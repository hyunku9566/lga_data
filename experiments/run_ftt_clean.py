"""
run_ftt_clean.py — FT-Transformer 정식 재측정 스크립트

규약 준수:
  1. 피처: lib_lga.build_v7() 120개
  2. 폴드: 2024 / 2023 듀얼 폴드 (lib_lga.fold_ctx)
  3. 최근성 가중 hl=2.0
  4. 조기중단: 학습 데이터 내부 6% 홀드아웃 손실 기준으로만 판정 (patience=6, max_epochs=40)
  5. 검증 폴드는 학습 완료 후 최적 가중치로 복원하여 단 1회만 평가
  6. 산출물: results_ftt_clean/ftt_2024.npy, ftt_2023.npy, log.txt
"""
import os, sys, json, time, math, warnings
import numpy as np, pandas as pd, scipy.special as sp
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lib_lga
warnings.filterwarnings('ignore')

D = '/home/lee/lga/'
OUT = D + 'results_ftt_clean/'
os.makedirs(OUT, exist_ok=True)
LOG_FILE = open(OUT + 'log.txt', 'w', buffering=1)

def log(*args):
    msg = ' '.join(str(x) for x in args)
    print(msg, flush=True)
    LOG_FILE.write(msg + '\n')

DEV = os.environ.get('LGA_DEV', 'cuda:0')
device = torch.device(DEV if torch.cuda.is_available() else 'cpu')
log(f'사용 디바이스: {device}')

# 1. 데이터 로드 및 피처 준비 (v7 120개 규약)
b = lib_lga.load_base()
RAW = b['RAW']
comp_labels, pitch_cls, valid_pt = lib_lga.recover_labels(RAW)
XK = lib_lga.build_v7(b=b)

CAT_COLS = ['pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id',
            'pitcher_hand', 'batter_hand', 'base_state', 'game_type', 'top_bottom']
NUM_COLS = [c for c in XK.columns if c not in CAT_COLS and not c.startswith('__')]

assert len(CAT_COLS) + len(NUM_COLS) == 120, f'피처 수 오류: {len(CAT_COLS) + len(NUM_COLS)} != 120'

# 범주형 변수 정수 인덱싱
cat_cards = []
X_cat = np.zeros((len(RAW), len(CAT_COLS)), dtype=np.int64)
for i, col in enumerate(CAT_COLS):
    s = RAW[col].astype(str)
    unq = sorted(s.unique())
    mp = {v: k for k, v in enumerate(unq)}
    X_cat[:, i] = s.map(mp).fillna(0).values
    cat_cards.append(len(unq) + 1)

# 수치형 변수 정규화
X_num_raw = np.nan_to_num(XK[NUM_COLS].values.astype(np.float32), 0.0)
mu = np.nanmean(X_num_raw, 0)
std = np.maximum(np.nanstd(X_num_raw, 0), 1e-6)
X_num = np.clip((X_num_raw - mu) / std, -6.0, 6.0)

y_main = b['y']
y_comp = comp_labels[['reverse', 'middle', 'ball', 'strike']].fillna(0.5).values.astype(np.float32)

log(f'피처 준비 완료: 수치형 {len(NUM_COLS)}개, 범주형 {len(CAT_COLS)}개 (총 {len(NUM_COLS)+len(CAT_COLS)}개 피처 토큰)')

# 2. FT-Transformer 아키텍처
class FTBlock(nn.Module):
    def __init__(self, d_token=48, n_heads=4, ffn_mult=2, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.mha = nn.MultiheadAttention(d_token, n_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * ffn_mult),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_token * ffn_mult, d_token),
            nn.Dropout(drop)
        )
    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.mha(h, h, h)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class FTTransformer(nn.Module):
    def __init__(self, n_num, cat_cards, d_token=48, n_layers=3, n_heads=4, drop=0.1):
        super().__init__()
        self.num_weights = nn.Parameter(torch.randn(n_num, d_token) * 0.02)
        self.num_biases = nn.Parameter(torch.zeros(n_num, d_token))
        
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, d_token) for card in cat_cards
        ])
        for e in self.cat_embeddings:
            nn.init.normal_(e.weight, std=0.02)
            
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        
        self.layers = nn.ModuleList([
            FTBlock(d_token=d_token, n_heads=n_heads, ffn_mult=2, drop=drop)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_token)
        self.head_main = nn.Linear(d_token, 1)
        self.head_comp = nn.Linear(d_token, 4)
        
    def forward(self, x_num, x_cat):
        t_num = x_num.unsqueeze(-1) * self.num_weights + self.num_biases
        t_cat = torch.stack([e(x_cat[:, i]) for i, e in enumerate(self.cat_embeddings)], dim=1)
        B = x_num.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, t_num, t_cat], dim=1)
        
        for layer in self.layers:
            tokens = layer(tokens)
        cls_rep = self.norm(tokens[:, 0])
        return self.head_main(cls_rep).squeeze(-1), self.head_comp(cls_rep)

# 3. 데이터셋 및 학습 파이프라인
class PitchDataset(Dataset):
    def __init__(self, idxs, weights):
        self.xn = torch.tensor(X_num[idxs], dtype=torch.float32)
        self.xc = torch.tensor(X_cat[idxs], dtype=torch.long)
        self.ym = torch.tensor(y_main[idxs], dtype=torch.float32)
        self.yc = torch.tensor(y_comp[idxs], dtype=torch.float32)
        self.w  = torch.tensor(weights, dtype=torch.float32)
    def __len__(self):
        return len(self.ym)
    def __getitem__(self, i):
        return self.xn[i], self.xc[i], self.ym[i], self.yc[i], self.w[i]

def train_and_eval_fold(vs, max_epochs=40, patience=6, batch_size=1024, lr=3e-4, seed=42):
    log(f'\n======================================================')
    log(f'=== [폴드 {vs}] FT-Transformer 정식 측정 시작 ===')
    log(f'======================================================')
    
    ctx = lib_lga.fold_ctx(vs)
    tr_mask, va_mask = ctx['tr'], ctx['va']
    season = b['season']
    
    # 1. 학습 인덱스 내부 6% 홀드아웃 분할 (run18 규약과 동일)
    ia = np.where(tr_mask)[0]
    rs = np.random.RandomState(seed)
    rs.shuffle(ia)
    nin = int(len(ia) * 0.06)
    iin, itr = ia[:nin], ia[nin:]
    
    # 최근성 가중치 계산 (hl=2.0)
    w_tr = (0.5 ** ((vs - 1 - season[itr]) / 2.0)).astype(np.float32)
    w_in = (0.5 ** ((vs - 1 - season[iin]) / 2.0)).astype(np.float32)
    
    log(f'학습 분할: 훈련 {len(itr):,}건 / 내부 홀드아웃 {len(iin):,}건 (6.0%) | 검증 폴드: {len(np.where(va_mask)[0]):,}건')
    
    train_ds = PitchDataset(itr, w_tr)
    holdout_ds = PitchDataset(iin, w_in)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    holdout_loader = DataLoader(holdout_ds, batch_size=batch_size * 2, shuffle=False)
    
    torch.manual_seed(seed)
    model = FTTransformer(n_num=len(NUM_COLS), cat_cards=cat_cards, d_token=48, n_layers=3, n_heads=4, drop=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    best_holdout_loss = float('inf')
    best_epoch = 0
    best_weights = None
    bad_epochs = 0
    
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        total_tr_loss = 0.0
        n_tr_batches = 0
        
        for xn, xc, ym, yc, w in train_loader:
            xn, xc, ym, yc, w = xn.to(device), xc.to(device), ym.to(device), yc.to(device), w.to(device)
            optimizer.zero_grad()
            out_m, out_c = model(xn, xc)
            
            loss_m = (bce_loss(out_m, ym) * w).mean()
            loss_c = (bce_loss(out_c, yc) * w.unsqueeze(-1)).mean()
            loss = loss_m + 0.3 * loss_c
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_tr_loss += loss.item()
            n_tr_batches += 1
            
        scheduler.step()
        avg_tr_loss = total_tr_loss / n_tr_batches
        
        # 내부 홀드아웃 평가 (검증 폴드가 아님!)
        model.eval()
        total_ho_loss = 0.0
        n_ho_samples = 0
        with torch.no_grad():
            for xn, xc, ym, yc, w in holdout_loader:
                xn, xc, ym, yc, w = xn.to(device), xc.to(device), ym.to(device), yc.to(device), w.to(device)
                out_m, out_c = model(xn, xc)
                loss_m = (bce_loss(out_m, ym) * w).sum().item()
                loss_c = (bce_loss(out_c, yc) * w.unsqueeze(-1)).sum().item()
                total_ho_loss += loss_m + 0.3 * loss_c
                n_ho_samples += len(ym)
                
        avg_ho_loss = total_ho_loss / n_ho_samples
        dt = time.time() - t0
        
        # 조기중단 판정
        if avg_ho_loss < best_holdout_loss - 1e-6:
            best_holdout_loss = avg_ho_loss
            best_epoch = ep
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            star = " ★ (홀드아웃 최저손실)"
        else:
            bad_epochs += 1
            star = f" (개선없음 {bad_epochs}/{patience})"
            
        log(f'Epoch {ep:2d}/{max_epochs} [{dt:4.1f}s] | 훈련 손실: {avg_tr_loss:.5f} | 홀드아웃 손실: {avg_ho_loss:.6f}{star}')
        
        if bad_epochs >= patience:
            log(f'--> 홀드아웃 손실이 {patience}에폭 연속 개선되지 않아 조기중단합니다. (최적 에폭: {best_epoch} / 최저 홀드아웃 손실: {best_holdout_loss:.6f})')
            break
            
    # 2. 최적 에폭 가중치 복원
    log(f'\n최적 가중치(에폭 {best_epoch}) 복원 중...')
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()
    
    # 3. 검증 폴드 단 1회 최종 평가
    log(f'검증 폴드({vs} 시즌) 단 1회 최종 평가 진행 중...')
    va_idx = np.where(va_mask)[0]
    val_ds = PitchDataset(va_idx, np.ones(len(va_idx)))
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    
    preds_list = []
    with torch.no_grad():
        for xn, xc, ym, yc, _ in val_loader:
            xn, xc = xn.to(device), xc.to(device)
            out_m, _ = model(xn, xc)
            preds_list.append(torch.sigmoid(out_m).float().cpu().numpy())
            
    val_preds = np.concatenate(preds_list)
    save_path = f'{OUT}ftt_{vs}.npy'
    np.save(save_path, val_preds.astype(np.float32))
    
    score = lib_lga.bss(val_preds, ctx['yv'], ctx['base'])
    log(f'>>> [폴드 {vs}] FT-Transformer 최종 실측 BSS: {score:.2f} (저장 완료: {save_path})')
    
    return score, best_epoch

# 실행
s24, ep24 = train_and_eval_fold(2024, max_epochs=40, patience=6, lr=3e-4)
s23, ep23 = train_and_eval_fold(2023, max_epochs=40, patience=6, lr=3e-4)

# 4. 저장물 재채점 및 앙상블 한계이득 산출
log(f'\n======================================================')
log(f'=== 저장된 .npy 직접 재채점 및 한계이득 실측 검증 ===')
log(f'======================================================')

p_ftt_24 = np.load(f'{OUT}ftt_2024.npy')
p_ftt_23 = np.load(f'{OUT}ftt_2023.npy')

ctx24 = lib_lga.get_ctx(2024)
ctx23 = lib_lga.get_ctx(2023)

score24 = lib_lga.bss(p_ftt_24, ctx24['yv'], ctx24['base'])
score23 = lib_lga.bss(p_ftt_23, ctx23['yv'], ctx23['base'])

log(f'\n[실측 재채점 결과]')
log(f'  • FT-Transformer 단독 폴드2024: {score24:.2f} (조기중단 최적에폭: {ep24})')
log(f'  • FT-Transformer 단독 폴드2023: {score23:.2f} (조기중단 최적에폭: {ep23})')

# 기준선 구성 (지시서 76-85줄 규약)
lg = lambda p: sp.logit(np.clip(p, 1e-6, 1-1e-6))

# 트리축
p_trees_24 = [np.load(f'{D}results32/p_{cfg}_s0_2024.npy') for cfg in ['A', 'B', 'C', 'D']]
tree_24 = sp.expit(np.mean([lg(p) for p in p_trees_24], 0))

p_trees_23 = [np.load(f'{D}results32/p_{cfg}_s0_2023.npy') for cfg in ['A', 'B', 'C', 'D']]
tree_23 = sp.expit(np.mean([lg(p) for p in p_trees_23], 0))

# 2024 NN 축
nn_sel = [x[0] for x in json.load(open(f'{D}v4_nn_sel.json'))]
p_old = [np.load(f'{D}results6/{tag}_2024.npy') for tag in nn_sel]
old_24 = sp.expit(np.mean([lg(p) for p in p_old], 0))

mt_tags = [f'L3_s{s}' for s in range(4)] + [f'L5_s{s}' for s in range(4)]
p_mt = [np.load(f'{D}results18/{tag}_2024.npy') for tag in mt_tags]
mt_24 = sp.expit(np.mean([lg(p) for p in p_mt], 0))

nn_24 = sp.expit(0.6 * lg(old_24) + 0.4 * lg(mt_24))
base_24 = sp.expit(0.70 * lg(tree_24) + 0.30 * lg(nn_24))
base_23 = tree_23

base_score24 = lib_lga.bss(base_24, ctx24['yv'], ctx24['base'])
base_score23 = lib_lga.bss(base_23, ctx23['yv'], ctx23['base'])

log(f'\n[기준선 BSS]')
log(f'  • 폴드2024 기준선 (Tree 0.70 + NN 0.30): {base_score24:.2f}')
log(f'  • 폴드2023 기준선 (Tree 1.00):           {base_score23:.2f}')

log(f'\n[FT-Transformer 앙상블 한계이득 스윕 (w=0.05~0.30)]')
best_gain24, best_w24 = -999.0, 0.0
for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    blend = sp.expit((1 - w) * lg(base_24) + w * lg(p_ftt_24))
    s = lib_lga.bss(blend, ctx24['yv'], ctx24['base'])
    diff = s - base_score24
    if diff > best_gain24: best_gain24, best_w24 = diff, w
    log(f'  폴드2024 w={w:.2f}: BSS={s:.2f} (한계이득: {diff:+5.2f})')

best_gain23, best_w23 = -999.0, 0.0
for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    blend = sp.expit((1 - w) * lg(base_23) + w * lg(p_ftt_23))
    s = lib_lga.bss(blend, ctx23['yv'], ctx23['base'])
    diff = s - base_score23
    if diff > best_gain23: best_gain23, best_w23 = diff, w
    log(f'  폴드2023 w={w:.2f}: BSS={s:.2f} (한계이득: {diff:+5.2f})')

log(f'\n[최종 요약]')
log(f'  • 폴드2024 최적 가중치 w={best_w24:.2f} -> 한계이득: {best_gain24:+5.2f}')
log(f'  • 폴드2023 최적 가중치 w={best_w23:.2f} -> 한계이득: {best_gain23:+5.2f}')
log(f'\n전체 완료')
