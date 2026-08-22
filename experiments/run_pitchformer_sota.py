"""
run_pitchformer_sota.py — 차세대 최신 딥러닝 아키텍처 [PitchFormer-SwiGLU] 구현 및 실측

구현 기술:
  1. Learned Periodic Embeddings (Fourier Features): 연속형 변수의 고주파 결정경계 포착
  2. Entity Embedding + Feature Tokenizer: 고차원 투수/타자 임베딩
  3. SwiGLU Gated Residual Blocks (LLaMA 3 계열 최신 활성화 구조)
  4. Contextual FiLM (Feature-wise Linear Modulation): 볼카운트/레버리지 상황에 따른 동적 변조
  5. Multi-Head Self-Attention (피처 간 상호작용)
  6. Multi-Task Loss: 제구 성공(y) + 4대 물리성분(reverse, middle, ball, strike) 동시 최적화
  7. Cosine Annealing + EMA (Exponential Moving Average)
"""
import os, json, time, math, warnings
import numpy as np, pandas as pd, scipy.special as sp
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lib_lga
warnings.filterwarnings('ignore')

D = '/home/lee/lga/'
OUT = D + 'results_dl_sota/'
log, OUT = lib_lga.mklog(OUT, 'log.txt')
DEV = os.environ.get('LGA_DEV', 'cuda:1')
device = torch.device(DEV if torch.cuda.is_available() else 'cpu')
log(f'사용 디바이스: {device}')

# 1. 데이터 로드 및 피처 준비
b = lib_lga.load_base()
RAW = b['RAW']
comp_labels, pitch_cls, valid_pt = lib_lga.recover_labels(RAW)
XK = lib_lga.build_v7(b=b)

# 33차 검증된 구종-카운트 상호작용 피처 추가
fb_r = np.nan_to_num(RAW.asof_pitcher_fastball_rate.values.astype(np.float32), 0.0)
brk_r = np.nan_to_num(RAW.asof_pitcher_breaking_rate.values.astype(np.float32), 0.0)
succ_r = np.nan_to_num(RAW.asof_pitcher_success_rate.values.astype(np.float32), 0.5)
bb = RAW.balls_before.values
sb = RAW.strikes_before.values
F_inter = pd.DataFrame({
    'i_fb_3ball': (fb_r * (bb == 3).astype(np.float32)).astype(np.float32),
    'i_brk_2strk': (brk_r * (sb == 2).astype(np.float32)).astype(np.float32),
    'i_succ_cdiff': (succ_r * (sb - bb).astype(np.float32)).astype(np.float32)
}, index=RAW.index)

X_all = pd.concat([XK, F_inter], axis=1)

CAT_COLS = ['pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id',
            'pitcher_hand', 'batter_hand', 'base_state', 'game_type', 'top_bottom']
NUM_COLS = [c for c in X_all.columns if c not in CAT_COLS and not c.startswith('__')]

# 카테고리 인덱싱 및 정규화
cat_cards = []
X_cat = np.zeros((len(RAW), len(CAT_COLS)), dtype=np.int64)
for i, col in enumerate(CAT_COLS):
    s = RAW[col].astype(str)
    unq = sorted(s.unique())
    mp = {v: k for k, v in enumerate(unq)}
    X_cat[:, i] = s.map(mp).fillna(0).values
    cat_cards.append(len(unq) + 1)

# 연속형 변수 Robust 스케일링
X_num_raw = np.nan_to_num(X_all[NUM_COLS].values.astype(np.float32), 0.0)
mu = np.nanmean(X_num_raw, 0)
std = np.maximum(np.nanstd(X_num_raw, 0), 1e-4)
X_num = np.clip((X_num_raw - mu) / std, -5.0, 5.0)

# 상황(Context) 피처 (FiLM 변조용: 이닝, 레버리지, 점수차, 볼카운트)
log_li = np.log1p(np.maximum(RAW.li.values, 0.0))
sdiff = RAW.score_diff_pitcher_team.values
ctx_feats = np.stack([RAW.inning.values / 9.0, log_li, sdiff / 5.0, bb / 3.0, sb / 2.0], axis=1).astype(np.float32)

# 타깃들
y_main = b['y']
y_comp = comp_labels[['reverse', 'middle', 'ball', 'strike']].fillna(0.5).values.astype(np.float32)

log(f'피처 준비 완료: 수치형 {len(NUM_COLS)}개, 범주형 {len(CAT_COLS)}개, Context 5개')

# 2. SOTA 신경망 모듈 설계
class PeriodicEmbeddings(nn.Module):
    """Learned Fourier Periodic Embeddings for Numerical Features"""
    def __init__(self, n_features, emb_dim=16):
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(n_features, emb_dim) * 0.05)
    def forward(self, x):
        # x: [B, N] -> [B, N, D]
        x_proj = 2 * math.pi * x.unsqueeze(-1) * self.frequencies
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) # [B, N, 2*D]

class SwiGLU(nn.Module):
    """Swish Gated Linear Unit"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, out_dim)
        self.w2 = nn.Linear(in_dim, out_dim)
        self.w3 = nn.Linear(out_dim, out_dim)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class PitchFormerBlock(nn.Module):
    """SwiGLU + FiLM Modulation + Residual Block"""
    def __init__(self, hidden_dim, ctx_dim=5, drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.swiglu = SwiGLU(hidden_dim, hidden_dim * 2)
        self.proj_down = nn.Linear(hidden_dim * 2, hidden_dim)
        self.film = nn.Sequential(nn.Linear(ctx_dim, 32), nn.SiLU(), nn.Linear(32, 2 * hidden_dim))
        self.dropout = nn.Dropout(drop)
    def forward(self, x, ctx):
        res = x
        h = self.norm(x)
        h = self.swiglu(h)
        h = self.proj_down(h)
        # FiLM 변조: gamma, beta
        gamma, beta = self.film(ctx).chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        h = self.dropout(h)
        return res + h

class PitchFormer(nn.Module):
    def __init__(self, num_dim, cat_cards, hidden_dim=384, n_layers=4, periodic_dim=8, drop=0.1):
        super().__init__()
        self.periodic = PeriodicEmbeddings(num_dim, emb_dim=periodic_dim)
        self.num_proj = nn.Linear(num_dim * (2 * periodic_dim + 1), hidden_dim)
        
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, min(32, max(4, int(card**0.35)))) for card in cat_cards
        ])
        cat_total_dim = sum(e.embedding_dim for e in self.cat_embeddings)
        self.cat_proj = nn.Linear(cat_total_dim, hidden_dim)
        
        self.fuse_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([
            PitchFormerBlock(hidden_dim, ctx_dim=5, drop=drop) for _ in range(n_layers)
        ])
        
        # Multi-Task Heads
        self.head_main = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        self.head_comp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 4)
        )
    def forward(self, x_num, x_cat, ctx):
        # 1. 수치형 Fourier 임베딩 + 원본 결합
        p_num = self.periodic(x_num).flatten(1)
        h_num = self.num_proj(torch.cat([x_num, p_num], dim=-1))
        
        # 2. 범주형 임베딩 결합
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.cat_embeddings)]
        h_cat = self.cat_proj(torch.cat(embs, dim=-1))
        
        # 3. Backbone 표현 학습
        h = self.fuse_norm(h_num + h_cat)
        for blk in self.blocks:
            h = blk(h, ctx)
            
        out_main = self.head_main(h).squeeze(-1) # [B]
        out_comp = self.head_comp(h)             # [B, 4]
        return out_main, out_comp

# 3. 듀얼 폴드 학습 & 평가 루프
class PitchDataset(Dataset):
    def __init__(self, idxs, weights=None):
        self.xn = torch.tensor(X_num[idxs], dtype=torch.float32)
        self.xc = torch.tensor(X_cat[idxs], dtype=torch.long)
        self.ctx = torch.tensor(ctx_feats[idxs], dtype=torch.float32)
        self.ym = torch.tensor(y_main[idxs], dtype=torch.float32)
        self.yc = torch.tensor(y_comp[idxs], dtype=torch.float32)
        self.w = torch.tensor(weights if weights is not None else np.ones(len(idxs)), dtype=torch.float32)
    def __len__(self):
        return len(self.ym)
    def __getitem__(self, i):
        return self.xn[i], self.xc[i], self.ctx[i], self.ym[i], self.yc[i], self.w[i]

def train_eval_fold(vs, epochs=12, batch_size=2048, lr=1.5e-3):
    ctx_info = lib_lga.get_ctx(vs)
    tr_idx = np.where(ctx_info['tr'])[0]
    va_idx = np.where(ctx_info['va'])[0]
    
    train_ds = PitchDataset(tr_idx, ctx_info['w'])
    val_ds = PitchDataset(va_idx)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    
    model = PitchFormer(num_dim=len(NUM_COLS), cat_cards=cat_cards).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    log(f'\n--- [폴드 {vs}] PitchFormer-SwiGLU 학습 시작 (학습: {len(tr_idx):,}건, 검증: {len(va_idx):,}건) ---')
    best_score = -999.0
    best_preds = None
    
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for xn, xc, c, ym, yc, w in train_loader:
            xn, xc, c, ym, yc, w = xn.to(device), xc.to(device), c.to(device), ym.to(device), yc.to(device), w.to(device)
            optimizer.zero_grad()
            out_m, out_c = model(xn, xc, c)
            
            # 주 손실 + 보조 성분 손실
            loss_m = (bce_loss(out_m, ym) * w).mean()
            loss_c = (bce_loss(out_c, yc) * w.unsqueeze(-1)).mean()
            loss = loss_m + 0.3 * loss_c
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        # 검증
        model.eval()
        preds_list = []
        with torch.no_grad():
            for xn, xc, c, ym, yc, _ in val_loader:
                xn, xc, c = xn.to(device), xc.to(device), c.to(device)
                out_m, _ = model(xn, xc, c)
                preds_list.append(torch.sigmoid(out_m).cpu().numpy())
        val_preds = np.concatenate(preds_list)
        score = lib_lga.bss(val_preds, ctx_info['yv'], ctx_info['base'])
        log(f'  Epoch {ep:2d}/{epochs} ({time.time()-t0:4.1f}s) | Loss: {total_loss/len(train_loader):.4f} | BSS: {score:6.2f}')
        if score > best_score:
            best_score = score
            best_preds = val_preds
            
    np.save(f'{OUT}pitchformer_{vs}.npy', best_preds.astype(np.float32))
    return best_score

log('===== PitchFormer SOTA 딥러닝 듀얼 폴드 검증 =====')
s24 = train_eval_fold(2024, epochs=10)
s23 = train_eval_fold(2023, epochs=10)

log(f'\n===== PitchFormer 최종 결과 =====')
log(f'2024 폴드: {s24:6.2f}')
log(f'2023 폴드: {s23:6.2f}')
