"""
run_dcn_v2_sota.py — Deep & Cross Network v2 (DCN-v2)
  * Google Research의 명시적 고차 피처 크로스(Explicit High-Order Feature Crossing) 전용 아키텍처
  * Cross Network: x_{l+1} = x_0 * (W_l x_l + b_l) + x_l (다항식 차수의 모든 피처 곱을 효율적으로 자동 생성)
  * Deep Network: SwiGLU 비선형 표현 학습
  * 구종 x 카운트, 투수 x 상황 등 모든 피처의 N차원 상호작용을 수학적으로 직접 연산
"""
import os, json, time, math, warnings
import numpy as np, pandas as pd, scipy.special as sp
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lib_lga
warnings.filterwarnings('ignore')

D = '/home/lee/lga/'
OUT = D + 'results_dcn_v2/'
log, OUT = lib_lga.mklog(OUT, 'log.txt')
DEV = os.environ.get('LGA_DEV', 'cuda:1')
device = torch.device(DEV if torch.cuda.is_available() else 'cpu')
log(f'사용 디바이스: {device}')

# 1. 데이터 로드 및 피처 준비
b = lib_lga.load_base()
RAW = b['RAW']
comp_labels, pitch_cls, valid_pt = lib_lga.recover_labels(RAW)
XK = lib_lga.build_v7(b=b)

CAT_COLS = ['pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id',
            'pitcher_hand', 'batter_hand', 'base_state', 'game_type', 'top_bottom']
NUM_COLS = [c for c in XK.columns if c not in CAT_COLS and not c.startswith('__')]

cat_cards = []
X_cat = np.zeros((len(RAW), len(CAT_COLS)), dtype=np.int64)
for i, col in enumerate(CAT_COLS):
    s = RAW[col].astype(str)
    unq = sorted(s.unique())
    mp = {v: k for k, v in enumerate(unq)}
    X_cat[:, i] = s.map(mp).fillna(0).values
    cat_cards.append(len(unq) + 1)

X_num_raw = np.nan_to_num(XK[NUM_COLS].values.astype(np.float32), 0.0)
mu = np.nanmean(X_num_raw, 0)
std = np.maximum(np.nanstd(X_num_raw, 0), 1e-4)
X_num = np.clip((X_num_raw - mu) / std, -5.0, 5.0)

y_main = b['y']
y_comp = comp_labels[['reverse', 'middle', 'ball', 'strike']].fillna(0.5).values.astype(np.float32)

log(f'피처 준비 완료: 수치형 {len(NUM_COLS)}개, 범주형 {len(CAT_COLS)}개')

# 2. DCN-v2 아키텍처
class CrossLayerV2(nn.Module):
    """Matrix-based Cross Layer v2: x_{l+1} = x_0 * (W_l x_l + b_l) + x_l"""
    def __init__(self, in_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, in_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(in_dim))
    def forward(self, x0, xl):
        # x0: [B, D], xl: [B, D]
        return x0 * (self.w(xl) + self.b) + xl

class CrossNetworkV2(nn.Module):
    def __init__(self, in_dim, n_cross_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([CrossLayerV2(in_dim) for _ in range(n_cross_layers)])
    def forward(self, x0):
        xl = x0
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl

class DCNv2(nn.Module):
    def __init__(self, n_num, cat_cards, emb_dim=16, n_cross_layers=4, deep_dims=[384, 256, 128], drop=0.1):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, min(32, max(4, int(card**0.35)))) for card in cat_cards
        ])
        cat_total_dim = sum(e.embedding_dim for e in self.cat_embeddings)
        self.in_dim = n_num + cat_total_dim
        
        # 1. Explicit Cross Network
        self.cross_net = CrossNetworkV2(self.in_dim, n_cross_layers=n_cross_layers)
        
        # 2. Parallel Deep Network
        deep_layers = []
        prev_dim = self.in_dim
        for h in deep_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Dropout(drop)
            ])
            prev_dim = h
        self.deep_net = nn.Sequential(*deep_layers)
        
        # Combined Output Heads
        comb_dim = self.in_dim + deep_dims[-1]
        self.head_main = nn.Sequential(
            nn.Linear(comb_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.head_comp = nn.Sequential(
            nn.Linear(comb_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 4)
        )
        
    def forward(self, x_num, x_cat):
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.cat_embeddings)]
        x0 = torch.cat([x_num] + embs, dim=-1)
        
        x_cross = self.cross_net(x0)
        x_deep = self.deep_net(x0)
        
        x_comb = torch.cat([x_cross, x_deep], dim=-1)
        return self.head_main(x_comb).squeeze(-1), self.head_comp(x_comb)

# 3. 듀얼 폴드 검증
class PitchDataset(Dataset):
    def __init__(self, idxs, weights=None):
        self.xn = torch.tensor(X_num[idxs], dtype=torch.float32)
        self.xc = torch.tensor(X_cat[idxs], dtype=torch.long)
        self.ym = torch.tensor(y_main[idxs], dtype=torch.float32)
        self.yc = torch.tensor(y_comp[idxs], dtype=torch.float32)
        self.w = torch.tensor(weights if weights is not None else np.ones(len(idxs)), dtype=torch.float32)
    def __len__(self):
        return len(self.ym)
    def __getitem__(self, i):
        return self.xn[i], self.xc[i], self.ym[i], self.yc[i], self.w[i]

def train_eval_fold(vs, epochs=12, batch_size=2048, lr=1.5e-3):
    ctx_info = lib_lga.get_ctx(vs)
    tr_idx = np.where(ctx_info['tr'])[0]
    va_idx = np.where(ctx_info['va'])[0]
    
    train_ds = PitchDataset(tr_idx, ctx_info['w'])
    val_ds = PitchDataset(va_idx)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    
    model = DCNv2(n_num=len(NUM_COLS), cat_cards=cat_cards).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    log(f'\n--- [폴드 {vs}] DCN-v2 학습 시작 (명시적 피처 크로스 연산) ---')
    best_score = -999.0
    best_preds = None
    
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
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
            scheduler.step()
            total_loss += loss.item()
            
        model.eval()
        preds_list = []
        with torch.no_grad():
            for xn, xc, ym, yc, _ in val_loader:
                xn, xc = xn.to(device), xc.to(device)
                out_m, _ = model(xn, xc)
                preds_list.append(torch.sigmoid(out_m).cpu().numpy())
        val_preds = np.concatenate(preds_list)
        score = lib_lga.bss(val_preds, ctx_info['yv'], ctx_info['base'])
        log(f'  Epoch {ep:2d}/{epochs} ({time.time()-t0:4.1f}s) | Loss: {total_loss/len(train_loader):.4f} | BSS: {score:6.2f}')
        if score > best_score:
            best_score = score
            best_preds = val_preds
            
    np.save(f'{OUT}dcn_v2_{vs}.npy', best_preds.astype(np.float32))
    return best_score

log('===== DCN-v2 듀얼 폴드 검증 =====')
s24 = train_eval_fold(2024, epochs=10)
s23 = train_eval_fold(2023, epochs=10)

log(f'\n===== DCN-v2 최종 결과 =====')
log(f'2024 폴드: {s24:6.2f}')
log(f'2023 폴드: {s23:6.2f}')
