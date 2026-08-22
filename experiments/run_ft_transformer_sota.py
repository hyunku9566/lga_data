"""
run_ft_transformer_sota.py — Turing GPU Compatible FT-Transformer
  * Feature-to-Feature Cross Attention across feature tokens
  * Turing GPU (RTX 2060S) memory efficient Attention backend
"""
import os, json, time, math, warnings
import numpy as np, pandas as pd, scipy.special as sp
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lib_lga
warnings.filterwarnings('ignore')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
D = '/home/lee/lga/'
OUT = D + 'results_ft_transformer/'
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

# 2. FT-Transformer 모델 아키텍처
class FTBlock(nn.Module):
    def __init__(self, d_token=48, n_heads=4, ffn_mult=2, drop=0.1):
        super().__init__()
        self.d_token = d_token
        self.n_heads = n_heads
        self.head_dim = d_token // n_heads
        
        self.norm1 = nn.LayerNorm(d_token)
        self.q_proj = nn.Linear(d_token, d_token)
        self.k_proj = nn.Linear(d_token, d_token)
        self.v_proj = nn.Linear(d_token, d_token)
        self.out_proj = nn.Linear(d_token, d_token)
        
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * ffn_mult),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_token * ffn_mult, d_token),
            nn.Dropout(drop)
        )
        self.dropout = nn.Dropout(drop)
        
    def forward(self, x):
        h = self.norm1(x)
        B, N, D = h.shape
        q = self.q_proj(h).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Turing GPU Compatible SDPA
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1 if self.training else 0.0)
            
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        x = x + self.dropout(self.out_proj(attn_out))
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

# 3. 듀얼 폴드 검증 루프
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

def train_eval_fold(vs, epochs=8, batch_size=512, lr=1e-3):
    ctx_info = lib_lga.get_ctx(vs)
    tr_idx = np.where(ctx_info['tr'])[0]
    va_idx = np.where(ctx_info['va'])[0]
    
    train_ds = PitchDataset(tr_idx, ctx_info['w'])
    val_ds = PitchDataset(va_idx)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    
    model = FTTransformer(n_num=len(NUM_COLS), cat_cards=cat_cards, d_token=48, n_layers=3, n_heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    scaler = torch.amp.GradScaler('cuda')
    
    log(f'\n--- [폴드 {vs}] FT-Transformer 학습 시작 (피처 토큰 간 Self-Attention) ---')
    best_score = -999.0
    best_preds = None
    
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for xn, xc, ym, yc, w in train_loader:
            xn, xc, ym, yc, w = xn.to(device), xc.to(device), ym.to(device), yc.to(device), w.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out_m, out_c = model(xn, xc)
                loss_m = (bce_loss(out_m, ym) * w).mean()
                loss_c = (bce_loss(out_c, yc) * w.unsqueeze(-1)).mean()
                loss = loss_m + 0.3 * loss_c
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            
        model.eval()
        preds_list = []
        with torch.no_grad():
            for xn, xc, ym, yc, _ in val_loader:
                xn, xc = xn.to(device), xc.to(device)
                with torch.amp.autocast('cuda'):
                    out_m, _ = model(xn, xc)
                preds_list.append(torch.sigmoid(out_m).float().cpu().numpy())
        val_preds = np.concatenate(preds_list)
        score = lib_lga.bss(val_preds, ctx_info['yv'], ctx_info['base'])
        log(f'  Epoch {ep:2d}/{epochs} ({time.time()-t0:4.1f}s) | Loss: {total_loss/len(train_loader):.4f} | BSS: {score:6.2f}')
        if score > best_score:
            best_score = score
            best_preds = val_preds
            
    np.save(f'{OUT}ft_transformer_{vs}.npy', best_preds.astype(np.float32))
    return best_score

log('===== FT-Transformer 듀얼 폴드 검증 =====')
s24 = train_eval_fold(2024, epochs=6)
s23 = train_eval_fold(2023, epochs=6)

log(f'\n===== FT-Transformer 최종 결과 =====')
log(f'2024 폴드: {s24:6.2f}')
log(f'2023 폴드: {s23:6.2f}')
