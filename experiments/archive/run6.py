"""
6차 — 대규모 신경망 스윕 (양 GPU 병렬)
사용법: python run6.py <device> <shard> <nshards>

【검증 설계 교정 — 이게 핵심】
  R(1군) 시즌 제구율: 2019 .5495 → 2020 .5269 → 2021 .5128 → 2022 .5037 → 2023 .5031 → 2024 .4897
  2024 에만 ABS 충격. 2023 은 퓨처스 ABS 로 F 가 붕괴.
  예측 대상 2025 = "ABS 2년차 → 안정 체제 내 연도 전환"
  => 같은 구조의 폴드는 2021, 2022 (동일 체제 내 전환)
  => 기존에 쓰던 2023/2024 는 둘 다 체제 전환 연도라 부적합했음.
     그 폴드가 "용량을 극단적으로 줄여라"로 수렴시켰고, 신경망도 그 기준으로 기각됐다.

【아키텍처】
  A TabM            파라미터 효율 앙상블 MLP (ICLR 2025)
  B TabM+PLR        주기적 수치 임베딩 추가
  C FT-Transformer  피처 토큰화 + 어텐션
  D HierShrink      정밀도 가중 계층 축소 (이 문제의 통계 구조를 직접 인코딩)
"""
import os, sys, time, json, warnings, traceback, itertools
import numpy as np, pandas as pd, scipy.special as sp
import torch, torch.nn as nn
warnings.filterwarnings('ignore')

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:0'
SHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NSH = int(sys.argv[3]) if len(sys.argv) > 3 else 1
D = '/home/lee/lga/'; OUT = D + f'results6/'
os.makedirs(OUT, exist_ok=True)
LOG = open(OUT + f'log_s{SHARD}.txt', 'a', buffering=1)
def log(*a):
    m = ' '.join(str(x) for x in a); print(m, flush=True); LOG.write(m + '\n')
T0 = time.time()
def el(): return f'[{(time.time()-T0)/60:6.1f}m]'
log(f'\n{"="*70}\n{el()} shard {SHARD}/{NSH} on {DEV}\n{"="*70}')

X = pd.read_parquet(D + 'X98.parquet')
y = X.__y.values.astype(np.float32); season = X.__season.values; isF = X.__F.values.astype(bool)
COLS = [c for c in X.columns if not c.startswith('__')]

# ── 교정된 폴드: 동일 체제 내 전환 (2025 와 같은 구조) ──
FOLDS = [2024, 2022]   # 2024=학습량 최대 주폴드, 2022=안정체제 보조폴드
REF   = []
def split(vs):
    tr = (season < vs) & ~(isF & (season <= 2022) & (vs >= 2023))
    va = (season == vs) & ~isF          # 검증은 R 만 (test 는 R 로 보임)
    return tr, va
def trend_prior(vs):
    m = season < vs; s = pd.Series(y[m]).groupby(season[m]).mean()
    return float(sp.expit(np.polyval(np.polyfit(s.index, sp.logit(s.values), 1), vs)))
RP = {vs: trend_prior(vs) for vs in FOLDS + REF}
def evaluate(p, yv, rp):
    r = yv.mean(); ref = r*(1-r)
    b = lambda q: 100000*max(0., 1 - np.mean((q-yv)**2)/ref)
    lo = sp.logit(np.clip(p, 1e-6, 1-1e-6))
    return dict(raw=b(p), trend=b(sp.expit(lo-lo.mean()+sp.logit(rp))),
                oracle=b(sp.expit(lo-lo.mean()+sp.logit(r))), sdp=float(p.std()))

CAT = ['pitcher_id','batter_id','pitcher_team_id','batter_team_id','pitcher_hand',
       'batter_hand','base_state','game_type','top_bottom']
NUM = [c for c in COLS if c not in CAT]
Xc = X[CAT].values.astype(np.int64); Xc = np.maximum(Xc, 0)
Xn = np.nan_to_num(X[NUM].values.astype(np.float32), nan=0., posinf=0., neginf=0.)
MU, SD = Xn.mean(0), Xn.std(0)+1e-6
Xn = np.clip((Xn-MU)/SD, -6, 6)
CARD = [int(Xc[:,i].max())+2 for i in range(len(CAT))]
mth = X.game_month.values.astype(np.float32)
Tf = np.stack([np.sin(2*np.pi*mth/12), np.cos(2*np.pi*mth/12),
               np.sin(4*np.pi*mth/12), np.cos(4*np.pi*mth/12),
               (season-2019)/6.0], 1).astype(np.float32)

# HierShrink 용 증거 집합: (rate, n) 쌍
def col(c): return X[c].values.astype(np.float32)
EVID = [  # (rate, n, 종류id)
    (col('asof_pitcher_success_rate'), col('asof_pitcher_n'), 0),
    (col('p_succ_ssn'),                col('p_succ_ssn_n'),   1),
    (col('asof_pitcher_prev1_game_success_rate'), np.full(len(X), 20., np.float32), 2),
    (col('asof_pitcher_prev3_game_success_rate'), np.full(len(X), 60., np.float32), 3),
    (col('asof_pitcher_prev5_game_success_rate'), np.full(len(X), 100., np.float32), 4),
    (col('p_sit_overall'),             col('asof_pitcher_n'), 5),
    (col('p_sit_matched')+col('p_sit_overall'), col('asof_pitcher_n')/8, 6),
]
EVIDB = [
    (col('asof_batter_success_rate'), col('asof_batter_n'), 0),
    (col('b_succ_ssn'),               col('b_succ_ssn_n'),   1),
    (col('pb_rate'),                  col('pb_n'),           2),
]
def pack(ev):
    r = np.stack([np.nan_to_num(e[0], nan=0.5) for e in ev], 1)
    n = np.stack([np.nan_to_num(e[1], nan=0.)   for e in ev], 1)
    return (sp.logit(np.clip(r, .02, .98)).astype(np.float32), np.log1p(n).astype(np.float32))
Ep_r, Ep_n = pack(EVID); Eb_r, Eb_n = pack(EVIDB)

t = lambda a: torch.tensor(a, device=DEV)
TN, TC, TT, TY = t(Xn), t(Xc), t(Tf), t(y)
TPR, TPN, TBR, TBN = t(Ep_r), t(Ep_n), t(Eb_r), t(Eb_n)
log(f'{el()} 수치 {Xn.shape[1]} 범주 {len(CAT)} 증거 P{len(EVID)}/B{len(EVIDB)}')

# ══════════════ 아키텍처 ══════════════
class PLR(nn.Module):
    def __init__(s, d_in, k=8, d=6):
        super().__init__(); s.c = nn.Parameter(torch.randn(d_in,k)*.05); s.l = nn.Linear(2*k,d)
    def forward(s, x):
        z = 2*np.pi*x.unsqueeze(-1)*s.c
        return torch.relu(s.l(torch.cat([torch.sin(z), torch.cos(z)],-1))).flatten(1)

class TabM(nn.Module):
    def __init__(s, n_num, card, k=32, h=256, L=2, emb=0, drop=.1, plr=False, film=True):
        super().__init__(); s.k, s.emb, s.film, s.plr = k, emb, film, None
        d = n_num
        if plr: s.plr = PLR(n_num); d += n_num*6
        if emb:
            s.e1 = nn.Embedding(card[0], emb); s.e2 = nn.Embedding(card[1], emb)
            nn.init.normal_(s.e1.weight, std=.01); nn.init.normal_(s.e2.weight, std=.01); d += 2*emb
        s.r1 = nn.Parameter(torch.randn(k, d)*.1+1)
        s.ls = nn.ModuleList([nn.Linear(d if i==0 else h, h) for i in range(L)])
        s.dp = nn.Dropout(drop)
        s.hd = nn.Parameter(torch.randn(k,h)*.02); s.hb = nn.Parameter(torch.zeros(k))
        if film: s.fn = nn.Sequential(nn.Linear(5,32), nn.ReLU(), nn.Linear(32,2*h))
    def forward(s, xn, xc, tt, *a):
        z = xn
        if s.plr is not None: z = torch.cat([z, s.plr(xn)], -1)
        if s.emb: z = torch.cat([z, s.e1(xc[:,0]), s.e2(xc[:,1])], -1)
        z = z.unsqueeze(1)*s.r1
        for i, l in enumerate(s.ls):
            z = torch.relu(l(z))
            if i == 0:
                z = s.dp(z)
                if s.film:
                    g, b = s.fn(tt).chunk(2,-1); z = z*(1+g.unsqueeze(1))+b.unsqueeze(1)
        return ((z*s.hd).sum(-1)+s.hb).mean(1)

class FTT(nn.Module):
    """FT-Transformer: 피처 토큰화 + 어텐션"""
    def __init__(s, n_num, card, d=32, L=3, heads=4, drop=.1, **kw):
        super().__init__()
        s.w = nn.Parameter(torch.randn(n_num, d)*.05); s.b = nn.Parameter(torch.zeros(n_num, d))
        s.ce = nn.ModuleList([nn.Embedding(c, d) for c in card[:4]])
        s.cls = nn.Parameter(torch.randn(1,1,d)*.05)
        s.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, d*2, drop, batch_first=True, norm_first=True), L)
        s.out = nn.Linear(d, 1)
    def forward(s, xn, xc, tt, *a):
        tok = xn.unsqueeze(-1)*s.w + s.b
        ct = torch.stack([e(xc[:,i]) for i,e in enumerate(s.ce)], 1)
        z = torch.cat([s.cls.expand(xn.shape[0],-1,-1), tok, ct], 1)
        return s.out(s.tr(z)[:,0]).squeeze(-1)

class HierShrink(nn.Module):
    """정밀도 가중 계층 축소 — 이 문제의 통계 구조를 직접 인코딩"""
    def __init__(s, n_num, card, np_ev=7, nb_ev=3, h=128, drop=.1, **kw):
        super().__init__()
        s.tp = nn.Embedding(np_ev, 8); s.tb = nn.Embedding(nb_ev, 8)
        s.gp = nn.Sequential(nn.Linear(9,32), nn.ReLU(), nn.Linear(32,1))
        s.gb = nn.Sequential(nn.Linear(9,32), nn.ReLU(), nn.Linear(32,1))
        s.a = nn.Parameter(torch.tensor(1.0)); s.bb = nn.Parameter(torch.tensor(-0.3))
        s.sit = nn.Sequential(nn.Linear(n_num,h), nn.ReLU(), nn.Dropout(drop),
                              nn.Linear(h,h), nn.ReLU(), nn.Linear(h,1))
        s.tm = nn.Sequential(nn.Linear(5,16), nn.ReLU(), nn.Linear(16,1))
        s.c = nn.Parameter(torch.zeros(1))
        s.ip = torch.arange(np_ev); s.ib = torch.arange(nb_ev)
    def pool(s, r, n, temb, g):
        B, K = r.shape
        f = torch.cat([n.unsqueeze(-1), temb.unsqueeze(0).expand(B,-1,-1)], -1)
        w = torch.softmax(g(f).squeeze(-1) + torch.log1p(n), 1)      # 정밀도 ∝ 표본수
        return (w*r).sum(1)
    def forward(s, xn, xc, tt, pr, pn, br, bn):
        th_p = s.pool(pr, pn, s.tp(s.ip.to(pr.device)), s.gp)
        th_b = s.pool(br, bn, s.tb(s.ib.to(br.device)), s.gb)
        return s.a*th_p + s.bb*th_b + s.sit(xn).squeeze(-1) + s.tm(tt).squeeze(-1) + s.c

ARCH = {'TabM': TabM, 'FTT': FTT, 'Hier': HierShrink}

# ══════════════ 학습 ══════════════
def train_eval(cfg, vs, seed=0, max_ep=120, patience=8, budget=900):
    tr, va = split(vs)
    ia = np.where(tr)[0]; rs = np.random.RandomState(1); rs.shuffle(ia)
    nin = int(len(ia)*.06); iin, itr = ia[:nin], ia[nin:]; iva = np.where(va)[0]
    torch.manual_seed(seed); np.random.seed(seed)
    net = ARCH[cfg['arch']](Xn.shape[1], CARD, **{k: v for k, v in cfg.items() if k != 'arch'
                            and k not in ('lr','wd','bs')}).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    # GPU 메모리에 맞춰 배치 자동 조절 (TabM 의 (B,k,h) 텐서가 지배적)
    mem = torch.cuda.get_device_properties(DEV).total_memory/2**30
    budget_act = 1.2e8 if mem < 10 else 3.0e8
    sz = cfg.get('k',1)*cfg.get('h', cfg.get('d',64))*cfg.get('L',2)
    B = int(np.clip(budget_act/max(sz,1), 1024, cfg['bs']))
    best, bad, bstate = 1e9, 0, None
    t_start = time.time()
    for ep in range(max_ep):
        net.train(); perm = np.random.RandomState(ep).permutation(itr)
        for j in range(0, len(perm), B):
            b = torch.tensor(perm[j:j+B], device=DEV)
            p = torch.sigmoid(net(TN[b], TC[b], TT[b], TPR[b], TPN[b], TBR[b], TBN[b]))
            loss = ((p - TY[b])**2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            se, cnt = 0.0, 0
            for q in range(0, len(iin), B):          # 내부검증도 배치 분할 (OOM 방지)
                bi = torch.tensor(iin[q:q+B], device=DEV)
                se += ((torch.sigmoid(net(TN[bi],TC[bi],TT[bi],TPR[bi],TPN[bi],TBR[bi],TBN[bi]))-TY[bi])**2).sum().item()
                cnt += len(bi)
            v = se/cnt
        if v < best - 1e-8:
            best, bad = v, 0
            bstate = {k: q.detach().clone() for k, q in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
        if time.time()-t_start > budget: break
    net.load_state_dict(bstate); net.eval(); ps = []
    with torch.no_grad():
        for j in range(0, len(iva), B):
            b = torch.tensor(iva[j:j+B], device=DEV)
            ps.append(torch.sigmoid(net(TN[b],TC[b],TT[b],TPR[b],TPN[b],TBR[b],TBN[b])).cpu().numpy())
    return np.concatenate(ps), evaluate(np.concatenate(ps), y[iva], RP[vs]), ep+1

# ══════════════ 설정 공간 ══════════════
CFGS = []
for k in [8, 32, 64]:
    for h in [128, 256, 512]:
        for L in [2, 3]:
            for wd in [1e-4, 1e-3, 1e-2]:
                for plr in [False, True]:
                    CFGS.append(dict(arch='TabM', k=k, h=h, L=L, emb=0, drop=0.1,
                                     plr=plr, film=True, lr=2e-3, wd=wd, bs=16384))
for d in [32, 64]:
    for L in [2, 3, 4]:
        for wd in [1e-4, 1e-3]:
            CFGS.append(dict(arch='FTT', d=d, L=L, heads=4, drop=0.1, lr=1e-3, wd=wd, bs=8192))
for h in [64, 128, 256]:
    for wd in [1e-4, 1e-3, 1e-2]:
        for lr in [1e-3, 3e-3]:
            CFGS.append(dict(arch='Hier', h=h, drop=0.1, lr=lr, wd=wd, bs=16384))
rng = np.random.RandomState(0); rng.shuffle(CFGS)
MINE = CFGS[SHARD::NSH]
log(f'{el()} 전체 {len(CFGS)} 설정 중 이 샤드 {len(MINE)}개')

R = []
best_avg = None
for ci, cfg in enumerate(MINE):
    try:
        sc = {}
        for vs in FOLDS:
            p, r, ep = train_eval(cfg, vs)
            sc[vs] = r
            np.save(OUT + f's{SHARD}_c{ci}_{vs}.npy', p.astype(np.float32))
        avg = np.mean([sc[v]['raw'] for v in FOLDS])
        rec = dict(shard=SHARD, ci=ci, avg=avg, **{f'{k}_{v}': q for v in sc for k, q in sc[v].items()},
                   **{k: v for k, v in cfg.items()})
        R.append(rec); pd.DataFrame(R).to_csv(OUT + f'res_s{SHARD}.csv', index=False)
        f = ''
        if best_avg is None or avg > best_avg: best_avg = avg; f = '  <== BEST'
        log(f'{el()} [{ci:3d}] {cfg["arch"]:5s} avg={avg:7.1f} '
            f'24:{sc[2024]["raw"]:7.1f} 22:{sc[2022]["raw"]:7.1f} '
            f'| {" ".join(f"{k}{v}" for k,v in cfg.items() if k not in ("arch","bs","drop","film","emb"))}{f}')
    except Exception:
        log(f'!! cfg{ci}\n' + traceback.format_exc())
log(f'\n{el()} ===== shard {SHARD} 완료 =====')
