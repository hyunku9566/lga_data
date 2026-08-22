"""
2차 실험 — 1차에서 배운 것 반영
  L1. 이 문제의 지배적 실패모드는 과적합. 모든 하이퍼파라미터가 "용량 최소화"를 가리킴
  L2. 1차 피처셋 대조는 과적합 파라미터로 돌려서 무효 -> 좋은 파라미터로 재판정
  L3. 최적점이 탐색 경계(lr=0.02, trees=500)에 걸림 -> 경계 밖으로 확장
  L4. 신경망 전멸 원인 = 조기종료 없음 + 과용량 -> 소형화 + 조기종료

Stage A 피처셋 재판정 / B 탐색확장 / C 신경망 재설계 / D 축소·블렌딩
"""
import os, time, warnings, traceback
import numpy as np, pandas as pd, scipy.special as sp
warnings.filterwarnings('ignore')

D = '/home/lee/lga/'; OUT = D + 'results2/'
os.makedirs(OUT, exist_ok=True)
LOG = open(OUT + 'log.txt', 'a', buffering=1)
def log(*a):
    m = ' '.join(str(x) for x in a); print(m); LOG.write(m + '\n')
T0 = time.time()
def el(): return f'[{time.time()-T0:7.1f}s]'

X = pd.read_parquet(D + 'features.parquet')
y = X.__y.values; season = X.__season.values; gtF = X.__gt_F.values.astype(bool)
COLS = [c for c in X.columns if not c.startswith('__')]
log(f'{el()} 로드 {X.shape}, 피처 {len(COLS)}')

def evaluate(p, yv, rp):
    r = yv.mean(); ref = r * (1 - r)
    b = lambda q: 100000 * max(0.0, 1 - np.mean((q - yv) ** 2) / ref)
    lo = sp.logit(np.clip(p, 1e-6, 1 - 1e-6))
    return dict(raw=b(p), trend=b(sp.expit(lo - lo.mean() + sp.logit(rp))),
                oracle=b(sp.expit(lo - lo.mean() + sp.logit(r))), meanp=float(p.mean()))

def trend_prior(vs):
    m = season < vs; s = pd.Series(y[m]).groupby(season[m]).mean()
    return float(sp.expit(np.polyval(np.polyfit(s.index, sp.logit(s.values), 1), vs)))

SP = {vs: ((season < vs) & ~(gtF & (season <= 2022)), season == vs) for vs in (2024, 2023)}
RP = {vs: trend_prior(vs) for vs in (2024, 2023)}
R = []
def save(): pd.DataFrame(R).to_csv(OUT + 'results2.csv', index=False)

# 1차 우승 파라미터
BEST = dict(n_estimators=500, learning_rate=0.02, max_depth=5, min_child_weight=400,
            subsample=0.6, colsample_bytree=0.8, reg_lambda=100.0,
            device='cuda', tree_method='hist', eval_metric='logloss', verbosity=0)

FEATSETS = {
    'base48':    [c for c in COLS if not c.startswith(('p_', 'b_', 'tm_', 'hand_mix'))],
    'deconv':    [c for c in COLS if not c.startswith('tm_') and 'clutch' not in c and 'ctrl' not in c],
    'deconv+tm': [c for c in COLS if 'clutch' not in c and 'ctrl' not in c],
    'deconv+cl': [c for c in COLS if not c.startswith('tm_')],
    'full':      COLS,
}

# ════════ Stage A. 피처셋 재판정 (좋은 파라미터로) ════════
def stageA():
    import xgboost as xgb
    log(f'\n{el()} ===== Stage A: 피처셋 재판정 (BEST 파라미터) =====')
    for fs, cc in FEATSETS.items():
        for vs, (tr, va) in SP.items():
            try:
                m = xgb.XGBClassifier(**BEST).fit(X.loc[tr, cc], y[tr])
                p = m.predict_proba(X.loc[va, cc])[:, 1]
                r = evaluate(p, y[va], RP[vs]); r.update(stage='A', name=fs, val=vs, nfeat=len(cc))
                R.append(r); np.save(OUT + f'pA_{fs}_{vs}.npy', p)
                log(f'{el()} [A] {fs:11s}({len(cc):2d}) val{vs} raw={r["raw"]:7.1f} trend={r["trend"]:7.1f} oracle={r["oracle"]:7.1f}')
            except Exception: log(f'!! A {fs}/{vs}\n' + traceback.format_exc())
    save()

# ════════ Stage B. 경계 밖으로 탐색 확장 ════════
def stageB():
    import xgboost as xgb
    log(f'\n{el()} ===== Stage B: 저용량 영역 탐색 80회 =====')
    rng = np.random.RandomState(7); best = None
    cc = FEATSETS['deconv']
    for i in range(80):
        prm = dict(n_estimators=int(rng.choice([150, 250, 400, 600, 900])),
                   learning_rate=float(rng.choice([0.005, 0.008, 0.012, 0.02, 0.03])),
                   max_depth=int(rng.choice([3, 4, 5, 6])),
                   min_child_weight=float(rng.choice([200, 600, 1500, 4000])),
                   subsample=float(rng.choice([0.5, 0.7, 0.9])),
                   colsample_bytree=float(rng.choice([0.3, 0.5, 0.7])),
                   reg_lambda=float(rng.choice([10, 50, 200, 800])),
                   reg_alpha=float(rng.choice([0, 1, 10])),
                   device='cuda', tree_method='hist', eval_metric='logloss', verbosity=0)
        try:
            sc = {}
            for vs, (tr, va) in SP.items():
                m = xgb.XGBClassifier(**prm).fit(X.loc[tr, cc], y[tr])
                p = m.predict_proba(X.loc[va, cc])[:, 1]
                sc[vs] = evaluate(p, y[va], RP[vs])
                np.save(OUT + f'pB_{i}_{vs}.npy', p)
            avg = (sc[2024]['raw'] + sc[2023]['raw']) / 2
            rec = dict(stage='B', i=i, avg_raw=avg,
                       **{f'{k}_{vs}': v for vs in sc for k, v in sc[vs].items()},
                       **{k: v for k, v in prm.items() if k not in ('device','tree_method','eval_metric','verbosity')})
            R.append(rec)
            f = ''
            if best is None or avg > best: best = avg; f = '  <== BEST'
            log(f'{el()} [B{i:2d}] avg={avg:7.1f} (24:{sc[2024]["raw"]:6.1f}/23:{sc[2023]["raw"]:6.1f}) '
                f'd{prm["max_depth"]} lr{prm["learning_rate"]} n{prm["n_estimators"]} mcw{prm["min_child_weight"]:.0f} '
                f'L2:{prm["reg_lambda"]:.0f} L1:{prm["reg_alpha"]:.0f}{f}')
        except Exception: log(f'!! B{i}\n' + traceback.format_exc())
        if i % 10 == 9: save()
    save()

# ════════ Stage C. 신경망 재설계 (소형 + 조기종료) ════════
def stageC():
    import torch, torch.nn as nn
    log(f'\n{el()} ===== Stage C: 소형 TabM + 조기종료 =====')
    dev = 'cuda:0'
    CAT = ['pitcher_id','batter_id','pitcher_team_id','batter_team_id',
           'pitcher_hand','batter_hand','base_state','game_type','top_bottom']
    cc = FEATSETS['deconv']
    num = [c for c in cc if c not in CAT]
    Xc = X[CAT].values.astype(np.int64)
    Xn = np.nan_to_num(X[num].values.astype(np.float32), nan=0., posinf=0., neginf=0.)
    Xn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-6)
    Xn = np.clip(Xn, -5, 5)
    card = [int(Xc[:, i].max()) + 2 for i in range(len(CAT))]
    Tf = np.stack([np.sin(2*np.pi*X.game_month.values/12), np.cos(2*np.pi*X.game_month.values/12),
                   np.sin(4*np.pi*X.game_month.values/12), np.cos(4*np.pi*X.game_month.values/12)],1).astype(np.float32)
    tn = torch.tensor(Xn, device=dev); tc = torch.tensor(Xc, device=dev)
    tt = torch.tensor(Tf, device=dev); ty = torch.tensor(y, device=dev)

    class Net(nn.Module):
        """소형 TabM. emb=0 이면 투수/타자 임베딩 없이 (암기 차단)"""
        def __init__(s, n_num, card, k=8, h=64, emb=0, drop=0.2, film=True):
            super().__init__(); s.k, s.emb, s.film = k, emb, film
            d_in = n_num
            if emb:
                s.e1 = nn.Embedding(card[0], emb); s.e2 = nn.Embedding(card[1], emb)
                nn.init.normal_(s.e1.weight, std=.01); nn.init.normal_(s.e2.weight, std=.01)
                d_in += 2 * emb
            s.r1 = nn.Parameter(torch.randn(k, d_in) * .1 + 1)
            s.l1 = nn.Linear(d_in, h); s.dp = nn.Dropout(drop); s.l2 = nn.Linear(h, h)
            s.hd = nn.Parameter(torch.randn(k, h) * .02); s.hb = nn.Parameter(torch.zeros(k))
            if film: s.fn = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2 * h))
        def forward(s, xn, xc, t):
            z = xn
            if s.emb: z = torch.cat([z, s.e1(xc[:,0]), s.e2(xc[:,1])], -1)
            z = z.unsqueeze(1) * s.r1
            z = s.dp(torch.relu(s.l1(z)))
            if s.film:
                g, b = s.fn(t).chunk(2, -1); z = z * (1 + g.unsqueeze(1)) + b.unsqueeze(1)
            z = torch.relu(s.l2(z))
            return ((z * s.hd).sum(-1) + s.hb).mean(1)

    CFG = [dict(k=8,  h=64,  emb=0, wd=1e-2, drop=0.2, tag='소형-임베딩없음'),
           dict(k=8,  h=64,  emb=4, wd=1e-2, drop=0.2, tag='소형-임베딩4'),
           dict(k=16, h=128, emb=0, wd=1e-2, drop=0.3, tag='중형-임베딩없음'),
           dict(k=16, h=128, emb=4, wd=3e-2, drop=0.3, tag='중형-강한정규화'),
           dict(k=4,  h=32,  emb=0, wd=1e-3, drop=0.1, tag='초소형'),
           dict(k=8,  h=64,  emb=0, wd=1e-2, drop=0.2, film=False, tag='소형-FiLM없음')]

    for cfg in CFG:
        for vs, (tr, va) in SP.items():
            try:
                torch.manual_seed(0)
                itr_all = np.where(tr)[0]
                rs = np.random.RandomState(1); rs.shuffle(itr_all)
                nin = int(len(itr_all) * 0.05)
                iin, itr = itr_all[:nin], itr_all[nin:]      # 조기종료용 내부 홀드아웃(학습 시즌에서만)
                iva = np.where(va)[0]
                net = Net(Xn.shape[1], card, k=cfg['k'], h=cfg['h'], emb=cfg['emb'],
                          drop=cfg['drop'], film=cfg.get('film', True)).to(dev)
                opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=cfg['wd'])
                B = 16384; bestv, bad, bstate = 1e9, 0, None
                for ep in range(40):
                    net.train(); perm = np.random.RandomState(ep).permutation(itr)
                    for j in range(0, len(perm), B):
                        b = torch.tensor(perm[j:j+B], device=dev)
                        loss = ((torch.sigmoid(net(tn[b], tc[b], tt[b])) - ty[b]) ** 2).mean()
                        opt.zero_grad(); loss.backward(); opt.step()
                    net.eval()
                    with torch.no_grad():
                        bi = torch.tensor(iin, device=dev)
                        v = ((torch.sigmoid(net(tn[bi], tc[bi], tt[bi])) - ty[bi]) ** 2).mean().item()
                    if v < bestv - 1e-7:
                        bestv, bad = v, 0
                        bstate = {k: t.detach().clone() for k, t in net.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= 4: break
                net.load_state_dict(bstate); net.eval()
                ps = []
                with torch.no_grad():
                    for j in range(0, len(iva), 32768):
                        b = torch.tensor(iva[j:j+32768], device=dev)
                        ps.append(torch.sigmoid(net(tn[b], tc[b], tt[b])).cpu().numpy())
                p = np.concatenate(ps)
                r = evaluate(p, y[va], RP[vs]); r.update(stage='C', name=cfg['tag'], val=vs, epochs=ep+1)
                R.append(r); np.save(OUT + f'pC_{cfg["tag"]}_{vs}.npy', p)
                log(f'{el()} [C] {cfg["tag"]:16s} val{vs} ep={ep+1:2d} raw={r["raw"]:7.1f} '
                    f'trend={r["trend"]:7.1f} oracle={r["oracle"]:7.1f} meanp={r["meanp"]:.4f} sd={p.std():.4f}')
            except Exception: log(f'!! C {cfg["tag"]}/{vs}\n' + traceback.format_exc())
            save()

# ════════ Stage D. 로짓 축소 + 블렌딩 ════════
def stageD():
    log(f'\n{el()} ===== Stage D: 축소 / 시드앙상블 / 블렌딩 =====')
    import xgboost as xgb
    df = pd.DataFrame(R)
    b = df[df.stage == 'B'].sort_values('avg_raw', ascending=False)
    if not len(b): log('B 결과 없음, skip'); return
    i = int(b.iloc[0]['i'])
    log(f'{el()} 최고 설정 B{i} 사용')

    # D1. 로짓 축소 스윕
    for vs, (tr, va) in SP.items():
        p = np.load(OUT + f'pB_{i}_{vs}.npy'); yv = y[va]
        lo = sp.logit(np.clip(p, 1e-6, 1-1e-6)); m = lo.mean()
        for s in [0.7, 0.85, 1.0, 1.15, 1.3]:
            r = evaluate(sp.expit(m + s*(lo-m)), yv, RP[vs])
            R.append(dict(stage='D_shrink', shrink=s, val=vs, **r))
            log(f'{el()} [D축소] val{vs} s={s:.2f} raw={r["raw"]:7.1f} trend={r["trend"]:7.1f}')

    # D2. 시드 앙상블
    prm = {k: b.iloc[0][k] for k in ['n_estimators','learning_rate','max_depth','min_child_weight',
                                      'subsample','colsample_bytree','reg_lambda','reg_alpha']}
    prm = {k: (int(v) if k in ('n_estimators','max_depth') else float(v)) for k, v in prm.items()}
    prm.update(device='cuda', tree_method='hist', eval_metric='logloss', verbosity=0)
    cc = FEATSETS['deconv']
    for vs, (tr, va) in SP.items():
        acc = []
        for sd in range(5):
            m = xgb.XGBClassifier(**prm, random_state=sd).fit(X.loc[tr, cc], y[tr])
            acc.append(m.predict_proba(X.loc[va, cc])[:, 1])
        p = np.mean(acc, 0); np.save(OUT + f'pD_seed_{vs}.npy', p)
        r = evaluate(p, y[va], RP[vs]); r.update(stage='D_seed5', val=vs)
        R.append(r); log(f'{el()} [D시드5] val{vs} raw={r["raw"]:7.1f} trend={r["trend"]:7.1f} oracle={r["oracle"]:7.1f}')

    # D3. XGB x NN 블렌딩
    for vs, (tr, va) in SP.items():
        px = np.load(OUT + f'pD_seed_{vs}.npy')
        for f in os.listdir(OUT):
            if f.startswith('pC_') and f.endswith(f'_{vs}.npy'):
                pn = np.load(OUT + f); nm = f[3:-9]
                for w in [0.1, 0.2, 0.3, 0.5]:
                    lo = (1-w)*sp.logit(np.clip(px,1e-6,1-1e-6)) + w*sp.logit(np.clip(pn,1e-6,1-1e-6))
                    r = evaluate(sp.expit(lo), y[va], RP[vs])
                    R.append(dict(stage='D_blend', nn=nm, w=w, val=vs, **r))
                    log(f'{el()} [D블렌드] val{vs} {nm:16s} w={w:.1f} raw={r["raw"]:7.1f} trend={r["trend"]:7.1f}')
    save()

for fn in (stageA, stageB, stageC, stageD):
    try: fn()
    except Exception: log(f'!! {fn.__name__}\n' + traceback.format_exc())
save()
log(f'\n{el()} ===== 2차 완료 =====')
