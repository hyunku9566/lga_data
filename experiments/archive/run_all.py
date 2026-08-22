"""
LG Aimers 9기 — 전체 실험 파이프라인 (장시간 백그라운드용)

원칙 (전부 코드로 강제):
  R1. 시즌 s 행의 baked 피처는 season <= s-1 데이터로만 계산  (test 2025 는 <=2024 만 가짐)
  R2. 트랙맨은 투수 단위 집계로만 내림. 현재 투구 측정값을 행에 붙이지 않음
  R3. 역산은 (현재 행 + 시즌시작 앵커) 만 사용 — 다른 평가행 미사용
  R4. 2019-2022 game_type=F 는 학습에서 제외 (ABS 이전 라벨 체제가 다름)

Stage 1 피처 빌드 -> 2 XGB 스윕(GPU) -> 3 TabM+FiLM(GPU) -> 4 블렌딩 -> 5 리포트
"""
import os, sys, json, time, warnings, traceback
import numpy as np, pandas as pd, scipy.special as sp
warnings.filterwarnings('ignore')

D = '/home/lee/lga/'
OUT = D + 'results/'
os.makedirs(OUT, exist_ok=True)
LOG = open(OUT + 'log.txt', 'a', buffering=1)
def log(*a):
    m = ' '.join(str(x) for x in a)
    print(m); LOG.write(m + '\n')

T0 = time.time()
def el(): return f'[{time.time()-T0:7.1f}s]'

# ════════════════════════════════════════════════════════════════
# Stage 1. 피처 빌드
# ════════════════════════════════════════════════════════════════
FEAT_CACHE = D + 'features.parquet'

def stage1():
    log(f'\n{el()} ===== Stage 1: 피처 빌드 =====')
    df = pd.read_csv(D + 'data/train.csv', encoding='utf-8-sig')
    raw_gt = df.game_type.values.copy()
    y = df.control_success.values.astype(np.float32)
    season = df.season.values
    log(f'{el()} train {df.shape}')

    # ---------- 1a. 경기/등판 복원 (train 은 시간순 정렬) ----------
    gid = (df.inning.diff().fillna(0) < 0).cumsum().values
    df['_gid'] = gid
    app_sz = df.groupby(['_gid', 'pitcher_id'], sort=False).size()
    # 역할(등판당 투구수)을 season 별 as-of 로 (R1)
    aps = df.groupby([df._gid, df.pitcher_id, df.season], sort=False).size().reset_index(name='np_')
    ppa_ss = aps.groupby(['pitcher_id', 'season']).np_.median().reset_index()
    ppa_asof = {}
    for s in range(2019, 2026):
        prev = ppa_ss[ppa_ss.season < s]
        ppa_asof[s] = prev.groupby('pitcher_id').np_.median() if len(prev) else pd.Series(dtype=float)
    log(f'{el()} 경기 {len(np.unique(gid))} 복원, 역할 as-of 완료')

    # ---------- 1b. 역산 (R3) ----------
    F = {}
    def sss(idcol, ncol, ratecol):
        t = df[[idcol, 'season', ncol, ratecol]].copy()
        t['succ'] = t[ncol] * t[ratecol].fillna(0)
        return t.loc[t.groupby([idcol, 'season'])[ncol].idxmin()].set_index([idcol, 'season'])[[ncol, 'succ']]

    specs = [('pitcher_id', 'asof_pitcher_n', 'asof_pitcher_success_rate', 'p_succ'),
             ('pitcher_id', 'asof_pitcher_n', 'asof_pitcher_reverse_rate', 'p_rev'),
             ('pitcher_id', 'asof_pitcher_n', 'asof_pitcher_middle_rate', 'p_mid'),
             ('pitcher_id', 'asof_pitcher_n', 'asof_pitcher_ball_rate', 'p_ball'),
             ('pitcher_id', 'asof_pitcher_n', 'asof_pitcher_strike_rate', 'p_stk'),
             ('batter_id', 'asof_batter_n', 'asof_batter_success_rate', 'b_succ'),
             ('batter_id', 'asof_batter_n', 'asof_batter_middle_rate', 'b_mid')]
    K = 150.0
    for idcol, ncol, ratecol, pref in specs:
        S = sss(idcol, ncol, ratecol)
        a = df[[idcol, 'season']].join(S, on=[idcol, 'season'])
        a_n = a[ncol].fillna(0).values; a_s = a['succ'].fillna(0).values
        n = df[ncol].values; r = df[ratecol].values
        dn = np.maximum(n - a_n, 0); ds = np.maximum(np.nan_to_num(n * r) - a_s, 0)
        pri = np.nanmean(r)
        F[pref + '_ssn'] = (ds + K * pri) / (dn + K)
        F[pref + '_ssn_vs_car'] = F[pref + '_ssn'] - np.nan_to_num(r, nan=pri)
        if pref in ('p_succ', 'b_succ'):
            F[pref + '_ssn_n'] = dn
    F['p_prev5_vs_car'] = (df.asof_pitcher_prev5_game_success_rate - df.asof_pitcher_success_rate).values
    F['p_prev1_vs_prev5'] = (df.asof_pitcher_prev1_game_success_rate - df.asof_pitcher_prev5_game_success_rate).values
    log(f'{el()} 역산 {len(F)}개')

    # ---------- 1c. 피로도 ----------
    ppa_v = np.array([ppa_asof[s].get(p, np.nan) for s, p in zip(season, df.pitcher_id.values)])
    med = np.nanmedian(ppa_v); ppa_v = np.where(np.isnan(ppa_v), med, ppa_v)
    F['p_ppa'] = ppa_v
    F['p_est_apps'] = F['p_succ_ssn_n'] / np.clip(ppa_v, 5, None)
    F['p_inning_x_role'] = df.inning.values * np.log1p(ppa_v)
    F['p_ssn_per_month'] = F['p_succ_ssn_n'] / np.clip(df.game_month.values, 3, None)

    # ---------- 1d. 압박 성향 + 제구형 지수 (R1: as-of s-1, 축소 적용) ----------
    hi = (df.li > 1.5).values; lo = (df.li < 0.5).values
    risp = ((df.runner_on_2b | df.runner_on_3b) == 1).values
    tmp = pd.DataFrame({'pid': df.pitcher_id.values, 's': season, 'y': y,
                        'hi': hi, 'lo': lo, 'risp': risp,
                        'ball': df.asof_pitcher_ball_rate.values,
                        'mid': df.asof_pitcher_middle_rate.values})
    SH = 400.0  # 축소 강도 (r~0.19 반영해 강하게)
    clutch = np.zeros(len(df), np.float32); ctrl = np.zeros(len(df), np.float32)
    for s in range(2019, 2026):
        prev = tmp[tmp.s < s]
        m = season == s
        if not m.any(): continue
        if len(prev) > 0:
            g = prev.groupby('pid')
            a = g.apply(lambda x: (x.y[x.hi].sum() + SH * x.y.mean()) / (x.hi.sum() + SH)
                                  - (x.y[x.lo].sum() + SH * x.y.mean()) / (x.lo.sum() + SH))
            b = g.apply(lambda x: (x.y[x.risp].sum() + SH * x.y.mean()) / (x.risp.sum() + SH)
                                  - (x.y[~x.risp].sum() + SH * x.y.mean()) / ((~x.risp).sum() + SH))
            pr = g[['ball', 'mid']].mean()
            ci = -(pr.ball.rank(pct=True) + pr.mid.rank(pct=True)) / 2
            pid_m = df.pitcher_id.values[m]
            clutch[m] = pd.Series(pid_m).map(a).fillna(0).values
            ctrl[m] = pd.Series(pid_m).map(ci).fillna(-0.5).values
            F.setdefault('p_clutch_risp', np.zeros(len(df), np.float32))
            F['p_clutch_risp'][m] = pd.Series(pid_m).map(b).fillna(0).values
    F['p_clutch_li'] = clutch
    F['p_ctrl_idx'] = ctrl
    F['p_ctrl_x_li'] = ctrl * df.li.values            # "극한상황일수록 제구형이 강해진다"
    F['p_clutch_x_li'] = clutch * df.li.values
    log(f'{el()} 압박성향/제구형 완료')

    # ---------- 1e. 트랙맨 구위 (R1 + R2) ----------
    try:
        pm = pd.read_csv(D + 'pitcher_map.csv')
        pm = pm[pm.conf >= 0.90]
        tm = pd.read_csv(D + 'data/trackman_history.csv', encoding='utf-8-sig',
                         usecols=['season', 'pitcher_trackman_id', 'pitch_type_group', 'rel_speed',
                                  'spin_rate', 'induced_vert_break', 'horz_break', 'extension',
                                  'rel_height', 'rel_side'])
        tm = tm.merge(pm[['pitcher_id', 'pitcher_trackman_id']], on='pitcher_trackman_id')
        log(f'{el()} 트랙맨 조인 {len(tm):,}행, 투수 {tm.pitcher_id.nunique()}명')

        # 투수x시즌 요약 -> 이후 s-1 까지 누적
        agg = tm.groupby(['pitcher_id', 'season']).agg(
            velo=('rel_speed', 'mean'), velo_sd=('rel_speed', 'std'),
            velo_max=('rel_speed', lambda x: x.quantile(0.95)),
            spin=('spin_rate', 'mean'), ivb=('induced_vert_break', 'mean'),
            hb=('horz_break', 'mean'), hb_abs=('horz_break', lambda x: x.abs().mean()),
            ext=('extension', 'mean'),
            # 커맨드 재현성: 릴리스 포인트 산포 (작을수록 반복성 높음)
            rel_h_sd=('rel_height', 'std'), rel_s_sd=('rel_side', 'std'), ext_sd=('extension', 'std'),
            n=('rel_speed', 'size')).reset_index()
        fb = tm[tm.pitch_type_group == 'fastball'].groupby(['pitcher_id', 'season']).agg(
            fb_velo=('rel_speed', 'mean'), fb_spin=('spin_rate', 'mean'),
            fb_ivb=('induced_vert_break', 'mean')).reset_index()
        agg = agg.merge(fb, on=['pitcher_id', 'season'], how='left')
        arse = tm.groupby(['pitcher_id', 'season']).pitch_type_group.nunique().rename('arsenal').reset_index()
        agg = agg.merge(arse, on=['pitcher_id', 'season'], how='left')

        tcols = [c for c in agg.columns if c not in ('pitcher_id', 'season')]
        for c in tcols: F['tm_' + c] = np.full(len(df), np.nan, np.float32)
        for s in range(2019, 2026):
            m = season == s
            if not m.any(): continue
            prev = agg[agg.season < s]
            if not len(prev): continue
            w = prev.groupby('pitcher_id').apply(
                lambda x: pd.Series({c: np.average(x[c], weights=x.n) if x[c].notna().any() else np.nan
                                     for c in tcols}))
            pid_m = pd.Series(df.pitcher_id.values[m])
            for c in tcols:
                F['tm_' + c][m] = pid_m.map(w[c]).values
        log(f'{el()} 트랙맨 구위 {len(tcols)}개 (as-of s-1)')
    except Exception:
        log(f'{el()} !! 트랙맨 실패:\n' + traceback.format_exc())

    # ---------- 1f. 조립 ----------
    CATS = ['top_bottom', 'game_type', 'base_state', 'pitcher_hand', 'batter_hand',
            'pitcher_team_id', 'batter_team_id', 'pitcher_id', 'batter_id']
    X = df.drop(columns=['row_id', 'control_success', 'asof_pitcher_pitchmix_n', '_gid'], errors='ignore')
    for c in CATS: X[c] = X[c].astype('category').cat.codes
    X['hand_mix'] = X.pitcher_hand * 2 + X.batter_hand
    for k, v in F.items(): X[k] = np.asarray(v, np.float32)
    X = X.astype(np.float32)
    X['__y'] = y; X['__gt_F'] = (raw_gt == 'F').astype(np.int8); X['__season'] = season
    X.to_parquet(FEAT_CACHE)
    log(f'{el()} 저장 {FEAT_CACHE}  shape={X.shape}')
    return X


# ════════════════════════════════════════════════════════════════
# 평가 유틸
# ════════════════════════════════════════════════════════════════
def evaluate(p, yv, r_prior):
    r = yv.mean(); ref = r * (1 - r)
    b = lambda q: 100000 * max(0.0, 1 - np.mean((q - yv) ** 2) / ref)
    lo = sp.logit(np.clip(p, 1e-6, 1 - 1e-6))
    return dict(raw=b(p), trend=b(sp.expit(lo - lo.mean() + sp.logit(r_prior))),
                oracle=b(sp.expit(lo - lo.mean() + sp.logit(r))), meanp=float(p.mean()))

def trend_prior(y, season, vs):
    m = season < vs
    s = pd.Series(y[m]).groupby(season[m]).mean()
    return float(sp.expit(np.polyval(np.polyfit(s.index, sp.logit(s.values), 1), vs)))

def splits(X):
    s = X.__season.values; f = X.__gt_F.values.astype(bool)
    out = {}
    for vs in (2024, 2023):
        tr = (s < vs) & ~(f & (s <= 2022))     # R4
        out[vs] = (tr, s == vs)
    return out


# ════════════════════════════════════════════════════════════════
# Stage 2. XGBoost 스윕 (GPU)
# ════════════════════════════════════════════════════════════════
FEATSETS = {
    'base48':   lambda c: [x for x in c if not x.startswith(('p_', 'b_', 'tm_', 'hand_mix'))],
    'deconv':   lambda c: [x for x in c if not x.startswith('tm_')],
    'full':     lambda c: c,
    'no_clutch':lambda c: [x for x in c if 'clutch' not in x and 'ctrl' not in x],
    'no_tm':    lambda c: [x for x in c if not x.startswith('tm_')],
}

def stage2(X, results):
    import xgboost as xgb
    log(f'\n{el()} ===== Stage 2: XGBoost GPU 스윕 =====')
    y = X.__y.values; season = X.__season.values
    cols = [c for c in X.columns if not c.startswith('__')]
    SP = splits(X)
    rng = np.random.RandomState(0)

    # 2a. 피처셋 비교 (고정 하이퍼파라미터)
    base = dict(n_estimators=800, learning_rate=0.04, max_depth=6, min_child_weight=200,
                subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0, device='cuda',
                tree_method='hist', eval_metric='logloss', verbosity=0)
    for fs, fn in FEATSETS.items():
        cc = fn(cols)
        for vs, (tr, va) in SP.items():
            try:
                m = xgb.XGBClassifier(**base).fit(X.loc[tr, cc], y[tr])
                p = m.predict_proba(X.loc[va, cc])[:, 1]
                r = evaluate(p, y[va], trend_prior(y, season, vs))
                r.update(stage='featset', name=fs, val=vs, nfeat=len(cc))
                results.append(r); log(f'{el()} [featset] {fs:10s} val{vs} raw={r["raw"]:7.1f} trend={r["trend"]:7.1f} oracle={r["oracle"]:7.1f}')
            except Exception:
                log(f'{el()} !! {fs}/{vs}\n' + traceback.format_exc())

    # 2b. 랜덤 서치 (full 피처셋)
    log(f'\n{el()} --- 랜덤 서치 60회 ---')
    best = None
    for i in range(60):
        prm = dict(n_estimators=int(rng.choice([500, 800, 1200, 2000])),
                   learning_rate=float(rng.choice([0.02, 0.03, 0.05, 0.08])),
                   max_depth=int(rng.choice([4, 5, 6, 7, 8])),
                   min_child_weight=float(rng.choice([50, 150, 400, 1000])),
                   subsample=float(rng.choice([0.6, 0.8, 1.0])),
                   colsample_bytree=float(rng.choice([0.4, 0.6, 0.8])),
                   reg_lambda=float(rng.choice([1, 5, 20, 100])),
                   device='cuda', tree_method='hist', eval_metric='logloss', verbosity=0)
        try:
            sc = {}
            for vs, (tr, va) in SP.items():
                m = xgb.XGBClassifier(**prm).fit(X.loc[tr, cols], y[tr])
                p = m.predict_proba(X.loc[va, cols])[:, 1]
                sc[vs] = evaluate(p, y[va], trend_prior(y, season, vs))
                if vs == 2024: np.save(OUT + f'pred_xgb_{i}_2024.npy', p)
            avg = (sc[2024]['raw'] + sc[2023]['raw']) / 2
            rec = dict(stage='search', i=i, avg_raw=avg, **{f'{k}_{vs}': v
                       for vs in sc for k, v in sc[vs].items()}, **{k: v for k, v in prm.items()
                       if k not in ('device', 'tree_method', 'eval_metric', 'verbosity')})
            results.append(rec)
            flag = ''
            if best is None or avg > best: best = avg; flag = '  <== BEST'
            log(f'{el()} [search {i:2d}] avg_raw={avg:7.1f} (24:{sc[2024]["raw"]:6.1f}/23:{sc[2023]["raw"]:6.1f}) '
                f'd{prm["max_depth"]} lr{prm["learning_rate"]} mcw{prm["min_child_weight"]:.0f} l2:{prm["reg_lambda"]:.0f}{flag}')
        except Exception:
            log(f'{el()} !! search {i}\n' + traceback.format_exc())
        pd.DataFrame(results).to_csv(OUT + 'results.csv', index=False)


# ════════════════════════════════════════════════════════════════
# Stage 3. TabM + FiLM (GPU)
# ════════════════════════════════════════════════════════════════
def stage3(X, results):
    import torch, torch.nn as nn
    log(f'\n{el()} ===== Stage 3: TabM + FiLM =====')
    dev = 'cuda:0'
    y = X.__y.values; season = X.__season.values
    CAT = ['pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id',
           'pitcher_hand', 'batter_hand', 'base_state', 'game_type', 'top_bottom']
    num = [c for c in X.columns if not c.startswith('__') and c not in CAT]
    Xc = X[CAT].values.astype(np.int64)
    Xn = X[num].values.astype(np.float32)
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    card = [int(Xc[:, i].max()) + 2 for i in range(len(CAT))]

    class PLR(nn.Module):
        """주기적 수치 임베딩 (Gorishniy et al.) — MLP 가 트리처럼 임계값을 학습하게 함"""
        def __init__(s, d_in, k=12, d=8):
            super().__init__(); s.c = nn.Parameter(torch.randn(d_in, k) * 0.05)
            s.lin = nn.Linear(2 * k, d); s.d_in, s.d = d_in, d
        def forward(s, x):
            z = 2 * np.pi * x.unsqueeze(-1) * s.c
            return torch.relu(s.lin(torch.cat([torch.sin(z), torch.cos(z)], -1))).flatten(1)

    class TabM(nn.Module):
        def __init__(s, n_num, card, k=32, h=256, emb=8, d_plr=8, film=True):
            super().__init__()
            s.k, s.film = k, film
            s.plr = PLR(n_num, d=d_plr)
            s.embs = nn.ModuleList([nn.Embedding(c, emb if i < 2 else 4) for i, c in enumerate(card)])
            d_in = n_num * d_plr + emb * 2 + 4 * (len(card) - 2)
            s.r1 = nn.Parameter(torch.randn(k, d_in) * 0.1 + 1)   # BatchEnsemble 입력 스케일
            s.l1 = nn.Linear(d_in, h); s.l2 = nn.Linear(h, h)
            s.heads = nn.Parameter(torch.zeros(k, h)); s.hb = nn.Parameter(torch.zeros(k))
            nn.init.normal_(s.heads, std=0.05)
            if film: s.film_net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 2 * h))
        def forward(s, xn, xc, t):
            e = [emb(xc[:, i]) for i, emb in enumerate(s.embs)]
            z = torch.cat([s.plr(xn)] + e, -1)                       # (B, d_in)
            z = z.unsqueeze(1) * s.r1                                # (B, k, d_in)
            z = torch.relu(s.l1(z))
            if s.film:
                g, b = s.film_net(t).chunk(2, -1)
                z = z * (1 + g.unsqueeze(1)) + b.unsqueeze(1)        # 시즌에 따라 층을 변조
            z = torch.relu(s.l2(z))
            out = (z * s.heads).sum(-1) + s.hb                       # (B, k)
            # 투수x타자 상호작용 (batter|pitcher)2vec 식
            inter = (e[0] * e[1]).sum(-1, keepdim=True)
            return out.mean(1) + inter.squeeze(-1) * 0.1

    def time_feat(sea, mon):
        s0 = (sea - 2019) / 6.0
        return np.stack([s0, s0 ** 2,
                         np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12),
                         np.sin(4 * np.pi * mon / 12), np.cos(4 * np.pi * mon / 12)], 1).astype(np.float32)
    Tf = time_feat(season.astype(np.float32), X.game_month.values.astype(np.float32))

    mu, sd = Xn.mean(0), Xn.std(0) + 1e-6
    Xn = (Xn - mu) / sd
    tn = torch.tensor(Xn, device=dev); tc = torch.tensor(Xc, device=dev)
    tt = torch.tensor(Tf, device=dev); ty = torch.tensor(y, device=dev)
    SP = splits(X)

    CFGS = [dict(k=32, h=256, film=True,  lr=2e-3, wd=1e-4, ep=12, tag='TabM+FiLM'),
            dict(k=32, h=256, film=False, lr=2e-3, wd=1e-4, ep=12, tag='TabM(FiLM없음)'),
            dict(k=1,  h=256, film=True,  lr=2e-3, wd=1e-4, ep=12, tag='단일MLP+FiLM'),
            dict(k=64, h=384, film=True,  lr=1e-3, wd=3e-4, ep=16, tag='TabM큰모델'),
            dict(k=32, h=256, film=True,  lr=2e-3, wd=1e-3, ep=12, tag='TabM+강한정규화'),
            dict(k=32, h=128, film=True,  lr=3e-3, wd=1e-4, ep=20, tag='TabM작은모델')]

    for cfg in CFGS:
        for vs, (tr, va) in SP.items():
            try:
                torch.manual_seed(0)
                net = TabM(Xn.shape[1], card, k=cfg['k'], h=cfg['h'], film=cfg['film']).to(dev)
                opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
                itr = np.where(tr)[0]; iva = np.where(va)[0]
                B = 16384
                for ep in range(cfg['ep']):
                    net.train(); perm = np.random.RandomState(ep).permutation(itr)
                    for j in range(0, len(perm), B):
                        b = torch.tensor(perm[j:j + B], device=dev)
                        p = torch.sigmoid(net(tn[b], tc[b], tt[b]))
                        loss = ((p - ty[b]) ** 2).mean()      # Brier 직접 최적화
                        opt.zero_grad(); loss.backward(); opt.step()
                net.eval(); ps = []
                with torch.no_grad():
                    for j in range(0, len(iva), 65536):
                        b = torch.tensor(iva[j:j + 65536], device=dev)
                        ps.append(torch.sigmoid(net(tn[b], tc[b], tt[b])).cpu().numpy())
                p = np.concatenate(ps)
                r = evaluate(p, y[va], trend_prior(y, season, vs))
                r.update(stage='nn', name=cfg['tag'], val=vs)
                results.append(r)
                np.save(OUT + f'pred_nn_{cfg["tag"]}_{vs}.npy', p)
                log(f'{el()} [nn] {cfg["tag"]:16s} val{vs} raw={r["raw"]:7.1f} trend={r["trend"]:7.1f} oracle={r["oracle"]:7.1f} meanp={r["meanp"]:.4f}')
            except Exception:
                log(f'{el()} !! nn {cfg["tag"]}/{vs}\n' + traceback.format_exc())
            pd.DataFrame(results).to_csv(OUT + 'results.csv', index=False)


# ════════════════════════════════════════════════════════════════
def main():
    results = []
    if os.path.exists(FEAT_CACHE):
        log(f'{el()} 캐시 로드 {FEAT_CACHE}'); X = pd.read_parquet(FEAT_CACHE)
    else:
        X = stage1()
    log(f'{el()} 피처 {X.shape[1]-3}개')
    try: stage2(X, results)
    except Exception: log('!! stage2\n' + traceback.format_exc())
    try: stage3(X, results)
    except Exception: log('!! stage3\n' + traceback.format_exc())
    pd.DataFrame(results).to_csv(OUT + 'results.csv', index=False)
    log(f'\n{el()} ===== 완료. results/results.csv, results/log.txt =====')

if __name__ == '__main__':
    main()
