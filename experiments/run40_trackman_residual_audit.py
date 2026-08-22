"""40차 — TrackMan 고정 프로필의 시간외 잔차 신호 감사 (CPU 전용).

목적은 TrackMan 피처를 다시 대량 추가하는 것이 아니다. 각 시즌 s 행에는
< s 공식 TrackMan 이력만으로 만든 투수 프로필을 붙인 뒤, 현 v7 XGB의
시간외 잔차를 그 프로필이 설명하는지 확인한다. 테스트 행 간 정보는 전혀
사용하지 않는다.
"""
import os, time, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

import lib_lga as L

OUT = '/home/lee/lga/results40/'
os.makedirs(OUT, exist_ok=True)
LOG = open(OUT + 'log.txt', 'w', buffering=1)
T0 = time.time()
def log(*a):
    s = ' '.join(map(str, a))
    print(f'[{(time.time()-T0)/60:5.1f}m] {s}', flush=True)
    LOG.write(f'[{(time.time()-T0)/60:5.1f}m] {s}\n')

# GPU를 전혀 쓰지 않는다. 사용자 허용 범위 안에서 CPU 30코어를 사용한다.
NJOB = 30
BASEP = dict(n_estimators=900, learning_rate=.01, max_depth=8, min_child_weight=5000,
             subsample=.7, colsample_bytree=.6, reg_lambda=50., reg_alpha=1.,
             tree_method='hist', device='cpu', n_jobs=NJOB, eval_metric='logloss', verbosity=0)
RESP = dict(n_estimators=500, learning_rate=.02, max_depth=3, min_child_weight=1200,
            subsample=.8, colsample_bytree=.8, reg_lambda=100., reg_alpha=5.,
            tree_method='hist', device='cpu', n_jobs=NJOB, objective='reg:squarederror', verbosity=0)

b = L.load_base(); raw=b['RAW']; y=b['y']; season=b['season']; isF=b['isF']
X = L.build_v7(b=b).astype(np.float32)
J = pd.read_parquet('/home/lee/lga/aligned.parquet')
log(f'v7={X.shape}, aligned TrackMan={J.shape}, CPU threads={NJOB}')

# 평균뿐 아니라 구종별 평균/산포/비중, 최근 시즌과 과거 평균의 차이를 만든다.
VALUE = ['rel_speed','spin_rate','induced_vert_break','horz_break','extension',
         'rel_height','rel_side','zone_speed']
TYPES = ['fastball','breaking','offspeed']
PID = raw.pitcher_id.values

def profile_asof():
    out = pd.DataFrame(index=raw.index)
    for s in range(2020, 2025):
        tgt = season == s
        hist = J[J.season < s]
        if not tgt.any() or len(hist) == 0:
            continue
        g = hist.groupby('pitcher_id')
        f = pd.DataFrame({'tm_n': g.size().astype('float32')})
        for c in VALUE:
            f['tm_' + c + '_mean'] = g[c].mean()
            f['tm_' + c + '_sd'] = g[c].std()
        for pt in TYPES:
            q = hist[hist.pitch_type_group == pt].groupby('pitcher_id')
            f['tm_' + pt + '_n'] = q.size()
            for c in ('rel_speed','spin_rate','induced_vert_break','horz_break','zone_speed'):
                f[f'tm_{pt}_{c}'] = q[c].mean()
        # 시즌간 변화: 직전 시즌 프로필 - 이전 전체 프로필. 추론 시에도 2024와 그 이전만 사용 가능.
        last = J[J.season == s-1].groupby('pitcher_id')
        for c in ('rel_speed','spin_rate','induced_vert_break','horz_break','zone_speed'):
            f['tm_last_' + c] = last[c].mean()
            f['tm_lastminus_' + c] = f['tm_last_' + c] - f['tm_' + c + '_mean']
        mapped = pd.DataFrame({'pitcher_id': PID[tgt]}).join(f, on='pitcher_id').drop(columns='pitcher_id')
        out.loc[tgt, mapped.columns] = mapped.values
    # log count makes coverage/confidence explicit; leave physical missingness as NaN for trees.
    out['tm_logn'] = np.log1p(out.tm_n)
    return out.astype(np.float32)

TM = profile_asof()
log(f'TM profile {TM.shape[1]} columns; coverage by season')
for s in (2023, 2024):
    m = (season == s) & ~isF
    log(f'  {s}: mapped={TM.tm_n[m].notna().mean():.3f}, median_n={TM.tm_n[m].median():.0f}')
TM.to_parquet(OUT+'tm_profiles_asof.parquet')

def weight(mask, vs):
    return (0.5 ** ((vs - 1 - season[mask]) / 2.)).astype(np.float32)

def base_fit_pred(tr, va, seed=0):
    m = xgb.XGBClassifier(**BASEP, random_state=seed)
    m.fit(X[tr], y[tr], sample_weight=weight(tr, int(season[va][0])))
    return m.predict_proba(X[va])[:,1]

rows=[]
for vs in (2023, 2024):
    tr, va = L.split(vs, b)
    # outer baseline: residual that a valid test-time correction would see
    pva = base_fit_pred(tr, va)
    # Train residual labels must be out-of-fold. Each source season q is predicted by <q only.
    poof = np.full(tr.sum(), np.nan, np.float32)
    ixtr = np.flatnonzero(tr)
    for q in range(2020, vs):
        inner_va = tr & (season == q)
        inner_tr = (season < q) & ~(isF & (season <= 2022) & (q >= 2023))
        if inner_va.sum() == 0 or inner_tr.sum() == 0: continue
        pq = base_fit_pred(inner_tr, inner_va)
        poof[np.isin(ixtr, np.flatnonzero(inner_va))] = pq
        log(f'fold{vs}: inner baseline {q} ({inner_tr.sum():,}->{inner_va.sum():,})')
    usable = np.isfinite(poof) & TM.iloc[ixtr].tm_n.notna().values
    # Restrict both fitting and scoring correction to mapped rows. Unmapped outputs stay exactly baseline.
    rtrain = y[ixtr][usable] - poof[usable]
    Ztr = TM.iloc[ixtr[usable]].copy(); Ztr['base_p'] = poof[usable]
    zva_all = TM.iloc[np.flatnonzero(va)].copy(); zva_all['base_p'] = pva
    mapped_va = zva_all.tm_n.notna().values
    rm = xgb.XGBRegressor(**RESP, random_state=0)
    rm.fit(Ztr, rtrain, sample_weight=weight(tr, vs)[usable])
    corr = np.zeros(va.sum(), np.float32)
    corr[mapped_va] = rm.predict(zva_all.loc[mapped_va])
    # prevent residual learner from creating implausible probability shifts
    pnew = np.clip(pva + corr, .001, .999)
    ctx=L.fold_ctx(vs, b=b); yv=ctx['yv']; base=ctx['base']
    old=L.bss(pva, yv, base); new=L.bss(pnew, yv, base)
    resid=yv-pva
    rho=spearmanr(corr[mapped_va], resid[mapped_va]).statistic
    pear=np.corrcoef(corr[mapped_va], resid[mapped_va])[0,1]
    log(f'fold{vs}: baseline={old:.2f}, correction={new:.2f}, delta={new-old:+.2f}; '
        f'resid corr pearson={pear:.4f}, spearman={rho:.4f}; mapped={mapped_va.mean():.3f}')
    nva=zva_all.tm_n.values
    for tag, gm in [('all', np.ones(len(nva),bool)), ('n>=200',nva>=200), ('n>=800',nva>=800), ('n<200',nva<200)]:
        gm &= np.isfinite(nva)
        if gm.sum() < 100: continue
        mse0=np.mean((pva[gm]-yv[gm])**2); mse1=np.mean((pnew[gm]-yv[gm])**2)
        log(f'  {tag:6s} n={gm.sum():,} correction MSE delta={(mse1-mse0):+.7f}')
    rows.append(dict(fold=vs, baseline=old, corrected=new, delta=new-old,
                     pearson=pear, spearman=rho, mapped=float(mapped_va.mean()), train_n=int(usable.sum())))

pd.DataFrame(rows).to_csv(OUT+'summary.csv', index=False)
both=all(r['delta']>0 for r in rows)
log(f'FINAL both-fold residual signal: {both}')
