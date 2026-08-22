"""
39차 — 투수x상황 축소강도(SHR) 스윕 + 표본수 피처

배경
  잔차 분해에서 접근 가능한 남은 구조: 투수 ID 120.0, 투수x볼카운트 178.8.
  p_sit_* 26개가 이 축을 담당하는데 mkfeat.py 에서 13개 상황 전부
  shrink=300 으로 균일 축소된다. 상황별 표본 크기가 10배 이상 차이나므로
  표본이 작은 볼카운트 상황(3ball/2strk)이 과하게 뭉개졌을 가능성이 크다.

  또 상황별 표본수 n 이 피처에 없다. 트리가 편차의 신뢰도를 알 수 없다.
  (매치업에는 pb_n 이 있는데 상황에는 없다)

측정 (전부 v7 120 피처의 p_sit_* 만 교체/추가, 나머지 동일)
  shr30 / shr100 / shr300(현행) / shr1000     축소강도
  +n                                          상황별 log1p(n) 12개 추가
  EB                                          상황별 경험베이즈 축소강도

채택 기준: bench2 (폴드2024 · 폴드2023 동시 개선)
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, xgboost as xgb
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/lee/lga')
import lib_lga as L

OUT = '/home/lee/lga/results39/'; os.makedirs(OUT, exist_ok=True)
LOG = open(OUT+'log.txt', 'a', buffering=1)
def log(*a):
    m = ' '.join(str(x) for x in a); print(m, flush=True); LOG.write(m+'\n')
T0 = time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'
DEV = os.environ.get('LGA_DEV', 'cuda:1')

b = L.load_base(); RAW = b['RAW']; y = b['y']; season = b['season']
X0 = L.build_v7(b=b)
SIT_COLS = [c for c in X0.columns if c.startswith('p_sit_')]
log(f'{el()} v7 {X0.shape[1]} 피처, 그중 p_sit_* {len(SIT_COLS)}개')

# ── mkfeat.py 의 상황 정의를 그대로 재현 ──
bb = RAW.balls_before.values; st = RAW.strikes_before.values
SITS = {'3ball': bb==3, '2strk': st==2, 'ahead': st>bb, 'behind': bb>st,
        'risp': (RAW.runner_on_2b.values | RAW.runner_on_3b.values)==1,
        'on1b': RAW.runner_on_1b.values==1, 'vsL': RAW.batter_hand.values==1,
        'vsR': RAW.batter_hand.values==2, 'late': RAW.inning.values>=7,
        'hiLI': RAW.li.values>1.5, 'loLI': RAW.li.values<0.5,
        'blowout': np.abs(RAW.score_diff_pitcher_team.values)>=5}
PID = RAW.pitcher_id.values

def asof(mask, shrink, want_n=False):
    """mkfeat.asof_rate 와 동일. want_n 이면 as-of 표본수도 같이 준다."""
    out = np.full(len(RAW), np.nan, np.float32)
    cnt = np.zeros(len(RAW), np.float32)
    for s in range(2020, 2026):
        prev = season < s; tgt = season == s
        if not tgt.any() or not prev.any(): continue
        sel = prev & mask
        g = pd.Series(y[sel]).groupby(PID[sel])
        tot = pd.Series(y[prev]).groupby(PID[prev]).mean()
        base = tot.reindex(g.size().index).fillna(y[prev].mean())
        k = pd.Series(PID[tgt])
        out[tgt] = k.map((g.sum()+shrink*base)/(g.size()+shrink)).values
        if want_n:
            cnt[tgt] = k.map(g.size()).fillna(0).values
    return (out, cnt) if want_n else out

def make_sit(shrink, add_n=False, per_sit=None):
    """p_sit_* 26개를 주어진 축소강도로 재생성. per_sit 은 상황별 강도 dict."""
    sh_ov = per_sit.get('overall', shrink) if per_sit else shrink
    ov = asof(np.ones(len(RAW), bool), sh_ov)
    F = {'p_sit_overall': ov}
    mt = np.full(len(RAW), np.nan, np.float32)
    for n, m in SITS.items():
        sh = per_sit.get(n, shrink) if per_sit else shrink
        if add_n:
            v, c = asof(m, sh, want_n=True); F['p_sit_'+n+'_ln'] = np.log1p(c)
        else:
            v = asof(m, sh)
        F['p_sit_'+n] = v; F['p_sit_'+n+'_d'] = v - ov
        mt = np.where(m & np.isnan(mt), v - ov, mt)
    F['p_sit_matched'] = mt
    return pd.DataFrame(F, index=RAW.index).astype(np.float32)

def swap(newF):
    """v7 에서 p_sit_* 를 교체하고 새 컬럼은 덧붙인다. 열 순서 고정."""
    keep = X0.drop(columns=SIT_COLS)
    return pd.concat([keep, newF], axis=1)

def score(X, name):
    cf = OUT+f'y_{name}.json'
    if os.path.exists(cf):
        r = {int(k): v for k, v in json.load(open(cf)).items()}
        log(f'{el()}   [{name}] 캐시 재사용'); return r
    o = {}
    for vs in (2024, 2023):
        f = L.fold_ctx(vs, b=b); tr, va, w, yv, bq = f['tr'], f['va'], f['w'], f['yv'], f['base']
        p = np.mean([xgb.XGBClassifier(
                n_estimators=2000, learning_rate=0.005, max_depth=10,
                min_child_weight=6000, subsample=0.7, colsample_bytree=0.5,
                reg_lambda=50., reg_alpha=1., tree_method='hist', device=DEV,
                eval_metric='logloss', verbosity=0, random_state=s)
              .fit(X[tr], y[tr], sample_weight=w).predict_proba(X[va])[:, 1]
              for s in range(2)], 0)
        o[vs] = L.bss(p, yv, bq)
        del f, tr, va, w, yv, p
    json.dump({str(k): v for k, v in o.items()}, open(cf, 'w'))
    return o

# ── 경험베이즈 축소강도: 상황별 within/between 분산비 ──
def eb_shrink():
    """shrink* = sigma2_within / sigma2_between (2020~2023 학습분만으로 추정)."""
    prev = season <= 2023
    out = {}
    for n, m in list(SITS.items()) + [('overall', np.ones(len(RAW), bool))]:
        sel = prev & m
        g = pd.Series(y[sel]).groupby(PID[sel])
        sz = g.size(); mu = g.mean()
        ok = sz >= 20
        if ok.sum() < 30: out[n] = 300.; continue
        sz, mu = sz[ok], mu[ok]
        gm = float(np.average(mu, weights=sz))
        s2w = gm*(1-gm)                                   # 이항 내분산
        v_tot = float(np.average((mu-gm)**2, weights=sz))  # 관측 분산
        v_noise = float(np.average(s2w/sz, weights=sz))    # 표본잡음 기여
        s2b = max(v_tot - v_noise, 1e-6)                   # 진짜 투수간 분산
        out[n] = float(np.clip(s2w/s2b, 5, 5000))
    return out

EB = eb_shrink()
log(f'{el()} 경험베이즈 축소강도')
for k in sorted(EB, key=EB.get):
    log(f'          {k:10s} {EB[k]:8.0f}')

ARMS = [('shr300_현행', lambda: X0),
        ('shr30',   lambda: swap(make_sit(30.))),
        ('shr100',  lambda: swap(make_sit(100.))),
        ('shr1000', lambda: swap(make_sit(1000.))),
        ('shr300_n', lambda: swap(make_sit(300., add_n=True))),
        ('EB',      lambda: swap(make_sit(300., per_sit=EB))),
        ('EB_n',    lambda: swap(make_sit(300., add_n=True, per_sit=EB)))]

log(f'\n{el()} ===== 스윕 =====')
res = {}
for name, mk in ARMS:
    X = mk()
    r = score(X, name); res[name] = r
    if name == 'shr300_현행':
        base = r; log(f'{el()}   {name:12s} ({X.shape[1]:3d}피처)  24: {r[2024]:7.1f}  23: {r[2023]:7.1f}   기준')
    else:
        d24 = r[2024]-base[2024]; d23 = r[2023]-base[2023]
        tag = 'O 채택가능' if (d24 > 0 and d23 > 0) else ''
        log(f'{el()}   {name:12s} ({X.shape[1]:3d}피처)  24: {r[2024]:7.1f} ({d24:+5.1f})  23: {r[2023]:7.1f} ({d23:+5.1f})  {tag}')
    del X

json.dump(res, open(OUT+'summary.json', 'w'), indent=1)
log(f'{el()} 완료')
