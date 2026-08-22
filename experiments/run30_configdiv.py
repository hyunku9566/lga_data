"""
30차 — 시드 다양성 vs 설정 다양성

■ 무엇을 왜 재는가
  현재 제출본은 '같은 설정 + 시드만 12개' 를 평균한다.
  그런데 27차에서 개별 성능이 비슷한(폴드2024 866~870) 서로 다른 설정이 여럿 나왔다.

      n4000/lr0.004/d10        869.7
      lossguide max_leaves512  868.1
      n2000/lr0.005/d10        867.7
      d12/n2000/lr0.005        866.0

  같은 설정 시드 평균은 초기값 잡음만 지우지만, 서로 다른 설정은 트리 구조 자체가
  달라 오차가 덜 겹친다. 앙상블 이론상 후자가 낫지만 이 데이터에서 실제로 그런지는
  측정된 바 없다.

  이 질문이 지금 중요한 이유: v10 에서 '다양성을 줄이는' 방향(블렌드 집중)이
  CV +17 인데 LB -11.66 이었다. LB 는 CV 보다 다양성을 훨씬 더 원한다는 신호다.
  설정 다양성은 그 방향을 트리 축 안에서 늘리는 가장 값싼 수단이다.

■ 비교
  (a) 단일 최고설정(n4000) 시드 4개 평균     — 현행 방식의 대표
  (b) 네 설정 x 각 1시드 평균                — 같은 4개 모델, 설정만 다양화
  (c) 네 설정 x 각 2시드 평균                — 8개 모델
  참고로 각 설정 단독(1시드) 도 함께 기록한다.

■ 채택 기준
  (b) 또는 (c) 가 **폴드2024 와 폴드2023 양쪽에서** (a) 를 이길 때만 채택한다.
  한쪽만 이기면 기각. 예측은 확률 공간에서 평균한다(배포 추론 코드와 동일 규약).
"""
import os, sys, time, itertools
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, '/home/lee/lga')
import lib_lga as L

OUT = '/home/lee/lga/results30/'
log, _ = L.mklog(OUT)

b = L.load_base()
y = b['y']
X = L.build_v7(b=b)
log(f'피처 {X.shape[1]}')

BASEP = dict(max_depth=10, min_child_weight=6000, subsample=0.7, colsample_bytree=0.5,
             reg_lambda=50., reg_alpha=1., tree_method='hist', device=L.DEV,
             eval_metric='logloss', verbosity=0)
CFGS = {
    'A n4000/lr.004/d10': dict(BASEP, n_estimators=4000, learning_rate=0.004),
    'B n2000/lr.005/d10': dict(BASEP, n_estimators=2000, learning_rate=0.005),
    'C lossguide L512':   dict(BASEP, n_estimators=1500, learning_rate=0.008,
                               grow_policy='lossguide', max_leaves=512, max_depth=0),
    'D d12/n2000/lr.005': dict(BASEP, n_estimators=2000, learning_rate=0.005, max_depth=12),
}
SEEDS = [0, 1]          # (b)/(c) 용
SINGLE_SEEDS = [0, 1, 2, 3]   # (a) 용
SINGLE = 'A n4000/lr.004/d10'


def pred(cfg_name, seed, vs):
    """(설정, 시드, 폴드) 예측 확률. 캐시해서 재실행 시 재사용."""
    tag = cfg_name.split()[0]
    f = OUT + f'p_{tag}_s{seed}_{vs}.npy'
    if os.path.exists(f):
        return np.load(f)
    ctx = L.get_ctx(vs)
    t0 = time.time()
    m = xgb.XGBClassifier(**CFGS[cfg_name], random_state=seed)
    m.fit(X[ctx['tr']], y[ctx['tr']], sample_weight=ctx['w'])
    p = m.predict_proba(X[ctx['va']])[:, 1].astype(np.float32)
    np.save(f, p)
    log(f'  fit {cfg_name:22s} seed{seed} 폴드{vs}  {(time.time()-t0)/60:.1f}분')
    return p


def score(members, vs):
    """members = [(cfg, seed), ...] 확률 평균 후 BSS."""
    ctx = L.get_ctx(vs)
    p = np.mean([pred(c, s, vs) for c, s in members], 0)
    return L.bss(p, ctx['yv'], ctx['base'])


log('\n===== 개별 설정 단독 (시드0) =====')
solo = {}
for c in CFGS:
    solo[c] = {vs: score([(c, 0)], vs) for vs in L.FOLDS}
    log(f'  {c:22s} 24:{solo[c][2024]:7.1f}  23:{solo[c][2023]:7.1f}')

log('\n===== (a) 단일 최고설정 시드 4개 =====')
a_mem = [(SINGLE, s) for s in SINGLE_SEEDS]
A = {vs: score(a_mem, vs) for vs in L.FOLDS}
log(f'  (a) {SINGLE} x4시드      24:{A[2024]:7.1f}  23:{A[2023]:7.1f}   [기준]')

R = [dict(name=f'(a) {SINGLE} x4시드', n=4, m24=A[2024], m23=A[2023], d24=0., d23=0., both=False)]


def report(name, members):
    s = {vs: score(members, vs) for vs in L.FOLDS}
    d24, d23 = s[2024] - A[2024], s[2023] - A[2023]
    both = bool(d24 > 0 and d23 > 0)
    log(f'  {name:34s} n={len(members):<2} 24:{s[2024]:7.1f} ({d24:+6.1f}) '
        f'23:{s[2023]:7.1f} ({d23:+6.1f}) {"O 채택가능" if both else ""}')
    R.append(dict(name=name, n=len(members), m24=s[2024], m23=s[2023],
                  d24=d24, d23=d23, both=both))
    pd.DataFrame(R).to_csv(OUT + 'res30.csv', index=False)


log('\n===== (b)/(c) 설정 다양성 =====')
report('(b) 네 설정 x 1시드', [(c, 0) for c in CFGS])
report('(c) 네 설정 x 2시드', [(c, s) for c in CFGS for s in SEEDS])
# 보조: 설정 3개 조합(어느 설정이 실제로 기여하는지)
for drop in CFGS:
    mem = [(c, s) for c in CFGS if c != drop for s in SEEDS]
    report(f'(c) - {drop.split()[0]} 제외', mem)

log('\n===== 결론 =====')
ok = [r for r in R[1:] if r['both']]
if ok:
    best = max(ok, key=lambda r: min(r['d24'], r['d23']))
    log(f'채택 권고: {best["name"]}  24 {best["d24"]:+.1f} / 23 {best["d23"]:+.1f} (두 폴드 모두 개선)')
else:
    log('채택할 구성 없음 — 설정 다양성이 시드 다양성을 두 폴드 모두에서 이기지 못했다.')
log(pd.DataFrame(R)[['name', 'n', 'm24', 'm23', 'd24', 'd23', 'both']].to_string(index=False))
