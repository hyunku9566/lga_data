"""
32차 — LightGBM 설정 다양성(Configuration Diversity) 검증

30차에서 XGB 에 대해 설정 다양성이 단일 최적 대비 두 폴드 모두 크게 개선됨을 확인했다:
  단일 최고 x 시드4:  24: 869.0  23: 748.7
  네 설정 x 시드1:   24: 870.5 (+1.5)  23: 763.7 (+15.1)  both=True

동일한 원리를 LightGBM 에 적용한다:
  (a) 기준: 현행 단일 설정(num_leaves=15, n=1200) x 시드 4개
  (b) 설정 다양성: 네 가지 상이한 트리 구조(A, B, C, D) x 시드 1개
  (c) 설정 다양성: 네 가지 상이한 트리 구조(A, B, C, D) x 시드 2개

검증 프로토콜: 폴드2024(주) + 폴드2023(보조) 동시 판정 (both=True 일 때만 채택)
"""
import os, json, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb, scipy.special as sp
import lib_lga
warnings.filterwarnings('ignore')

D = '/home/lee/lga/'
OUT = D + 'results32/'
log, OUT = lib_lga.mklog(OUT, 'log.txt')

b = lib_lga.load_base()
XK = lib_lga.build_v7(b=b)
log(f'피처 {XK.shape[1]}개 준비 완료')

# 4가지 상호보완적 LightGBM 설정
CFGS = {
    'A': dict(n_estimators=1200, learning_rate=0.01, num_leaves=15, min_child_samples=6000,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=50., reg_alpha=1.,
              verbose=-1, n_jobs=12),
    'B': dict(n_estimators=2000, learning_rate=0.005, num_leaves=63, max_depth=8, min_child_samples=4000,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=50., reg_alpha=1.,
              verbose=-1, n_jobs=12),
    'C': dict(n_estimators=1500, learning_rate=0.008, num_leaves=31, min_child_samples=6000,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.4, reg_lambda=50., reg_alpha=1.,
              extra_trees=True, verbose=-1, n_jobs=12),
    'D': dict(n_estimators=1500, learning_rate=0.008, num_leaves=25, min_child_samples=6000,
              boosting_type='goss', top_rate=0.2, other_rate=0.1, colsample_bytree=0.5,
              reg_lambda=50., reg_alpha=1., verbose=-1, n_jobs=12),
}

log('\n===== 개별 설정 단독 학습 및 검증 (시드 0) =====')
preds = {}
for tag, prm in CFGS.items():
    preds[tag] = {}
    for vs in lib_lga.FOLDS:
        ctx = lib_lga.get_ctx(vs)
        fn = f'{OUT}p_{tag}_s0_{vs}.npy'
        if os.path.exists(fn):
            p = np.load(fn)
            log(f'  load {tag} seed0 폴드{vs}  [캐시사용]  BSS: {lib_lga.bss(p, ctx["yv"], ctx["base"]):.1f}')
        else:
            t0 = time.time()
            m = lgb.LGBMClassifier(**prm, random_state=0)
            m.fit(XK[ctx['tr']], b['y'][ctx['tr']], sample_weight=ctx['w'])
            p = m.predict_proba(XK[ctx['va']])[:, 1]
            np.save(fn, p.astype(np.float32))
            log(f'  fit {tag} seed0 폴드{vs}  {time.time()-t0:.1f}초  BSS: {lib_lga.bss(p, ctx["yv"], ctx["base"]):.1f}')
        preds[tag][(vs, 0)] = p

log('\n===== (a) 단일 최고설정(A) x 시드 4개 (기준선) =====')
for sd in range(1, 4):
    for vs in lib_lga.FOLDS:
        ctx = lib_lga.get_ctx(vs)
        fn = f'{OUT}p_A_s{sd}_{vs}.npy'
        if os.path.exists(fn):
            p = np.load(fn)
        else:
            m = lgb.LGBMClassifier(**CFGS['A'], random_state=sd)
            m.fit(XK[ctx['tr']], b['y'][ctx['tr']], sample_weight=ctx['w'])
            p = m.predict_proba(XK[ctx['va']])[:, 1]
            np.save(fn, p.astype(np.float32))
        preds['A'][(vs, sd)] = p

p_a = {}
for vs in lib_lga.FOLDS:
    ctx = lib_lga.get_ctx(vs)
    pa = np.mean([preds['A'][(vs, s)] for s in range(4)], 0)
    p_a[vs] = lib_lga.bss(pa, ctx['yv'], ctx['base'])
log(f'  (a) A x4시드      24: {p_a[2024]:6.1f}  23: {p_a[2023]:6.1f}   [기준]')

log('\n===== (b) 설정 다양성: 네 설정(A,B,C,D) x 시드 1개 (총 4개 모델) =====')
p_b = {}
for vs in lib_lga.FOLDS:
    ctx = lib_lga.get_ctx(vs)
    pb = np.mean([preds[tag][(vs, 0)] for tag in CFGS], 0)
    p_b[vs] = lib_lga.bss(pb, ctx['yv'], ctx['base'])

d24_b = p_b[2024] - p_a[2024]
d23_b = p_b[2023] - p_a[2023]
both_b = (d24_b > 0 and d23_b > 0)
mark_b = 'O 채택가능' if both_b else '  '
log(f'  (b) 네 설정 x 1시드  24: {p_b[2024]:6.1f} ({d24_b:+5.1f}) 23: {p_b[2023]:6.1f} ({d23_b:+5.1f}) {mark_b}')

log('\n===== (c) 설정 다양성: 네 설정(A,B,C,D) x 시드 2개 (총 8개 모델) =====')
for tag in ['B', 'C', 'D']:
    for vs in lib_lga.FOLDS:
        ctx = lib_lga.get_ctx(vs)
        fn = f'{OUT}p_{tag}_s1_{vs}.npy'
        if os.path.exists(fn):
            p = np.load(fn)
        else:
            m = lgb.LGBMClassifier(**CFGS[tag], random_state=1)
            m.fit(XK[ctx['tr']], b['y'][ctx['tr']], sample_weight=ctx['w'])
            p = m.predict_proba(XK[ctx['va']])[:, 1]
            np.save(fn, p.astype(np.float32))
        preds[tag][(vs, 1)] = p

p_c = {}
for vs in lib_lga.FOLDS:
    ctx = lib_lga.get_ctx(vs)
    pc = np.mean([preds[tag][(vs, s)] for tag in CFGS for s in range(2)], 0)
    p_c[vs] = lib_lga.bss(pc, ctx['yv'], ctx['base'])

d24_c = p_c[2024] - p_a[2024]
d23_c = p_c[2023] - p_a[2023]
both_c = (d24_c > 0 and d23_c > 0)
mark_c = 'O 채택가능' if both_c else '  '
log(f'  (c) 네 설정 x 2시드  24: {p_c[2024]:6.1f} ({d24_c:+5.1f}) 23: {p_c[2023]:6.1f} ({d23_c:+5.1f}) {mark_c}')

# 서브셋 탐색
rows = [
    dict(name='(a) A x4시드', n=4, m24=p_a[2024], m23=p_a[2023], d24=0., d23=0., both=False),
    dict(name='(b) 네 설정 x 1시드', n=4, m24=p_b[2024], m23=p_b[2023], d24=d24_b, d23=d23_b, both=both_b),
    dict(name='(c) 네 설정 x 2시드', n=8, m24=p_c[2024], m23=p_c[2023], d24=d24_c, d23=d23_c, both=both_c),
]
for drop_tag in CFGS:
    rem = [t for t in CFGS if t != drop_tag]
    for vs in lib_lga.FOLDS:
        ctx = lib_lga.get_ctx(vs)
        p_sub = np.mean([preds[t][(vs, s)] for t in rem for s in range(2)], 0)
        p_c[vs] = lib_lga.bss(p_sub, ctx['yv'], ctx['base'])
    d24_s = p_c[2024] - p_a[2024]
    d23_s = p_c[2023] - p_a[2023]
    b_s = (d24_s > 0 and d23_s > 0)
    mark_s = 'O 채택가능' if b_s else '  '
    log(f'  (c) - {drop_tag} 제외     24: {p_c[2024]:6.1f} ({d24_s:+5.1f}) 23: {p_c[2023]:6.1f} ({d23_s:+5.1f}) {mark_s}')
    rows.append(dict(name=f'(c) - {drop_tag} 제외', n=len(rem)*2, m24=p_c[2024], m23=p_c[2023], d24=d24_s, d23=d23_s, both=b_s))

df_res = pd.DataFrame(rows)
df_res.to_csv(OUT + 'res32.csv', index=False)
log('\n===== 최종 요약 =====')
log(df_res.to_string(index=False))
log('\n완료')
