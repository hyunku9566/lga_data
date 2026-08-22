"""
57차 — 폴드2022 는 regime 때문이 아니라 F리그 오염 때문에 이상했던 것인가

발견 (데이터 직접 관측)
  * 퓨처스(F) 제구율이 2022-10 .7076 -> 2023-04 .4936 으로 21.4%p 계단 단절.
    같은 경계에서 F 구장효과 상한이 220.6 -> 16.0 붕괴. 두 신호가 같은 곳을 가리킨다.
  * 반면 1군(R) 은 2023-10 .510 -> 2024-03 .504 로 단절이 없다. 완만한 하강뿐이다.
    즉 'R리그 2024 ABS regime 단절' 은 데이터에 없다.
  * lib_lga.split 의 (vs>=2023) 가드 탓에 폴드2022 만 F2019-2021(~72k, +18%p) 이
    학습셋에 남는다. 폴드2023/2024 는 깨끗하다.

측정: 폴드2022 를 현행(오염) vs 정제(F 제외) 로 재고 차이를 본다.
      정제 후 정상 거동하면 폴드2022 는 3번째 검증 폴드로 복구된다.
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/lee/lga')
import lib_lga as L

OUT = '/home/lee/lga/results57/'
log, _ = L.mklog(OUT)

b = L.load_base()
X = L.build_v7(b=b).astype(np.float32)
y, season, isF = b['y'], b['season'], b['isF']
log(f'피처 {X.shape[1]}  행 {X.shape[0]}')

def ctx_for(vs, drop_f, hl=2.0):
    tr = (season < vs)
    if drop_f:
        tr = tr & ~(isF & (season <= 2022))
    va = (season == vs) & ~isF
    w = (0.5 ** ((vs - 1 - season[tr]) / hl)).astype(np.float32)
    yv = y[va].astype(np.float64)
    return dict(vs=vs, tr=tr, va=va, w=w, yv=yv, base=yv.mean() * (1 - yv.mean()))

rows = []
for vs in (2022, 2023, 2024):
    for drop_f in (False, True):
        c = ctx_for(vs, drop_f)
        ntr = int(c['tr'].sum()); nf = int((isF & (season <= 2022) & c['tr']).sum())
        t0 = time.time()
        p = L.fit_predict(X, y, L.XP_TUNED, c, nseed=2)
        s = L.bss(p, c['yv'], c['base'])
        log(f'  폴드{vs}  {"정제(F제외)" if drop_f else "현행       "}  '
            f'학습{ntr:8d} (F오염 {nf:6d})  BSS {s:8.1f}   [{time.time()-t0:5.0f}s]')
        rows.append(dict(vs=vs, drop_f=drop_f, ntr=ntr, nF=nf, bss=s))
        np.save(OUT + f'p_{vs}_{"clean" if drop_f else "cur"}.npy', p.astype(np.float32))

r = pd.DataFrame(rows); r.to_csv(OUT + 'res.csv', index=False)
log('\n=== 정제 효과 ===')
for vs in (2022, 2023, 2024):
    a = r[(r.vs == vs) & ~r.drop_f].bss.iloc[0]; c = r[(r.vs == vs) & r.drop_f].bss.iloc[0]
    log(f'  폴드{vs}  현행 {a:8.1f} -> 정제 {c:8.1f}   Δ {c-a:+7.1f}')
log('\n폴드2023/2024 는 마스크가 이미 같으므로 Δ≈0 이어야 한다 (대조군).')
