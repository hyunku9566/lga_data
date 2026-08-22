"""
59차 — pbc_* 의 축소 경로를 고친다 (병렬 추가가 아니라 대체)

58차에서 배운 것
    ph_* 를 '추가' 하면 24 -5.7 / 23 +5.8 로 부호가 갈린다.
    p_sit_vsL/vsR 가 이미 있어 상관 컬럼을 하나 더 붙이는 꼴이었다.
    문제는 피처 부재가 아니라 **축소 경로**다.

현행 pbc_* (build_v17.py)
    셀 = pid x 타자손 x 정확카운트   ->  부모 = (투수 전체율, 리그 카운트율) 반반, k=100
    즉 [셀] -> [투수]  로 **투수x손 층을 건너뛴다.** 손 정보가 축소에서 씻긴다.

수정판 pbc2_*
    [셀] -> [투수x손] -> [투수] -> [리그]  3단 계층으로 순차 축소.
    피처 이름/개수는 동일(4개)하게 두고 값만 바꿔 '한 번에 하나만' 원칙을 지킨다.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/lee/lga')
import lib_lga as L

OUT = '/home/lee/lga/results59/'
log, _ = L.mklog(OUT)
b = L.load_base(); R = b['RAW']; y = b['y']; season = b['season']
X0 = L.build_v7(b=b).astype(np.float32)
B3 = pd.read_parquet('/home/lee/lga/results45/B3.parquet')
log(f'현행 pbc 컬럼: {B3.columns.tolist()}')

pid = R.pitcher_id.values.astype(np.int64)
bh  = R.batter_hand.values.astype(np.int64)
cnt = R.balls_before.values.astype(np.int64) * 10 + R.strikes_before.values.astype(np.int64)
cell = pid * 1000 + bh * 100 + cnt
ph   = pid * 10 + bh

def build(k_cell, k_ph, k_pit):
    F = {c: np.full(len(R), np.nan, np.float32) for c in ['pbc_rate','pbc_n','pbc_logn','pbc_delta']}
    for s_ in range(2020, 2026):
        tgt = (season == s_); prev = (season < s_)
        if not tgt.any() or not prev.any(): continue
        yp = pd.Series(y[prev]); mu = float(yp.mean())
        gC = yp.groupby(cell[prev]).agg(['sum','size'])
        gH = yp.groupby(ph[prev]).agg(['sum','size'])
        gP = yp.groupby(pid[prev]).agg(['sum','size'])
        gL = yp.groupby(cnt[prev]).mean()
        ct, ht, pt, nt = pd.Series(cell[tgt]), pd.Series(ph[tgt]), pd.Series(pid[tgt]), pd.Series(cnt[tgt])
        g = lambda s_map, key, col: key.map(s_map[col]).fillna(0).values.astype(np.float64)
        pn, ps = g(gP,pt,'size'), g(gP,pt,'sum')
        hn, hs = g(gH,ht,'size'), g(gH,ht,'sum')
        cn, cs = g(gC,ct,'size'), g(gC,ct,'sum')
        lg = nt.map(gL).fillna(mu).values.astype(np.float64)
        r_pit = (ps + k_pit * mu) / (pn + k_pit)                 # 투수 -> 리그
        r_ph  = (hs + k_ph  * r_pit) / (hn + k_ph)               # 투수x손 -> 투수
        parent = 0.5 * r_ph + 0.5 * lg                           # 부모 = 손보정 투수율 + 리그카운트율
        r_cell = (cs + k_cell * parent) / (cn + k_cell)          # 셀 -> 부모
        F['pbc_rate'][tgt]=r_cell; F['pbc_n'][tgt]=cn
        F['pbc_logn'][tgt]=np.log1p(cn); F['pbc_delta'][tgt]=r_cell-r_pit
    return pd.DataFrame(F, index=R.index)

r0 = L.bench2(pd.concat([X0, B3], axis=1), name='현행 pbc (기준선 124)', log=log)
res=[r0]
for kc, kh, kp in [(100., 150., 200.), (100., 400., 200.), (200., 150., 200.)]:
    P = build(kc, kh, kp)
    r = L.bench2(pd.concat([X0, P], axis=1), name=f'pbc2 3단계층 k={kc:.0f}/{kh:.0f}/{kp:.0f}',
                 baseline=(r0['m24'], r0['m23']), log=log)
    res.append(r)
pd.DataFrame(res).to_csv(OUT+'res.csv', index=False)
log('\nboth=True 인 팔만 블렌드 수준 재검증 대상이다.')
