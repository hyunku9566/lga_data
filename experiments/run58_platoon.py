"""
58차 — 투수 x 타자손(플래툰) 전용 계층 피처

근거 (57차 잔차 감사, 124피처 LGB 기준)
    투수x타자손 잔차 상한 +202.9 (순열 대조군 17.3)  <- 최대 잔여 축
    플래툰 스플릿(vsL-vsR) 안정성 <=2023 -> 2024  r=+0.426,  스플릿 sd 6.34%p
    (참고) 구장 r=+0.080 -> 불안정, 기각

원인 가설
    pbc_* 의 축소 경로가  [투수x손x카운트] -> [투수 전체율]  로 **투수x손 층을 건너뛴다**.
    p_sit_vsL/vsR 는 있으나 '현재 타자 손에 맞춘 값'이 아니라 별도 컬럼이고,
    p_sit_matched 는 sits 순회 순서상 손보다 카운트 상황이 먼저 잡힌다.
    => 손 정보가 축소 과정에서 씻긴다.

측정: v7(120) + pbc(4) = 124 기준선 대비, ph_* 4개 추가 (128) 를 두 폴드에서 잰다.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/lee/lga')
import lib_lga as L

OUT = '/home/lee/lga/results58/'
log, _ = L.mklog(OUT)

b = L.load_base(); R = b['RAW']; y = b['y']; season = b['season']
X0 = L.build_v7(b=b).astype(np.float32)
B3 = pd.read_parquet('/home/lee/lga/results45/B3.parquet')
BASE = pd.concat([X0, B3], axis=1)
log(f'기준선 피처 {BASE.shape[1]}')

pid = R.pitcher_id.values.astype(np.int64)
bh  = R.batter_hand.values.astype(np.int64)
key = pid * 10 + bh

# 시즌 인과: 시즌 s 행은 season<s 만 사용. 2단 축소 (셀 -> 투수 -> 리그)
K_CELL, K_PIT = 150.0, 200.0
F = {c: np.full(len(R), np.nan, np.float32) for c in ['ph_rate', 'ph_n', 'ph_logn', 'ph_delta']}
for s_ in range(2020, 2026):
    tgt = (season == s_); prev = (season < s_)
    if not tgt.any() or not prev.any(): continue
    mu = float(y[prev].mean())
    gk = pd.Series(y[prev]).groupby(key[prev]).agg(['sum', 'size'])   # 투수x손
    gp = pd.Series(y[prev]).groupby(pid[prev]).agg(['sum', 'size'])   # 투수 전체
    kt, pt = pd.Series(key[tgt]), pd.Series(pid[tgt])
    n  = kt.map(gk['size']).fillna(0).values.astype(np.float64)
    sy = kt.map(gk['sum']).fillna(0).values.astype(np.float64)
    pn = pt.map(gp['size']).fillna(0).values.astype(np.float64)
    ps = pt.map(gp['sum']).fillna(0).values.astype(np.float64)
    pit = (ps + K_PIT * mu) / (pn + K_PIT)          # 1단: 투수 전체율 (리그로 축소)
    rate = (sy + K_CELL * pit) / (n + K_CELL)       # 2단: 셀을 '투수x손 부모'가 아닌 투수로 축소
    F['ph_rate'][tgt] = rate
    F['ph_n'][tgt] = n
    F['ph_logn'][tgt] = np.log1p(n)
    F['ph_delta'][tgt] = rate - pit                 # 순수 플래툰 성분
PH = pd.DataFrame(F, index=R.index)
log('ph_* 생성 완료  결측률 ' + str((PH.isna().mean().round(4)).to_dict()))

r0 = L.bench2(BASE, name='기준선 124', log=log)
r1 = L.bench2(pd.concat([BASE, PH], axis=1), name='+ ph_* (플래툰)',
              baseline=(r0['m24'], r0['m23']), log=log)
r2 = L.bench2(pd.concat([BASE, PH[['ph_delta', 'ph_logn']]], axis=1), name='+ ph_delta/logn 만',
              baseline=(r0['m24'], r0['m23']), log=log)
pd.DataFrame([r0, r1, r2]).to_csv(OUT + 'res.csv', index=False)
log('\n채택 기준: both=True (두 폴드 동시 개선). 통과 시 블렌드 수준에서 재검증 필요.')
