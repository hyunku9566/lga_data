"""
60차 — [진단 전용, 제출 금지] test 내부 투수별 집계의 값어치를 정량화

동기
    리더보드에 제출 2~4회로 1140+ 인 팀이 다수다. 우리는 30회 제출에 1073 이다.
    test.csv 에는 정답이 없지만, **투수별로 test 행을 모으면** 그 투수의
    asof_pitcher_n 최대 행에서 2025 시즌 최종 성적을 읽을 수 있다.
    = 데이콘이 명시 금지한 "test.csv 내부 행들을 이용한 선수별 누적 통계" (실격 사유).

    이 스크립트는 그 연산의 값어치만 잰다. 결과는 제출에 절대 쓰지 않는다.
    목적은 '우리가 쫓는 격차가 합법 영역에 존재하는가'를 판정하는 것이다.

측정
    폴드2024 에서, 각 투수의 **2024 시즌 최종(사후) 성적**을 피처로 주면 얼마나 오르나.
    합법 파이프라인은 '그 투구 직전까지' 만 쓴다 (p_succ_ssn).
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/lee/lga')
import lib_lga as L

OUT = '/home/lee/lga/results60/'
log, _ = L.mklog(OUT)
log('*** 진단 전용. 이 피처는 제출 파이프라인에 넣지 않는다. ***')
b = L.load_base(); R = b['RAW']; y = b['y']; season = b['season']
X0 = L.build_v7(b=b).astype(np.float32)
B3 = pd.read_parquet('/home/lee/lga/results45/B3.parquet')
BASE = pd.concat([X0, B3], axis=1)

pid = R.pitcher_id.values
# 사후(hindsight): 그 투수의 '해당 시즌 전체' 성적 — test 집계로 얻어지는 것과 동형
df = pd.DataFrame(dict(pid=pid, season=season, y=y))
fs = df.groupby(['pid', 'season']).y.agg(['sum', 'size'])
mu = float(y.mean()); K = 50.0
fs['rate'] = (fs['sum'] + K * mu) / (fs['size'] + K)
key = pd.MultiIndex.from_arrays([pid, season])
H = pd.DataFrame({
    'hs_rate': fs['rate'].reindex(key).values.astype(np.float32),
    'hs_n':    fs['size'].reindex(key).values.astype(np.float32),
}, index=R.index)
H['hs_vs_asof'] = (H.hs_rate.values - np.nan_to_num(
    R.asof_pitcher_success_rate.values, nan=mu)).astype(np.float32)

r0 = L.bench2(BASE, name='합법 기준선 124', log=log)
r1 = L.bench2(pd.concat([BASE, H], axis=1), name='+ 사후 시즌최종성적 (금지)',
              baseline=(r0['m24'], r0['m23']), log=log)
pd.DataFrame([r0, r1]).to_csv(OUT + 'res.csv', index=False)
log('\n해석: 이 Δ 가 리더보드 격차와 같은 자릿수면, 상위 점수대는 합법 영역이 아니다.')
