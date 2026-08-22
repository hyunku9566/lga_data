"""
33차 — 볼카운트 및 위기상황 상호작용 피처(Count & Stress Interaction Features) 검증

배경:
  - 투수의 제구 성공률은 볼카운트(3볼/2스트라이크/유불리)와 레버리지(li/득점권/후반 접전)에 따라 비선형적으로 변화한다.
  - 100% 행 독립적인(row-independent) 파생 피처 후보들을 생성하고,
  - lib_lga.bench2 (폴드2024 + 폴드2023 듀얼 검증)로 엄격히 채택 여부를 판정한다.
"""
import os, json, time, warnings
import numpy as np, pandas as pd, scipy.special as sp
import lib_lga
warnings.filterwarnings('ignore')

D = '/home/lee/lga/'
OUT = D + 'results33/'
log, OUT = lib_lga.mklog(OUT, 'log.txt')

b = lib_lga.load_base()
RAW = b['RAW']
XK = lib_lga.build_v7(b=b)
log(f'기존 v7 피처 수: {XK.shape[1]}')

# 1. 볼카운트 역학 파생 피처
F_count = pd.DataFrame(index=RAW.index)
bb = RAW.balls_before.values
sb = RAW.strikes_before.values
li = RAW.li.values
inn = RAW.inning.values
sdiff = RAW.score_diff_pitcher_team.values
risp = (RAW.runner_on_2b.values | RAW.runner_on_3b.values).astype(float)

F_count['c_diff'] = (sb - bb).astype(np.float32)
F_count['c_ahead'] = (sb > bb).astype(np.float32)
F_count['c_behind'] = (bb > sb).astype(np.float32)
F_count['c_3ball'] = (bb == 3).astype(np.float32)
F_count['c_2strk'] = (sb == 2).astype(np.float32)
F_count['c_full'] = ((bb == 3) & (sb == 2)).astype(np.float32)
F_count['c_pitches_in_pa'] = (bb + sb + 1).astype(np.float32)

# 2. 스트레스 & 상황 상호작용
F_stress = pd.DataFrame(index=RAW.index)
log_li = np.log1p(np.maximum(li, 0)).astype(np.float32)
F_stress['s_risp_li'] = (risp * log_li).astype(np.float32)
F_stress['s_3ball_li'] = (F_count['c_3ball'].values * log_li).astype(np.float32)
F_stress['s_late_close'] = ((inn >= 7) & (np.abs(sdiff) <= 2)).astype(np.float32)

# 3. 투수 구종/제구 특성과 카운트의 상호작용
F_inter = pd.DataFrame(index=RAW.index)
fb_rate = RAW.asof_pitcher_fastball_rate.values.astype(np.float32)
brk_rate = RAW.asof_pitcher_breaking_rate.values.astype(np.float32)
succ_rate = RAW.asof_pitcher_success_rate.values.astype(np.float32)
F_inter['i_fb_3ball'] = np.nan_to_num(fb_rate * F_count['c_3ball'].values, 0).astype(np.float32)
F_inter['i_brk_2strk'] = np.nan_to_num(brk_rate * F_count['c_2strk'].values, 0).astype(np.float32)
F_inter['i_succ_cdiff'] = np.nan_to_num(succ_rate * F_count['c_diff'].values, 0).astype(np.float32)

# 기준선 측정
log('\n===== 1. 기준선 (120 피처) =====')
res_base = lib_lga.bench2(XK, name='기준선(v7 120피처)', nseed=2, log=log)

# 그룹별 검증
candidates = [
    ('그룹1: 볼카운트 역학 (+7개)', [F_count]),
    ('그룹2: 스트레스/상황 상호작용 (+3개)', [F_stress]),
    ('그룹3: 구종-카운트 상호작용 (+3개)', [F_inter]),
    ('전체 결합 (+13개)', [F_count, F_stress, F_inter]),
]

res_list = [res_base]
for name, dfs in candidates:
    X_new = pd.concat([XK] + dfs, axis=1)
    r = lib_lga.bench2(X_new, name=name, nseed=2, baseline=(res_base['m24'], res_base['m23']), log=log)
    res_list.append(r)

df_all = pd.DataFrame(res_list)
df_all.to_csv(OUT + 'res33.csv', index=False)
log('\n===== 최종 요약 =====')
log(df_all.to_string(index=False))
log('\n완료')
