"""
34차 — 구종별(Fastball / Breaking / Offspeed) 제구 분해 및 엔트로피 피처 검증

배경:
  - 투수는 구종(패스트볼/슬라이더·커브/체인지업·포크볼)마다 제구 성공 확률과 스트라이크 존 공략 전략이 완전히 다르다.
  - 투수별 구종 배분(Pitch Mix) 엔트로피와 구종별 지배력, 구종별 기대 제구력을 100% 행 독립적으로 산출하여
  - lib_lga.bench2 (2024+2023 듀얼 검증)로 검증한다.
"""
import os, json, time, warnings
import numpy as np, pandas as pd, scipy.special as sp
import lib_lga
warnings.filterwarnings('ignore')

D = '/home/lee/lga/'
OUT = D + 'results34/'
log, OUT = lib_lga.mklog(OUT, 'log.txt')

b = lib_lga.load_base()
RAW = b['RAW']
XK = lib_lga.build_v7(b=b)
log(f'기존 v7 피처 수: {XK.shape[1]}')

# 1. 구종 배분(Pitch Mix) 물리 피처
F_pitch = pd.DataFrame(index=RAW.index)
fb = np.nan_to_num(RAW.asof_pitcher_fastball_rate.values.astype(np.float32), 0.0)
brk = np.nan_to_num(RAW.asof_pitcher_breaking_rate.values.astype(np.float32), 0.0)
off = np.nan_to_num(RAW.asof_pitcher_offspeed_rate.values.astype(np.float32), 0.0)

# 합계 정규화
tot = np.maximum(fb + brk + off, 1e-6)
p_fb, p_brk, p_off = fb / tot, brk / tot, off / tot

# 1) 구종 다양성 (Shannon Entropy)
eps = 1e-6
ent = -(p_fb * np.log(p_fb + eps) + p_brk * np.log(p_brk + eps) + p_off * np.log(p_off + eps))
F_pitch['pm_entropy'] = ent.astype(np.float32)

# 2) 직구 지배력 vs 변화구 지배력
F_pitch['pm_fb_dominance'] = (p_fb - np.maximum(p_brk, p_off)).astype(np.float32)
F_pitch['pm_brk_dominance'] = (p_brk - np.maximum(p_fb, p_off)).astype(np.float32)
F_pitch['pm_off_dominance'] = (p_off - np.maximum(p_fb, p_brk)).astype(np.float32)

# 3) 구종 다양성에 따른 제구 상호작용
succ_rate = np.nan_to_num(RAW.asof_pitcher_success_rate.values.astype(np.float32), 0.5)
F_pitch['pm_succ_x_entropy'] = (succ_rate * ent).astype(np.float32)
F_pitch['pm_succ_x_fb_dom'] = (succ_rate * F_pitch['pm_fb_dominance'].values).astype(np.float32)

# 기준선 측정
log('\n===== 1. 기준선 (120 피처) =====')
res_base = lib_lga.bench2(XK, name='기준선(v7 120피처)', nseed=2, log=log)

# 구종 피처 결합 검증
X_new = pd.concat([XK, F_pitch], axis=1)
res_pitch = lib_lga.bench2(X_new, name='구종배분 & 엔트로피 (+6개)', nseed=2,
                           baseline=(res_base['m24'], res_base['m23']), log=log)

df_all = pd.DataFrame([res_base, res_pitch])
df_all.to_csv(OUT + 'res34.csv', index=False)
log('\n===== 최종 요약 =====')
log(df_all.to_string(index=False))
log('\n완료')
