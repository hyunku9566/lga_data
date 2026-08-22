"""
7차 — 단조 제약 (CPU, NN 스윕과 병렬)

논리: "그 투수의 과거 제구 성공률이 높으면 예측 확률도 높아야 한다"는 건
      데이터로 배울 게 아니라 물리적으로 확실한 사전지식이다.
      신호가 0.7% 뿐인 문제에서 이런 형태 제약은 가장 강력한 정규화다.
      트리는 제약이 없으면 노이즈를 따라 비단조 분할을 만든다.
"""
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb
import warnings, time, json; warnings.filterwarnings('ignore')
D = '/home/lee/lga/'
X = pd.read_parquet(D+'X98.parquet')
y = X.__y.values; season = X.__season.values; isF = X.__F.values.astype(bool)
C = [c for c in X.columns if not c.startswith('__')]

# 방향이 물리적으로 확실한 피처만 제약. 나머지는 0(자유).
UP_EXACT = {'asof_pitcher_success_rate','asof_pitcher_strike_rate','asof_batter_success_rate',
            'asof_pitcher_prev1_game_success_rate','asof_pitcher_prev3_game_success_rate',
            'asof_pitcher_prev5_game_success_rate','p_succ_ssn','p_succ_ssn_vs_car',
            'b_succ_ssn','b_succ_ssn_vs_car','p_stk_ssn','p_stk_ssn_vs_car',
            'p_sit_overall','p_sit_matched','pb_rate','p_prev5_vs_car','p_prev1_vs_prev5'}
DOWN_EXACT = {'asof_pitcher_reverse_rate','asof_pitcher_middle_rate','asof_pitcher_ball_rate',
              'asof_batter_middle_rate','asof_pitcher_prev1_game_middle_rate',
              'asof_pitcher_prev3_game_middle_rate','asof_pitcher_prev5_game_middle_rate',
              'p_rev_ssn','p_rev_ssn_vs_car','p_mid_ssn','p_mid_ssn_vs_car',
              'p_ball_ssn','p_ball_ssn_vs_car','b_mid_ssn','b_mid_ssn_vs_car'}
def mono(cols, mode):
    v = []
    for c in cols:
        if mode == 'off': v.append(0)
        elif c in UP_EXACT or (mode == 'wide' and c.startswith('p_sit_') and c.endswith('_d')): v.append(1)
        elif c in DOWN_EXACT: v.append(-1)
        else: v.append(0)
    return '(' + ','.join(map(str, v)) + ')'

def tp(vs):
    m = season < vs; s = pd.Series(y[m]).groupby(season[m]).mean()
    return float(sp.expit(np.polyval(np.polyfit(s.index, sp.logit(s.values), 1), vs)))
def ev(p, yv, rp):
    r = yv.mean(); ref = r*(1-r); b = lambda q: 100000*max(0., 1-np.mean((q-yv)**2)/ref)
    lo = sp.logit(np.clip(p, 1e-6, 1-1e-6))
    return b(p), b(sp.expit(lo-lo.mean()+sp.logit(rp))), b(sp.expit(lo-lo.mean()+sp.logit(r)))

PRM = dict(n_estimators=600, learning_rate=0.008, max_depth=6, min_child_weight=1500,
           subsample=0.7, colsample_bytree=0.5, reg_lambda=50., reg_alpha=1.,
           tree_method='hist', eval_metric='logloss', verbosity=0, n_jobs=8)
FOLDS = [2024, 2022]
NSEED = 5
print(f'제약 대상: 증가 {len(UP_EXACT & set(C))}개, 감소 {len(DOWN_EXACT & set(C))}개 / 전체 {len(C)}', flush=True)

R = []
for mode in ['off', 'exact', 'wide']:
    mc = mono(C, mode)
    row = {'mode': mode}
    for vs in FOLDS:
        tr = (season < vs) & ~(isF & (season <= 2022) & (vs >= 2023))
        va = (season == vs) & ~isF
        sc = []
        for sd in range(NSEED):
            p = xgb.XGBClassifier(**PRM, random_state=sd,
                                  monotone_constraints=(None if mode=='off' else mc)
                                  ).fit(X.loc[tr, C], y[tr]).predict_proba(X.loc[va, C])[:, 1]
            sc.append(ev(p, y[va], tp(vs)))
        a = np.array(sc)
        row[f'raw{vs}'] = a[:,0].mean(); row[f'sd{vs}'] = a[:,0].std()
        row[f'trend{vs}'] = a[:,1].mean()
        print(f'  [{mode:6s}] val{vs} raw={a[:,0].mean():7.1f}±{a[:,0].std():4.1f} '
              f'trend={a[:,1].mean():7.1f} oracle={a[:,2].mean():7.1f}', flush=True)
    row['avg'] = (row[f'raw{FOLDS[0]}'] + row[f'raw{FOLDS[1]}'])/2
    R.append(row); print(f'  >>> {mode}: avg={row["avg"]:7.1f}\n', flush=True)
    pd.DataFrame(R).to_csv(D+'results7_mono.csv', index=False)

b = pd.DataFrame(R).sort_values('avg', ascending=False)
print(b.to_string(index=False))
print(f'\n단조 제약 순효과: {b.iloc[0]["avg"] - [r for r in R if r["mode"]=="off"][0]["avg"]:+.1f}')
