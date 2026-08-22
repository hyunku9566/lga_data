"""
42차 — 투수 × 정확 볼카운트(12개) 4대 물리 성분 이력 확장 검증

배경:
  - 41차에서 제구 성공(y)의 정확 카운트 이력이 듀얼 폴드 동시 폭등(+5.1 / +3.6 both=True)으로 공식 검증 통과!
  - 로드맵 2단계: 4대 물리 성분(reverse, middle, ball, strike)의 정확 카운트 이력을 추가하여 추가 상승 검증.
"""
import os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import lib_lga as L

DEV = os.environ.get('LGA_DEV', 'cuda:1')
OUT = '/home/lee/lga/results42/'
log, OUT = L.mklog(OUT, 'log.txt')

b = L.load_base()
R = b['RAW']; y = b['y']; season = b['season']; isF = b['isF']
comp_labels, pitch_cls, valid_pt = L.recover_labels(R)
X_base = L.build_v7(b=b).astype(np.float32)

pid = R.pitcher_id.values.astype(np.int64)
bb = R.balls_before.values.astype(np.int16)
ss = R.strikes_before.values.astype(np.int16)
cnt = bb * 10 + ss
key = pid * 100 + cnt

# 1. 41차 통과한 exact count success 피처 (8개: k=30, 100, 300, 1000 + delta, n, logn)
def exact_y():
    F = {c: np.full(len(R), np.nan, np.float32) for c in ['pc_rate', 'pc_n', 'pc_logn', 'pc_delta']}
    for k in [30, 100, 300, 1000]:
        F[f'pc_k{k}'] = np.full(len(R), np.nan, np.float32)
    for s in range(2020, 2026):
        tgt = (season == s); prev = (season < s)
        if not tgt.any() or not prev.any(): continue
        dp = pd.DataFrame({'pid': pid[prev], 'cnt': cnt[prev], 'key': key[prev], 'y': y[prev]})
        gp = dp.groupby('pid').y.agg(['sum', 'size'])
        prior_pid = gp['sum'] / gp['size']
        gc_league = dp.groupby('cnt').y.agg(['sum', 'size'])
        prior_count = gc_league['sum'] / gc_league['size']
        gk = dp.groupby('key').y.agg(['sum', 'size'])
        
        mu = float(y[prev].mean())
        kt, pt, ct = pd.Series(key[tgt]), pd.Series(pid[tgt]), pd.Series(cnt[tgt])
        n = kt.map(gk['size']).fillna(0).values.astype(np.float32)
        sy = kt.map(gk['sum']).fillna(0).values.astype(np.float64)
        pp = pt.map(prior_pid).fillna(mu).values.astype(np.float64)
        pc = ct.map(prior_count).fillna(mu).values.astype(np.float64)
        
        rate = (sy + 50.0 * pp + 50.0 * pc) / (n + 100.0)
        overall = pt.map(prior_pid).fillna(mu).values.astype(np.float64)
        F['pc_rate'][tgt] = rate.astype(np.float32)
        F['pc_n'][tgt] = n
        F['pc_logn'][tgt] = np.log1p(n)
        F['pc_delta'][tgt] = (rate - overall).astype(np.float32)
        for k in [30, 100, 300, 1000]:
            F[f'pc_k{k}'][tgt] = ((sy + (k * 0.5) * pp + (k * 0.5) * pc) / (n + k)).astype(np.float32)
    return pd.DataFrame(F, index=R.index).astype(np.float32)

E_succ = exact_y()
X41 = pd.concat([X_base, E_succ], axis=1)

# 2. 4대 물리 성분 exact count 피처 생성 함수
def exact_comp(shr=100):
    F_comp = {}
    COMP_NAMES = ['reverse', 'middle', 'ball', 'strike']
    for c in COMP_NAMES:
        F_comp[f'pc_{c}_rate'] = np.full(len(R), np.nan, np.float32)
        F_comp[f'pc_{c}_delta'] = np.full(len(R), np.nan, np.float32)
        
    for s in range(2020, 2026):
        tgt = (season == s); prev = (season < s)
        if not tgt.any() or not prev.any(): continue
        
        for c in COMP_NAMES:
            yc = comp_labels[c].values
            valid_c = ~np.isnan(yc) & prev
            
            dp = pd.DataFrame({
                'pid': pid[valid_c],
                'cnt': cnt[valid_c],
                'key': key[valid_c],
                'yc': yc[valid_c]
            })
            gp = dp.groupby('pid').yc.agg(['sum', 'size'])
            prior_pid = gp['sum'] / gp['size']
            gc_league = dp.groupby('cnt').yc.agg(['sum', 'size'])
            prior_count = gc_league['sum'] / gc_league['size']
            gk = dp.groupby('key').yc.agg(['sum', 'size'])
            
            mu = float(np.nanmean(yc[prev]))
            kt, pt, ct = pd.Series(key[tgt]), pd.Series(pid[tgt]), pd.Series(cnt[tgt])
            n = kt.map(gk['size']).fillna(0).values.astype(np.float32)
            sy = kt.map(gk['sum']).fillna(0).values.astype(np.float64)
            pp = pt.map(prior_pid).fillna(mu).values.astype(np.float64)
            pc = ct.map(prior_count).fillna(mu).values.astype(np.float64)
            
            rate = (sy + (shr * 0.5) * pp + (shr * 0.5) * pc) / (n + shr)
            overall = pt.map(prior_pid).fillna(mu).values.astype(np.float64)
            
            F_comp[f'pc_{c}_rate'][tgt] = rate.astype(np.float32)
            F_comp[f'pc_{c}_delta'][tgt] = (rate - overall).astype(np.float32)
            
    return pd.DataFrame(F_comp, index=R.index).astype(np.float32)

log(f'41차 통과 기준선 피처 수: {X41.shape[1]}')
base = L.bench2(X41, name='41차 승인 기준 (128피처)', nseed=2, log=log)

results = []
for shr in (100, 300):
    E_c = exact_comp(shr)
    log(f'\n--- [성분 이력 shr={shr} (+8개)] ---')
    r = L.bench2(pd.concat([X41, E_c], axis=1),
                 name=f'성분이력 shr{shr}', nseed=2, baseline=(base['m24'], base['m23']), log=log)
    results.append(r)

df_res = pd.DataFrame([base] + results)
df_res.to_csv(OUT + 'summary.csv', index=False)
log('\n===== 최종 요약 =====')
log(df_res.to_string())
log('\n완료')
