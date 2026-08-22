"""
41차 — 투수 × 정확 볼카운트(12개) 계층적 과거 이력 검증

핵심 설계:
  1. 12개 카운트(0-0, 0-1, ..., 3-2)별 투수 과거 제구 성향 복원 (key = pid*100 + cnt)
  2. 계층적 경험적 베이즈 축소 (Hierarchical Empirical Bayes Shrinkage):
     - 투수 전체 제구율(prior_pid) + 리그 해당 카운트 제구율(prior_count)로 축소
     - rate = (sy + shr*0.5*pp + shr*0.5*pc) / (n + shr)
  3. 편차 및 표본수 피처:
     - pc_delta = rate - overall_pid (투수 평균 대비 이 카운트에서의 제구 편차)
     - pc_n, pc_logn = log1p(n)
  4. 엄격한 과거 시점(<s) 한정 조회로 미래 및 테스트셋 누수 0% (행 독립성 100% 준수)
  5. 듀얼 폴드(2024 주 / 2023 보조) 실측 검증
"""
import os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import lib_lga as L

DEV = os.environ.get('LGA_DEV', 'cuda:0')
OUT = '/home/lee/lga/results41/'
log, OUT = L.mklog(OUT, 'log.txt')

b = L.load_base()
R = b['RAW']; y = b['y']; season = b['season']; isF = b['isF']
X = L.build_v7(b=b).astype(np.float32)

pid = R.pitcher_id.values.astype(np.int64)
bb = R.balls_before.values.astype(np.int16)
ss = R.strikes_before.values.astype(np.int16)
cnt = bb * 10 + ss                # 0~32 (12개 카운트)
key = pid * 100 + cnt            # 투수 x 카운트 고유 키

def exact(shr, include_all=False):
    F = {c: np.full(len(R), np.nan, np.float32) for c in ['pc_rate', 'pc_n', 'pc_logn', 'pc_delta']}
    if include_all:
        for k in [30, 100, 300, 1000]:
            F[f'pc_k{k}'] = np.full(len(R), np.nan, np.float32)
            
    for s in range(2020, 2026):
        tgt = (season == s)
        prev = (season < s)
        if not tgt.any() or not prev.any():
            continue
            
        dp = pd.DataFrame({
            'pid': pid[prev],
            'cnt': cnt[prev],
            'key': key[prev],
            'y': y[prev]
        })
        
        # 1. 투수 전체 사전확률 (prior_pid)
        gp = dp.groupby('pid').y.agg(['sum', 'size'])
        prior_pid = gp['sum'] / gp['size']
        
        # 2. 리그 해당 카운트 전체 사전확률 (prior_count)
        gc_league = dp.groupby('cnt').y.agg(['sum', 'size'])
        prior_count = gc_league['sum'] / gc_league['size']
        
        # 3. 투수 x 카운트 통계 (g_key)
        gk = dp.groupby('key').y.agg(['sum', 'size'])
        
        mu = float(y[prev].mean())
        kt = pd.Series(key[tgt])
        pt = pd.Series(pid[tgt])
        ct = pd.Series(cnt[tgt])
        
        n = kt.map(gk['size']).fillna(0).values.astype(np.float32)
        sy = kt.map(gk['sum']).fillna(0).values.astype(np.float64)
        pp = pt.map(prior_pid).fillna(mu).values.astype(np.float64)
        pc = ct.map(prior_count).fillna(mu).values.astype(np.float64)
        
        rate = (sy + (shr * 0.5) * pp + (shr * 0.5) * pc) / (n + shr)
        overall = pt.map(prior_pid).fillna(mu).values.astype(np.float64)
        
        F['pc_rate'][tgt] = rate.astype(np.float32)
        F['pc_n'][tgt] = n
        F['pc_logn'][tgt] = np.log1p(n)
        F['pc_delta'][tgt] = (rate - overall).astype(np.float32)
        
        if include_all:
            for k in [30, 100, 300, 1000]:
                F[f'pc_k{k}'][tgt] = ((sy + (k * 0.5) * pp + (k * 0.5) * pc) / (n + k)).astype(np.float32)
                
    return pd.DataFrame(F, index=R.index).astype(np.float32)

log(f'기준 v7 피처 수: {X.shape[1]}')
base = L.bench2(X, name='기준 v7 (120피처)', nseed=2, log=log)

results = []
for shr in (30, 100, 300, 1000):
    E = exact(shr, False)
    log(f'\n--- [shr={shr}] 유효비율: {E.pc_n.notna().mean():.3f} ---')
    r = L.bench2(pd.concat([X, E[['pc_rate', 'pc_n', 'pc_logn', 'pc_delta']]], axis=1),
                 name=f'정확카운트 shr{shr}', nseed=2, baseline=(base['m24'], base['m23']), log=log)
    results.append(r)

E_all = exact(100, True)
log(f'\n--- [all-shr (다중 k축소 결합)] ---')
r_all = L.bench2(pd.concat([X, E_all], axis=1),
                 name='정확카운트 all-shr', nseed=2, baseline=(base['m24'], base['m23']), log=log)
results.append(r_all)

df_res = pd.DataFrame([base] + results)
df_res.to_csv(OUT + 'summary.csv', index=False)
log('\n===== 최종 요약 =====')
log(df_res.to_string())
log('\n완료')
