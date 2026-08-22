"""
45차 — 타자축 및 교차 이력  (GPU1)

배경
  pc_* (투수x정확카운트) 가 블렌드 수준 +4.66/+1.08 로 살아남았다.
  그런데 대칭인 타자축은 한 번도 만든 적이 없다. 제공 컬럼에
  asof_batter_n / asof_batter_success_rate / asof_batter_middle_rate 가 있는데
  타자x카운트 조회표가 없다. 투수축과 겹치지 않는 독립 정보다.

측정 (전부 시즌 인과적, 조회키에 test 행 간 참조 없음)
  B1 타자x정확카운트 이력
  B2 타자 x 투수손
  B3 투수 x 타자손 x 카운트
  B4 투수 x 이닝구간
  B5 전부 결합
  B6 기검증 후보 합산 (pc_* + run33 그룹3) — 서로 겹치는지 확인

채택 기준: bench2 (두 폴드 동시 개선)
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, xgboost as xgb
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga')
import lib_lga as L

OUT='/home/lee/lga/results45/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'
DEV=os.environ.get('LGA_DEV','cuda:1')

b=L.load_base(); R=b['RAW']; y=b['y']; season=b['season']
X0=L.build_v7(b=b).astype(np.float32)
pid=R.pitcher_id.values.astype(np.int64); bid=R.batter_id.values.astype(np.int64)
ph=R.pitcher_hand.values.astype(np.int64); bh=R.batter_hand.values.astype(np.int64)
cnt=R.balls_before.values.astype(np.int64)*10+R.strikes_before.values.astype(np.int64)
inn=np.clip(R.inning.values.astype(np.int64),1,9)
log(f'{el()} v7 {X0.shape[1]} 피처')

def hier(keys, base_keys, pref, k=100.):
    """계층 축소 조회 피처. keys=(세분키, 상위키) 배열 튜플.
       rate / n / logn / delta 4개를 만든다. 전부 시즌 인과적."""
    kf, kb = keys
    F={f'{pref}_rate':np.full(len(R),np.nan,np.float32),
       f'{pref}_n':np.full(len(R),np.nan,np.float32),
       f'{pref}_logn':np.full(len(R),np.nan,np.float32),
       f'{pref}_delta':np.full(len(R),np.nan,np.float32)}
    for s in range(2020,2026):
        tgt=season==s; prev=season<s
        if not tgt.any() or not prev.any(): continue
        mu=float(y[prev].mean())
        gf=pd.Series(y[prev]).groupby(kf[prev]).agg(['sum','size'])
        gb=pd.Series(y[prev]).groupby(kb[prev]).mean()
        gl=pd.Series(y[prev]).groupby(base_keys[prev]).mean()
        tf=pd.Series(kf[tgt]); tb=pd.Series(kb[tgt]); tl=pd.Series(base_keys[tgt])
        num=tf.map(gf['sum']).fillna(0).values.astype(np.float64)
        den=tf.map(gf['size']).fillna(0).values.astype(np.float64)
        pa=tb.map(gb).fillna(mu).values.astype(np.float64)
        pb=tl.map(gl).fillna(mu).values.astype(np.float64)
        rate=(num+(k*.5)*pa+(k*.5)*pb)/(den+k)
        F[f'{pref}_rate'][tgt]=rate; F[f'{pref}_n'][tgt]=den
        F[f'{pref}_logn'][tgt]=np.log1p(den); F[f'{pref}_delta'][tgt]=rate-pa
    return pd.DataFrame(F,index=R.index).astype(np.float32)

t=time.time()
B1=hier((bid*100+cnt, bid), cnt, 'bc')                    # 타자x카운트
B2=hier((bid*10+ph,  bid), ph,  'bp')                     # 타자x투수손
B3=hier((pid*1000+bh*100+cnt, pid), cnt, 'pbc')           # 투수x타자손x카운트
B4=hier((pid*100+inn, pid), inn, 'pi')                    # 투수x이닝
log(f'{el()} 피처 생성 완료 ({time.time()-t:.0f}s)  bc/bp/pbc/pi 각 4개')
for nm,E in [('B1',B1),('B2',B2),('B3',B3),('B4',B4)]: E.to_parquet(OUT+f'{nm}.parquet')

log(f'\n{el()} ===== 스윕 =====')
base=L.bench2(X0, name='기준 v7', nseed=2, log=log)
BL=(base['m24'],base['m23'])
ARMS=[('B1 타자x카운트',B1), ('B2 타자x투수손',B2),
      ('B3 투수x타자손x카운트',B3), ('B4 투수x이닝',B4),
      ('B5 전부결합',pd.concat([B1,B2,B3,B4],axis=1))]
out=[base]
for nm,E in ARMS:
    out.append(L.bench2(pd.concat([X0,E],axis=1), name=nm, nseed=2, baseline=BL, log=log))
pd.DataFrame(out).to_csv(OUT+'summary.csv',index=False)
log(f'{el()} 완료')
