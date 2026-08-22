"""
47차 — pbc 키를 성분 라벨에 적용  (GPU0)

배경
  46차에서 pbc_*(투수 x 타자손 x 정확카운트, y 성공률)가 통과했고 v16 이 LB 1071.89 로
  최고 기록을 세웠다. 같은 조회키를 y 대신 성분 라벨에 적용한다.

  성분 라벨은 y 보다 신호가 2배 강하다 (폴드2024 단독 스킬):
      ball 1778 / reverse 1431 / strike 1211 / middle 887   vs   y 809
  y = ¬reverse ∧ ¬middle ∧ Z 이므로 성분별 투수 성향은 y 와 다른 정보를 담는다.

기준선은 v16 (v7 120 + pbc_* 4 = 124). 그 위에 얼마나 더 붙는지를 잰다.
채택 기준: bench2 두 폴드 동시 개선.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga')
import lib_lga as L

OUT='/home/lee/lga/results47/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'

b=L.load_base(); R=b['RAW']; y=b['y']; season=b['season']
X0=L.build_v7(b=b).astype(np.float32)
B3=pd.read_parquet('/home/lee/lga/results45/B3.parquet')
XV16=pd.concat([X0,B3],axis=1)
log(f'{el()} 기준선 v16 = {XV16.shape[1]} 피처')

pid=R.pitcher_id.values.astype(np.int64)
bh=R.batter_hand.values.astype(np.int64)
cnt=R.balls_before.values.astype(np.int64)*10+R.strikes_before.values.astype(np.int64)
KEY=pid*1000+bh*100+cnt

comp, cls, valid = L.recover_labels(R)
log(f'{el()} 성분 라벨 {list(comp.columns)}  결측률 {comp.isna().mean().round(3).to_dict()}')

def pbc_target(t, pref, k=100.):
    """pbc 키로 임의 타깃 t 의 계층축소 비율. t 는 0/1 (NaN 허용)."""
    F={f'{pref}_r':np.full(len(R),np.nan,np.float32),
       f'{pref}_d':np.full(len(R),np.nan,np.float32)}
    ok=np.isfinite(t)
    for s in range(2020,2026):
        tgt=season==s; prev=(season<s)&ok
        if not tgt.any() or not prev.any(): continue
        tv=t[prev].astype(np.float64); mu=float(tv.mean())
        gk=pd.Series(tv).groupby(KEY[prev]).agg(['sum','size'])
        gp=pd.Series(tv).groupby(pid[prev]).mean()
        gl=pd.Series(tv).groupby(cnt[prev]).mean()
        kt=pd.Series(KEY[tgt]); pt=pd.Series(pid[tgt]); ct=pd.Series(cnt[tgt])
        n =kt.map(gk['size']).fillna(0).values.astype(np.float64)
        sy=kt.map(gk['sum']).fillna(0).values.astype(np.float64)
        pa=pt.map(gp).fillna(mu).values.astype(np.float64)
        pb=ct.map(gl).fillna(mu).values.astype(np.float64)
        rate=(sy+(k*.5)*pa+(k*.5)*pb)/(n+k)
        F[f'{pref}_r'][tgt]=rate; F[f'{pref}_d'][tgt]=rate-pa
    return pd.DataFrame(F,index=R.index).astype(np.float32)

t=time.time(); CF={}
for c in ['reverse','middle','ball','strike']:
    CF[c]=pbc_target(comp[c].values.astype(np.float64), f'cb_{c[:4]}')
    CF[c].to_parquet(OUT+f'cb_{c}.parquet')
log(f'{el()} 성분 pbc 피처 생성 완료 ({time.time()-t:.0f}s)  각 2개')

log(f'\n{el()} ===== 스윕 (기준선 = v16) =====')
base=L.bench2(XV16, name='기준 v16', nseed=3, log=log)
BL=(base['m24'],base['m23'])
ARMS=[('+ball',        CF['ball']),
      ('+reverse',     CF['reverse']),
      ('+strike',      CF['strike']),
      ('+middle',      CF['middle']),
      ('+ball+reverse',pd.concat([CF['ball'],CF['reverse']],axis=1)),
      ('+성분4종',      pd.concat([CF[c] for c in ['reverse','middle','ball','strike']],axis=1))]
out=[base]
for nm,E in ARMS:
    out.append(L.bench2(pd.concat([XV16,E],axis=1), name=nm, nseed=3, baseline=BL, log=log))
pd.DataFrame(out).to_csv(OUT+'summary.csv',index=False)
log(f'{el()} 완료')
