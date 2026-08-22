"""
48차 — pbc 계열 변형  (GPU1)

기준선 v16 (124 피처) 위에서 pbc 조회키/축소강도를 변형해 더 붙는지 본다.
  V1 주자상황 추가   투수 x 타자손 x 카운트 x 주자유무
  V2 아웃카운트 추가 투수 x 타자손 x 카운트 x 아웃
  V3 k=30            현재 k=100 보다 약한 축소
  V4 k=300           강한 축소
  V5 구종배합        투수 x 타자손 x 카운트별 fb/br/os 비율
     (44차에서 타자손 없이 했을 때는 24 +3.1 / 23 -7.5 로 기각됐다)
채택 기준: bench2 두 폴드 동시 개선.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga')
import lib_lga as L

OUT='/home/lee/lga/results48/'; os.makedirs(OUT,exist_ok=True)
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
onb=(R.num_runners_on.values>0).astype(np.int64)
out_=np.clip(R.outs_before.values.astype(np.int64),0,2)
comp, cls, valid = L.recover_labels(R)

def hier(kf, pref, k=100., tgt_arr=None):
    t = y.astype(np.float64) if tgt_arr is None else tgt_arr
    F={f'{pref}_r':np.full(len(R),np.nan,np.float32),
       f'{pref}_n':np.full(len(R),np.nan,np.float32),
       f'{pref}_d':np.full(len(R),np.nan,np.float32)}
    ok=np.isfinite(t)
    for s in range(2020,2026):
        tg=season==s; prev=(season<s)&ok
        if not tg.any() or not prev.any(): continue
        tv=t[prev]; mu=float(tv.mean())
        gk=pd.Series(tv).groupby(kf[prev]).agg(['sum','size'])
        gp=pd.Series(tv).groupby(pid[prev]).mean()
        gl=pd.Series(tv).groupby(cnt[prev]).mean()
        kt=pd.Series(kf[tg]); pt=pd.Series(pid[tg]); ct=pd.Series(cnt[tg])
        n =kt.map(gk['size']).fillna(0).values.astype(np.float64)
        sy=kt.map(gk['sum']).fillna(0).values.astype(np.float64)
        pa=pt.map(gp).fillna(mu).values.astype(np.float64)
        pb=ct.map(gl).fillna(mu).values.astype(np.float64)
        rate=(sy+(k*.5)*pa+(k*.5)*pb)/(n+k)
        F[f'{pref}_r'][tg]=rate; F[f'{pref}_n'][tg]=n; F[f'{pref}_d'][tg]=rate-pa
    return pd.DataFrame(F,index=R.index).astype(np.float32)

K0=pid*1000+bh*100+cnt
t=time.time()
V1=hier(K0*10+onb,  'v1')
V2=hier(K0*10+out_, 'v2')
V3=hier(K0,'v3',k=30.)
V4=hier(K0,'v4',k=300.)
mixF={}
for i,nm in enumerate(('fb','br','os')):
    ind=np.where(valid,(cls==i).astype(np.float64),np.nan)
    mixF[nm]=hier(K0,f'v5{nm}',tgt_arr=ind)[[f'v5{nm}_r',f'v5{nm}_d']]
V5=pd.concat(list(mixF.values()),axis=1)
log(f'{el()} 변형 피처 생성 완료 ({time.time()-t:.0f}s)')

log(f'\n{el()} ===== 스윕 (기준선 = v16) =====')
base=L.bench2(XV16, name='기준 v16', nseed=3, log=log)
BL=(base['m24'],base['m23'])
out=[base]
for nm,E in [('V1 +주자유무',V1),('V2 +아웃',V2),('V3 k=30',V3),
             ('V4 k=300',V4),('V5 구종배합',V5)]:
    out.append(L.bench2(pd.concat([XV16,E],axis=1), name=nm, nseed=3, baseline=BL, log=log))
pd.DataFrame(out).to_csv(OUT+'summary.csv',index=False)
log(f'{el()} 완료')
