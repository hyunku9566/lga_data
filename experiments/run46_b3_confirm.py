"""
46차 — B3(투수 x 타자손 x 카운트) 확정 검증

45차에서 24: +5.3 / 23: +15.7 로 오늘 유일하게 두 폴드 크게 동시 개선.
그러나 폴드2023 은 분산이 큰 폴드다 (30차에서 XGB 설정만 바꿔도 748.7<->766.1, 17점 진폭).
시드 2개로는 확정할 수 없으므로 아래를 잰다.

  C1 시드 5개 재측정        +15.7 이 시드 잡음인지
  C2 pc_* 와 결합           독립 정보인지 겹치는지
  C3 블렌드 수준            XGB/LGB/CB 3축에서도 살아남는지 (43차와 같은 방식)
  C4 이력 분할              이득이 어느 투수 구간에서 나오는지
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, scipy.special as sp
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier, Pool
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga')
import lib_lga as L

OUT='/home/lee/lga/results46/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'
DEV=os.environ.get('LGA_DEV','cuda:1')
lgt=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))

b=L.load_base(); R=b['RAW']; y=b['y']; season=b['season']
X0=L.build_v7(b=b).astype(np.float32)
pid=R.pitcher_id.values.astype(np.int64)
bh=R.batter_hand.values.astype(np.int64)
cnt=R.balls_before.values.astype(np.int64)*10+R.strikes_before.values.astype(np.int64)

B3=pd.read_parquet('/home/lee/lga/results45/B3.parquet')
log(f'{el()} B3 {B3.shape}')

def pc_feats(k=100.):
    """41차 pc_* 와 동일 (투수 x 정확카운트, 계층축소)."""
    key=pid*100+cnt
    names=['pc_rate','pc_n','pc_logn','pc_delta']
    F={c:np.full(len(R),np.nan,np.float32) for c in names}
    for s in range(2020,2026):
        tgt=season==s; prev=season<s
        if not tgt.any() or not prev.any(): continue
        mu=float(y[prev].mean())
        gk=pd.Series(y[prev]).groupby(key[prev]).agg(['sum','size'])
        gp=pd.Series(y[prev]).groupby(pid[prev]).mean()
        gl=pd.Series(y[prev]).groupby(cnt[prev]).mean()
        kt=pd.Series(key[tgt]); pt=pd.Series(pid[tgt]); ct=pd.Series(cnt[tgt])
        n=kt.map(gk['size']).fillna(0).values.astype(np.float64)
        sy=kt.map(gk['sum']).fillna(0).values.astype(np.float64)
        pa=pt.map(gp).fillna(mu).values.astype(np.float64)
        pb=ct.map(gl).fillna(mu).values.astype(np.float64)
        rate=(sy+(k*.5)*pa+(k*.5)*pb)/(n+k)
        F['pc_rate'][tgt]=rate; F['pc_n'][tgt]=n
        F['pc_logn'][tgt]=np.log1p(n); F['pc_delta'][tgt]=rate-pa
    return pd.DataFrame(F,index=R.index).astype(np.float32)
PC=pc_feats()

log(f'\n{el()} ===== C1 시드 5개 재측정 =====')
base=L.bench2(X0, name='기준 v7', nseed=5, log=log)
BL=(base['m24'],base['m23'])
r_b3=L.bench2(pd.concat([X0,B3],axis=1), name='B3 (시드5)', nseed=5, baseline=BL, log=log)

log(f'\n{el()} ===== C2 pc_* 와의 관계 =====')
L.bench2(pd.concat([X0,PC],axis=1),          name='pc_* 단독 (시드5)', nseed=5, baseline=BL, log=log)
L.bench2(pd.concat([X0,PC,B3],axis=1),       name='pc_* + B3 (시드5)', nseed=5, baseline=BL, log=log)

log(f'\n{el()} ===== C3 블렌드 수준 =====')
XP=dict(n_estimators=2000,learning_rate=0.005,max_depth=10,min_child_weight=6000,
        subsample=0.7,colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,
        tree_method='hist',device=DEV,eval_metric='logloss',verbosity=0)
LP=dict(n_estimators=1200,learning_rate=0.01,num_leaves=15,min_child_samples=6000,
        subsample=0.7,subsample_freq=1,colsample_bytree=0.5,reg_lambda=50.,verbose=-1,n_jobs=24)
CB=dict(iterations=2000,learning_rate=0.03,depth=6,l2_leaf_reg=50.,loss_function='Logloss',
        random_seed=0,verbose=0,task_type='GPU',devices=DEV.split(':')[-1])
CATF=['pitcher_id','batter_id','pitcher_team_id','batter_team_id','pitcher_hand','batter_hand']
W3=dict(xgb=.45,lgb=.35,cb=.20); WN=0.30
def axes(X,ctx,tag):
    ck=OUT+f'ax_{tag}.npz'
    if os.path.exists(ck):
        d=np.load(ck); log(f'{el()}   [{tag}] 캐시'); return d['px'],d['pl'],d['pc']
    tr,va,w=ctx['tr'],ctx['va'],ctx['w']
    px=np.mean([xgb.XGBClassifier(**XP,random_state=s).fit(X[tr],y[tr],sample_weight=w)
                .predict_proba(X[va])[:,1] for s in range(2)],0)
    pl=np.mean([lgb.LGBMClassifier(**LP,random_state=s).fit(X[tr],y[tr],sample_weight=w)
                .predict_proba(X[va])[:,1] for s in range(3)],0)
    Z=X.copy(); cf=[c for c in CATF if c in Z.columns]
    for c in cf: Z[c]=Z[c].fillna(-1).astype(np.int64)
    pcb=CatBoostClassifier(**CB).fit(Pool(Z[tr],y[tr],weight=w,cat_features=cf)).predict_proba(Z[va])[:,1]
    np.savez(ck,px=px,pl=pl,pc=pcb); log(f'{el()}   [{tag}] 3축 완료')
    return px,pl,pcb
XB=pd.concat([X0,B3],axis=1)
for vs in (2024,2023):
    ctx=L.get_ctx(vs,2.0); yv,bq=ctx['yv'],ctx['base']
    a0=axes(X0,ctx,f'{vs}_base'); a1=axes(XB,ctx,f'{vs}_b3')
    zn=lgt(a0[0])
    def bl3(a): return sp.expit(W3['xgb']*lgt(a[0])+W3['lgb']*lgt(a[1])+W3['cb']*lgt(a[2]))
    def bl4(a): return sp.expit((1-WN)*(W3['xgb']*lgt(a[0])+W3['lgb']*lgt(a[1])+W3['cb']*lgt(a[2]))+WN*zn)
    log(f'{el()}  폴드{vs}')
    for nm,f in [('XGB 단독',lambda a:a[0]),('트리 3축',bl3),('4축(NN대리)',bl4)]:
        s0=L.bss(f(a0),yv,bq); s1=L.bss(f(a1),yv,bq)
        log(f'          {nm:12s} {s0:7.1f} → {s1:7.1f}   {s1-s0:+6.2f}')
    if vs==2024:
        hist=pd.Series(y[season<2024]).groupby(pid[season<2024]).size()
        h=pd.Series(pid[ctx['va']]).map(hist).fillna(0).values
        p0=bl3(a0); p1=bl3(a1); N=len(yv)
        log(f'{el()} ===== C4 이력 분할 (폴드2024) =====')
        for lo,hi,nm in [(-1,0,'미등장'),(0,500,'1~500'),(500,2000,'500~2k'),(2000,10**9,'2k+')]:
            m=(h>lo)&(h<=hi)
            if not m.any(): continue
            c0=-100000*np.sum((p0[m]-yv[m])**2)/N/bq; c1=-100000*np.sum((p1[m]-yv[m])**2)/N/bq
            log(f'          {nm:8s} {m.sum():>7,} {m.mean()*100:5.1f}%   {c1-c0:+8.2f}')
log(f'{el()} 완료')
