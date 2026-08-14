"""
26차 — LGB / CatBoost 재탐색 (마지막 미탐색 영역)

23차에서 XGB 는 d6->d10, mcw1500->6000 으로 폴드2024 +41.5 가 나왔다.
그런데 LGB(num_leaves=31, 깊이 약 5)와 CatBoost(depth 6)는 한 번도
재탐색된 적이 없다. XGB 가 훨씬 큰 모델을 원한다면 이쪽도 같을 가능성이 크다.

기준은 폴드2024 단독. (폴드2022 는 regime 이 달라 참고만)
"""
import os, json, time, warnings
import numpy as np, pandas as pd, scipy.special as sp
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results26/'; os.makedirs(OUT,exist_ok=True)
S='/tmp/claude-1000/-home-lee-lga/97188a20-36c6-4af4-9fed-509e8b2fcd01/scratchpad/'
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'

RAW=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig')
X98=pd.read_parquet(D+'X98.parquet'); TM=pd.read_parquet(D+'results14/tm5.parquet')
y=X98.__y.values.astype(np.float32); season=X98.__season.values; isF=X98.__F.values.astype(bool)
CORE=[c for c in X98.columns if not c.startswith('__')]
TMSEL=json.load(open(D+'v6_tmsel.json'))
def multi_k():
    F={}
    for idc,nc,rc,pf in [('pitcher_id','asof_pitcher_n','asof_pitcher_success_rate','p_succ'),
                         ('batter_id','asof_batter_n','asof_batter_success_rate','b_succ')]:
        t=RAW[[idc,'season',nc,rc]].copy(); t['succ']=t[nc]*t[rc].fillna(0)
        Sx=t.loc[t.groupby([idc,'season'])[nc].idxmin()].set_index([idc,'season'])[[nc,'succ']]
        a=RAW[[idc,'season']].join(Sx,on=[idc,'season'])
        dn=np.maximum(RAW[nc].values-a[nc].fillna(0).values,0)
        ds=np.maximum(np.nan_to_num(RAW[nc].values*RAW[rc].values)-a['succ'].fillna(0).values,0)
        lgv=np.nanmean(RAW[rc])
        for k in [25,75,400,1000]: F[f'{pf}_k{k}']=(ds+k*lgv)/(dn+k)
    return pd.DataFrame(F,index=RAW.index).astype(np.float32)
OF=pd.read_parquet(S+'oof_comp.parquet')
G1=pd.read_parquet(D+'results24/G1.parquet'); G2=pd.read_parquet(D+'results24/G2.parquet')
V10=pd.concat([X98[CORE],multi_k(),TM[TMSEL],OF,G1,G2],axis=1)   # 24차 채택 G1+G2 포함
log(f'{el()} v10 피처 {V10.shape[1]}')

HL=2.0; NS=2
tr=(season<2024)&~(isF&(season<=2022)); va=(season==2024)&~isF
W=(0.5**((2023-season[tr])/HL)).astype(np.float32)
yv=y[va]; bq=yv.mean()*(1-yv.mean())
def bss(p): return 100000*max(0.,1-np.mean((p-yv)**2)/bq)
CATF=['pitcher_id','batter_id','pitcher_team_id','batter_team_id','pitcher_hand',
      'batter_hand','base_state','game_type','top_bottom']

R=[]
log(f'\n{el()} ===== CatBoost =====')
Z=V10.copy()
for c in CATF: Z[c]=Z[c].fillna(-1).astype(np.int32).astype(str)
ptr=Pool(Z[tr],y[tr],weight=W,cat_features=CATF); pva=Pool(Z[va],cat_features=CATF)
def cgo(prm,name):
    m=CatBoostClassifier(**prm,loss_function='Logloss',random_seed=0,verbose=0,
                         task_type='GPU',devices='0').fit(ptr)
    p=m.predict_proba(pva)[:,1]; v=bss(p)
    R.append(dict(model='cb',name=name,bss=v,**prm)); pd.DataFrame(R).to_csv(OUT+'res26.csv',index=False)
    log(f'{el()} CB  {name:38s} BSS {v:7.1f}')
    np.save(OUT+f'cb_{name.replace(" ","_")}.npy',p.astype(np.float32)); return v
cgo(dict(iterations=2000,learning_rate=0.03,depth=6,l2_leaf_reg=50.),'현재 d6 n2000 lr0.03')
for d in [8,10]:
    cgo(dict(iterations=2000,learning_rate=0.03,depth=d,l2_leaf_reg=50.),f'd{d} n2000 lr0.03')
cgo(dict(iterations=4000,learning_rate=0.015,depth=8,l2_leaf_reg=50.),'d8 n4000 lr0.015')
cgo(dict(iterations=2000,learning_rate=0.03,depth=8,l2_leaf_reg=200.),'d8 n2000 l2=200')
log(f'\n{el()} ===== LightGBM =====')
base=dict(n_estimators=1200,learning_rate=0.01,num_leaves=31,min_child_samples=1500,
          subsample=0.7,subsample_freq=1,colsample_bytree=0.5,reg_lambda=50.)
def lgo(prm,name):
    p=np.mean([lgb.LGBMClassifier(**prm,random_state=s,verbose=-1,n_jobs=24)
               .fit(V10[tr],y[tr],sample_weight=W).predict_proba(V10[va])[:,1] for s in range(NS)],0)
    v=bss(p); R.append(dict(model='lgb',name=name,bss=v,**prm))
    pd.DataFrame(R).to_csv(OUT+'res26.csv',index=False)
    log(f'{el()} LGB {name:38s} BSS {v:7.1f}')
    np.save(OUT+f'lgb_{name.replace(" ","_")}.npy',p.astype(np.float32)); return v
# 1차 관측: leaves31 796.3 -> leaves127 698.0 (-98).
# LGB 는 XGB 와 반대로 키우면 급격히 나빠진다. '더 작게 / 더 규제' 쪽만 본다.
b=lgo(base,'현재 leaves31 mcs1500')
for lv,mcs in [(15,1500),(15,6000),(31,6000),(31,12000),(63,6000)]:
    p=dict(base); p.update(num_leaves=lv,min_child_samples=mcs)
    lgo(p,f'leaves{lv} mcs{mcs}')
for lv,mcs,n,lr in [(31,6000,2500,0.005),(15,6000,2500,0.005)]:
    p=dict(base); p.update(num_leaves=lv,min_child_samples=mcs,n_estimators=n,learning_rate=lr)
    lgo(p,f'leaves{lv} mcs{mcs} n{n} lr{lr}')

log(f'\n{el()} 완료')
d=pd.DataFrame(R)
for mdl in ['lgb','cb']:
    s=d[d.model==mdl].sort_values('bss',ascending=False)
    if len(s): log(f'{mdl} 최고: {s.name.iloc[0]}  {s.bss.iloc[0]:.1f}  (현재 {s.bss.iloc[-1]:.1f})')
