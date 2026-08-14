"""
27차 — 자원 제약 없는 '무겁고 구조적으로 다른' 학습기 탐색

측정된 사실:
  XGB      d6 818.9 -> d10 829.4   키우면 좋아짐 (튜닝 후 863)
  LGB      leaves31 796 -> 127 698 키우면 나빠짐
  CatBoost d6 745 -> d8 640        키우면 나빠짐
  => '무겁게'가 보편 레버는 아니고 XGB 한정. 그러면 XGB 를 더 밀거나,
     아직 안 써본 '구조가 다른' 학습기를 찾아야 한다.

이번에 보는 것 (전부 폴드2024 단독 기준, 자원 제약 없음)
  H1 XGB 초대형   n4000/lr0.004, d12, lossguide+max_leaves 대형
  H2 XGB dart     부스팅 단계에 드롭아웃. 느리지만 과적합에 강함
  H3 XGB exact    hist 대신 정확 분할 탐색. 훨씬 느리지만 분할이 정밀
  H4 LGB linear_tree  리프에 선형모델을 적합. 저신호·매끄러운 관계에 유리할 수 있음
  H5 XGB 초저lr 장시간  n8000/lr0.002
"""
import os, json, time, warnings
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb, lightgbm as lgb
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results27/'; os.makedirs(OUT,exist_ok=True)
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
V10=pd.concat([X98[CORE],multi_k(),TM[TMSEL],OF,G1,G2],axis=1)
log(f'{el()} 피처 {V10.shape[1]}')

HL=2.0; NS=2; DEV='cuda:1'
tr=(season<2024)&~(isF&(season<=2022)); va=(season==2024)&~isF
W=(0.5**((2023-season[tr])/HL)).astype(np.float32)
yv=y[va]; bq=yv.mean()*(1-yv.mean())
def bss(p): return 100000*max(0.,1-np.mean((p-yv)**2)/bq)
R=[]
def xgo(prm,name,ns=NS):
    t=time.time()
    p=np.mean([xgb.XGBClassifier(**prm,random_state=s,eval_metric='logloss',verbosity=0)
               .fit(V10[tr],y[tr],sample_weight=W).predict_proba(V10[va])[:,1] for s in range(ns)],0)
    v=bss(p); R.append(dict(name=name,bss=v,sec=time.time()-t))
    pd.DataFrame(R).to_csv(OUT+'res27.csv',index=False)
    np.save(OUT+f'{name.replace(" ","_").replace("/","-")}.npy',p.astype(np.float32))
    log(f'{el()} {name:40s} BSS {v:7.1f}  ({(time.time()-t)/60:.1f}분)')
    return v

# 23차 폴드2024 최적을 기준선으로
BEST=dict(n_estimators=600,learning_rate=0.015,max_depth=10,min_child_weight=6000,subsample=0.7,
          colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,tree_method='hist',device=DEV)
log(f'\n{el()} ===== 기준: 23차 최적 =====')
b=xgo(BEST,'기준 d10/mcw6000/n600/lr0.015')

log(f'\n{el()} ===== H1 초대형 =====')
for n,lr,d in [(2000,0.005,10),(4000,0.004,10),(2000,0.005,12)]:
    p=dict(BEST); p.update(n_estimators=n,learning_rate=lr,max_depth=d)
    xgo(p,f'H1 n{n} lr{lr} d{d}')
for ml in [512,2048]:
    p=dict(BEST); p.update(grow_policy='lossguide',max_leaves=ml,max_depth=0,
                           n_estimators=1500,learning_rate=0.008)
    xgo(p,f'H1 lossguide leaves{ml}')

log(f'\n{el()} ===== H2 dart =====')
p=dict(BEST); p.update(booster='dart',rate_drop=0.1,skip_drop=0.5,n_estimators=1000,learning_rate=0.02)
xgo(p,'H2 dart n1000 lr0.02',ns=1)

log(f'\n{el()} ===== H5 초저lr 장시간 =====')
p=dict(BEST); p.update(n_estimators=8000,learning_rate=0.002)
xgo(p,'H5 n8000 lr0.002',ns=1)

log(f'\n{el()} ===== H4 LGB linear_tree =====')
for lv,lam in [(31,50.),(15,200.)]:
    t=time.time()
    try:
        pp=np.mean([lgb.LGBMClassifier(n_estimators=800,learning_rate=0.02,num_leaves=lv,
                    min_child_samples=6000,subsample=0.7,subsample_freq=1,colsample_bytree=0.5,
                    reg_lambda=lam,linear_tree=True,random_state=s,verbose=-1,n_jobs=36)
                    .fit(V10[tr],y[tr],sample_weight=W).predict_proba(V10[va])[:,1] for s in range(1)],0)
        v=bss(pp); R.append(dict(name=f'H4 linear_tree leaves{lv}',bss=v,sec=time.time()-t))
        pd.DataFrame(R).to_csv(OUT+'res27.csv',index=False)
        np.save(OUT+f'H4_linear_leaves{lv}.npy',pp.astype(np.float32))
        log(f'{el()} {"H4 linear_tree leaves"+str(lv):40s} BSS {v:7.1f}  ({(time.time()-t)/60:.1f}분)')
    except Exception as e:
        log(f'{el()} H4 linear_tree leaves{lv} 실패: {type(e).__name__}: {e}')

log(f'\n{el()} ===== H3 exact (마지막, 매우 느림) =====')
p=dict(BEST); p.update(tree_method='exact',device='cpu',n_estimators=300,learning_rate=0.03,n_jobs=36)
xgo(p,'H3 exact n300 lr0.03',ns=1)

log(f'\n{el()} 완료')
d=pd.DataFrame(R).sort_values('bss',ascending=False)
log(d.to_string(index=False))
