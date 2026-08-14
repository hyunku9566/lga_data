"""
28차 — v10 구성 확정용 최종 확인

질문: 튜닝 이득(+48)은 163피처(G1+G2 포함)에서 쟀다.
      G1/G2 는 추론 코드에 새로 구현해야 하는 부담이 있다.
      120피처(=v9 그대로)만으로도 같은 점수가 나오면 그 작업이 불필요하다.

      추가로 제출 제약(zip 크기 / 600초 추론)을 위해
      n_estimators 를 줄였을 때의 손실도 같이 잰다.
"""
import os, json, time, warnings
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results28/'; os.makedirs(OUT,exist_ok=True)
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
V7=pd.concat([X98[CORE],multi_k(),TM[TMSEL],OF],axis=1)                     # 120
G1=pd.read_parquet(D+'results24/G1.parquet'); G2=pd.read_parquet(D+'results24/G2.parquet')
V10=pd.concat([V7,G1,G2],axis=1)                                            # 163
V7G2=pd.concat([V7,G2],axis=1)                                              # 136 (G2 는 구현이 쉬움)
log(f'{el()} V7 {V7.shape[1]} / V7+G2 {V7G2.shape[1]} / V10 {V10.shape[1]}')

HL=2.0; DEV='cuda:0'
tr=(season<2024)&~(isF&(season<=2022)); va=(season==2024)&~isF
W=(0.5**((2023-season[tr])/HL)).astype(np.float32)
yv=y[va]; bq=yv.mean()*(1-yv.mean())
def bss(p): return 100000*max(0.,1-np.mean((p-yv)**2)/bq)
P=dict(max_depth=10,min_child_weight=6000,subsample=0.7,colsample_bytree=0.5,
       reg_lambda=50.,reg_alpha=1.,tree_method='hist',device=DEV,eval_metric='logloss',verbosity=0)
R=[]
def go(Xa,n,lr,name,ns=2):
    t=time.time()
    ms=[]; ps=[]
    for s in range(ns):
        m=xgb.XGBClassifier(**P,n_estimators=n,learning_rate=lr,random_state=s).fit(Xa[tr],y[tr],sample_weight=W)
        ps.append(m.predict_proba(Xa[va])[:,1]); ms.append(m)
    p=np.mean(ps,0)
    f=OUT+'tmp.json'; ms[0].get_booster().save_model(f); mb=os.path.getsize(f)/2**20
    v=bss(p); R.append(dict(name=name,nfeat=Xa.shape[1],n=n,lr=lr,bss=v,mb=mb))
    pd.DataFrame(R).to_csv(OUT+'res28.csv',index=False)
    log(f'{el()} {name:34s}({Xa.shape[1]:3d}) n{n:<5} BSS {v:7.1f} | 모델 {mb:6.1f}MB | {(time.time()-t)/60:.1f}분')
    return v

log(f'\n{el()} ===== 피처셋 비교 (n4000/lr0.004) =====')
go(V7,   4000,0.004,'A V7 120피처')
go(V7G2, 4000,0.004,'B V7+G2 136피처')
go(V10,  4000,0.004,'C V10 163피처')
log(f'\n{el()} ===== 트리 수 축소 손실 (최고 피처셋) =====')
for n,lr in [(2000,0.005),(1200,0.008),(800,0.012)]:
    go(V10,n,lr,f'D n{n} lr{lr}')
log(f'\n{el()} 완료')
log(pd.DataFrame(R).sort_values('bss',ascending=False).to_string(index=False))
