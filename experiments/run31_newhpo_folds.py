"""
31차-C — 실제 출시된 모델(v10 HPO)의 폴드2022/2023 예측 생성

31차-B 의 결함: 블렌드 가중치 분석을 results19(옛 HPO XGB) 로 했다.
그런데 실제 LB 실패(-11.66)는 '새 HPO + 집중 블렌드' 조합에서 났다.
큰 XGB 는 2025 로의 외삽이 더 나쁠 수 있으므로 HPO x 블렌드 상호작용이 의심된다.
=> 출시본과 같은 하이퍼파라미터로 폴드2022/2024 예측을 만들어 다시 판정한다.

GPU cuda:1 사용 (cuda:0 은 run31_fold2023.py 가 점유 중).
"""
import os, json, time, warnings
import numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results31/'; os.makedirs(OUT,exist_ok=True)
S='/tmp/claude-1000/-home-lee-lga/97188a20-36c6-4af4-9fed-509e8b2fcd01/scratchpad/'
LOG=open(OUT+'log_newhpo.txt','a',buffering=1)
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
V7=pd.concat([X98[CORE],multi_k(),TM[TMSEL],OF],axis=1)
log(f'{el()} 피처 {V7.shape[1]}')

HL=2.0; NS=3
XPnew=dict(n_estimators=2000,learning_rate=0.005,max_depth=10,min_child_weight=6000,
           subsample=0.7,colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,
           tree_method='hist',device='cuda:1',eval_metric='logloss',verbosity=0)
LPnew=dict(n_estimators=1200,learning_rate=0.01,num_leaves=15,min_child_samples=6000,
           subsample=0.7,subsample_freq=1,colsample_bytree=0.5,reg_lambda=50.)
for vs in [2024,2022]:
    tr=(season<vs)&~(isF&(season<=2022)&(vs>=2023)); va=(season==vs)&~isF
    W=(0.5**((vs-1-season[tr])/HL)).astype(np.float32)
    yv=y[va]; b=yv.mean()*(1-yv.mean())
    def bss(p): return 100000*max(0.,1-np.mean((p-yv)**2)/b)
    px=np.mean([xgb.XGBClassifier(**XPnew,random_state=s).fit(V7[tr],y[tr],sample_weight=W)
                .predict_proba(V7[va])[:,1] for s in range(NS)],0)
    np.save(OUT+f'xgbNEW_{vs}.npy',px.astype(np.float32))
    log(f'{el()} [{vs}] XGB(새HPO) {bss(px):7.1f}')
    pl=np.mean([lgb.LGBMClassifier(**LPnew,random_state=s,verbose=-1,n_jobs=20)
                .fit(V7[tr],y[tr],sample_weight=W).predict_proba(V7[va])[:,1] for s in range(2)],0)
    np.save(OUT+f'lgbNEW_{vs}.npy',pl.astype(np.float32))
    log(f'{el()} [{vs}] LGB(새HPO) {bss(pl):7.1f}')
log(f'{el()} 완료')
