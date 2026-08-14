"""
23차 — 하이퍼파라미터 재탐색 (22차에서 과소적합 확인됨)

22차 단일 변경 실측(폴드2024, v7 120피처, 최근성 hl2):
    현재 d6/mcw1500  818.9
    d8               826.9   d10  829.4
    mcw6000          830.6
두 방향 모두 개선 -> 현재 설정은 명백히 최적이 아니다.
현재값은 98피처 시절(run3) 탐색 결과라 환경이 바뀐 뒤 재탐색된 적이 없다.

폴드2024/2022 양쪽에서 재고, 두 폴드 모두 개선되는 조합만 채택한다.
"""
import os, json, time, itertools, warnings
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results23/'; os.makedirs(OUT,exist_ok=True)
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
V7=pd.concat([X98[CORE],multi_k(),TM[TMSEL],OF],axis=1)
log(f'{el()} 피처 {V7.shape[1]}')

FOLDS=[2024,2022]; HL=2.0; NS=2
CTX={}
for vs in FOLDS:
    tr=(season<vs)&~(isF&(season<=2022)&(vs>=2023)); va=(season==vs)&~isF
    CTX[vs]=(tr,va,(0.5**((vs-1-season[tr])/HL)).astype(np.float32),
             y[va].mean()*(1-y[va].mean()))
def run(prm):
    o={}
    for vs in FOLDS:
        tr,va,w,b=CTX[vs]
        p=np.mean([xgb.XGBClassifier(**prm,random_state=s,tree_method='hist',device='cuda:0',
                                     eval_metric='logloss',verbosity=0)
                   .fit(V7[tr],y[tr],sample_weight=w).predict_proba(V7[va])[:,1] for s in range(NS)],0)
        o[vs]=100000*max(0.,1-np.mean((p-y[va])**2)/b)
    return o

CUR=dict(n_estimators=600,learning_rate=0.008,max_depth=6,min_child_weight=1500,
         subsample=0.7,colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.)
base=run(CUR)
log(f'{el()} {"현재 설정":44s} 24:{base[2024]:7.1f} 22:{base[2022]:7.1f} 평균 {(base[2024]+base[2022])/2:7.1f}')

GRID=[]
for d in [8,10,12]:
    for mcw in [3000,6000,12000]:
        for n,lr in [(600,0.008),(1000,0.008),(600,0.015)]:
            GRID.append(dict(max_depth=d,min_child_weight=mcw,n_estimators=n,learning_rate=lr,
                             subsample=0.7,colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.))
log(f'{el()} 조합 {len(GRID)}개\n')
R=[]
for i,g in enumerate(GRID):
    o=run(g); avg=(o[2024]+o[2022])/2
    ok='O' if (o[2024]>base[2024] and o[2022]>base[2022]) else ' '
    R.append(dict(**g,m24=o[2024],m22=o[2022],avg=avg,both=ok))
    pd.DataFrame(R).sort_values('avg',ascending=False).to_csv(OUT+'res23.csv',index=False)
    log(f'{el()} [{i:2d}] d{g["max_depth"]:<2} mcw{g["min_child_weight"]:<5} n{g["n_estimators"]:<4} '
        f'lr{g["learning_rate"]:<5} | 24:{o[2024]:7.1f} 22:{o[2022]:7.1f} 평균 {avg:7.1f} {ok}')
B=pd.DataFrame(R).sort_values('avg',ascending=False)
log(f'\n{el()} ===== 상위 5 =====\n'+B.head(5)[['max_depth','min_child_weight','n_estimators',
    'learning_rate','m24','m22','avg','both']].to_string(index=False))
log(f'{el()} 기준 평균 {(base[2024]+base[2022])/2:.1f} -> 최고 {B.avg.iloc[0]:.1f} ({B.avg.iloc[0]-(base[2024]+base[2022])/2:+.1f})')
