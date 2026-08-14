"""
22차 — 병목 진단: 모델인가 데이터인가

세 가지를 재면 답이 나온다.

  A 학습곡선   : 학습행 25% / 50% / 100%
                 아직 오르는 중이면 -> 데이터량/모델 병목
                 포화됐으면        -> 정보량 병목 (더 짜낼 게 없음)

  B 용량곡선   : depth 4/6/8/10, 트리 600/2000
                 더 키워서 오르면 -> 모델이 과소적합 (모델 병목)
                 떨어지면        -> 이미 충분 (정보 병목)

  C 시간분할 vs 무작위분할
                 무작위분할이 훨씬 높으면 -> 시간 일반화가 병목 (드리프트/신규선수)
                 비슷하면              -> 신호 자체의 천장

  ※ C 의 무작위분할은 같은 경기/같은 투수 행이 양쪽에 들어가므로
    규칙상 제출에 못 쓴다. 오직 '천장 측정' 용도다.
"""
import os, json, time, warnings
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results22/'; os.makedirs(OUT,exist_ok=True)
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

BASEP=dict(n_estimators=600,learning_rate=0.008,max_depth=6,min_child_weight=1500,subsample=0.7,
           colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,tree_method='hist',device='cuda:0',
           eval_metric='logloss',verbosity=0)
NS=2
def fit_eval(trm,vam,prm=None,w=None,tag=''):
    P=dict(BASEP);
    if prm: P.update(prm)
    yv=y[vam]; b=yv.mean()*(1-yv.mean())
    p=np.mean([xgb.XGBClassifier(**P,random_state=s).fit(V7[trm],y[trm],sample_weight=w)
               .predict_proba(V7[vam])[:,1] for s in range(NS)],0)
    v=100000*max(0.,1-np.mean((p-yv)**2)/b)
    log(f'{el()} {tag:38s} 학습{trm.sum():>9,} BSS {v:7.1f}')
    return v

FOLD=2024
tr=(season<FOLD)&~(isF&(season<=2022)); va=(season==FOLD)&~isF
rs=np.random.RandomState(0)

log(f'\n{el()} ===== A 학습곡선 (시간분할, 폴드2024) =====')
idx=np.where(tr)[0]
for frac in [0.25,0.5,1.0]:
    m=np.zeros(len(y),bool); sel=rs.choice(idx,int(len(idx)*frac),replace=False); m[sel]=True
    fit_eval(m,va,tag=f'A 학습 {int(frac*100)}%')

log(f'\n{el()} ===== B 용량곡선 =====')
for d in [4,6,8,10]:
    fit_eval(tr,va,prm=dict(max_depth=d),tag=f'B depth {d}')
for n,lr in [(2000,0.008),(2000,0.02),(600,0.03)]:
    fit_eval(tr,va,prm=dict(n_estimators=n,learning_rate=lr),tag=f'B n{n} lr{lr}')
for mcw in [200,1500,6000]:
    fit_eval(tr,va,prm=dict(min_child_weight=mcw),tag=f'B mcw {mcw}')

log(f'\n{el()} ===== C 시간분할 vs 무작위분할 (천장 측정) =====')
allm=~(isF&(season<=2022))
ai=np.where(allm)[0]; rs2=np.random.RandomState(1); rs2.shuffle(ai)
cut=int(len(ai)*0.8)
trR=np.zeros(len(y),bool); trR[ai[:cut]]=True
vaR=np.zeros(len(y),bool); vaR[ai[cut:]]=True
fit_eval(trR,vaR,tag='C 무작위 80/20 (전 시즌 혼합)')
# 같은 크기의 시간분할 대조
fit_eval(tr,va,tag='C 시간분할 (<=2023 -> 2024)')
log(f'\n{el()} 완료')
