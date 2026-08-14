"""
21차 — 타깃 인수분해 (지금까지 안 해본 마지막 구조적 아이디어)

y = ¬reverse ∧ ¬middle ∧ Z  이므로
    P(y=1|x) = P(¬rev ∧ ¬mid | x) · P(y=1 | ¬rev,¬mid, x)

3차에서 실패했던 혼합분해( P(y)=Σ P(c|x)P(y|c) , 651점 )와는 다르다.
그건 셀 안에서 x 를 버렸고, 이건 두 인수 모두 x 를 유지한다.
원리상 직접 모델링보다 표현력이 크거나 같다. 문제는 추정 노이즈뿐이다.

  F1 : 'not-bad' = ¬rev ∧ ¬mid 를 이진 타깃으로 (라벨 정확히 보유)
  F2 : ¬rev∧¬mid 인 행만 골라 y 를 학습
  최종 : p = p_F1 * p_F2
"""
import os, json, time, warnings
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results21/'; os.makedirs(OUT,exist_ok=True)
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

# 성분 라벨
ordr=np.lexsort((RAW.asof_pitcher_n.values, RAW.pitcher_id.values))
pid=RAW.pitcher_id.values[ordr]; n=RAW.asof_pitcher_n.values[ordr].astype(np.float64)
last=np.append(pid[1:]!=pid[:-1],True)
LB={}
for c in ['reverse','middle']:
    cum=np.nan_to_num(n*RAW[f'asof_pitcher_{c}_rate'].values[ordr])
    d=np.append(cum[1:]-cum[:-1],np.nan); d[last]=np.nan
    v=np.round(d); v[np.abs(d-v)>0.3]=np.nan
    o=np.full(len(RAW),np.nan,np.float32); o[ordr]=v; LB[c]=o
L=pd.DataFrame(LB); ok=L.notna().all(1).values
notbad=((L.reverse.values==0)&(L.middle.values==0)).astype(np.float32)
log(f'{el()} not-bad 비율 {notbad[ok].mean():.4f} | 그 안에서 y 비율 {y[ok&(notbad==1)].mean():.4f}')

XP=dict(n_estimators=600,learning_rate=0.008,max_depth=6,min_child_weight=1500,subsample=0.7,
        colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,tree_method='hist',device='cuda:0',
        eval_metric='logloss',verbosity=0)
HL=2.0; NS=3
lgt=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))
for FOLD in [2024,2022]:
    tr=(season<FOLD)&~(isF&(season<=2022)&(FOLD>=2023)); va=(season==FOLD)&~isF
    yv=y[va]; bse=yv.mean()*(1-yv.mean())
    def bss(p): return 100000*max(0.,1-np.mean((p-yv)**2)/bse)
    W=(0.5**((FOLD-1-season[tr])/HL)).astype(np.float32)
    log(f'\n{el()} ===== 폴드 {FOLD} =====')
    pd_=np.mean([xgb.XGBClassifier(**XP,random_state=s).fit(V7[tr],y[tr],sample_weight=W)
                 .predict_proba(V7[va])[:,1] for s in range(NS)],0)
    log(f'{el()} 직접 P(y|x)                BSS {bss(pd_):7.1f}')
    t1=tr&ok
    p1=np.mean([xgb.XGBClassifier(**XP,random_state=s).fit(V7[t1],notbad[t1],
                sample_weight=(0.5**((FOLD-1-season[t1])/HL)).astype(np.float32))
                .predict_proba(V7[va])[:,1] for s in range(NS)],0)
    t2=tr&ok&(notbad==1)
    p2=np.mean([xgb.XGBClassifier(**XP,random_state=s).fit(V7[t2],y[t2],
                sample_weight=(0.5**((FOLD-1-season[t2])/HL)).astype(np.float32))
                .predict_proba(V7[va])[:,1] for s in range(NS)],0)
    log(f'{el()} F1 P(not-bad) 단독          BSS(자체) — 학습 {t1.sum():,}')
    log(f'{el()} F2 P(y|not-bad) 단독        학습 {t2.sum():,}')
    pf=np.clip(p1*p2,1e-6,1-1e-6)
    log(f'{el()} 인수분해 p1*p2             BSS {bss(pf):7.1f}   순효과 {bss(pf)-bss(pd_):+6.1f}')
    for a in [0.3,0.5,0.7]:
        pb=sp.expit((1-a)*lgt(pd_)+a*lgt(pf))
        log(f'{el()}   직접과 {a:.1f} 블렌딩        BSS {bss(pb):7.1f}   순효과 {bss(pb)-bss(pd_):+6.1f}')
    np.save(OUT+f'fact_{FOLD}.npy',pf.astype(np.float32)); np.save(OUT+f'direct_{FOLD}.npy',pd_.astype(np.float32))
log(f'\n{el()} 완료')
