"""
19차 — CatBoost (트리 계열 중 유일한 미시도)

v7 구성(120피처 = 기준114 + 성분OOF 6)에 최근성 가중 hl=2 를 그대로 적용.
관심사는 단독 성능이 아니라 'XGB+LGB 축에 얹었을 때의 이득'.

CatBoost 를 쓰는 이유: ordered target statistics 로 고카디널리티 범주를
XGB/LGB 와 다른 방식으로 처리한다. 다양성이 붙을 여지가 여기에 있다.
  A 수치만        (XGB 와 동일 입력)
  B 범주 네이티브   (pitcher_id/batter_id 등을 cat_features 로)
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd, scipy.special as sp
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results19/'; os.makedirs(OUT,exist_ok=True)
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
        S=t.loc[t.groupby([idc,'season'])[nc].idxmin()].set_index([idc,'season'])[[nc,'succ']]
        a=RAW[[idc,'season']].join(S,on=[idc,'season'])
        dn=np.maximum(RAW[nc].values-a[nc].fillna(0).values,0)
        ds=np.maximum(np.nan_to_num(RAW[nc].values*RAW[rc].values)-a['succ'].fillna(0).values,0)
        lgv=np.nanmean(RAW[rc])
        for k in [25,75,400,1000]: F[f'{pf}_k{k}']=(ds+k*lgv)/(dn+k)
    return pd.DataFrame(F,index=RAW.index).astype(np.float32)
OF=pd.read_parquet('/tmp/claude-1000/-home-lee-lga/97188a20-36c6-4af4-9fed-509e8b2fcd01/scratchpad/oof_comp.parquet')
XK=pd.concat([X98[CORE],multi_k(),TM[TMSEL],OF],axis=1)
log(f'{el()} 피처 {XK.shape[1]}')

CATF=['pitcher_id','batter_id','pitcher_team_id','batter_team_id','pitcher_hand',
      'batter_hand','base_state','game_type','top_bottom']
FOLDS=[2024,2022]; HL=2.0
lg_=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))
def split(vs): return (season<vs)&~(isF&(season<=2022)&(vs>=2023)), (season==vs)&~isF
def bss(p,vs):
    va=(season==vs)&~isF; yv=y[va]; r=yv.mean()
    return 100000*max(0.,1-np.mean((p-yv)**2)/(r*(1-r)))

# ── 트리 축 기준선 (XGB+LGB, v7 구성) ──
XP=dict(n_estimators=600,learning_rate=0.008,max_depth=6,min_child_weight=1500,subsample=0.7,
        colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,tree_method='hist',device='cuda:0',
        eval_metric='logloss',verbosity=0)
import lightgbm as lgb
LP=dict(n_estimators=1200,learning_rate=0.01,num_leaves=31,min_child_samples=1500,
        subsample=0.7,subsample_freq=1,colsample_bytree=0.5,reg_lambda=50.)
TREE={}; B0={}
for vs in FOLDS:
    tr,va=split(vs); w=(0.5**((vs-1-season[tr])/HL)).astype(np.float32)
    px=np.mean([xgb.XGBClassifier(**XP,random_state=s).fit(XK[tr],y[tr],sample_weight=w)
                .predict_proba(XK[va])[:,1] for s in range(3)],0)
    pl=np.mean([lgb.LGBMClassifier(**LP,random_state=s,verbose=-1,n_jobs=24)
                .fit(XK[tr],y[tr],sample_weight=w).predict_proba(XK[va])[:,1] for s in range(2)],0)
    np.save(OUT+f'xgb7_{vs}.npy',px.astype(np.float32)); np.save(OUT+f'lgb7_{vs}.npy',pl.astype(np.float32))
    TREE[vs]=(0.45*lg_(px)+0.35*lg_(pl))/0.8; B0[vs]=bss(sp.expit(TREE[vs]),vs)
    log(f'{el()} [{vs}] XGB {bss(px,vs):7.1f} | LGB {bss(pl,vs):7.1f} | 축 {B0[vs]:7.1f}')

CB=dict(iterations=2000,learning_rate=0.03,depth=6,l2_leaf_reg=50.,
        loss_function='Logloss',random_seed=0,verbose=0,task_type='GPU',devices='0')
def bench(name,use_cat):
    out={}
    for vs in FOLDS:
        tr,va=split(vs); w=(0.5**((vs-1-season[tr])/HL)).astype(np.float32)
        if use_cat:
            Z=XK.copy()
            for c in CATF: Z[c]=Z[c].fillna(-1).astype(np.int32).astype(str)
            ptr=Pool(Z[tr],y[tr],weight=w,cat_features=CATF); pva=Pool(Z[va],cat_features=CATF)
        else:
            Z=XK.drop(columns=CATF)
            ptr=Pool(Z[tr],y[tr],weight=w); pva=Pool(Z[va])
        m=CatBoostClassifier(**CB).fit(ptr)
        p=m.predict_proba(pva)[:,1]
        np.save(OUT+f'cb_{name}_{vs}.npy',p.astype(np.float32))
        g=max(bss(sp.expit((1-q)*TREE[vs]+q*lg_(p)),vs)-B0[vs] for q in [0.1,0.2,0.3,0.4,0.5])
        out[vs]=(bss(p,vs),g)
        log(f'{el()} [{vs}] CB {name:14s} 단독 {out[vs][0]:7.1f} | 축에 얹은 이득 {g:+6.1f}')
    return out

log(f'\n{el()} ===== CatBoost =====')
bench('A_수치만',False)
bench('B_범주네이티브',True)
log(f'{el()} 완료')
