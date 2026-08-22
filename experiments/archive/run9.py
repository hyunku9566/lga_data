"""
9차 — 블렌딩 이득의 선택편향 제거 + 다양성 원천 비교

문제: run8 의 '+13.1' 은 60개 조합 중 같은 폴드에서 최댓값을 고른 값 -> 과대평가.
방법: (a) 체리피킹 없이 전체 NN 평균과 블렌딩
      (b) 폴드2022 로만 파트너를 고르고 폴드2024 로 평가 (정직한 선택)
      (c) NN 없이 XGB 다양성(하이퍼파라미터/피처부분집합)만으로 블렌딩
          -> 이게 통하면 제출에 torch 를 안 실어도 되므로 훨씬 안전
"""
import os, glob, warnings
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results9/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')

RAW=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig')
X=pd.read_parquet(D+'X98.parquet')
y=X.__y.values; season=X.__season.values; isF=X.__F.values.astype(bool)
CORE=[c for c in X.columns if not c.startswith('__')]
FOLDS=[2024,2022]
def tp(vs):
    m=season<vs; s=pd.Series(y[m]).groupby(season[m]).mean()
    return float(sp.expit(np.polyval(np.polyfit(s.index,sp.logit(s.values),1),vs)))
RP={v:tp(v) for v in FOLDS}
def bss(p,vs):
    va=(season==vs)&~isF; yv=y[va]; r=yv.mean()
    return 100000*max(0.,1-np.mean((p-yv)**2)/(r*(1-r)))

# 다중축소강도 (run8 에서 +6.1 확인)
def multi_k():
    F={}
    for idcol,ncol,ratecol,pref in [('pitcher_id','asof_pitcher_n','asof_pitcher_success_rate','p_succ'),
                                    ('batter_id','asof_batter_n','asof_batter_success_rate','b_succ')]:
        t=RAW[[idcol,'season',ncol,ratecol]].copy(); t['succ']=t[ncol]*t[ratecol].fillna(0)
        S=t.loc[t.groupby([idcol,'season'])[ncol].idxmin()].set_index([idcol,'season'])[[ncol,'succ']]
        a=RAW[[idcol,'season']].join(S,on=[idcol,'season'])
        dn=np.maximum(RAW[ncol].values-a[ncol].fillna(0).values,0)
        ds=np.maximum(np.nan_to_num(RAW[ncol].values*RAW[ratecol].values)-a['succ'].fillna(0).values,0)
        lg=np.nanmean(RAW[ratecol])
        for k in [25,75,400,1000]: F[f'{pref}_k{k}']=(ds+k*lg)/(dn+k)
    return pd.DataFrame(F,index=RAW.index).astype(np.float32)
XK=pd.concat([X[CORE],multi_k()],axis=1)
log(f'기준 피처 {XK.shape[1]}개 (다중축소강도 포함)')

def fit(Xa,vs,prm,seeds=4):
    tr=(season<vs)&~(isF&(season<=2022)&(vs>=2023)); va=(season==vs)&~isF
    return np.mean([xgb.XGBClassifier(**prm,random_state=s).fit(Xa.loc[tr],y[tr])
                    .predict_proba(Xa.loc[va])[:,1] for s in range(seeds)],0)

BASE=dict(n_estimators=600,learning_rate=0.008,max_depth=6,min_child_weight=1500,
          subsample=0.7,colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,
          tree_method='hist',device='cuda:1',eval_metric='logloss',verbosity=0)
PX={vs:fit(XK,vs,BASE) for vs in FOLDS}
b0={vs:bss(PX[vs],vs) for vs in FOLDS}
log(f'\n기준(다중축소 포함)  avg={np.mean(list(b0.values())):7.1f}  24:{b0[2024]:7.1f} 22:{b0[2022]:7.1f}')

lg_=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))

# ── (a) 체리피킹 없이 전체 NN 평균 ──
log('\n=== (a) NN 전체 평균과 블렌딩 (체리피킹 없음) ===')
tags=[]
for f in sorted(glob.glob(D+'results6/s*_c*_2024.npy')):
    t=os.path.basename(f).replace('_2024.npy','')
    if os.path.exists(D+f'results6/{t}_2022.npy'): tags.append(t)
log(f'  사용 가능한 NN 예측 {len(tags)}개')
if tags:
    NN={vs:np.mean([lg_(np.load(D+f'results6/{t}_{vs}.npy')) for t in tags],0) for vs in FOLDS}
    for w in [0.05,0.1,0.15,0.2,0.3]:
        sc={vs:bss(sp.expit((1-w)*lg_(PX[vs])+w*NN[vs]),vs) for vs in FOLDS}
        log(f'  w={w:.2f}  avg={np.mean(list(sc.values())):7.1f}  24:{sc[2024]:7.1f} 22:{sc[2022]:7.1f}')

# ── (b) 폴드2022 로 파트너 선택 -> 폴드2024 로 평가 ──
log('\n=== (b) 정직한 선택: 2022 로 고르고 2024 로 평가 ===')
if tags:
    gains=[]
    for t in tags:
        n22=lg_(np.load(D+f'results6/{t}_2022.npy'))
        g=bss(sp.expit(0.9*lg_(PX[2022])+0.1*n22),2022)-b0[2022]
        gains.append((g,t))
    gains.sort(reverse=True)
    for topn in [1,3,5]:
        sel=[t for _,t in gains[:topn]]
        n24=np.mean([lg_(np.load(D+f'results6/{t}_2024.npy')) for t in sel],0)
        for w in [0.1,0.2]:
            s=bss(sp.expit((1-w)*lg_(PX[2024])+w*n24),2024)
            log(f'  top{topn} w={w:.1f}: 2024 = {s:7.1f}  (기준 {b0[2024]:7.1f}, 순이득 {s-b0[2024]:+6.1f})')

# ── (c) NN 없이 XGB 다양성만으로 ──
log('\n=== (c) XGB 다양성 블렌딩 (torch 불필요) ===')
VAR=[('깊은얕은',dict(BASE,max_depth=3,n_estimators=1500,learning_rate=0.02)),
     ('큰L2',    dict(BASE,reg_lambda=1000.,min_child_weight=6000)),
     ('피처절반', dict(BASE,colsample_bytree=0.2)),
     ('선형부스터',dict(booster='gblinear',n_estimators=300,learning_rate=0.1,
                     reg_lambda=10.,device='cuda:1',eval_metric='logloss',verbosity=0)),
     ('Brier직접',dict(BASE,objective='reg:squarederror'))]
POOL={vs:[lg_(PX[vs])] for vs in FOLDS}
for nm,prm in VAR:
    try:
        pv={}
        for vs in FOLDS:
            tr=(season<vs)&~(isF&(season<=2022)&(vs>=2023)); va=(season==vs)&~isF
            if prm.get('objective')=='reg:squarederror':
                m=xgb.XGBRegressor(**{k:v for k,v in prm.items() if k!='eval_metric'}, random_state=0).fit(XK.loc[tr],y[tr])
                pv[vs]=np.clip(m.predict(XK.loc[va]),.02,.98)
            else:
                pv[vs]=fit(XK,vs,prm,seeds=2)
        s={vs:bss(pv[vs],vs) for vs in FOLDS}
        log(f'  [{nm:9s}] 단독 avg={np.mean(list(s.values())):7.1f}  24:{s[2024]:7.1f} 22:{s[2022]:7.1f}')
        for vs in FOLDS: POOL[vs].append(lg_(pv[vs]))
    except Exception as e: log(f'  [{nm}] 실패 {type(e).__name__}: {e}')
for n in range(2,len(POOL[2024])+1):
    sc={vs:bss(sp.expit(np.mean(POOL[vs][:n],0)),vs) for vs in FOLDS}
    log(f'  XGB {n}종 평균: avg={np.mean(list(sc.values())):7.1f}  24:{sc[2024]:7.1f} 22:{sc[2022]:7.1f}')
log('\n완료')
