"""
43차 — 왜 CV 이득이 LB 로 전이되지 않는가

발견
  lib_lga.fit_predict 는 XGBoost 단독으로 예측한다. 즉 bench2 가 내놓는
  모든 수치(run23 +41.5, run41 +5.10, run38 +4.0 ...)는 XGB 단독 이득이다.
  그런데 제출본은 4축 블렌드이고 XGB 실효 가중치는 0.315 다.
      XGB .315  LGB .245  CB .140  NN .300
  기록된 실측 전이율 0.06~0.5배가 이 희석과 같은 범위다.

측정
  A 희석계수    같은 변경을 XGB 단독 / 트리3축 / 전체블렌드(NN 대리) 로 각각 재고
                이득이 얼마나 줄어드는지 실측한다
  B 이력 분할   폴드2024 를 투수 과거이력 크기로 나눠 이득이 어디서 나오는지 본다
                (2025 는 미등장 투수 비중이 다를 수 있다)
  C 미등장 추이 2021~2024 각 시즌의 미등장 투수 비율 -> 2025 외삽

대상 변경: pc_* 정확 볼카운트 이력 8종 (run41 에서 XGB 단독 +5.10/+3.59)
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, scipy.special as sp
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier, Pool
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/lee/lga')
import lib_lga as L

OUT='/home/lee/lga/results43/'; os.makedirs(OUT, exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'
DEV=os.environ.get('LGA_DEV','cuda:0')
lgt=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))

b=L.load_base(); R=b['RAW']; y=b['y']; season=b['season']; isF=b['isF']
X0=L.build_v7(b=b).astype(np.float32)
pid=R.pitcher_id.values.astype(np.int64)
cnt=R.balls_before.values.astype(np.int64)*10+R.strikes_before.values.astype(np.int64)
key=pid*100+cnt
log(f'{el()} v7 {X0.shape[1]} 피처')

# ── run41 과 동일한 pc_* 생성 (계층 축소) ──
def exact_count(shr_list=(30,100,300,1000)):
    names=['pc_rate','pc_n','pc_logn','pc_delta']+[f'pc_k{k}' for k in shr_list]
    F={c:np.full(len(R),np.nan,np.float32) for c in names}
    for s in range(2020,2026):
        tgt=season==s; prev=season<s
        if not tgt.any() or not prev.any(): continue
        dp=pd.DataFrame({'pid':pid[prev],'cnt':cnt[prev],'key':key[prev],'y':y[prev]})
        gp=dp.groupby('pid').y.agg(['sum','size']); prior_pid=gp['sum']/gp['size']
        gl=dp.groupby('cnt').y.agg(['sum','size']); prior_cnt=gl['sum']/gl['size']
        gk=dp.groupby('key').y.agg(['sum','size'])
        mu=float(y[prev].mean())
        kt=pd.Series(key[tgt]); pt=pd.Series(pid[tgt]); ct=pd.Series(cnt[tgt])
        n =kt.map(gk['size']).fillna(0).values.astype(np.float64)
        sy=kt.map(gk['sum']).fillna(0).values.astype(np.float64)
        pp=pt.map(prior_pid).fillna(mu).values.astype(np.float64)
        pc=ct.map(prior_cnt).fillna(mu).values.astype(np.float64)
        rate=(sy+50.*pp+50.*pc)/(n+100.)
        F['pc_rate'][tgt]=rate; F['pc_n'][tgt]=n; F['pc_logn'][tgt]=np.log1p(n)
        F['pc_delta'][tgt]=rate-pp
        for k in shr_list:
            F[f'pc_k{k}'][tgt]=(sy+(k*.5)*pp+(k*.5)*pc)/(n+k)
    return pd.DataFrame(F,index=R.index).astype(np.float32)

XP=dict(n_estimators=2000,learning_rate=0.005,max_depth=10,min_child_weight=6000,
        subsample=0.7,colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,
        tree_method='hist',device=DEV,eval_metric='logloss',verbosity=0)
LP=dict(n_estimators=1200,learning_rate=0.01,num_leaves=15,min_child_samples=6000,
        subsample=0.7,subsample_freq=1,colsample_bytree=0.5,reg_lambda=50.,verbose=-1,n_jobs=24)
CB=dict(iterations=2000,learning_rate=0.03,depth=6,l2_leaf_reg=50.,loss_function='Logloss',
        random_seed=0,verbose=0,task_type='GPU',devices=DEV.split(':')[-1])
CATF=['pitcher_id','batter_id','pitcher_team_id','batter_team_id','pitcher_hand','batter_hand']

def axes(X, ctx, tag):
    """XGB / LGB / CB 세 축 예측을 각각 돌려준다."""
    ck=OUT+f'ax_{tag}.npz'
    if os.path.exists(ck):
        d=np.load(ck); log(f'{el()}   [{tag}] 캐시'); return d['px'],d['pl'],d['pc']
    tr,va,w=ctx['tr'],ctx['va'],ctx['w']
    px=np.mean([xgb.XGBClassifier(**XP,random_state=s).fit(X[tr],y[tr],sample_weight=w)
                .predict_proba(X[va])[:,1] for s in range(2)],0)
    log(f'{el()}   [{tag}] XGB 완료')
    pl=np.mean([lgb.LGBMClassifier(**LP,random_state=s).fit(X[tr],y[tr],sample_weight=w)
                .predict_proba(X[va])[:,1] for s in range(3)],0)
    log(f'{el()}   [{tag}] LGB 완료')
    Z=X.copy()
    cf=[c for c in CATF if c in Z.columns]
    for c in cf: Z[c]=Z[c].fillna(-1).astype(np.int64)
    pc=CatBoostClassifier(**CB).fit(Pool(Z[tr],y[tr],weight=w,cat_features=cf))\
        .predict_proba(Z[va])[:,1]
    log(f'{el()}   [{tag}] CB 완료')
    np.savez(ck,px=px,pl=pl,pc=pc)
    return px,pl,pc

W3=dict(xgb=.45,lgb=.35,cb=.20); WN=0.30
def blend3(px,pl,pc):
    return sp.expit(W3['xgb']*lgt(px)+W3['lgb']*lgt(pl)+W3['cb']*lgt(pc))
def blend4(px,pl,pc,zn):
    t=W3['xgb']*lgt(px)+W3['lgb']*lgt(pl)+W3['cb']*lgt(pc)
    return sp.expit((1-WN)*t+WN*zn)

E=exact_count(); XP_=pd.concat([X0,E],axis=1)
log(f'{el()} pc_* 8종 생성, 확장 피처 {XP_.shape[1]}')

log(f'\n{el()} ===== A. 희석계수 =====')
res={}
for vs in (2024,2023):
    ctx=L.get_ctx(vs,2.0); yv,base=ctx['yv'],ctx['base']
    a0=axes(X0 ,ctx,f'{vs}_base'); a1=axes(XP_,ctx,f'{vs}_pc')
    zn=lgt(a0[0])                       # NN 대리: 두 arm 사이에 고정된 축
    rows=[]
    for nm,f in [('XGB 단독',   lambda a: a[0]),
                 ('트리 3축',   lambda a: blend3(*a)),
                 ('4축(NN대리)', lambda a: blend4(*a,zn))]:
        s0=L.bss(f(a0),yv,base); s1=L.bss(f(a1),yv,base)
        rows.append((nm,s0,s1,s1-s0))
    res[vs]=rows
    log(f'{el()}  폴드{vs}')
    d0=rows[0][3]
    for nm,s0,s1,d in rows:
        log(f'          {nm:12s} 기준 {s0:7.1f} → {s1:7.1f}   이득 {d:+6.2f}   희석 {d/d0 if d0 else float("nan"):5.2f}배')
    np.savez(OUT+f'pred_{vs}.npz', y=yv, base=base,
             b_px=a0[0],b_pl=a0[1],b_pc=a0[2], p_px=a1[0],p_pl=a1[1],p_pc=a1[2])

log(f'\n{el()} ===== B. 투수 이력 분할 (폴드2024) =====')
ctx=L.get_ctx(2024,2.0); yv,base=ctx['yv'],ctx['base']; va=ctx['va']
hist=pd.Series(y[season<2024]).groupby(pid[season<2024]).size()
h=pd.Series(pid[va]).map(hist).fillna(0).values
BINS=[(-1,0,'미등장'),(0,500,'1~500'),(500,2000,'500~2k'),(2000,10**9,'2k+')]
a0=axes(X0,ctx,'2024_base'); a1=axes(XP_,ctx,'2024_pc')
p0=blend3(*a0); p1=blend3(*a1)
N=len(yv)
log(f'{el()}   구간      행수    비중    기준BSS기여   pc후    이득기여')
for lo,hi,nm in BINS:
    m=(h>lo)&(h<=hi)
    if not m.any(): continue
    c0=-100000*np.sum((p0[m]-yv[m])**2)/N/base
    c1=-100000*np.sum((p1[m]-yv[m])**2)/N/base
    log(f'          {nm:8s} {m.sum():>7,}  {m.mean()*100:5.1f}%  {c0:11.1f}  {c1:9.1f}  {c1-c0:+9.2f}')

log(f'\n{el()} ===== C. 미등장 투수 비율 추이 =====')
for s in (2021,2022,2023,2024):
    prev=set(pid[season<s]); m=season==s
    log(f'{el()}   시즌{s} (이력 {s-2019}시즌)  미등장 {np.mean([p not in prev for p in pid[m]])*100:5.2f}%')
json.dump({str(k):[list(r) for r in v] for k,v in res.items()},open(OUT+'summary.json','w'),indent=1)
log(f'{el()} 완료')
