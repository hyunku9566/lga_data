"""
56차 — 새 LGB/CB 설정의 블렌드 수준 검증

49~55차에서 축 단독 이득이 나왔다.
    LGB  leaves15+extra_trees+colsample0.4   현행 대비 24 +35.1 / 23 +28.1
    CB   d4+bagtemp2+l2_500+n4000/lr.015     현행 대비 24 +105.9 / 23 +159.4
그러나 축 단독 이득이 4축 블렌드로 얼마나 넘어오는지는 별개다.
pc_* 는 XGB 단독 폴드2023 +3.59 였는데 블렌드에서 +1.08 로 소멸했고 LB 에서 졌다.

측정
  A 블렌드 이득   구설정 3축 vs 신설정 3축 (XGB 는 동일, 46차 캐시 재사용)
  B 가중치 지형   신설정에서 xgb/lgb/cb 비중을 훑어 CV 최적점 위치 확인
                  (가중치는 LB 로 정하지만, 최적점이 어느 방향으로 이동했는지는 참고)
"""
import os, sys, time, json, warnings, itertools
import numpy as np, pandas as pd, scipy.special as sp
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga')
import lib_lga as L

OUT='/home/lee/lga/results56/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'
lgt=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))
DEV=os.environ.get('LGA_DEV','cuda:0').split(':')[-1]

b=L.load_base(); R=b['RAW']; y=b['y']
X0=L.build_v7(b=b).astype(np.float32)
B3=pd.read_parquet('/home/lee/lga/results45/B3.parquet')
X=pd.concat([X0,B3],axis=1)
CATF=[c for c in ['pitcher_id','batter_id','pitcher_team_id','batter_team_id',
                  'pitcher_hand','batter_hand'] if c in X.columns]
Z=X.copy()
for c in CATF: Z[c]=Z[c].fillna(-1).astype(np.int64)

LP_NEW=dict(n_estimators=1200, learning_rate=0.01, num_leaves=15, min_child_samples=6000,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.4, reg_lambda=50.,
            extra_trees=True, verbose=-1, n_jobs=36)
CB_NEW=dict(iterations=4000, learning_rate=0.015, depth=4, l2_leaf_reg=500.,
            bagging_temperature=2., loss_function='Logloss', random_seed=0, verbose=0,
            task_type='GPU', devices=DEV)

WN=0.30
def blend(lx,ll,lc,w,zn):
    t=w[0]*lx+w[1]*ll+w[2]*lc
    return sp.expit((1-WN)*t/sum(w)+WN*zn)

for vs in (2024,2023):
    ctx=L.get_ctx(vs,2.0); yv,bq=ctx['yv'],ctx['base']
    old=np.load(f'/home/lee/lga/results46/ax_{vs}_b3.npz')
    px,pl_o,pc_o=old['px'],old['pl'],old['pc']

    f1=OUT+f'lgb_{vs}.npy'
    if os.path.exists(f1): pl_n=np.load(f1)
    else:
        t=time.time()
        pl_n=np.mean([lgb.LGBMClassifier(**LP_NEW,random_state=s)
                      .fit(X[ctx['tr']],y[ctx['tr']],sample_weight=ctx['w'])
                      .predict_proba(X[ctx['va']])[:,1] for s in range(3)],0)
        np.save(f1,pl_n); log(f'{el()} 폴드{vs} 신 LGB {time.time()-t:.0f}s')
    f2=OUT+f'cb_{vs}.npy'
    if os.path.exists(f2): pc_n=np.load(f2)
    else:
        t=time.time()
        m=CatBoostClassifier(**CB_NEW).fit(Pool(Z[ctx['tr']],y[ctx['tr']],
                             weight=ctx['w'],cat_features=CATF))
        pc_n=m.predict_proba(Z[ctx['va']])[:,1]; del m
        np.save(f2,pc_n); log(f'{el()} 폴드{vs} 신 CB {time.time()-t:.0f}s')

    log(f'\n{el()} ===== 폴드{vs} =====')
    log(f'          축 단독   XGB {L.bss(px,yv,bq):7.1f}   '
        f'LGB {L.bss(pl_o,yv,bq):7.1f} -> {L.bss(pl_n,yv,bq):7.1f}   '
        f'CB {L.bss(pc_o,yv,bq):7.1f} -> {L.bss(pc_n,yv,bq):7.1f}')
    lx=lgt(px); zn=lx
    W=(0.35,0.45,0.20)                      # v16wB (현재 최고 1073.39)
    s_o=L.bss(blend(lx,lgt(pl_o),lgt(pc_o),W,zn),yv,bq)
    s_n=L.bss(blend(lx,lgt(pl_n),lgt(pc_n),W,zn),yv,bq)
    log(f'          A 블렌드(v16wB 가중치)  구 {s_o:7.2f} -> 신 {s_n:7.2f}   {s_n-s_o:+6.2f}')

    log(f'          B 가중치 지형 (신설정)')
    best=None
    for wx in (0.15,0.25,0.35,0.45):
        row=[]
        for wl in (0.35,0.45,0.55,0.65):
            for wc in (0.10,0.20,0.30):
                s=L.bss(blend(lx,lgt(pl_n),lgt(pc_n),(wx,wl,wc),zn),yv,bq)
                if best is None or s>best[1]: best=((wx,wl,wc),s)
            row.append(f'{max(L.bss(blend(lx,lgt(pl_n),lgt(pc_n),(wx,wl,wc),zn),yv,bq) for wc in (0.10,0.20,0.30)):7.1f}')
        log(f'            xgb {wx:.2f}  lgb별최고 {" ".join(row)}')
    log(f'            CV 최적 (xgb,lgb,cb) = {best[0]}  {best[1]:.2f}')
log(f'{el()} 완료')
