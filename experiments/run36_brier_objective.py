"""
36차 — 트리를 Brier(제곱오차)로 직접 최적화하면 나은가

발견: 우리 지표는 Brier(=MSE)인데 트리 3종은 전부 logloss 를 최적화하고 있다.
      NN 은 이미 MSE 로 학습한다(run18: ((o[:,0]-y)**2).mean()).
      즉 앙상블 주력 축만 지표와 다른 목적함수를 쓰고 있었다.

둘 다 proper scoring rule 이지만 오차 가중이 다르다. logloss 는 확신에 찬
오답을 훨씬 크게 벌한다. 예측이 전부 0.5 근처에 몰리는 극저신호 문제에서
이 차이가 다른 트리를 만들 수 있다.

채택 기준: 폴드2024/2023 동시 개선 (lib_lga.bench2 의 both)
"""
import os, sys, time, warnings, json
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga')
import lib_lga as L

OUT='/home/lee/lga/results36/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'

b=L.load_base(); y=b['y']; X=L.build_v7(b=b)
log(f'{el()} 피처 {X.shape[1]}')
DEV=os.environ.get('LGA_DEV','cuda:1')

BASE=dict(max_depth=10,min_child_weight=6000,subsample=0.7,colsample_bytree=0.5,
          reg_lambda=50.,reg_alpha=1.,tree_method='hist',device=DEV,verbosity=0,
          n_estimators=2000,learning_rate=0.005)
NS=2

def run(vs, obj, name):
    f = L.fold_ctx(vs, b=b)
    tr,va,w,yv,bq = f['tr'],f['va'],f['w'],f['yv'],f['base']
    ps=[]
    for s in range(NS):
        p=dict(BASE, random_state=s)
        if obj=='logloss':
            m=xgb.XGBClassifier(**p, objective='binary:logistic', eval_metric='logloss')
            m.fit(X[tr],y[tr],sample_weight=w); pr=m.predict_proba(X[va])[:,1]
        else:   # Brier = 0/1 타깃에 대한 제곱오차 직접 최적화
            m=xgb.XGBRegressor(**p, objective='reg:squarederror', eval_metric='rmse')
            m.fit(X[tr],y[tr],sample_weight=w); pr=np.clip(m.predict(X[va]),1e-6,1-1e-6)
        ps.append(pr)
    p=np.mean(ps,0)
    np.save(OUT+f'{name}_{vs}.npy', p.astype(np.float32))
    return L.bss(p, yv, bq), p, float(np.var(p)), float(np.mean(p))

log(f'\n{el()} ===== 목적함수 비교 (XGB, 시드{NS}) =====')
res={}
for obj,name in [('logloss','logloss'),('brier','brier')]:
    for vs in (2024,2023):
        s,_,v,mu=run(vs,obj,name); res[(obj,vs)]=s
        log(f'{el()}   {name:8s} 폴드{vs}  BSS {s:7.1f}  예측평균 {mu:.4f}  예측분산 {v:.6f}')
d24=res[('brier',2024)]-res[('logloss',2024)]
d23=res[('brier',2023)]-res[('logloss',2023)]
both='O 채택가능' if (d24>0 and d23>0) else ''
log(f'\n{el()} 순효과  2024 {d24:+6.1f}  2023 {d23:+6.1f}  {both}')

# 두 목적함수를 섞으면? (서로 다른 트리라면 앙상블 값어치가 있다)
log(f'\n{el()} ===== 두 목적함수 혼합 =====')
lg=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))
for vs in (2024,2023):
    f=L.fold_ctx(vs,b=b); va,yv,bq = f['va'],f['yv'],f['base']
    a=np.load(OUT+f'logloss_{vs}.npy'); c=np.load(OUT+f'brier_{vs}.npy')
    base=L.bss(a,yv,bq)
    ea=(yv-a)**2; ec=(yv-c)**2
    log(f'{el()}   폴드{vs} logloss 단독 {base:7.1f}')
    log(f'{el()}     예측 로짓상관 {np.corrcoef(lg(a),lg(c))[0,1]:.4f}   오차상관 {np.corrcoef(ea,ec)[0,1]:.4f}')
    log(f'{el()}     예측분산  logloss {np.var(a):.6f}  brier {np.var(c):.6f}')
    for m in (0.3,0.5,0.7):
        s=L.bss(sp.expit((1-m)*lg(a)+m*lg(c)),yv,bq)
        log(f'{el()}     brier {m:.1f} 혼합  {s:7.1f}  ({s-base:+5.1f})')
log(f'{el()} 완료')
