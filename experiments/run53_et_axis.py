"""
53차 — ExtraTrees 를 5번째 블렌드 축으로

배경
  20일 가중치 실측에서 XGB 실효 .385/.315/.245 -> 1070.18/1071.89/1073.39 로,
  개별 성능이 가장 좋은 축(XGB 872.9)의 비중을 줄일수록 점수가 올랐다.
  즉 이 앙상블은 개별 정확도보다 축 간 다양성에 값을 매긴다.

  그리고 49차에서 LGB 의 extra_trees(분할점을 무작위로) 가 단일 최대 이득
  (폴드2024 +33.1 / 폴드2023 +23.4) 을 냈다. 무작위성이 이 문제에 잘 맞는다.

  그렇다면 완전 무작위 분할인 sklearn ExtraTrees 를 별도 축으로 넣을 가치가 있다.

측정
  A 단독 성능        폴드2024/2023 BSS
  B 탈상관도         기존 3축(XGB/LGB/CB) 예측과의 로짓 상관
  C 5번째 축 효과    캐시된 3축에 ET 를 더해 가중치를 훑으며 블렌드 이득 측정

기존 3축 예측은 46차 캐시(results46/ax_*.npz, v16 피처 기준)를 재사용한다.
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, scipy.special as sp
from sklearn.ensemble import ExtraTreesClassifier
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga')
import lib_lga as L

OUT='/home/lee/lga/results53/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'
lgt=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))
NTH=int(os.environ.get('LGA_THREADS','36'))

b=L.load_base(); R=b['RAW']; y=b['y']
X0=L.build_v7(b=b).astype(np.float32)
B3=pd.read_parquet('/home/lee/lga/results45/B3.parquet')
X=pd.concat([X0,B3],axis=1)
Xf=X.fillna(-999.).values.astype(np.float32)     # sklearn 은 NaN 처리 없음
log(f'{el()} v16 {X.shape[1]}피처 · 스레드 {NTH}')

ETP=dict(n_estimators=400, max_features=0.15, min_samples_leaf=6000,
         bootstrap=False, n_jobs=NTH, random_state=0)

W3=dict(xgb=.35, lgb=.45, cb=.20); WN=0.30      # v16wB (현재 최고 1073.39)
res={}
for vs in (2024,2023):
    ctx=L.get_ctx(vs,2.0); yv,bq=ctx['yv'],ctx['base']
    d=np.load(f'/home/lee/lga/results46/ax_{vs}_b3.npz')
    px,pl,pc=d['px'],d['pl'],d['pc']

    f=OUT+f'et_{vs}.npy'
    if os.path.exists(f):
        pe=np.load(f); log(f'{el()} 폴드{vs} ET 캐시')
    else:
        t=time.time()
        m=ExtraTreesClassifier(**ETP).fit(Xf[ctx['tr']], y[ctx['tr']],
                                          sample_weight=ctx['w'])
        pe=m.predict_proba(Xf[ctx['va']])[:,1]; del m
        np.save(f,pe); log(f'{el()} 폴드{vs} ET 학습 {time.time()-t:.0f}s')

    log(f'\n{el()} ===== 폴드{vs} =====')
    log(f'          A 단독   XGB {L.bss(px,yv,bq):7.1f}  LGB {L.bss(pl,yv,bq):7.1f}  '
        f'CB {L.bss(pc,yv,bq):7.1f}  ET {L.bss(pe,yv,bq):7.1f}')
    lx,ll,lc,le=map(lgt,(px,pl,pc,pe))
    log(f'          B 로짓상관  ET-XGB {np.corrcoef(le,lx)[0,1]:.4f}  '
        f'ET-LGB {np.corrcoef(le,ll)[0,1]:.4f}  ET-CB {np.corrcoef(le,lc)[0,1]:.4f}   '
        f'(참고 XGB-LGB {np.corrcoef(lx,ll)[0,1]:.4f})')

    tree0=W3['xgb']*lx+W3['lgb']*ll+W3['cb']*lc
    zn=lx                                        # NN 대리 (두 arm 사이 고정)
    s0=L.bss(sp.expit((1-WN)*tree0+WN*zn), yv, bq)
    log(f'          C 5축 효과  기준(ET 없음) {s0:7.2f}')
    best=(0.,s0)
    for we in (0.05,0.10,0.15,0.20,0.30):
        w={k:v*(1-we) for k,v in W3.items()}
        tree=w['xgb']*lx+w['lgb']*ll+w['cb']*lc+we*le
        s=L.bss(sp.expit((1-WN)*tree+WN*zn), yv, bq)
        log(f'                      ET 비중 {we:.2f}  {s:7.2f}  ({s-s0:+6.2f})')
        if s>best[1]: best=(we,s)
    res[vs]=dict(base=s0, best_w=best[0], best=best[1], gain=best[1]-s0)

log(f'\n{el()} ===== 요약 =====')
for vs in (2024,2023):
    r=res[vs]
    log(f'          폴드{vs}  최적 ET 비중 {r["best_w"]:.2f}  이득 {r["gain"]:+6.2f}')
both = res[2024]['gain']>0 and res[2023]['gain']>0
log(f'          두 폴드 동시 개선: {both}')
json.dump({str(k):v for k,v in res.items()}, open(OUT+'summary.json','w'), indent=1)
log(f'{el()} 완료')
