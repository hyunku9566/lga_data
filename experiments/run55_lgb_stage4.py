"""
55차 (LGB 4단계 — leaves15+ET 기준) — LightGBM 재탐색 (블렌드 주력 축이 된 뒤 처음)

배경
  20일 블렌드 가중치 실측에서 XGB 실효 .385/.315/.245 -> 1070.18/1071.89/1073.39 로
  XGB 를 줄일수록 단조 상승했다. 꼭짓점이 -0.193 로 구간 밖이라 우리 범위 전체에서
  LGB 가 XGB 보다 낫다는 뜻이다.

  그런데 LGB 파라미터는 26차에서 leaves31 vs leaves15 만 대충 비교하고 정한 값이다.
  XGB 는 23/27차에서 두 번 정밀 재탐색해 폴드2024 821.7 -> 869.7 (+48) 을 얻었는데
  LGB 는 그런 작업을 한 적이 없다. 주력 축에 자원을 쓴 적이 없는 셈이다.

  현행: n1200 / lr0.01 / leaves15 / mcs6000 / colsample0.5 / lambda50 / subsample0.7

주의
  lib_lga.bench2 는 XGBoost 를 쓴다. 여기서는 LGB 전용 채점기를 쓴다.
  기준선 피처는 v16 (v7 120 + pbc_* 4 = 124).

사용: python run49_lgb_retune.py A|B
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga')
import lib_lga as L

GRP=(sys.argv[1] if len(sys.argv)>1 else 'A').upper()
NTH=int(os.environ.get('LGA_THREADS','20'))
OUT='/home/lee/lga/results55/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+f'log_{GRP}.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'

b=L.load_base(); R=b['RAW']; y=b['y']
X0=L.build_v7(b=b).astype(np.float32)
B3=pd.read_parquet('/home/lee/lga/results45/B3.parquet')
X=pd.concat([X0,B3],axis=1)
log(f'{el()} [{GRP}] v16 {X.shape[1]}피처 · leaves10+extra_trees 기준 · 스레드 {NTH}')

BASE=dict(n_estimators=1200, learning_rate=0.01, num_leaves=15, min_child_samples=6000,
          subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=50., extra_trees=True,
          verbose=-1, n_jobs=NTH)

def score(prm, name, nseed=2):
    ck=OUT+f'r_{name.replace(" ","_").replace("/","-")}.json'
    if os.path.exists(ck):
        d=json.load(open(ck)); log(f'{el()}   {name:26s} 캐시  24: {d["m24"]:7.1f}  23: {d["m23"]:7.1f}'); return d
    o={}
    for vs in (2024,2023):
        ctx=L.get_ctx(vs,2.0)
        p=np.mean([lgb.LGBMClassifier(**prm,random_state=s)
                   .fit(X[ctx['tr']], y[ctx['tr']], sample_weight=ctx['w'])
                   .predict_proba(X[ctx['va']])[:,1] for s in range(nseed)],0)
        o[f'm{str(vs)[2:]}']=L.bss(p, ctx['yv'], ctx['base'])
    json.dump(o,open(ck,'w'))
    return o

GA=[('기준 leaves15+ET', {}),
    ('lambda200',     dict(reg_lambda=200.)),
    ('lambda500',     dict(reg_lambda=500.)),
    ('lambda200+n2400', dict(reg_lambda=200., n_estimators=2400, learning_rate=0.005)),
    ('n2400/lr.005',  dict(n_estimators=2400, learning_rate=0.005)),
    ('n4800/lr.0025', dict(n_estimators=4800, learning_rate=0.0025))]
GB=[('기준 leaves15+ET', {}),
    ('leaves18',      dict(num_leaves=18)),
    ('leaves12',      dict(num_leaves=12)),
    ('mcs12000',      dict(min_child_samples=12000)),
    ('mcs3000',       dict(min_child_samples=3000)),
    ('colsample.4',   dict(colsample_bytree=0.4)),
    ('colsample.6',   dict(colsample_bytree=0.6)),
    ('lambda200+mcs12000', dict(reg_lambda=200., min_child_samples=12000))]
ARMS=GA if GRP=='A' else GB

log(f'\n{el()} ===== 스윕 {GRP} ({len(ARMS)}개) =====')
base=None; rows=[]
for nm,d in ARMS:
    prm=dict(BASE); prm.update(d)
    r=score(prm,nm)
    if base is None:
        base=r; log(f'{el()}   {nm:26s} 24: {r["m24"]:7.1f}  23: {r["m23"]:7.1f}   [기준]')
    else:
        d24=r['m24']-base['m24']; d23=r['m23']-base['m23']
        tag='O 채택가능' if (d24>0 and d23>0) else ''
        log(f'{el()}   {nm:26s} 24: {r["m24"]:7.1f} ({d24:+5.1f})  23: {r["m23"]:7.1f} ({d23:+5.1f})  {tag}')
    rows.append(dict(name=nm,**r))
pd.DataFrame(rows).to_csv(OUT+f'summary_{GRP}.csv',index=False)
log(f'{el()} 완료')
