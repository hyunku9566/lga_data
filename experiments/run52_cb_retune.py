"""
52차 — CatBoost 재탐색 (블렌드 3번째 축, 한 번도 제대로 튜닝 안 함)

배경
  49~51차에서 LGB 를 재탐색하니 leaves10+extra_trees 로 폴드2024 +30.4 / 폴드2023 +22.3
  이 나왔다. 방향은 일관되게 "용량을 줄이고 트리 간 다양성을 키운다" 였다.
  CatBoost 는 26차에서 d6 vs d10 만 비교하고(d10 은 붕괴) 그 뒤로 손댄 적이 없다.
  현행: iterations2000 / lr0.03 / depth6 / l2_leaf_reg50

  CB 실효 가중치는 .140 이고 GPU 에서 돈다. LGB 스윕이 CPU 를 다 쓰는 동안
  놀고 있는 GPU 를 여기에 쓴다.

기준선 피처는 v16 (v7 120 + pbc_* 4 = 124).
사용: python run52_cb_retune.py A|B
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga')
import lib_lga as L

GRP=(sys.argv[1] if len(sys.argv)>1 else 'A').upper()
DEV=os.environ.get('LGA_DEV','cuda:0').split(':')[-1]
OUT='/home/lee/lga/results52/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+f'log_{GRP}.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'

b=L.load_base(); R=b['RAW']; y=b['y']
X0=L.build_v7(b=b).astype(np.float32)
B3=pd.read_parquet('/home/lee/lga/results45/B3.parquet')
X=pd.concat([X0,B3],axis=1)
CATF=[c for c in ['pitcher_id','batter_id','pitcher_team_id','batter_team_id',
                  'pitcher_hand','batter_hand'] if c in X.columns]
Z=X.copy()
for c in CATF: Z[c]=Z[c].fillna(-1).astype(np.int64)
log(f'{el()} [{GRP}] v16 {X.shape[1]}피처 · 범주형 {len(CATF)} · GPU{DEV}')

BASE=dict(iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=50.,
          loss_function='Logloss', random_seed=0, verbose=0,
          task_type='GPU', devices=DEV)

def score(prm, name):
    ck=OUT+f'r_{name.replace(" ","_").replace("/","-").replace(".","p")}.json'
    if os.path.exists(ck):
        d=json.load(open(ck)); log(f'{el()}   {name:22s} 캐시  24: {d["m24"]:7.1f}  23: {d["m23"]:7.1f}'); return d
    o={}
    try:
        for vs in (2024,2023):
            ctx=L.get_ctx(vs,2.0)
            m=CatBoostClassifier(**prm).fit(Pool(Z[ctx['tr']],y[ctx['tr']],
                                weight=ctx['w'],cat_features=CATF))
            o[f'm{str(vs)[2:]}']=L.bss(m.predict_proba(Z[ctx['va']])[:,1], ctx['yv'], ctx['base'])
            del m
    except Exception as e:
        log(f'{el()}   {name:22s} 실패: {str(e)[:70]}'); return None
    json.dump(o,open(ck,'w'))
    return o

GA=[('기준 현행', {}),
    ('depth4',        dict(depth=4)),
    ('depth5',        dict(depth=5)),
    ('depth8',        dict(depth=8)),
    ('n4000/lr.015',  dict(iterations=4000, learning_rate=0.015)),
    ('n8000/lr.0075', dict(iterations=8000, learning_rate=0.0075)),
    ('d4+n4000',      dict(depth=4, iterations=4000, learning_rate=0.015))]
GB=[('기준 현행', {}),
    ('l2_200',        dict(l2_leaf_reg=200.)),
    ('l2_500',        dict(l2_leaf_reg=500.)),
    ('rsm.3',         dict(rsm=0.3)),
    ('rand_str5',     dict(random_strength=5.)),
    ('bagtemp2',      dict(bagging_temperature=2.)),
    ('border64',      dict(border_count=64)),
    ('l2_200+d4',     dict(l2_leaf_reg=200., depth=4))]
ARMS=GA if GRP=='A' else GB

log(f'\n{el()} ===== CatBoost 스윕 {GRP} ({len(ARMS)}개) =====')
base=None; rows=[]
for nm,d in ARMS:
    prm=dict(BASE); prm.update(d)
    r=score(prm,nm)
    if r is None: continue
    if base is None:
        base=r; log(f'{el()}   {nm:22s} 24: {r["m24"]:7.1f}  23: {r["m23"]:7.1f}   [기준]')
    else:
        d24=r['m24']-base['m24']; d23=r['m23']-base['m23']
        tag='O 채택가능' if (d24>0 and d23>0) else ''
        log(f'{el()}   {nm:22s} 24: {r["m24"]:7.1f} ({d24:+5.1f})  23: {r["m23"]:7.1f} ({d23:+5.1f})  {tag}')
    rows.append(dict(name=nm,**r))
pd.DataFrame(rows).to_csv(OUT+f'summary_{GRP}.csv',index=False)
log(f'{el()} 완료')
