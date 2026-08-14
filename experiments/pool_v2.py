"""
블렌드 풀 전면 재선택 (정직한 프로토콜)

문제: 현행 8개 NN 풀(v4_nn_sel.json)은 폴드 2024/2022 양쪽을 다 보고 골라졌다.
      그래서 그 두 폴드에서 성능이 부풀려져 있고, 새 모델(멀티태스크)을 그 기준선과
      비교하면 새 모델이 구조적으로 불리하다.

방법: 후보 전체(results6 75종 + results18 멀티태스크 12종)를 놓고
      '한 폴드에서만 전진선택 → 반대 폴드로 평가' 를 양방향으로 수행한다.
      기존 8개 풀도 동일한 절차로 다시 뽑아 같은 조건에서 비교한다.
"""
import os, glob, json, warnings
import numpy as np, pandas as pd, scipy.special as sp
warnings.filterwarnings('ignore')
D='/home/lee/lga/'
R=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig',usecols=['season','game_type','control_success'])
y=R.control_success.values.astype(np.float64); season=R.season.values; isF=(R.game_type.values=='F')
lg=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))
FOLDS=[2024,2022]

# ── 후보 수집 ──
cand={}
for f in glob.glob(D+'results6/*_2024.npy'):
    n=os.path.basename(f).replace('_2024.npy','')
    if os.path.exists(D+f'results6/{n}_2022.npy'): cand[('old',n)]=(D+'results6/'+n+'_%d.npy')
for a in ['L3','L1','L5']:
    for s in range(4):
        n=f'{a}_s{s}'
        if all(os.path.exists(D+f'results18/{n}_{v}.npy') for v in FOLDS):
            cand[('mt',n)]=(D+'results18/'+n+'_%d.npy')
print(f'후보 {len(cand)}종 (기존 {sum(1 for k in cand if k[0]=="old")} / 멀티태스크 {sum(1 for k in cand if k[0]=="mt")})')

Z={}; C={}
for vs in FOLDS:
    va=(season==vs)&~isF; yv=y[va]
    zx=lg(np.load(D+f'results14/xgb6_{vs}.npy')); zl=lg(np.load(D+f'results14/lgb6_{vs}.npy'))
    C[vs]=dict(yv=yv,base=yv.mean()*(1-yv.mean()),zt=(0.45*zx+0.35*zl)/0.8)
    for k,pat in cand.items(): Z[(k,vs)]=lg(np.load(pat%vs))
def bss(vs,p): c=C[vs]; return 100000*max(0.,1-np.mean((p-c['yv'])**2)/c['base'])
WS=np.arange(0.05,0.46,0.05)
def score(vs,keys,w=None):
    zn=np.mean([Z[(k,vs)] for k in keys],0)
    if w is None: return max(bss(vs,sp.expit((1-q)*C[vs]['zt']+q*zn)) for q in WS)
    return bss(vs,sp.expit((1-w)*C[vs]['zt']+w*zn))
def best_w(vs,keys):
    zn=np.mean([Z[(k,vs)] for k in keys],0)
    return max(((bss(vs,sp.expit((1-q)*C[vs]['zt']+q*zn)),q) for q in WS))[1]

def forward(pick, pool, maxn=12):
    """선택폴드에서만 전진선택"""
    chosen=[]; cur=-1e9
    while len(chosen)<maxn:
        bestk,bestv=None,cur
        for k in pool:
            if k in chosen: continue
            v=score(pick,chosen+[k])
            if v>bestv: bestk,bestv=k,v
        if bestk is None: break
        chosen.append(bestk); cur=bestv
    return chosen

ALL=list(cand); OLD=[k for k in cand if k[0]=='old']
print(f"\n{'='*74}\n정직한 프로토콜: 선택폴드에서 구성+w 결정 -> 반대폴드 평가\n{'='*74}")
rows=[]
for pick,ev in [(2024,2022),(2022,2024)]:
    for label,pool in [('기존후보만 재선택',OLD),('전체후보(멀티태스크 포함)',ALL)]:
        ch=forward(pick,pool); w=best_w(pick,ch)
        v=score(ev,ch,w)
        nmt=sum(1 for k in ch if k[0]=='mt')
        rows.append(dict(pick=pick,ev=ev,label=label,n=len(ch),n_mt=nmt,w=w,ev_score=v))
        print(f"  선택={pick} 평가={ev} | {label:24s} {len(ch):2d}개(MT {nmt}) w={w:.2f} -> {v:7.1f}")
    # 현행 8개 풀 (선택 편향이 있는 기준선)
    sel=[('old',n) for n,_ in json.load(open(D+'v4_nn_sel.json'))]
    sel=[k for k in sel if k in cand]
    w=best_w(pick,sel); v=score(ev,sel,w)
    print(f"  선택={pick} 평가={ev} | {'현행 v4 8개 풀(참고)':24s} {len(sel):2d}개      w={w:.2f} -> {v:7.1f}")
    print()
df=pd.DataFrame(rows)
print('평균 평가점수')
for label in df.label.unique():
    print(f"  {label:26s} {df[df.label==label].ev_score.mean():7.1f}")
# 최종 추천: 양 폴드에서 각각 뽑아 교집합/합집합 확인
print(f"\n{'='*74}\n최종 후보 구성\n{'='*74}")
for pick in FOLDS:
    ch=forward(pick,ALL)
    print(f"  폴드{pick} 선택: "+', '.join(f'{k[1]}' for k in ch))
    json.dump([[k[0],k[1]] for k in ch],open(D+f'pool_v2_pick{pick}.json','w'))
