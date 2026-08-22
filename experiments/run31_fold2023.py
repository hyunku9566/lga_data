"""
31차 — 폴드2023 보조 검증 (선택 편향 진단)

배경: 23차에서 27개 HPO 조합을 폴드2024 하나로 고르고 같은 폴드로 "+41.5" 라고 보고했다.
      실제 LB 이득은 +2.99 (전이율 0.06배). 자기 채점이었다.

이번엔 폴드2023 을 제3의 심판으로 세운다.
  폴드2023 = 학습 (season<2023, F경기 제외) / 검증 (season==2023, F 제외)
  최근성 가중 기준연도 = vs-1 = 2022 (기존 코드 규약과 동일)

성분 OOF 누수 확인 완료: exp_stack.py 의 tr=(season<s) 이므로
  - 2023 행의 OOF = 2019~2022 학습 모델 산출 (검증셋, 미래 미사용) OK
  - 2021/2022 행의 OOF = 각각 <2021 / <2022 학습 (학습셋, 미래 미사용) OK
따라서 폴드2023 평가에 그대로 써도 누수 없음.
"""
import os, json, time, warnings
import numpy as np, pandas as pd, xgboost as xgb
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results31/'; os.makedirs(OUT, exist_ok=True)
S='/tmp/claude-1000/-home-lee-lga/97188a20-36c6-4af4-9fed-509e8b2fcd01/scratchpad/'
LOG=open(OUT+'log_f2023.txt','a',buffering=1)
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

# 누수 점검: 폴드2023 학습/검증 구간의 OOF 결측 구조
for s in [2019,2020,2021,2022,2023,2024]:
    m=season==s
    log(f'      시즌{s} 성분OOF 결측률 {OF[m].isna().mean().mean():.3f}')

HL=2.0; NS=2; DEV='cuda:0'
def ctx(vs):
    tr=(season<vs)&~(isF&(season<=2022)&(vs>=2023)); va=(season==vs)&~isF
    return tr, va, (0.5**((vs-1-season[tr])/HL)).astype(np.float32), y[va].mean()*(1-y[va].mean())
CT={vs:ctx(vs) for vs in [2023,2024]}
def run(prm,vs):
    tr,va,W,b=CT[vs]
    p=np.mean([xgb.XGBClassifier(**prm,random_state=s,tree_method='hist',device=DEV,
                                 eval_metric='logloss',verbosity=0)
               .fit(V7[tr],y[tr],sample_weight=W).predict_proba(V7[va])[:,1] for s in range(NS)],0)
    return 100000*max(0.,1-np.mean((p-y[va])**2)/b)

def P(d,mcw,n,lr):
    return dict(max_depth=d,min_child_weight=mcw,n_estimators=n,learning_rate=lr,
                subsample=0.7,colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.)

CFG=[
 ('현재 v9  d6/mcw1500/n600/lr.008',   P(6,1500,600,0.008)),
 ('#1 d10/mcw6000/n600/lr.015',       P(10,6000,600,0.015)),
 ('#2 d8/mcw6000/n1000/lr.008',       P(8,6000,1000,0.008)),
 ('#3 d12/mcw6000/n1000/lr.008',      P(12,6000,1000,0.008)),
 ('#4 d10/mcw6000/n1000/lr.008',      P(10,6000,1000,0.008)),
 ('#5 d12/mcw6000/n600/lr.015',       P(12,6000,600,0.015)),
 ('mcw3000 d8/n1000/lr.008',          P(8,3000,1000,0.008)),
 ('mcw3000 d10/n1000/lr.008',         P(10,3000,1000,0.008)),
 ('v10 출시본 d10/mcw6000/n2000/lr.005',P(10,6000,2000,0.005)),
 ('27차최고 d10/mcw6000/n4000/lr.004', P(10,6000,4000,0.004)),
]
log(f'\n{el()} ===== 폴드2023 + 폴드2024 재측정 (시드{NS}) =====')
log(f'{"설정":38s} {"폴드2023":>9} {"폴드2024":>9}')
R=[]
for name,prm in CFG:
    v23=run(prm,2023); v24=run(prm,2024)
    R.append(dict(name=name,f2023=v23,f2024=v24,**prm))
    pd.DataFrame(R).to_csv(OUT+'res_f2023.csv',index=False)
    log(f'{el()} {name:38s} {v23:9.1f} {v24:9.1f}')

d=pd.DataFrame(R)
base=d.iloc[0]
d['d23']=d.f2023-base.f2023; d['d24']=d.f2024-base.f2024
log(f'\n{el()} ===== 현재 설정 대비 증분 =====')
log(f'{"설정":38s} {"Δ2023":>8} {"Δ2024":>8} {"양쪽개선":>8}')
for _,r in d.iterrows():
    ok='O' if (r.d23>0 and r.d24>0) else ('X' if r.d23<0<r.d24 else ' ')
    log(f'{r["name"]:38s} {r.d23:+8.1f} {r.d24:+8.1f} {ok:>8}')
log(f'\n{el()} 폴드2023 vs 폴드2024 순위상관 spearman {d.f2023.corr(d.f2024,method="spearman"):.4f}')
log(f'{el()} 완료')
