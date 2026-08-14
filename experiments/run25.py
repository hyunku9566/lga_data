"""
25차 — ABS regime 대응 (가장 중요한 구조 문제)

전제: 2024 시즌부터 ABS(자동 볼판정) 도입. 학습데이터 6시즌 중
      2025 와 같은 regime 은 2024 한 시즌뿐이다.

문제: 이걸 검증할 폴드가 없다. 폴드2024 는 학습이 전부 non-ABS(<=2023)이고,
      실제 제출은 학습에 ABS(2024)가 포함된다. 상황이 다르다.

대리 실험: 2024 를 시간으로 쪼갠다.
      학습 = ... + 2024 전반(3~6월),  검증 = 2024 후반(7~10월)
      이러면 '같은 regime 데이터를 일부 갖고 같은 regime 을 예측' 하는
      실제 제출과 같은 구조가 된다.

  T1 non-ABS 만 (2019-2023)          — 데이터 많지만 regime 다름
  T2 ABS 만 (2024 전반)               — 데이터 적지만 regime 같음
  T3 전부 균등
  T4~ 반감기별 가중 (hl 0.25/0.5/1/2)
  T8 regime 지시자 피처 추가

이 비교로 '같은 regime 소량' vs '다른 regime 대량' 의 교환비를 잰다.
"""
import os, json, time, warnings
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results25/'; os.makedirs(OUT,exist_ok=True)
S='/tmp/claude-1000/-home-lee-lga/97188a20-36c6-4af4-9fed-509e8b2fcd01/scratchpad/'
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'

RAW=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig')
X98=pd.read_parquet(D+'X98.parquet'); TM=pd.read_parquet(D+'results14/tm5.parquet')
y=X98.__y.values.astype(np.float32); season=X98.__season.values; isF=X98.__F.values.astype(bool)
month=RAW.game_month.values
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

# ── regime 불연속 확인 ──
log(f'{el()} 월별 제구성공률 (정규경기)')
reg=~isF
t=pd.DataFrame({'s':season[reg],'m':month[reg],'y':y[reg]})
pv=t.groupby(['s','m']).y.mean().unstack().round(4)
log(pv.to_string())

# ── 대리 실험 설정 ──
EARLY=(season==2024)&(month<=6)&~isF
LATE =(season==2024)&(month>=7)&~isF
OLD  =(season<2024)&~(isF&(season<=2022))
log(f'\n{el()} 2019-23 {OLD.sum():,} | 2024전반 {EARLY.sum():,} | 2024후반(검증) {LATE.sum():,}')
log(f'{el()} 제구성공률  2019-23 {y[OLD].mean():.4f} | 2024전반 {y[EARLY].mean():.4f} | 2024후반 {y[LATE].mean():.4f}')

XP=dict(n_estimators=1000,learning_rate=0.008,max_depth=8,min_child_weight=6000,subsample=0.7,
        colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,tree_method='hist',device='cuda:0',
        eval_metric='logloss',verbosity=0)   # 23차 폴드2024 최적
yv=y[LATE]; bq=yv.mean()*(1-yv.mean()); NS=2
def bss(p): return 100000*max(0.,1-np.mean((p-yv)**2)/bq)
def go(trm,w,name,Xa=None):
    Xa=V7 if Xa is None else Xa
    p=np.mean([xgb.XGBClassifier(**XP,random_state=s).fit(Xa[trm],y[trm],sample_weight=w)
               .predict_proba(Xa[LATE])[:,1] for s in range(NS)],0)
    bias=p.mean()-yv.mean()
    log(f'{el()} {name:34s} 학습{trm.sum():>9,} BSS {bss(p):7.1f} 편향 {bias:+.5f}')
    return bss(p)

log(f'\n{el()} ===== 대리 실험: 2024전반 -> 2024후반 =====')
go(OLD,None,'T1 non-ABS 만 (2019-23)')
go(EARLY,None,'T2 ABS 만 (2024 전반)')
BOTH=OLD|EARLY
go(BOTH,None,'T3 전부 균등')
for hl in [0.25,0.5,1.0,2.0]:
    w=(0.5**((2024-season[BOTH])/hl)).astype(np.float32)
    go(BOTH,w,f'T4 반감기 {hl}')
# ABS 지시자 + 배수 가중
for mult in [3,10]:
    w=np.where(season[BOTH]==2024,float(mult),1.).astype(np.float32)
    go(BOTH,w,f'T5 2024 만 x{mult}')
VA=V7.copy(); VA['is_abs']=(season>=2024).astype(np.float32)
go(BOTH,None,'T6 regime 지시자 피처',Xa=VA)
w=(0.5**((2024-season[BOTH])/0.5)).astype(np.float32)
go(BOTH,w,'T7 반감기0.5 + 지시자',Xa=VA)
log(f'{el()} 완료')
