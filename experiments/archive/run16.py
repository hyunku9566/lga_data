"""
16차 — 트랙맨 6차. 진단(diag15)이 가리킨 세 방향을 전부 구현

진단 근거
  ① 경험 구간: 잔차 상관이 3k+ 에서 ±0.003, 300-1k 에서 0.020~0.027
     -> 트랙맨은 '결과 이력이 얇을 때만' 값어치. 경험과의 상호작용으로 표현
  ② 상황 상호작용: trait 3분위 효과가 0볼 +0.021 -> 3볼 +0.037 로 단조 증가
     -> trait x 카운트 교차항
  ③ 등판 내 구속 저하: 분할반분 r=0.42~0.56 (압박성향 0.19, 결과체력 0.01 대비 강함)
     -> 형질로 확정. 추론 시 '이닝x역할'(등판 내 진행도 추정)과 곱해 시변화

G1 경험 상호작용  : trait x log(1+asof_n), trait/(경험) 등
G2 상황 상호작용  : trait x balls, trait x (balls>=3), trait x li
G3 등판내 붕괴    : 구속저하·릴리스이동 형질 + 등판진행도 추정치와의 곱
"""
import os, json, time, warnings, traceback
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb, lightgbm as lgb
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results16/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'

RAW=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig')
X98=pd.read_parquet(D+'X98.parquet'); TM=pd.read_parquet(D+'results14/tm5.parquet')
y=X98.__y.values; season=X98.__season.values; isF=X98.__F.values.astype(bool)
CORE=[c for c in X98.columns if not c.startswith('__')]
TMSEL=json.load(open(D+'v6_tmsel.json'))
def multi_k():
    F={}
    for idc,nc,rc,pf in [('pitcher_id','asof_pitcher_n','asof_pitcher_success_rate','p_succ'),
                         ('batter_id','asof_batter_n','asof_batter_success_rate','b_succ')]:
        t=RAW[[idc,'season',nc,rc]].copy(); t['succ']=t[nc]*t[rc].fillna(0)
        S=t.loc[t.groupby([idc,'season'])[nc].idxmin()].set_index([idc,'season'])[[nc,'succ']]
        a=RAW[[idc,'season']].join(S,on=[idc,'season'])
        dn=np.maximum(RAW[nc].values-a[nc].fillna(0).values,0)
        ds=np.maximum(np.nan_to_num(RAW[nc].values*RAW[rc].values)-a['succ'].fillna(0).values,0)
        lgv=np.nanmean(RAW[rc])
        for k in [25,75,400,1000]: F[f'{pf}_k{k}']=(ds+k*lgv)/(dn+k)
    return pd.DataFrame(F,index=RAW.index).astype(np.float32)
BASE=pd.concat([X98[CORE],multi_k(),TM[TMSEL]],axis=1)   # = v6 (114)
log(f'{el()} v6 기준 {BASE.shape[1]}피처')

# ═══════ G3 재료: 등판 내 붕괴 형질 (as-of 시즌별) ═══════
J=pd.read_parquet(D+'aligned.parquet')
JF=J[J.pitch_type_group=='fastball'].copy()
JF['bin']=pd.cut(JF.pitch_of_app,[-1,15,45,300],labels=['e','m','l'])
def decay(x):
    g=x.groupby('bin')[['rel_height','rel_side','rel_speed']].mean()
    if len(g)<3 or g.isna().any().any(): return pd.Series({'dc_relmove':np.nan,'dc_velo':np.nan,'w':len(x)})
    a=g.loc['e',['rel_height','rel_side']].values; b=g.loc['l',['rel_height','rel_side']].values
    return pd.Series({'dc_relmove':float(np.linalg.norm(b-a)),
                      'dc_velo':float(g.loc['l','rel_speed']-g.loc['e','rel_speed']),'w':len(x)})
DC={c:np.full(len(RAW),np.nan,np.float32) for c in ['dc_relmove','dc_velo']}
for s in range(2020,2026):
    tgt=season==s; hist=JF[JF.season<s]
    if not tgt.any() or not len(hist): continue
    tb=hist.groupby('pitcher_id').apply(decay)
    tb=tb[tb.w>=300]
    pid=pd.Series(RAW.pitcher_id.values[tgt])
    for c in ['dc_relmove','dc_velo']: DC[c][tgt]=pid.map(tb[c]).values
DCF=pd.DataFrame(DC,index=RAW.index).astype(np.float32)
log(f'{el()} G3 붕괴형질 생성, 결측 {DCF.isna().mean().mean():.3f}')

# ═══════ 세 그룹 피처 ═══════
n_exp=RAW.asof_pitcher_n.values.astype(np.float32)
logn=np.log1p(n_exp)
balls=RAW.balls_before.values.astype(np.float32)
li=RAW.li.values.astype(np.float32)
inn=RAW.inning.values.astype(np.float32)
ppa=X98['p_ppa'].values.astype(np.float32)        # 등판당 투구수 = 역할
KEY=['v_vaa_sd','a_anom_mean','m2_press_shift','m2_press_velo','v_vaa','v_haa_sd']

G1={}   # 경험 상호작용
for c in KEY:
    v=TM[c].values.astype(np.float32)
    G1[f'{c}_x_logn']=v*logn
    G1[f'{c}_x_thin']=v*np.exp(-n_exp/1000.)      # 경험 얇을수록 가중 (300-1k 구간 강조)
G1=pd.DataFrame(G1,index=RAW.index).astype(np.float32)

G2={}   # 상황 상호작용
for c in KEY:
    v=TM[c].values.astype(np.float32)
    G2[f'{c}_x_balls']=v*balls
    G2[f'{c}_x_3ball']=v*(balls>=3)
    G2[f'{c}_x_li']=v*li
G2=pd.DataFrame(G2,index=RAW.index).astype(np.float32)

G3={}   # 등판내 붕괴 x 진행도
prog=inn*np.log1p(ppa)                             # 등판 내 진행도 추정 (이닝 x 역할)
G3['dc_relmove']=DCF.dc_relmove.values
G3['dc_velo']=DCF.dc_velo.values
G3['dc_relmove_x_prog']=DCF.dc_relmove.values*prog
G3['dc_velo_x_prog']=DCF.dc_velo.values*prog
G3['dc_velo_x_inn']=DCF.dc_velo.values*inn
G3['prog']=prog
G3=pd.DataFrame(G3,index=RAW.index).astype(np.float32)
log(f'{el()} G1 {G1.shape[1]} / G2 {G2.shape[1]} / G3 {G3.shape[1]}개')

# ═══════ 평가 ═══════
FOLDS=[2024,2022]; NSEED=5
XP=dict(n_estimators=600,learning_rate=0.008,max_depth=6,min_child_weight=1500,subsample=0.7,
        colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,tree_method='hist',device='cuda:0',
        eval_metric='logloss',verbosity=0)
LP=dict(n_estimators=1200,learning_rate=0.01,num_leaves=31,min_child_samples=1500,
        subsample=0.7,subsample_freq=1,colsample_bytree=0.5,reg_lambda=50.)
def bss(p,vs):
    va=(season==vs)&~isF; yv=y[va]; r=yv.mean()
    return 100000*max(0.,1-np.mean((p-yv)**2)/(r*(1-r)))
R=[]
def bench(Xa,name,both=False):
    o={}
    for vs in FOLDS:
        tr=(season<vs)&~(isF&(season<=2022)&(vs>=2023)); va=(season==vs)&~isF
        sc=[bss(xgb.XGBClassifier(**XP,random_state=sd).fit(Xa.loc[tr],y[tr])
                .predict_proba(Xa.loc[va])[:,1],vs) for sd in range(NSEED)]
        o[vs]=(float(np.mean(sc)),float(np.std(sc)))
        if both:
            pl=np.mean([lgb.LGBMClassifier(**LP,random_state=sd,verbose=-1,n_jobs=24)
                        .fit(Xa.loc[tr],y[tr]).predict_proba(Xa.loc[va])[:,1] for sd in range(2)],0)
            np.save(OUT+f'lgb_{name}_{vs}.npy',pl.astype(np.float32))
            o[str(vs)+'L']=bss(pl,vs)
    avg=(o[2024][0]+o[2022][0])/2; se=np.hypot(o[2024][1],o[2022][1])/2
    R.append(dict(name=name,nfeat=Xa.shape[1],avg=avg,se=se,m24=o[2024][0],m22=o[2022][0]))
    pd.DataFrame(R).to_csv(OUT+'res16.csv',index=False)
    ex=f"  LGB 24:{o.get('2024L',0):7.1f} 22:{o.get('2022L',0):7.1f}" if both else ''
    log(f'{el()} {name:28s}({Xa.shape[1]:3d}) avg={avg:7.1f}±{se:4.1f}  '
        f'24:{o[2024][0]:7.1f}±{o[2024][1]:4.1f}  22:{o[2022][0]:7.1f}±{o[2022][1]:4.1f}{ex}')
    return avg

log(f'\n{el()} ===== 벤치마크 (시드 {NSEED}) =====')
b0=bench(BASE,'v6 기준')
gains={}
for nm,G in [('G1 경험상호작용',G1),('G2 상황상호작용',G2),('G3 등판내붕괴',G3)]:
    gains[nm]=bench(pd.concat([BASE,G],axis=1),nm)-b0
    log(f'{el()}    -> 순효과 {gains[nm]:+6.1f}')
log(f'\n{el()} ===== 조합 =====')
bench(pd.concat([BASE,G1,G2],axis=1),'G1+G2')
bench(pd.concat([BASE,G1,G3],axis=1),'G1+G3')
bench(pd.concat([BASE,G2,G3],axis=1),'G2+G3')
allg=bench(pd.concat([BASE,G1,G2,G3],axis=1),'G1+G2+G3 전체',both=True)
pos=[G for nm,G in [('G1 경험상호작용',G1),('G2 상황상호작용',G2),('G3 등판내붕괴',G3)] if gains.get(nm,0)>0]
if pos and len(pos)<3:
    bench(pd.concat([BASE]+pos,axis=1),f'양수그룹만({len(pos)})',both=True)
log(f'\n{el()} ===== 완료 =====')
