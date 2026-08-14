"""
v7 = v6 + 세 가지 실측 레버
  L1 성분 스태킹 : asof_pitcher_*_rate x asof_pitcher_n 차분으로 매 투구의
                   reverse/middle/ball/strike 라벨을 역산(일치율 1.000000).
                   이를 보조 타깃으로 학습한 모델의 예측을 y 모델의 피처로 (+9.3)
  L2 최근성 가중 : 시즌 지수감쇠 sample_weight (+5.6)
  L3 드리프트 보정: 리그 제구율이 매년 하락(.5495->.4897)하는데 학습에 2025가 없어
                   모델이 2024 수준으로 예측한다. 로짓 시프트로 레벨만 내린다.
                   ※ 학습 데이터의 시즌 추세만 사용. 테스트 분포는 일절 보지 않음.
"""
import os, json, shutil, warnings, time
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb, lightgbm as lgb
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; B=D+'submit_v7/'; M=B+'model/'
HL=2.0          # 최근성 반감기(시즌). None 이면 미적용
DRIFT=-0.020    # 로짓 시프트. 0 이면 미적용
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'
log=print

shutil.rmtree(B, ignore_errors=True); shutil.copytree(D+'submit_v6/', B)
for f in os.listdir(M):
    if f.startswith(('xgb_','lgb_','cmp_')): os.remove(M+f)

RAW=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig')
X98=pd.read_parquet(D+'X98.parquet'); TM=pd.read_parquet(D+'results14/tm5.parquet')
y=X98.__y.values.astype(np.float32); season=X98.__season.values; isF=X98.__F.values.astype(bool)
CORE=[c for c in X98.columns if not c.startswith('__')]
TMSEL=json.load(open(D+'v6_tmsel.json'))

# ── v6 와 동일한 114 피처 ──
MULTIK=[('p_succ','pitcher_id','asof_pitcher_n','asof_pitcher_success_rate'),
        ('b_succ','batter_id','asof_batter_n','asof_batter_success_rate')]
F={}
for pref,idcol,ncol,ratecol in MULTIK:
    t=RAW[[idcol,'season',ncol,ratecol]].copy(); t['succ']=t[ncol]*t[ratecol].fillna(0)
    S=t.loc[t.groupby([idcol,'season'])[ncol].idxmin()].set_index([idcol,'season'])[[ncol,'succ']]
    a=RAW[[idcol,'season']].join(S,on=[idcol,'season'])
    dn=np.maximum(RAW[ncol].values-a[ncol].fillna(0).values,0)
    ds=np.maximum(np.nan_to_num(RAW[ncol].values*RAW[ratecol].values)-a['succ'].fillna(0).values,0)
    lgv=np.nanmean(RAW[ratecol])
    for k in [25,75,400,1000]: F[f'{pref}_k{k}']=(ds+k*lgv)/(dn+k)
BASE=pd.concat([X98[CORE],pd.DataFrame(F,index=RAW.index).astype(np.float32),TM[TMSEL]],axis=1)
BFEAT=list(BASE.columns)
log(f'{el()} 기준 피처 {len(BFEAT)}')

# ── L1: 성분 라벨 역산 ──
ordr=np.lexsort((RAW.asof_pitcher_n.values, RAW.pitcher_id.values))
pid_s=RAW.pitcher_id.values[ordr]; n_s=RAW.asof_pitcher_n.values[ordr].astype(np.float64)
last=np.append(pid_s[1:]!=pid_s[:-1],True)
COMP=['reverse','middle','ball','strike']
LAB={}
for c in COMP:
    cum=np.nan_to_num(n_s*RAW[f'asof_pitcher_{c}_rate'].values[ordr])
    d=np.append(cum[1:]-cum[:-1],np.nan); d[last]=np.nan
    v=np.round(d); v[np.abs(d-v)>0.3]=np.nan
    out=np.full(len(RAW),np.nan,np.float32); out[ordr]=v; LAB[c]=out
L=pd.DataFrame(LAB); okl=L.notna().all(1).values
chk=(L['reverse'].notna())&(season>=0)
log(f'{el()} 성분라벨 역산 {okl.sum():,}/{len(L):,} (역산불가 = 투수별 마지막 1구)')

XP=dict(n_estimators=600,learning_rate=0.008,max_depth=6,min_child_weight=1500,subsample=0.7,
        colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,tree_method='hist',device='cuda:0',
        eval_metric='logloss',verbosity=0)
LP=dict(n_estimators=1200,learning_rate=0.01,num_leaves=31,min_child_samples=1500,
        subsample=0.7,subsample_freq=1,colsample_bytree=0.5,reg_lambda=50.)
lgt=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))

def cmp_frame(d):
    """성분 로짓 4개 + 파생 2개. 학습/추론 동일 규약."""
    o=pd.DataFrame({f'cmp_{c}':d[c] for c in COMP})
    o['cmp_bad']=o.cmp_reverse+o.cmp_middle
    o['cmp_zone']=o.cmp_strike-o.cmp_ball
    return o.astype(np.float32)

# 학습행용 OOF (시즌 s 는 <s 로 학습). 2019-20 은 결측
OOF={c:np.full(len(RAW),np.nan,np.float32) for c in COMP}
for s in range(2021,2025):
    tr=(season<s)&~(isF&(season<=2022)&(s>=2023))&okl; tg=season==s
    for c in COMP:
        m=xgb.XGBClassifier(**XP,random_state=0).fit(BASE.loc[tr],L[c].values[tr])
        OOF[c][tg]=lgt(m.predict_proba(BASE.loc[tg])[:,1])
    log(f'{el()}   성분 OOF 시즌{s} (학습 {tr.sum():,})')
OF=cmp_frame(OOF); OF.index=RAW.index
log(f'{el()} 성분피처 결측률 {OF.isna().mean().mean():.3f}')

# 추론용 성분 모델: 전체 학습데이터로 재학습해 저장
trc=(~(isF&(season<=2022)))&okl
for c in COMP:
    m=xgb.XGBClassifier(**XP,random_state=0).fit(BASE.loc[trc],L[c].values[trc])
    m.get_booster().save_model(M+f'cmp_{c}.json')
log(f'{el()} 추론용 성분 모델 4개 저장 ({trc.sum():,}행)')

# ── 최종 학습 ──
XK=pd.concat([BASE,OF],axis=1)
feat=list(XK.columns); json.dump(feat,open(M+'feat_xgb.json','w'))
TR=~(isF&(season<=2022))
W=(0.5**((2024-season[TR])/HL)).astype(np.float32) if HL else None
log(f'{el()} 최종 피처 {len(feat)} / 학습 {TR.sum():,}행 / 가중 hl={HL}')
if W is not None:
    log('      시즌별 가중: '+' '.join(f'{s}:{0.5**((2024-s)/HL):.3f}' for s in range(2019,2025)))
log(f'{el()} XGB 15시드')
for sd in range(15):
    xgb.XGBClassifier(**XP,random_state=sd).fit(XK[TR],y[TR],sample_weight=W)\
       .get_booster().save_model(M+f'xgb_{sd}.json')
log(f'{el()} LGB 7시드')
for sd in range(7):
    lgb.LGBMClassifier(**LP,random_state=sd,verbose=-1,n_jobs=32)\
       .fit(XK[TR],y[TR],sample_weight=W).booster_.save_model(M+f'lgb_{sd}.txt')

c=json.load(open(M+'consts.json'))
c['base_feat']=BFEAT; c['comp']=COMP; c['drift']=DRIFT
json.dump(c,open(M+'consts.json','w'))
shutil.copy(D+'script_v7.py', B+'script.py')
log(f'{el()} 완료. drift={DRIFT} hl={HL}')
