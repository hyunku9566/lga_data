"""
20차 — 구종 라벨 역산의 값어치 측정 (근본 재검토)

발견: asof_pitcher_pitchmix_n 이 증분 1의 완전 수열(결측 0%)이라
      매 투구의 구종(fastball/breaking/offspeed/other)이 정확히 역산된다.
      지금까지 이 정보는 통째로 미사용이었다.

측정 세 가지
  1) 오라클 : 진짜 구종을 피처로 (규칙상 사용 불가. 천장을 재는 용도)
  2) 합법   : 구종 분포를 예측하는 다중분류 모델의 OOF 확률을 피처로
  3) 성분+구종 : 이미 검증된 성분 스태킹과 합쳤을 때
"""
import os, json, time, warnings
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results20/'; os.makedirs(OUT,exist_ok=True)
S='/tmp/claude-1000/-home-lee-lga/97188a20-36c6-4af4-9fed-509e8b2fcd01/scratchpad/'
LOG=open(OUT+'log.txt','a',buffering=1)
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
BASE=pd.concat([X98[CORE],multi_k(),TM[TMSEL]],axis=1)           # 114 (v6)
V7=pd.concat([BASE,OF],axis=1)                                    # 120 (v7)
log(f'{el()} BASE {BASE.shape[1]} / V7 {V7.shape[1]}')

# ── 구종 라벨 역산 ──
ordr=np.lexsort((RAW.asof_pitcher_n.values, RAW.pitcher_id.values))
pid=RAW.pitcher_id.values[ordr]; mx=RAW.asof_pitcher_pitchmix_n.values[ordr].astype(np.float64)
last=np.append(pid[1:]!=pid[:-1],True)
PT={}
for c in ['fastball','breaking','offspeed']:
    cum=np.nan_to_num(mx*RAW[f'asof_pitcher_{c}_rate'].values[ordr])
    d=np.append(cum[1:]-cum[:-1],np.nan); d[last]=np.nan
    v=np.round(d); v[np.abs(d-v)>0.3]=np.nan
    o=np.full(len(RAW),np.nan,np.float32); o[ordr]=v; PT[c]=o
P=pd.DataFrame(PT)
ok=P.notna().all(1).values
# 4번째 범주: 셋 다 0 이면 other
cls=np.full(len(RAW),-1,np.int64)
cls[ok&(P.fastball.values==1)]=0
cls[ok&(P.breaking.values==1)]=1
cls[ok&(P.offspeed.values==1)]=2
cls[ok&(P.fastball.values==0)&(P.breaking.values==0)&(P.offspeed.values==0)]=3
valid=cls>=0
log(f'{el()} 구종 역산 {valid.sum():,}/{len(RAW):,}  분포 {pd.Series(cls[valid]).value_counts(normalize=True).round(4).to_dict()}')
log(f'{el()}   (0=fastball 1=breaking 2=offspeed 3=other)')
# 구종별 제구 성공률
t=pd.DataFrame({'c':cls[valid],'y':y[valid]}).groupby('c').agg(n=('y','size'),y=('y','mean'))
log(f'{el()} 구종별 제구성공률:\n{t.round(4).to_string()}')

XP=dict(n_estimators=600,learning_rate=0.008,max_depth=6,min_child_weight=1500,subsample=0.7,
        colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,tree_method='hist',device='cuda:1',
        eval_metric='logloss',verbosity=0)
lgt=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))
FOLD=2024; HL=2.0
tr=(season<FOLD)&~(isF&(season<=2022)); va=(season==FOLD)&~isF
yv=y[va]; bse=yv.mean()*(1-yv.mean())
def bss(p): return 100000*max(0.,1-np.mean((p-yv)**2)/bse)
W=(0.5**((FOLD-1-season[tr])/HL)).astype(np.float32)
def go(Xa,name,ns=3):
    p=np.mean([xgb.XGBClassifier(**XP,random_state=s).fit(Xa[tr],y[tr],sample_weight=W)
               .predict_proba(Xa[va])[:,1] for s in range(ns)],0)
    log(f'{el()} {name:34s}({Xa.shape[1]:3d}) BSS {bss(p):7.1f}')
    return bss(p)

# ── OOF 구종 예측 (합법 경로) ──
PQ=dict(XP); PQ.pop('eval_metric')
OOFP=np.full((len(RAW),3),np.nan,np.float32)
for s in range(2021,2025):
    t0=(season<s)&~(isF&(season<=2022)&(s>=2023))&valid; tg=season==s
    m=xgb.XGBClassifier(**PQ,objective='multi:softprob',num_class=3,
                        eval_metric='mlogloss',random_state=0).fit(BASE[t0],cls[t0])
    OOFP[tg]=m.predict_proba(BASE[tg])
    log(f'{el()}   구종 OOF 시즌{s} (학습 {t0.sum():,})')
PO=pd.DataFrame(lgt(np.clip(OOFP,1e-6,1-1e-6)),columns=[f'pt_{c}' for c in ['fb','br','os']],index=RAW.index)
acc=(np.argmax(OOFP[va],1)==cls[va]).mean()
log(f'{el()} 구종 예측 정확도(폴드2024) {acc:.4f}  (최빈 {pd.Series(cls[va]).value_counts(normalize=True).max():.4f})')

# 오라클 피처
ORC=pd.DataFrame({f'true_{c}':(cls==i).astype(np.float32) for i,c in enumerate(['fb','br','os'])},index=RAW.index)
ORC[~valid]=np.nan

log(f'\n{el()} ===== 폴드2024 =====')
b=go(V7,'A v7 기준')
go(pd.concat([V7,ORC],axis=1),'B +진짜구종(오라클,사용불가)')
go(pd.concat([V7,PO],axis=1),'C +구종OOF(합법)')
go(pd.concat([BASE,PO],axis=1),'D 성분빼고 구종만')
PO.to_parquet(OUT+'oof_pitchtype.parquet')
np.save(OUT+'cls.npy',cls)
log(f'{el()} 완료')
