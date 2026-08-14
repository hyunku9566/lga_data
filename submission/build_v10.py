"""
v10 = v9 + 전면 재튜닝 (26/27/28차 결과 반영)

폴드2024 단독 기준(ABS regime 때문에 폴드2022 는 참고만):
  XGB       d6/mcw1500/n600/lr0.008  821.7  ->  d10/mcw6000/n2000/lr0.005  867.7
  LGB       leaves31/mcs1500         796.3  ->  leaves15/mcs6000           830.8
  CatBoost  d6/n2000/lr0.03          745.1     (재탐색 결과 현재가 최적, 재사용)
  블렌드     기존 가중치는 XGB 단독보다 나빴다(859.4 < 869.7). 재최적화 -> 874.6

피처는 v9 그대로 120개. (G1/G2 는 튜닝 후 +1.9 뿐이라 추론 구현 부담 대비 기각)
CatBoost/성분모델/NN 은 v9 산출물을 그대로 재사용하고 XGB/LGB 만 재학습한다.
"""
import os, sys, json, shutil, zipfile, time, warnings
import numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; SRC=D+'submit_v9/'; B=D+'submit_v10/'; M=B+'model/'
S='/tmp/claude-1000/-home-lee-lga/97188a20-36c6-4af4-9fed-509e8b2fcd01/scratchpad/'
NSEED_X=int(sys.argv[1]) if len(sys.argv)>1 else 12
NSEED_L=int(sys.argv[2]) if len(sys.argv)>2 else 7
HL=2.0
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'

shutil.rmtree(B, ignore_errors=True); shutil.copytree(SRC, B)
for sub in ('data','output'): shutil.rmtree(B+sub, ignore_errors=True)
for f in os.listdir(M):
    if f.startswith(('xgb_','lgb_')): os.remove(M+f)

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
XK=pd.concat([X98[CORE],multi_k(),TM[TMSEL],OF],axis=1)
feat=json.load(open(M+'feat_xgb.json'))
assert list(XK.columns)==feat, f'피처 불일치 {len(XK.columns)} vs {len(feat)}'
TR=~(isF&(season<=2022))
W=(0.5**((2024-season[TR])/HL)).astype(np.float32)
print(f'{el()} 피처 {len(feat)} / 학습 {TR.sum():,}행')

XP=dict(n_estimators=2000,learning_rate=0.005,max_depth=10,min_child_weight=6000,subsample=0.7,
        colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,tree_method='hist',device='cuda:0',
        eval_metric='logloss',verbosity=0)
print(f'{el()} XGB {NSEED_X}시드 (d10/mcw6000/n2000/lr0.005)')
for sd in range(NSEED_X):
    xgb.XGBClassifier(**XP,random_state=sd).fit(XK[TR],y[TR],sample_weight=W)\
       .get_booster().save_model(M+f'xgb_{sd}.json')
LP=dict(n_estimators=1200,learning_rate=0.01,num_leaves=15,min_child_samples=6000,
        subsample=0.7,subsample_freq=1,colsample_bytree=0.5,reg_lambda=50.)
print(f'{el()} LGB {NSEED_L}시드 (leaves15/mcs6000)')
for sd in range(NSEED_L):
    lgb.LGBMClassifier(**LP,random_state=sd,verbose=-1,n_jobs=36)\
       .fit(XK[TR],y[TR],sample_weight=W).booster_.save_model(M+f'lgb_{sd}.txt')

c=json.load(open(M+'consts.json'))
c['blend3']={'xgb':0.70,'lgb':0.10,'cb':0.10,'nn':0.10}
c['mt_alpha']=0.3
json.dump(c,open(M+'consts.json','w'))
xs=sum(os.path.getsize(M+f) for f in os.listdir(M) if f.startswith('xgb_'))/2**20
ls=sum(os.path.getsize(M+f) for f in os.listdir(M) if f.startswith('lgb_'))/2**20
print(f'{el()} XGB {xs:.1f}MB / LGB {ls:.1f}MB / blend={c["blend3"]} alpha={c["mt_alpha"]} drift={c["drift"]}')

out=D+'submit_v10.zip'
if os.path.exists(out): os.remove(out)
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for root,_,fs in os.walk(B):
        for f in fs:
            fp=os.path.join(root,f); rel=os.path.relpath(fp,B)
            if rel.split(os.sep)[0] in ('data','output'): continue
            z.write(fp,rel)
print(f'{el()} {out}  {os.path.getsize(out)/2**20:.1f}MB')
