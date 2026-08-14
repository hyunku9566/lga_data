"""
v9 = v8 + CatBoost 축

19차 실측(폴드2024/2022, 두 폴드 모두 양수):
    CatBoost 범주 네이티브를 트리축에 q=0.20 으로 얹으면 +7.4 / +3.3

v8 의 XGB/LGB/성분모델/NN 은 전부 재사용하고 CatBoost 만 추가 학습한다.
사용법: python build_v9.py <q> <nn_weight>   예) python build_v9.py 0.20 0.30
"""
import os, sys, json, shutil, zipfile, time, warnings
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; SRC=D+'submit_v8/'; B=D+'submit_v9/'; M=B+'model/'
S='/tmp/claude-1000/-home-lee-lga/97188a20-36c6-4af4-9fed-509e8b2fcd01/scratchpad/'
Q=float(sys.argv[1]) if len(sys.argv)>1 else 0.20
WN=float(sys.argv[2]) if len(sys.argv)>2 else 0.30
HL=2.0
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'

shutil.rmtree(B, ignore_errors=True); shutil.copytree(SRC, B)
for sub in ('data','output'): shutil.rmtree(B+sub, ignore_errors=True)

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
print(f'{el()} 피처 {len(feat)} 일치 확인')

CATF=['pitcher_id','batter_id','pitcher_team_id','batter_team_id','pitcher_hand',
      'batter_hand','base_state','game_type','top_bottom']
TR=~(isF&(season<=2022))
W=(0.5**((2024-season[TR])/HL)).astype(np.float32)
Z=XK.copy()
for c in CATF: Z[c]=Z[c].fillna(-1).astype(np.int32).astype(str)
CB=dict(iterations=2000,learning_rate=0.03,depth=6,l2_leaf_reg=50.,
        loss_function='Logloss',random_seed=0,verbose=200,task_type='GPU',devices='0')
print(f'{el()} CatBoost 학습 ({TR.sum():,}행)')
m=CatBoostClassifier(**CB).fit(Pool(Z[TR],y[TR],weight=W,cat_features=CATF))
m.save_model(M+'cb.cbm')
print(f'{el()} 저장 {os.path.getsize(M+"cb.cbm")/2**20:.1f}MB')

c=json.load(open(M+'consts.json'))
# 트리축 내부 비율 xgb:lgb = 0.45:0.35 를 유지하며 CatBoost 가 q 를 차지하도록
c['blend3']={'xgb':0.45/0.8*(1-Q),'lgb':0.35/0.8*(1-Q),'cb':Q,'nn':WN}
c['cat_feat']=CATF
json.dump(c,open(M+'consts.json','w'))
shutil.copy(D+'script_v9.py', B+'script.py')
rq=open(B+'requirements.txt').read()
if 'catboost' not in rq: open(B+'requirements.txt','w').write(rq.rstrip()+'\ncatboost==1.2.10\n')
print(f'{el()} blend={c["blend3"]}  alpha={c["mt_alpha"]}  drift={c["drift"]}')

tag=f'q{int(round(Q*100)):03d}_w{int(round(WN*100)):03d}'
out=D+f'submit_v9_{tag}.zip'
if os.path.exists(out): os.remove(out)
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for root,_,fs in os.walk(B):
        for f in fs:
            fp=os.path.join(root,f); rel=os.path.relpath(fp,B)
            if rel.split(os.sep)[0] in ('data','output'): continue
            z.write(fp,rel)
print(f'{el()} {out}  {os.path.getsize(out)/2**20:.1f}MB')
