"""
67차 — 수축 사전확률(prior)이 낡았다: 고정상수 -> 시즌 인과 추세외삽

발견
    consts.json: p_succ_ssn = (ds + K*pri)/(dn + K),  K=150, pri=0.5352 고정
    pri 는 2019~2024 전체 평균이다. 1군 시즌율은 .5495 -> .4897 로 하강 중이고
    2025 추세 외삽은 .4763 -> **pri 가 +0.0589 낡았다.**
    사전확률 가중 w=K/(dn+K): 37.2% 행이 w>=0.30, 2024년 3월 평균 w=0.764.
    => 시즌 초 행은 주력 피처의 76% 가 5.9%p 틀린 상수다. dn 의존이라 전역 drift 로 못 고친다.

수정
    모든 행의 prior 를 '그 시즌보다 이전 시즌들만으로 추세 외삽한 값' 으로 교체.
    학습과 추론이 동일 절차가 된다 (2025 도 같은 방식으로 계산 가능 -> 행 독립).

재계산 (X98 에 박힌 값을 정확히 역산해서 prior 만 교체)
    ds = f*(dn+K) - K*pri_old  ->  f_new = (ds + K*pri_new)/(dn+K)
    투수 성분 7종은 dn 공유(asof_pitcher_n), 타자 2종은 b_succ_ssn_n 공유.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga'); import lib_lga as L
OUT='/home/lee/lga/results67/'; log,_=L.mklog(OUT)
b=L.load_base(); R=b['RAW']; y=b['y']; season=b['season']; isF=b['isF']
C=json.load(open('/home/lee/lga/submit_v16/model/consts.json')); K=C['K']; LGR=C['lg_rate']
X0=L.build_v7(b=b).astype(np.float32)
B3=pd.read_parquet('/home/lee/lga/results45/B3.parquet')
BASE=pd.concat([X0,B3],axis=1)

# 시즌별 1군 리그율 -> 각 시즌에 대해 '그 이전 시즌들만' 으로 추세 외삽
lg=pd.DataFrame(dict(s=season[~isF],y=y[~isF])).groupby('s').y.mean()
pri_by_season={}
for s_ in range(2019,2026):
    hist=lg[lg.index<s_]
    if len(hist)>=2:
        co=np.polyfit(hist.index.values,hist.values,1); pri_by_season[s_]=float(np.polyval(co,s_))
    elif len(hist)==1: pri_by_season[s_]=float(hist.iloc[0])
    else: pri_by_season[s_]=float(y.mean())
log('시즌별 인과 사전확률: '+str({k:round(v,4) for k,v in pri_by_season.items()}))
log(f'기존 고정 사전확률 p_succ={LGR["p_succ"]:.4f}')
pri_row=pd.Series(season).map(pri_by_season).values

SPEC=[('p_succ_ssn','p_succ','p_succ_ssn_n'),('p_rev_ssn','p_rev','p_succ_ssn_n'),
      ('p_mid_ssn','p_mid','p_succ_ssn_n'),('p_ball_ssn','p_ball','p_succ_ssn_n'),
      ('p_stk_ssn','p_stk','p_succ_ssn_n'),('b_succ_ssn','b_succ','b_succ_ssn_n'),
      ('b_mid_ssn','b_mid','b_succ_ssn_n')]
# 성분별 리그율도 같은 비율로 이동시킨다 (p_succ 의 상대변화를 적용)
scale=pri_row/LGR['p_succ']
NEW=BASE.copy()
for col,key,ncol in SPEC:
    if col not in BASE.columns: log(f'  !! {col} 없음'); continue
    dn=BASE[ncol].values.astype(np.float64); f=BASE[col].values.astype(np.float64)
    ds=f*(dn+K)-K*LGR[key]
    pri_new=(LGR[key]*scale) if key!='p_succ' else pri_row
    NEW[col]=((ds+K*pri_new)/(dn+K)).astype(np.float32)
    vc=col+'_vs_car'
    if vc in BASE.columns:
        NEW[vc]=(NEW[col].values-(BASE[col].values-BASE[vc].values)).astype(np.float32)
# multi_k 도 같은 prior 로 재계산
RAWt=R
for idc,nc,rc,pf in [('pitcher_id','asof_pitcher_n','asof_pitcher_success_rate','p_succ'),
                     ('batter_id','asof_batter_n','asof_batter_success_rate','b_succ')]:
    t=RAWt[[idc,'season',nc,rc]].copy(); t['succ']=t[nc]*t[rc].fillna(0)
    S=t.loc[t.groupby([idc,'season'])[nc].idxmin()].set_index([idc,'season'])[[nc,'succ']]
    a=RAWt[[idc,'season']].join(S,on=[idc,'season'])
    dn=np.maximum(RAWt[nc].values-a[nc].fillna(0).values,0)
    ds=np.maximum(np.nan_to_num(RAWt[nc].values*RAWt[rc].values)-a['succ'].fillna(0).values,0)
    pnew=pri_row if pf=='p_succ' else LGR['b_succ']*scale
    for k in [25,75,400,1000]:
        NEW[f'{pf}_k{k}']=((ds+k*pnew)/(dn+k)).astype(np.float32)
chg=[c for c in NEW.columns if not NEW[c].equals(BASE[c])]
log(f'변경된 컬럼 {len(chg)}개: {chg}')
r0=L.bench2(BASE, name='현행 (고정 사전확률)', log=log)
r1=L.bench2(NEW,  name='시즌 인과 사전확률', baseline=(r0['m24'],r0['m23']), log=log)
pd.DataFrame([r0,r1]).to_csv(OUT+'res.csv',index=False)
