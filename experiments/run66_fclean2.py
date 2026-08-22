"""
64차 — 1군/2군 오염 보정 (주최측 asof 는 R+F 를 섞어 누적한다)

발견
    asof_pitcher_n 은 train 전체(R+F)에 대한 조밀 카운터다 -> asof 통계는 1군+2군 혼합.
    그런데 학습 타깃/검증/test 는 전부 1군(R)이다.
    혼용 투수 136명: 1군율 .4797  vs  2군율 .6117  (13.2%p 차)
    통합율 - 1군전용율 편향: 평균 +1.6%p, 2%p 초과 125명, 5%p 초과 47명
    최악 pid 22478: asof .5946 vs 실제 1군 .3777 (21.7%p)
    633/792 투수가 F 투구 보유. build_v17 의 pbc_* 도 F 필터 없음.

피처 (전부 시즌 인과 + 행 독립)
    pR_rate   1군 전용 커리어율 (직전 시즌까지, EB 수축)
    pR_logn   1군 전용 표본수
    p_fshare  커리어 중 2군 비중
    p_ctm     asof커리어율 - pR_rate   <- '주력 피처가 얼마나 오염됐나'
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga'); import lib_lga as L
OUT='/home/lee/lga/results66/'; log,_=L.mklog(OUT)
b=L.load_base(); R=b['RAW']; y=b['y']; season=b['season']; isF=b['isF']
X0=L.build_v7(b=b).astype(np.float32)
B3=pd.read_parquet('/home/lee/lga/results45/B3.parquet')
BASE=pd.concat([X0,B3],axis=1)
pid=R.pitcher_id.values
n_as=R.asof_pitcher_n.values.astype(np.float64)
r_as=np.nan_to_num(R.asof_pitcher_success_rate.values, nan=float(y.mean()))
K=200.
F={c:np.full(len(R),np.nan,np.float32) for c in ['pR_rate','pR_logn','p_fshare','p_ctm']}
for s_ in range(2020,2026):
    tgt=(season==s_); prev=(season<s_)
    if not tgt.any() or not prev.any(): continue
    Rw=prev&~isF
    mu=float(y[Rw].mean())
    gR=pd.Series(y[Rw]).groupby(pid[Rw]).agg(['sum','size'])
    gAn=pd.Series(y[prev]).groupby(pid[prev]).agg(['sum','size'])
    gF=pd.Series(isF[prev].astype(float)).groupby(pid[prev]).mean()
    pt=pd.Series(pid[tgt])
    nR=pt.map(gR['size']).fillna(0).values.astype(np.float64)
    sR=pt.map(gR['sum']).fillna(0).values.astype(np.float64)
    nA=pt.map(gAn['size']).fillna(0).values.astype(np.float64)
    sA=pt.map(gAn['sum']).fillna(0).values.astype(np.float64)
    muA=float(y[prev].mean())
    rate=(sR+K*mu)/(nR+K)
    pool_rate=(sA+K*muA)/(nA+K)
    F['pR_rate'][tgt]=rate
    F['pR_logn'][tgt]=np.log1p(nR)
    F['p_fshare'][tgt]=pt.map(gF).fillna(0.).values
    F['p_ctm'][tgt]=pool_rate-rate   # 같은 층(직전시즌까지) 끼리 비교
FC=pd.DataFrame(F,index=R.index)
log('결측률 '+str(FC.isna().mean().round(4).to_dict()))
log(f"p_ctm(수정) 평균 {np.nanmean(FC.p_ctm):+.4f}  sd {np.nanstd(FC.p_ctm):.4f}  |>0.02| {np.nanmean(np.abs(FC.p_ctm)>0.02):.3f}")

r0=L.bench2(BASE, name='기준선 124', log=log)
r1=L.bench2(pd.concat([BASE,FC],axis=1), name='+ 1군오염보정 4개', baseline=(r0['m24'],r0['m23']), log=log)
r2=L.bench2(pd.concat([BASE,FC[['p_ctm']]],axis=1), name='+ p_ctm 만', baseline=(r0['m24'],r0['m23']), log=log)
r3=L.bench2(pd.concat([BASE,FC[['pR_rate','p_ctm']]],axis=1), name='+ pR_rate/p_ctm', baseline=(r0['m24'],r0['m23']), log=log)
pd.DataFrame([r0,r1,r2,r3]).to_csv(OUT+'res.csv',index=False)
