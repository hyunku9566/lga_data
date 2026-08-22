"""
65차 — F(2군) 를 학습에서 전부 뺀다

57차는 F<=2022 만 뺐다 (폴드2024 학습셋에 F2023 잔존).
타깃/검증/test 가 전부 1군인데 학습의 11% 가 2군이고 제구율이 다르다
(2024 기준 R .4897 vs F .4579, 2022 이전은 R .50 vs F .70).
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga'); import lib_lga as L
OUT='/home/lee/lga/results65/'; log,_=L.mklog(OUT)
b=L.load_base(); R=b['RAW']; y=b['y']; season=b['season']; isF=b['isF']
X0=L.build_v7(b=b).astype(np.float32)
B3=pd.read_parquet('/home/lee/lga/results45/B3.parquet')
X=pd.concat([X0,B3],axis=1)
def ctx(vs,mode,hl=2.0):
    if mode=='현행':   tr=(season<vs)&~(isF&(season<=2022)&(vs>=2023))
    elif mode=='F전체제외': tr=(season<vs)&~isF
    va=(season==vs)&~isF
    w=(0.5**((vs-1-season[tr])/hl)).astype(np.float32)
    yv=y[va].astype(np.float64)
    return dict(tr=tr,va=va,w=w,yv=yv,base=yv.mean()*(1-yv.mean()))
res=[]
for vs in (2023,2024):
    for mode in ('현행','F전체제외'):
        c=ctx(vs,mode); p=L.fit_predict(X,y,L.XP_TUNED,c,nseed=2)
        s=L.bss(p,c['yv'],c['base'])
        log(f'  폴드{vs}  {mode:9s}  학습{int(c["tr"].sum()):8d}  BSS {s:8.1f}')
        res.append(dict(vs=vs,mode=mode,bss=s))
        np.save(OUT+f'p_{vs}_{mode}.npy',p.astype(np.float32))
r=pd.DataFrame(res); r.to_csv(OUT+'res.csv',index=False)
log('')
for vs in (2023,2024):
    a=r[(r.vs==vs)&(r['mode']=='현행')].bss.iloc[0]; c2=r[(r.vs==vs)&(r['mode']=='F전체제외')].bss.iloc[0]
    log(f'  폴드{vs}  현행 {a:8.1f} -> F전체제외 {c2:8.1f}   Δ {c2-a:+7.1f}')
