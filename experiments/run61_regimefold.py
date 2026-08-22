"""
61차 — regime 일치 폴드: 2024 를 학습에 넣고 2024 를 맞힌다

문제
    실제 제출  = 학습[2019..2024, 2024는 ABS] -> 예측[2025, ABS]
    폴드2024   = 학습[2019..2023, ABS 거의 없음] -> 예측[2024, ABS]
    두 과제는 '최근 ABS 시즌이 학습에 있는가' 에서 다르다.
    v7 이후 CV 개선 7건이 연속으로 LB 에서 실패했다(v17 -5.84 포함).
    측정 장치가 과제와 다른 것을 재고 있을 가능성을 검사한다.

새 폴드 H (regime 일치)
    학습 = 2019..2023 전부 + 2024 전반(3~6월)
    검증 = 2024 후반(7~10월)
    -> 학습에 ABS 시즌이 들어있고, 검증도 ABS. 실제 과제와 같은 모양이다.

측정
    기존 폴드에서 '실패' 판정났던 후보들을 폴드H 에서 다시 재서
    폴드H 가 LB 실측과 더 잘 맞는 판정을 내리는지 본다.
    LB 실측 정답지: pbc_* = +3.30(성공) / v17 LGB·CB재튜닝 = -5.84(실패)
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga')
import lib_lga as L

OUT='/home/lee/lga/results61/'
log,_=L.mklog(OUT)
b=L.load_base(); R=b['RAW']; y=b['y']; season=b['season']; isF=b['isF']
X0=L.build_v7(b=b).astype(np.float32)
B3=pd.read_parquet('/home/lee/lga/results45/B3.parquet')
mth=R.game_month.values

def ctxH(hl=2.0):
    tr=((season<2024)&~(isF&(season<=2022))) | ((season==2024)&~isF&(mth<=6))
    va=(season==2024)&~isF&(mth>=7)
    w=(0.5**((2024-season[tr])/hl)).astype(np.float32)
    yv=y[va].astype(np.float64)
    return dict(vs='H',tr=tr,va=va,w=w,yv=yv,base=yv.mean()*(1-yv.mean()))

c=ctxH()
log(f"폴드H  학습 {int(c['tr'].sum())}  검증 {int(c['va'].sum())}  검증 제구율 {c['yv'].mean():.4f}")

def run(Xa,name,base=None):
    t0=time.time(); p=L.fit_predict(Xa,y,L.XP_TUNED,c,nseed=2)
    s=L.bss(p,c['yv'],c['base'])
    d=f" ({s-base:+6.1f})" if base is not None else "  [기준]"
    log(f"  {name:34s}({Xa.shape[1]:3d})  H:{s:7.1f}{d}   [{time.time()-t0:4.0f}s]")
    np.save(OUT+f'p_{name.replace(" ","_").replace("/","-")}.npy',p.astype(np.float32))
    return s

# 1) pbc_* 유무 — LB 에서 +3.30 으로 성공한 변경. 폴드H 가 이걸 양수로 잡는가?
s_no  = run(X0, 'pbc 없음 (v7 120)')
s_yes = run(pd.concat([X0,B3],axis=1), 'pbc 있음 (124)', s_no)
# 2) 58/59 차에서 부호가 갈렸던 플래툰 후보를 폴드H 에서 판정
PH=pd.read_parquet(OUT+'ph.parquet') if os.path.exists(OUT+'ph.parquet') else None
log('\n판정: 폴드H 가 pbc_* 를 양수(+)로 잡으면 LB 정답지와 일치한다.')
