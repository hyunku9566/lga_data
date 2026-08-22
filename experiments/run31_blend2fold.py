"""
31차-B — 블렌드 가중치 질문을 '두 폴드 동시'로 재판정 (CPU 전용, 재학습 없음)

질문: v10 에서 블렌드를 분산(v9) -> 집중(v10)으로 바꿨더니 LB -11.66 이었다.
      이 실패를 CV 로 미리 잡을 수 있었는가?
      폴드2024 단독으로는 집중이 최적이었다(876.4). 폴드2022 도 그렇게 말했는가?

재료: results19 에 XGB/LGB/CatBoost 가 두 폴드 모두 저장돼 있다(v7 피처, 최근성 hl2).
      NN 은 results6(기존 8종) / results18(멀티태스크) 이 두 폴드 모두 보유.
      => 재학습 없이 블렌드 가중치만 바꿔가며 두 폴드에서 동시에 평가 가능.
"""
import os, json, itertools, warnings
import numpy as np, pandas as pd, scipy.special as sp
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results31/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+'log_blend.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')

R=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig',usecols=['season','game_type','control_success'])
y=R.control_success.values.astype(np.float64); season=R.season.values; isF=(R.game_type.values=='F')
lg=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))
FOLDS=[2024,2022]
sel=[n for n,_ in json.load(open(D+'v4_nn_sel.json'))]
C={}
for vs in FOLDS:
    va=(season==vs)&~isF; yv=y[va]
    C[vs]=dict(yv=yv, base=yv.mean()*(1-yv.mean()),
        X=lg(np.load(D+f'results19/xgb7_{vs}.npy')),
        G=lg(np.load(D+f'results19/lgb7_{vs}.npy')),
        B=lg(np.load(D+f'results19/cb_B_범주네이티브_{vs}.npy')),
        No=np.mean([lg(np.load(D+f'results6/{n}_{vs}.npy')) for n in sel],0),
        Nm=np.mean([lg(np.load(D+f'results18/{a}_s{s}_{vs}.npy'))
                    for a in ['L3','L5'] for s in range(4)],0))
def bss(vs,p):
    c=C[vs]; return 100000*max(0.,1-np.mean((p-c['yv'])**2)/c['base'])
def sc(vs,wx,wl,wc,wn,al):
    c=C[vs]; tw=wx+wl+wc
    tree=(wx*c['X']+wl*c['G']+wc*c['B'])/tw
    zn=(1-al)*c['No']+al*c['Nm']
    return bss(vs, sp.expit((1-wn)*tree+wn*zn))

log('=== 축별 단독 (폴드2024 / 폴드2022) ===')
for k,nm in [('X','XGB'),('G','LGB'),('B','CatBoost'),('No','NN 기존8'),('Nm','NN 멀티8')]:
    log(f'  {nm:10s} {bss(2024,sp.expit(C[2024][k])):8.1f} {bss(2022,sp.expit(C[2022][k])):8.1f}')

# v9 / v10 실제 배분 (실효 가중치 기준)
log('\n=== 실제 제출된 두 배분 ===')
CASES={
 'v9  분산 (XGB.315/LGB.245/CB.14/NN.30 a.4)': (0.45,0.35,0.20,0.30,0.4),
 'v10 집중 (XGB.70/LGB.10/CB.10/NN.10 a.3)':   (0.70,0.10,0.10,0.10,0.3),
}
log(f'{"배분":46s} {"폴드2024":>9} {"폴드2022":>9} {"평균":>9}')
for nm,(wx,wl,wc,wn,al) in CASES.items():
    a=sc(2024,wx,wl,wc,wn,al); b=sc(2022,wx,wl,wc,wn,al)
    log(f'{nm:46s} {a:9.1f} {b:9.1f} {(a+b)/2:9.1f}')

# 폴드별 최적 배분
log('\n=== 폴드별 최적 배분 탐색 ===')
GR=[]
for wx in np.arange(0.2,0.85,0.05):
 for wl in np.arange(0.0,0.45,0.05):
  for wc in np.arange(0.0,0.35,0.05):
   if abs(wx+wl+wc-1.0)>1e-6: continue
   for wn in np.arange(0.0,0.45,0.05):
    for al in [0.0,0.3,0.5,1.0]:
     GR.append((round(wx,2),round(wl,2),round(wc,2),round(wn,2),al))
log(f'  격자 {len(GR)}개')
res={vs:[(sc(vs,*g),g) for g in GR] for vs in FOLDS}
for vs in FOLDS:
    v,g=max(res[vs])
    log(f'  폴드{vs} 최적 XGB{g[0]:.2f}/LGB{g[1]:.2f}/CB{g[2]:.2f} NN{g[3]:.2f} a{g[4]} -> {v:.1f}')
    other=2022 if vs==2024 else 2024
    log(f'        이 배분을 폴드{other} 에 적용하면 {sc(other,*g):.1f}')

# 핵심 판정: XGB 비중을 올리면 각 폴드가 어떻게 반응하는가
log('\n=== XGB 집중도에 대한 두 폴드의 반응 (LGB/CB/NN 은 비례 축소) ===')
log(f'{"XGB 트리축 비중":>16} {"폴드2024":>9} {"폴드2022":>9}')
for wx in [0.40,0.50,0.5625,0.65,0.75,0.85,1.00]:
    rest=1-wx; wl=rest*0.35/0.55; wc=rest*0.20/0.55
    a=sc(2024,wx,wl,wc,0.30,0.4); b=sc(2022,wx,wl,wc,0.30,0.4)
    log(f'{wx:16.3f} {a:9.1f} {b:9.1f}')
log('\n=== NN 축 비중에 대한 두 폴드의 반응 (트리축 v9 비율 고정) ===')
log(f'{"NN 비중":>16} {"폴드2024":>9} {"폴드2022":>9}')
for wn in [0.0,0.10,0.20,0.30,0.40]:
    a=sc(2024,0.45,0.35,0.20,wn,0.4); b=sc(2022,0.45,0.35,0.20,wn,0.4)
    log(f'{wn:16.2f} {a:9.1f} {b:9.1f}')
log('\n완료')
