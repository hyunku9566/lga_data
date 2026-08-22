"""
31차-D — 출시본 HPO 로 블렌드 재판정 + HPO x 블렌드 상호작용 검증

31차-C 결과:
  XGB 새HPO  폴드2024 866.5(+43.6) / 폴드2022 676.2(-21.4)  -> 폴드2024 과적합
  LGB 새HPO  폴드2024 823.0(+16.5) / 폴드2022 706.9(+15.8)  -> 양쪽 개선(진짜)

가설: v10 의 LB -11.66 은 '큰(과적합된) XGB 에 70% 를 몰아준' 결과다.
      옛 HPO 로는 집중해도 손해가 없었다(31차-B 에서 두 배분이 동률 781.7).
      => HPO x 블렌드 상호작용이 있는지 두 폴드로 확인한다.
"""
import os, json, itertools, warnings
import numpy as np, pandas as pd, scipy.special as sp
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results31/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+'log_blend_newhpo.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')

R=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig',usecols=['season','game_type','control_success'])
y=R.control_success.values.astype(np.float64); season=R.season.values; isF=(R.game_type.values=='F')
lg=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))
sel=[n for n,_ in json.load(open(D+'v4_nn_sel.json'))]
FOLDS=[2024,2022]
C={}
for vs in FOLDS:
    va=(season==vs)&~isF; yv=y[va]
    C[vs]=dict(yv=yv, base=yv.mean()*(1-yv.mean()),
        Xold=lg(np.load(D+f'results19/xgb7_{vs}.npy')),
        Xnew=lg(np.load(D+f'results31/xgbNEW_{vs}.npy')),
        Gold=lg(np.load(D+f'results19/lgb7_{vs}.npy')),
        Gnew=lg(np.load(D+f'results31/lgbNEW_{vs}.npy')),
        B=lg(np.load(D+f'results19/cb_B_범주네이티브_{vs}.npy')),
        No=np.mean([lg(np.load(D+f'results6/{n}_{vs}.npy')) for n in sel],0),
        Nm=np.mean([lg(np.load(D+f'results18/{a}_s{s}_{vs}.npy'))
                    for a in ['L3','L5'] for s in range(4)],0))
def bss(vs,p):
    c=C[vs]; return 100000*max(0.,1-np.mean((p-c['yv'])**2)/c['base'])
def sc(vs,wx,wl,wc,wn,al,xk='Xnew',gk='Gnew'):
    c=C[vs]; tw=wx+wl+wc
    tree=(wx*c[xk]+wl*c[gk]+wc*c['B'])/tw
    return bss(vs, sp.expit((1-wn)*tree+wn*((1-al)*c['No']+al*c['Nm'])))

log('=== HPO x 블렌드 상호작용: XGB 트리축 비중 스윕 (NN .30, a .4 고정) ===')
log(f'{"XGB 비중":>10} | {"옛HPO 2024":>10} {"옛HPO 2022":>10} | {"새HPO 2024":>10} {"새HPO 2022":>10}')
for wx in [0.40,0.45,0.55,0.65,0.78,0.90,1.00]:
    rest=1-wx; wl=rest*0.35/0.55; wc=rest*0.20/0.55
    o24=sc(2024,wx,wl,wc,0.30,0.4,'Xold','Gold'); o22=sc(2022,wx,wl,wc,0.30,0.4,'Xold','Gold')
    n24=sc(2024,wx,wl,wc,0.30,0.4); n22=sc(2022,wx,wl,wc,0.30,0.4)
    log(f'{wx:10.2f} | {o24:10.1f} {o22:10.1f} | {n24:10.1f} {n22:10.1f}')

log('\n=== 실제 제출 배분 재현 (새HPO 기준) ===')
log(f'{"배분":44s} {"2024":>9} {"2022":>9} {"평균":>9}')
for nm,(wx,wl,wc,wn,al) in {
 'v9/v10a 분산 (.45/.35/.20, NN.30 a.4)': (0.45,0.35,0.20,0.30,0.4),
 'v10 집중   (.70/.10/.10, NN.10 a.3)':   (0.70,0.10,0.10,0.10,0.3),
}.items():
    a=sc(2024,wx,wl,wc,wn,al); b=sc(2022,wx,wl,wc,wn,al)
    log(f'{nm:44s} {a:9.1f} {b:9.1f} {(a+b)/2:9.1f}')

log('\n=== 두 폴드 동시 개선(both) 을 만족하는 배분 탐색 ===')
BASE24=sc(2024,0.45,0.35,0.20,0.30,0.4); BASE22=sc(2022,0.45,0.35,0.20,0.30,0.4)
log(f'  기준(v10a 배분) 2024 {BASE24:.1f} / 2022 {BASE22:.1f}')
rows=[]
for wx in np.arange(0.25,0.85,0.05):
 for wl in np.arange(0.0,0.55,0.05):
  wc=1-wx-wl
  if wc<-1e-9 or wc>0.40: continue
  for wn in [0.10,0.15,0.20,0.25,0.30,0.35]:
   for al in [0.0,0.3,0.5,0.7]:
    a=sc(2024,wx,wl,wc,wn,al); b=sc(2022,wx,wl,wc,wn,al)
    rows.append(dict(wx=round(wx,2),wl=round(wl,2),wc=round(wc,2),wn=wn,al=al,
                     f24=a,f22=b,d24=a-BASE24,d22=b-BASE22,
                     both=(a>BASE24 and b>BASE22),mn=min(a-BASE24,b-BASE22)))
d=pd.DataFrame(rows); d.to_csv(OUT+'blend_newhpo.csv',index=False)
log(f'  격자 {len(d)}개 중 both 통과 {d.both.sum()}개')
top=d[d.both].sort_values('mn',ascending=False).head(8)
log(f'\n{"XGB":>5}{"LGB":>6}{"CB":>6}{"NN":>6}{"a":>5} {"2024":>9} {"2022":>9} {"Δ2024":>8} {"Δ2022":>8}')
for _,r in top.iterrows():
    log(f'{r.wx:5.2f}{r.wl:6.2f}{r.wc:6.2f}{r.wn:6.2f}{r.al:5.1f} {r.f24:9.1f} {r.f22:9.1f} {r.d24:+8.1f} {r.d22:+8.1f}')
log('\n완료')
