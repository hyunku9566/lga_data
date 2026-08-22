"""
17차 — 공유 trunk 멀티태스크 NN

배경
  asof_pitcher_*_rate x asof_pitcher_n 을 차분하면 매 투구의
  reverse/middle/ball/strike 라벨이 역산된다 (control_success 와 일치율 1.000000).
  같은 114피처/같은 XGB 로 잰 각 라벨의 스킬:
      y 809 | middle 887 | strike 1211 | reverse 1431 | ball 1778
  y 는 ¬reverse ∧ ¬middle ∧ Z 의 논리곱이라 가장 시끄럽다.

  16차에서 성분을 '투수x상황 피처'로 만든 건 실패했고(-11),
  '조건부 예측을 피처로' 넣은 스태킹은 성공했다(+9.3).
  => 성분의 값어치는 주변확률이 아니라 조건부 모델링에 있다.
  => 표현 자체를 5개 라벨로 같이 깎으면 상한이 더 높은가? 가 이번 질문.

설계
  TabM trunk 공유 + 헤드 5개 (y, reverse, middle, ball, strike)
  loss = Brier(y) + lam * mean_c Brier(c)
  lam=0 이 정확히 단일태스크 대조군 (그 외 모든 조건 동일)

평가는 run10 프로토콜 그대로: 단독 성능이 아니라 '트리 축과의 블렌딩 이득'
"""
import os, sys, json, time, warnings, traceback
import numpy as np, pandas as pd, scipy.special as sp
import torch, torch.nn as nn
warnings.filterwarnings('ignore')
DEV=sys.argv[1] if len(sys.argv)>1 else 'cuda:0'
# argv[2]: 실행할 설정 인덱스 (쉼표구분). 두 GPU 로 쪼개 돌릴 때 사용
SEL=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else None
TAG=('s'+''.join(map(str,SEL))) if SEL else 'all'
D='/home/lee/lga/'; OUT=D+'results17/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+f'log_{TAG}.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:6.1f}m]'

RAW=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig')
X98=pd.read_parquet(D+'X98.parquet'); TM=pd.read_parquet(D+'results14/tm5.parquet')
y=X98.__y.values.astype(np.float32); season=X98.__season.values; isF=X98.__F.values.astype(bool)
CORE=[c for c in X98.columns if not c.startswith('__')]
TMSEL=json.load(open(D+'v6_tmsel.json'))
FOLDS=[2024,2022]
def split(vs): return (season<vs)&~(isF&(season<=2022)&(vs>=2023)), (season==vs)&~isF
def bss(p,vs):
    va=(season==vs)&~isF; yv=y[va]; r=yv.mean()
    return 100000*max(0.,1-np.mean((p-yv)**2)/(r*(1-r)))
lg_=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))

def multi_k():
    F={}
    for idc,nc,rc,pf in [('pitcher_id','asof_pitcher_n','asof_pitcher_success_rate','p_succ'),
                         ('batter_id','asof_batter_n','asof_batter_success_rate','b_succ')]:
        t=RAW[[idc,'season',nc,rc]].copy(); t['succ']=t[nc]*t[rc].fillna(0)
        S=t.loc[t.groupby([idc,'season'])[nc].idxmin()].set_index([idc,'season'])[[nc,'succ']]
        a=RAW[[idc,'season']].join(S,on=[idc,'season'])
        dn=np.maximum(RAW[nc].values-a[nc].fillna(0).values,0)
        ds=np.maximum(np.nan_to_num(RAW[nc].values*RAW[rc].values)-a['succ'].fillna(0).values,0)
        lgv=np.nanmean(RAW[rc])
        for k in [25,75,400,1000]: F[f'{pf}_k{k}']=(ds+k*lgv)/(dn+k)
    return pd.DataFrame(F,index=RAW.index).astype(np.float32)
BASE=pd.concat([X98[CORE],multi_k(),TM[TMSEL]],axis=1)

# ── 성분 라벨 역산 ──
ordr=np.lexsort((RAW.asof_pitcher_n.values, RAW.pitcher_id.values))
pid_s=RAW.pitcher_id.values[ordr]; n_s=RAW.asof_pitcher_n.values[ordr].astype(np.float64)
last=np.append(pid_s[1:]!=pid_s[:-1],True)
COMP=['reverse','middle','ball','strike']
LB={}
for c in COMP:
    cum=np.nan_to_num(n_s*RAW[f'asof_pitcher_{c}_rate'].values[ordr])
    d=np.append(cum[1:]-cum[:-1],np.nan); d[last]=np.nan
    v=np.round(d); v[np.abs(d-v)>0.3]=np.nan
    o=np.full(len(RAW),np.nan,np.float32); o[ordr]=v; LB[c]=o
CL=pd.DataFrame(LB)
CM=CL.notna().all(1).values                      # 성분 손실 마스크
log(f'{el()} 성분라벨 {CM.sum():,}/{len(CL):,}')

# ── 트리 축 (v6 XGB+LGB) = 블렌딩 상대 ──
W6=json.load(open(D+'v6_weights.json'))
TREE={}
for vs in FOLDS:
    zx=lg_(np.load(D+f'results14/xgb6_{vs}.npy')); zl=lg_(np.load(D+f'results14/lgb6_{vs}.npy'))
    TREE[vs]=(W6['a']*zx+W6['b']*zl)/(W6['a']+W6['b'])
B0={vs:bss(sp.expit(TREE[vs]),vs) for vs in FOLDS}
log(f'{el()} 트리 축 기준  24:{B0[2024]:7.1f}  22:{B0[2022]:7.1f}')

# ── 입력 ──
CAT=['pitcher_id','batter_id','pitcher_team_id','batter_team_id','pitcher_hand',
     'batter_hand','base_state','game_type','top_bottom']
NUM=[c for c in BASE.columns if c not in CAT]
Z=np.nan_to_num(BASE[NUM].values.astype(np.float32),nan=0.,posinf=0.,neginf=0.)
Z=np.clip((Z-Z.mean(0))/(Z.std(0)+1e-6),-6,6)
TN=torch.tensor(Z,device=DEV)
Xc=np.maximum(BASE[CAT].values.astype(np.int64),0)
CARD=[int(Xc[:,i].max())+2 for i in range(len(CAT))]
TC=torch.tensor(Xc,device=DEV)
mth=RAW.game_month.values.astype(np.float32)
TT=torch.tensor(np.stack([np.sin(2*np.pi*mth/12),np.cos(2*np.pi*mth/12),
    np.sin(4*np.pi*mth/12),np.cos(4*np.pi*mth/12),(season-2019)/6.0],1).astype(np.float32),device=DEV)
TY=torch.tensor(y,device=DEV)
TCL=torch.tensor(np.nan_to_num(CL.values.astype(np.float32)),device=DEV)   # (N,4)
TCMask=torch.tensor(CM.astype(np.float32),device=DEV)
log(f'{el()} 수치피처 {len(NUM)} / 범주 {len(CAT)}')

class PLR(nn.Module):
    def __init__(s,d_in,k=8,d=6):
        super().__init__(); s.c=nn.Parameter(torch.randn(d_in,k)*.05); s.l=nn.Linear(2*k,d)
    def forward(s,x):
        z=2*np.pi*x.unsqueeze(-1)*s.c
        return torch.relu(s.l(torch.cat([torch.sin(z),torch.cos(z)],-1))).flatten(1)

class MTabM(nn.Module):
    """TabM trunk 공유 + 헤드 nh 개. nh=1 이면 기존 TabM 과 동일 구조."""
    def __init__(s,n,card,k=64,h=512,L=3,emb=0,drop=.1,plr=True,film=True,nh=5):
        super().__init__(); s.k,s.emb,s.film,s.plr,s.nh=k,emb,film,None,nh; d=n
        if plr: s.plr=PLR(n); d+=n*6
        if emb:
            s.es=nn.ModuleList([nn.Embedding(c,emb) for c in card[:2]])
            for e in s.es: nn.init.normal_(e.weight,std=.01)
            d+=2*emb
        s.r1=nn.Parameter(torch.randn(k,d)*.1+1)
        s.ls=nn.ModuleList([nn.Linear(d if i==0 else h,h) for i in range(L)])
        s.dp=nn.Dropout(drop)
        s.hd=nn.Parameter(torch.randn(nh,k,h)*.02); s.hb=nn.Parameter(torch.zeros(nh,k))
        if film: s.fn=nn.Sequential(nn.Linear(5,32),nn.ReLU(),nn.Linear(32,2*h))
    def forward(s,xn,xc,tt):
        z=xn
        if s.plr is not None: z=torch.cat([z,s.plr(xn)],-1)
        if s.emb: z=torch.cat([z]+[e(xc[:,i]) for i,e in enumerate(s.es)],-1)
        z=z.unsqueeze(1)*s.r1
        for i,l in enumerate(s.ls):
            z=torch.relu(l(z))
            if i==0:
                z=s.dp(z)
                if s.film:
                    g,b=s.fn(tt).chunk(2,-1); z=z*(1+g.unsqueeze(1))+b.unsqueeze(1)
        o=(z.unsqueeze(1)*s.hd).sum(-1)+s.hb      # (B,nh,k)
        return o.mean(-1)                          # (B,nh)

CFGS=[
    dict(name='L0 단일태스크(대조)', lam=0.0,  ann=False, emb=0),
    dict(name='L1 lam=0.3',        lam=0.3,  ann=False, emb=0),
    dict(name='L2 lam=1.0',        lam=1.0,  ann=False, emb=0),
    dict(name='L3 lam=1.0 anneal', lam=1.0,  ann=True,  emb=0),
    dict(name='L4 단일+emb8(대조)',  lam=0.0,  ann=False, emb=8),
    dict(name='L5 lam=1.0 +emb8',  lam=1.0,  ann=False, emb=8),
]
ARCH=dict(k=64,h=512,L=3,drop=0.1,plr=True,film=True)
LR,WD,EPS=2e-3,1e-4,60
R=[]
for ci,cfg in enumerate(CFGS):
    if SEL is not None and ci not in SEL: continue
    try:
        sc={}; preds={}
        for vs in FOLDS:
            tr,va=split(vs)
            ia=np.where(tr)[0]; rs=np.random.RandomState(1); rs.shuffle(ia)
            nin=int(len(ia)*.06); iin,itr=ia[:nin],ia[nin:]; iva=np.where(va)[0]
            torch.manual_seed(0)
            net=MTabM(len(NUM),CARD,emb=cfg['emb'],nh=5,**ARCH).to(DEV)
            opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WD)
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPS)
            d_in=len(NUM)*7+2*cfg['emb']
            Bs=int(np.clip(1.2e8/max(ARCH['k']*max(ARCH['h'],d_in)*3,1),1024,16384))
            best,bad,bst=1e9,0,None; t0=time.time()
            for ep in range(EPS):
                lam=cfg['lam']*(max(0.,1-ep/40.) if cfg['ann'] else 1.)
                net.train(); perm=np.random.RandomState(ep).permutation(itr)
                for j in range(0,len(perm),Bs):
                    b=torch.tensor(perm[j:j+Bs],device=DEV)
                    o=torch.sigmoid(net(TN[b],TC[b],TT[b]))
                    loss=((o[:,0]-TY[b])**2).mean()
                    if lam>0:
                        m=TCMask[b].unsqueeze(1)
                        lc=(((o[:,1:]-TCL[b])**2)*m).sum()/(m.sum()*4+1e-6)
                        loss=loss+lam*lc
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step()
                sch.step(); net.eval()
                with torch.no_grad():          # 조기중단은 y 헤드 기준
                    se,c=0.,0
                    for q in range(0,len(iin),Bs):
                        b=torch.tensor(iin[q:q+Bs],device=DEV)
                        se+=((torch.sigmoid(net(TN[b],TC[b],TT[b])[:,0])-TY[b])**2).sum().item(); c+=len(b)
                    v=se/c
                if v<best-1e-8: best,bad=v,0; bst={q:t.detach().clone() for q,t in net.state_dict().items()}
                else:
                    bad+=1
                    if bad>=6: break
                if time.time()-t0>600: break
            net.load_state_dict(bst); net.eval(); ps=[]
            with torch.no_grad():
                for j in range(0,len(iva),Bs):
                    b=torch.tensor(iva[j:j+Bs],device=DEV)
                    ps.append(torch.sigmoid(net(TN[b],TC[b],TT[b])[:,0]).cpu().numpy())
            p=np.concatenate(ps); preds[vs]=p; sc[vs]=bss(p,vs)
            np.save(OUT+f'mt_c{ci}_{vs}.npy',p.astype(np.float32))
        gains={w:np.mean([bss(sp.expit((1-w)*TREE[vs]+w*lg_(preds[vs])),vs)-B0[vs] for vs in FOLDS])
               for w in [0.15,0.2,0.25,0.3]}
        bw=max(gains,key=gains.get)
        R.append(dict(ci=ci,name=cfg['name'],lam=cfg['lam'],ann=cfg['ann'],emb=cfg['emb'],
                      solo24=sc[2024],solo22=sc[2022],solo=np.mean(list(sc.values())),
                      best_w=bw,gain=gains[bw]))
        pd.DataFrame(R).to_csv(OUT+f'res17_{TAG}.csv',index=False)
        log(f'{el()} [{ci}] {cfg["name"]:20s} | 단독 24:{sc[2024]:7.1f} 22:{sc[2022]:7.1f} '
            f'(평균 {np.mean(list(sc.values())):6.1f}) | 블렌딩이득 {gains[bw]:+6.1f} (w={bw})')
    except Exception:
        log(f'!! c{ci}\n'+traceback.format_exc())
log(f'\n{el()} ===== 완료 =====')
if R:
    b=pd.DataFrame(R).sort_values('gain',ascending=False)
    log(b[['name','solo','best_w','gain']].to_string(index=False))
