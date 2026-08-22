"""
18차 — 멀티태스크 NN 생산 학습 (폴드 예측 생성)

17차 결론 (폴드2024 블렌딩이득, 대조군 L0 는 +1.3):
    L3 lam1.0 annealing  +11.6   ← 채택
    L1 lam0.3            +7.2    ← 채택 (다양성)
    L5 lam1.0 emb8       +6.9    ← 채택 (다양성, 임베딩 계열)

여기서는 세 원형을 시드별로 뽑아 폴드 예측을 만든다.
이후 pool 단계에서 기존 8개 NN 풀에 합쳐 구성/가중치를 정직하게 재최적화한다.

사용법: python run18.py <device> <seed들,쉼표>   예) python run18.py cuda:0 0,1
"""
import os, sys, json, time, warnings, traceback
import numpy as np, pandas as pd, scipy.special as sp
import torch, torch.nn as nn
warnings.filterwarnings('ignore')
DEV=sys.argv[1] if len(sys.argv)>1 else 'cuda:0'
SEEDS=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else [0]
TAG='sd'+''.join(map(str,SEEDS))
D='/home/lee/lga/'; OUT=D+'results18/'; os.makedirs(OUT,exist_ok=True)
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

ordr=np.lexsort((RAW.asof_pitcher_n.values, RAW.pitcher_id.values))
pid_s=RAW.pitcher_id.values[ordr]; n_s=RAW.asof_pitcher_n.values[ordr].astype(np.float64)
last=np.append(pid_s[1:]!=pid_s[:-1],True)
COMP=['reverse','middle','ball','strike']
LBd={}
for c in COMP:
    cum=np.nan_to_num(n_s*RAW[f'asof_pitcher_{c}_rate'].values[ordr])
    d=np.append(cum[1:]-cum[:-1],np.nan); d[last]=np.nan
    v=np.round(d); v[np.abs(d-v)>0.3]=np.nan
    o=np.full(len(RAW),np.nan,np.float32); o[ordr]=v; LBd[c]=o
CL=pd.DataFrame(LBd); CM=CL.notna().all(1).values

CAT=['pitcher_id','batter_id','pitcher_team_id','batter_team_id','pitcher_hand',
     'batter_hand','base_state','game_type','top_bottom']
NUM=[c for c in BASE.columns if c not in CAT]
Zr=np.nan_to_num(BASE[NUM].values.astype(np.float32),nan=0.,posinf=0.,neginf=0.)
MU,SD=Zr.mean(0),Zr.std(0)+1e-6
Z=np.clip((Zr-MU)/SD,-6,6)
TN=torch.tensor(Z,device=DEV)
Xc=np.maximum(BASE[CAT].values.astype(np.int64),0)
CARD=[int(Xc[:,i].max())+2 for i in range(len(CAT))]
TC=torch.tensor(Xc,device=DEV)
mth=RAW.game_month.values.astype(np.float32)
TT=torch.tensor(np.stack([np.sin(2*np.pi*mth/12),np.cos(2*np.pi*mth/12),
    np.sin(4*np.pi*mth/12),np.cos(4*np.pi*mth/12),(season-2019)/6.0],1).astype(np.float32),device=DEV)
TY=torch.tensor(y,device=DEV)
TCL=torch.tensor(np.nan_to_num(CL.values.astype(np.float32)),device=DEV)
TCMask=torch.tensor(CM.astype(np.float32),device=DEV)
np.savez(OUT+'nn_norm_mt.npz',mu=MU,sd=SD)
json.dump(NUM,open(OUT+'feat_nn_mt.json','w'))
json.dump(CARD,open(OUT+'card_mt.json','w'))
log(f'{el()} 수치 {len(NUM)} / 범주 {len(CAT)} / 성분라벨 {CM.sum():,}')

class PLR(nn.Module):
    def __init__(s,d_in,k=8,d=6):
        super().__init__(); s.c=nn.Parameter(torch.randn(d_in,k)*.05); s.l=nn.Linear(2*k,d)
    def forward(s,x):
        z=2*np.pi*x.unsqueeze(-1)*s.c
        return torch.relu(s.l(torch.cat([torch.sin(z),torch.cos(z)],-1))).flatten(1)
class MTabM(nn.Module):
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
        o=(z.unsqueeze(1)*s.hd).sum(-1)+s.hb
        return o.mean(-1)

ARCHS={'L3':dict(lam=1.0,ann=True, emb=0),
       'L1':dict(lam=0.3,ann=False,emb=0),
       'L5':dict(lam=1.0,ann=False,emb=8)}
BASEARCH=dict(k=64,h=512,L=3,drop=0.1,plr=True,film=True)
LR,WD,EPS=2e-3,1e-4,60

def train(cfg,sd,tr_mask,seed_tag,save_full=False):
    ia=np.where(tr_mask)[0]; rs=np.random.RandomState(1+sd); rs.shuffle(ia)
    nin=int(len(ia)*.06); iin,itr=ia[:nin],ia[nin:]
    torch.manual_seed(sd)
    net=MTabM(len(NUM),CARD,emb=cfg['emb'],nh=5,**BASEARCH).to(DEV)
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WD)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPS)
    d_in=len(NUM)*7+2*cfg['emb']
    Bs=int(np.clip(1.2e8/max(BASEARCH['k']*max(BASEARCH['h'],d_in)*3,1),1024,16384))
    best,bad,bst=1e9,0,None; t0=time.time()
    for ep in range(EPS):
        lam=cfg['lam']*(max(0.,1-ep/40.) if cfg['ann'] else 1.)
        net.train(); perm=np.random.RandomState(ep+sd*100).permutation(itr)
        for j in range(0,len(perm),Bs):
            b=torch.tensor(perm[j:j+Bs],device=DEV)
            o=torch.sigmoid(net(TN[b],TC[b],TT[b]))
            loss=((o[:,0]-TY[b])**2).mean()
            if lam>0:
                m=TCMask[b].unsqueeze(1)
                loss=loss+lam*((((o[:,1:]-TCL[b])**2)*m).sum()/(m.sum()*4+1e-6))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            se,c=0.,0
            for q in range(0,len(iin),Bs):
                b=torch.tensor(iin[q:q+Bs],device=DEV)
                se+=((torch.sigmoid(net(TN[b],TC[b],TT[b])[:,0])-TY[b])**2).sum().item(); c+=len(b)
            v=se/c
        if v<best-1e-8: best,bad=v,0; bst={q:t.detach().clone() for q,t in net.state_dict().items()}
        else:
            bad+=1
            if bad>=6: break
        if time.time()-t0>900: break
    net.load_state_dict(bst); net.eval()
    if save_full: torch.save(net.state_dict(), OUT+f'nn_{seed_tag}.pt')
    return net,Bs

def predict(net,idx,Bs):
    ps=[]
    with torch.no_grad():
        for j in range(0,len(idx),Bs):
            b=torch.tensor(idx[j:j+Bs],device=DEV)
            ps.append(torch.sigmoid(net(TN[b],TC[b],TT[b])[:,0]).cpu().numpy())
    return np.concatenate(ps)

R=[]
for sd in SEEDS:
    for an,cfg in ARCHS.items():
        tag=f'{an}_s{sd}'
        try:
            sc={}
            for vs in FOLDS:                      # 폴드 예측 (풀 최적화용)
                tr,va=split(vs)
                net,Bs=train(cfg,sd,tr,tag)
                p=predict(net,np.where(va)[0],Bs)
                np.save(OUT+f'{tag}_{vs}.npy',p.astype(np.float32)); sc[vs]=bss(p,vs)
                del net; torch.cuda.empty_cache()
            # 제출용: 전체 학습데이터로 재학습해 가중치 저장
            TRF=~(isF&(season<=2022))
            net,Bs=train(cfg,sd,TRF,tag,save_full=True)
            del net; torch.cuda.empty_cache()
            R.append(dict(arch=an,seed=sd,solo24=sc[2024],solo22=sc[2022]))
            pd.DataFrame(R).to_csv(OUT+f'res18_{TAG}.csv',index=False)
            log(f'{el()} {tag:8s} 단독 24:{sc[2024]:7.1f} 22:{sc[2022]:7.1f}  +제출용 저장')
        except Exception:
            log(f'!! {tag}\n'+traceback.format_exc())
log(f'\n{el()} ===== 완료 =====')
