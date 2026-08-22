"""
11차 — FT-Transformer 계열 (아직 한 번도 측정 안 된 아키텍처)
사용법: python run11.py <device> <shard> <nshards>

동기: run6 에 FTT 를 넣었으나 15/138 에서 중단돼 단 하나도 완료되지 않았음.
      run10 은 TabM 만. => 트랜스포머는 미측정 상태.

트랜스포머의 값어치는 '병렬성'이 아니라 (MLP 도 완전 병렬) '어텐션이 피처 상호작용을
스스로 발견' 하는 것. 우리가 손으로 만들던 교차항을 모델이 찾게 한다.

3종:
  FTT      표준 FT-Transformer (피처 토큰화 + self-attention + CLS)
  FTT-lite 토큰 수를 줄인 경량형 (수치 피처를 그룹으로 묶어 토큰화)
  AFN      어텐션 없이 피처쌍 상호작용만 (대조군)

평가 기준은 run10 과 동일: 단독 성능이 아니라 'XGB 와 블렌딩했을 때 이득'
"""
import os, sys, time, warnings, traceback
import numpy as np, pandas as pd, scipy.special as sp
import torch, torch.nn as nn
import xgboost as xgb
warnings.filterwarnings('ignore')

DEV=sys.argv[1] if len(sys.argv)>1 else 'cuda:0'
SHARD=int(sys.argv[2]) if len(sys.argv)>2 else 0
NSH=int(sys.argv[3]) if len(sys.argv)>3 else 1
D='/home/lee/lga/'; OUT=D+'results11/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+f'log_s{SHARD}.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:6.1f}m]'
log(f'\n{"="*66}\n{el()} FTT shard {SHARD}/{NSH} @ {DEV}\n{"="*66}')

RAW=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig')
X=pd.read_parquet(D+'X98.parquet')
y=X.__y.values.astype(np.float32); season=X.__season.values; isF=X.__F.values.astype(bool)
CORE=[c for c in X.columns if not c.startswith('__')]
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
        lg=np.nanmean(RAW[rc])
        for k in [25,75,400,1000]: F[f'{pf}_k{k}']=(ds+k*lg)/(dn+k)
    return pd.DataFrame(F,index=RAW.index).astype(np.float32)
XK=pd.concat([X[CORE],multi_k()],axis=1)
PRM=dict(n_estimators=600,learning_rate=0.008,max_depth=6,min_child_weight=1500,subsample=0.7,
         colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,tree_method='hist',device=DEV,
         eval_metric='logloss',verbosity=0)
XGP={}
for vs in FOLDS:
    f=OUT+f'../results10/xgb_{vs}.npy'
    if os.path.exists(f): XGP[vs]=np.load(f)
    else:
        tr,va=split(vs)
        XGP[vs]=np.mean([xgb.XGBClassifier(**PRM,random_state=s).fit(XK.loc[tr],y[tr])
                         .predict_proba(XK.loc[va])[:,1] for s in range(4)],0)
B0={vs:bss(XGP[vs],vs) for vs in FOLDS}
log(f'{el()} XGB 기준  24:{B0[2024]:7.1f}  22:{B0[2022]:7.1f}')

CAT=['pitcher_id','batter_id','pitcher_team_id','batter_team_id','pitcher_hand',
     'batter_hand','base_state','game_type','top_bottom']
NUM=[c for c in CORE if c not in CAT]
Z=np.nan_to_num(X[NUM].values.astype(np.float32),nan=0.,posinf=0.,neginf=0.)
Z=np.clip((Z-Z.mean(0))/(Z.std(0)+1e-6),-6,6)
TN=torch.tensor(Z,device=DEV)
Xc=np.maximum(X[CAT].values.astype(np.int64),0)
CARD=[int(Xc[:,i].max())+2 for i in range(len(CAT))]
TC=torch.tensor(Xc,device=DEV)
mth=X.game_month.values.astype(np.float32)
TT=torch.tensor(np.stack([np.sin(2*np.pi*mth/12),np.cos(2*np.pi*mth/12),
    np.sin(4*np.pi*mth/12),np.cos(4*np.pi*mth/12),(season-2019)/6.0],1).astype(np.float32),device=DEV)
TY=torch.tensor(y,device=DEV)
NN_=Z.shape[1]
log(f'{el()} 수치 {NN_} 범주 {len(CAT)}')

class FTT(nn.Module):
    """표준 FT-Transformer: 피처마다 토큰 1개 + CLS, self-attention"""
    def __init__(s,n,card,d=32,L=2,heads=4,drop=.1,ncat=4,**kw):
        super().__init__()
        s.w=nn.Parameter(torch.randn(n,d)*.05); s.b=nn.Parameter(torch.zeros(n,d))
        s.ce=nn.ModuleList([nn.Embedding(card[i],d) for i in range(2,2+ncat)])  # 팀/손/상태만
        s.cls=nn.Parameter(torch.randn(1,1,d)*.05)
        s.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d,heads,d*2,drop,batch_first=True,norm_first=True,activation='gelu'),L)
        s.tm=nn.Sequential(nn.Linear(5,d),nn.GELU())
        s.out=nn.Linear(d,1)
    def forward(s,xn,xc,tt):
        tok=xn.unsqueeze(-1)*s.w+s.b
        ct=torch.stack([e(xc[:,i+2]) for i,e in enumerate(s.ce)],1)
        z=torch.cat([s.cls.expand(xn.shape[0],-1,-1),s.tm(tt).unsqueeze(1),tok,ct],1)
        return s.out(s.tr(z)[:,0]).squeeze(-1)

class FTTLite(nn.Module):
    """경량형: 수치피처를 g개 그룹으로 묶어 토큰화 -> 어텐션 비용 대폭 감소"""
    def __init__(s,n,card,d=48,L=2,heads=4,drop=.1,g=12,**kw):
        super().__init__()
        s.g=g; s.sz=int(np.ceil(n/g)); s.pad=s.g*s.sz-n
        s.proj=nn.Linear(s.sz,d)
        s.pos=nn.Parameter(torch.randn(1,g,d)*.05)
        s.cls=nn.Parameter(torch.randn(1,1,d)*.05)
        s.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d,heads,d*2,drop,batch_first=True,norm_first=True,activation='gelu'),L)
        s.tm=nn.Sequential(nn.Linear(5,d),nn.GELU())
        s.out=nn.Linear(d,1)
    def forward(s,xn,xc,tt):
        x=torch.nn.functional.pad(xn,(0,s.pad)).view(xn.shape[0],s.g,s.sz)
        z=s.proj(x)+s.pos
        z=torch.cat([s.cls.expand(xn.shape[0],-1,-1),s.tm(tt).unsqueeze(1),z],1)
        return s.out(s.tr(z)[:,0]).squeeze(-1)

class AFN(nn.Module):
    """대조군: 어텐션 없이 저차원 피처쌍 상호작용(팩터라이제이션)만"""
    def __init__(s,n,card,d=16,h=128,drop=.1,**kw):
        super().__init__()
        s.v=nn.Parameter(torch.randn(n,d)*.05)
        s.lin=nn.Linear(n,1)
        s.mlp=nn.Sequential(nn.Linear(d+5,h),nn.GELU(),nn.Dropout(drop),nn.Linear(h,1))
    def forward(s,xn,xc,tt):
        vx=xn.unsqueeze(-1)*s.v
        inter=0.5*((vx.sum(1)**2)-(vx**2).sum(1))          # FM 2차 상호작용
        return s.lin(xn).squeeze(-1)+s.mlp(torch.cat([inter,tt],-1)).squeeze(-1)

ARCH={'FTT':FTT,'FTTLite':FTTLite,'AFN':AFN}
CFGS=[]
for d in [16,32,64]:
    for L in [1,2,3]:
        for wd in [1e-3,1e-2]:
            CFGS.append(dict(arch='FTT',d=d,L=L,heads=4,drop=0.1,wd=wd,lr=1e-3,bs=4096))
for d in [32,48,64]:
    for L in [2,3]:
        for g in [8,16]:
            for wd in [1e-3,1e-2]:
                CFGS.append(dict(arch='FTTLite',d=d,L=L,g=g,heads=4,drop=0.1,wd=wd,lr=1e-3,bs=8192))
for d in [8,16,32]:
    for h in [64,128]:
        for wd in [1e-3,1e-2]:
            CFGS.append(dict(arch='AFN',d=d,h=h,drop=0.1,wd=wd,lr=2e-3,bs=16384))
rng=np.random.RandomState(9); rng.shuffle(CFGS)
MINE=CFGS[SHARD::NSH]
log(f'{el()} 전체 {len(CFGS)} 중 이 샤드 {len(MINE)}개\n')

R=[]
for ci,cfg in enumerate(MINE):
    try:
        sc={}; preds={}
        for vs in FOLDS:
            tr,va=split(vs)
            ia=np.where(tr)[0]; rs=np.random.RandomState(1); rs.shuffle(ia)
            nin=int(len(ia)*.06); iin,itr=ia[:nin],ia[nin:]; iva=np.where(va)[0]
            torch.manual_seed(0)
            net=ARCH[cfg['arch']](NN_,CARD,**{k:v for k,v in cfg.items()
                                  if k not in ('arch','wd','lr','bs')}).to(DEV)
            opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40)
            Bs=cfg['bs']; best,bad,bst=1e9,0,None; t0=time.time()
            for ep in range(40):
                net.train(); perm=np.random.RandomState(ep).permutation(itr)
                for j in range(0,len(perm),Bs):
                    b=torch.tensor(perm[j:j+Bs],device=DEV)
                    loss=((torch.sigmoid(net(TN[b],TC[b],TT[b]))-TY[b])**2).mean()
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step()
                sch.step(); net.eval()
                with torch.no_grad():
                    se,c=0.,0
                    for q in range(0,len(iin),Bs):
                        b=torch.tensor(iin[q:q+Bs],device=DEV)
                        se+=((torch.sigmoid(net(TN[b],TC[b],TT[b]))-TY[b])**2).sum().item(); c+=len(b)
                    v=se/c
                if v<best-1e-8: best,bad=v,0; bst={q:t.detach().clone() for q,t in net.state_dict().items()}
                else:
                    bad+=1
                    if bad>=5: break
                if time.time()-t0>900: break
            net.load_state_dict(bst); net.eval(); ps=[]
            with torch.no_grad():
                for j in range(0,len(iva),Bs):
                    b=torch.tensor(iva[j:j+Bs],device=DEV)
                    ps.append(torch.sigmoid(net(TN[b],TC[b],TT[b])).cpu().numpy())
            p=np.concatenate(ps); preds[vs]=p; sc[vs]=bss(p,vs)
            np.save(OUT+f'nn_s{SHARD}_c{ci}_{vs}.npy',p.astype(np.float32))
        gains={w:np.mean([bss(sp.expit((1-w)*lg_(XGP[vs])+w*lg_(preds[vs])),vs)-B0[vs] for vs in FOLDS])
               for w in [0.05,0.1,0.15,0.2,0.3]}
        bw=max(gains,key=gains.get)
        R.append(dict(shard=SHARD,ci=ci,solo=np.mean(list(sc.values())),solo24=sc[2024],solo22=sc[2022],
                      best_w=bw,gain=gains[bw],ep=ep+1,**cfg))
        pd.DataFrame(R).to_csv(OUT+f'res_s{SHARD}.csv',index=False)
        log(f'{el()} [{ci:2d}] {cfg["arch"]:8s} d{cfg.get("d")} L{cfg.get("L","-")} '
            f'g{cfg.get("g","-")} wd{cfg["wd"]:<6} ep{ep+1:<3}| 단독 {np.mean(list(sc.values())):6.1f} '
            f'| 블렌딩이득 {gains[bw]:+6.1f} (w={bw})')
    except Exception:
        log(f'!! c{ci}\n'+traceback.format_exc())
log(f'\n{el()} ===== FTT shard {SHARD} 완료 =====')
