"""
10차 — NN 다양성 극대화 (양 GPU)
사용법: python run10.py <device> <shard> <nshards>

근거: v3 에서 NN 블렌딩이 LB +49 를 냈다. 그런데 블렌딩에 쓴 최고 파트너는
      단독 171점짜리였다. 즉 NN 의 값어치는 정확도가 아니라 'XGB 와 다름' 이다.
      => NN 에게 우리가 만든 피처를 덜 줄수록 더 달라지고 블렌딩 이득이 커질 수 있다.

입력 세트
  A raw   : 원본 48컬럼만 (NN 이 전부 스스로 학습)
  B raw+d : 원본 48 + 역산 (역산은 앵커 룩업이 필요해 NN 이 유도 불가)
  C full  : 현재 98 엔지니어링 피처 (v3 와 동일)

평가는 '단독 성능' 이 아니라 'XGB 와 블렌딩했을 때의 이득' 으로 한다.
"""
import os, sys, time, warnings, traceback
import numpy as np, pandas as pd, scipy.special as sp
import torch, torch.nn as nn
import xgboost as xgb
warnings.filterwarnings('ignore')

DEV=sys.argv[1] if len(sys.argv)>1 else 'cuda:0'
SHARD=int(sys.argv[2]) if len(sys.argv)>2 else 0
NSH=int(sys.argv[3]) if len(sys.argv)>3 else 1
D='/home/lee/lga/'; OUT=D+'results10/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+f'log_s{SHARD}.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:6.1f}m]'
log(f'\n{"="*66}\n{el()} shard {SHARD}/{NSH} @ {DEV}\n{"="*66}')

RAW=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig')
X=pd.read_parquet(D+'X98.parquet')
y=X.__y.values.astype(np.float32); season=X.__season.values; isF=X.__F.values.astype(bool)
CORE=[c for c in X.columns if not c.startswith('__')]
FOLDS=[2024,2022]
def split(vs):
    return (season<vs)&~(isF&(season<=2022)&(vs>=2023)), (season==vs)&~isF
def bss(p,vs):
    va=(season==vs)&~isF; yv=y[va]; r=yv.mean()
    return 100000*max(0.,1-np.mean((p-yv)**2)/(r*(1-r)))
lg_=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))

# ── XGB 기준 (다중축소강도 포함 = v3 구성) ──
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
XGP={}
PRM=dict(n_estimators=600,learning_rate=0.008,max_depth=6,min_child_weight=1500,subsample=0.7,
         colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,tree_method='hist',device=DEV,
         eval_metric='logloss',verbosity=0)
for vs in FOLDS:
    tr,va=split(vs)
    XGP[vs]=np.mean([xgb.XGBClassifier(**PRM,random_state=s).fit(XK.loc[tr],y[tr])
                     .predict_proba(XK.loc[va])[:,1] for s in range(4)],0)
    np.save(OUT+f'xgb_{vs}.npy',XGP[vs].astype(np.float32))
B0={vs:bss(XGP[vs],vs) for vs in FOLDS}
log(f'{el()} XGB 기준  24:{B0[2024]:7.1f}  22:{B0[2022]:7.1f}')

# ── 입력 세트 3종 ──
CAT=['pitcher_id','batter_id','pitcher_team_id','batter_team_id','pitcher_hand',
     'batter_hand','base_state','game_type','top_bottom']
RAW48=[c for c in CORE if not c.startswith(('p_','b_','tm_','hand_mix'))]
DECON=[c for c in CORE if '_ssn' in c or c in ('p_prev5_vs_car','p_prev1_vs_prev5')]
SETS={'A_raw':   [c for c in RAW48 if c not in CAT],
      'B_raw_d': [c for c in RAW48+DECON if c not in CAT],
      'C_full':  [c for c in CORE if c not in CAT]}
PREP={}
for nm,cols in SETS.items():
    Z=np.nan_to_num(X[cols].values.astype(np.float32),nan=0.,posinf=0.,neginf=0.)
    Z=np.clip((Z-Z.mean(0))/(Z.std(0)+1e-6),-6,6)
    PREP[nm]=torch.tensor(Z,device=DEV)
    log(f'{el()} 입력 {nm}: {len(cols)}개')
Xc=np.maximum(X[CAT].values.astype(np.int64),0)
CARD=[int(Xc[:,i].max())+2 for i in range(len(CAT))]
TC=torch.tensor(Xc,device=DEV)
mth=X.game_month.values.astype(np.float32)
TT=torch.tensor(np.stack([np.sin(2*np.pi*mth/12),np.cos(2*np.pi*mth/12),
    np.sin(4*np.pi*mth/12),np.cos(4*np.pi*mth/12),(season-2019)/6.0],1).astype(np.float32),device=DEV)
TY=torch.tensor(y,device=DEV)

class PLR(nn.Module):
    def __init__(s,d_in,k=8,d=6):
        super().__init__(); s.c=nn.Parameter(torch.randn(d_in,k)*.05); s.l=nn.Linear(2*k,d)
    def forward(s,x):
        z=2*np.pi*x.unsqueeze(-1)*s.c
        return torch.relu(s.l(torch.cat([torch.sin(z),torch.cos(z)],-1))).flatten(1)
class TabM(nn.Module):
    def __init__(s,n,card,k=32,h=256,L=2,emb=0,drop=.1,plr=False,film=True):
        super().__init__(); s.k,s.emb,s.film,s.plr=k,emb,film,None; d=n
        if plr: s.plr=PLR(n); d+=n*6
        if emb:
            s.es=nn.ModuleList([nn.Embedding(c,emb) for c in card[:2]])
            for e in s.es: nn.init.normal_(e.weight,std=.01)
            d+=2*emb
        s.r1=nn.Parameter(torch.randn(k,d)*.1+1)
        s.ls=nn.ModuleList([nn.Linear(d if i==0 else h,h) for i in range(L)])
        s.dp=nn.Dropout(drop); s.hd=nn.Parameter(torch.randn(k,h)*.02); s.hb=nn.Parameter(torch.zeros(k))
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
        return ((z*s.hd).sum(-1)+s.hb).mean(1)

CFGS=[]
for iset in ['A_raw','B_raw_d','C_full']:
    for k in [8,32,64]:
        for h in [128,256,512]:
            for plr in [False,True]:
                for emb in [0,4]:
                    for wd in [1e-3,1e-2]:
                        CFGS.append(dict(iset=iset,k=k,h=h,L=2,plr=plr,emb=emb,wd=wd,lr=2e-3,drop=0.1))
rng=np.random.RandomState(5); rng.shuffle(CFGS)
MINE=CFGS[SHARD::NSH][:60]
log(f'{el()} 이 샤드 {len(MINE)}개 설정\n')

R=[]
for ci,cfg in enumerate(MINE):
    try:
        TN=PREP[cfg['iset']]; nnum=TN.shape[1]
        sc={}; preds={}
        for vs in FOLDS:
            tr,va=split(vs)
            ia=np.where(tr)[0]; rs=np.random.RandomState(1); rs.shuffle(ia)
            nin=int(len(ia)*.06); iin,itr=ia[:nin],ia[nin:]; iva=np.where(va)[0]
            torch.manual_seed(0)
            net=TabM(nnum,CARD,k=cfg['k'],h=cfg['h'],L=cfg['L'],emb=cfg['emb'],
                     drop=cfg['drop'],plr=cfg['plr']).to(DEV)
            opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60)
            d_in=nnum*(7 if cfg['plr'] else 1)+2*cfg['emb']
            Bs=int(np.clip(1.2e8/max(cfg['k']*max(cfg['h'],d_in)*3,1),1024,16384))
            best,bad,bst=1e9,0,None; t0=time.time()
            for ep in range(60):
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
                    if bad>=6: break
                if time.time()-t0>420: break
            net.load_state_dict(bst); net.eval(); ps=[]
            with torch.no_grad():
                for j in range(0,len(iva),Bs):
                    b=torch.tensor(iva[j:j+Bs],device=DEV)
                    ps.append(torch.sigmoid(net(TN[b],TC[b],TT[b])).cpu().numpy())
            p=np.concatenate(ps); preds[vs]=p; sc[vs]=bss(p,vs)
            np.save(OUT+f'nn_s{SHARD}_c{ci}_{vs}.npy',p.astype(np.float32))
        # 핵심 지표: 단독이 아니라 '블렌딩 이득'
        gains={}
        for w in [0.15,0.2,0.25,0.3]:
            gains[w]=np.mean([bss(sp.expit((1-w)*lg_(XGP[vs])+w*lg_(preds[vs])),vs)-B0[vs] for vs in FOLDS])
        bw=max(gains,key=gains.get)
        R.append(dict(shard=SHARD,ci=ci,solo24=sc[2024],solo22=sc[2022],
                      solo=np.mean(list(sc.values())),best_w=bw,gain=gains[bw],**cfg))
        pd.DataFrame(R).to_csv(OUT+f'res_s{SHARD}.csv',index=False)
        log(f'{el()} [{ci:2d}] {cfg["iset"]:8s} k{cfg["k"]:<2} h{cfg["h"]:<3} plr{int(cfg["plr"])} emb{cfg["emb"]} '
            f'wd{cfg["wd"]:<6} | 단독 {np.mean(list(sc.values())):6.1f} | 블렌딩이득 {gains[bw]:+6.1f} (w={bw})')
    except Exception:
        log(f'!! c{ci}\n'+traceback.format_exc())
log(f'\n{el()} ===== shard {SHARD} 완료 =====')
