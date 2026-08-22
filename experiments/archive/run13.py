"""
13차 — 세 가지 제안 검증
  A LightGBM  (리프 단위 성장 + 범주형 네이티브) : 단독 / XGB 블렌딩
  B 스태킹    (NN 예측을 XGB 피처로) vs 현행 블렌딩
  C 트랙맨을 NN 에 (지금까지 XGB 에만 넣어봤음)

평가: 폴드 2024 / 2022, 기준은 현행 XGB(106피처)
"""
import os, time, json, warnings, traceback
import numpy as np, pandas as pd, scipy.special as sp
import xgboost as xgb, lightgbm as lgb
import torch, torch.nn as nn
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results13/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'
DEV='cuda:0'

RAW=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig')
X98=pd.read_parquet(D+'X98.parquet')
FE=pd.read_parquet(D+'features.parquet')
y=X98.__y.values.astype(np.float32); season=X98.__season.values; isF=X98.__F.values.astype(bool)
CORE=[c for c in X98.columns if not c.startswith('__')]
TMC=[c for c in FE.columns if c.startswith('tm_')]
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
XK=pd.concat([X98[CORE],multi_k()],axis=1)
XGPRM=dict(n_estimators=600,learning_rate=0.008,max_depth=6,min_child_weight=1500,subsample=0.7,
           colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,tree_method='hist',device=DEV,
           eval_metric='logloss',verbosity=0)
XGP={}
for vs in FOLDS:
    f=D+f'results10/xgb_{vs}.npy'
    XGP[vs]=np.load(f) if os.path.exists(f) else None
    if XGP[vs] is None:
        tr,va=split(vs)
        XGP[vs]=np.mean([xgb.XGBClassifier(**XGPRM,random_state=s).fit(XK.loc[tr],y[tr])
                         .predict_proba(XK.loc[va])[:,1] for s in range(4)],0)
B0={vs:bss(XGP[vs],vs) for vs in FOLDS}
log(f'{el()} XGB 기준  24:{B0[2024]:7.1f}  22:{B0[2022]:7.1f}\n')
R=[]
def rec(**kw):
    R.append(kw); pd.DataFrame(R).to_csv(OUT+'res13.csv',index=False)

# ═══════ A. LightGBM ═══════
CATIDX=['pitcher_id','batter_id','pitcher_team_id','batter_team_id','base_state','game_type']
def stageA():
    log(f'{el()} ===== A. LightGBM =====')
    Xl=XK.copy()
    for c in CATIDX: Xl[c]=Xl[c].astype(int).astype('category')
    for name,natcat,prm in [
        ('LGB 기본',       False, dict(n_estimators=1200,learning_rate=0.01,num_leaves=31,
                                      min_child_samples=1500,subsample=0.7,subsample_freq=1,
                                      colsample_bytree=0.5,reg_lambda=50.)),
        ('LGB 범주형네이티브', True, dict(n_estimators=1200,learning_rate=0.01,num_leaves=31,
                                      min_child_samples=1500,subsample=0.7,subsample_freq=1,
                                      colsample_bytree=0.5,reg_lambda=50.)),
        ('LGB 얕은',       True,  dict(n_estimators=2500,learning_rate=0.01,num_leaves=15,
                                      min_child_samples=3000,subsample=0.7,subsample_freq=1,
                                      colsample_bytree=0.4,reg_lambda=200.))]:
        try:
            sc={}; pv={}
            for vs in FOLDS:
                tr,va=split(vs)
                XX=Xl if natcat else XK
                m=lgb.LGBMClassifier(**prm,random_state=0,verbose=-1,n_jobs=16).fit(XX.loc[tr],y[tr])
                p=m.predict_proba(XX.loc[va])[:,1]; pv[vs]=p; sc[vs]=bss(p,vs)
                np.save(OUT+f'lgb_{name.replace(" ","_")}_{vs}.npy',p.astype(np.float32))
            gains={w:np.mean([bss(sp.expit((1-w)*lg_(XGP[v])+w*lg_(pv[v])),v)-B0[v] for v in FOLDS])
                   for w in [0.2,0.3,0.4,0.5]}
            bw=max(gains,key=gains.get)
            rec(stage='A',name=name,solo24=sc[2024],solo22=sc[2022],
                solo=np.mean(list(sc.values())),best_w=bw,gain=gains[bw])
            log(f'{el()} {name:16s} 단독 24:{sc[2024]:7.1f} 22:{sc[2022]:7.1f} | XGB블렌딩이득 {gains[bw]:+6.1f} (w={bw})')
        except Exception: log('!! '+name+'\n'+traceback.format_exc())

# ═══════ NN 유틸 ═══════
CAT=['pitcher_id','batter_id','pitcher_team_id','batter_team_id','pitcher_hand',
     'batter_hand','base_state','game_type','top_bottom']
class PLR(nn.Module):
    def __init__(s,d_in,k=8,d=6):
        super().__init__(); s.c=nn.Parameter(torch.randn(d_in,k)*.05); s.l=nn.Linear(2*k,d)
    def forward(s,x):
        z=2*np.pi*x.unsqueeze(-1)*s.c
        return torch.relu(s.l(torch.cat([torch.sin(z),torch.cos(z)],-1))).flatten(1)
class TabM(nn.Module):
    def __init__(s,n,k=32,h=256,L=3,drop=.1,plr=True,film=True):
        super().__init__(); s.k,s.film,s.plr=k,film,None; d=n
        if plr: s.plr=PLR(n); d+=n*6
        s.r1=nn.Parameter(torch.randn(k,d)*.1+1)
        s.ls=nn.ModuleList([nn.Linear(d if i==0 else h,h) for i in range(L)])
        s.dp=nn.Dropout(drop); s.hd=nn.Parameter(torch.randn(k,h)*.02); s.hb=nn.Parameter(torch.zeros(k))
        if film: s.fn=nn.Sequential(nn.Linear(5,32),nn.ReLU(),nn.Linear(32,2*h))
    def forward(s,xn,tt):
        z=xn
        if s.plr is not None: z=torch.cat([z,s.plr(xn)],-1)
        z=z.unsqueeze(1)*s.r1
        for i,l in enumerate(s.ls):
            z=torch.relu(l(z))
            if i==0:
                z=s.dp(z)
                if s.film:
                    g,b=s.fn(tt).chunk(2,-1); z=z*(1+g.unsqueeze(1))+b.unsqueeze(1)
        return ((z*s.hd).sum(-1)+s.hb).mean(1)

def prep(cols):
    Z=np.nan_to_num(cols.values.astype(np.float32),nan=0.,posinf=0.,neginf=0.)
    return np.clip((Z-Z.mean(0))/(Z.std(0)+1e-6),-6,6)
mth=X98.game_month.values.astype(np.float32)
TF=torch.tensor(np.stack([np.sin(2*np.pi*mth/12),np.cos(2*np.pi*mth/12),
    np.sin(4*np.pi*mth/12),np.cos(4*np.pi*mth/12),(season-2019)/6.0],1).astype(np.float32),device=DEV)
TY=torch.tensor(y,device=DEV)

def train_nn(TN, trmask, vamask, k=32,h=256,L=3,plr=True,lr=2e-3,wd=1e-4,seed=0,budget=1500):
    ia=np.where(trmask)[0]; rs=np.random.RandomState(1); rs.shuffle(ia)
    nin=int(len(ia)*.06); iin,itr=ia[:nin],ia[nin:]; iva=np.where(vamask)[0]
    torch.manual_seed(seed)
    net=TabM(TN.shape[1],k=k,h=h,L=L,plr=plr).to(DEV)
    d_in=TN.shape[1]*(7 if plr else 1)
    Bs=int(np.clip(6.0e8/max(k*max(h,d_in)*(L+1),1),2048,32768))
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=wd)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80)
    best,bad,bst=1e9,0,None; t0=time.time()
    for ep in range(80):
        net.train(); perm=np.random.RandomState(ep).permutation(itr)
        for j in range(0,len(perm),Bs):
            b=torch.tensor(perm[j:j+Bs],device=DEV)
            loss=((torch.sigmoid(net(TN[b],TF[b]))-TY[b])**2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            se,c=0.,0
            for q in range(0,len(iin),Bs):
                b=torch.tensor(iin[q:q+Bs],device=DEV)
                se+=((torch.sigmoid(net(TN[b],TF[b]))-TY[b])**2).sum().item(); c+=len(b)
            v=se/c
        if v<best-1e-8: best,bad=v,0; bst={q:t.detach().clone() for q,t in net.state_dict().items()}
        else:
            bad+=1
            if bad>=6: break
        if time.time()-t0>budget: break
    net.load_state_dict(bst); net.eval(); ps=[]
    with torch.no_grad():
        for j in range(0,len(iva),Bs):
            b=torch.tensor(iva[j:j+Bs],device=DEV)
            ps.append(torch.sigmoid(net(TN[b],TF[b])).cpu().numpy())
    return np.concatenate(ps)

# ═══════ C. 트랙맨을 NN 에 ═══════
def stageC():
    log(f'\n{el()} ===== C. 트랙맨을 NN 에 =====')
    base=[c for c in CORE if c not in CAT]
    TN_no=torch.tensor(prep(X98[base]),device=DEV)
    TN_tm=torch.tensor(prep(pd.concat([X98[base],FE[TMC]],axis=1)),device=DEV)
    for name,TN in [('NN 트랙맨없음',TN_no),('NN 트랙맨포함',TN_tm)]:
        try:
            sc={}; pv={}
            for vs in FOLDS:
                tr,va=split(vs)
                p=train_nn(TN,tr,va); pv[vs]=p; sc[vs]=bss(p,vs)
                np.save(OUT+f'nnC_{name.replace(" ","_")}_{vs}.npy',p.astype(np.float32))
            gains={w:np.mean([bss(sp.expit((1-w)*lg_(XGP[v])+w*lg_(pv[v])),v)-B0[v] for v in FOLDS])
                   for w in [0.15,0.2,0.25,0.3]}
            bw=max(gains,key=gains.get)
            rec(stage='C',name=name,solo24=sc[2024],solo22=sc[2022],
                solo=np.mean(list(sc.values())),best_w=bw,gain=gains[bw],nfeat=TN.shape[1])
            log(f'{el()} {name:14s}({TN.shape[1]:3d}) 단독 24:{sc[2024]:7.1f} 22:{sc[2022]:7.1f} | 블렌딩이득 {gains[bw]:+6.1f} (w={bw})')
        except Exception: log('!! '+name+'\n'+traceback.format_exc())

# ═══════ B. 스태킹 ═══════
def stageB():
    log(f'\n{el()} ===== B. 스태킹 (NN 예측을 XGB 피처로) =====')
    base=[c for c in CORE if c not in CAT]
    TN=torch.tensor(prep(X98[base]),device=DEV)
    # OOF: 시즌 s 예측은 s 이전 시즌들로만 학습한 NN 으로
    oof=np.full(len(X98),np.nan,np.float32)
    for s in [2021,2022,2023,2024]:
        tr=(season<s)&~(isF&(season<=2022)&(s>=2023)); va=(season==s)
        if not tr.any() or not va.any(): continue
        oof[va]=train_nn(TN,tr,va,budget=900)
        log(f'{el()}   OOF 시즌{s} 완료 (학습 {tr.sum():,}행)')
    np.save(OUT+'nn_oof.npy',oof)
    log(f'{el()}   OOF 결측률 {np.isnan(oof).mean():.3f} (2019-20 은 이전데이터 없어 결측)')
    Xs=XK.copy(); Xs['nn_oof']=sp.logit(np.clip(oof,1e-6,1-1e-6))
    for vs in FOLDS:
        tr,va=split(vs)
        p=np.mean([xgb.XGBClassifier(**XGPRM,random_state=s).fit(Xs.loc[tr],y[tr])
                   .predict_proba(Xs.loc[va])[:,1] for s in range(3)],0)
        g=bss(p,vs)-B0[vs]
        rec(stage='B',name='스태킹',val=vs,score=bss(p,vs),gain=g)
        log(f'{el()} 스태킹 폴드{vs}: {bss(p,vs):7.1f}  (XGB단독 {B0[vs]:7.1f}, 이득 {g:+6.1f})')
        # 참고: 같은 NN 을 블렌딩했을 때
        nnv=oof[(season==vs)&~isF] if vs!=2019 else None
        if nnv is not None and not np.isnan(nnv).all():
            gb=max(bss(sp.expit((1-w)*lg_(XGP[vs])+w*lg_(nnv)),vs)-B0[vs] for w in [0.15,0.2,0.25,0.3])
            log(f'{el()}    (동일 NN 블렌딩 시 이득 {gb:+6.1f})')

for fn in (stageA, stageC, stageB):
    try: fn()
    except Exception: log(f'!! {fn.__name__}\n'+traceback.format_exc())
log(f'\n{el()} ===== 완료 =====')
