"""
12차 — FTT 의 폴드별 부호 반전이 '학습 데이터 크기' 때문인지 판별

관측: FTT 블렌딩 이득이 폴드2022(학습3시즌) -15.3, 폴드2024(학습5시즌) +20.1
가설: 트랜스포머는 데이터를 많이 먹는다. 실제 제출은 6시즌이므로 2024 쪽 거동이 맞다.
검증: 중간점 폴드2023(학습 4시즌)을 찍는다. 단조 증가면 가설 성립.
      추가로 폴드2024 를 학습량 축소(최근 3/4/5시즌)로 재현해 직접 확인.
"""
import os, time, warnings, traceback
import numpy as np, pandas as pd, scipy.special as sp
import torch, torch.nn as nn, xgboost as xgb
warnings.filterwarnings('ignore')
DEV='cuda:0'
D='/home/lee/lga/'; OUT=D+'results12/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'

RAW=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig')
X=pd.read_parquet(D+'X98.parquet')
y=X.__y.values.astype(np.float32); season=X.__season.values; isF=X.__F.values.astype(bool)
CORE=[c for c in X.columns if not c.startswith('__')]
def bss(p,vs,mask=None):
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
TY=torch.tensor(y,device=DEV); NN_=Z.shape[1]

class FTT(nn.Module):
    def __init__(s,n,card,d=64,L=2,heads=4,drop=.1,ncat=4):
        super().__init__()
        s.w=nn.Parameter(torch.randn(n,d)*.05); s.b=nn.Parameter(torch.zeros(n,d))
        s.ce=nn.ModuleList([nn.Embedding(card[i],d) for i in range(2,2+ncat)])
        s.cls=nn.Parameter(torch.randn(1,1,d)*.05)
        s.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d,heads,d*2,drop,batch_first=True,norm_first=True,activation='gelu'),L)
        s.tm=nn.Sequential(nn.Linear(5,d),nn.GELU()); s.out=nn.Linear(d,1)
    def forward(s,xn,xc,tt):
        tok=xn.unsqueeze(-1)*s.w+s.b
        ct=torch.stack([e(xc[:,i+2]) for i,e in enumerate(s.ce)],1)
        z=torch.cat([s.cls.expand(xn.shape[0],-1,-1),s.tm(tt).unsqueeze(1),tok,ct],1)
        return s.out(s.tr(z)[:,0]).squeeze(-1)

def train_ftt(trmask, vamask, seed=0, d=64, L=2, wd=1e-3):
    ia=np.where(trmask)[0]; rs=np.random.RandomState(1); rs.shuffle(ia)
    nin=int(len(ia)*.06); iin,itr=ia[:nin],ia[nin:]; iva=np.where(vamask)[0]
    torch.manual_seed(seed)
    net=FTT(NN_,CARD,d=d,L=L).to(DEV)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=wd)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40)
    Bs=4096; best,bad,bst=1e9,0,None; t0=time.time()
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
        if time.time()-t0>1500: break
    net.load_state_dict(bst); net.eval(); ps=[]
    with torch.no_grad():
        for j in range(0,len(iva),Bs):
            b=torch.tensor(iva[j:j+Bs],device=DEV)
            ps.append(torch.sigmoid(net(TN[b],TC[b],TT[b])).cpu().numpy())
    return np.concatenate(ps), ep+1

PRM=dict(n_estimators=600,learning_rate=0.008,max_depth=6,min_child_weight=1500,subsample=0.7,
         colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,tree_method='hist',device=DEV,
         eval_metric='logloss',verbosity=0)

log(f'{el()} ===== 실험 1: 폴드별 (학습 시즌 수가 자연히 달라짐) =====')
res=[]
for vs in [2022,2023,2024]:
    tr=(season<vs)&~(isF&(season<=2022)&(vs>=2023)); va=(season==vs)&~isF
    nseas=len(np.unique(season[tr]))
    px=np.mean([xgb.XGBClassifier(**PRM,random_state=s).fit(XK.loc[tr],y[tr])
                .predict_proba(XK.loc[va])[:,1] for s in range(3)],0)
    b0=bss(px,vs)
    pn,ep=train_ftt(tr,va)
    row=dict(fold=vs,nseas=nseas,ntrain=int(tr.sum()),xgb=b0,ftt_solo=bss(pn,vs),ep=ep)
    for w in [0.2,0.3]:
        row[f'gain_w{w}']=bss(sp.expit((1-w)*lg_(px)+w*lg_(pn)),vs)-b0
    res.append(row); pd.DataFrame(res).to_csv(OUT+'fold_size.csv',index=False)
    log(f'{el()} 폴드{vs} 학습{nseas}시즌 {tr.sum():,}행 | XGB {b0:6.1f} | FTT단독 {row["ftt_solo"]:6.1f} '
        f'| 이득 w0.2 {row["gain_w0.2"]:+6.1f}  w0.3 {row["gain_w0.3"]:+6.1f}')

log(f'\n{el()} ===== 실험 2: 폴드2024 고정, 학습량만 축소 =====')
res2=[]
for start in [2021,2020,2019]:
    tr=(season>=start)&(season<2024)&~(isF&(season<=2022)); va=(season==2024)&~isF
    nseas=len(np.unique(season[tr]))
    px=np.mean([xgb.XGBClassifier(**PRM,random_state=s).fit(XK.loc[tr],y[tr])
                .predict_proba(XK.loc[va])[:,1] for s in range(3)],0)
    b0=bss(px,2024); pn,ep=train_ftt(tr,va)
    row=dict(start=start,nseas=nseas,ntrain=int(tr.sum()),xgb=b0,ftt_solo=bss(pn,2024))
    for w in [0.2,0.3]:
        row[f'gain_w{w}']=bss(sp.expit((1-w)*lg_(px)+w*lg_(pn)),2024)-b0
    res2.append(row); pd.DataFrame(res2).to_csv(OUT+'trainsize.csv',index=False)
    log(f'{el()} 학습 {start}~2023 ({nseas}시즌 {tr.sum():,}행) | XGB {b0:6.1f} | FTT단독 {row["ftt_solo"]:6.1f} '
        f'| 이득 w0.2 {row["gain_w0.2"]:+6.1f}  w0.3 {row["gain_w0.3"]:+6.1f}')
log(f'\n{el()} ===== 완료 =====')
