"""
run_ft_pure — 트리 제거 FT-Transformer 순수
  118피처 동일, FTT 토큰 attention, 홀드아웃 6%
"""
import os, sys, json, time, warnings, argparse
import numpy as np, pandas as pd, scipy.special as sp
import torch, torch.nn as nn
warnings.filterwarnings('ignore')
D='/home/lee/lga/'
parser=argparse.ArgumentParser()
parser.add_argument('--fold',type=int,default=2024,choices=[2024,2023])
parser.add_argument('--dev',type=str,default='cuda:0')
parser.add_argument('--epochs',type=int,default=40)
parser.add_argument('--patience',type=int,default=6)
parser.add_argument('--bs',type=int,default=1024)
parser.add_argument('--lr',type=float,default=0.0005)
args=parser.parse_args()
DEV=args.dev; FOLD=args.fold
OUT=D+'results_ft_pure/'
os.makedirs(OUT,exist_ok=True)
LOGF=open(f"{OUT}/log_{FOLD}.txt",'w',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOGF.write(m+'\n')
T0=time.time()
def el(): return f"[{(time.time()-T0)/60:5.1f}m]"
log(f"{el()} === FT pure vs={FOLD} dev={DEV} ===")
RAW=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig')
X98=pd.read_parquet(D+'X98.parquet')
TM=pd.read_parquet(D+'results14/tm5.parquet')
y=X98.__y.values.astype(np.float32); season=X98.__season.values; isF=X98.__F.values.astype(bool)
CORE=[c for c in X98.columns if not c.startswith('__')]
TMSEL=json.load(open(D+'v6_tmsel.json'))
def multi_k():
    F={}
    for idc,nc,rc,pf in [('pitcher_id','asof_pitcher_n','asof_pitcher_success_rate','p_succ'),('batter_id','asof_batter_n','asof_batter_success_rate','b_succ')]:
        t=RAW[[idc,'season',nc,rc]].copy(); t['succ']=t[nc]*t[rc].fillna(0)
        S=t.loc[t.groupby([idc,'season'])[nc].idxmin()].set_index([idc,'season'])[[nc,'succ']]
        a=RAW[[idc,'season']].join(S,on=[idc,'season'])
        dn=np.maximum(RAW[nc].values-a[nc].fillna(0).values,0); ds=np.maximum(np.nan_to_num(RAW[nc].values*RAW[rc].values)-a['succ'].fillna(0).values,0)
        lgv=np.nanmean(RAW[rc])
        for k in [25,75,400,1000]: F[f'{pf}_k{k}']=(ds+k*lgv)/(dn+k)
    return pd.DataFrame(F,index=RAW.index).astype(np.float32)
BASE114=pd.concat([X98[CORE],multi_k(),TM[TMSEL]],axis=1)
pid=RAW.pitcher_id.values.astype(np.int64); bh=RAW.batter_hand.values.astype(np.int64); cnt=RAW.balls_before.values.astype(np.int64)*10+RAW.strikes_before.values.astype(np.int64); key=pid*1000+bh*100+cnt
F_b3={c: np.full(len(RAW),np.nan,np.float32) for c in ['pbc_rate','pbc_n','pbc_logn','pbc_delta']}
for s_ in range(2020,2026):
    tgt=(season==s_); prev=(season < s_)
    if not tgt.any() or not prev.any(): continue
    mu=float(y[prev].mean()); gk=pd.Series(y[prev]).groupby(key[prev]).agg(['sum','size']); gp=pd.Series(y[prev]).groupby(pid[prev]).mean(); gl=pd.Series(y[prev]).groupby(cnt[prev]).mean()
    gk_size=gk['size'].to_dict(); gk_sum=gk['sum'].to_dict(); gp_d=gp.to_dict(); gl_d=gl.to_dict()
    n=np.array([gk_size.get(k,0) for k in key[tgt]],dtype=np.float64); sy=np.array([gk_sum.get(k,0) for k in key[tgt]],dtype=np.float64)
    pa=np.array([gp_d.get(p,mu) for p in pid[tgt]],dtype=np.float64); pb=np.array([gl_d.get(c,mu) for c in cnt[tgt]],dtype=np.float64)
    rate=(sy+50*pa+50*pb)/(n+100)
    F_b3['pbc_rate'][tgt]=rate.astype(np.float32); F_b3['pbc_n'][tgt]=n.astype(np.float32); F_b3['pbc_logn'][tgt]=np.log1p(n).astype(np.float32); F_b3['pbc_delta'][tgt]=(rate-pa).astype(np.float32)
X118=pd.concat([BASE114,pd.DataFrame(F_b3,index=RAW.index).astype(np.float32)],axis=1)
log(f"{el()} X118 {X118.shape}")
def split(vs): return (season<vs)&~(isF&(season<=2022)&(vs>=2023)), (season==vs)&~isF
tr_all,va=split(FOLD)
hl=2.0; w_tr=(0.5**((FOLD-1-season[tr_all])/hl)).astype(np.float32)
rng=np.random.RandomState(FOLD); idx_tr=np.where(tr_all)[0]; rng.shuffle(idx_tr); n_hold=int(len(idx_tr)*0.06); idx_hold=idx_tr[:n_hold]; idx_train=idx_tr[n_hold:]
log(f"{el()} train {len(idx_train):,} hold {len(idx_hold):,} val {va.sum():,}")
CAT=['pitcher_id','batter_id','pitcher_team_id','batter_team_id','pitcher_hand','batter_hand','base_state','game_type','top_bottom']
NUM=[c for c in X118.columns if c not in CAT]
Z=X118[NUM].values.astype(np.float32); mu=Z[idx_train].mean(0); sd=Z[idx_train].std(0)+1e-6; Z=(Z-mu)/sd; Z=np.clip(Z,-6,6); Z=np.nan_to_num(Z,nan=0.)
Xc=X118[CAT].values.astype(np.int64); Xc=np.maximum(Xc,0)
CARD=[int(Xc[:,i].max())+2 for i in range(len(CAT))]
mth=RAW.game_month.values.astype(np.float32); TT=np.stack([np.sin(2*np.pi*mth/12),np.cos(2*np.pi*mth/12),np.sin(4*np.pi*mth/12),np.cos(4*np.pi*mth/12),(season-2019)/6.0],1).astype(np.float32)
device=torch.device(DEV if torch.cuda.is_available() else 'cpu')
log(f"{el()} device={device} NUM={len(NUM)} CARD={CARD}")

# FTT 118토큰용
class FTT(nn.Module):
    def __init__(s,n,card,d=32,L=3,heads=4,drop=.1):
        super().__init__()
        s.w=nn.Parameter(torch.randn(n,d)*.05); s.b=nn.Parameter(torch.zeros(n,d))
        s.ce=nn.ModuleList([nn.Embedding(c,d) for c in card])
        s.cls=nn.Parameter(torch.randn(1,1,d)*.05)
        s.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,heads,d*2,drop,batch_first=True,norm_first=True),L)
        s.out=nn.Linear(d,1)
    def forward(s,xn,xc,tt): # tt not used but keep signature
        tok=xn.unsqueeze(-1)*s.w + s.b
        ct=torch.stack([e(xc[:,i]) for i,e in enumerate(s.ce)],1)
        z=torch.cat([s.cls.expand(xn.shape[0],-1,-1),tok,ct],1)
        return s.out(s.tr(z)[:,0]).squeeze(-1)

Warr=np.ones(len(y),dtype=np.float32); Warr[idx_tr]=w_tr
import torch.utils.data as tud
class DS2(tud.Dataset):
    def __init__(s,idx): s.idx=idx
    def __len__(s): return len(s.idx)
    def __getitem__(s,i):
        j=s.idx[i]; return Z[j],Xc[j],TT[j],float(y[j]),float(Warr[j])
train_ds=DS2(idx_train); hold_ds=DS2(idx_hold)
train_loader=tud.DataLoader(train_ds,batch_size=args.bs,shuffle=True,num_workers=4,pin_memory=True)
hold_loader=tud.DataLoader(hold_ds,batch_size=args.bs*2,shuffle=False,num_workers=2)
def bss(p,yv,base): return 100000*max(0.,1-np.mean((p-yv)**2)/base)
yv=y[va]; base=yv.mean()*(1-yv.mean())
log(f"{el()} base {base:.5f}")
model=FTT(len(NUM),CARD,d=32,L=3,heads=4,drop=.1).to(device)
opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
best_loss=float('inf'); best_state=None; pat=0
for epoch in range(1,args.epochs+1):
    model.train(); tr=0; n=0
    for xn,xc,tt,y0,w in train_loader:
        xn=xn.to(device); xc=xc.to(device); tt=tt.to(device); y0=y0.to(device); w=w.to(device)
        opt.zero_grad(); out=model(xn,xc,tt)
        loss=nn.functional.binary_cross_entropy_with_logits(out,y0,reduction='none'); loss=(loss*w).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tr+=loss.item()*len(xn); n+=len(xn)
    tr/=n; sched.step()
    model.eval(); hl=0; hn=0
    with torch.no_grad():
        for xn,xc,tt,y0,w in hold_loader:
            xn=xn.to(device); xc=xc.to(device); tt=tt.to(device); y0=y0.to(device); w=w.to(device)
            out=model(xn,xc,tt)
            loss=nn.functional.binary_cross_entropy_with_logits(out,y0,reduction='none'); loss=(loss*w).mean()
            hl+=loss.item()*len(xn); hn+=len(xn)
    hl/=hn; is_best=hl<best_loss
    if is_best: best_loss=hl; best_state={k:v.cpu() for k,v in model.state_dict().items()}; pat=0; mark="★"
    else: pat+=1; mark=f"({pat}/{args.patience})"
    log(f"{el()} Epoch {epoch}/{args.epochs} tr {tr:.5f} hold {hl:.5f} {mark} lr {opt.param_groups[0]['lr']:.2e}")
    if pat>=args.patience: log(f"{el()} early stop"); break
if best_state is not None: model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
model.eval()
with torch.no_grad():
    val_idx=np.where(va)[0]; all=[]
    for s in range(0,len(val_idx),args.bs*2):
        idx=val_idx[s:s+args.bs*2]
        xn=torch.tensor(Z[idx],device=device); xc=torch.tensor(Xc[idx],device=device); tt=torch.tensor(TT[idx],device=device)
        all.append(torch.sigmoid(model(xn,xc,tt)).cpu().numpy())
    p_val=np.concatenate(all); score=bss(p_val,yv,base)
    log(f"{el()} VAL BSS {score:.1f}")
    np.save(f"{OUT}/pred_{FOLD}.npy",p_val.astype(np.float32))
    torch.save(best_state,f"{OUT}/model_{FOLD}.pt")
    np.savez(f"{OUT}/norm_{FOLD}.npz",mu=mu,sd=sd)
    import json; json.dump({"NUM":NUM,"CARD":CARD,"val_bss":float(score)},open(f"{OUT}/meta_{FOLD}.json","w"))
    log(f"{el()} saved")
