"""
run_nn_pure — 트리 완전 제거 순수 NN 파이프라인
  베이스 118피처 = X98 CORE(98) + multi_k 8 + TM 8 + pbc 4  (OOF 6 없음)
  MTabM 멀티태스크 (y + 4성분) trunk 공유, 홀드아웃 6% early-stop
  두 폴드(2024,2023) 동시 실행: LGA_DEV=cuda:0/1 병렬
"""
import os, sys, json, time, warnings, argparse
import numpy as np, pandas as pd, scipy.special as sp
import torch, torch.nn as nn
warnings.filterwarnings('ignore')

D = '/home/lee/lga/'
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=2024, choices=[2024,2023])
parser.add_argument('--dev', type=str, default='cuda:0')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--patience', type=int, default=6)
parser.add_argument('--bs', type=int, default=2048)
parser.add_argument('--lr', type=float, default=0.0008)
args = parser.parse_args()
DEV = args.dev
FOLD = args.fold

OUT = D + 'results_nn_pure/'
os.makedirs(OUT, exist_ok=True)
LOGF = open(f"{OUT}/log_{FOLD}.txt", 'w', buffering=1)
def log(*a):
    m = ' '.join(str(x) for x in a)
    print(m, flush=True); LOGF.write(m+'\n')
T0=time.time()
def el(): return f"[{(time.time()-T0)/60:5.1f}m]"

log(f"{el()} === 트리제거 순수 NN vs={FOLD} dev={DEV} ===")
log(f"{el()} 피처: 118 (base114 + pbc4) OOF 없음")

# ── 로딩 ──
RAW = pd.read_csv(D+'data/train.csv', encoding='utf-8-sig')
X98 = pd.read_parquet(D+'X98.parquet')
TM = pd.read_parquet(D+'results14/tm5.parquet')
y = X98.__y.values.astype(np.float32)
season = X98.__season.values
isF = X98.__F.values.astype(bool)
CORE = [c for c in X98.columns if not c.startswith('__')]
TMSEL = json.load(open(D+'v6_tmsel.json'))

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

# base114 트리프리
BASE114 = pd.concat([X98[CORE], multi_k(), TM[TMSEL]], axis=1)
log(f"{el()} BASE114 {BASE114.shape}")

# pbc 4 계층
pid = RAW.pitcher_id.values.astype(np.int64)
bh = RAW.batter_hand.values.astype(np.int64)
cnt = RAW.balls_before.values.astype(np.int64)*10 + RAW.strikes_before.values.astype(np.int64)
key = pid*1000 + bh*100 + cnt
F_b3 = {c: np.full(len(RAW), np.nan, np.float32) for c in ['pbc_rate','pbc_n','pbc_logn','pbc_delta']}
for s_ in range(2020,2026):
    tgt=(season==s_); prev=(season < s_)
    if not tgt.any() or not prev.any(): continue
    mu=float(y[prev].mean())
    gk=pd.Series(y[prev]).groupby(key[prev]).agg(['sum','size'])
    gp=pd.Series(y[prev]).groupby(pid[prev]).mean()
    gl=pd.Series(y[prev]).groupby(cnt[prev]).mean()
    kt,nan=pd.Series(key[tgt]),None
    # map via dict for speed
    gk_size=gk['size'].to_dict(); gk_sum=gk['sum'].to_dict(); gp_d=gp.to_dict(); gl_d=gl.to_dict()
    n=np.array([gk_size.get(k,0) for k in key[tgt]],dtype=np.float64)
    sy=np.array([gk_sum.get(k,0) for k in key[tgt]],dtype=np.float64)
    pa=np.array([gp_d.get(p,mu) for p in pid[tgt]],dtype=np.float64)
    pb=np.array([gl_d.get(c,mu) for c in cnt[tgt]],dtype=np.float64)
    rate=(sy+50*pa+50*pb)/(n+100)
    F_b3['pbc_rate'][tgt]=rate.astype(np.float32)
    F_b3['pbc_n'][tgt]=n.astype(np.float32)
    F_b3['pbc_logn'][tgt]=np.log1p(n).astype(np.float32)
    F_b3['pbc_delta'][tgt]=(rate-pa).astype(np.float32)

X118 = pd.concat([BASE114, pd.DataFrame(F_b3,index=RAW.index).astype(np.float32)],axis=1)
log(f"{el()} X118 {X118.shape} cols={list(X118.columns[-6:])}")

# ── 성분 라벨 역산 (멀티태스크용, 소스는 RAW 누적 그대로 - 트리 아님) ──
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
CL=np.vstack([LB[c] for c in COMP]).T  # (N,4) nan 포함
log(f"{el()} 성분라벨 역산 완료")

# ── 폴드 정의 (lib_lga.split과 동일) ──
def split(vs):
    tr=(season<vs) & ~(isF & (season<=2022) & (vs>=2023))
    va=(season==vs) & ~isF
    return tr,va
tr_all,va = split(FOLD)
y_np=y; season_np=season
# 가중 최근성 hl2.0
hl=2.0
w_tr=(0.5**((FOLD-1 - season_np[tr_all])/hl)).astype(np.float32)

# 내부 홀드아웃 6% (ANTIGRAVITY 규약)
rng=np.random.RandomState(FOLD)
idx_tr=np.where(tr_all)[0]
rng.shuffle(idx_tr)
n_hold=int(len(idx_tr)*0.06)
idx_hold=idx_tr[:n_hold]
idx_train=idx_tr[n_hold:]
log(f"{el()} 폴드{FOLD} train {len(idx_train):,} hold {len(idx_hold):,} val {va.sum():,}")

# ── 입력 준비 ──
CAT=['pitcher_id','batter_id','pitcher_team_id','batter_team_id','pitcher_hand','batter_hand','base_state','game_type','top_bottom']
NUM=[c for c in X118.columns if c not in CAT]
# 정규화: train 기준
Z=X118[NUM].values.astype(np.float32)
mu=Z[idx_train].mean(0); sd=Z[idx_train].std(0)+1e-6
Z=(Z-mu)/sd; Z=np.clip(Z,-6,6); Z=np.nan_to_num(Z,nan=0.)
# cat 카드 - 전역 max로 계산 (val 미노출 방지)
Xc=X118[CAT].values.astype(np.int64)
Xc=np.maximum(Xc,0)
CARD=[int(Xc[:,i].max())+2 for i in range(len(CAT))]
# time feat
mth=RAW.game_month.values.astype(np.float32)
TT=np.stack([np.sin(2*np.pi*mth/12),np.cos(2*np.pi*mth/12),np.sin(4*np.pi*mth/12),np.cos(4*np.pi*mth/12),(season_np-2019)/6.0],1).astype(np.float32)

# 텐서
device=torch.device(DEV if torch.cuda.is_available() else 'cpu')
log(f"{el()} device={device} CARD={CARD[:3]} NUM={len(NUM)}")

# ── 모델 정의 (MTabM 순수) ──
class PLR(nn.Module):
    def __init__(s,d_in,k=8,d=6):
        super().__init__(); s.c=nn.Parameter(torch.randn(d_in,k)*.05); s.l=nn.Linear(2*k,d)
    def forward(s,x):
        z=2*np.pi*x.unsqueeze(-1)*s.c
        return torch.relu(s.l(torch.cat([torch.sin(z),torch.cos(z)],-1))).flatten(1)
class MTabM(nn.Module):
    def __init__(s,n,card,k=32,h=256,L=3,emb=16,drop=.1,plr=True,film=True,nh=5):
        super().__init__(); s.k,s.emb,s.film,s.plr,s.nh=k,emb,film,None,nh; d=n
        if plr: s.plr=PLR(n); d+=n*6
        if emb:
            s.es=nn.ModuleList([nn.Embedding(c,emb) for c in card]); d+=len(card)*emb
        s.r1=nn.Parameter(torch.randn(k,d)*.1+1)
        s.ls=nn.ModuleList([nn.Linear(d if i==0 else h, h) for i in range(L)])
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
        return ((z.unsqueeze(1)*s.hd).sum(-1)+s.hb).mean(-1)  # (B,nh)

# ── 데이터로더 ──
import torch.utils.data as tud
class DS(tud.Dataset):
    def __init__(s,idx): 
        s.idx=idx
        s.w = w_tr[np.searchsorted(idx_tr[n_hold:], idx)] if len(idx)==len(idx_train) else np.ones(len(idx),dtype=np.float32) # hold는 weight 1
        # simpler: map via dict
        s.wmap=dict(zip(idx_tr, w_tr))
    def __len__(s): return len(s.idx)
    def __getitem__(s,i):
        j=s.idx[i]
        # y + 4성분 (nan은 -1 마스크)
        y0=float(y_np[j])
        comps=np.array([LB[c][j] for c in COMP],dtype=np.float32)
        mask=np.isfinite(comps).astype(np.float32)
        comps=np.nan_to_num(comps,nan=0.)
        w=float(s.wmap.get(j,1.0))
        return Z[j], Xc[j], TT[j], y0, comps, mask, w

# faster: precompute w array
Warr=np.ones(len(y_np),dtype=np.float32); Warr[idx_tr]=w_tr
class DS2(tud.Dataset):
    def __init__(s,idx): s.idx=idx
    def __len__(s): return len(s.idx)
    def __getitem__(s,i):
        j=s.idx[i]
        y0=float(y_np[j])
        comps=np.array([LB[c][j] for c in COMP],dtype=np.float32)
        mask=np.isfinite(comps).astype(np.float32)
        comps=np.nan_to_num(comps,nan=0.)
        w=float(Warr[j])
        return Z[j], Xc[j], TT[j], y0, comps, mask, w

train_ds=DS2(idx_train); hold_ds=DS2(idx_hold)
train_loader=tud.DataLoader(train_ds,batch_size=args.bs,shuffle=True,num_workers=4,pin_memory=True,drop_last=False)
hold_loader=tud.DataLoader(hold_ds,batch_size=args.bs*2,shuffle=False,num_workers=2)

def bss(p, yv, base):
    return 100000*max(0.,1 - np.mean((p-yv)**2)/base)
# val base
yv=y_np[va]; base=yv.mean()*(1-yv.mean())
log(f"{el()} val base {base:.5f} mean {yv.mean():.4f}")

model=MTabM(len(NUM),CARD,k=32,h=256,L=3,emb=16,drop=.1,plr=True,film=True,nh=5).to(device)
opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.02)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)

best_loss=float('inf'); best_state=None; pat=0
for epoch in range(1,args.epochs+1):
    model.train(); tr_loss=0; n=0
    for xn,xc,tt,y0,comps,mask,w in train_loader:
        xn=xn.to(device); xc=xc.to(device); tt=tt.to(device); y0=y0.to(device); comps=comps.to(device); mask=mask.to(device); w=w.to(device)
        opt.zero_grad()
        out=model(xn,xc,tt)  # (B,5)
        # y head 0, comp 1-4
        bce_y=nn.functional.binary_cross_entropy_with_logits(out[:,0], y0, reduction='none')
        loss_y=(bce_y*w).mean()
        # comp loss (only where mask=1)
        # comps shape (B,4) align with out[:,1:5]
        bce_c=nn.functional.binary_cross_entropy_with_logits(out[:,1:5], comps, reduction='none')
        # mask 적용
        loss_c=(bce_c*mask).sum() / (mask.sum()+1e-6)
        lam=0.6  # comp 가중
        loss=loss_y + lam*loss_c
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tr_loss+=loss.item()*len(xn); n+=len(xn)
    sched.step()
    tr_loss/=n
    # hold
    model.eval()
    hold_loss=0; hn=0
    with torch.no_grad():
        for xn,xc,tt,y0,comps,mask,w in hold_loader:
            xn=xn.to(device); xc=xc.to(device); tt=tt.to(device); y0=y0.to(device); comps=comps.to(device); mask=mask.to(device); w=w.to(device)
            out=model(xn,xc,tt)
            bce_y=nn.functional.binary_cross_entropy_with_logits(out[:,0], y0, reduction='none')
            loss_y=(bce_y*w).mean()
            bce_c=nn.functional.binary_cross_entropy_with_logits(out[:,1:5], comps, reduction='none')
            loss_c=(bce_c*mask).sum()/(mask.sum()+1e-6)
            loss=loss_y+0.6*loss_c
            hold_loss+=loss.item()*len(xn); hn+=len(xn)
    hold_loss/=hn
    is_best=hold_loss < best_loss
    if is_best:
        best_loss=hold_loss; best_state={k:v.cpu() for k,v in model.state_dict().items()}; pat=0
        mark="★"
    else:
        pat+=1; mark=f"({pat}/{args.patience})"
    log(f"{el()} Epoch {epoch}/{args.epochs} tr {tr_loss:.5f} hold {hold_loss:.5f} {mark} lr {opt.param_groups[0]['lr']:.2e}")
    if pat>=args.patience:
        log(f"{el()} early stop at {epoch}")
        break

if best_state is not None:
    model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
# val 예측
model.eval()
with torch.no_grad():
    # batch infer
    N=len(y_np)
    preds=np.full(N, np.nan, dtype=np.float32)
    # only val needed but do full val set
    val_idx=np.where(va)[0]
    # chunk
    bs=args.bs*2
    all_p=[]
    for s in range(0,len(val_idx),bs):
        idx=val_idx[s:s+bs]
        xn=torch.tensor(Z[idx],device=device)
        xc=torch.tensor(Xc[idx],device=device)
        tt=torch.tensor(TT[idx],device=device)
        out=model(xn,xc,tt)[:,0]
        p=torch.sigmoid(out).cpu().numpy()
        all_p.append(p)
    p_val=np.concatenate(all_p)
    score=bss(p_val, yv, base)
    log(f"{el()} VAL BSS {score:.1f} (base {base:.5f})")
    # 저장
    np.save(f"{OUT}/pred_{FOLD}.npy", p_val.astype(np.float32))
    torch.save(best_state, f"{OUT}/model_{FOLD}.pt")
    # 정규화 통계 저장
    import json
    np.savez(f"{OUT}/norm_{FOLD}.npz", mu=mu, sd=sd)
    json.dump({"NUM":NUM,"CAT":CAT,"CARD":CARD,"best_hold":float(best_loss),"val_bss":float(score)}, open(f"{OUT}/meta_{FOLD}.json","w"))
    log(f"{el()} saved pred/model {OUT}")
