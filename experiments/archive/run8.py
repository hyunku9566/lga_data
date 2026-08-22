"""
8차 — (A) 시즌별 능력 이력 복원  (B) 다중 축소강도  (C) NN 블렌딩

핵심 아이디어(A):
  asof_* 는 커리어 누적이고, 각 시즌 첫 행의 asof 값 = 그 시즌 시작 시점의 누적 상태.
  따라서 시즌 s 성적 = start[s+1] - start[s] 로 '연도별 제구율'이 정확히 복원된다.
  지금까지는 '커리어 평균 1개 + 올해 1개'로 뭉개 썼다.
  -> 시간감쇠 가중 능력치 / 추세 / 연도간 변동성 을 만든다.
  전부 (현재 행 + train 유래 룩업) 이라 규칙 준수.
"""
import os, time, warnings, traceback
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results8/'
os.makedirs(OUT, exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'

RAW=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig')
X0=pd.read_parquet(D+'X98.parquet')
y=X0.__y.values; season=X0.__season.values; isF=X0.__F.values.astype(bool)
CORE=[c for c in X0.columns if not c.startswith('__')]
FOLDS=[2024,2022]
def tp(vs):
    m=season<vs; s=pd.Series(y[m]).groupby(season[m]).mean()
    return float(sp.expit(np.polyval(np.polyfit(s.index,sp.logit(s.values),1),vs)))
RP={v:tp(v) for v in FOLDS}
def ev(p,yv,rp):
    r=yv.mean(); ref=r*(1-r); b=lambda q:100000*max(0.,1-np.mean((q-yv)**2)/ref)
    lo=sp.logit(np.clip(p,1e-6,1-1e-6))
    return dict(raw=b(p),trend=b(sp.expit(lo-lo.mean()+sp.logit(rp))),oracle=b(sp.expit(lo-lo.mean()+sp.logit(r))))
PRM=dict(n_estimators=600,learning_rate=0.008,max_depth=6,min_child_weight=1500,
         subsample=0.7,colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,
         tree_method='hist',device='cuda:1',n_jobs=8,eval_metric='logloss',verbosity=0)
NSEED=4
R=[]
def bench(Xa,name,keep=False):
    o={}; preds={}
    for vs in FOLDS:
        tr=(season<vs)&~(isF&(season<=2022)&(vs>=2023)); va=(season==vs)&~isF
        sc=[]; acc=[]
        for sd in range(NSEED):
            p=xgb.XGBClassifier(**PRM,random_state=sd).fit(Xa.loc[tr],y[tr]).predict_proba(Xa.loc[va])[:,1]
            sc.append(ev(p,y[va],RP[vs])['raw']); acc.append(p)
        o[vs]=(float(np.mean(sc)),float(np.std(sc))); preds[vs]=np.mean(acc,0)
    avg=(o[2024][0]+o[2022][0])/2; se=np.hypot(o[2024][1],o[2022][1])/2
    R.append(dict(name=name,nfeat=Xa.shape[1],avg=avg,se=se,m24=o[2024][0],s24=o[2024][1],
                  m22=o[2022][0],s22=o[2022][1]))
    log(f'{el()} {name:34s}({Xa.shape[1]:3d}) avg={avg:7.1f}±{se:4.1f}  '
        f'24:{o[2024][0]:7.1f}±{o[2024][1]:4.1f}  22:{o[2022][0]:7.1f}±{o[2022][1]:4.1f}')
    pd.DataFrame(R).to_csv(OUT+'res8.csv',index=False)
    if keep:
        for vs in FOLDS: np.save(OUT+f'p_{name}_{vs}.npy', preds[vs].astype(np.float32))
    return avg

# ═══════ A. 시즌별 능력 이력 ═══════
RATES=[('pitcher_id','asof_pitcher_n','asof_pitcher_success_rate','ps'),
       ('pitcher_id','asof_pitcher_n','asof_pitcher_middle_rate','pm'),
       ('batter_id','asof_batter_n','asof_batter_success_rate','bs')]
SEASONS=list(range(2019,2026))

def season_history(vs):
    """시즌 s(<vs) 별 (n, succ) 를 복원하고, 각 행 시점에서 쓸 수 있는 이력 피처 생성.
       vs 이후 정보는 절대 쓰지 않음."""
    F={}
    for idcol,ncol,ratecol,pref in RATES:
        t=RAW[[idcol,'season',ncol,ratecol]].copy()
        t['succ']=t[ncol]*t[ratecol].fillna(0)
        st=t.loc[t.groupby([idcol,'season'])[ncol].idxmin()].set_index([idcol,'season'])[[ncol,'succ']]
        en=t.loc[t.groupby([idcol,'season'])[ncol].idxmax()].set_index([idcol,'season'])[[ncol,'succ']]
        # 시즌 s 성적 = end[s] - start[s]
        per=(en-st).rename(columns={ncol:'n','succ':'sc'}).reset_index()
        per=per[per.n>0]
        lg=np.nanmean(RAW[ratecol]); K0=120.
        per['rate']=(per.sc+K0*lg)/(per.n+K0)
        cols={}
        for hl in [0.7,0.5,0.3]:                      # 시간감쇠 반감기 (작을수록 최근 중시)
            cols[f'{pref}_ewm{hl}']=np.full(len(RAW),np.nan,np.float32)
        for nm in ['last','prev2','trend','vol','nseas']:
            cols[f'{pref}_{nm}']=np.full(len(RAW),np.nan,np.float32)
        ids=RAW[idcol].values
        for s in SEASONS:
            tgt=season==s
            if not tgt.any(): continue
            h=per[per.season<s]
            if not len(h): continue
            g=h.sort_values('season')
            agg={}
            for pid,grp in g.groupby(idcol):
                ss=grp.season.values; rr=sp.logit(np.clip(grp.rate.values,.05,.95)); nn=grp.n.values
                d={}
                for hl in [0.7,0.5,0.3]:
                    w=(hl**(s-1-ss))*np.sqrt(nn)
                    d[f'ewm{hl}']=float(np.sum(w*rr)/np.sum(w))
                d['last']=float(rr[-1]); d['prev2']=float(rr[-2]) if len(rr)>1 else np.nan
                d['trend']=float(np.polyfit(ss,rr,1)[0]) if len(rr)>1 else 0.0
                d['vol']=float(np.std(rr)) if len(rr)>1 else 0.0
                d['nseas']=float(len(rr))
                agg[pid]=d
            for k in ['ewm0.7','ewm0.5','ewm0.3','last','prev2','trend','vol','nseas']:
                mp={p:v[k] for p,v in agg.items()}
                cols[f'{pref}_{k}'][tgt]=pd.Series(ids[tgt]).map(mp).values
        F.update(cols)
    return pd.DataFrame(F,index=RAW.index).astype(np.float32)

log(f'{el()} 시즌별 이력 생성...')
H=season_history(2026)
log(f'{el()} 이력 피처 {H.shape[1]}개')

# ═══════ B. 다중 축소강도 ═══════
def multi_k():
    F={}
    for idcol,ncol,ratecol,pref in [('pitcher_id','asof_pitcher_n','asof_pitcher_success_rate','p_succ'),
                                    ('batter_id','asof_batter_n','asof_batter_success_rate','b_succ')]:
        t=RAW[[idcol,'season',ncol,ratecol]].copy(); t['succ']=t[ncol]*t[ratecol].fillna(0)
        S=t.loc[t.groupby([idcol,'season'])[ncol].idxmin()].set_index([idcol,'season'])[[ncol,'succ']]
        a=RAW[[idcol,'season']].join(S,on=[idcol,'season'])
        dn=np.maximum(RAW[ncol].values-a[ncol].fillna(0).values,0)
        ds=np.maximum(np.nan_to_num(RAW[ncol].values*RAW[ratecol].values)-a['succ'].fillna(0).values,0)
        lg=np.nanmean(RAW[ratecol])
        for k in [25,75,400,1000]:
            F[f'{pref}_k{k}']=(ds+k*lg)/(dn+k)
    return pd.DataFrame(F,index=RAW.index).astype(np.float32)
MK=multi_k()

log(f'\n{el()} ===== 벤치마크 =====')
b0=bench(X0[CORE],'기준선(현 제출 v2)',keep=True)
bench(pd.concat([X0[CORE],MK],axis=1),'+ 다중축소강도')
bench(pd.concat([X0[CORE],H],axis=1),'+ 시즌별 능력이력')
bench(pd.concat([X0[CORE],H,MK],axis=1),'+ 이력 + 다중축소',keep=True)

# ═══════ C. NN 블렌딩 ═══════
log(f'\n{el()} ===== NN 블렌딩 =====')
import glob
best=None
for f in glob.glob(D+'results6/s*_c*_2024.npy'):
    tag=os.path.basename(f).replace('_2024.npy','')
    f22=f.replace('_2024.npy','_2022.npy')
    if not os.path.exists(f22): continue
    try:
        pn={2024:np.load(f),2022:np.load(f22)}
        px={vs:np.load(OUT+f'p_기준선(현 제출 v2)_{vs}.npy') for vs in FOLDS}
        for w in [0.05,0.1,0.2,0.3]:
            sc={}
            for vs in FOLDS:
                va=(season==vs)&~isF
                lo=(1-w)*sp.logit(np.clip(px[vs],1e-6,1-1e-6))+w*sp.logit(np.clip(pn[vs],1e-6,1-1e-6))
                sc[vs]=ev(sp.expit(lo),y[va],RP[vs])['raw']
            a=(sc[2024]+sc[2022])/2
            R.append(dict(name=f'blend_{tag}_w{w}',avg=a,m24=sc[2024],m22=sc[2022]))
            if best is None or a>best[0]:
                best=(a,tag,w); log(f'{el()}   blend {tag} w={w} avg={a:7.1f} (기준 {b0:7.1f})')
    except Exception: pass
log(f'{el()} 최고 블렌딩: {best}')
pd.DataFrame(R).to_csv(OUT+'res8.csv',index=False)
log(f'\n{el()} ===== 완료 =====')
