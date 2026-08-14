"""
24차 — 시간 일반화 3종 (22차 진단이 가리킨 방향)

22차: 무작위분할 1146.4 vs 시간분할 818.9. 손실은 정보 부족이 아니라
      '올해 선수 상태를 모른 채 다음 해를 예측하는' 과정에서 난다.

G1 최근성 가중 피처
   현행 p_sit_* / matchup / 투수 전체율은 이전 시즌을 전부 균등 가중한다.
   리그 제구율이 매년 -1.2%p 씩 밀리는데 2019년 성적과 2023년 성적을
   같은 무게로 쓰고 있다. 시즌 지수감쇠(반감기 2시즌)를 넣는다.

G2 커리어 기준 동적 축소  ← 현행 구조의 명백한 결함
   현행 p_succ_ssn = (ds + K*리그평균)/(dn + K).
   당해 성적을 '리그 평균'으로 축소하고 있다. 올바른 경험적 베이즈는
   '그 투수의 커리어 성적'으로 축소하는 것이다. 커리어가 훨씬 좋은 사전분포다.
   추가로 당해 표본이 쌓일수록 커리어 비중이 자동으로 줄도록 신뢰도까지 준다.

G3 콜드스타트
   신인/복귀 투수는 커리어 사전분포가 없거나 낡았다. 이를 명시적으로 표시한다.
   (22차에서 시간 일반화가 병목으로 나왔으므로 '처음 보는 투수' 처리가 중요)
"""
import os, json, time, warnings
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results24/'; os.makedirs(OUT,exist_ok=True)
S='/tmp/claude-1000/-home-lee-lga/97188a20-36c6-4af4-9fed-509e8b2fcd01/scratchpad/'
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'

RAW=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig')
X98=pd.read_parquet(D+'X98.parquet'); TM=pd.read_parquet(D+'results14/tm5.parquet')
y=X98.__y.values.astype(np.float32); season=X98.__season.values; isF=X98.__F.values.astype(bool)
CORE=[c for c in X98.columns if not c.startswith('__')]
TMSEL=json.load(open(D+'v6_tmsel.json'))
def multi_k():
    F={}
    for idc,nc,rc,pf in [('pitcher_id','asof_pitcher_n','asof_pitcher_success_rate','p_succ'),
                         ('batter_id','asof_batter_n','asof_batter_success_rate','b_succ')]:
        t=RAW[[idc,'season',nc,rc]].copy(); t['succ']=t[nc]*t[rc].fillna(0)
        Sx=t.loc[t.groupby([idc,'season'])[nc].idxmin()].set_index([idc,'season'])[[nc,'succ']]
        a=RAW[[idc,'season']].join(Sx,on=[idc,'season'])
        dn=np.maximum(RAW[nc].values-a[nc].fillna(0).values,0)
        ds=np.maximum(np.nan_to_num(RAW[nc].values*RAW[rc].values)-a['succ'].fillna(0).values,0)
        lgv=np.nanmean(RAW[rc])
        for k in [25,75,400,1000]: F[f'{pf}_k{k}']=(ds+k*lgv)/(dn+k)
    return pd.DataFrame(F,index=RAW.index).astype(np.float32)
OF=pd.read_parquet(S+'oof_comp.parquet')
V7=pd.concat([X98[CORE],multi_k(),TM[TMSEL],OF],axis=1)
log(f'{el()} v7 기준 {V7.shape[1]}피처')

pid=RAW.pitcher_id.values; bid=RAW.batter_id.values
b=RAW.balls_before.values; st=RAW.strikes_before.values
SITS={'3ball':b==3,'2strk':st==2,'ahead':st>b,'behind':b>st,
      'risp':(RAW.runner_on_2b.values|RAW.runner_on_3b.values)==1,
      'on1b':RAW.runner_on_1b.values==1,'vsL':RAW.batter_hand.values==1,
      'vsR':RAW.batter_hand.values==2,'late':RAW.inning.values>=7,
      'hiLI':RAW.li.values>1.5,'loLI':RAW.li.values<0.5,
      'blowout':np.abs(RAW.score_diff_pitcher_team.values)>=5}

# ═══════ G1 최근성 가중 as-of 테이블 ═══════
HLF=2.0; SHR=300.
def wasof(mask,ids,shrink=SHR):
    out=np.full(len(RAW),np.nan,np.float32)
    for s in range(2020,2025):
        prev=season<s; tgt=season==s
        if not tgt.any() or not prev.any(): continue
        w=(0.5**((s-1-season[prev])/HLF)).astype(np.float64)
        dfp=pd.DataFrame({'id':ids[prev],'y':y[prev].astype(np.float64),'w':w,
                          'm':mask[prev].astype(np.float64)})
        g=dfp.groupby('id')
        tw=g.apply(lambda d:(d.w*d.m).sum()); ts=g.apply(lambda d:(d.w*d.m*d.y).sum())
        aw=g.w.sum(); asum=g.apply(lambda d:(d.w*d.y).sum())
        ov=(asum/aw)                                  # 투수 전체(가중)
        gmu=float((dfp.w*dfp.y).sum()/dfp.w.sum())
        base=ov.reindex(tw.index).fillna(gmu)
        rate=(ts+shrink*base)/(tw+shrink)
        out[tgt]=pd.Series(ids[tgt]).map(rate).values
    return out
G1={}
ov_w=wasof(np.ones(len(RAW),bool),pid); G1['w_p_overall']=ov_w
for n,m in SITS.items():
    v=wasof(m,pid); G1[f'w_p_{n}']=v; G1[f'w_p_{n}_d']=v-ov_w
# 가중 매치업
pk=pid.astype(np.int64)*100000+bid
mm=np.full(len(RAW),np.nan,np.float32); mn=np.zeros(len(RAW),np.float32)
for s in range(2020,2025):
    prev=season<s; tgt=season==s
    if not tgt.any() or not prev.any(): continue
    w=(0.5**((s-1-season[prev])/HLF)).astype(np.float64)
    dfp=pd.DataFrame({'k':pk[prev],'y':y[prev].astype(np.float64),'w':w}); g=dfp.groupby('k')
    cw=g.w.sum(); sw=g.apply(lambda d:(d.w*d.y).sum())
    mu=float((dfp.w*dfp.y).sum()/dfp.w.sum())
    kk=pd.Series(pk[tgt]); c=kk.map(cw).fillna(0).values; v=kk.map(sw).fillna(0).values
    mm[tgt]=(v+30*mu)/(c+30); mn[tgt]=c
G1['w_pb_rate']=mm; G1['w_pb_n']=mn
G1=pd.DataFrame(G1,index=RAW.index).astype(np.float32)
log(f'{el()} G1 최근성가중 {G1.shape[1]}개')

# ═══════ G2 커리어 기준 동적 축소 ═══════
G2={}
for pref,idc,nc,rc in [('p','pitcher_id','asof_pitcher_n','asof_pitcher_success_rate'),
                       ('b','batter_id','asof_batter_n','asof_batter_success_rate')]:
    t=RAW[[idc,'season',nc,rc]].copy(); t['succ']=t[nc]*t[rc].fillna(0)
    A=t.loc[t.groupby([idc,'season'])[nc].idxmin()].set_index([idc,'season'])[[nc,'succ']]
    a=RAW[[idc,'season']].join(A,on=[idc,'season'])
    an=a[nc].fillna(0).values; asu=a['succ'].fillna(0).values      # 시즌 시작 시점 커리어
    n_now=RAW[nc].values; s_now=np.nan_to_num(RAW[nc].values*RAW[rc].values)
    dn=np.maximum(n_now-an,0); ds=np.maximum(s_now-asu,0)          # 당해 진행분
    lgv=float(np.nanmean(RAW[rc]))
    car=np.where(an>0,asu/np.maximum(an,1),np.nan)                 # 커리어 사전분포
    prior=np.where(np.isnan(car),lgv,car)
    for k in [25,75,200,600]:
        G2[f'{pref}_car_k{k}']=(ds+k*prior)/(dn+k)                 # 커리어로 축소
    G2[f'{pref}_trust']=dn/(dn+200.)                               # 당해 신뢰도
    G2[f'{pref}_car_prior']=prior
    G2[f'{pref}_ssn_minus_car']=np.where(dn>0,ds/np.maximum(dn,1)-prior,np.nan)
    G2[f'{pref}_car_n']=an
G2=pd.DataFrame(G2,index=RAW.index).astype(np.float32)
log(f'{el()} G2 커리어축소 {G2.shape[1]}개')

# ═══════ G3 콜드스타트 ═══════
sp_seen=RAW.groupby(['pitcher_id','season']).size().reset_index()[['pitcher_id','season']]
first={}; last={}
for p_,s_ in zip(sp_seen.pitcher_id.values,sp_seen.season.values):
    first[p_]=min(first.get(p_,9999),s_); last.setdefault(p_,[]).append(s_)
G3={}
an_p=G2['p_car_n'].values
G3['p_is_rookie']=(an_p==0).astype(np.float32)
G3['p_car_n_log']=np.log1p(an_p)
prev_last=np.full(len(RAW),np.nan,np.float32); nseas=np.full(len(RAW),np.nan,np.float32)
for s in range(2020,2025):
    tgt=season==s
    if not tgt.any(): continue
    pl={}; ns={}
    for p_,ss in last.items():
        ps=[x for x in ss if x<s]
        if ps: pl[p_]=max(ps); ns[p_]=len(set(ps))
    k=pd.Series(pid[tgt])
    prev_last[tgt]=s-k.map(pl).values          # 마지막 등판 시즌과의 간격
    nseas[tgt]=k.map(ns).fillna(0).values      # 경험 시즌 수
G3['p_gap_seasons']=prev_last
G3['p_n_seasons']=nseas
G3['p_rookie_x_ssn_n']=G3['p_is_rookie']*G2['p_trust'].values
G3['p_thin_prior']=np.exp(-an_p/500.)          # 사전분포가 얇을수록 1
G3=pd.DataFrame(G3,index=RAW.index).astype(np.float32)
log(f'{el()} G3 콜드스타트 {G3.shape[1]}개  신인비율 {G3.p_is_rookie.mean():.3f}')

# ═══════ 평가 ═══════
XP=dict(n_estimators=600,learning_rate=0.008,max_depth=6,min_child_weight=1500,subsample=0.7,
        colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,tree_method='hist',device='cuda:1',
        eval_metric='logloss',verbosity=0)
FOLDS=[2024,2022]; HL=2.0; NS=3
CTX={}
for vs in FOLDS:
    tr=(season<vs)&~(isF&(season<=2022)&(vs>=2023)); va=(season==vs)&~isF
    CTX[vs]=(tr,va,(0.5**((vs-1-season[tr])/HL)).astype(np.float32),y[va].mean()*(1-y[va].mean()))
R=[]
def bench(Xa,name):
    o={}
    for vs in FOLDS:
        tr,va,w,bq=CTX[vs]
        p=np.mean([xgb.XGBClassifier(**XP,random_state=s).fit(Xa[tr],y[tr],sample_weight=w)
                   .predict_proba(Xa[va])[:,1] for s in range(NS)],0)
        o[vs]=100000*max(0.,1-np.mean((p-y[va])**2)/bq)
    avg=(o[2024]+o[2022])/2
    R.append(dict(name=name,nfeat=Xa.shape[1],m24=o[2024],m22=o[2022],avg=avg))
    pd.DataFrame(R).to_csv(OUT+'res24.csv',index=False)
    log(f'{el()} {name:26s}({Xa.shape[1]:3d}) 24:{o[2024]:7.1f} 22:{o[2022]:7.1f} 평균 {avg:7.1f}')
    return avg
log(f'\n{el()} ===== 벤치마크 (시드{NS}) =====')
b0=bench(V7,'기준 v7')
g={}
for nm,G in [('G1 최근성가중',G1),('G2 커리어축소',G2),('G3 콜드스타트',G3)]:
    g[nm]=bench(pd.concat([V7,G],axis=1),nm)-b0
    log(f'{el()}    -> 순효과 {g[nm]:+6.1f}')
log(f'\n{el()} ===== 조합 =====')
bench(pd.concat([V7,G1,G2],axis=1),'G1+G2')
bench(pd.concat([V7,G2,G3],axis=1),'G2+G3')
bench(pd.concat([V7,G1,G2,G3],axis=1),'G1+G2+G3')
pos=[G for nm,G in [('G1 최근성가중',G1),('G2 커리어축소',G2),('G3 콜드스타트',G3)] if g.get(nm,0)>0]
if pos and len(pos)<3: bench(pd.concat([V7]+pos,axis=1),f'양수그룹만({len(pos)})')
for nm,G in [('G1',G1),('G2',G2),('G3',G3)]: G.to_parquet(OUT+f'{nm}.parquet')
log(f'{el()} 완료')
