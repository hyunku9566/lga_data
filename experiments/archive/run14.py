"""
14차 — 트랙맨 5차. 이번엔 업계(MLB) 검증 방법론 기반 + 자체 발상 3개

【업계】
 K  Kirby Index 계열 : 릴리스 일관성. 단 원본의 VRA/HRA 는 플레이트 위치가 필요해
                       직접 계산 불가 -> 3D 릴리스 분산 + 익스텐션 산포로 대체·확장
                       (run5 T2 는 높이·좌우 2개만 썼음. 이번엔 익스텐션·마할라노비스 추가)
 V  VAA/HAA 성분     : 업계 공식에서 플레이트 항만 제외한 구조적 성분
                       VAA = -11.6236 + .0921*구속 - 1.0763*릴리스높이 - .0244*익스텐션
                             + 1.0976*플레이트높이 + .1777*수직무브먼트
 T  터널링           : 구종별 릴리스 중심 간 거리 (작을수록 같은 릴리스에서 나옴)
 A  릴리스 이상치     : 자기 중심에서 벗어난 정도 (평균/95분위)
 D  항력·회전효율     : zone_speed/rel_speed 및 잔차

【자체 발상】
 M1 릴리스-무브먼트 결합 일관성
     릴리스 변수로 무브먼트를 회귀한 뒤 '잔차 분산'.
     릴리스가 같은데 공이 다르게 휘면 로케이션이 흩어진다.
     -> Kirby 가 쓰는 릴리스 각도(우리는 관측 불가)의 대리 지표
 M2 압박 시 릴리스 이동
     3볼 카운트 릴리스 중심 - 0볼 카운트 릴리스 중심의 거리.
     압박에서 메커니즘이 흔들리는 투수를 물리적으로 포착
 M3 구종 간 무브먼트분리 / 릴리스분리 비율 (기만 효율)
"""
import os, time, warnings, traceback
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb
warnings.filterwarnings('ignore')
D='/home/lee/lga/'; OUT=D+'results14/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'

RAW=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig')
X98=pd.read_parquet(D+'X98.parquet')
y=X98.__y.values; season=X98.__season.values; isF=X98.__F.values.astype(bool)
CORE=[c for c in X98.columns if not c.startswith('__')]
J=pd.read_parquet(D+'aligned.parquet')     # run5 에서 만든 투구 단위 정렬(94.8% 정확)
log(f'{el()} 정렬 트랙맨 {len(J):,}구, 투수 {J.pitcher_id.nunique()}명')
TYPES=['fastball','breaking','offspeed']
J=J[J.pitch_type_group.isin(TYPES)].copy()
# VAA/HAA 구조 성분 (플레이트 항 제외)
J['vaa_c']=(-11.6236 + .0921*J.rel_speed - 1.0763*J.rel_height - .0244*J.extension + .1777*J.induced_vert_break)
J['haa_c']=(.0921*J.rel_speed - 1.0763*J.rel_side.abs() - .0244*J.extension + .1777*J.horz_break.abs())
J['drag']=J.zone_speed/J.rel_speed.replace(0,np.nan)

def zsafe(v):
    v=np.asarray(v,float); s=np.nanstd(v)
    return (v-np.nanmean(v))/(s if s>1e-9 else 1.0)

def build_pitcher_table(hist):
    """hist: 특정 시즌 이전 트랙맨. -> 투수별 지표 테이블"""
    out={}
    g=hist.groupby('pitcher_id')
    # K: 구종 내 릴리스 산포 (익스텐션 포함) + 3D 마할라노비스
    KEYS=['k_relh_sd','k_rels_sd','k_ext_sd','k_comb','k_rel3d_vol',
          'v_vaa','v_vaa_sd','v_haa','v_haa_sd','d_drag','d_drag_sd','t_tunnel',
          'a_anom_mean','a_anom_p95','m1_mov_resid','m2_press_shift','m2_press_velo','m3_decept']
    def kirby(x):
        d={k:np.nan for k in KEYS}
        sds=[]
        for t,gg in x.groupby('pitch_type_group'):
            if len(gg)<30: continue
            sds.append((len(gg),gg.rel_height.std(),gg.rel_side.std(),gg.extension.std()))
        if sds:
            w=np.array([s[0] for s in sds],float); w/=w.sum()
            d['k_relh_sd']=float(np.dot(w,[s[1] for s in sds]))
            d['k_rels_sd']=float(np.dot(w,[s[2] for s in sds]))
            d['k_ext_sd'] =float(np.dot(w,[s[3] for s in sds]))
            d['k_comb']=d['k_relh_sd']+d['k_rels_sd']+0.5*d['k_ext_sd']
        R=x[['rel_height','rel_side','extension']].dropna()
        if len(R)>50:
            C=np.cov(R.values.T)+np.eye(3)*1e-6
            d['k_rel3d_vol']=float(np.linalg.det(C))**(1/6)      # 릴리스 산포 부피
        # V: VAA/HAA 성분
        d['v_vaa']=float(x.vaa_c.mean()); d['v_vaa_sd']=float(x.vaa_c.std())
        d['v_haa']=float(x.haa_c.mean()); d['v_haa_sd']=float(x.haa_c.std())
        # D: 항력
        d['d_drag']=float(x.drag.mean()); d['d_drag_sd']=float(x.drag.std())
        # T: 터널링 = 구종 릴리스 중심 간 평균 거리
        cen=x.groupby('pitch_type_group')[['rel_height','rel_side','extension']].mean().values
        if len(cen)>1:
            dd=[np.linalg.norm(cen[i]-cen[j]) for i in range(len(cen)) for j in range(i+1,len(cen))]
            d['t_tunnel']=float(np.mean(dd))
        # A: 릴리스 이상치
        R2=x[['rel_height','rel_side']].dropna()
        if len(R2)>50:
            mu=R2.mean().values; C=np.cov(R2.values.T)+np.eye(2)*1e-6
            Ci=np.linalg.inv(C); dv=R2.values-mu
            md=np.sqrt(np.einsum('ij,jk,ik->i',dv,Ci,dv))
            d['a_anom_mean']=float(md.mean()); d['a_anom_p95']=float(np.percentile(md,95))
        # M1: 릴리스로 무브먼트를 설명한 뒤 남는 잔차 분산 (= 관측불가한 릴리스각도의 대리)
        res=[]
        for t,gg in x.groupby('pitch_type_group'):
            gg=gg[['rel_height','rel_side','extension','rel_speed','induced_vert_break','horz_break']].dropna()
            if len(gg)<80: continue
            A=np.c_[np.ones(len(gg)),gg[['rel_height','rel_side','extension','rel_speed']].values]
            for tgt in ['induced_vert_break','horz_break']:
                b,*_=np.linalg.lstsq(A,gg[tgt].values,rcond=None)
                res.append((len(gg),float(np.std(gg[tgt].values-A@b))))
        if res:
            w=np.array([r[0] for r in res],float); w/=w.sum()
            d['m1_mov_resid']=float(np.dot(w,[r[1] for r in res]))
        # M2: 압박(3볼) vs 여유(0볼) 릴리스 중심 이동
        hi=x[x.balls_before>=3]; lo=x[x.balls_before==0]
        if len(hi)>40 and len(lo)>40:
            a=hi[['rel_height','rel_side','extension']].mean().values
            b=lo[['rel_height','rel_side','extension']].mean().values
            d['m2_press_shift']=float(np.linalg.norm(a-b))
            d['m2_press_velo']=float(hi.rel_speed.mean()-lo.rel_speed.mean())
        # M3: 무브먼트 분리 / 릴리스 분리
        mc=x.groupby('pitch_type_group')[['induced_vert_break','horz_break']].mean().values
        if len(mc)>1 and 't_tunnel' in d and d['t_tunnel']>1e-6:
            mm=[np.linalg.norm(mc[i]-mc[j]) for i in range(len(mc)) for j in range(i+1,len(mc))]
            d['m3_decept']=float(np.mean(mm))/d['t_tunnel']
        d['_n']=len(x)
        return pd.Series({k:d.get(k,np.nan) for k in KEYS+['_n']})
    return g.apply(kirby)

COLS=None
FEATS={}
for s in range(2020,2026):
    tgt=season==s
    if not tgt.any(): continue
    hist=J[J.season<s]
    if not len(hist): continue
    tb=build_pitcher_table(hist)
    if COLS is None:
        COLS=[c for c in tb.columns if c!='_n']
        for c in COLS: FEATS[c]=np.full(len(RAW),np.nan,np.float32)
    pid=pd.Series(RAW.pitcher_id.values[tgt])
    for c in COLS:
        if c in tb.columns: FEATS[c][tgt]=pid.map(tb[c]).values
    log(f'{el()} 시즌{s} 지표 생성 (이전 {len(hist):,}구, 투수 {tb.shape[0]}명)')
TM=pd.DataFrame(FEATS,index=RAW.index).astype(np.float32)
TM.to_parquet(OUT+'tm5.parquet')
log(f'{el()} 트랙맨 5차 피처 {TM.shape[1]}개, 평균결측 {TM.isna().mean().mean():.3f}')
log('  ' + ', '.join(TM.columns))

# ═══════ 평가 ═══════
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
FOLDS=[2024,2022]; NSEED=5
PRM=dict(n_estimators=600,learning_rate=0.008,max_depth=6,min_child_weight=1500,subsample=0.7,
         colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,tree_method='hist',device='cuda:0',
         eval_metric='logloss',verbosity=0)
def bss(p,vs):
    va=(season==vs)&~isF; yv=y[va]; r=yv.mean()
    return 100000*max(0.,1-np.mean((p-yv)**2)/(r*(1-r)))
R=[]
def bench(Xa,name):
    o={}
    for vs in FOLDS:
        tr=(season<vs)&~(isF&(season<=2022)&(vs>=2023)); va=(season==vs)&~isF
        sc=[bss(xgb.XGBClassifier(**PRM,random_state=sd).fit(Xa.loc[tr],y[tr])
                .predict_proba(Xa.loc[va])[:,1],vs) for sd in range(NSEED)]
        o[vs]=(float(np.mean(sc)),float(np.std(sc)))
    avg=(o[2024][0]+o[2022][0])/2; se=np.hypot(o[2024][1],o[2022][1])/2
    R.append(dict(name=name,nfeat=Xa.shape[1],avg=avg,se=se,m24=o[2024][0],s24=o[2024][1],
                  m22=o[2022][0],s22=o[2022][1]))
    pd.DataFrame(R).to_csv(OUT+'res14.csv',index=False)
    log(f'{el()} {name:26s}({Xa.shape[1]:3d}) avg={avg:7.1f}±{se:4.1f}  '
        f'24:{o[2024][0]:7.1f}±{o[2024][1]:4.1f}  22:{o[2022][0]:7.1f}±{o[2022][1]:4.1f}')
    return avg

log(f'\n{el()} ===== 벤치마크 (시드 {NSEED}회) =====')
b0=bench(XK,'기준선 (v4 XGB)')
GRP={'K 릴리스일관성':[c for c in TM if c.startswith('k_')],
     'V VAA/HAA':[c for c in TM if c.startswith('v_')],
     'T 터널링':[c for c in TM if c.startswith('t_')],
     'A 릴리스이상치':[c for c in TM if c.startswith('a_')],
     'D 항력':[c for c in TM if c.startswith('d_')],
     'M1 무브먼트잔차':[c for c in TM if c.startswith('m1_')],
     'M2 압박릴리스이동':[c for c in TM if c.startswith('m2_')],
     'M3 기만효율':[c for c in TM if c.startswith('m3_')]}
res={}
for nm,cs in GRP.items():
    if not cs: continue
    res[nm]=bench(pd.concat([XK,TM[cs]],axis=1),nm)-b0
    log(f'{el()}    -> 순효과 {res[nm]:+6.1f}')
log(f'\n{el()} ===== 전체 결합 =====')
bench(pd.concat([XK,TM],axis=1),'트랙맨5차 전체')
pos=[c for nm,cs in GRP.items() if res.get(nm,0)>0 for c in cs]
if pos: bench(pd.concat([XK,TM[pos]],axis=1),f'양수그룹만({len(pos)})')
log(f'\n{el()} ===== 완료 =====')
