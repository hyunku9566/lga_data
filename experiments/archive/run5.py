"""
5차 — 트랙맨 3차 시도: 구종별 '커맨드' 축 (GPU1)

앞선 두 번의 실패:
  1차 구위 축 (구속/회전/무브먼트 평균)        -> 측정불가 (오차 내)
  2차 제어난이도 축 (무브먼트 크기/릴리스산포)   -> -3.2 (유의하게 해로움)

3차의 근거: 정렬된 118.9만 구에서 직접 측정한 구종군별 제구 성공률
  fastball .5438 / offspeed .5148 / breaking .4857   -> 격차 5.8pp
  (우리가 찾은 모든 상황효과 < 1.5pp 대비 압도적)
  그런데 현재 모델은 '사용 비율'만 알고 '그 투수의 그 구종 제구력'을 모른다.

핵심 설계 (규칙 준수):
  기대제구 = Σ_구종군 [asof_pitcher_*_rate: 공식 입력피처] × [투수x구종 제구율: ≤s-1 룩업]
  현재 투구의 구종도, 측정값도 사용하지 않음. '투수 단위 요약값'만 사용.

기준선 = deconv + S1(투수x상황성적표) + S2(매치업이력)  = 771.4 ± 2.0
"""
import os, time, warnings, traceback
import numpy as np, pandas as pd, scipy.special as sp
warnings.filterwarnings('ignore')

D='/home/lee/lga/'; OUT=D+'results5/'
os.makedirs(OUT, exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:6.1f}m]'
R=[]
def save(): pd.DataFrame(R).to_csv(OUT+'results5.csv', index=False)

log(f'{el()} 로드')
RAW=pd.read_csv(D+'data/train.csv', encoding='utf-8-sig')
y=RAW.control_success.values.astype(np.float32)
season=RAW.season.values; gtF=(RAW.game_type.values=='F')
BASE=pd.read_parquet(D+'features.parquet')
CORE=[c for c in BASE.columns if not c.startswith(('__','tm_')) and 'clutch' not in c and 'ctrl' not in c]
SP={vs:((season<vs)&~(gtF&(season<=2022)), season==vs) for vs in (2024,2023)}
def trend_prior(vs):
    m=season<vs; s=pd.Series(y[m]).groupby(season[m]).mean()
    return float(sp.expit(np.polyval(np.polyfit(s.index,sp.logit(s.values),1),vs)))
RP={vs:trend_prior(vs) for vs in (2024,2023)}
def evaluate(p,yv,rp):
    r=yv.mean(); ref=r*(1-r)
    b=lambda q:100000*max(0.,1-np.mean((q-yv)**2)/ref)
    lo=sp.logit(np.clip(p,1e-6,1-1e-6))
    return dict(raw=b(p), trend=b(sp.expit(lo-lo.mean()+sp.logit(rp))), oracle=b(sp.expit(lo-lo.mean()+sp.logit(r))))
PRM=dict(n_estimators=600, learning_rate=0.008, max_depth=6, min_child_weight=1500,
         subsample=0.7, colsample_bytree=0.5, reg_lambda=50., reg_alpha=1.,
         device='cuda:1', tree_method='hist', eval_metric='logloss', verbosity=0)
NSEED=7
def bench(Xa,name):
    import xgboost as xgb
    o={}
    for vs,(tr,va) in SP.items():
        sc=[]
        for sd in range(NSEED):
            m=xgb.XGBClassifier(**PRM, random_state=sd).fit(Xa.loc[tr], y[tr])
            sc.append(evaluate(m.predict_proba(Xa.loc[va])[:,1], y[va], RP[vs])['raw'])
        o[vs]=(float(np.mean(sc)), float(np.std(sc)))
    avg=(o[2024][0]+o[2023][0])/2; se=np.hypot(o[2024][1],o[2023][1])/2
    R.append(dict(name=name,nfeat=Xa.shape[1],avg=avg,se=se,
                  m2024=o[2024][0],s2024=o[2024][1],m2023=o[2023][0],s2023=o[2023][1]))
    log(f'{el()} {name:30s}({Xa.shape[1]:3d}) avg={avg:7.1f}±{se:4.1f}  '
        f'24:{o[2024][0]:7.1f}±{o[2024][1]:4.1f}  23:{o[2023][0]:7.1f}±{o[2023][1]:4.1f}')
    save(); return avg

# ═══════ 기준선 재구성: S1 + S2 ═══════
def asof_rate(mask,key,shrink=300.):
    out=np.full(len(RAW),np.nan,np.float32); ids=RAW[key].values
    for s in range(2020,2026):
        prev=season<s; tgt=season==s
        if not tgt.any() or not prev.any(): continue
        sel=prev&mask
        g=pd.Series(y[sel]).groupby(ids[sel])
        tot=pd.Series(y[prev]).groupby(ids[prev]).mean()
        num=g.sum(); den=g.size()
        base=tot.reindex(num.index).fillna(y[prev].mean())
        out[tgt]=pd.Series(ids[tgt]).map((num+shrink*base)/(den+shrink)).values
    return out

def build_S1():
    b=RAW.balls_before.values; st=RAW.strikes_before.values
    sits={'p_sit_3ball':b==3,'p_sit_2strk':st==2,'p_sit_ahead':st>b,'p_sit_behind':b>st,
          'p_sit_risp':(RAW.runner_on_2b.values|RAW.runner_on_3b.values)==1,
          'p_sit_on1b':RAW.runner_on_1b.values==1,'p_sit_vsL':RAW.batter_hand.values==1,
          'p_sit_vsR':RAW.batter_hand.values==2,'p_sit_late':RAW.inning.values>=7,
          'p_sit_hiLI':RAW.li.values>1.5,'p_sit_loLI':RAW.li.values<0.5,
          'p_sit_blowout':np.abs(RAW.score_diff_pitcher_team.values)>=5}
    F={}; ov=asof_rate(np.ones(len(RAW),bool),'pitcher_id'); F['p_sit_overall']=ov
    match=np.full(len(RAW),np.nan,np.float32)
    for n,m in sits.items():
        v=asof_rate(m,'pitcher_id'); F[n]=v; F[n+'_d']=v-ov
        match=np.where(m & np.isnan(match), F[n+'_d'], match)
    F['p_sit_matched']=match
    return pd.DataFrame(F,index=RAW.index).astype(np.float32)

def build_S2():
    key=RAW.pitcher_id.values.astype(np.int64)*100000+RAW.batter_id.values
    n=np.zeros(len(RAW),np.float32); r=np.full(len(RAW),np.nan,np.float32)
    for s in range(2020,2026):
        prev=season<s; tgt=season==s
        if not tgt.any() or not prev.any(): continue
        g=pd.Series(y[prev]).groupby(key[prev]); cnt=g.size(); sm=g.sum(); mu=y[prev].mean()
        k=pd.Series(key[tgt]); c=k.map(cnt).fillna(0).values; v=k.map(sm).fillna(0).values
        n[tgt]=c; r[tgt]=(v+30*mu)/(c+30)
    return pd.DataFrame({'pb_n':n,'pb_rate':r,'pb_logn':np.log1p(n)},index=RAW.index).astype(np.float32)

log(f'{el()} S1/S2 재구성')
X0=pd.concat([BASE[CORE], build_S1(), build_S2()], axis=1)
log(f'{el()} 기준선 피처 {X0.shape[1]}개')

# ═══════ 투구 단위 정렬 (94.8% 정확) ═══════
JOIN=D+'aligned.parquet'
if os.path.exists(JOIN):
    J=pd.read_parquet(JOIN); log(f'{el()} 정렬 캐시 로드 {len(J):,}')
else:
    log(f'{el()} 트랙맨 정렬 시작')
    tm=pd.read_csv(D+'data/trackman_history.csv', encoding='utf-8-sig',
        usecols=['season','trackman_game_id','pitch_no','inning','top_bottom','balls_before',
                 'strikes_before','outs_before','pitch_type_group','rel_speed','spin_rate',
                 'induced_vert_break','horz_break','extension','rel_height','rel_side'])
    tm=tm.sort_values(['trackman_game_id','pitch_no']); tm['top_bottom']=tm.top_bottom.str[0]
    tm=tm[tm.inning>=1]
    tr=RAW[['season','inning','top_bottom','balls_before','strikes_before','outs_before']].copy()
    tr['gid']=(RAW.inning.diff().fillna(0)<0).cumsum().values
    tr['ridx']=np.arange(len(RAW))
    def sig(df,g,n=30):
        s=(df.inning.astype(str)+df.top_bottom+df.balls_before.astype(str)
           +df.strikes_before.astype(str)+df.outs_before.astype(str))
        return s.groupby(df[g]).apply(lambda x:'|'.join(x.head(n)))
    A=pd.DataFrame({'sig':sig(tr,'gid'),'season':tr.groupby('gid').season.first()})
    B=pd.DataFrame({'sig':sig(tm,'trackman_game_id'),'season':tm.groupby('trackman_game_id').season.first()})
    M=A.reset_index().merge(B.reset_index(),on=['sig','season']).drop_duplicates('gid').drop_duplicates('trackman_game_id')
    t2=tr.merge(M[['gid','trackman_game_id']],on='gid'); t2['k']=t2.groupby('gid').cumcount()
    m2=tm[tm.trackman_game_id.isin(M.trackman_game_id)].copy(); m2['k']=m2.groupby('trackman_game_id').cumcount()
    J=t2.merge(m2,on=['trackman_game_id','k'],suffixes=('','_tm'))
    J=J[(J.inning==J.inning_tm)&(J.balls_before==J.balls_before_tm)&
        (J.strikes_before==J.strikes_before_tm)&(J.outs_before==J.outs_before_tm)]
    J['pitcher_id']=RAW.pitcher_id.values[J.ridx.values]
    J['y']=y[J.ridx.values]
    J['pitch_of_app']=J.groupby(['trackman_game_id','pitcher_id']).cumcount()
    J.to_parquet(JOIN)
    log(f'{el()} 정렬 완료 {len(J):,}구, 투수 {J.pitcher_id.nunique()}명')

# ═══════ 트랙맨 3차 피처 ═══════
GROUPS={}
TYPES=['fastball','breaking','offspeed']

def pv_asof(dfagg, cols, prefix):
    """(pitcher_id, season) 집계를 s-1 까지 누적 가중평균해 행에 배치"""
    F={f'{prefix}{c}':np.full(len(RAW),np.nan,np.float32) for c in cols}
    ar=dfagg.reset_index()
    for s in range(2020,2026):
        tgt=season==s; prev=ar[ar.season<s]
        if not tgt.any() or not len(prev): continue
        w=prev.groupby('pitcher_id').apply(lambda x: pd.Series(
            {c: np.average(x[c].fillna(np.nanmean(x[c])), weights=x['w'])
                if x[c].notna().any() else np.nan for c in cols}))
        pid=pd.Series(RAW.pitcher_id.values[tgt])
        for c in cols: F[f'{prefix}{c}'][tgt]=pid.map(w[c]).values
    return pd.DataFrame(F,index=RAW.index).astype(np.float32)

# --- T1. 구종군별 제구 성공률 + 사용비율 가중 기대제구 ---
def T1():
    a=J.groupby(['pitcher_id','season','pitch_type_group']).y.agg(['sum','size']).reset_index()
    a=a[a.pitch_type_group.isin(TYPES)]
    piv=a.pivot_table(index=['pitcher_id','season'],columns='pitch_type_group',
                      values=['sum','size'],fill_value=0)
    piv.columns=[f'{b}_{c}' for b,c in piv.columns]
    piv['w']=piv[[f'size_{t}' for t in TYPES]].sum(1)
    lg=J[J.pitch_type_group.isin(TYPES)].groupby('pitch_type_group').y.mean()
    SH=250.
    for t in TYPES:
        piv[f'cmd_{t}']=(piv[f'sum_{t}']+SH*lg[t])/(piv[f'size_{t}']+SH)
        piv[f'use_{t}']=piv[f'size_{t}']/piv['w'].clip(lower=1)
    cols=[f'cmd_{t}' for t in TYPES]+[f'use_{t}' for t in TYPES]
    Fd=pv_asof(piv, cols, 'tc_')
    # 주어진 사용비율(공식 피처)로 가중한 기대제구
    uf=RAW.asof_pitcher_fastball_rate.values; ub=RAW.asof_pitcher_breaking_rate.values
    uo=RAW.asof_pitcher_offspeed_rate.values
    tot=np.nan_to_num(uf)+np.nan_to_num(ub)+np.nan_to_num(uo)
    tot=np.where(tot<=0,1,tot)
    Fd['tc_expected']=(np.nan_to_num(uf)*Fd.tc_cmd_fastball.values
                      +np.nan_to_num(ub)*Fd.tc_cmd_breaking.values
                      +np.nan_to_num(uo)*Fd.tc_cmd_offspeed.values)/tot
    # 구종별 커맨드 편차 (전체 대비)
    ov=np.nanmean([Fd.tc_cmd_fastball.values,Fd.tc_cmd_breaking.values,Fd.tc_cmd_offspeed.values],0)
    for t in TYPES: Fd[f'tc_d_{t}']=Fd[f'tc_cmd_{t}'].values-ov
    Fd['tc_cmd_spread']=np.nanmax([Fd[f'tc_cmd_{t}'].values for t in TYPES],0) \
                       -np.nanmin([Fd[f'tc_cmd_{t}'].values for t in TYPES],0)
    # 사용비율 변화 (트랙맨 시점 대비 현재)
    Fd['tc_use_shift_brk']=np.nan_to_num(ub)-Fd.tc_use_breaking.values
    return Fd.astype(np.float32)

# --- T2. 구종 내 릴리스 산포 (진짜 반복성) ---
def T2():
    g=J[J.pitch_type_group.isin(TYPES)].groupby(['pitcher_id','season','pitch_type_group'])
    s=g[['rel_height','rel_side','extension','rel_speed']].std()
    n=g.size().rename('n')
    s=s.join(n).reset_index()
    s=s[s.n>=30]
    a=s.groupby(['pitcher_id','season']).apply(lambda x: pd.Series({
        'wt_relh_sd':np.average(x.rel_height.fillna(x.rel_height.mean()),weights=x.n),
        'wt_rels_sd':np.average(x.rel_side.fillna(x.rel_side.mean()),weights=x.n),
        'wt_ext_sd' :np.average(x.extension.fillna(x.extension.mean()),weights=x.n),
        'wt_velo_sd':np.average(x.rel_speed.fillna(x.rel_speed.mean()),weights=x.n),
        'w':x.n.sum()}))
    return pv_asof(a,[c for c in a.columns if c!='w'],'t2_')

# --- T3. 등판 내 구속 저하 (물리적 체력) ---
def T3():
    K=J[J.pitch_type_group=='fastball'].copy()
    K['bin']=pd.cut(K.pitch_of_app,[-1,15,40,70,300],labels=['e','m','l','x'])
    p=K.pivot_table(index=['pitcher_id','season'],columns='bin',values='rel_speed',aggfunc='mean')
    c=K.pivot_table(index=['pitcher_id','season'],columns='bin',values='rel_speed',aggfunc='size')
    p.columns=[f'v_{x}' for x in p.columns]; p['w']=c.sum(1)
    p['fade_m']=p.v_m-p.v_e; p['fade_l']=p.v_l-p.v_e; p['fade_x']=p.v_x-p.v_e
    return pv_asof(p,['v_e','fade_m','fade_l','fade_x'],'t3_')

# --- T4. 압박 카운트에서의 위축 ---
def T4():
    K=J.copy()
    K['press']=(K.balls_before>=2)&(K.balls_before>K.strikes_before)
    a=K.groupby(['pitcher_id','season']).apply(lambda x: pd.Series({
        'velo_press': x.rel_speed[x.press].mean()-x.rel_speed[~x.press].mean(),
        'fb_press'  : (x.pitch_type_group[x.press]=='fastball').mean()
                      -(x.pitch_type_group[~x.press]=='fastball').mean(),
        'brk_press' : (x.pitch_type_group[x.press]=='breaking').mean()
                      -(x.pitch_type_group[~x.press]=='breaking').mean(),
        'w': len(x)}))
    return pv_asof(a,['velo_press','fb_press','brk_press'],'t4_')

# --- T5. 무브먼트 크기 x 사용비중 ---
def T5():
    K=J[J.pitch_type_group.isin(TYPES)].copy()
    K['brk']=np.hypot(K.induced_vert_break,K.horz_break)
    a=K.groupby(['pitcher_id','season']).apply(lambda x: pd.Series({
        'brk_mean':x.brk.mean(), 'brk_sd':x.brk.std(),
        'brk_wt':(x.brk*(x.pitch_type_group!='fastball')).mean(),
        'velo':x.rel_speed.mean(), 'w':len(x)}))
    return pv_asof(a,['brk_mean','brk_sd','brk_wt','velo'],'t5_')

# --- T6/T7. 시즌간 릴리스 이동 + 구속 추세 ---
def T67():
    a=J.groupby(['pitcher_id','season']).agg(rh=('rel_height','mean'),rs=('rel_side','mean'),
                                             ex=('extension','mean'),v=('rel_speed','mean'),
                                             w=('rel_speed','size')).reset_index()
    a=a.sort_values(['pitcher_id','season'])
    for c in ['rh','rs','ex','v']: a[f'd_{c}']=a.groupby('pitcher_id')[c].diff()
    a['mech_shift']=np.hypot(a.d_rh,a.d_rs)
    a=a.set_index(['pitcher_id','season'])
    return pv_asof(a,['d_rh','d_rs','d_ex','d_v','mech_shift'],'t6_')

# --- T8. 아스널 복잡도 ---
def T8():
    a=J.groupby(['pitcher_id','season']).apply(lambda x: pd.Series({
        'ntypes':x.pitch_type_group.nunique(),
        'entropy':-(x.pitch_type_group.value_counts(normalize=True)
                    *np.log(x.pitch_type_group.value_counts(normalize=True))).sum(),
        'w':len(x)}))
    return pv_asof(a,['ntypes','entropy'],'t8_')

for nm,fn in [('T1 구종별커맨드',T1),('T2 구종내릴리스산포',T2),('T3 등판내구속저하',T3),
              ('T4 압박시위축',T4),('T5 무브먼트x비중',T5),('T6-7 메커니즘이동/구속추세',T67),
              ('T8 아스널복잡도',T8)]:
    try:
        t=time.time(); GROUPS[nm]=fn()
        log(f'{el()} 빌드 {nm}: {GROUPS[nm].shape[1]}개 ({time.time()-t:.0f}s)')
    except Exception: log(f'!! 빌드 {nm}\n'+traceback.format_exc())

log(f'\n{el()} ===== 기준선 (deconv+S1+S2) =====')
b0=bench(X0,'기준선 S1+S2')
log(f'\n{el()} ===== 트랙맨 3차 개별 검증 =====')
res={}
for nm,Fd in GROUPS.items():
    try:
        res[nm]=bench(pd.concat([X0,Fd],axis=1), nm)-b0
        log(f'{el()}    -> 순효과 {res[nm]:+7.1f}')
    except Exception: log(f'!! {nm}\n'+traceback.format_exc())

log(f'\n{el()} ===== 양수 누적 결합 =====')
cum=X0.copy(); acc=[]
for nm in sorted(res,key=res.get,reverse=True):
    if res[nm]<=0: break
    cum=pd.concat([cum,GROUPS[nm]],axis=1); acc.append(nm)
    bench(cum,'결합: '+'+'.join(a.split()[0] for a in acc))
log(f'\n{el()} ===== 트랙맨 전체결합 =====')
bench(pd.concat([X0]+list(GROUPS.values()),axis=1),'트랙맨 전체')
save()
log(f'\n{el()} ===== 5차 완료 =====')
