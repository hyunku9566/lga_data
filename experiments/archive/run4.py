"""
4차 — 야구 직관 피처 검증 (GPU1 전용, run3 와 병렬)

방법론 교정:
  이전 트랙맨/clutch 판정은 "설정 1개, 반복 0회" 였음. 효과크기 ±5~8점 < 측정오차.
  -> 이번엔 시드 7회 반복으로 평균±표준편차를 내고, 효과가 오차를 넘는지로 판정한다.

검증할 직관:
  S1 투수×상황 과거 성적표 (as-of)  ← 제안만 하고 미구현이었던 것
  S2 투수x타자 매치업 이력 (as-of)
  S3 트랙맨: 무브먼트 크기 = 제구 난이도 (구위가 아니라 '제어 난이도' 축)
  S4 트랙맨: 구속-제구 트레이드오프 잔차
  S5 트랙맨: 릴리스 반복성 (경기내 산포)
  S6 압박성향 재설계 (차이값 대신 상황별 원값 2개)
  S7 홈/원정, 추운달 민감도
  S8 주자 견제 부담 (1루주자 x 투수별 민감도)
"""
import os, time, warnings, traceback
import numpy as np, pandas as pd, scipy.special as sp
warnings.filterwarnings('ignore')

D='/home/lee/lga/'; OUT=D+'results4/'
os.makedirs(OUT, exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:6.1f}m]'

R=[]
def save(): pd.DataFrame(R).to_csv(OUT+'results4.csv', index=False)

log(f'{el()} 로드')
RAW=pd.read_csv(D+'data/train.csv', encoding='utf-8-sig')
y=RAW.control_success.values.astype(np.float32)
season=RAW.season.values; gtF=(RAW.game_type.values=='F')
BASE=pd.read_parquet(D+'features.parquet')
CORE=[c for c in BASE.columns if not c.startswith(('__','tm_')) and 'clutch' not in c and 'ctrl' not in c]
X0=BASE[CORE].copy()
log(f'{el()} 기준 피처셋 {len(CORE)}개')

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

def bench(Xa, name):
    """시드 NSEED 회 반복 -> 평균±sd. 효과가 오차를 넘는지 판정 가능하게."""
    import xgboost as xgb
    out={}
    for vs,(tr,va) in SP.items():
        sc=[]
        for sd in range(NSEED):
            m=xgb.XGBClassifier(**PRM, random_state=sd).fit(Xa.loc[tr], y[tr])
            p=m.predict_proba(Xa.loc[va])[:,1]
            sc.append(evaluate(p,y[va],RP[vs])['raw'])
        out[vs]=(float(np.mean(sc)), float(np.std(sc)))
    avg=(out[2024][0]+out[2023][0])/2
    se=np.hypot(out[2024][1],out[2023][1])/2
    R.append(dict(name=name, nfeat=Xa.shape[1], avg=avg, se=se,
                  m2024=out[2024][0], s2024=out[2024][1], m2023=out[2023][0], s2023=out[2023][1]))
    log(f'{el()} {name:26s}({Xa.shape[1]:3d}) avg={avg:7.1f}±{se:4.1f}  '
        f'24:{out[2024][0]:7.1f}±{out[2024][1]:4.1f}  23:{out[2023][0]:7.1f}±{out[2023][1]:4.1f}')
    save(); return avg

# ═══════ as-of 헬퍼: season<s 데이터로만 선수별 통계 ═══════
def asof_rate(mask, key, shrink=300.0, name='x'):
    """선수(key)별, 조건(mask) 하에서의 과거 성공률을 as-of 로. 축소 적용."""
    out=np.full(len(RAW), np.nan, np.float32)
    ids=RAW[key].values
    for s in range(2020, 2026):
        prev=(season<s)
        tgt=(season==s)
        if not tgt.any() or not prev.any(): continue
        sel=prev&mask
        gm=pd.Series(y[sel]).groupby(ids[sel])
        tot=pd.Series(y[prev]).groupby(ids[prev]).mean()
        num=gm.sum(); den=gm.size()
        base=tot.reindex(num.index).fillna(y[prev].mean())
        rate=(num+shrink*base)/(den+shrink)
        out[tgt]=pd.Series(ids[tgt]).map(rate).values
    return out

# ═══════ S1. 투수x상황 과거 성적표 ═══════
def s1():
    b=RAW.balls_before.values; st=RAW.strikes_before.values
    sits={
      'p_sit_3ball' : b==3,
      'p_sit_2strk' : st==2,
      'p_sit_ahead' : (st>b),
      'p_sit_behind': (b>st),
      'p_sit_risp'  : (RAW.runner_on_2b.values|RAW.runner_on_3b.values)==1,
      'p_sit_on1b'  : RAW.runner_on_1b.values==1,
      'p_sit_vsL'   : RAW.batter_hand.values==1,
      'p_sit_vsR'   : RAW.batter_hand.values==2,
      'p_sit_late'  : RAW.inning.values>=7,
      'p_sit_hiLI'  : RAW.li.values>1.5,
      'p_sit_loLI'  : RAW.li.values<0.5,
      'p_sit_blowout':np.abs(RAW.score_diff_pitcher_team.values)>=5,
    }
    F={}
    ov=asof_rate(np.ones(len(RAW),bool),'pitcher_id',name='ov')
    F['p_sit_overall']=ov
    for n,m in sits.items():
        v=asof_rate(m,'pitcher_id',name=n); F[n]=v; F[n+'_d']=v-ov   # 전체 대비 편차
    # 현재 행 상황에 매칭되는 값 하나로 압축
    match=np.full(len(RAW),np.nan,np.float32)
    for n,m in sits.items():
        match=np.where(m & np.isnan(match), F[n+'_d'], match)
    F['p_sit_matched']=match
    return pd.DataFrame(F, index=RAW.index).astype(np.float32)

# ═══════ S2. 투수x타자 매치업 이력 ═══════
def s2():
    key=RAW.pitcher_id.values.astype(np.int64)*100000+RAW.batter_id.values
    n=np.zeros(len(RAW),np.float32); r=np.full(len(RAW),np.nan,np.float32)
    for s in range(2020,2026):
        prev=season<s; tgt=season==s
        if not tgt.any() or not prev.any(): continue
        g=pd.Series(y[prev]).groupby(key[prev])
        cnt=g.size(); sm=g.sum(); mu=y[prev].mean()
        k=pd.Series(key[tgt])
        c=k.map(cnt).fillna(0).values; v=k.map(sm).fillna(0).values
        n[tgt]=c; r[tgt]=(v+30*mu)/(c+30)
    return pd.DataFrame({'pb_n':n,'pb_rate':r,'pb_logn':np.log1p(n)}, index=RAW.index).astype(np.float32)

# ═══════ S3~S5. 트랙맨 재설계 (제어 난이도 축) ═══════
def s345():
    pm=pd.read_csv(D+'pitcher_map.csv'); pm=pm[pm.conf>=0.90]
    tm=pd.read_csv(D+'data/trackman_history.csv', encoding='utf-8-sig',
        usecols=['season','trackman_game_id','pitcher_trackman_id','pitch_type_group','rel_speed',
                 'spin_rate','induced_vert_break','horz_break','extension','rel_height','rel_side'])
    tm=tm.merge(pm[['pitcher_id','pitcher_trackman_id']], on='pitcher_trackman_id')
    tm['brk']=np.hypot(tm.induced_vert_break, tm.horz_break)     # 총 무브먼트 크기
    # 경기내 릴리스 산포 -> 투수x시즌 평균 (진짜 반복성)
    ing=tm.groupby(['pitcher_id','season','trackman_game_id'])[['rel_height','rel_side','extension']].std()
    ing=ing.groupby(['pitcher_id','season']).mean().rename(columns=lambda c:'ig_'+c)
    a=tm.groupby(['pitcher_id','season']).agg(
        velo=('rel_speed','mean'), brk=('brk','mean'), brk_sd=('brk','std'),
        hb_abs=('horz_break',lambda x:x.abs().mean()), ivb=('induced_vert_break','mean'),
        spin=('spin_rate','mean'), ext=('extension','mean'), n=('rel_speed','size'))
    mixbrk=tm[tm.pitch_type_group.isin(['breaking','offspeed'])].groupby(['pitcher_id','season']).agg(
        brk_share=('brk','size'), brk_mag=('brk','mean'))
    tot=tm.groupby(['pitcher_id','season']).size().rename('tot')
    a=a.join(ing).join(mixbrk).join(tot)
    a['brk_share']=a.brk_share/a.tot
    cols=[c for c in a.columns if c not in ('n','tot')]
    F={f'tx_{c}':np.full(len(RAW),np.nan,np.float32) for c in cols}
    ar=a.reset_index()
    for s in range(2020,2026):
        tgt=season==s; prev=ar[ar.season<s]
        if not tgt.any() or not len(prev): continue
        w=prev.groupby('pitcher_id').apply(lambda x: pd.Series(
            {c: np.average(x[c].fillna(x[c].mean()), weights=x.n) if x[c].notna().any() else np.nan for c in cols}))
        pid=pd.Series(RAW.pitcher_id.values[tgt])
        for c in cols: F[f'tx_{c}'][tgt]=pid.map(w[c]).values
    Fd=pd.DataFrame(F, index=RAW.index)
    # S4 구속-제구 트레이드오프 잔차: 구속으로 설명되는 제구 성공률 대비 실제
    ovr=asof_rate(np.ones(len(RAW),bool),'pitcher_id')
    ok=~np.isnan(Fd.tx_velo.values) & ~np.isnan(ovr)
    if ok.sum()>1000:
        c=np.polyfit(Fd.tx_velo.values[ok], ovr[ok], 1)
        Fd['tx_cmd_resid']=ovr-np.polyval(c, Fd.tx_velo.values)
    return Fd.astype(np.float32)

# ═══════ S6. 압박성향 재설계 (차이 대신 원값 2개) ═══════
def s6():
    hi=RAW.li.values>1.5; lo=RAW.li.values<0.5
    return pd.DataFrame({'p_hiLI_rate':asof_rate(hi,'pitcher_id',shrink=600),
                         'p_loLI_rate':asof_rate(lo,'pitcher_id',shrink=600),
                         'b_hiLI_rate':asof_rate(hi,'batter_id',shrink=600)},
                        index=RAW.index).astype(np.float32)

# ═══════ S7. 홈/원정 + 추운달 ═══════
def s7():
    home=(RAW.top_bottom.values=='T').astype(np.float32)   # 초 = 홈팀이 투구
    cold=RAW.game_month.isin([3,4,10]).values.astype(np.float32)
    cr=asof_rate(RAW.game_month.isin([3,4,10]).values,'pitcher_id',shrink=400)
    ov=asof_rate(np.ones(len(RAW),bool),'pitcher_id')
    return pd.DataFrame({'is_home_pitcher':home,'is_cold':cold,
                         'p_cold_rate_d':cr-ov,'home_x_li':home*RAW.li.values},
                        index=RAW.index).astype(np.float32)

# ═══════ S8. 주자 견제 부담 (투수별 민감도) ═══════
def s8():
    on1=RAW.runner_on_1b.values==1
    r1=asof_rate(on1,'pitcher_id',shrink=400); r0=asof_rate(~on1,'pitcher_id',shrink=400)
    return pd.DataFrame({'p_hold_pen':r1-r0, 'p_hold_x_on1':(r1-r0)*on1.astype(np.float32)},
                        index=RAW.index).astype(np.float32)

# ═══════ 실행 ═══════
GROUPS={}
for nm,fn in [('S1 투수x상황성적표',s1), ('S2 매치업이력',s2), ('S3-5 트랙맨(제어난이도)',s345),
              ('S6 압박성향(원값)',s6), ('S7 홈/추운달',s7), ('S8 견제부담',s8)]:
    try:
        t=time.time(); GROUPS[nm]=fn()
        log(f'{el()} 빌드 {nm}: {GROUPS[nm].shape[1]}개 ({time.time()-t:.0f}s)')
    except Exception: log(f'!! 빌드 {nm}\n'+traceback.format_exc())

log(f'\n{el()} ===== 기준선 (시드 {NSEED}회) =====')
b0=bench(X0, '기준선(deconv)')

log(f'\n{el()} ===== 개별 그룹 추가 =====')
res={}
for nm,Fd in GROUPS.items():
    try:
        Xa=pd.concat([X0, Fd], axis=1)
        res[nm]=bench(Xa, nm)-b0
        log(f'{el()}    -> 순효과 {res[nm]:+7.1f}')
    except Exception: log(f'!! {nm}\n'+traceback.format_exc())

log(f'\n{el()} ===== 양수 그룹 누적 결합 =====')
order=sorted(res, key=res.get, reverse=True)
cum=X0.copy(); acc=[]
for nm in order:
    if res[nm]<=0: break
    cum=pd.concat([cum,GROUPS[nm]],axis=1); acc.append(nm)
    bench(cum, '결합: '+'+'.join(a.split()[0] for a in acc))

log(f'\n{el()} ===== 전체 결합 (음수 포함) =====')
bench(pd.concat([X0]+list(GROUPS.values()),axis=1), '전체결합')

log(f'\n{el()} ===== 최종 하이퍼파라미터 재탐색 (최고 조합, 60회) =====')
Xb=cum if acc else X0
import xgboost as xgb
rng=np.random.RandomState(23); best=None
for i in range(60):
    prm=dict(n_estimators=int(rng.choice([300,500,700,1000])),
             learning_rate=float(rng.choice([0.005,0.008,0.012,0.02])),
             max_depth=int(rng.choice([4,5,6,7])),
             min_child_weight=float(rng.choice([500,1500,3000,6000])),
             subsample=float(rng.choice([0.6,0.8,1.0])),
             colsample_bytree=float(rng.choice([0.25,0.4,0.6])),
             reg_lambda=float(rng.choice([10,50,200,800])),
             reg_alpha=float(rng.choice([0,1,10])),
             device='cuda:1', tree_method='hist', eval_metric='logloss', verbosity=0)
    try:
        sc={}
        for vs,(tr,va) in SP.items():
            m=xgb.XGBClassifier(**prm, random_state=0).fit(Xb.loc[tr], y[tr])
            p=m.predict_proba(Xb.loc[va])[:,1]
            sc[vs]=evaluate(p,y[va],RP[vs])
            np.save(OUT+f'p4_{i}_{vs}.npy', p.astype(np.float32))
        a=(sc[2024]['raw']+sc[2023]['raw'])/2
        R.append(dict(name=f'HP{i}', avg=a, stage='hp',
                      **{f'{k}_{vs}':v for vs in sc for k,v in sc[vs].items()},
                      **{k:v for k,v in prm.items() if k not in ('device','tree_method','eval_metric','verbosity')}))
        f=''
        if best is None or a>best: best=a; f='  <== BEST'
        log(f'{el()} [HP{i:2d}] avg={a:7.1f} 24:{sc[2024]["raw"]:7.1f} 23:{sc[2023]["raw"]:7.1f} '
            f'd{prm["max_depth"]} lr{prm["learning_rate"]} n{prm["n_estimators"]}{f}')
    except Exception: log(f'!! HP{i}\n'+traceback.format_exc())
    if i%10==9: save()
save()
log(f'\n{el()} ===== 4차 완료 =====')
