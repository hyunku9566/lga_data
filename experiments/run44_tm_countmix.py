"""
44차 — 투수 x 볼카운트별 구종배합 / 물리 프로파일  (GPU0)

배경
  잔차 분해: 투수x구종 625.9 (오라클), 투수x볼카운트 178.8.
  현재 피처에는 투수 '전체' 구종배합(asof_pitcher_fastball/breaking/offspeed_rate)만
  있고 카운트별 배합이 없다. 투수는 3-0 에서 직구, 0-2 에서 변화구를 던진다.
  구종 예측기가 55.1% 에 머문 것도, y 모델이 카운트별 배합을 못 보는 것도 여기서 온다.

  트랙맨 tm5 8개는 전부 가공 집계지표(VAA/HAA/이상치/터널링)라 배합과 무관하다.

측정 (전부 시즌 인과적 <s, 조회키 = 투수 x 카운트. 행 독립성 유지)
  M1 구종배합   투수x카운트 fb/br/os 비율 + 전체배합 대비 편차          (6)
  M2 물리       투수x카운트 평균 구속/회전/수직무브/수평무브             (4)
  M3 둘 다
  M4 구종예측기 개선  M1+M2 를 구종 예측기 입력에 넣어 정확도/회수율 재측정

채택 기준: bench2 (두 폴드 동시 개선)
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga')
import lib_lga as L

OUT='/home/lee/lga/results44/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'
DEV=os.environ.get('LGA_DEV','cuda:0')
lgt=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))

b=L.load_base(); R=b['RAW']; y=b['y']; season=b['season']; isF=b['isF']
X0=L.build_v7(b=b).astype(np.float32)
pid=R.pitcher_id.values.astype(np.int64)
cnt=R.balls_before.values.astype(np.int64)*10+R.strikes_before.values.astype(np.int64)
log(f'{el()} v7 {X0.shape[1]} 피처')

comp, cls, valid = L.recover_labels(R)
log(f'{el()} 역산 구종 {valid.sum():,} ({valid.mean()*100:.1f}%)')

AL=pd.read_parquet('/home/lee/lga/aligned.parquet')
log(f'{el()} aligned {AL.shape[0]:,}행')
AL['cnt']=AL.balls_before.values*10+AL.strikes_before.values
PHYS=['rel_speed','spin_rate','induced_vert_break','horz_break']

def shrunk(num, den, prior_a, prior_b, k=100.):
    """계층 축소: 투수전체(prior_a) 와 리그카운트(prior_b) 로 반반."""
    return (num + (k*.5)*prior_a + (k*.5)*prior_b)/(den + k)

def mix_feats():
    """M1 투수x카운트 구종배합 (역산 라벨 사용, 커버리지 99.9%)."""
    names=[f'mx_{c}' for c in ('fb','br','os')]+[f'mx_{c}_d' for c in ('fb','br','os')]
    F={c:np.full(len(R),np.nan,np.float32) for c in names}
    for s in range(2020,2026):
        tgt=season==s; prev=(season<s)&valid
        if not tgt.any() or not prev.any(): continue
        dp=pd.DataFrame({'pid':pid[prev],'cnt':cnt[prev],'c':cls[prev]})
        pt=pd.Series(pid[tgt]); ct=pd.Series(cnt[tgt])
        kk=pd.Series(pid[tgt]*100+cnt[tgt])
        kprev=dp.pid.values*100+dp.cnt.values
        for i,nm in enumerate(('fb','br','os')):
            ind=(dp.c.values==i).astype(np.float64)
            gk=pd.Series(ind).groupby(kprev).agg(['sum','size'])
            gp=pd.Series(ind).groupby(dp.pid.values).mean()
            gl=pd.Series(ind).groupby(dp.cnt.values).mean()
            mu=float(ind.mean())
            num=kk.map(gk['sum']).fillna(0).values; den=kk.map(gk['size']).fillna(0).values
            pa=pt.map(gp).fillna(mu).values; pb=ct.map(gl).fillna(mu).values
            v=shrunk(num,den,pa,pb)
            F[f'mx_{nm}'][tgt]=v; F[f'mx_{nm}_d'][tgt]=v-pa
    return pd.DataFrame(F,index=R.index).astype(np.float32)

def phys_feats():
    """M2 투수x카운트 평균 물리량 (트랙맨)."""
    names=[f'ph_{c}' for c in PHYS]+[f'ph_{c}_d' for c in PHYS]
    F={c:np.full(len(R),np.nan,np.float32) for c in names}
    for s in range(2020,2026):
        tgt=season==s; pv=AL.season.values<s
        if not tgt.any() or not pv.any(): continue
        A=AL[pv]
        kprev=A.pitcher_id.values.astype(np.int64)*100+A.cnt.values
        kk=pd.Series(pid[tgt]*100+cnt[tgt]); pt=pd.Series(pid[tgt]); ct=pd.Series(cnt[tgt])
        for c in PHYS:
            v0=A[c].values.astype(np.float64)
            ok=np.isfinite(v0)
            gk=pd.Series(v0[ok]).groupby(kprev[ok]).agg(['sum','size'])
            gp=pd.Series(v0[ok]).groupby(A.pitcher_id.values[ok]).mean()
            gl=pd.Series(v0[ok]).groupby(A.cnt.values[ok]).mean()
            mu=float(np.nanmean(v0))
            num=kk.map(gk['sum']).fillna(0).values; den=kk.map(gk['size']).fillna(0).values
            pa=pt.map(gp).fillna(mu).values; pb=ct.map(gl).fillna(mu).values
            v=np.where(den>0, shrunk(num,den,pa,pb,k=20.), pa)
            F[f'ph_{c}'][tgt]=v; F[f'ph_{c}_d'][tgt]=v-pa
    return pd.DataFrame(F,index=R.index).astype(np.float32)

t=time.time(); M1=mix_feats(); log(f'{el()} M1 구종배합 {M1.shape[1]}개 ({time.time()-t:.0f}s)')
t=time.time(); M2=phys_feats(); log(f'{el()} M2 물리 {M2.shape[1]}개 ({time.time()-t:.0f}s)')
M1.to_parquet(OUT+'M1.parquet'); M2.to_parquet(OUT+'M2.parquet')

log(f'\n{el()} ===== y 모델 이득 =====')
base=L.bench2(X0, name='기준 v7', nseed=2, log=log)
BL=(base['m24'],base['m23'])
for nm,E in [('M1 구종배합', M1), ('M2 물리', M2), ('M3 둘다', pd.concat([M1,M2],axis=1))]:
    L.bench2(pd.concat([X0,E],axis=1), name=nm, nseed=2, baseline=BL, log=log)

# ── M4: 개선된 구종 예측기 ──
log(f'\n{el()} ===== M4 구종 예측기 개선 =====')
PRM=dict(n_estimators=2000,learning_rate=0.005,max_depth=10,min_child_weight=6000,
         subsample=0.7,colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.,
         tree_method='hist',device=DEV,verbosity=0,
         objective='multi:softprob',num_class=3,eval_metric='mlogloss')
def pt_oof(Xp, tag):
    f=OUT+f'pt_{tag}.npy'
    if os.path.exists(f): return np.load(f)
    O=np.full((len(R),3),np.nan,np.float32)
    for s in range(2021,2025):
        tr=(season<s)&~(isF&(season<=2022)&(s>=2023))&valid; tg=season==s
        m=xgb.XGBClassifier(**PRM,random_state=0).fit(Xp[tr],cls[tr])
        O[tg]=m.predict_proba(Xp[tg]); del m
        log(f'{el()}   [{tag}] 시즌{s}')
    np.save(f,O); return O
for tag,Xp in [('plain',X0), ('mix',pd.concat([X0,M1,M2],axis=1))]:
    O=pt_oof(Xp,tag)
    for vs in (2024,2023):
        va=(season==vs)&~isF&valid; pr=O[va]; t2=cls[va]
        acc=(pr.argmax(1)==t2).mean()
        ll=-np.mean(np.log(np.clip(pr[np.arange(len(t2)),t2],1e-9,1)))
        log(f'{el()}   구종예측 {tag:5s} 폴드{vs}  정확도 {acc:.4f}  로그손실 {ll:.4f}')
    E=pd.DataFrame(lgt(np.clip(O,1e-6,1-1e-6)),columns=[f'pt_{c}' for c in ('fb','br','os')],index=R.index)
    L.bench2(pd.concat([X0,E],axis=1), name=f'+구종OOF({tag})', nseed=2, baseline=BL, log=log)
log(f'{el()} 완료')
