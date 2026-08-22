"""
38차 — 구종 예측 모델 재구축 (옛 하이퍼파라미터로 남아 있던 마지막 축)

배경
  잔차 분해에서 남은 구조의 대부분이 구종에 몰려 있다.
      투수 x 구종  625.9   구종 단독 137.6   (둘 다 오라클, 사용 불가)
  구종 자체는 규칙상 못 쓰지만 '예측'은 합법이다.
  20차에서 해봤을 때 정확도 54.9%(최빈 52.3%), y 모델 이득 +1.8 로 접었다.

  그런데 그 구종 예측 모델이 옛 설정(d6/mcw1500/n600/lr0.008)이었다.
  이후 트리 재튜닝으로 d10/mcw6000/n2000/lr0.005 가 +48 을 냈는데
  구종 모델만 그 혜택을 못 받았다.

측정
  A 예측 품질   정확도 / 다중로그손실 / 클래스별
  B y 모델 이득  구종 OOF 를 피처로 넣었을 때 폴드2024/2023 동시 개선 여부
  C 오라클 대비  진짜 구종을 넣었을 때(상한)의 몇 %를 회수하나

채택 기준: both (두 폴드 동시 개선)
"""
import os, sys, time, warnings, json
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga')
import lib_lga as L

OUT='/home/lee/lga/results38/'; os.makedirs(OUT,exist_ok=True)
LOG=open(OUT+'log.txt','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
T0=time.time()
def el(): return f'[{(time.time()-T0)/60:5.1f}m]'
DEV=os.environ.get('LGA_DEV','cuda:1')
lgt=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))

b=L.load_base(); y=b['y']; RAW=b['RAW']; season=b['season']; isF=b['isF']
BASE=L.build_base114(b=b) if hasattr(L,'build_base114') else None
X=L.build_v7(b=b)
log(f'{el()} v7 피처 {X.shape[1]}')

# ── 구종 라벨 역산 ──
ordr=np.lexsort((RAW.asof_pitcher_n.values, RAW.pitcher_id.values))
pid=RAW.pitcher_id.values[ordr]; mx=RAW.asof_pitcher_pitchmix_n.values[ordr].astype(np.float64)
last=np.append(pid[1:]!=pid[:-1],True)
PT={}
for c in ['fastball','breaking','offspeed']:
    cum=np.nan_to_num(mx*RAW[f'asof_pitcher_{c}_rate'].values[ordr])
    d=np.append(cum[1:]-cum[:-1],np.nan); d[last]=np.nan
    v=np.round(d); v[np.abs(d-v)>0.3]=np.nan
    o=np.full(len(RAW),np.nan,np.float32); o[ordr]=v; PT[c]=o
P=pd.DataFrame(PT); ok=P.notna().all(1).values
cls=np.full(len(RAW),-1,np.int64)
for i,c in enumerate(['fastball','breaking','offspeed']):
    cls[ok&(P[c].values==1)]=i
valid=cls>=0
log(f'{el()} 구종 라벨 {valid.sum():,}  분포 {pd.Series(cls[valid]).value_counts(normalize=True).round(4).to_dict()}')

OLD=dict(n_estimators=600,learning_rate=0.008,max_depth=6,min_child_weight=1500,
         subsample=0.7,colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.)
NEW=dict(n_estimators=2000,learning_rate=0.005,max_depth=10,min_child_weight=6000,
         subsample=0.7,colsample_bytree=0.5,reg_lambda=50.,reg_alpha=1.)
COMMON=dict(tree_method='hist',device=DEV,verbosity=0,
            objective='multi:softprob',num_class=3,eval_metric='mlogloss')

def rss():
    try: return int(open(f'/proc/{os.getpid()}/status').read().split('VmRSS:')[1].split()[0])//1024
    except Exception: return -1

def pt_oof(prm, tag):
    """시즌별 인과 OOF (시즌 s 는 <s 로 학습). 시즌마다 체크포인트를 남긴다."""
    fin=OUT+f'pt_{tag}.npy'
    if os.path.exists(fin):
        log(f'{el()}   [{tag}] 완성본 재사용'); return np.load(fin)
    ck=OUT+f'pt_{tag}_ck.npy'
    O=np.load(ck) if os.path.exists(ck) else np.full((len(RAW),3),np.nan,np.float32)
    for s in range(2021,2025):
        tg=season==s
        if not np.isnan(O[tg]).any():
            log(f'{el()}   [{tag}] 시즌{s} 체크포인트 재사용'); continue
        tr=(season<s)&~(isF&(season<=2022)&(s>=2023))&valid
        m=xgb.XGBClassifier(**prm,**COMMON,random_state=0).fit(X[tr],cls[tr])
        O[tg]=m.predict_proba(X[tg])
        np.save(ck,O); del m
        log(f'{el()}   [{tag}] 구종 OOF 시즌{s} (학습 {tr.sum():,})  RSS {rss()}MB')
    np.save(fin,O)
    if os.path.exists(ck): os.remove(ck)
    return O

def quality(O, vs):
    va=(season==vs)&~isF&valid
    pr=O[va]; t=cls[va]
    acc=(pr.argmax(1)==t).mean()
    ll=-np.mean(np.log(np.clip(pr[np.arange(len(t)),t],1e-9,1)))
    maj=pd.Series(t).value_counts(normalize=True).max()
    return acc, ll, maj

def yscore(extra, name):
    cf=OUT+f'y_{name}.json'
    if os.path.exists(cf):
        r={int(k):v for k,v in json.load(open(cf)).items()}
        log(f'{el()}   [{name}] 캐시 재사용'); return r
    o={}
    for vs in (2024,2023):
        f=L.fold_ctx(vs,b=b); tr,va,w,yv,bq=f['tr'],f['va'],f['w'],f['yv'],f['base']
        Xa=X if extra is None else pd.concat([X,extra],axis=1)
        p=np.mean([xgb.XGBClassifier(n_estimators=2000,learning_rate=0.005,max_depth=10,
                   min_child_weight=6000,subsample=0.7,colsample_bytree=0.5,reg_lambda=50.,
                   reg_alpha=1.,tree_method='hist',device=DEV,eval_metric='logloss',
                   verbosity=0,random_state=s).fit(Xa[tr],y[tr],sample_weight=w)
                   .predict_proba(Xa[va])[:,1] for s in range(2)],0)
        o[vs]=L.bss(p,yv,bq)
        del f,tr,va,w,yv,Xa,p
    json.dump({str(k):v for k,v in o.items()},open(cf,'w'))
    return o

log(f'\n{el()} ===== A. 구종 예측 품질 =====')
Oo=pt_oof(OLD,'old'); On=pt_oof(NEW,'new')
for tag,O in (('옛설정',Oo),('신설정',On)):
    for vs in (2024,2023):
        a,l,mj=quality(O,vs)
        log(f'{el()}   {tag} 폴드{vs}  정확도 {a:.4f} (최빈 {mj:.4f})  다중로그손실 {l:.4f}')

log(f'\n{el()} ===== B. y 모델 이득 =====')
base=yscore(None,'기준')
log(f'{el()}   기준 (v7 120)          24: {base[2024]:7.1f}  23: {base[2023]:7.1f}')
res={}
for tag,O in (('옛설정',Oo),('신설정',On)):
    E=pd.DataFrame(lgt(np.clip(O,1e-6,1-1e-6)),columns=[f'pt_{c}' for c in ('fb','br','os')],index=RAW.index)
    r=yscore(E,tag); res[tag]=r
    d24=r[2024]-base[2024]; d23=r[2023]-base[2023]
    both='O 채택가능' if (d24>0 and d23>0) else ''
    log(f'{el()}   +구종OOF {tag:5s}       24: {r[2024]:7.1f} ({d24:+5.1f})  23: {r[2023]:7.1f} ({d23:+5.1f})  {both}')

log(f'\n{el()} ===== C. 오라클 상한 =====')
ORC=pd.DataFrame({f'true_{c}':(cls==i).astype(np.float32) for i,c in enumerate(('fb','br','os'))},index=RAW.index)
ORC[~valid]=np.nan
orc=yscore(ORC,'오라클')
log(f'{el()}   +진짜구종 (사용불가)      24: {orc[2024]:7.1f} ({orc[2024]-base[2024]:+5.1f})  23: {orc[2023]:7.1f} ({orc[2023]-base[2023]:+5.1f})')
for tag in ('옛설정','신설정'):
    g=res[tag][2024]-base[2024]; o=orc[2024]-base[2024]
    log(f'{el()}   {tag} 회수율 (폴드2024)  {100*g/o if o>0 else float("nan"):.1f}%')
log(f'{el()} 완료')
