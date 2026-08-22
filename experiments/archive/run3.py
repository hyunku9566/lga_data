"""
3차 심층 실험 (~4시간). 지금까지 확정된 사실 위에서만 판다.

확정:
  F1. 역산이 유일한 대형 레버 (base48 -> deconv: 2024 +168, 2023 +243, 양폴드 일관)
  F2. 트랙맨 구위 / 압박성향 = 순효과 0 -> 주력에서 제외, 대조군으로만 유지
  F3. 지배적 실패모드는 과적합. 최적 하이퍼파라미터는 저용량 극단에 있음
  F4. 트렌드 재보정은 폴드마다 부호가 뒤집힘 -> raw 를 기본으로

3차에서 팔 것:
  E 역산 자체를 깊게 튜닝 (축소강도/축소대상/원시카운트/오프셋)
  F 구체제 F 처리 정밀화 (제외 vs 가중 vs 체제플래그)
  G 대규모 하이퍼파라미터 탐색 (저용량 영역 집중, 250회)
  H 손실함수 (logloss vs Brier 직접)
  I 신경망 3차 (소형 + 조기종료 스윕)
  J 캘리브레이션 대안 + 앙상블
"""
import os, time, warnings, traceback, itertools
import numpy as np, pandas as pd, scipy.special as sp
warnings.filterwarnings('ignore')

D = '/home/lee/lga/'; OUT = D + 'results3/'
os.makedirs(OUT, exist_ok=True)
LOG = open(OUT + 'log.txt', 'a', buffering=1)
def log(*a):
    m = ' '.join(str(x) for x in a); print(m); LOG.write(m + '\n')
T0 = time.time()
def el(): return f'[{(time.time()-T0)/60:6.1f}m]'

R = []
def save(): pd.DataFrame(R).to_csv(OUT + 'results3.csv', index=False)

# ═══════════ 데이터 ═══════════
log(f'{el()} 원본 로드')
RAW = pd.read_csv(D + 'data/train.csv', encoding='utf-8-sig')
y = RAW.control_success.values.astype(np.float32)
season = RAW.season.values
gtF = (RAW.game_type.values == 'F')
BASE = pd.read_parquet(D + 'features.parquet')
KEEP = [c for c in BASE.columns if not c.startswith(('__', 'tm_')) and 'clutch' not in c and 'ctrl' not in c]
log(f'{el()} 기본 피처셋 {len(KEEP)}개 (트랙맨/clutch 제외 = F2)')

SP = {vs: ((season < vs) & ~(gtF & (season <= 2022)), season == vs) for vs in (2024, 2023)}

def trend_prior(vs):
    m = season < vs; s = pd.Series(y[m]).groupby(season[m]).mean()
    return float(sp.expit(np.polyval(np.polyfit(s.index, sp.logit(s.values), 1), vs)))
RP = {vs: trend_prior(vs) for vs in (2024, 2023)}

def evaluate(p, yv, rp):
    r = yv.mean(); ref = r * (1 - r)
    b = lambda q: 100000 * max(0.0, 1 - np.mean((q - yv) ** 2) / ref)
    lo = sp.logit(np.clip(p, 1e-6, 1 - 1e-6))
    return dict(raw=b(p), trend=b(sp.expit(lo - lo.mean() + sp.logit(rp))),
                oracle=b(sp.expit(lo - lo.mean() + sp.logit(r))), meanp=float(p.mean()), sdp=float(p.std()))

# ═══════════ 역산 빌더 (변형 가능) ═══════════
SPECS = [('pitcher_id','asof_pitcher_n','asof_pitcher_success_rate','p_succ'),
         ('pitcher_id','asof_pitcher_n','asof_pitcher_reverse_rate','p_rev'),
         ('pitcher_id','asof_pitcher_n','asof_pitcher_middle_rate','p_mid'),
         ('pitcher_id','asof_pitcher_n','asof_pitcher_ball_rate','p_ball'),
         ('pitcher_id','asof_pitcher_n','asof_pitcher_strike_rate','p_stk'),
         ('batter_id','asof_batter_n','asof_batter_success_rate','b_succ'),
         ('batter_id','asof_batter_n','asof_batter_middle_rate','b_mid')]

_anchor_cache = {}
def anchors(idcol, ncol, ratecol):
    key = (idcol, ncol, ratecol)
    if key not in _anchor_cache:
        t = RAW[[idcol,'season',ncol,ratecol]].copy()
        t['succ'] = t[ncol]*t[ratecol].fillna(0)
        _anchor_cache[key] = t.loc[t.groupby([idcol,'season'])[ncol].idxmin()] \
                              .set_index([idcol,'season'])[[ncol,'succ']]
    return _anchor_cache[key]

def build_deconv(k=150.0, target='league', raw_counts=False):
    """target: 'league'=리그평균으로 축소, 'career'=그 선수 커리어값으로 축소(경험적 베이즈)"""
    F = {}
    for idcol, ncol, ratecol, pref in SPECS:
        S = anchors(idcol, ncol, ratecol)
        a = RAW[[idcol,'season']].join(S, on=[idcol,'season'])
        a_n = a[ncol].fillna(0).values; a_s = a['succ'].fillna(0).values
        n = RAW[ncol].values; r = RAW[ratecol].values
        dn = np.maximum(n-a_n, 0); ds = np.maximum(np.nan_to_num(n*r)-a_s, 0)
        lg = np.nanmean(r)
        tgt = np.nan_to_num(r, nan=lg) if target == 'career' else np.full(len(RAW), lg)
        F[pref+'_ssn'] = (ds + k*tgt)/(dn + k)
        F[pref+'_ssn_vs_car'] = F[pref+'_ssn'] - np.nan_to_num(r, nan=lg)
        if raw_counts:
            F[pref+'_ssn_succ'] = ds
            F[pref+'_ssn_rawrate'] = np.where(dn > 0, ds/np.maximum(dn,1), np.nan)
        if pref in ('p_succ','b_succ'):
            F[pref+'_ssn_n'] = dn
            F[pref+'_ssn_logn'] = np.log1p(dn)
    return pd.DataFrame(F, index=RAW.index).astype(np.float32)

def assemble(dec):
    X = BASE[[c for c in KEEP if not c.startswith(('p_succ_ssn','p_rev_ssn','p_mid_ssn','p_ball_ssn',
                                                   'p_stk_ssn','b_succ_ssn','b_mid_ssn'))]].copy()
    for c in dec.columns: X[c] = dec[c].values
    return X

# ═══════════ 공통 fit ═══════════
GOOD = dict(n_estimators=600, learning_rate=0.008, max_depth=6, min_child_weight=1500,
            subsample=0.7, colsample_bytree=0.5, reg_lambda=50.0, reg_alpha=1.0,
            device='cuda', tree_method='hist', eval_metric='logloss', verbosity=0)

def run_cfg(X, prm, tag, extra=None, base_margin=None, weights=None, tell=True):
    import xgboost as xgb
    sc = {}
    for vs, (tr, va) in SP.items():
        m = xgb.XGBClassifier(**prm)
        kw = {}
        if base_margin is not None: kw['base_margin'] = base_margin[tr]
        if weights is not None: kw['sample_weight'] = weights[tr]
        m.fit(X.loc[tr], y[tr], **kw)
        if base_margin is not None:
            p = sp.expit(m.predict(X.loc[va], output_margin=True) + base_margin[va])
        else:
            p = m.predict_proba(X.loc[va])[:, 1]
        sc[vs] = evaluate(p, y[va], RP[vs])
        np.save(OUT + f'p_{tag}_{vs}.npy', p.astype(np.float32))
    avg = (sc[2024]['raw'] + sc[2023]['raw']) / 2
    rec = dict(tag=tag, avg_raw=avg, **{f'{k}_{vs}': v for vs in sc for k, v in sc[vs].items()})
    if extra: rec.update(extra)
    R.append(rec)
    if tell:
        log(f'{el()} {tag:38s} avg={avg:7.1f}  24:{sc[2024]["raw"]:7.1f}({sc[2024]["trend"]:7.1f})  '
            f'23:{sc[2023]["raw"]:7.1f}({sc[2023]["trend"]:7.1f})')
    return avg

# ═══════════ E. 역산 튜닝 ═══════════
def stageE():
    log(f'\n{el()} ===== E. 역산 심층 튜닝 =====')
    for k in [25, 50, 100, 150, 300, 600, 1200]:
        for tgt in ['league', 'career']:
            X = assemble(build_deconv(k=k, target=tgt))
            run_cfg(X, GOOD, f'E_k{k}_{tgt}', dict(stage='E', k=k, target=tgt))
    log(f'\n{el()} --- 원시 카운트 추가 ---')
    for k in [100, 300]:
        X = assemble(build_deconv(k=k, target='career', raw_counts=True))
        run_cfg(X, GOOD, f'E_k{k}_career_raw', dict(stage='E', k=k, target='career_raw'))
    save()

# ═══════════ F. 구체제 F 처리 ═══════════
def stageF(dec):
    log(f'\n{el()} ===== F. game_type=F 처리 정밀화 =====')
    X = assemble(dec); Xf = X.copy()
    Xf['is_oldF'] = (gtF & (season <= 2022)).astype(np.float32)
    Xf['is_F'] = gtF.astype(np.float32)
    global SP
    orig = SP
    for name, mk in [('제외(기준)',   lambda vs: (season<vs) & ~(gtF & (season<=2022))),
                     ('전부사용',      lambda vs: season<vs),
                     ('F전체제외',     lambda vs: (season<vs) & ~gtF),
                     ('R만+최근F',    lambda vs: (season<vs) & (~gtF | (season>=2023)))]:
        SP = {vs: (mk(vs), season==vs) for vs in (2024,2023)}
        run_cfg(X, GOOD, f'F_{name}', dict(stage='F', how=name))
    SP = orig
    run_cfg(Xf, GOOD, 'F_체제플래그피처', dict(stage='F', how='flag'))
    for w in [0.1, 0.3, 0.5]:
        ww = np.where(gtF & (season<=2022), w, 1.0).astype(np.float32)
        SP = {vs: ((season<vs), season==vs) for vs in (2024,2023)}
        run_cfg(X, GOOD, f'F_가중{w}', dict(stage='F', how=f'weight{w}'), weights=ww)
    SP = orig
    # 최근시즌 가중
    for hl in [1.0, 2.0, 4.0]:
        ww = (0.5 ** ((season.max()-season)/hl)).astype(np.float32)
        run_cfg(X, GOOD, f'F_최근가중hl{hl}', dict(stage='F', how=f'recency{hl}'), weights=ww)
    save()

# ═══════════ G. 대규모 하이퍼파라미터 탐색 ═══════════
def stageG(dec, n=250):
    log(f'\n{el()} ===== G. 하이퍼파라미터 탐색 {n}회 =====')
    X = assemble(dec); rng = np.random.RandomState(11); best = None
    for i in range(n):
        prm = dict(n_estimators=int(rng.choice([200,350,500,700,1000,1400])),
                   learning_rate=float(rng.choice([0.003,0.005,0.008,0.012,0.02,0.03])),
                   max_depth=int(rng.choice([3,4,5,6,7])),
                   min_child_weight=float(rng.choice([300,800,1500,3000,6000])),
                   subsample=float(rng.choice([0.5,0.7,0.9,1.0])),
                   colsample_bytree=float(rng.choice([0.25,0.4,0.55,0.7])),
                   reg_lambda=float(rng.choice([5,30,100,400,1500])),
                   reg_alpha=float(rng.choice([0,1,5,30])),
                   max_bin=int(rng.choice([64,128,256])),
                   device='cuda', tree_method='hist', eval_metric='logloss', verbosity=0)
        try:
            a = run_cfg(X, prm, f'G{i}', dict(stage='G', i=i, **{k:v for k,v in prm.items()
                        if k not in ('device','tree_method','eval_metric','verbosity')}), tell=False)
            f = ''
            if best is None or a > best: best = a; f = '  <== BEST'
            log(f'{el()} [G{i:3d}] avg={a:7.1f} d{prm["max_depth"]} lr{prm["learning_rate"]} '
                f'n{prm["n_estimators"]} mcw{prm["min_child_weight"]:.0f} L2:{prm["reg_lambda"]:.0f} '
                f'L1:{prm["reg_alpha"]:.0f} bin{prm["max_bin"]}{f}')
        except Exception: log(f'!! G{i}\n'+traceback.format_exc())
        if i % 20 == 19: save()
    save()

# ═══════════ H. 손실함수 + 오프셋 ═══════════
def stageH(dec):
    log(f'\n{el()} ===== H. 손실함수 / 오프셋 모델 =====')
    X = assemble(dec)
    for obj in ['binary:logistic', 'reg:squarederror']:
        prm = dict(GOOD); prm['objective'] = obj
        if obj == 'reg:squarederror':
            import xgboost as xgb
            sc = {}
            for vs,(tr,va) in SP.items():
                m = xgb.XGBRegressor(**{k:v for k,v in prm.items() if k!='eval_metric'}).fit(X.loc[tr], y[tr])
                p = np.clip(m.predict(X.loc[va]), 0.01, 0.99)
                sc[vs] = evaluate(p, y[va], RP[vs]); np.save(OUT+f'p_H_brier_{vs}.npy', p.astype(np.float32))
            avg=(sc[2024]['raw']+sc[2023]['raw'])/2
            R.append(dict(tag='H_brier직접', avg_raw=avg, stage='H',
                          **{f'{k}_{vs}':v for vs in sc for k,v in sc[vs].items()}))
            log(f'{el()} H_brier직접  avg={avg:7.1f}  24:{sc[2024]["raw"]:7.1f}  23:{sc[2023]["raw"]:7.1f}')
        else:
            run_cfg(X, prm, 'H_logloss', dict(stage='H', obj=obj))
    # 오프셋: 역산 시즌율을 base_margin 으로 넣고 잔차만 학습
    log(f'{el()} --- 오프셋(base_margin) 모델 ---')
    for a in [0.5, 1.0]:
        pr = np.clip(dec['p_succ_ssn'].values, 0.02, 0.98)
        bm = (a * sp.logit(pr)).astype(np.float32)
        run_cfg(X, GOOD, f'H_offset{a}', dict(stage='H', offset=a), base_margin=bm)
    save()

# ═══════════ I. 신경망 3차 ═══════════
def stageI(dec):
    import torch, torch.nn as nn
    log(f'\n{el()} ===== I. 소형 신경망 스윕 =====')
    dev='cuda:0'; X = assemble(dec)
    CAT=['pitcher_id','batter_id','pitcher_team_id','batter_team_id','pitcher_hand',
         'batter_hand','base_state','game_type','top_bottom']
    num=[c for c in X.columns if c not in CAT]
    Xc=X[CAT].values.astype(np.int64)
    Xn=np.nan_to_num(X[num].values.astype(np.float32),nan=0.,posinf=0.,neginf=0.)
    Xn=np.clip((Xn-Xn.mean(0))/(Xn.std(0)+1e-6),-5,5)
    card=[int(Xc[:,i].max())+2 for i in range(len(CAT))]
    mth=RAW.game_month.values.astype(np.float32)
    Tf=np.stack([np.sin(2*np.pi*mth/12),np.cos(2*np.pi*mth/12),
                 np.sin(4*np.pi*mth/12),np.cos(4*np.pi*mth/12)],1).astype(np.float32)
    tn=torch.tensor(Xn,device=dev); tc=torch.tensor(Xc,device=dev)
    tt=torch.tensor(Tf,device=dev); ty=torch.tensor(y,device=dev)

    class Net(nn.Module):
        def __init__(s,n,card,k=8,h=64,emb=0,drop=.2,film=True):
            super().__init__(); s.k,s.emb,s.film=k,emb,film; d=n
            if emb:
                s.e1=nn.Embedding(card[0],emb); s.e2=nn.Embedding(card[1],emb)
                nn.init.normal_(s.e1.weight,std=.01); nn.init.normal_(s.e2.weight,std=.01); d+=2*emb
            s.r1=nn.Parameter(torch.randn(k,d)*.1+1); s.l1=nn.Linear(d,h)
            s.dp=nn.Dropout(drop); s.l2=nn.Linear(h,h)
            s.hd=nn.Parameter(torch.randn(k,h)*.02); s.hb=nn.Parameter(torch.zeros(k))
            if film: s.fn=nn.Sequential(nn.Linear(4,16),nn.ReLU(),nn.Linear(16,2*h))
        def forward(s,xn,xc,t):
            z=xn
            if s.emb: z=torch.cat([z,s.e1(xc[:,0]),s.e2(xc[:,1])],-1)
            z=z.unsqueeze(1)*s.r1; z=s.dp(torch.relu(s.l1(z)))
            if s.film:
                g,b=s.fn(t).chunk(2,-1); z=z*(1+g.unsqueeze(1))+b.unsqueeze(1)
            z=torch.relu(s.l2(z))
            return ((z*s.hd).sum(-1)+s.hb).mean(1)

    grid=[]
    for k in [4,8,16]:
        for h in [32,64,128]:
            for wd in [3e-3,1e-2,5e-2]:
                for emb in [0,4]:
                    grid.append(dict(k=k,h=h,wd=wd,emb=emb,drop=0.2,lr=1e-3))
    rng=np.random.RandomState(3); rng.shuffle(grid); grid=grid[:22]
    for ci,cfg in enumerate(grid):
        sc={}
        try:
            for vs,(tr,va) in SP.items():
                torch.manual_seed(0)
                ia=np.where(tr)[0]; rs=np.random.RandomState(1); rs.shuffle(ia)
                nin=int(len(ia)*.05); iin,itr=ia[:nin],ia[nin:]; iva=np.where(va)[0]
                net=Net(Xn.shape[1],card,k=cfg['k'],h=cfg['h'],emb=cfg['emb'],drop=cfg['drop']).to(dev)
                opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
                B=16384; bv,bad,bs=1e9,0,None
                for ep in range(50):
                    net.train(); perm=np.random.RandomState(ep).permutation(itr)
                    for j in range(0,len(perm),B):
                        b=torch.tensor(perm[j:j+B],device=dev)
                        l=((torch.sigmoid(net(tn[b],tc[b],tt[b]))-ty[b])**2).mean()
                        opt.zero_grad(); l.backward(); opt.step()
                    net.eval()
                    with torch.no_grad():
                        bi=torch.tensor(iin,device=dev)
                        v=((torch.sigmoid(net(tn[bi],tc[bi],tt[bi]))-ty[bi])**2).mean().item()
                    if v<bv-1e-7: bv,bad=v,0; bs={q:t.detach().clone() for q,t in net.state_dict().items()}
                    else:
                        bad+=1
                        if bad>=5: break
                net.load_state_dict(bs); net.eval(); ps=[]
                with torch.no_grad():
                    for j in range(0,len(iva),32768):
                        b=torch.tensor(iva[j:j+32768],device=dev)
                        ps.append(torch.sigmoid(net(tn[b],tc[b],tt[b])).cpu().numpy())
                p=np.concatenate(ps); sc[vs]=evaluate(p,y[va],RP[vs])
                np.save(OUT+f'p_I{ci}_{vs}.npy',p.astype(np.float32))
            avg=(sc[2024]['raw']+sc[2023]['raw'])/2
            R.append(dict(tag=f'I{ci}',avg_raw=avg,stage='I',**cfg,
                          **{f'{k}_{vs}':v for vs in sc for k,v in sc[vs].items()}))
            log(f'{el()} [I{ci:2d}] avg={avg:7.1f} 24:{sc[2024]["raw"]:7.1f} 23:{sc[2023]["raw"]:7.1f} '
                f'k{cfg["k"]} h{cfg["h"]} wd{cfg["wd"]} emb{cfg["emb"]}')
        except Exception: log(f'!! I{ci}\n'+traceback.format_exc())
        save()

# ═══════════ J. 캘리브레이션 + 앙상블 ═══════════
def stageJ():
    log(f'\n{el()} ===== J. 캘리브레이션 / 앙상블 =====')
    df=pd.DataFrame(R)
    g=df[df.stage=='G'].sort_values('avg_raw',ascending=False)
    tags=list(g.tag.head(8)) if len(g) else []
    nn=df[df.stage=='I'].sort_values('avg_raw',ascending=False)
    if len(nn): tags+= list(nn.tag.head(2))
    log(f'{el()} 앙상블 후보 {tags}')
    for vs,(tr,va) in SP.items():
        yv=y[va]; ps=[]
        for t in tags:
            f=OUT+f'p_{t}_{vs}.npy'
            if os.path.exists(f): ps.append(np.load(f))
        if len(ps)<2: continue
        L=np.mean([sp.logit(np.clip(p,1e-6,1-1e-6)) for p in ps],0)
        r=evaluate(sp.expit(L),yv,RP[vs]); r.update(stage='J',tag=f'앙상블{len(ps)}',val=vs)
        R.append(r); log(f'{el()} [J] 로짓평균앙상블({len(ps)}) val{vs} raw={r["raw"]:7.1f} trend={r["trend"]:7.1f} oracle={r["oracle"]:7.1f}')
        m=L.mean()
        for s in [0.8,0.9,1.0,1.1,1.2]:
            r=evaluate(sp.expit(m+s*(L-m)),yv,RP[vs])
            R.append(dict(stage='J_shrink',shrink=s,val=vs,**r))
            log(f'{el()}   축소 s={s:.1f} raw={r["raw"]:7.1f}')
        # 월별 재보정 (학습시즌 월패턴으로)
        for b in [-0.02,-0.01,0.0,0.01,0.02]:
            r=evaluate(np.clip(sp.expit(L)+b,.01,.99),yv,RP[vs])
            R.append(dict(stage='J_shift',shift=b,val=vs,**r))
            log(f'{el()}   절편이동 {b:+.3f} raw={r["raw"]:7.1f}')
    save()

# ═══════════ main ═══════════
dec_default = build_deconv(k=150., target='league')
for fn, args in [(stageE,()), (stageF,(dec_default,)), (stageG,(dec_default,250)),
                 (stageH,(dec_default,)), (stageI,(dec_default,)), (stageJ,())]:
    try: fn(*args)
    except Exception: log(f'!! {fn.__name__}\n'+traceback.format_exc())
save()
log(f'\n{el()} ===== 3차 완료 =====')
