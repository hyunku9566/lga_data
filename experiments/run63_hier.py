"""
63차 — 처음부터 다시: 계층 베이즈 모형 (기존 124피처 파이프라인 미사용)

설계 근거는 전부 데이터에서 직접 측정한 것이다.
    분산성분(확률단위)  투수간 .002106 / 시즌|투수 .000283 / 경기|시즌 .003878
    경기 자기상관 ρ=0.179,  시즌간 자기상관 -0.049 (평균회귀)
    접근가능 상한 재계산: 투수간 842 + 시즌 113 + 경기 50 + 타자/상황 ~150 ≈ 1155

모형 (전부 로짓 공간)
    logit P(y) = λ(시즌) + θ_p(t) + φ_b + s(상황)
      λ  리그 수준 (학습기간 추세로 외삽 — 미래를 안 본다)
      θ_p(t) 투수 현재능력의 EB 사후평균. 4계층 정밀도 가중:
             리그 <- 커리어(직전시즌까지) <- 당해시즌 진행분 <- 최근 5경기
             각 단계의 사전분산 = 위에서 측정한 혁신분산
      φ_b  타자 EB 효과
      s()  상황 효과 (볼카운트12 x 아웃 x 주자) — 잔차에서 추정

추론 입력은 전부 행 하나에서 나온다:
    커리어 총계 = train 조회(투수/타자 id),  당해시즌 = asof 역산,  최근폼 = prev1/3/5
"""
import os, sys, warnings
import numpy as np, pandas as pd, scipy.special as sp
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga'); import lib_lga as L

OUT='/home/lee/lga/results63/'; log,_=L.mklog(OUT)
b=L.load_base(); R=b['RAW']; y=b['y']; season=b['season']; isF=b['isF']
lgt=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))

pid=R.pitcher_id.values; bid=R.batter_id.values
n_as=R.asof_pitcher_n.values.astype(np.float64)
cum =n_as*np.nan_to_num(R.asof_pitcher_success_rate.values)
bn_as=R.asof_batter_n.values.astype(np.float64)
bcum =bn_as*np.nan_to_num(R.asof_batter_success_rate.values)
prev1=R.asof_pitcher_prev1_game_success_rate.values
prev3=R.asof_pitcher_prev3_game_success_rate.values
prev5=R.asof_pitcher_prev5_game_success_rate.values
cnt=(R.balls_before.values*10+R.strikes_before.values).astype(np.int64)
sit=cnt*100+R.outs_before.values.astype(np.int64)*10+np.minimum(R.num_runners_on.values,3).astype(np.int64)

# 확률단위 분산성분 -> 로짓단위로 환산 (p≈0.5 에서 dlogit/dp = 1/(p(1-p)) = 4)
V_PIT,V_SSN,V_GAME=np.load('/tmp/claude-1000/-home-lee-lga/a68aaebc-1ad3-4cca-a241-1603962ba966/scratchpad/vc.npy')
J=1/0.25
T_PIT,T_SSN,T_GAME=V_PIT*J*J, V_SSN*J*J, V_GAME*J*J*0.179**2
T_BAT=0.0006*J*J
log(f'로짓 사전분산  투수 {T_PIT:.4f}  시즌 {T_SSN:.4f}  경기(전이분) {T_GAME:.4f}')

def eb(obs_n, obs_s, prior_mu, tau):
    """이항관측을 정밀도 가중으로 prior_mu(로짓)에 수축. 관측정밀도 = n*p(1-p)"""
    n=np.maximum(obs_n,0); rate=np.divide(obs_s,np.maximum(n,1))
    z=lgt(np.clip(rate,1e-3,1-1e-3))
    prec_o=n*0.25                       # 로짓 관측정밀도
    w=prec_o/(prec_o+1.0/max(tau,1e-12))
    return np.where(n>0, w*z+(1-w)*prior_mu, prior_mu)

def run(vs):
    tr=(season<vs)&~(isF&(season<=2022)); va=(season==vs)&~isF
    # λ: 학습기간 시즌별 리그율에 선형추세를 맞춰 vs 로 외삽 (미래 미사용)
    ss=pd.DataFrame(dict(s=season[tr&~isF],y=y[tr&~isF])).groupby('s').y.mean()
    co=np.polyfit(ss.index.values, lgt(ss.values), 1); lam=np.polyval(co, vs)
    # 커리어(직전 시즌까지) 총계
    P=pd.DataFrame(dict(k=pid[tr],y=y[tr])).groupby('k').y.agg(['sum','size'])
    B=pd.DataFrame(dict(k=bid[tr],y=y[tr])).groupby('k').y.agg(['sum','size'])
    def arrs(idx):
        pn=pd.Series(pid[idx]).map(P['size']).fillna(0).values.astype(float)
        ps=pd.Series(pid[idx]).map(P['sum']).fillna(0).values.astype(float)
        bnn=pd.Series(bid[idx]).map(B['size']).fillna(0).values.astype(float)
        bs=pd.Series(bid[idx]).map(B['sum']).fillna(0).values.astype(float)
        return pn,ps,bnn,bs
    def theta(idx):
        pn,ps,bnn,bs=arrs(idx)
        th=eb(pn,ps,lam,T_PIT)                                   # 1) 커리어
        n_in=np.maximum(n_as[idx]-pn,0); s_in=np.maximum(cum[idx]-ps,0)
        th=eb(n_in,s_in,th,T_SSN)                                # 2) 당해 시즌
        p5=prev5[idx]; n5=np.where(np.isnan(p5),0.,150.)
        th=eb(n5,np.nan_to_num(p5)*n5,th,T_GAME)                 # 3) 최근 5경기
        ph=eb(bnn,bs,lam,T_BAT)                                  # 타자
        return th+(ph-lam)
    zt=theta(np.where(tr)[0]); zv=theta(np.where(va)[0])
    # 상황 효과: 학습기간 잔차에서 (볼카운트x아웃x주자) 셀별 로짓 오프셋, 축소 k=2000
    res=pd.DataFrame(dict(c=sit[tr],y=y[tr],p=sp.expit(zt)))
    g=res.groupby('c').apply(lambda t: (t.y.sum()-t.p.sum())/(t.p*(1-t.p)).sum()
                             if (t.p*(1-t.p)).sum()>0 else 0.)
    nsz=res.groupby('c').size(); g=g*nsz/(nsz+2000.)
    zv2=zv+pd.Series(sit[va]).map(g).fillna(0.).values
    yv=y[va].astype(np.float64); base=yv.mean()*(1-yv.mean())
    a=L.bss(sp.expit(zv),yv,base); c2=L.bss(sp.expit(zv2),yv,base)
    log(f'  폴드{vs}   투수+타자 {a:7.1f}   +상황 {c2:7.1f}   실제 {yv.mean():.4f} 예측 {sp.expit(zv2).mean():.4f} 편향 {sp.expit(zv2).mean()-yv.mean():+.4f}')
    np.save(OUT+f'p_{vs}.npy', sp.expit(zv2).astype(np.float32))
    return c2

log('\n계층 베이즈 단독 (124피처 GBDT 비교: 24:872.9 / 23:781.8)')
for vs in (2023,2024): run(vs)
