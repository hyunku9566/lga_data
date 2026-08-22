"""
62차 — 상태공간 투수능력 추정기 (기존 GBDT 구조와 무관한 별도 모형)

데이터에서 직접 추정한 분산성분 (1군, 확률단위)
    σ²_투수간   0.002106  (sd 4.6%p)   33.6%
    σ²_시즌|투수 0.000283  (sd 1.7%p)    4.5%
    σ²_경기|시즌 0.003878  (sd 6.2%p)   61.9%   <- 자기상관 ρ=0.179
    관측잡음     p(1-p) ≈ 0.25 / 투구

구조
    θ_p,g = 투수 p 의 경기 g 시점 잠재능력 (로짓)
      θ = μ_리그(시즌) + α_p + β_p,시즌 + η_p,g,   η ~ AR(1), ρ=0.179
    사후평균을 정밀도 가중으로 닫힌 형태로 계산한다.
    입력은 전부 행 하나에서 나온다:
      커리어 총계(train 조회) / 2025 시즌내(역산) / prev1·3·5 경기율

    기존 파이프라인의 k25/75/400/1000 은 이 사후평균을 여러 대역폭으로
    손 근사한 것이다. 여기서는 최적 가중을 분산성분에서 직접 푼다.

평가: 폴드2023/2024 에서 **단독** BSS. 블렌드 다양성 축 후보로서의 값어치를 본다.
"""
import os, sys, warnings
import numpy as np, pandas as pd, scipy.special as sp
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga')
import lib_lga as L

OUT='/home/lee/lga/results62/'; log,_=L.mklog(OUT)
b=L.load_base(); R=b['RAW']; y=b['y']; season=b['season']; isF=b['isF']
V_PIT, V_SSN, V_GAME = np.load('/tmp/claude-1000/-home-lee-lga/a68aaebc-1ad3-4cca-a241-1603962ba966/scratchpad/vc.npy')
RHO=0.179
log(f'분산성분 투수간 {V_PIT:.6f} / 시즌 {V_SSN:.6f} / 경기 {V_GAME:.6f}  ρ={RHO}')

pid=R.pitcher_id.values
n_asof=R.asof_pitcher_n.values.astype(np.float64)
r_asof=np.nan_to_num(R.asof_pitcher_success_rate.values)
cum=n_asof*r_asof
prev1=R.asof_pitcher_prev1_game_success_rate.values
prev5=R.asof_pitcher_prev5_game_success_rate.values

def fit_eval(vs):
    tr,va=L.split(vs,b)
    mu=float(y[tr].mean())                     # 학습기간 리그 수준
    # 투수별: 검증시즌 '이전' 총계 (= 시즌 첫 행의 asof 값)
    prior=pd.DataFrame(dict(pid=pid[tr],y=y[tr])).groupby('pid').y.agg(['sum','size'])
    P_n=pd.Series(pid).map(prior['size']).fillna(0).values
    P_s=pd.Series(pid).map(prior['sum']).fillna(0).values
    # 당해 시즌 진행분 = 현재 누적 - 이전 총계
    n_in=np.maximum(n_asof-P_n,0); s_in=np.maximum(cum-P_s,0)

    def post(idx):
        """정밀도 가중 사후평균 (확률단위 근사; 관측정밀도 = n/(p(1-p)))"""
        v=mu*(1-mu)
        # 1층: 커리어 (투수간 분산으로 리그에 수축)
        tau1=V_PIT
        w1=(P_n[idx]/v)/((P_n[idx]/v)+1/tau1)
        car=w1*np.divide(P_s[idx],np.maximum(P_n[idx],1))+(1-w1)*mu
        # 2층: 당해 시즌 (시즌 혁신분산만큼 커리어에서 이탈 허용)
        tau2=V_SSN
        w2=(n_in[idx]/v)/((n_in[idx]/v)+1/tau2)
        ssn=w2*np.divide(s_in[idx],np.maximum(n_in[idx],1))+(1-w2)*car
        # 3층: 최근 경기 (경기 혁신분산, AR(1) 감쇠 ρ 적용)
        p5=prev5[idx]; p1=prev1[idx]
        n5=np.where(np.isnan(p5),0,150.); n1=np.where(np.isnan(p1),0,30.)
        obs=np.nan_to_num(p5)*n5+np.nan_to_num(p1)*n1
        nn=n5+n1
        tau3=V_GAME*RHO**2                      # 다음 경기로 전이되는 몫만
        w3=(nn/v)/((nn/v)+1/max(tau3,1e-9))
        cur=w3*np.divide(obs,np.maximum(nn,1))+(1-w3)*ssn
        return np.clip(cur,1e-4,1-1e-4)

    ptr=post(np.where(tr)[0]); pva=post(np.where(va)[0])
    yv=y[va].astype(np.float64); base=yv.mean()*(1-yv.mean())
    # 상황 보정: 학습기간에서 잔차를 볼카운트 12셀로만 (상황 정보량이 31 이라 이거면 충분)
    cnt=R.balls_before.values*10+R.strikes_before.values
    ctr=cnt[tr]; cva=cnt[va]
    adj=pd.DataFrame(dict(c=ctr,r=y[tr]-ptr)).groupby('c').r.mean()
    pva2=np.clip(pva+pd.Series(cva).map(adj).fillna(0).values,1e-4,1-1e-4)
    # 리그수준 시프트는 LB 로 정하는 영역이므로 여기선 학습기간 평균만 맞춘다
    return L.bss(pva,yv,base), L.bss(pva2,yv,base), yv.mean(), pva.mean()

log('\n폴드   상태공간 단독   +볼카운트보정   검증실제   예측평균')
for vs in (2023,2024):
    a,c,ym,pm=fit_eval(vs)
    log(f'  {vs}      {a:8.1f}      {c:8.1f}      {ym:.4f}   {pm:.4f}')
log('\n비교: 같은 폴드 XGB 124피처 단독 = 24:872.9 / 23:781.8')
