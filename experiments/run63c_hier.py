"""63차b — 계층 베이즈, 각 관측을 '그 시절 리그 수준'으로 디민 후 수축"""
import os, sys, warnings
import numpy as np, pandas as pd, scipy.special as sp
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/lee/lga'); import lib_lga as L
OUT='/home/lee/lga/results63/'; log,_=L.mklog(OUT,'log_c.txt')
b=L.load_base(); R=b['RAW']; y=b['y']; season=b['season']; isF=b['isF']
lgt=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))
pid=R.pitcher_id.values; bid=R.batter_id.values
n_as=R.asof_pitcher_n.values.astype(np.float64)
cum =n_as*np.nan_to_num(R.asof_pitcher_success_rate.values)
bn_as=R.asof_batter_n.values.astype(np.float64)
bcum =bn_as*np.nan_to_num(R.asof_batter_success_rate.values)
prev5=R.asof_pitcher_prev5_game_success_rate.values
cnt=(R.balls_before.values*10+R.strikes_before.values).astype(np.int64)
sit=cnt*100+R.outs_before.values.astype(np.int64)*10+np.minimum(R.num_runners_on.values,3).astype(np.int64)
V_PIT,V_SSN,V_GAME=np.load('/tmp/claude-1000/-home-lee-lga/a68aaebc-1ad3-4cca-a241-1603962ba966/scratchpad/vc.npy')
J=4.0
T_PIT,T_SSN,T_GAME,T_BAT=V_PIT*J*J, V_SSN*J*J, V_GAME*J*J*0.179**2, 0.0006*J*J

def eb_eff(n, s, tau, base_logit):
    """관측(n,s)을 base_logit 기준 '효과'로 바꾼 뒤 0 으로 수축한 값을 돌려준다"""
    n=np.maximum(n,0.); rate=np.divide(s,np.maximum(n,1.))
    e=lgt(np.clip(rate,1e-3,1-1e-3))-base_logit
    w=(n*0.25)/((n*0.25)+1.0/max(tau,1e-12))
    return np.where(n>0, w*e, 0.)

def run(vs):
    tr=(season<vs)&~(isF&(season<=2022)); va=(season==vs)&~isF
    m=tr&~isF
    ss=pd.DataFrame(dict(s=season[m],y=y[m])).groupby('s').y.mean()
    L_s=pd.Series(lgt(ss.values), index=ss.index)          # 시즌별 리그 로짓
    co=np.polyfit(ss.index.values, L_s.values, 1); lam=float(np.polyval(co, vs))
    log(f'  폴드{vs}  시즌리그로짓 {dict(L_s.round(4))}  -> 외삽 λ={lam:.4f} (p={sp.expit(lam):.4f})')
    # 각 학습행의 '그 시즌 리그 로짓'
    row_L=pd.Series(season[tr]).map(L_s).values
    # 커리어: 리그수준 제거한 잔차 성공수로 집계 (로짓 대신 확률잔차 누적 근사)
    dfp=pd.DataFrame(dict(k=pid[tr], y=y[tr], L=row_L))
    agg=dfp.groupby('k').agg(n=('y','size'), s=('y','sum'), Lm=('L','mean'))
    dfb=pd.DataFrame(dict(k=bid[tr], y=y[tr], L=row_L))
    aggb=dfb.groupby('k').agg(n=('y','size'), s=('y','sum'), Lm=('L','mean'))
    def theta(idx, base):
        pn=pd.Series(pid[idx]).map(agg['n']).fillna(0).values.astype(float)
        ps=pd.Series(pid[idx]).map(agg['s']).fillna(0).values.astype(float)
        pl=pd.Series(pid[idx]).map(agg['Lm']).fillna(np.mean(base)).values.astype(float)
        e=eb_eff(pn,ps,T_PIT,pl)                                  # 커리어 효과
        n_in=np.maximum(n_as[idx]-pn,0.); s_in=np.maximum(cum[idx]-ps,0.)
        e=e+eb_eff(n_in,s_in,T_SSN,base+e)                         # 당해시즌 증분
        p5=prev5[idx]; n5=np.where(np.isnan(p5),0.,150.)
        e=e+eb_eff(n5,np.nan_to_num(p5)*n5,T_GAME,base+e)          # 최근5경기 증분
        bn=pd.Series(bid[idx]).map(aggb['n']).fillna(0).values.astype(float)
        bs=pd.Series(bid[idx]).map(aggb['s']).fillna(0).values.astype(float)
        bl=pd.Series(bid[idx]).map(aggb['Lm']).fillna(np.mean(base)).values.astype(float)
        eb_=eb_eff(bn,bs,T_BAT,bl)
        return base+e+eb_
    tri=np.where(tr)[0]; vai=np.where(va)[0]
    zt=theta(tri, pd.Series(season[tri]).map(L_s).values)   # 학습행은 자기 시즌 수준
    zv=theta(vai, np.full(len(vai), lam))                   # 검증행은 외삽 λ
    res=pd.DataFrame(dict(c=sit[tr],y=y[tr],p=sp.expit(zt)))
    num=res.groupby('c').apply(lambda t:(t.y.sum()-t.p.sum())); den=res.groupby('c').apply(lambda t:(t.p*(1-t.p)).sum())
    nsz=res.groupby('c').size(); g=(num/den.replace(0,np.nan)).fillna(0.)*nsz/(nsz+2000.)
    zv2=zv+pd.Series(sit[va]).map(g).fillna(0.).values
    yv=y[va].astype(np.float64); base=yv.mean()*(1-yv.mean())
    for nm,z in (('투수+타자',zv),('+상황',zv2)):
        p=sp.expit(z); log(f'      {nm:8s} BSS {L.bss(p,yv,base):7.1f}   예측평균 {p.mean():.4f}  편향 {p.mean()-yv.mean():+.4f}  예측sd {p.std():.4f}')
    np.save(OUT+f'pc_{vs}.npy', sp.expit(zv2).astype(np.float32))

log('계층 베이즈 (디민 수정)  — 124피처 GBDT: 24:872.9 / 23:781.8')
for vs in (2023,2024): run(vs)
