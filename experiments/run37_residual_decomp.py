"""
37차 — 현재 모델 잔차에 구조가 남아 있는가 (천장 판별)

방법: 그룹 g 의 잔차 평균 r̄_g 를 본다. 모델이 그 그룹에 대해 편향이 없으면
      r̄_g 는 0 근처에 표본 노이즈 σ²/n_g 만큼만 흩어진다.
      관측된 Σ w_g·r̄_g² 가 그 노이즈 기대값을 초과하면 = 미사용 신호.

      회수가능 BSS ≈ 100000 × (Σ w_g·r̄_g² − Σ w_g·σ²/n_g) / (r(1-r))

이건 '이 그룹 변수를 완벽히 쓰면 얼마나 더 얻나'의 상한이다.
음수면 그 축에는 남은 구조가 없다는 뜻.
"""
import numpy as np, pandas as pd, scipy.special as sp, json, os
D='/home/lee/lga/'
RAW=pd.read_csv(D+'data/train.csv',encoding='utf-8-sig')
y=RAW.control_success.values.astype(np.float64)
season=RAW.season.values; isF=(RAW.game_type.values=='F')
lg=lambda p: sp.logit(np.clip(p,1e-6,1-1e-6))
vs=2024; va=(season==vs)&~isF
yv=y[va]; base=yv.mean()*(1-yv.mean())

# 현재 제출 모델과 동등한 블렌드 재구성 (폴드2024)
tree=np.mean([lg(np.load(f'results32/p_{t}_s0_2024.npy')) for t in 'ABCD'],0)
sel=[n for n,_ in json.load(open('v4_nn_sel.json'))]
old=np.mean([lg(np.load(f'results6/{n}_2024.npy')) for n in sel],0)
mt =np.mean([lg(np.load(f'results18/{a}_s{s}_2024.npy')) for a in ('L3','L5') for s in range(4)],0)
z=0.70*tree+0.30*(0.6*old+0.4*mt) - 0.020      # 드리프트 보정까지 동일
p=sp.expit(z)
bss=100000*max(0.,1-np.mean((p-yv)**2)/base)
r=yv-p
sig2=float(np.mean(r**2))
print(f'기준 모델 (폴드2024)  BSS {bss:7.1f}   잔차평균 {r.mean():+.5f}   σ² {sig2:.5f}\n')

sub=RAW.loc[va].reset_index(drop=True)
# 구종 라벨 역산 (검증행에 대해)
ordr=np.lexsort((RAW.asof_pitcher_n.values, RAW.pitcher_id.values))
pid=RAW.pitcher_id.values[ordr]; mx=RAW.asof_pitcher_pitchmix_n.values[ordr].astype(np.float64)
last=np.append(pid[1:]!=pid[:-1],True)
PT={}
for c in ['fastball','breaking','offspeed']:
    cum=np.nan_to_num(mx*RAW[f'asof_pitcher_{c}_rate'].values[ordr])
    d=np.append(cum[1:]-cum[:-1],np.nan); d[last]=np.nan
    v=np.round(d); v[np.abs(d-v)>0.3]=np.nan
    o=np.full(len(RAW),np.nan); o[ordr]=v; PT[c]=o
ptype=np.where(PT['fastball']==1,'FB',np.where(PT['breaking']==1,'BR',
        np.where(PT['offspeed']==1,'OS','?')))[va]

G={
 '투수 ID'          : sub.pitcher_id.values,
 '타자 ID'          : sub.batter_id.values,
 '볼카운트'          : sub.balls_before.astype(str)+'-'+sub.strikes_before.astype(str),
 '구종(역산)'        : ptype,
 '이닝'             : sub.inning.clip(1,10).values,
 '월'               : sub.game_month.values,
 '주자상황'          : sub.base_state.values,
 '좌우조합'          : sub.pitcher_hand.astype(str)+'x'+sub.batter_hand.astype(str),
 '레버리지 10분위'    : pd.qcut(sub.li,10,labels=False,duplicates='drop'),
 '예측확률 20분위'    : pd.qcut(p,20,labels=False,duplicates='drop'),
 '투수 x 볼카운트'    : sub.pitcher_id.astype(str)+'|'+sub.balls_before.astype(str)+sub.strikes_before.astype(str),
 '투수 x 구종'       : sub.pitcher_id.astype(str)+'|'+pd.Series(ptype),
 '투수경험 20분위'    : pd.qcut(sub.asof_pitcher_n,20,labels=False,duplicates='drop'),
}
print(f"{'그룹 축':22s} {'그룹수':>7} {'회수가능 BSS':>13}")
print('-'*46)
rows=[]
for name,g in G.items():
    df=pd.DataFrame({'g':pd.Series(g).astype(str).values,'r':r})
    a=df.groupby('g')['r'].agg(['mean','size'])
    w=a['size'].values/len(r)
    obs=float(np.sum(w*a['mean'].values**2))
    noise=float(np.sum(w*sig2/a['size'].values))
    rec=100000*(obs-noise)/base
    rows.append((name,len(a),rec))
    print(f'{name:22s} {len(a):7,} {rec:13.1f}')
print('\n※ 회수가능 BSS = 그 축의 잔차 편향을 완벽히 제거했을 때의 상한')
print('※ 0 이하 = 남은 구조 없음 (이미 그 축은 다 썼다)')
