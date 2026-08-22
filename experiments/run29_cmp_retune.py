"""
29차 — 성분모델(cmp_*) 재튜닝

■ 무엇을 왜 재는가
  y 모델(XGB)은 23/27차에서 재튜닝돼 폴드2024 821.7 -> 869.7 이 됐다.
  그런데 그 y 모델의 입력으로 들어가는 성분모델 4개
  (cmp_reverse / cmp_middle / cmp_ball / cmp_strike) 는
  아직 옛 하이퍼파라미터(d6/mcw1500/n600/lr0.008) 그대로다.

  성분 라벨은 y 보다 신호가 2배 강하다 (폴드2024 단독 스킬):
      ball 1778 / reverse 1431 / strike 1211 / middle 887   vs   y 809
  신호가 강한 타깃일수록 용량을 더 쓸 수 있으므로 재튜닝 여지가 크다.

  단계 A : 성분 예측 '자체'의 품질을 재튜닝으로 얼마나 올릴 수 있나
  단계 B : 그 개선이 y 모델 성능으로 이어지나 (이게 진짜 질문)

■ 누수 방지
  성분 OOF 는 시즌별 인과 규약을 지킨다: 시즌 s 행 = (season<s) 로 학습한 모델의 예측.
  이러면 폴드2024(학습<=2023) 와 폴드2023(학습<=2022) 양쪽에서 미래 정보가 섞이지 않는다.
  기존 oof_comp.parquet 도 같은 규약으로 만들어졌음을 확인했다
  (학습행 커버리지 폴드2024 61.3% / 폴드2023 50.3%, 검증행 100%).

■ 채택 기준
  단계 B 에서 **폴드2024 와 폴드2023 이 둘 다 개선될 때만** 채택한다.
  한쪽만 오르면 기각한다. 폴드2024 단독으로 고르면 LB 전이율이 0.06배까지
  떨어진 전례가 있다(HPO: CV +48 -> LB +3).
"""
import os, sys, json, time
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, '/home/lee/lga')
import lib_lga as L

OUT = '/home/lee/lga/results29/'
log, _ = L.mklog(OUT)

b = L.load_base()
y, season, isF = b['y'], b['season'], b['isF']
BASE = L.build_base114(b)                       # 성분모델 입력 = 기준 114 피처
comp, cls, _ = L.recover_labels(b['RAW'])
okl = comp.notna().all(1).values                # 성분 4종이 모두 역산된 행
log(f'기준 피처 {BASE.shape[1]} / 성분라벨 {okl.sum():,}/{len(okl):,}')

# 성분모델 후보. 성분모델은 학습 시 sample_weight 를 쓰지 않는다(배포 파이프라인과 동일).
CFGS = {
    'C0 옛설정 d6/mcw1500/n600':   dict(L.XP_OLD),
    'C1 중간 d8/mcw3000/n1200':    dict(L.XP_TUNED, max_depth=8, min_child_weight=3000,
                                        n_estimators=1200, learning_rate=0.008),
    'C2 튜닝 d10/mcw6000/n2000':   dict(L.XP_TUNED),
}


# ───────────────────── 단계 A: 성분 예측 자체의 품질 ─────────────────────
def comp_quality(prm, name):
    """각 성분을 자기 기저율 대비 BSS 로 평가. 폴드2024/2023 양쪽."""
    row = {'cfg': name}
    for vs in L.FOLDS:
        tr, va = L.split(vs, b)
        trm = tr & okl
        vam = va & okl
        for c in L.COMP:
            t = comp[c].values
            m = xgb.XGBClassifier(**prm, random_state=0).fit(BASE[trm], t[trm])
            p = m.predict_proba(BASE[vam])[:, 1]
            tv = t[vam].astype(np.float64)
            row[f'{c}_{vs}'] = L.bss(p, tv, tv.mean() * (1 - tv.mean()))
        log(f'  {name:28s} 폴드{vs} ' + ' '.join(
            f'{c[:4]}:{row[f"{c}_{vs}"]:7.1f}' for c in L.COMP))
    return row


log('\n===== 단계 A: 성분 예측 자체의 품질 =====')
A = []
for nm, prm in CFGS.items():
    A.append(comp_quality(prm, nm))
    pd.DataFrame(A).to_csv(OUT + 'res29_A.csv', index=False)

dfA = pd.DataFrame(A)
# 두 폴드 평균 성분 스킬 합으로 순위 (성분 자체 품질은 참고 지표일 뿐)
dfA['score'] = dfA[[f'{c}_{vs}' for c in L.COMP for vs in L.FOLDS]].mean(axis=1)
log('\n단계 A 요약 (성분 스킬 평균)')
for _, r in dfA.sort_values('score', ascending=False).iterrows():
    log(f'  {r.cfg:28s} {r.score:8.1f}')
winner = dfA.sort_values('score', ascending=False).cfg.iloc[0]
log(f'단계 A 최고: {winner}')


# ───────────────────── 단계 B: y 모델로 이어지나 ─────────────────────
def make_oof(prm, tag):
    """시즌별 인과 OOF 생성. 시즌 s = (season<s) 로 학습한 모델의 예측(로짓)."""
    f = OUT + f'oof_{tag}.parquet'
    if os.path.exists(f):
        log(f'  OOF 재사용 {f}')
        return pd.read_parquet(f)
    O = {c: np.full(len(y), np.nan, np.float32) for c in L.COMP}
    for s in range(2021, 2025):
        trm = (season < s) & ~(isF & (season <= 2022) & (s >= 2023)) & okl
        tgm = season == s
        for c in L.COMP:
            m = xgb.XGBClassifier(**prm, random_state=0).fit(BASE[trm], comp[c].values[trm])
            O[c][tgm] = L.logit(m.predict_proba(BASE[tgm])[:, 1])
        log(f'  OOF[{tag}] 시즌{s} 완료 (학습 {trm.sum():,})')
    of = L.cmp_frame(O)
    of.to_parquet(f)
    return of


log('\n===== 단계 B: y 모델 성능으로 이어지나 =====')
# 기준선 = 현행 성분 OOF (옛 설정으로 만들어져 있음)
X_old = L.build_v7()
r0 = L.bench2(X_old, L.XP_TUNED, '기준 현행 성분OOF', nseed=2, log=log, save_dir=OUT)
base = (r0['m24'], r0['m23'])

B = [r0]
# 단계 A 우승 설정 + (다르면) 튜닝 설정을 각각 y 모델까지 태워본다
cand = [winner] + [k for k in ('C2 튜닝 d10/mcw6000/n2000',) if k != winner]
for nm in cand:
    tag = nm.split()[0]
    of = make_oof(CFGS[nm], tag)
    Xn = L.build_v7(oof=of, b=b)
    B.append(L.bench2(Xn, L.XP_TUNED, f'성분OOF={tag}', nseed=2,
                      baseline=base, log=log, save_dir=OUT))
    pd.DataFrame(B).to_csv(OUT + 'res29_B.csv', index=False)

log('\n===== 결론 =====')
ok = [r for r in B[1:] if r.get('both')]
if ok:
    best = max(ok, key=lambda r: min(r['d24'], r['d23']))
    log(f'채택 권고: {best["name"]}  24 {best["d24"]:+.1f} / 23 {best["d23"]:+.1f} (두 폴드 모두 개선)')
else:
    log('채택할 설정 없음 — 두 폴드 모두 개선된 구성이 없다. 성분모델은 현행 유지.')
log(pd.DataFrame(B)[['name', 'm24', 'm23', 'd24', 'd23', 'both']].to_string(index=False))
