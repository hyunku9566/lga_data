"""
35차 — CatBoost 아키텍처 다양성 및 시드 앙상블 검증

배경:
  - 현재 CatBoost는 단일 설정(depth=6, seed=0) 1개만 사용 중 (트리축 20% 비중 차지).
  - CatBoost에도 depth 4, 6, 8 및 시드 앙상블을 적용하여 듀얼 폴드(2024+2023) 개선을 실측한다.
"""
import os, json, time, warnings
import numpy as np, pandas as pd, scipy.special as sp
from catboost import CatBoostClassifier, Pool
import lib_lga
warnings.filterwarnings('ignore')

D = '/home/lee/lga/'
OUT = D + 'results35/'
log, OUT = lib_lga.mklog(OUT, 'log.txt')

b = lib_lga.load_base()
RAW = b['RAW']
XK = lib_lga.build_v7(b=b)

CATF = ['pitcher_id','batter_id','pitcher_team_id','batter_team_id','pitcher_hand',
        'batter_hand','base_state','game_type','top_bottom']
Z = XK.copy()
for c in CATF:
    Z[c] = Z[c].fillna(-1).astype(np.int32).astype(str)

DEV = os.environ.get('LGA_DEV', 'cuda:1')
dev_idx = '1' if '1' in DEV else '0'

CFGS = {
    'd6': dict(iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=50.,
               loss_function='Logloss', verbose=0, task_type='GPU', devices=dev_idx),
    'd4': dict(iterations=2500, learning_rate=0.03, depth=4, l2_leaf_reg=30.,
               loss_function='Logloss', verbose=0, task_type='GPU', devices=dev_idx),
    'd8': dict(iterations=1500, learning_rate=0.02, depth=8, l2_leaf_reg=70.,
               loss_function='Logloss', verbose=0, task_type='GPU', devices=dev_idx),
}

preds = {}
for tag, prm in CFGS.items():
    preds[tag] = {}
    for vs in lib_lga.FOLDS:
        ctx = lib_lga.get_ctx(vs)
        ps = []
        for s in range(2):
            t0 = time.time()
            m = CatBoostClassifier(**prm, random_seed=s)
            ptr = Pool(Z.iloc[ctx['tr']], b['y'][ctx['tr']], weight=ctx['w'], cat_features=CATF)
            pva = Pool(Z.iloc[ctx['va']], cat_features=CATF)
            m.fit(ptr)
            p = m.predict_proba(pva)[:, 1]
            ps.append(p)
            log(f'  fit CB {tag} s{s} 폴드{vs}  {time.time()-t0:.1f}초  BSS: {lib_lga.bss(p, ctx["yv"], ctx["base"]):.1f}')
        preds[tag][vs] = np.mean(ps, 0)
        np.save(f'{OUT}cb_{tag}_{vs}.npy', preds[tag][vs].astype(np.float32))

# 기준선 (d6 단독)
base24 = lib_lga.bss(preds['d6'][2024], lib_lga.get_ctx(2024)['yv'], lib_lga.get_ctx(2024)['base'])
base23 = lib_lga.bss(preds['d6'][2023], lib_lga.get_ctx(2023)['yv'], lib_lga.get_ctx(2023)['base'])
log(f'\n기준선 (CB d6 x2시드): 2024={base24:.1f}, 2023={base23:.1f}')

# d4 + d6 + d8 결합
p_ens24 = (preds['d4'][2024] + preds['d6'][2024] + preds['d8'][2024]) / 3
p_ens23 = (preds['d4'][2023] + preds['d6'][2023] + preds['d8'][2023]) / 3
s24 = lib_lga.bss(p_ens24, lib_lga.get_ctx(2024)['yv'], lib_lga.get_ctx(2024)['base'])
s23 = lib_lga.bss(p_ens23, lib_lga.get_ctx(2023)['yv'], lib_lga.get_ctx(2023)['base'])

d24 = s24 - base24
d23 = s23 - base23
both = (d24 > 0 and d23 > 0)
mark = 'O 채택가능' if both else '  '
log(f'CB 다양성 (d4+d6+d8): 2024={s24:.1f} ({d24:+5.1f}) | 2023={s23:.1f} ({d23:+5.1f}) {mark}')

res_df = pd.DataFrame([
    dict(name='CB d6 기준', m24=base24, m23=base23, d24=0., d23=0., both=False),
    dict(name='CB d4+d6+d8 다양성', m24=s24, m23=s23, d24=d24, d23=d23, both=both),
])
res_df.to_csv(OUT + 'res35.csv', index=False)
log('\n완료')
