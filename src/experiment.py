"""
experiment — 아이디어 하나를 넣으면 '기준선 대비 유의미한가'가 한눈에 나오는 러너

왜 이렇게까지 하나
  폴드2024 CV +52.9 를 믿고 제출했다가 LB -8.67 을 맞았다. 그 +52.9 중 상당 부분이
  27개 조합에서 최고를 고른 선택 편향이었고, 개별 차이 일부는 시드 노이즈였다.
  그래서 '노이즈 막대 없는 델타는 보고하지 않는다' 를 파이프라인에 강제한다.

판정 규칙 (se = 기준선 시드 표준편차로 만든 차이의 표준오차)
  |delta| < 2*se                      -> 노이즈    (유의하지 않음)
  delta > 2*se  이고 두 폴드 모두 양수  -> 채택 후보
  delta > 2*se  인데 한쪽만 양수        -> 보류      (폴드 불일치)
  delta < -2*se                       -> 기각

기준선 (고정, 모든 아이디어의 비교 대상)
  피처  v7 120개
  모델  XGB d6/mcw1500/n600/lr0.008/sub0.7/col0.5/L2=50/L1=1   (현재 제출본과 동일)
  가중  최근성 반감기 2.0 시즌
  시드  5개 -> 폴드별 평균과 표준편차를 캐시

사용 예
    from experiment import run_experiment, DEFAULT_XGB
    run_experiment(name="xgb_depth_sweep", kind="hparam",
                   grid={"max_depth":[6,8,10], "min_child_weight":[1500,6000]},
                   base_params=DEFAULT_XGB, seeds=2, runner="홍길동")
"""
import os
import sys
import json
import time
import itertools
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C          # noqa: E402
import lib_lga as L         # noqa: E402

warnings.filterwarnings('ignore')

__all__ = ['run_experiment', 'DEFAULT_XGB', 'get_baseline', 'ledger_path',
           'read_all_ledgers', 'format_report', 'TRANSFER', 'VERDICTS']

# ────────────────────────── 상수 ──────────────────────────

# 현재 제출본(v16wB)의 XGB 설정. 기준선이자 스윕의 출발점.
# 2026-08-22 갱신: 종전에는 v10a 시절(d6/mcw1500/n600/lr.008)이 박혀 있었다.
# 낡은 기준선에 대고 튜닝하면 이미 채택된 이득을 다시 세게 되므로 반드시 현행을 써야 한다.
DEFAULT_XGB = dict(n_estimators=2000, learning_rate=0.005, max_depth=10, min_child_weight=6000,
                   subsample=0.7, colsample_bytree=0.5, reg_lambda=50., reg_alpha=1.,
                   tree_method='hist', eval_metric='logloss', verbosity=0)
# 참고용 옛 설정 (비교 기준이 필요할 때만)
LEGACY_XGB_V10A = dict(n_estimators=600, learning_rate=0.008, max_depth=6, min_child_weight=1500,
                       subsample=0.7, colsample_bytree=0.5, reg_lambda=50., reg_alpha=1.,
                       tree_method='hist', eval_metric='logloss', verbosity=0)

BASELINE_SEEDS = 5
HL = 2.0

# 실측 전이율. CV 이득에 곱해 LB 기대범위를 만든다.
#   v8   CV +4.3  -> LB +0.9   (0.21배)
#   v9   CV +5.3  -> LB +2.8   (0.53배)
#   HPO  CV +48   -> LB +3.0   (0.06배)
#   pbc  CV +6.1  -> LB +3.3   (0.54배)   <- 피처 추가
#   v17  CV +6.9  -> LB **-5.84**         <- 모델설정 변경, 부호가 반대
# 2026-08-22: 모델 설정 변경은 v7 이후 **7전 7패**다 (v10/v11a/v11c/v12/v13/v17).
# 성공한 것은 피처 추가(pbc_*)뿐이다. 하이퍼파라미터 스윕의 기대값은 0 에 가깝다고 보라.
TRANSFER = {'낙관': 0.5, '비관': 0.0}

BLEND_WARNING = (
    '※ 이 LB 기대범위는 모델/피처 변경에만 적용된다. 블렌드 가중치 변경은 전이율이 '
    '음수였다 (CV +14.2 -> LB -11.66). 블렌드는 반드시 LB 실측으로만 정해라.'
)

VERDICTS = ('채택후보', '보류', '노이즈', '기각')

LEDGER_COLS = ['timestamp', 'runner', 'name', 'kind', 'params_json',
               'm24', 'm23', 'm24_sd', 'm23_sd', 'delta24', 'delta23',
               'se24', 'se23', 'both', 'verdict', 'lb_lo', 'lb_hi',
               'seeds', 'nfeat', 'sec', 'notes']


# ────────────────────────── 기준선 ──────────────────────────

def _baseline_cache_path():
    return os.path.join(C.LEDGER_DIR, 'baseline.json')


def _seed_scores(Xa, prm, vs, seeds, target=None):
    """폴드 vs 에서 시드별 BSS 리스트를 돌려준다 (표준편차를 구하기 위함).

    lib_lga.fit_predict 는 시드 평균만 주므로, 시드별 값이 필요한 여기서만
    같은 규약(get_ctx / bss)으로 직접 돈다. 폴드 정의·가중은 lib_lga 와 동일하다.
    """
    import xgboost as xgb
    b = L.load_base()
    tgt = b['y'] if target is None else target
    ctx = L.get_ctx(vs, HL)
    out = []
    for s in range(seeds):
        m = xgb.XGBClassifier(**prm, random_state=s, device=C.DEV)
        m.fit(Xa[ctx['tr']], tgt[ctx['tr']], sample_weight=ctx['w'])
        p = m.predict_proba(Xa[ctx['va']])[:, 1]
        out.append(L.bss(p, ctx['yv'], ctx['base']))
    return out


def get_baseline(force=False, verbose=True):
    """고정 기준선을 계산(또는 캐시 로드)한다.

    반환 dict(m24, m24_sd, m23, m23_sd, seeds, params, nfeat)
    """
    p = _baseline_cache_path()
    if os.path.exists(p) and not force:
        d = json.load(open(p))
        if verbose:
            print(f'기준선 캐시 로드: 2024 {d["m24"]:.1f}±{d["m24_sd"]:.1f}  '
                  f'2023 {d["m23"]:.1f}±{d["m23_sd"]:.1f}  (시드 {d["seeds"]})')
        return d
    if verbose:
        print(f'기준선 계산 중 (시드 {BASELINE_SEEDS} x 폴드 2) — 처음 한 번만, 10~20분')
    Xa = L.build_v16()          # 124 = v7 120 + pbc 4 (현재 제출본과 동일)
    prm = dict(DEFAULT_XGB)
    t0 = time.time()
    s24 = _seed_scores(Xa, prm, 2024, BASELINE_SEEDS)
    s23 = _seed_scores(Xa, prm, 2023, BASELINE_SEEDS)
    d = dict(m24=float(np.mean(s24)), m24_sd=float(np.std(s24, ddof=1)),
             m23=float(np.mean(s23)), m23_sd=float(np.std(s23, ddof=1)),
             seeds=BASELINE_SEEDS, nfeat=int(Xa.shape[1]),
             params=prm, sec=time.time() - t0,
             scores24=[float(x) for x in s24], scores23=[float(x) for x in s23])
    os.makedirs(C.LEDGER_DIR, exist_ok=True)
    json.dump(d, open(p, 'w'), ensure_ascii=False, indent=1)
    if verbose:
        print(f'기준선 저장: 2024 {d["m24"]:.1f}±{d["m24_sd"]:.1f}  '
              f'2023 {d["m23"]:.1f}±{d["m23_sd"]:.1f}  ({d["sec"]/60:.1f}분)')
    return d


# ────────────────────────── 유의성 판정 ──────────────────────────

def _se(sd, n_exp, n_base=BASELINE_SEEDS):
    """두 평균 차이의 표준오차.
    se = sd * sqrt(1/n_base + 1/n_exp).  n_base==n_exp==n 이면 sqrt(2)*sd/sqrt(n).
    """
    return float(sd * np.sqrt(1.0 / max(n_base, 1) + 1.0 / max(n_exp, 1)))


def _mark(delta, se):
    if abs(delta) < 2 * se:
        return '노이즈'
    return '개선' if delta > 0 else '기각'


def judge(m24, m23, base, seeds):
    """판정 + LB 기대범위."""
    se24 = _se(base['m24_sd'], seeds)
    se23 = _se(base['m23_sd'], seeds)
    d24 = m24 - base['m24']
    d23 = m23 - base['m23']
    k24, k23 = _mark(d24, se24), _mark(d23, se23)

    # 두 폴드 표식이 같을 때만 확정 판정을 내린다. 엇갈리면 전부 '보류'.
    # 폴드2024/2023 은 HPO 순위 Spearman -0.806 으로 서로 역전되는 관계라,
    # 한쪽 신호만으로 확정하면 안 된다.
    if k24 == k23 == '개선':
        verdict = '채택후보'
    elif k24 == k23 == '기각':
        verdict = '기각'
    elif k24 == k23 == '노이즈':
        verdict = '노이즈'
    else:
        verdict = '보류'

    # LB 기대범위는 주 폴드(2024) 효과에 실측 전이율을 곱해 만든다.
    lb_lo = d24 * TRANSFER['비관']
    lb_hi = d24 * TRANSFER['낙관']
    if lb_lo > lb_hi:
        lb_lo, lb_hi = lb_hi, lb_lo
    return dict(d24=d24, d23=d23, se24=se24, se23=se23,
                k24=k24, k23=k23, verdict=verdict,
                both=bool(d24 > 0 and d23 > 0),
                lb_lo=float(lb_lo), lb_hi=float(lb_hi))


def format_report(name, kind, runner, seeds, m24, m23, base, j):
    """실험 하나가 끝날 때마다 찍는 고정 블록."""
    W = 60
    reason = {
        '채택후보': '두 폴드 모두 유의하게 개선',
        '기각': '두 폴드 모두 유의하게 악화',
        '노이즈': '두 폴드 모두 노이즈 범위 — 효과 없음',
        '보류': f'폴드 불일치 (2024 {j["k24"]}, 2023 {j["k23"]})',
    }[j['verdict']]
    lines = [
        '═' * W,
        f'아이디어  : {name}',
        f'분류      : {kind:<12s}  실행자: {runner:<10s}  시드: {seeds}',
        '─' * W,
        f'{"":12s}{"폴드2024":<16s}{"폴드2023":<16s}',
        f'{"기준선":<12s}{base["m24"]:>7.1f} ±{base["m24_sd"]:<6.1f}{base["m23"]:>7.1f} ±{base["m23_sd"]:<6.1f}',
        f'{"결과":<12s}{m24:>7.1f}{"":<8s}{m23:>7.1f}',
        f'{"효과":<12s}{j["d24"]:>+7.1f}{"":<8s}{j["d23"]:>+7.1f}',
        f'{"유의성":<12s}±{j["se24"]*2:<6.1f}(2se){"":<3s}±{j["se23"]*2:<6.1f}(2se)',
        f'{"":<12s}{j["k24"]:<16s}{j["k23"]:<16s}',
        '─' * W,
        f'판정      : {j["verdict"]} — {reason}',
        f'LB 기대   : 관측 전이율 {TRANSFER["비관"]}~{TRANSFER["낙관"]}배 적용 '
        f'→ {j["lb_lo"]:+.1f} ~ {j["lb_hi"]:+.1f}',
    ]
    if j['verdict'] in ('보류', '노이즈'):
        lines.append('            ※ 폴드 불일치/노이즈 시 LB 예측 신뢰도 매우 낮음')
    lines.append('═' * W)
    return '\n'.join(lines)


# ────────────────────────── 원장 ──────────────────────────

def ledger_path(runner):
    """팀원별 파일로 쓴다. 동시 append 충돌을 피하기 위함이고, 집계 때 합친다."""
    safe = ''.join(ch for ch in str(runner) if ch.isalnum() or ch in '-_')
    os.makedirs(C.LEDGER_DIR, exist_ok=True)
    return os.path.join(C.LEDGER_DIR, f'ledger_{safe or "unknown"}.csv')


def _append(row, runner):
    p = ledger_path(runner)
    df = pd.DataFrame([{c: row.get(c) for c in LEDGER_COLS}])
    df.to_csv(p, mode='a', header=not os.path.exists(p), index=False, encoding='utf-8-sig')


def _done_keys(runner):
    """이미 원장에 있는 (name, params_json) — 재개 시 건너뛰기용."""
    p = ledger_path(runner)
    if not os.path.exists(p):
        return set()
    try:
        d = pd.read_csv(p, encoding='utf-8-sig')
        return set(zip(d.name.astype(str), d.params_json.astype(str)))
    except Exception:
        return set()


def read_all_ledgers(ledger_dir=None):
    """모든 팀원 원장을 하나로 합친다."""
    ledger_dir = ledger_dir or C.LEDGER_DIR
    fs = sorted(f for f in os.listdir(ledger_dir)
                if f.startswith('ledger_') and f.endswith('.csv')) if os.path.isdir(ledger_dir) else []
    if not fs:
        return pd.DataFrame(columns=LEDGER_COLS)
    ds = []
    for f in fs:
        try:
            ds.append(pd.read_csv(os.path.join(ledger_dir, f), encoding='utf-8-sig'))
        except Exception:
            pass
    return pd.concat(ds, ignore_index=True) if ds else pd.DataFrame(columns=LEDGER_COLS)


# ────────────────────────── 러너 ──────────────────────────

def _grid_rows(grid):
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]


# 부스터별 현재 설정 (v10a 제출본 기준). 스윕의 출발점이자 자기 기준선.
# 현재 제출본(v16wB) 의 세 축 설정.
# 주의: 49~56차에서 찾은 LGB(extra_trees/cs0.4) + CB(d4/bagtemp2/l2_500) 신설정은
#       CV 상 크게 좋았으나 **LB 에서 -5.84 로 실패했다(v17)**. 그래서 현행은 v16 설정이다.
CURRENT = {
    'xgb': DEFAULT_XGB,
    'lgb': dict(n_estimators=1200, learning_rate=0.01, num_leaves=15, min_child_samples=6000,
                subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=50.),
    'cb':  dict(iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=50.),
}

CAT_FEATURES = ['pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id', 'pitcher_hand',
                'batter_hand', 'base_state', 'game_type', 'top_bottom']


def _eval_model(model, Xa, prm, vs, seeds, target=None):
    """폴드 vs 에서 시드 평균 BSS. 폴드 정의·가중은 lib_lga 와 동일하다.

    model='xgb' 는 lib_lga.bench2 경로를 쓰고, lgb/cb 는 같은 fold ctx 위에서
    같은 규약으로 돈다 (bench2 가 XGB 전용이라 어쩔 수 없다).
    """
    b = L.load_base()
    tgt = b['y'] if target is None else target
    ctx = L.get_ctx(vs, HL)
    ps = []
    if model == 'xgb':
        import xgboost as xgb
        for s in range(seeds):
            m = xgb.XGBClassifier(**prm, random_state=s, device=C.DEV)
            m.fit(Xa[ctx['tr']], tgt[ctx['tr']], sample_weight=ctx['w'])
            ps.append(m.predict_proba(Xa[ctx['va']])[:, 1])
    elif model == 'lgb':
        import lightgbm as lgb
        for s in range(seeds):
            m = lgb.LGBMClassifier(**prm, random_state=s, verbose=-1, n_jobs=-1)
            m.fit(Xa[ctx['tr']], tgt[ctx['tr']], sample_weight=ctx['w'])
            ps.append(m.predict_proba(Xa[ctx['va']])[:, 1])
    elif model == 'cb':
        from catboost import CatBoostClassifier, Pool
        Z = Xa.copy()
        cats = [c for c in CAT_FEATURES if c in Z.columns]
        for c in cats:
            Z[c] = Z[c].fillna(-1).astype('int32').astype(str)
        ptr = Pool(Z[ctx['tr']], tgt[ctx['tr']], weight=ctx['w'], cat_features=cats)
        pva = Pool(Z[ctx['va']], cat_features=cats)
        for s in range(seeds):
            m = CatBoostClassifier(**prm, loss_function='Logloss', random_seed=s,
                                   verbose=0, task_type='GPU',
                                   devices=C.DEV.split(':')[-1]).fit(ptr)
            ps.append(m.predict_proba(pva)[:, 1])
    else:
        raise ValueError(f'알 수 없는 model: {model}')
    return L.bss(np.mean(ps, 0), ctx['yv'], ctx['base'])


def run_experiment(name, kind='hparam', grid=None, base_params=None, features=None,
                   seeds=2, runner='unknown', feature_fn=None, notes='',
                   baseline=None, verbose=True, model='xgb', ref_params=None):
    """아이디어 하나를 기준선과 비교한다.

    name        아이디어 이름 (원장 키)
    kind        'hparam' | 'feature' | 'blend'
    grid        하이퍼파라미터 격자. {'max_depth':[6,8,10], ...}
    base_params 격자에 덮어씌울 기본 설정 (기본: 해당 model 의 CURRENT)
    features    평가에 쓸 피처 DataFrame. None 이면 v16 124개 (현재 제출본과 동일)
    seeds       시드 수 (많을수록 노이즈 막대가 줄어든다)
    runner      팀원 이름 — 원장 파일이 팀원별로 갈린다
    feature_fn  kind='feature' 일 때 fn(RAW, X98, base) -> DataFrame.
                반환한 열이 기존 피처에 붙는다.
    model       'xgb' | 'lgb' | 'cb'
    ref_params  이 설정을 먼저 재서 '자기 기준선' 으로 삼는다.
                LGB/CatBoost 는 XGB 기준선(≈868)과 직접 비교하면 델타가 -38 처럼
                나와 무의미하므로, 해당 부스터의 현재 설정을 기준으로 삼아야 한다.
                None 이고 model!='xgb' 이면 CURRENT[model] 을 자동으로 쓴다.

    반환: 결과 행들의 DataFrame
    """
    if kind == 'blend':
        print('!! kind="blend" 는 CV 로 판정하지 않는다. ' + BLEND_WARNING)
        print('   블렌드 후보는 원장에 기록만 하고 LB 로 검증해라.')

    base = baseline or get_baseline(verbose=verbose)
    b = L.load_base()
    Xbase = L.build_v16(b=b) if features is None else features

    if kind == 'feature':
        if feature_fn is None:
            raise ValueError('kind="feature" 에는 feature_fn 이 필요하다')
        add = feature_fn(b['RAW'], b['X98'], b)
        add = add.copy()
        add.index = Xbase.index
        Xa = pd.concat([Xbase, add], axis=1)
        if verbose:
            print(f'피처 {Xbase.shape[1]} + {add.shape[1]} = {Xa.shape[1]}')
    else:
        Xa = Xbase

    # 자기 기준선: 다른 부스터는 XGB 기준선과 비교하면 의미가 없다.
    if ref_params is None and model != 'xgb':
        ref_params = CURRENT[model]
    if ref_params is not None:
        rp = dict(ref_params)
        if model == 'xgb':
            rp['device'] = C.DEV
        if verbose:
            print(f'자기 기준선 계산 중 ({model} 현재 설정, 시드 {seeds}) ...')
        r24 = _eval_model(model, Xa, rp, 2024, seeds)
        r23 = _eval_model(model, Xa, rp, 2023, seeds)
        base = dict(base, m24=r24, m23=r23)
        if verbose:
            print(f'자기 기준선  2024 {r24:.1f}  2023 {r23:.1f}   '
                  f'(노이즈 척도 sd 는 전역 기준선 값 ±{base["m24_sd"]:.1f}/±{base["m23_sd"]:.1f} 을 쓴다)')

    done = _done_keys(runner)
    rows = []
    combos = _grid_rows(grid)
    if verbose:
        print(f'조합 {len(combos)}개 (이미 완료된 것은 건너뛴다)')

    for i, g in enumerate(combos):
        prm = dict(base_params or CURRENT.get(model, DEFAULT_XGB))
        prm.update(g)
        if model == 'xgb':
            prm.pop('device', None)
            prm['device'] = C.DEV
        pj = json.dumps(g, sort_keys=True, ensure_ascii=False)
        if (str(name), pj) in done:
            if verbose:
                print(f'[{i+1}/{len(combos)}] 건너뜀 (원장에 있음): {pj}')
            continue

        t0 = time.time()
        label = f'{name} {pj}' if g else name
        if model == 'xgb':
            # 두 폴드 판정은 반드시 lib_lga.bench2 를 통과시킨다.
            r = L.bench2(Xa, prm=prm, name=label, nseed=seeds, hl=HL,
                         baseline=(base['m24'], base['m23']),
                         log=(print if verbose else (lambda *a: None)))
        else:
            r = dict(m24=_eval_model(model, Xa, prm, 2024, seeds),
                     m23=_eval_model(model, Xa, prm, 2023, seeds))
        j = judge(r['m24'], r['m23'], base, seeds)
        row = dict(timestamp=datetime.now().isoformat(timespec='seconds'),
                   runner=runner, name=name, kind=kind, params_json=pj,
                   m24=round(r['m24'], 2), m23=round(r['m23'], 2),
                   m24_sd=round(base['m24_sd'], 2), m23_sd=round(base['m23_sd'], 2),
                   delta24=round(j['d24'], 2), delta23=round(j['d23'], 2),
                   se24=round(j['se24'], 2), se23=round(j['se23'], 2),
                   both=j['both'], verdict=j['verdict'],
                   lb_lo=round(j['lb_lo'], 2), lb_hi=round(j['lb_hi'], 2),
                   seeds=seeds, nfeat=int(Xa.shape[1]),
                   sec=round(time.time() - t0, 1), notes=notes)
        _append(row, runner)          # 조합마다 즉시 기록 (런타임이 끊겨도 남는다)
        rows.append(row)
        if verbose:
            print(format_report(label, kind, runner, seeds, r['m24'], r['m23'], base, j))
            print(BLEND_WARNING if kind == 'blend' else '')

    return pd.DataFrame(rows)


if __name__ == '__main__':
    print(C.describe())
    print()
    print('기준선:', json.dumps({k: v for k, v in get_baseline().items()
                               if k not in ('scores24', 'scores23', 'params')},
                              ensure_ascii=False))
