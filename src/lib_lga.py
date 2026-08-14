"""
lib_lga — LG Aimers 제구 예측 공통 모듈

run20~run28 이 매번 복사·붙여넣기 하던 로딩/피처/폴드/평가 코드를 한곳에 모았다.

핵심 규약 (오늘 얻은 교훈이 반영돼 있다)
  * 검증은 폴드2024(주) + 폴드2023(보조) 두 곳에서 한다.
    폴드2022 는 2024 ABS 도입 전후로 regime 이 달라 제외한다.
  * 채택 기준은 '두 폴드 모두 개선'. 한쪽만 오르면 채택하지 않는다.
    - 폴드2024 단독으로 고르고 같은 폴드로 검증했더니 LB 전이율이 0.06배였다
      (HPO: CV +48 -> LB +3).
    - 다양성을 줄이는 방향은 CV 가 올라도 LB 가 크게 나빠졌다
      (블렌드 집중: CV +17 -> LB -11.66).

폴드 정의 (기존 run* 과 동일)
    split(vs) = (season<vs) & ~(isF & season<=2022 & vs>=2023),  (season==vs) & ~isF
학습 가중  = 0.5 ** ((vs-1 - season) / hl),  기본 hl=2.0 (반감기 2시즌)
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd, scipy.special as sp
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

D = C.ROOT
OOF_COMP = C.resolve_cache('oof_comp') or C.cache_path('oof_comp')

COMP = ['reverse', 'middle', 'ball', 'strike']          # 결과 성분 4종
PITCH = ['fastball', 'breaking', 'offspeed']            # 구종 3종
FOLDS = (2024, 2023)                                    # 주 / 보조

# GPU 지정. 두 스크립트를 각 GPU 에 나눠 돌리려면 LGA_DEV=cuda:1 로 실행한다.
DEV = C.DEV

# 23차+27차 폴드2024 최적. 성분모델·y모델 공통 출발점으로 쓴다.
XP_TUNED = dict(n_estimators=2000, learning_rate=0.005, max_depth=10, min_child_weight=6000,
                subsample=0.7, colsample_bytree=0.5, reg_lambda=50., reg_alpha=1.,
                tree_method='hist', device=DEV, eval_metric='logloss', verbosity=0)
# v9 이전에 쓰던 옛 설정 (비교 기준선)
XP_OLD = dict(n_estimators=600, learning_rate=0.008, max_depth=6, min_child_weight=1500,
              subsample=0.7, colsample_bytree=0.5, reg_lambda=50., reg_alpha=1.,
              tree_method='hist', device=DEV, eval_metric='logloss', verbosity=0)

_C = {}   # 로딩 캐시


# ────────────────────────────── 로딩 ──────────────────────────────
def load_base():
    """RAW / X98 / TM 과 파생 배열을 한 번만 읽어 캐시한다."""
    if 'base' in _C:
        return _C['base']
    RAW = pd.read_csv(os.path.join(C.DATA_DIR, 'train.csv'), encoding='utf-8-sig')
    _x98 = C.resolve_cache('x98')
    _tm5 = C.resolve_cache('tm5')
    if _x98 is None:
        raise FileNotFoundError('X98.parquet 이 없다. prepare_data.py --steps 1,2 를 먼저 돌려라.')
    if _tm5 is None:
        raise FileNotFoundError('tm5.parquet 이 없다. prepare_data.py --steps 3,4 를 먼저 돌려라.')
    X98 = pd.read_parquet(_x98)
    TM = pd.read_parquet(_tm5)
    b = dict(
        RAW=RAW, X98=X98, TM=TM,
        y=X98.__y.values.astype(np.float32),
        season=X98.__season.values,
        isF=X98.__F.values.astype(bool),
        CORE=[c for c in X98.columns if not c.startswith('__')],
        TMSEL=json.load(open(C.asset('v6_tmsel.json'))),
    )
    _C['base'] = b
    return b


def multi_k(RAW=None):
    """투수/타자 시즌 진행분을 여러 축소강도(k)로 만든 8개 피처.
    앵커(시즌 첫 행)를 빼서 '당해 누적'을 복원한 뒤 리그평균으로 축소한다."""
    if RAW is None:
        RAW = load_base()['RAW']
    F = {}
    for idc, nc, rc, pf in [('pitcher_id', 'asof_pitcher_n', 'asof_pitcher_success_rate', 'p_succ'),
                            ('batter_id', 'asof_batter_n', 'asof_batter_success_rate', 'b_succ')]:
        t = RAW[[idc, 'season', nc, rc]].copy()
        t['succ'] = t[nc] * t[rc].fillna(0)
        S = t.loc[t.groupby([idc, 'season'])[nc].idxmin()].set_index([idc, 'season'])[[nc, 'succ']]
        a = RAW[[idc, 'season']].join(S, on=[idc, 'season'])
        dn = np.maximum(RAW[nc].values - a[nc].fillna(0).values, 0)
        ds = np.maximum(np.nan_to_num(RAW[nc].values * RAW[rc].values) - a['succ'].fillna(0).values, 0)
        lgv = np.nanmean(RAW[rc])
        for k in [25, 75, 400, 1000]:
            F[f'{pf}_k{k}'] = (ds + k * lgv) / (dn + k)
    return pd.DataFrame(F, index=RAW.index).astype(np.float32)


def comp_oof(path=OOF_COMP):
    """저장된 성분 OOF 6열 (cmp_reverse/middle/ball/strike + cmp_bad/cmp_zone).
    시즌 s 행은 <s 로 학습한 모델의 예측이라 폴드2024/2023 양쪽에서 누수가 없다."""
    return pd.read_parquet(path)


def build_base114(b=None):
    """v6 기준 114 피처 (X98 CORE + multi_k 8 + 트랙맨 선택 8)."""
    b = b or load_base()
    return pd.concat([b['X98'][b['CORE']], multi_k(b['RAW']), b['TM'][b['TMSEL']]], axis=1)


def build_v7(oof=None, b=None):
    """v7/v9/v10 이 쓰는 120 피처 = 기준 114 + 성분 OOF 6.
    oof 를 넘기면 그것으로 교체한다 (성분모델 재튜닝 실험용)."""
    b = b or load_base()
    of = comp_oof() if oof is None else oof
    of = of.copy()
    of.index = b['RAW'].index
    return pd.concat([build_base114(b), of], axis=1)


# ──────────────────────── 숨은 라벨 역산 ────────────────────────
def _diff_labels(RAW, count_col, rate_cols, prefix=''):
    """누적비율 x 누적개수를 차분해 투구별 개별 라벨을 복원한다.
    투수별로 count_col 이 증분 1의 완전 수열이라 정확히 역산된다."""
    ordr = np.lexsort((RAW.asof_pitcher_n.values, RAW.pitcher_id.values))
    pid = RAW.pitcher_id.values[ordr]
    n = RAW[count_col].values[ordr].astype(np.float64)
    last = np.append(pid[1:] != pid[:-1], True)
    out = {}
    for c in rate_cols:
        cum = np.nan_to_num(n * RAW[f'asof_pitcher_{c}_rate'].values[ordr])
        d = np.append(cum[1:] - cum[:-1], np.nan)
        d[last] = np.nan                       # 투수별 마지막 1구는 역산 불가
        v = np.round(d)
        v[np.abs(d - v) > 0.3] = np.nan        # 반올림 잡음 방어
        o = np.full(len(RAW), np.nan, np.float32)
        o[ordr] = v
        out[prefix + c] = o
    return pd.DataFrame(out, index=RAW.index)


def recover_labels(RAW=None):
    """성분 4종 + 구종 3종 라벨 역산.

    반환 (comp, cls, valid)
      comp  : DataFrame[reverse, middle, ball, strike]  (0/1, 일부 NaN)
      cls   : int64 배열, 0=fastball 1=breaking 2=offspeed, 역산불가 -1
      valid : cls>=0 마스크
    역산한 success 라벨은 control_success 와 일치율 1.000000 으로 검증됐다.
    타깃 구조: y = ¬reverse ∧ ¬middle ∧ Z
    """
    if RAW is None:
        RAW = load_base()['RAW']
    comp = _diff_labels(RAW, 'asof_pitcher_n', COMP)
    pt = _diff_labels(RAW, 'asof_pitcher_pitchmix_n', PITCH)
    ok = pt.notna().all(1).values
    cls = np.full(len(RAW), -1, np.int64)
    for i, c in enumerate(PITCH):
        cls[ok & (pt[c].values == 1)] = i
    return comp, cls, cls >= 0


def cmp_frame(d):
    """성분 로짓 4개 + 파생 2개. 학습/추론 동일 규약(build_v7.py 와 같음)."""
    o = pd.DataFrame({f'cmp_{c}': d[c] for c in COMP})
    o['cmp_bad'] = o.cmp_reverse + o.cmp_middle
    o['cmp_zone'] = o.cmp_strike - o.cmp_ball
    return o.astype(np.float32)


# ────────────────────────── 폴드 / 평가 ──────────────────────────
def split(vs, b=None):
    """기존 run* 과 완전히 동일한 분할."""
    b = b or load_base()
    season, isF = b['season'], b['isF']
    tr = (season < vs) & ~(isF & (season <= 2022) & (vs >= 2023))
    va = (season == vs) & ~isF
    return tr, va


def fold_ctx(vs, hl=2.0, b=None):
    """폴드 하나의 학습/검증 마스크 + 최근성 가중 + 평가 기저분산."""
    b = b or load_base()
    y, season = b['y'], b['season']
    tr, va = split(vs, b)
    w = (0.5 ** ((vs - 1 - season[tr]) / hl)).astype(np.float32) if hl else None
    yv = y[va].astype(np.float64)
    return dict(vs=vs, tr=tr, va=va, w=w, yv=yv, base=yv.mean() * (1 - yv.mean()))


def bss(p, yv, base):
    """대회 지표: 100000 * (1 - Brier / (r(1-r)))"""
    return 100000 * max(0., 1 - np.mean((np.asarray(p, np.float64) - yv) ** 2) / base)


def logit(p):
    return sp.logit(np.clip(p, 1e-6, 1 - 1e-6))


_CTX = {}


def get_ctx(vs, hl=2.0):
    if (vs, hl) not in _CTX:
        _CTX[(vs, hl)] = fold_ctx(vs, hl)
    return _CTX[(vs, hl)]


def fit_predict(Xa, target, prm, ctx, nseed=2, sample_weight=True):
    """ctx 폴드에서 nseed 개 시드 평균 확률을 반환."""
    import xgboost as xgb
    w = ctx['w'] if sample_weight else None
    ps = []
    for s in range(nseed):
        m = xgb.XGBClassifier(**prm, random_state=s)
        m.fit(Xa[ctx['tr']], target[ctx['tr']], sample_weight=w)
        ps.append(m.predict_proba(Xa[ctx['va']])[:, 1])
    return np.mean(ps, 0)


def bench2(Xa, prm=None, name='', nseed=2, hl=2.0, baseline=None, log=print,
           target=None, save_dir=None):
    """폴드2024(주) + 폴드2023(보조) 양쪽에서 재고 결과를 반환한다.

    baseline 으로 (m24, m23) 을 넘기면 '두 폴드 모두 개선' 여부를 함께 판정한다.
    채택 기준은 both=True 일 때만이다. 한쪽만 오르는 건 채택하지 않는다.

    반환: dict(name, nfeat, m24, m23, d24, d23, both, sec)
    """
    prm = prm or XP_TUNED
    b = load_base()
    tgt = b['y'] if target is None else target
    t0 = time.time()
    out = {}
    for vs in FOLDS:
        ctx = get_ctx(vs, hl)
        p = fit_predict(Xa, tgt, prm, ctx, nseed=nseed)
        out[vs] = bss(p, ctx['yv'], ctx['base'])
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f'{name.replace(" ","_").replace("/","-")}_{vs}.npy'),
                    p.astype(np.float32))
    r = dict(name=name, nfeat=int(Xa.shape[1]), m24=out[2024], m23=out[2023],
             sec=time.time() - t0)
    if baseline is not None:
        r['d24'] = out[2024] - baseline[0]
        r['d23'] = out[2023] - baseline[1]
        r['both'] = bool(r['d24'] > 0 and r['d23'] > 0)
        mark = 'O 채택가능' if r['both'] else '  '
        log(f'  {name:34s}({Xa.shape[1]:3d}) 24:{out[2024]:7.1f} ({r["d24"]:+6.1f}) '
            f'23:{out[2023]:7.1f} ({r["d23"]:+6.1f}) {mark}')
    else:
        r['d24'] = r['d23'] = 0.0
        r['both'] = False
        log(f'  {name:34s}({Xa.shape[1]:3d}) 24:{out[2024]:7.1f}  23:{out[2023]:7.1f}  [기준]')
    return r


def mklog(out_dir, fname='log.txt'):
    """결과 디렉토리를 만들고 (log 함수, 경로) 를 돌려준다.
    셸 리다이렉트가 디렉토리보다 먼저 실행돼 실패한 사고가 있어 스크립트가 직접 만든다."""
    os.makedirs(out_dir, exist_ok=True)
    fh = open(os.path.join(out_dir, fname), 'a', buffering=1)
    t0 = time.time()

    def log(*a):
        m = ' '.join(str(x) for x in a)
        print(f'[{(time.time()-t0)/60:5.1f}m] {m}', flush=True)
        fh.write(f'[{(time.time()-t0)/60:5.1f}m] {m}\n')
    return log, out_dir
