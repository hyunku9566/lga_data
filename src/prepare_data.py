"""
prepare_data — 흩어진 파생 캐시 생성을 하나로 묶은 스크립트

원본 로직을 그대로 옮겼다 (재작성하면 수치가 바뀌므로 손대지 않았다).
  step1 features  <- run_all.py stage1()
  step2 x98       <- mkfeat.py
  step3 aligned   <- run5.py 의 트랙맨 투구단위 정렬
  step4 tm5       <- run14.py 의 투수 지표 18종
  step5 oof_comp  <- build_v7.py 의 성분 OOF 생성부

사용법
  python prepare_data.py                 # 없는 캐시만 순서대로 생성
  python prepare_data.py --steps 3,4     # 일부만
  python prepare_data.py --force 5       # 이미 있어도 다시 만들기

주의
  * step3/4 는 trackman_history.csv 가 있어야 한다. 없으면 건너뛰고,
    tm5 없이도 114피처 중 트랙맨 8개만 빠진 채로 동작한다(권장하지 않음).
  * step5 는 GPU 를 쓴다 (XGBoost 4개 타깃 x 4시즌 = 16회 학습).
  * pitcher_map.csv 는 생성 스크립트가 없어 레포에 동봉한다. step1 의
    트랙맨 블록에서만 쓰이고, 그 산출 컬럼(tm_*)은 step2 에서 전부 버려지므로
    없어도 최종 파이프라인에는 영향이 없다.
"""
import os
import sys
import time
import json
import argparse
import warnings
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

warnings.filterwarnings('ignore')

T0 = time.time()


def log(*a):
    m = ' '.join(str(x) for x in a)
    print(f'[{(time.time() - T0) / 60:5.1f}m] {m}', flush=True)


def _train():
    return pd.read_csv(os.path.join(C.DATA_DIR, 'train.csv'), encoding='utf-8-sig')


# ════════════════════════════════════════════════════════════════
# step1. features.parquet   (run_all.py stage1)
# ════════════════════════════════════════════════════════════════
def step1_features(out):
    df = _train()
    raw_gt = df.game_type.values.copy()
    y = df.control_success.values.astype(np.float32)
    season = df.season.values
    log(f'train {df.shape}')

    # 1a. 경기/등판 복원 (train 은 시간순 정렬)
    gid = (df.inning.diff().fillna(0) < 0).cumsum().values
    df['_gid'] = gid
    aps = df.groupby([df._gid, df.pitcher_id, df.season], sort=False).size().reset_index(name='np_')
    ppa_ss = aps.groupby(['pitcher_id', 'season']).np_.median().reset_index()
    ppa_asof = {}
    for s in range(2019, 2026):
        prev = ppa_ss[ppa_ss.season < s]
        ppa_asof[s] = prev.groupby('pitcher_id').np_.median() if len(prev) else pd.Series(dtype=float)
    log(f'경기 {len(np.unique(gid)):,} 복원, 역할 as-of 완료')

    # 1b. 역산
    F = {}

    def sss(idcol, ncol, ratecol):
        t = df[[idcol, 'season', ncol, ratecol]].copy()
        t['succ'] = t[ncol] * t[ratecol].fillna(0)
        return t.loc[t.groupby([idcol, 'season'])[ncol].idxmin()].set_index([idcol, 'season'])[[ncol, 'succ']]

    specs = [('pitcher_id', 'asof_pitcher_n', 'asof_pitcher_success_rate', 'p_succ'),
             ('pitcher_id', 'asof_pitcher_n', 'asof_pitcher_reverse_rate', 'p_rev'),
             ('pitcher_id', 'asof_pitcher_n', 'asof_pitcher_middle_rate', 'p_mid'),
             ('pitcher_id', 'asof_pitcher_n', 'asof_pitcher_ball_rate', 'p_ball'),
             ('pitcher_id', 'asof_pitcher_n', 'asof_pitcher_strike_rate', 'p_stk'),
             ('batter_id', 'asof_batter_n', 'asof_batter_success_rate', 'b_succ'),
             ('batter_id', 'asof_batter_n', 'asof_batter_middle_rate', 'b_mid')]
    K = 150.0
    for idcol, ncol, ratecol, pref in specs:
        S = sss(idcol, ncol, ratecol)
        a = df[[idcol, 'season']].join(S, on=[idcol, 'season'])
        a_n = a[ncol].fillna(0).values
        a_s = a['succ'].fillna(0).values
        n = df[ncol].values
        r = df[ratecol].values
        dn = np.maximum(n - a_n, 0)
        ds = np.maximum(np.nan_to_num(n * r) - a_s, 0)
        pri = np.nanmean(r)
        F[pref + '_ssn'] = (ds + K * pri) / (dn + K)
        F[pref + '_ssn_vs_car'] = F[pref + '_ssn'] - np.nan_to_num(r, nan=pri)
        if pref in ('p_succ', 'b_succ'):
            F[pref + '_ssn_n'] = dn
    F['p_prev5_vs_car'] = (df.asof_pitcher_prev5_game_success_rate - df.asof_pitcher_success_rate).values
    F['p_prev1_vs_prev5'] = (df.asof_pitcher_prev1_game_success_rate
                             - df.asof_pitcher_prev5_game_success_rate).values
    log(f'역산 {len(F)}개')

    # 1c. 피로도 / 역할
    ppa_v = np.array([ppa_asof[s].get(p, np.nan) for s, p in zip(season, df.pitcher_id.values)])
    med = np.nanmedian(ppa_v)
    ppa_v = np.where(np.isnan(ppa_v), med, ppa_v)
    F['p_ppa'] = ppa_v
    F['p_est_apps'] = F['p_succ_ssn_n'] / np.clip(ppa_v, 5, None)
    F['p_inning_x_role'] = df.inning.values * np.log1p(ppa_v)
    F['p_ssn_per_month'] = F['p_succ_ssn_n'] / np.clip(df.game_month.values, 3, None)

    # 1d. 압박 성향 + 제구형 지수  (step2 에서 버려지지만 원본 스키마 유지를 위해 그대로 둔다)
    hi = (df.li > 1.5).values
    lo = (df.li < 0.5).values
    risp = ((df.runner_on_2b | df.runner_on_3b) == 1).values
    tmp = pd.DataFrame({'pid': df.pitcher_id.values, 's': season, 'y': y,
                        'hi': hi, 'lo': lo, 'risp': risp,
                        'ball': df.asof_pitcher_ball_rate.values,
                        'mid': df.asof_pitcher_middle_rate.values})
    SH = 400.0
    clutch = np.zeros(len(df), np.float32)
    ctrl = np.zeros(len(df), np.float32)
    for s in range(2019, 2026):
        prev = tmp[tmp.s < s]
        m = season == s
        if not m.any():
            continue
        if len(prev) > 0:
            g = prev.groupby('pid')
            a = g.apply(lambda x: (x.y[x.hi].sum() + SH * x.y.mean()) / (x.hi.sum() + SH)
                        - (x.y[x.lo].sum() + SH * x.y.mean()) / (x.lo.sum() + SH))
            bb = g.apply(lambda x: (x.y[x.risp].sum() + SH * x.y.mean()) / (x.risp.sum() + SH)
                         - (x.y[~x.risp].sum() + SH * x.y.mean()) / ((~x.risp).sum() + SH))
            pr = g[['ball', 'mid']].mean()
            ci = -(pr.ball.rank(pct=True) + pr.mid.rank(pct=True)) / 2
            pid_m = df.pitcher_id.values[m]
            clutch[m] = pd.Series(pid_m).map(a).fillna(0).values
            ctrl[m] = pd.Series(pid_m).map(ci).fillna(-0.5).values
            F.setdefault('p_clutch_risp', np.zeros(len(df), np.float32))
            F['p_clutch_risp'][m] = pd.Series(pid_m).map(bb).fillna(0).values
    F['p_clutch_li'] = clutch
    F['p_ctrl_idx'] = ctrl
    F['p_ctrl_x_li'] = ctrl * df.li.values
    F['p_clutch_x_li'] = clutch * df.li.values
    log('압박성향/제구형 완료')

    # 1e. 트랙맨 구위 (pitcher_map 필요. 실패해도 진행 — step2 에서 tm_* 는 버려진다)
    try:
        pm = pd.read_csv(C.asset('pitcher_map.csv'))
        pm = pm[pm.conf >= 0.90]
        tm = pd.read_csv(os.path.join(C.DATA_DIR, 'trackman_history.csv'), encoding='utf-8-sig',
                         usecols=['season', 'pitcher_trackman_id', 'pitch_type_group', 'rel_speed',
                                  'spin_rate', 'induced_vert_break', 'horz_break', 'extension',
                                  'rel_height', 'rel_side'])
        tm = tm.merge(pm[['pitcher_id', 'pitcher_trackman_id']], on='pitcher_trackman_id')
        log(f'트랙맨 조인 {len(tm):,}행, 투수 {tm.pitcher_id.nunique()}명')
        agg = tm.groupby(['pitcher_id', 'season']).agg(
            velo=('rel_speed', 'mean'), velo_sd=('rel_speed', 'std'),
            velo_max=('rel_speed', lambda x: x.quantile(0.95)),
            spin=('spin_rate', 'mean'), ivb=('induced_vert_break', 'mean'),
            hb=('horz_break', 'mean'), hb_abs=('horz_break', lambda x: x.abs().mean()),
            ext=('extension', 'mean'),
            rel_h_sd=('rel_height', 'std'), rel_s_sd=('rel_side', 'std'), ext_sd=('extension', 'std'),
            n=('rel_speed', 'size')).reset_index()
        fb = tm[tm.pitch_type_group == 'fastball'].groupby(['pitcher_id', 'season']).agg(
            fb_velo=('rel_speed', 'mean'), fb_spin=('spin_rate', 'mean'),
            fb_ivb=('induced_vert_break', 'mean')).reset_index()
        agg = agg.merge(fb, on=['pitcher_id', 'season'], how='left')
        arse = tm.groupby(['pitcher_id', 'season']).pitch_type_group.nunique().rename('arsenal').reset_index()
        agg = agg.merge(arse, on=['pitcher_id', 'season'], how='left')
        tcols = [c for c in agg.columns if c not in ('pitcher_id', 'season')]
        for c in tcols:
            F['tm_' + c] = np.full(len(df), np.nan, np.float32)
        for s in range(2019, 2026):
            m = season == s
            if not m.any():
                continue
            prev = agg[agg.season < s]
            if not len(prev):
                continue
            w = prev.groupby('pitcher_id').apply(
                lambda x: pd.Series({c: np.average(x[c], weights=x.n) if x[c].notna().any() else np.nan
                                     for c in tcols}))
            pid_m = pd.Series(df.pitcher_id.values[m])
            for c in tcols:
                F['tm_' + c][m] = pid_m.map(w[c]).values
        log(f'트랙맨 구위 {len(tcols)}개 (as-of s-1)')
    except Exception:
        log('!! 트랙맨 블록 건너뜀 (step2 에서 어차피 버려짐):\n' + traceback.format_exc())

    # 1f. 조립
    CATS = ['top_bottom', 'game_type', 'base_state', 'pitcher_hand', 'batter_hand',
            'pitcher_team_id', 'batter_team_id', 'pitcher_id', 'batter_id']
    X = df.drop(columns=['row_id', 'control_success', 'asof_pitcher_pitchmix_n', '_gid'], errors='ignore')
    for c in CATS:
        X[c] = X[c].astype('category').cat.codes
    X['hand_mix'] = X.pitcher_hand * 2 + X.batter_hand
    for k, v in F.items():
        X[k] = np.asarray(v, np.float32)
    X = X.astype(np.float32)
    X['__y'] = y
    X['__gt_F'] = (raw_gt == 'F').astype(np.int8)
    X['__season'] = season
    X.to_parquet(out)
    log(f'저장 {out}  shape={X.shape}')


# ════════════════════════════════════════════════════════════════
# step2. X98.parquet   (mkfeat.py)
# ════════════════════════════════════════════════════════════════
def step2_x98(out):
    RAW = _train()
    y = RAW.control_success.values.astype(np.float32)
    season = RAW.season.values
    src = C.resolve_cache('features')
    if src is None:
        raise FileNotFoundError('features.parquet 이 없다. step1 을 먼저 돌려라.')
    BASE = pd.read_parquet(src)
    CORE = [c for c in BASE.columns
            if not c.startswith(('__', 'tm_')) and 'clutch' not in c and 'ctrl' not in c]

    def asof_rate(mask, key, shrink=300.):
        o = np.full(len(RAW), np.nan, np.float32)
        ids = RAW[key].values
        for s in range(2020, 2026):
            prev = season < s
            tgt = season == s
            if not tgt.any() or not prev.any():
                continue
            sel = prev & mask
            g = pd.Series(y[sel]).groupby(ids[sel])
            tot = pd.Series(y[prev]).groupby(ids[prev]).mean()
            base = tot.reindex(g.size().index).fillna(y[prev].mean())
            o[tgt] = pd.Series(ids[tgt]).map((g.sum() + shrink * base) / (g.size() + shrink)).values
        return o

    b = RAW.balls_before.values
    st = RAW.strikes_before.values
    SITS = {'3ball': b == 3, '2strk': st == 2, 'ahead': st > b, 'behind': b > st,
            'risp': (RAW.runner_on_2b.values | RAW.runner_on_3b.values) == 1,
            'on1b': RAW.runner_on_1b.values == 1, 'vsL': RAW.batter_hand.values == 1,
            'vsR': RAW.batter_hand.values == 2, 'late': RAW.inning.values >= 7,
            'hiLI': RAW.li.values > 1.5, 'loLI': RAW.li.values < 0.5,
            'blowout': np.abs(RAW.score_diff_pitcher_team.values) >= 5}

    F = {}
    ov = asof_rate(np.ones(len(RAW), bool), 'pitcher_id')
    F['p_sit_overall'] = ov
    mt = np.full(len(RAW), np.nan, np.float32)
    for n, m in SITS.items():
        v = asof_rate(m, 'pitcher_id')
        F['p_sit_' + n] = v
        F['p_sit_' + n + '_d'] = v - ov
        mt = np.where(m & np.isnan(mt), v - ov, mt)
    F['p_sit_matched'] = mt
    log(f'S1 투수x상황 {len(F)}개')

    pk = RAW.pitcher_id.values.astype(np.int64) * 100000 + RAW.batter_id.values
    nn = np.zeros(len(RAW), np.float32)
    rr = np.full(len(RAW), np.nan, np.float32)
    for s in range(2020, 2026):
        prev = season < s
        tgt = season == s
        if not tgt.any() or not prev.any():
            continue
        g = pd.Series(y[prev]).groupby(pk[prev])
        k = pd.Series(pk[tgt])
        c = k.map(g.size()).fillna(0).values
        v = k.map(g.sum()).fillna(0).values
        nn[tgt] = c
        rr[tgt] = (v + 30 * y[prev].mean()) / (c + 30)
    F['pb_n'] = nn
    F['pb_rate'] = rr
    F['pb_logn'] = np.log1p(nn)

    X = pd.concat([BASE[CORE], pd.DataFrame(F, index=RAW.index).astype(np.float32)], axis=1)
    X['__y'] = y
    X['__season'] = season
    X['__F'] = (RAW.game_type.values == 'F').astype(np.int8)
    X.to_parquet(out)
    log(f'저장 {out}  shape={X.shape}')


# ════════════════════════════════════════════════════════════════
# step3. aligned.parquet   (run5.py 트랙맨 투구단위 정렬)
# ════════════════════════════════════════════════════════════════
def step3_aligned(out):
    RAW = _train()
    y = RAW.control_success.values.astype(np.float32)
    th = os.path.join(C.DATA_DIR, 'trackman_history.csv')
    if not os.path.exists(th):
        raise FileNotFoundError(f'{th} 없음. 트랙맨 단계는 건너뛸 수 있다.')
    # 주의: 원본 run5.py 의 usecols 에는 zone_speed 가 없지만, step4(run14) 가
    #      J.zone_speed 로 drag 를 계산한다. 실제 aligned.parquet 에도 들어있다.
    tm = pd.read_csv(th, encoding='utf-8-sig',
                     usecols=['season', 'trackman_game_id', 'pitch_no', 'inning', 'top_bottom',
                              'balls_before', 'strikes_before', 'outs_before', 'pitch_type_group',
                              'rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break',
                              'extension', 'rel_height', 'rel_side', 'zone_speed'])
    tm = tm.sort_values(['trackman_game_id', 'pitch_no'])
    tm['top_bottom'] = tm.top_bottom.str[0]
    tm = tm[tm.inning >= 1]
    tr = RAW[['season', 'inning', 'top_bottom', 'balls_before', 'strikes_before', 'outs_before']].copy()
    tr['gid'] = (RAW.inning.diff().fillna(0) < 0).cumsum().values
    tr['ridx'] = np.arange(len(RAW))

    def sig(df, g, n=30):
        s = (df.inning.astype(str) + df.top_bottom + df.balls_before.astype(str)
             + df.strikes_before.astype(str) + df.outs_before.astype(str))
        return s.groupby(df[g]).apply(lambda x: '|'.join(x.head(n)))

    A = pd.DataFrame({'sig': sig(tr, 'gid'), 'season': tr.groupby('gid').season.first()})
    B = pd.DataFrame({'sig': sig(tm, 'trackman_game_id'),
                      'season': tm.groupby('trackman_game_id').season.first()})
    M = (A.reset_index().merge(B.reset_index(), on=['sig', 'season'])
         .drop_duplicates('gid').drop_duplicates('trackman_game_id'))
    t2 = tr.merge(M[['gid', 'trackman_game_id']], on='gid')
    t2['k'] = t2.groupby('gid').cumcount()
    m2 = tm[tm.trackman_game_id.isin(M.trackman_game_id)].copy()
    m2['k'] = m2.groupby('trackman_game_id').cumcount()
    J = t2.merge(m2, on=['trackman_game_id', 'k'], suffixes=('', '_tm'))
    J = J[(J.inning == J.inning_tm) & (J.balls_before == J.balls_before_tm)
          & (J.strikes_before == J.strikes_before_tm) & (J.outs_before == J.outs_before_tm)]
    J['pitcher_id'] = RAW.pitcher_id.values[J.ridx.values]
    J['y'] = y[J.ridx.values]
    J['pitch_of_app'] = J.groupby(['trackman_game_id', 'pitcher_id']).cumcount()
    J.to_parquet(out)
    log(f'저장 {out}  {len(J):,}구, 투수 {J.pitcher_id.nunique()}명')


# ════════════════════════════════════════════════════════════════
# step4. tm5.parquet   (run14.py 투수 지표 18종)
# ════════════════════════════════════════════════════════════════
def step4_tm5(out):
    RAW = _train()
    season = RAW.season.values
    src = C.resolve_cache('aligned')
    if src is None:
        raise FileNotFoundError('aligned.parquet 이 없다. step3 을 먼저 돌려라.')
    J = pd.read_parquet(src)
    TYPES = ['fastball', 'breaking', 'offspeed']
    J = J[J.pitch_type_group.isin(TYPES)].copy()
    J['vaa_c'] = (-11.6236 + .0921 * J.rel_speed - 1.0763 * J.rel_height
                  - .0244 * J.extension + .1777 * J.induced_vert_break)
    J['haa_c'] = (.0921 * J.rel_speed - 1.0763 * J.rel_side.abs()
                  - .0244 * J.extension + .1777 * J.horz_break.abs())
    J['drag'] = J.zone_speed / J.rel_speed.replace(0, np.nan)

    KEYS = ['k_relh_sd', 'k_rels_sd', 'k_ext_sd', 'k_comb', 'k_rel3d_vol',
            'v_vaa', 'v_vaa_sd', 'v_haa', 'v_haa_sd', 'd_drag', 'd_drag_sd', 't_tunnel',
            'a_anom_mean', 'a_anom_p95', 'm1_mov_resid', 'm2_press_shift', 'm2_press_velo', 'm3_decept']

    def build_pitcher_table(hist):
        g = hist.groupby('pitcher_id')

        def kirby(x):
            d = {k: np.nan for k in KEYS}
            sds = []
            for t, gg in x.groupby('pitch_type_group'):
                if len(gg) < 30:
                    continue
                sds.append((len(gg), gg.rel_height.std(), gg.rel_side.std(), gg.extension.std()))
            if sds:
                w = np.array([s[0] for s in sds], float)
                w /= w.sum()
                d['k_relh_sd'] = float(np.dot(w, [s[1] for s in sds]))
                d['k_rels_sd'] = float(np.dot(w, [s[2] for s in sds]))
                d['k_ext_sd'] = float(np.dot(w, [s[3] for s in sds]))
                d['k_comb'] = d['k_relh_sd'] + d['k_rels_sd'] + 0.5 * d['k_ext_sd']
            R = x[['rel_height', 'rel_side', 'extension']].dropna()
            if len(R) > 50:
                Cv = np.cov(R.values.T) + np.eye(3) * 1e-6
                d['k_rel3d_vol'] = float(np.linalg.det(Cv)) ** (1 / 6)
            d['v_vaa'] = float(x.vaa_c.mean())
            d['v_vaa_sd'] = float(x.vaa_c.std())
            d['v_haa'] = float(x.haa_c.mean())
            d['v_haa_sd'] = float(x.haa_c.std())
            d['d_drag'] = float(x.drag.mean())
            d['d_drag_sd'] = float(x.drag.std())
            cen = x.groupby('pitch_type_group')[['rel_height', 'rel_side', 'extension']].mean().values
            if len(cen) > 1:
                dd = [np.linalg.norm(cen[i] - cen[j]) for i in range(len(cen)) for j in range(i + 1, len(cen))]
                d['t_tunnel'] = float(np.mean(dd))
            R2 = x[['rel_height', 'rel_side']].dropna()
            if len(R2) > 50:
                mu = R2.mean().values
                Cv = np.cov(R2.values.T) + np.eye(2) * 1e-6
                Ci = np.linalg.inv(Cv)
                dv = R2.values - mu
                md = np.sqrt(np.einsum('ij,jk,ik->i', dv, Ci, dv))
                d['a_anom_mean'] = float(md.mean())
                d['a_anom_p95'] = float(np.percentile(md, 95))
            res = []
            for t, gg in x.groupby('pitch_type_group'):
                gg = gg[['rel_height', 'rel_side', 'extension', 'rel_speed',
                         'induced_vert_break', 'horz_break']].dropna()
                if len(gg) < 80:
                    continue
                A = np.c_[np.ones(len(gg)), gg[['rel_height', 'rel_side', 'extension', 'rel_speed']].values]
                for tgt in ['induced_vert_break', 'horz_break']:
                    bb, *_ = np.linalg.lstsq(A, gg[tgt].values, rcond=None)
                    res.append((len(gg), float(np.std(gg[tgt].values - A @ bb))))
            if res:
                w = np.array([r[0] for r in res], float)
                w /= w.sum()
                d['m1_mov_resid'] = float(np.dot(w, [r[1] for r in res]))
            hi = x[x.balls_before >= 3]
            lo = x[x.balls_before == 0]
            if len(hi) > 40 and len(lo) > 40:
                a = hi[['rel_height', 'rel_side', 'extension']].mean().values
                bq = lo[['rel_height', 'rel_side', 'extension']].mean().values
                d['m2_press_shift'] = float(np.linalg.norm(a - bq))
                d['m2_press_velo'] = float(hi.rel_speed.mean() - lo.rel_speed.mean())
            mc = x.groupby('pitch_type_group')[['induced_vert_break', 'horz_break']].mean().values
            if len(mc) > 1 and 't_tunnel' in d and d['t_tunnel'] and d['t_tunnel'] > 1e-6:
                mm = [np.linalg.norm(mc[i] - mc[j]) for i in range(len(mc)) for j in range(i + 1, len(mc))]
                d['m3_decept'] = float(np.mean(mm)) / d['t_tunnel']
            d['_n'] = len(x)
            return pd.Series({k: d.get(k, np.nan) for k in KEYS + ['_n']})

        return g.apply(kirby)

    COLS = None
    FEATS = {}
    for s in range(2020, 2026):
        tgt = season == s
        if not tgt.any():
            continue
        hist = J[J.season < s]
        if not len(hist):
            continue
        tb = build_pitcher_table(hist)
        if COLS is None:
            COLS = [c for c in tb.columns if c != '_n']
            for c in COLS:
                FEATS[c] = np.full(len(RAW), np.nan, np.float32)
        pid = pd.Series(RAW.pitcher_id.values[tgt])
        for c in COLS:
            if c in tb.columns:
                FEATS[c][tgt] = pid.map(tb[c]).values
        log(f'  시즌{s} 지표 생성 (이전 {len(hist):,}구, 투수 {tb.shape[0]}명)')
    TM = pd.DataFrame(FEATS, index=RAW.index).astype(np.float32)
    TM.to_parquet(out)
    log(f'저장 {out}  {TM.shape[1]}개, 평균결측 {TM.isna().mean().mean():.3f}')


# ════════════════════════════════════════════════════════════════
# step5. oof_comp.parquet   (build_v7.py 성분 OOF)
# ════════════════════════════════════════════════════════════════
def step5_oof(out):
    import xgboost as xgb
    import scipy.special as sp
    RAW = _train()
    x98p = C.resolve_cache('x98')
    tm5p = C.resolve_cache('tm5')
    if x98p is None:
        raise FileNotFoundError('X98.parquet 이 없다. step2 를 먼저 돌려라.')
    X98 = pd.read_parquet(x98p)
    season = X98.__season.values
    isF = X98.__F.values.astype(bool)
    CORE = [c for c in X98.columns if not c.startswith('__')]
    parts = [X98[CORE]]
    # multi_k
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
    parts.append(pd.DataFrame(F, index=RAW.index).astype(np.float32))
    if tm5p:
        tmsel = json.load(open(C.asset('v6_tmsel.json')))
        parts.append(pd.read_parquet(tm5p)[tmsel])
    else:
        log('!! tm5 없음 -> 트랙맨 8개 없이 성분 OOF 를 만든다 (권장하지 않음)')
    BASE = pd.concat(parts, axis=1)
    log(f'성분모델 입력 {BASE.shape[1]}피처')

    # 라벨 역산
    ordr = np.lexsort((RAW.asof_pitcher_n.values, RAW.pitcher_id.values))
    pid_s = RAW.pitcher_id.values[ordr]
    n_s = RAW.asof_pitcher_n.values[ordr].astype(np.float64)
    last = np.append(pid_s[1:] != pid_s[:-1], True)
    COMP = ['reverse', 'middle', 'ball', 'strike']
    LAB = {}
    for c in COMP:
        cum = np.nan_to_num(n_s * RAW[f'asof_pitcher_{c}_rate'].values[ordr])
        d = np.append(cum[1:] - cum[:-1], np.nan)
        d[last] = np.nan
        v = np.round(d)
        v[np.abs(d - v) > 0.3] = np.nan
        o = np.full(len(RAW), np.nan, np.float32)
        o[ordr] = v
        LAB[c] = o
    L = pd.DataFrame(LAB)
    okl = L.notna().all(1).values
    log(f'성분라벨 {okl.sum():,}/{len(L):,}')

    XP = dict(n_estimators=600, learning_rate=0.008, max_depth=6, min_child_weight=1500,
              subsample=0.7, colsample_bytree=0.5, reg_lambda=50., reg_alpha=1.,
              tree_method='hist', device=C.DEV, eval_metric='logloss', verbosity=0)
    lgt = lambda p: sp.logit(np.clip(p, 1e-6, 1 - 1e-6))  # noqa: E731
    OOF = {c: np.full(len(RAW), np.nan, np.float32) for c in COMP}
    for s in range(2021, 2025):
        tr = (season < s) & ~(isF & (season <= 2022) & (s >= 2023)) & okl
        tg = season == s
        for c in COMP:
            m = xgb.XGBClassifier(**XP, random_state=0).fit(BASE.loc[tr], L[c].values[tr])
            OOF[c][tg] = lgt(m.predict_proba(BASE.loc[tg])[:, 1])
        log(f'  성분 OOF 시즌{s} (학습 {tr.sum():,})')
    o = pd.DataFrame({f'cmp_{c}': OOF[c] for c in COMP})
    o['cmp_bad'] = o.cmp_reverse + o.cmp_middle
    o['cmp_zone'] = o.cmp_strike - o.cmp_ball
    o = o.astype(np.float32)
    o.index = RAW.index
    o.to_parquet(out)
    log(f'저장 {out}  결측률 {o.isna().mean().mean():.3f}')


STEPS = {
    1: ('features', step1_features, 'CPU  ~6-12분'),
    2: ('x98',      step2_x98,      'CPU  ~4-8분'),
    3: ('aligned',  step3_aligned,  'CPU  ~5-10분  (trackman_history.csv 필요)'),
    4: ('tm5',      step4_tm5,      'CPU  ~10-20분 (무겁다)'),
    5: ('oof_comp', step5_oof,      'GPU  ~10-15분'),
}


def main():
    ap = argparse.ArgumentParser(description='파생 캐시 생성')
    ap.add_argument('--steps', default='1,2,3,4,5', help='쉼표구분 (기본 전체)')
    ap.add_argument('--force', default='', help='이미 있어도 다시 만들 단계 (쉼표구분)')
    a = ap.parse_args()
    want = [int(x) for x in a.steps.split(',') if x.strip()]
    force = {int(x) for x in a.force.split(',') if x.strip()}

    C.ensure_dirs()
    print(C.describe())
    print()
    for i in want:
        key, fn, cost = STEPS[i]
        out = C.cache_path(key)
        have = C.resolve_cache(key)
        if have and i not in force:
            log(f'step{i} {key:9s} 이미 있음 -> 건너뜀  ({have})')
            continue
        log(f'───── step{i} {key} 생성 시작  [{cost}]')
        t = time.time()
        try:
            fn(out)
            log(f'───── step{i} {key} 완료  {(time.time()-t)/60:.1f}분')
        except FileNotFoundError as e:
            log(f'───── step{i} {key} 건너뜀: {e}')
        except Exception:
            log(f'───── step{i} {key} 실패:\n' + traceback.format_exc())
    print()
    print(C.describe())


if __name__ == '__main__':
    main()
