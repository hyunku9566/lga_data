# script.py — 평가 서버 진입점 (v4: XGB 15시드 + NN 8종 블렌딩, GPU 추론 + CPU 폴백)
import os, json, time
import numpy as np, pandas as pd, scipy.special as sp, xgboost as xgb
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.join(HERE, "model"); DDIR = os.path.join(HERE, "data"); ODIR = os.path.join(HERE, "output")
ID, TGT = "row_id", "control_success"

def sit_masks(df):
    b = df.balls_before.values; s = df.strikes_before.values
    return {"3ball": b==3, "2strk": s==2, "ahead": s>b, "behind": b>s,
            "risp": (df.runner_on_2b.values | df.runner_on_3b.values)==1,
            "on1b": df.runner_on_1b.values==1, "vsL": df.batter_hand.values==1,
            "vsR": df.batter_hand.values==2, "late": df.inning.values>=7,
            "hiLI": df.li.values>1.5, "loLI": df.li.values<0.5,
            "blowout": np.abs(df.score_diff_pitcher_team.values)>=5}

def featurize(df, art):
    C = art["consts"]; cols = {}
    for c in df.columns:
        if c in (ID, TGT, "asof_pitcher_pitchmix_n"): continue
        cols[c] = df[c].values
    for c, m in C["catmap"].items():
        s = df[c]
        key = (s.astype(str) if s.dtype == object
               else pd.to_numeric(s, errors="coerce").astype("Int64").astype(str))
        cols[c] = key.map(m).fillna(-1).astype(np.int32).values
    cols["hand_mix"] = cols["pitcher_hand"]*2 + cols["batter_hand"]
    for idcol, ncol, ratecol, pref in C["specs"]:
        A = art["anchor_"+idcol]
        j = df[[idcol]].join(A.set_index(idcol), on=idcol)
        a_n = j["a_n"].fillna(0).values; a_s = j["succ_"+pref].fillna(0).values
        n = df[ncol].values; r = df[ratecol].values
        dn = np.maximum(n-a_n, 0); ds = np.maximum(np.nan_to_num(n*r)-a_s, 0)
        pri = C["lg_rate"][pref]
        cols[pref+"_ssn"] = (ds + C["K"]*pri)/(dn + C["K"])
        cols[pref+"_ssn_vs_car"] = cols[pref+"_ssn"] - np.nan_to_num(r, nan=pri)
        if pref in ("p_succ","b_succ"): cols[pref+"_ssn_n"] = dn
    cols["p_prev5_vs_car"] = (df.asof_pitcher_prev5_game_success_rate - df.asof_pitcher_success_rate).values
    cols["p_prev1_vs_prev5"] = (df.asof_pitcher_prev1_game_success_rate - df.asof_pitcher_prev5_game_success_rate).values
    pv = df.pitcher_id.map(art["ppa"].set_index("pitcher_id")["ppa"]).fillna(C["ppa_med"]).values
    cols["p_ppa"] = pv
    cols["p_est_apps"] = cols["p_succ_ssn_n"]/np.clip(pv,5,None)
    cols["p_inning_x_role"] = df.inning.values*np.log1p(pv)
    cols["p_ssn_per_month"] = cols["p_succ_ssn_n"]/np.clip(df.game_month.values,3,None)
    S = art["sit"].set_index("pitcher_id")
    o = df.pitcher_id.map(S["overall"]).values; cols["p_sit_overall"] = o
    mk = sit_masks(df); match = np.full(len(df), np.nan)
    for n2 in C["sits"]:
        v = df.pitcher_id.map(S["sit_"+n2]).values
        cols["p_sit_"+n2] = v; cols["p_sit_"+n2+"_d"] = v - o
        match = np.where(mk[n2] & np.isnan(match), v - o, match)
    cols["p_sit_matched"] = match
    MP = art["matchup"].set_index("key")
    kk = pd.Series(df.pitcher_id.values.astype(np.int64)*100000 + df.batter_id.values)
    c2 = kk.map(MP["n"]).fillna(0).values; s2 = kk.map(MP["succ"]).fillna(0).values
    cols["pb_n"] = c2; cols["pb_rate"] = (s2 + C["MSH"]*C["mu"])/(c2 + C["MSH"])
    cols["pb_logn"] = np.log1p(c2)
    TP = art.get("tm")
    if TP is not None:
        Tm = TP.set_index("pitcher_id")
        for c in C.get("tmsel", []):
            cols[c] = df.pitcher_id.map(Tm[c]).values if c in Tm.columns else np.nan
    for pref, idcol, ncol, ratecol in C["multik"]:
        A = art["anchor_"+idcol]
        j = df[[idcol]].join(A.set_index(idcol), on=idcol)
        a_n = j["a_n"].fillna(0).values; a_s = j["succ_"+pref].fillna(0).values
        n = df[ncol].values; r = df[ratecol].values
        dn = np.maximum(n-a_n, 0); ds = np.maximum(np.nan_to_num(n*r)-a_s, 0)
        lg = C["lg_rate_mk"][pref]
        for k in C["ks"]: cols[f"{pref}_k{k}"] = (ds + k*lg)/(dn + k)
    return pd.DataFrame(cols, index=df.index).astype(np.float32)

class PLR(nn.Module):
    def __init__(s, d_in, k=8, d=6):
        super().__init__(); s.c = nn.Parameter(torch.randn(d_in,k)*.05); s.l = nn.Linear(2*k,d)
    def forward(s, x):
        z = 2*np.pi*x.unsqueeze(-1)*s.c
        return torch.relu(s.l(torch.cat([torch.sin(z), torch.cos(z)],-1))).flatten(1)
class TabM(nn.Module):
    def __init__(s, n, card, k=32, h=256, L=2, emb=0, drop=.1, plr=False, film=True):
        super().__init__(); s.k, s.emb, s.film, s.plr = k, emb, film, None; d = n
        if plr: s.plr = PLR(n); d += n*6
        if emb:
            s.e1 = nn.Embedding(card[0], emb); s.e2 = nn.Embedding(card[1], emb); d += 2*emb
        s.r1 = nn.Parameter(torch.randn(k,d)*.1+1)
        s.ls = nn.ModuleList([nn.Linear(d if i==0 else h, h) for i in range(L)])
        s.dp = nn.Dropout(drop); s.hd = nn.Parameter(torch.randn(k,h)*.02); s.hb = nn.Parameter(torch.zeros(k))
        if film: s.fn = nn.Sequential(nn.Linear(5,32), nn.ReLU(), nn.Linear(32,2*h))
    def forward(s, xn, xc, tt):
        z = xn
        if s.plr is not None: z = torch.cat([z, s.plr(xn)], -1)
        if s.emb: z = torch.cat([z, s.e1(xc[:,0]), s.e2(xc[:,1])], -1)
        z = z.unsqueeze(1)*s.r1
        for i, l in enumerate(s.ls):
            z = torch.relu(l(z))
            if i == 0:
                z = s.dp(z)
                if s.film:
                    g, b = s.fn(tt).chunk(2,-1); z = z*(1+g.unsqueeze(1))+b.unsqueeze(1)
        return ((z*s.hd).sum(-1)+s.hb).mean(1)
class FTT(nn.Module):
    """run6 원본 FT-Transformer (시간토큰 없음, card[:4], 기본 ReLU)"""
    def __init__(s, n, card, d=32, L=3, heads=4, drop=.1):
        super().__init__()
        s.w = nn.Parameter(torch.randn(n,d)*.05); s.b = nn.Parameter(torch.zeros(n,d))
        s.ce = nn.ModuleList([nn.Embedding(c,d) for c in card[:4]])
        s.cls = nn.Parameter(torch.randn(1,1,d)*.05)
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, heads, d*2, drop, batch_first=True, norm_first=True), L)
        s.out = nn.Linear(d,1)
    def forward(s, xn, xc, tt):
        tok = xn.unsqueeze(-1)*s.w + s.b
        ct = torch.stack([e(xc[:,i]) for i,e in enumerate(s.ce)], 1)
        z = torch.cat([s.cls.expand(xn.shape[0],-1,-1), tok, ct], 1)
        return s.out(s.tr(z)[:,0]).squeeze(-1)

class MTabM(nn.Module):
    """멀티태스크 TabM: trunk 공유 + 헤드 5개 (y, reverse, middle, ball, strike).
       추론에서는 헤드 0(y)만 쓴다. 나머지 헤드는 학습 때 표현을 깎는 역할만 했다."""
    def __init__(s, n, card, k=64, h=512, L=3, emb=0, drop=.1, plr=True, film=True, nh=5):
        super().__init__(); s.k, s.emb, s.film, s.plr, s.nh = k, emb, film, None, nh; d = n
        if plr: s.plr = PLR(n); d += n*6
        if emb:
            s.es = nn.ModuleList([nn.Embedding(c, emb) for c in card[:2]]); d += 2*emb
        s.r1 = nn.Parameter(torch.randn(k,d)*.1+1)
        s.ls = nn.ModuleList([nn.Linear(d if i==0 else h, h) for i in range(L)])
        s.dp = nn.Dropout(drop)
        s.hd = nn.Parameter(torch.randn(nh,k,h)*.02); s.hb = nn.Parameter(torch.zeros(nh,k))
        if film: s.fn = nn.Sequential(nn.Linear(5,32), nn.ReLU(), nn.Linear(32,2*h))
    def forward(s, xn, xc, tt):
        z = xn
        if s.plr is not None: z = torch.cat([z, s.plr(xn)], -1)
        if s.emb: z = torch.cat([z]+[e(xc[:,i]) for i,e in enumerate(s.es)], -1)
        z = z.unsqueeze(1)*s.r1
        for i, l in enumerate(s.ls):
            z = torch.relu(l(z))
            if i == 0:
                z = s.dp(z)
                if s.film:
                    g, b = s.fn(tt).chunk(2,-1); z = z*(1+g.unsqueeze(1))+b.unsqueeze(1)
        return ((z.unsqueeze(1)*s.hd).sum(-1)+s.hb).mean(-1)[:, 0]

def _timefeat(df):
    mth = df.game_month.values.astype(np.float32); ssn = df.season.values.astype(np.float32)
    return np.stack([np.sin(2*np.pi*mth/12), np.cos(2*np.pi*mth/12),
                     np.sin(4*np.pi*mth/12), np.cos(4*np.pi*mth/12),
                     (ssn-2019)/6.0], 1).astype(np.float32)

def nn_predict_mt(X, df, art, dev, budget, t0):
    """멀티태스크 NN 축. 전용 피처목록/정규화를 쓰므로 기존 NN 과 경로가 다르다."""
    NUM = art["feat_mt"]; nz = art["norm_mt"]; CARD = art["card_mt"]
    Xn = np.nan_to_num(X[NUM].values.astype(np.float32), nan=0., posinf=0., neginf=0.)
    Xn = np.clip((Xn-nz["mu"])/nz["sd"], -6, 6)
    CAT = ["pitcher_id","batter_id","pitcher_team_id","batter_team_id","pitcher_hand",
           "batter_hand","base_state","game_type","top_bottom"]
    Xc = np.maximum(X[CAT].values.astype(np.int64), 0)
    for i, c in enumerate(CARD): Xc[:, i] = np.minimum(Xc[:, i], c-1)
    tn = torch.tensor(Xn, device=dev); tc = torch.tensor(Xc, device=dev)
    tt = torch.tensor(_timefeat(df), device=dev)
    cap = 1.5e8 if dev == "cpu" else 3.0e8
    outs = []; last = 0.0
    for name, cfg in art["mt_cfg"]:
        f = os.path.join(MDIR, f"nn_{name}.pt")
        if not os.path.exists(f): continue
        if time.time()-t0 + last*1.3 > budget:
            print(f"    시간예산 예측 초과, MT-NN {len(outs)}개에서 중단", flush=True); break
        tm = time.time()
        net = MTabM(Xn.shape[1], CARD, k=cfg["k"], h=cfg["h"], L=cfg["L"], emb=cfg["emb"],
                    drop=cfg["drop"], plr=cfg["plr"], film=cfg["film"])
        bs = int(np.clip(cap/max(cfg["k"]*cfg["h"]*3, 1), 512, 65536))
        net.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
        net = net.to(dev).eval(); ps = []
        try:
            with torch.no_grad():
                for j in range(0, len(Xn), bs):
                    sl = slice(j, j+bs)
                    ps.append(net(tn[sl], tc[sl], tt[sl]).float().cpu().numpy())
            outs.append(np.concatenate(ps)); last = time.time()-tm
            print(f"    MT-NN {name} bs={bs} ok  {time.time()-t0:.0f}s (+{last:.0f}s)", flush=True)
        except Exception as e:
            print(f"    !! MT-NN {name} 건너뜀: {type(e).__name__}", flush=True)
        del net
        if dev != "cpu": torch.cuda.empty_cache()
    if not outs: raise RuntimeError("MT-NN 전부 실패")
    # net 출력은 로짓이다(기존 nn_predict 와 동일 규약). 여기서 시그모이드를 씌우지 않는다.
    return np.mean(outs, 0), len(outs)

def nn_predict(X, df, art, dev, budget, t0):
    NUM = art["feat_num"]; nz = art["norm"]; CARD = art["card"]
    Xn = np.nan_to_num(X[NUM].values.astype(np.float32), nan=0., posinf=0., neginf=0.)
    Xn = np.clip((Xn-nz["mu"])/nz["sd"], -6, 6)
    CAT = ["pitcher_id","batter_id","pitcher_team_id","batter_team_id","pitcher_hand",
           "batter_hand","base_state","game_type","top_bottom"]
    Xc = np.maximum(X[CAT].values.astype(np.int64), 0)
    for i, c in enumerate(CARD): Xc[:, i] = np.minimum(Xc[:, i], c-1)
    mth = df.game_month.values.astype(np.float32); ssn = df.season.values.astype(np.float32)
    Tf = np.stack([np.sin(2*np.pi*mth/12), np.cos(2*np.pi*mth/12),
                   np.sin(4*np.pi*mth/12), np.cos(4*np.pi*mth/12), (ssn-2019)/6.0], 1).astype(np.float32)
    tn = torch.tensor(Xn, device=dev); tc = torch.tensor(Xc, device=dev); tt = torch.tensor(Tf, device=dev)
    # 모델별 배치 자동 산정 (활성화 텐서 크기 기준). GPU 는 VRAM, CPU 는 고정.
    if dev != "cpu":
        vram = torch.cuda.get_device_properties(dev).total_memory/2**30
        cap = 1.5e8 if vram < 16 else 4.0e8
    else:
        cap = 4.0e7
    outs = []; last = 0.0
    for name, cfg in art["nn_cfg"]:
        f = os.path.join(MDIR, f"nn_{name}.pt")
        if not os.path.exists(f): continue
        # 직전 모델 소요시간으로 다음 모델을 예측해 '시작 전에' 중단 (초과 방지)
        if time.time()-t0 + last*1.3 > budget:
            print(f"    시간예산 예측 초과, NN {len(outs)}개에서 중단", flush=True); break
        tm = time.time()
        if cfg["arch"] == "TabM":
            net = TabM(Xn.shape[1], CARD, k=cfg["k"], h=cfg["h"], L=cfg["L"], emb=cfg["emb"],
                       drop=cfg["drop"], plr=cfg["plr"], film=cfg["film"])
            d_in = Xn.shape[1]*(7 if cfg["plr"] else 1) + 2*cfg["emb"]
            bs = int(np.clip(cap/max(cfg["k"]*max(cfg["h"], d_in), 1), 512, 65536))
        else:
            net = FTT(Xn.shape[1], CARD, d=cfg["d"], L=cfg["L"], heads=cfg["heads"], drop=cfg["drop"])
            ntok = Xn.shape[1]+5
            bs = int(np.clip(cap/max(ntok*ntok*cfg["heads"], 1), 512, 65536))
        net.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
        net = net.to(dev).eval(); ps = []
        try:
            with torch.no_grad():
                for j in range(0, len(Xn), bs):
                    sl = slice(j, j+bs)
                    ps.append(net(tn[sl], tc[sl], tt[sl]).float().cpu().numpy())
            outs.append(np.concatenate(ps)); last = time.time()-tm
            print(f"    NN {name} ({cfg['arch']}) bs={bs} ok  {time.time()-t0:.0f}s (+{last:.0f}s)", flush=True)
        except Exception as e:
            print(f"    !! NN {name} 건너뜀: {type(e).__name__}", flush=True)
        del net
        if dev != "cpu": torch.cuda.empty_cache()
    if not outs: raise RuntimeError("NN 전부 실패")
    return np.mean(outs, 0), len(outs)

def main():
    t0 = time.time()
    art = dict(consts=json.load(open(os.path.join(MDIR,"consts.json"))),
               anchor_pitcher_id=pd.read_csv(os.path.join(MDIR,"anchor_pitcher_id.csv")),
               anchor_batter_id=pd.read_csv(os.path.join(MDIR,"anchor_batter_id.csv")),
               sit=pd.read_csv(os.path.join(MDIR,"sit_pitcher.csv")),
               matchup=pd.read_csv(os.path.join(MDIR,"matchup.csv")),
               ppa=pd.read_csv(os.path.join(MDIR,"ppa.csv")),
               feat_num=json.load(open(os.path.join(MDIR,"feat_num.json"))),
               card=json.load(open(os.path.join(MDIR,"nn_card.json"))),
               nn_cfg=json.load(open(os.path.join(MDIR,"nn_cfg.json"))),
               norm=np.load(os.path.join(MDIR,"nn_norm.npz")),
               feat_mt=(json.load(open(os.path.join(MDIR,"feat_nn_mt.json")))
                        if os.path.exists(os.path.join(MDIR,"feat_nn_mt.json")) else None),
               card_mt=(json.load(open(os.path.join(MDIR,"card_mt.json")))
                        if os.path.exists(os.path.join(MDIR,"card_mt.json")) else None),
               mt_cfg=(json.load(open(os.path.join(MDIR,"mt_cfg.json")))
                       if os.path.exists(os.path.join(MDIR,"mt_cfg.json")) else []),
               norm_mt=(np.load(os.path.join(MDIR,"nn_norm_mt.npz"))
                        if os.path.exists(os.path.join(MDIR,"nn_norm_mt.npz")) else None),
               tm=(pd.read_csv(os.path.join(MDIR,"tm_profile.csv"))
                   if os.path.exists(os.path.join(MDIR,"tm_profile.csv")) else None))
    fx = json.load(open(os.path.join(MDIR,"feat_xgb.json")))
    fn = json.load(open(os.path.join(MDIR,"feat_nn.json")))
    test = pd.read_csv(os.path.join(DDIR,"test.csv"), encoding="utf-8-sig")
    sub  = pd.read_csv(os.path.join(DDIR,"sample_submission.csv"), encoding="utf-8-sig")
    print(f"test {test.shape}", flush=True)
    X = featurize(test, art)
    C = art["consts"]
    # ── 성분 스태킹: reverse/middle/ball/strike 예측 로짓을 y 모델의 피처로 ──
    # (성분 라벨은 학습 때 보조 타깃으로만 쓰였고, 여기선 입력으로 쓰지 않는다)
    BF = C.get("base_feat")
    if BF:
        for c in BF:
            if c not in X.columns: X[c] = np.nan
        dmb = xgb.DMatrix(X[BF], missing=np.nan); got = {}
        for c in C.get("comp", []):
            f = os.path.join(MDIR, f"cmp_{c}.json")
            if not os.path.exists(f): continue
            bst = xgb.Booster(); bst.load_model(f)
            got["cmp_"+c] = sp.logit(np.clip(bst.predict(dmb), 1e-6, 1-1e-6))
        for k, v in got.items(): X[k] = v
        if {"cmp_reverse","cmp_middle"} <= set(got):
            X["cmp_bad"] = got["cmp_reverse"] + got["cmp_middle"]
        if {"cmp_strike","cmp_ball"} <= set(got):
            X["cmp_zone"] = got["cmp_strike"] - got["cmp_ball"]
        print(f"  성분모델 {len(got)}개  {time.time()-t0:.0f}s", flush=True)
    for c in set(fx)|set(fn):
        if c not in X.columns: X[c] = np.nan
    print(f"  피처 {time.time()-t0:.0f}s", flush=True)

    dm = xgb.DMatrix(X[fx], missing=np.nan); ps = []
    for f in sorted(os.listdir(MDIR)):
        if f.startswith("xgb_") and f.endswith(".json"):
            bst = xgb.Booster(); bst.load_model(os.path.join(MDIR, f)); ps.append(bst.predict(dm))
    px = np.clip(np.mean(ps, 0), 1e-6, 1-1e-6)
    print(f"  XGB {len(ps)}개 mean={px.mean():.4f}  {time.time()-t0:.0f}s", flush=True)
    pl = None
    try:
        import lightgbm as lgbm
        lp = []
        for f in sorted(os.listdir(MDIR)):
            if f.startswith("lgb_") and f.endswith(".txt"):
                lp.append(lgbm.Booster(model_file=os.path.join(MDIR, f)).predict(X[fx].values))
        if lp:
            pl = np.clip(np.mean(lp, 0), 1e-6, 1-1e-6)
            print(f"  LGB {len(lp)}개 mean={pl.mean():.4f}  {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"  !! LGB 실패 -> XGB 만 사용: {type(e).__name__}: {e}", flush=True)

    pc = None
    try:
        from catboost import CatBoostClassifier, Pool
        cbf = os.path.join(MDIR, "cb.cbm")
        if os.path.exists(cbf):
            CF = art["consts"]["cat_feat"]
            Zc = X[fx].copy()
            for c in CF: Zc[c] = Zc[c].fillna(-1).astype(np.int32).astype(str)
            cb = CatBoostClassifier(); cb.load_model(cbf)
            pc = np.clip(cb.predict_proba(Pool(Zc, cat_features=CF))[:, 1], 1e-6, 1-1e-6)
            print(f"  CatBoost mean={pc.mean():.4f}  {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"  !! CatBoost 실패 -> 제외: {type(e).__name__}: {e}", flush=True)

    # 4축 가중치. 사용 불가한 축은 0 으로 두고 나머지를 재정규화한다.
    B3 = art["consts"].get("blend3") or {"xgb": 1-art["consts"]["blend_w"],
                                         "lgb": 0.0, "nn": art["consts"]["blend_w"]}
    wx, wl, wn = float(B3["xgb"]), float(B3["lgb"]), float(B3["nn"])
    wc = float(B3.get("cb", 0.0))
    if pl is None: wl = 0.0
    if pc is None: wc = 0.0
    lx = sp.logit(px); ll = sp.logit(pl) if pl is not None else 0.0
    lc = sp.logit(pc) if pc is not None else 0.0
    tree = wx*lx + wl*ll + wc*lc            # 트리 축 (가중합, 정규화 전)
    tw = max(wx+wl+wc, 1e-6)
    p = np.clip(sp.expit(tree/tw), 1e-6, 1-1e-6)   # NN 실패 시 최종값
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    # 전체 한도 600초. 피처+XGB 소요를 빼고 NN 에 배정, 저장 여유 60초 확보.
    LIMIT = 600.0
    devs = ([("cuda", LIMIT-90-(time.time()-t0)), ("cpu", LIMIT-90-(time.time()-t0))]
            if torch.cuda.is_available() else [("cpu", LIMIT-90-(time.time()-t0))])
    for dev, budget in devs:
        try:
            zn, k = nn_predict(X[fn], test, art, dev, budget, t0)
            if k == 0: raise RuntimeError("NN 없음")
            # NN 축 = (1-alpha)*기존 풀 + alpha*멀티태스크 풀. 실패하면 alpha=0 으로 되돌린다.
            al = float(art["consts"].get("mt_alpha", 0.0)); km = 0
            if al > 0 and art["feat_mt"] and art["mt_cfg"]:
                try:
                    zm, km = nn_predict_mt(X, test, art, dev,
                                           LIMIT-60-(time.time()-t0), t0)
                    if km: zn = (1.0-al)*zn + al*zm
                    else: al = 0.0
                except Exception as e:
                    print(f"  !! MT-NN 실패 -> alpha=0: {type(e).__name__}: {e}", flush=True)
                    al = 0.0
            sc = (1.0-wn)/tw                    # 트리 축 합이 (1-wn) 이 되도록
            p = np.clip(sp.expit(tree*sc + wn*zn), 1e-6, 1-1e-6)
            print(f"  4축 블렌딩 XGB{wx:.2f}/LGB{wl:.2f}/CB{wc:.2f}/NN{wn:.2f} "
                  f"(기존NN {k}개 + MT {km}개, alpha={al:.2f}, {dev}) "
                  f"mean={p.mean():.4f}  {time.time()-t0:.0f}s", flush=True)
            break
        except Exception as e:
            print(f"  !! NN({dev}) 실패: {type(e).__name__}: {e}", flush=True)
    else:
        print("  !! NN 전부 실패 -> XGB 단독", flush=True)

    # ── 시즌 드리프트 보정 ──
    # 리그 제구율은 2019 .5495 -> 2024 .4897 로 매년 하락하는데 학습에 2025 가 없어
    # 모델이 2024 수준으로 예측한다. 학습 시즌 추세로 추정한 만큼만 레벨을 내린다.
    # 평가 데이터의 분포는 전혀 참조하지 않는 상수 시프트다.
    dft = float(art["consts"].get("drift", 0.0))
    if dft:
        before = p.mean()
        p = np.clip(sp.expit(sp.logit(np.clip(p, 1e-6, 1-1e-6)) + dft), 1e-6, 1-1e-6)
        print(f"  드리프트 보정 {dft:+.4f}  mean {before:.4f} -> {p.mean():.4f}", flush=True)

    pm = dict(zip(test[ID].values, p))
    # 폴백은 학습 데이터에서 온 상수만 쓴다. 평가 데이터 전체 평균을 쓰면
    # 그 행의 예측이 다른 행에 의존하게 되어 행 독립 예측 원칙에 어긋난다.
    fb = float(art["consts"]["mu"])
    sub[TGT] = [pm.get(r, fb) for r in sub[ID].values]
    os.makedirs(ODIR, exist_ok=True)
    sub.to_csv(os.path.join(ODIR,"submission.csv"), index=False, encoding="utf-8")
    print(f"saved {len(sub)}  총 {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
