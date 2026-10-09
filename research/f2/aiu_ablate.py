# SPDX-License-Identifier: GPL-3.0-only
"""F2 for the coordinator's AIU-2026 question (10-08 23:0x): what does the A3 corrector's stage 2 lose without its AIU columns?
Local measurement only (no candidate is built). Corrector code = src/ctx_corr3.py's (imported, unchanged); stage 1 is fitted once and
shared by all variants (it has no AIU column). Stage-2 column sets:
  all       : 514's (ctx_v3: aiu_m_txo, aiu_m_add, aiu_m_valid, aiu_d_pdd, aiu_d_txo + aiu_m_gap, aiu_d_gap)
  no_month  : without the monthly taxi_out_additional_time columns (aiu_m_txo, aiu_m_add, aiu_m_valid, aiu_m_gap)
  no_day    : without the daily all_pre_departure_delays columns (aiu_d_pdd, aiu_d_txo, aiu_d_gap)
  no_aiu    : without all seven
Settings: val (ten day folds, pairwise-nested stage-1 inputs, as ctx_corr --setting val), fwd (January-only stage 1 and stage 2 applied to
the fwd base, as ctx_corr --setting fwd), and optionally F (the ten val S_k averaged and applied to a leave-Feb/Jun-out base, as full).
Delta = MSE(stage 1 + 0.5 x stage 2 + LOBT window) of the variant minus that of "all", day-block bootstrap (2,000 draws, seed 0).
    python -u research/f2/aiu_ablate.py --vbase <val base tag> [--fwdbase <tag>] [--fbase <fj base tag>] --ctx <ctx table> --ctx_sha <16> --tag <out>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "research/f2")
import src.adsb_corr as ac  # noqa: E402
import src.ctx_corr as cc  # noqa: E402
from src.load import CACHE  # noqa: E402
import m3_corr as m3  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
T0 = time.time()
DROP = {"all": [], "no_month": ["aiu_m_txo", "aiu_m_add", "aiu_m_valid", "aiu_m_gap"], "no_day": ["aiu_d_pdd", "aiu_d_txo", "aiu_d_gap"]}
DROP["no_aiu"] = DROP["no_month"] + DROP["no_day"]


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def fit_cols(M, t, names, a):
    import lightgbm as lgb
    params = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=a.threads, seed=a.seed, deterministic=True, force_col_wise=True)
    ds = lgb.Dataset(M, np.clip(t, -cc.CLIP, cc.CLIP), feature_name=names, categorical_feature=[c for c in cc.CAT2 if c in names], free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=a.rounds2)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--vbase", required=True)
    pa.add_argument("--fwdbase", default="")
    pa.add_argument("--fbase", default="")
    pa.add_argument("--ctx", required=True)
    pa.add_argument("--ctx_sha", required=True)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--threads", type=int, default=2)
    g = pa.parse_args()
    a = SimpleNamespace(adsb="adsb_v1nd", adsb_sha="0d03ff06267e0dee", geo="geo_v1nd", geo_sha="43310b9bbeefe81b", feats="feats_v25", folds=10,
                        rounds=300, rounds2=400, max_pred=3600.0, threads=g.threads, seed=g.seed, scale2=0.5)
    ac.below_normal()
    m3.setup()
    X2 = list(cc.X2)
    idx = {v: [i for i, c in enumerate(X2) if c not in set(d)] for v, d in DROP.items()}
    names = {v: [X2[i] for i in idx[v]] for v in DROP}
    A = cc.stage1_inputs(a)
    ctx = cc.checked(g.ctx, g.ctx_sha).set_index("MVT_ID_mvt")
    V = pd.read_parquet(CACHE / f"val_{g.vbase}_s{g.seed}.parquet")
    T = ac.design(V, A)
    T = T[np.isfinite(T["y"].to_numpy())]
    RV = cc.Rows(V, A, a, ctx)
    yV = V["y"].to_numpy(dtype=float)
    keepV = np.isfinite(yV)
    fo, fT, models = cc.stage1_folds(T, a)
    oof = np.zeros(len(T))
    for j, m in enumerate(models):
        oof[fT == j] = ac.predict(m, T[fT == j])
    d1V = np.zeros(len(V))
    T_pos = T["pos"].to_numpy()
    d1V[T_pos] = oof
    s1V = np.clip(V["pred"].to_numpy(dtype=float) + d1V, 0, None)
    MV = RV.design2(s1V, d1V).astype(np.float32)
    appV = s1V < a.max_pred
    fold_row = pd.Series(RV.day).map(fo).to_numpy()
    log("stage 1 done, nested fits next")
    nested = cc.nested_stage1(T, fT, a)
    x2V = {v: np.zeros(len(V)) for v in DROP}
    S = {v: [] for v in DROP}
    for k in range(a.folds):
        dm = np.zeros(len(T))
        for j in range(a.folds):
            if j != k:
                dm[fT == j] = nested[(j, k)]
        delta = RV.stage1(dm, T_pos)
        s1 = np.clip(V["pred"].to_numpy(dtype=float) + delta, 0, None)
        tr = (fold_row != k) & keepV & (s1 < a.max_pred)
        Mk = RV.design2(s1, delta).astype(np.float32)
        sel = (fold_row == k) & appV
        for v in DROP:
            m = fit_cols(Mk[tr][:, idx[v]], (yV - s1)[tr], names[v], a)
            S[v].append(m)
            x2V[v][sel] = m.predict(MV[sel][:, idx[v]])
        del Mk
        log(f"val fold {k} done")
    rep = {"tag": g.tag, "vbase": g.vbase, "fwdbase": g.fwdbase, "fbase": g.fbase, "ctx": g.ctx, "drop": DROP}
    se = lambda p, y: (p - y) ** 2  # noqa: E731
    dayV = RV.day
    s1w = se(m3.windowed(V, s1V, a.feats), yV)
    pv = {v: se(m3.windowed(V, np.clip(s1V + a.scale2 * x2V[v], 0, None), a.feats), yV) for v in DROP}
    rep["val"] = {v: {"stage2": m3.boot(pv[v] - s1w, dayV), "vs_all": m3.boot(pv[v] - pv["all"], dayV)} for v in DROP}
    log("val: " + json.dumps({v: (rep["val"][v]["stage2"]["dmse"], rep["val"][v]["vs_all"]["dmse"]) for v in DROP}))
    if g.fwdbase:
        B = pd.read_parquet(CACHE / f"fwd_{g.fwdbase}_s{g.seed}.parquet")
        RB = cc.Rows(B, A, a, ctx)
        jan_T = (T["day"].str[5:7] == "01").to_numpy()
        mdl = ac.fit(T[jan_T], a.seed, a.rounds, a.threads)
        d1B = np.zeros(len(B))
        if len(RB.X):
            d1B[RB.X["pos"].to_numpy()] = ac.predict(mdl, RB.X)
        dm = np.zeros(len(T))
        for j in range(a.folds):
            inj = jan_T & (fT == j)
            if inj.any():
                m = ac.fit(T[jan_T & (fT != j)], a.seed, a.rounds, a.threads)
                dm[inj] = ac.predict(m, T[inj])
        dv = RV.stage1(dm, T_pos)
        s1v = np.clip(V["pred"].to_numpy(dtype=float) + dv, 0, None)
        jan = (pd.Series(RV.day).str[5:7] == "01").to_numpy()
        tr = jan & keepV & (s1v < a.max_pred)
        Mj = RV.design2(s1v, dv).astype(np.float32)[tr]
        s1B = np.clip(B["pred"].to_numpy(dtype=float) + d1B, 0, None)
        appB = s1B < a.max_pred
        MB = RB.design2(s1B, d1B).astype(np.float32)
        yB = B["y"].to_numpy(dtype=float)
        s1wB = se(m3.windowed(B, s1B, a.feats), yB)
        pb = {}
        for v in DROP:
            m = fit_cols(Mj[:, idx[v]], (yV - s1v)[tr], names[v], a)
            x = np.zeros(len(B))
            x[appB] = m.predict(MB[appB][:, idx[v]])
            pb[v] = se(m3.windowed(B, np.clip(s1B + a.scale2 * x, 0, None), a.feats), yB)
        rep["fwd"] = {v: {"stage2": m3.boot(pb[v] - s1wB, RB.day), "vs_all": m3.boot(pb[v] - pb["all"], RB.day)} for v in DROP}
        log("fwd: " + json.dumps({v: (rep["fwd"][v]["stage2"]["dmse"], rep["fwd"][v]["vs_all"]["dmse"]) for v in DROP}))
    if g.fbase:
        F = pd.read_parquet(CACHE / f"fj_{g.fbase}_s{g.seed}.parquet")
        RF = cc.Rows(F, A, a, ctx)
        d1F = np.zeros(len(F))
        if len(RF.X):
            d1F[RF.X["pos"].to_numpy()] = np.mean([ac.predict(m, RF.X) for m in models], axis=0)
        s1F = np.clip(F["pred"].to_numpy(dtype=float) + d1F, 0, None)
        appF = s1F < a.max_pred
        MF = RF.design2(s1F, d1F).astype(np.float32)
        yF = F["y"].to_numpy(dtype=float)
        s1wF = se(m3.windowed(F, s1F, a.feats), yF)
        pf = {}
        for v in DROP:
            x = np.zeros(len(F))
            x[appF] = np.mean([m.predict(MF[appF][:, idx[v]]) for m in S[v]], axis=0)
            pf[v] = se(m3.windowed(F, np.clip(s1F + a.scale2 * x, 0, None), a.feats), yF)
        rep["F"] = {v: {"stage2": m3.boot(pf[v] - s1wF, RF.day), "vs_all": m3.boot(pf[v] - pf["all"], RF.day)} for v in DROP}
        log("F: " + json.dumps({v: (rep["F"][v]["stage2"]["dmse"], rep["F"][v]["vs_all"]["dmse"]) for v in DROP}))
    rep["seconds"] = round(time.time() - T0)
    Path("research/f2/out").mkdir(parents=True, exist_ok=True)
    (Path("research/f2/out") / f"aiu_ablate_{g.tag}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    log("saved")


if __name__ == "__main__":
    main()
