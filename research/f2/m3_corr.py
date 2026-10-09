# SPDX-License-Identifier: GPL-3.0-only
"""F2 ruler M3 for the A3 corrector (vault research/2026-10-08-f2-four-months.md section 1): does a corrector fitted on Jan/Jul 2025
transfer to months it never saw (Feb/Jun 2025), and would adding Feb/Jun 2025 rows to its training help?

Two LightGBM-only worlds built with the same leg command (m6r27skwin + g_band), differing only in the held-out months:
  V = cache/val_<vbase>_s<seed>.parquet  (legs trained without Jan/Jul 2025, rows = Jan + Jul 2025)
  F = cache/fj_<fbase>_s<seed>.parquet   (legs trained without Feb/Jun 2025, rows = Feb + Jun 2025; research/f2/model3_fj.py)
The corrector code is src/ctx_corr3.py's (imported, unchanged): stage-1 ADS-B + geo LightGBM, stage-2 context + AIU LightGBM, --scale2,
the LOBT window last. The context table must hold Jan/Feb/Jun/Jul 2025 rows (f2ctx_v3 = src/feats_ctx.py --months 1,2,6,7 + v2 + aiu).
Stage-1 tables (adsb_v1nd, geo_v1nd) hold no Feb/Jun 2025 day, so on F stage 1 changes nothing and the measure is stage 2.

Variants (stage-2 delta = MSE(after) - MSE(before stage 2), both with the window, divided by the evaluated rows):
  I_V : val recipe on V (day folds, pairwise-nested stage-1 inputs) = the in-month reference (as ctx_corr --setting val)
  T   : the ten val S_k averaged and applied to F (as ctx_corr --setting full applies them to the test rows) = transfer
  I_F : stage 2 fitted on F's own day folds (no Jan/Jul rows) = in-month on Feb/Jun
  T1  : one stage-2 model on all V rows (stage-1 out-of-fold inputs), applied to F = single-model transfer
  U_F : as T1 plus the F rows of the other F folds, applied to F fold k = four-month corrector on Feb/Jun   (decision: U_F - T1)
  U_V : as I_V with all F rows added to each S_k's training rows = four-month corrector on Jan/Jul         (decision: U_V - I_V)
Day-block bootstrap (UTC day of MVT from the context table, 2,000 draws, seed 0) for every delta and contrast.
Writes research/f2/out/m3_corr_<tag>.json and cache/fj_<tag>T_s<seed>.parquet (F after T), cache/val_<tag>V_s<seed>.parquet (V after I_V).
    python -u research/f2/m3_corr.py --vbase m6r27skwing --fbase f2skwing --seed 0 --ctx f2ctx_v3 --ctx_sha <16> --tag f2m3 --threads 4
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
import src.adsb_corr as ac  # noqa: E402
import src.ctx_corr as cc  # noqa: E402
import src.ctx_corr2 as c2  # noqa: E402
import src.ctx_corr3 as c3  # noqa: E402
from src.load import CACHE  # noqa: E402
from src.lobt_window import feats_d_lobt, mu_of, project  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
T0 = time.time()
OUT = Path("research/f2/out")


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def setup():
    """The same global stage-2 configuration src/ctx_corr3.main() sets (without --drop2, --aiu_date mvt)."""
    cc.CTX_COLS = cc.CTX_COLS + c2.NEW_CTX + c3.AIU_CTX
    cc.X2 = cc.X2 + c2.NEW_CTX + list(c2.PRED_GAP) + c3.AIU_CTX + c3.AIU_GAP
    cc.Rows = c3.Rows3


def windowed(b: pd.DataFrame, s: np.ndarray, feats: str) -> np.ndarray:
    _, mu = mu_of(b)
    p_art, g, dsch = b["p_art"].to_numpy(dtype=float), b["g"].to_numpy().astype(bool), b["dsch"].to_numpy(dtype=float)
    pw, _, _ = project(s, mu, p_art, g, dsch, feats_d_lobt(feats, b["MVT_ID_mvt"].to_numpy()))
    return pw


def windowed_full(b: pd.DataFrame, s: np.ndarray, feats: str):
    _, mu = mu_of(b)
    p_art, g, dsch = b["p_art"].to_numpy(dtype=float), b["g"].to_numpy().astype(bool), b["dsch"].to_numpy(dtype=float)
    return project(s, mu, p_art, g, dsch, feats_d_lobt(feats, b["MVT_ID_mvt"].to_numpy()))


def boot(d: np.ndarray, day: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Day-block bootstrap of mean(d): point, sd, P(<0)."""
    codes, days = pd.factorize(day)
    s = np.bincount(codes, weights=d)
    n = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    est = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return {"dmse": round(float(d.mean()), 2), "sd": round(float(est.std()), 2), "p_lt0": round(float((est < 0).mean()), 4), "days": int(len(s))}


def effect(se_new, se_old, day, month, ap) -> dict:
    d = se_new - se_old
    r = boot(d, day)
    r["by_month"] = {str(m): round(float(d[month == m].sum() / len(d)), 2) for m in sorted(np.unique(month))}
    r["by_month_own_rows"] = {str(m): round(float(d[month == m].mean()), 2) for m in sorted(np.unique(month))}
    r["by_airport"] = {k: round(float(d[ap == k].sum() / len(d)), 2) for k in sorted(np.unique(ap))}
    return r


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--vbase", required=True)
    pa.add_argument("--fbase", required=True)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--ctx", required=True)
    pa.add_argument("--ctx_sha", required=True)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--threads", type=int, default=4)
    g = pa.parse_args()
    a = SimpleNamespace(adsb="adsb_v1nd", adsb_sha="0d03ff06267e0dee", geo="geo_v1nd", geo_sha="43310b9bbeefe81b", feats="feats_v25", folds=10,
                        rounds=300, rounds2=400, max_pred=3600.0, threads=g.threads, seed=g.seed, scale2=0.5, ctx=g.ctx, ctx_sha=g.ctx_sha)
    ac.below_normal()
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    A = cc.stage1_inputs(a)
    ctx = cc.checked(a.ctx, a.ctx_sha).set_index("MVT_ID_mvt")
    V = pd.read_parquet(CACHE / f"val_{g.vbase}_s{g.seed}.parquet")
    F = pd.read_parquet(CACHE / f"fj_{g.fbase}_s{g.seed}.parquet")
    assert F["y"].notna().all() and V["y"].notna().all()
    T = ac.design(V, A)
    T = T[np.isfinite(T["y"].to_numpy())]
    RV = cc.Rows(V, A, a, ctx)
    assert set(np.unique(RV.day)) == set(T["day"].unique())
    RF = cc.Rows(F, A, a, ctx)
    log(f"V rows {len(V):,} (matched {len(T):,}), F rows {len(F):,} (matched {len(RF.X):,}), stage-2 columns {len(cc.X2)}")
    yV, yF = V["y"].to_numpy(dtype=float), F["y"].to_numpy(dtype=float)
    keepV, keepF = np.isfinite(yV), np.isfinite(yF)
    # stage 1 on V (val recipe) and its use on F (full recipe: mean of the fold models)
    fo, fT, models = cc.stage1_folds(T, a)
    oof = np.zeros(len(T))
    for j, m in enumerate(models):
        oof[fT == j] = ac.predict(m, T[fT == j])
    d1V = np.zeros(len(V))
    d1V[T["pos"].to_numpy()] = oof
    d1F = np.zeros(len(F))
    if len(RF.X):
        d1F[RF.X["pos"].to_numpy()] = np.mean([ac.predict(m, RF.X) for m in models], axis=0)
    log("stage 1 done, nested fits next")
    nested = cc.nested_stage1(T, fT, a)
    S, fold_row = cc.stage2_val_models(RV, T, fT, fo, models, nested, a, keepV)
    s1V = np.clip(V["pred"].to_numpy(dtype=float) + d1V, 0, None)
    s1F = np.clip(F["pred"].to_numpy(dtype=float) + d1F, 0, None)
    MV, MF = RV.design2(s1V, d1V), RF.design2(s1F, d1F)
    appV, appF = s1V < a.max_pred, s1F < a.max_pred
    # I_V and T
    d2 = {}
    x = np.zeros(len(V))
    for k, m in enumerate(S):
        sel = (fold_row == k) & appV
        x[sel] = m.predict(MV[sel])
    d2["I_V"] = x
    x = np.zeros(len(F))
    x[appF] = np.mean([m.predict(MF[appF]) for m in S], axis=0)
    d2["T"] = x
    log("I_V and T done")
    # I_F: F's own day folds
    foF = ac.day_folds(np.unique(RF.day), a.folds, a.seed)
    fold_F = pd.Series(RF.day).map(foF).to_numpy()
    tgtF = yF - s1F
    x = np.zeros(len(F))
    for k in range(a.folds):
        tr = (fold_F != k) & keepF & appF
        m = cc.fit2(MF[tr], tgtF[tr], a.seed, a.rounds2, a.threads)
        sel = (fold_F == k) & appF
        x[sel] = m.predict(MF[sel])
    d2["I_F"] = x
    log("I_F done")
    # T1 and U_F: one model on all V rows (stage-1 out-of-fold inputs), with or without the other F folds
    trV = keepV & appV
    tgtV = yV - s1V
    m1 = cc.fit2(MV[trV], tgtV[trV], a.seed, a.rounds2, a.threads)
    x = np.zeros(len(F))
    x[appF] = m1.predict(MF[appF])
    d2["T1"] = x
    x = np.zeros(len(F))
    for k in range(a.folds):
        trF = (fold_F != k) & keepF & appF
        m = cc.fit2(np.vstack([MV[trV], MF[trF]]), np.concatenate([tgtV[trV], tgtF[trF]]), a.seed, a.rounds2, a.threads)
        sel = (fold_F == k) & appF
        x[sel] = m.predict(MF[sel])
    d2["U_F"] = x
    log("T1 and U_F done")
    # U_V: the val recipe for S_k with every F row added to its training rows
    T_pos = T["pos"].to_numpy()
    allF = keepF & appF
    x = np.zeros(len(V))
    for k in range(a.folds):
        dm = np.zeros(len(T))
        for j in range(a.folds):
            if j != k:
                dm[fT == j] = nested[(j, k)]
        delta = RV.stage1(dm, T_pos)
        s1 = np.clip(V["pred"].to_numpy(dtype=float) + delta, 0, None)
        tr = (fold_row != k) & keepV & (s1 < a.max_pred)
        Mk = RV.design2(s1, delta)
        m = cc.fit2(np.vstack([Mk[tr], MF[allF]]), np.concatenate([(yV - s1)[tr], tgtF[allF]]), a.seed, a.rounds2, a.threads)
        sel = (fold_row == k) & appV
        x[sel] = m.predict(MV[sel])
        del Mk
    d2["U_V"] = x
    log("U_V done")
    # scores, all with the window last, stage 2 x scale2
    feats = a.feats
    dayV, dayF = RV.day, RF.day
    monV = pd.Series(dayV).str[5:7].astype(int).to_numpy()
    monF = pd.Series(dayF).str[5:7].astype(int).to_numpy()
    apV, apF = RV.ap, RF.ap
    se = lambda p, y: (p - y) ** 2  # noqa: E731
    baseV = se(windowed(V, V["pred"].to_numpy(dtype=float), feats), yV)
    s1wV = se(windowed(V, s1V, feats), yV)
    baseF = se(windowed(F, F["pred"].to_numpy(dtype=float), feats), yF)
    s1wF = se(windowed(F, s1F, feats), yF)
    rep = {"tag": g.tag, "vbase": g.vbase, "fbase": g.fbase, "seed": g.seed, "ctx": a.ctx, "ctx_sha": a.ctx_sha, "scale2": a.scale2,
           "rows": {"V": int(len(V)), "F": int(len(F)), "V_matched": int(len(T)), "F_matched": int(len(RF.X))},
           "rmse": {"V_base_window": round(float(np.sqrt(baseV.mean())), 3), "V_stage1_window": round(float(np.sqrt(s1wV.mean())), 3),
                    "F_base_window": round(float(np.sqrt(baseF.mean())), 3), "F_stage1_window": round(float(np.sqrt(s1wF.mean())), 3)},
           "stage1_V": effect(s1wV, baseV, dayV, monV, apV)}
    after = {}
    for k, v in d2.items():
        if k.endswith("_V") or k == "I_V":
            p = se(windowed(V, np.clip(s1V + a.scale2 * v, 0, None), feats), yV)
            after[k] = p
            rep[k] = effect(p, s1wV, dayV, monV, apV)
            rep[k]["rmse"] = round(float(np.sqrt(p.mean())), 3)
        else:
            p = se(windowed(F, np.clip(s1F + a.scale2 * v, 0, None), feats), yF)
            after[k] = p
            rep[k] = effect(p, s1wF, dayF, monF, apF)
            rep[k]["rmse"] = round(float(np.sqrt(p.mean())), 3)
    rep["contrast_U_F_minus_T1"] = boot(after["U_F"] - after["T1"], dayF)
    rep["contrast_U_F_minus_T"] = boot(after["U_F"] - after["T"], dayF)
    rep["contrast_T1_minus_T"] = boot(after["T1"] - after["T"], dayF)
    rep["contrast_I_F_minus_T"] = boot(after["I_F"] - after["T"], dayF)
    rep["contrast_U_V_minus_I_V"] = boot(after["U_V"] - after["I_V"], dayV)
    rep["seconds"] = round(time.time() - T0)
    (OUT / f"m3_corr_{g.tag}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    # corrected bases for the tail measurements (pred and p_art replaced, as ctx_corr writes them)
    for nm, b, s1, v in (("fj", F, s1F, d2["T"]), ("val", V, s1V, d2["I_V"])):
        pw, p2, _ = windowed_full(b, np.clip(s1 + a.scale2 * v, 0, None), feats)
        out = b.copy()
        out["pred"], out["p_art"] = pw, p2
        sfx = "T" if nm == "fj" else "V"
        out.to_parquet(CACHE / f"{nm}_{g.tag}{sfx}_s{g.seed}.parquet", index=False)
    print(json.dumps({k: (rep[k] if not isinstance(rep[k], dict) or "dmse" not in rep[k] else {kk: rep[k][kk] for kk in ("dmse", "sd", "p_lt0", "rmse", "by_month_own_rows") if kk in rep[k]})
                      for k in rep if k not in ("tag",)}, ensure_ascii=False, indent=1))
    log("saved")


if __name__ == "__main__":
    main()
