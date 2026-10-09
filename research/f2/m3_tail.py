# SPDX-License-Identifier: GPL-3.0-only
"""F2 ruler M3 for three tail stages fitted on Jan/Jul (private notes research/2026-10-08-f2-four-months.md section 1.3): artmm, arthist and
aiu_shift (--group block, lam 0.5, cmax 60) on the corrected LightGBM-only worlds written by research/f2/m3_corr.py.

  V = cache/val_<tag>V_s0.parquet (Jan + Jul 2025 after the in-month corrector), F = cache/fj_<tag>T_s0.parquet (Feb + Jun 2025 after the
  Jan/Jul-trained corrector), bases cache/val_<vbase>_s0.parquet and cache/fj_<fbase>_s0.parquet.
The stage code is the tools' own (imported unchanged: tools/artmm.py shift(), tools/arthist.py frame()/betas()/shift(), src/aiu_shift.py
add_classes()/fit_g()/offsets()/shift_days()/block_days()); the only addition is a setting "fj" in artmm's and arthist's FITM (2025 months
other than 2 and 6). Chain order as in 514: corrector -> artmm -> arthist -> (G stages, skipped here) -> aiu_shift. The G stages and antiwin
are not part of this world (antiwin rows are not masked in aiu_shift; about ten rows).
For each stage: delta MSE on V (the val recipe: artmm/arthist val FITM and day-fold beta, aiu_shift other-month fit) and on F (the final-file
recipe: artmm/arthist on the fj FITM with the full beta from V, aiu_shift fitted on all of V), day-block bootstrap, by month and airport.
Report only (registered in 1.3): aiu_shift on F with airport offsets from the labels of the ten months other than Feb/Jun; aiu_shift on F
fitted in-month (Feb rows from Jun, Jun rows from Feb).
    python -u research/f2/m3_tail.py --tag f2m3 --vbase m6r27skwing --fbase f2skwing --ctx f2ctx_v3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
T0 = time.time()
OUT = Path("research/f2/out")
HALF = 3606.0
FJ = [m for m in range(1, 13) if m not in (2, 6)]


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def boot(d: np.ndarray, day: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    codes, _ = pd.factorize(day)
    s = np.bincount(codes, weights=d)
    n = np.bincount(codes).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    est = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return {"dmse": round(float(d.mean()), 2), "sd": round(float(est.std()), 2), "p_lt0": round(float((est < 0).mean()), 4),
            "p_gt0": round(float((est > 0).mean()), 4), "days": int(len(s))}


def effect(new, old, y, day, ap) -> dict:
    d = (new - y) ** 2 - (old - y) ** 2
    r = boot(d, day)
    mon = pd.Series(day).str[5:7].astype(int).to_numpy()
    r["by_month_own_rows"] = {str(m): round(float(d[mon == m].mean()), 2) for m in sorted(np.unique(mon))}
    r["by_airport"] = {k: round(float(d[ap == k].sum() / len(d)), 2) for k in sorted(np.unique(ap))}
    r["changed"] = int((np.abs(new - old) > 1e-9).sum())
    r["move_mse"] = round(float(((new - old) ** 2).mean()), 2)
    return r


def clip_window(v: np.ndarray, dl: np.ndarray) -> np.ndarray:
    v = np.where(np.isfinite(dl), np.clip(v, dl - HALF, dl + HALF), v)
    return np.clip(v, 0, None)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--tag", required=True)
    pa.add_argument("--vbase", required=True)
    pa.add_argument("--fbase", required=True)
    pa.add_argument("--ctx", required=True)
    g = pa.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    am = load("tools/artmm.py", "artmm_f2")
    ah = load("tools/arthist.py", "arthist_f2")
    ai = load("src/aiu_shift.py", "aiu_f2")
    am.FITM["fj"] = FJ
    ah.FITM["fj"] = FJ
    C = {"val": pd.read_parquet(ROOT / "cache" / f"val_{g.tag}V_s0.parquet"), "fj": pd.read_parquet(ROOT / "cache" / f"fj_{g.tag}T_s0.parquet")}
    B = {"val": pd.read_parquet(ROOT / "cache" / f"val_{g.vbase}_s0.parquet"), "fj": pd.read_parquet(ROOT / "cache" / f"fj_{g.fbase}_s0.parquet")}
    t = pq.read_table(ROOT / "cache" / "feats_v25.parquet", columns=["MVT_ID_mvt", "ap", "d_lobt", "dd"]).to_pandas().set_index("MVT_ID_mvt")
    cx = pq.read_table(ROOT / "cache" / f"{g.ctx}.parquet", columns=["MVT_ID_mvt", "day", "t"]).to_pandas().set_index("MVT_ID_mvt")
    cx["day"] = cx["day"].astype(str)
    rep = {"tag": g.tag}
    pred = {}
    meta = {}
    for s in ("val", "fj"):
        c = C[s]
        ids = c["MVT_ID_mvt"].to_numpy()
        meta[s] = {"ids": ids, "y": c["y"].to_numpy(dtype=float), "day": cx["day"].reindex(ids).to_numpy(), "ap": t["ap"].astype(str).reindex(ids).to_numpy(),
                   "dl": t["d_lobt"].reindex(ids).to_numpy(dtype=float)}
        assert not pd.isna(meta[s]["day"]).any()
        pred[(s, "corr")] = c["pred"].to_numpy(dtype=float)
    # ---- artmm ----
    K, Ct = am.build_table()
    for s in ("val", "fj"):
        c = C[s]
        p_new, delta, rows, kap = am.shift(s, c, K, Ct)
        m = meta[s]
        ch = delta != 0
        nv = pred[(s, "corr")].copy()
        nv[ch] = clip_window(nv[ch] + delta[ch], m["dl"][ch])
        pred[(s, "am")] = nv
        cam = c.copy()
        cam["p_art"] = p_new
        cam.to_parquet(ROOT / "cache" / f"{s}_{g.tag}am_s0.parquet", index=False)
        rep[f"artmm_{s}"] = effect(nv, pred[(s, "corr")], m["y"], m["day"], m["ap"]) | {"kappa": round(kap, 4), "cells": rows}
        log(f"artmm {s}: {rep[f'artmm_{s}']['dmse']} (P<0 {rep[f'artmm_{s}']['p_lt0']}), cells {[(r['ap'], r['car'], r['mon'], r['da']) for r in rows]}")
    # ---- arthist (src = artmm cache, corr = corrector cache, beta from V) ----
    for s in ("val", "fj"):
        C[s].to_parquet(ROOT / "cache" / f"{s}_{g.tag}cr_s0.parquet", index=False)
    T = ah.build_table()
    Bt = ah.betas(f"{g.tag}am", f"{g.tag}cr", T)
    rep["arthist_beta_all"] = round(float(Bt["all"]), 5)
    for s in ("val", "fj"):
        cache, p_new, delta, info = ah.shift(s, f"{g.tag}am", f"{g.tag}cr", T, Bt)
        m = meta[s]
        assert np.array_equal(cache["MVT_ID_mvt"].to_numpy(), m["ids"])
        ch = delta != 0
        nv = pred[(s, "am")].copy()
        nv[ch] = clip_window(nv[ch] + delta[ch], m["dl"][ch])
        pred[(s, "fh")] = nv
        rep[f"arthist_{s}"] = effect(nv, pred[(s, "am")], m["y"], m["day"], m["ap"]) | {"info": info}
        log(f"arthist {s}: {rep[f'arthist_{s}']['dmse']} (P<0 {rep[f'arthist_{s}']['p_lt0']}), beta {info['beta'] if s != 'val' else 'folds'}")
    # ---- aiu_shift (group block, lam 0.5, cmax 60) on the post-arthist predictions ----
    tot = []
    for rel in ai.AIU.values():
        a = pd.read_csv(ROOT / rel)
        a = a[a["APT_ICAO"].isin(ai.APS)]
        tot.append(pd.Series(60.0 * a["DLY_ALL_PRE_2"].to_numpy(), index=(a["APT_ICAO"] + "|" + a["FLT_DATE"]).to_numpy()))
    tot = pd.concat(tot)
    tt = cx["t"]
    D = {}
    for s in ("val", "fj"):
        c, m = C[s], meta[s]
        b = B[s].set_index("MVT_ID_mvt")["pred"]
        Ds = pd.DataFrame({"id": m["ids"], "fc": c["pred"].to_numpy(), "f": pred[(s, "fh")], "p": c["p_art"].to_numpy(), "mu": c["mu"].to_numpy(),
                           "g": c["g"].to_numpy().astype(bool), "d": c["dsch"].to_numpy(), "base": b.reindex(m["ids"]).to_numpy(), "y": m["y"]})
        Ds["ap"], Ds["day"], Ds["dl"] = m["ap"], m["day"], m["dl"]
        Ds["aw"] = False
        Ds["m"] = Ds["mu"] + (Ds["fc"] - Ds["base"])
        Ds = ai.add_classes(Ds, t["dd"])
        assert not Ds[["f", "base", "ap", "day"]].isna().any().any()
        D[s] = Ds
    def app(Dm):
        return Dm.assign(day=ai.block_days(Dm, tt, "mn"))
    def fitd(Dm):
        return Dm.assign(day=ai.block_days(Dm, tt, "y"))
    V, Fd = D["val"], D["fj"]
    jan = (V["day"].str[5:7] == "01").to_numpy()
    fV = V["f"].to_numpy().copy()
    for msk, fit in ((jan, ~jan), (~jan, jan)):  # the val recipe
        Dm = V[msk].reset_index(drop=True)
        fm, _ = ai.shift_days(app(Dm), ai.fit_g(V[fit]), ai.offsets(fitd(V[fit]), tot), tot, 0.5, 60.0)
        fV[msk] = fm
    rep["aiu_val"] = effect(fV, V["f"].to_numpy(), meta["val"]["y"], meta["val"]["day"], meta["val"]["ap"])
    off_v = ai.offsets(fitd(V), tot)
    fF, cs = ai.shift_days(app(Fd), ai.fit_g(V), off_v, tot, 0.5, 60.0)  # the final-file recipe: g and offsets from all of V
    rep["aiu_fj"] = effect(fF, Fd["f"].to_numpy(), meta["fj"]["y"], meta["fj"]["day"], meta["fj"]["ap"]) | {"offsets": {k: round(v, 1) for k, v in off_v.items()}}
    log(f"aiu val {rep['aiu_val']['dmse']} (P<0 {rep['aiu_val']['p_lt0']}), fj {rep['aiu_fj']['dmse']} (P<0 {rep['aiu_fj']['p_lt0']})")
    # report only: in-month on F (Feb from Jun, Jun from Feb) and offsets from the ten months' labels
    feb = (Fd["day"].str[5:7] == "02").to_numpy()
    fI = Fd["f"].to_numpy().copy()
    for msk, fit in ((feb, ~feb), (~feb, feb)):
        Dm = Fd[msk].reset_index(drop=True)
        fm, _ = ai.shift_days(app(Dm), ai.fit_g(Fd[fit]), ai.offsets(fitd(Fd[fit]), tot), tot, 0.5, 60.0)
        fI[msk] = fm
    rep["aiu_fj_inmonth_report"] = effect(fI, Fd["f"].to_numpy(), meta["fj"]["y"], meta["fj"]["day"], meta["fj"]["ap"])
    A = pq.read_table(ROOT / "cache" / "all.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "t_mvt", "t_sch", "y", "src"]).to_pandas()
    A = A[(A["PHASE_mvt"].astype(str) == "DEP") & A["src"].isin(FJ)]
    L = pd.DataFrame({"id": A["MVT_ID_mvt"].to_numpy(), "ap": A["ADEP_mvt"].astype(str).to_numpy(), "d": (A["t_mvt"] - A["t_sch"]).to_numpy(dtype=float),
                      "y": A["y"].to_numpy(dtype=float)})
    L["day"] = pd.to_datetime(A["t_mvt"].to_numpy(dtype=float) - L["y"].to_numpy(), unit="s").strftime("%Y-%m-%d")  # off-block date (group block fit)
    off10 = ai.offsets(L, tot)
    fO, _ = ai.shift_days(app(Fd), ai.fit_g(V), off10, tot, 0.5, 60.0)
    rep["aiu_fj_off10_report"] = effect(fO, Fd["f"].to_numpy(), meta["fj"]["y"], meta["fj"]["day"], meta["fj"]["ap"]) | {"offsets": {k: round(v, 1) for k, v in off10.items()}}
    log(f"aiu fj in-month {rep['aiu_fj_inmonth_report']['dmse']}, fj with ten-month offsets {rep['aiu_fj_off10_report']['dmse']}")
    rep["rmse"] = {s: {k: round(float(np.sqrt(((pred[(s, k)] - meta[s]["y"]) ** 2).mean())), 3) for k in ("corr", "am", "fh")} for s in ("val", "fj")}
    rep["rmse"]["val"]["db"] = round(float(np.sqrt(((fV - meta["val"]["y"]) ** 2).mean())), 3)
    rep["rmse"]["fj"]["db"] = round(float(np.sqrt(((fF - meta["fj"]["y"]) ** 2).mean())), 3)
    rep["seconds"] = round(time.time() - T0)
    (OUT / f"m3_tail_{g.tag}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log("saved")


if __name__ == "__main__":
    main()
