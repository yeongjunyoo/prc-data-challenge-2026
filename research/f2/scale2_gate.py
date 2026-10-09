# SPDX-License-Identifier: GPL-3.0-only
"""F2 1.7 (private notes research/2026-10-08-f2-four-months.md 1.7, registered 2026-10-09 09:58 KST before any number): scale2 0.75 against 0.5 for
Feb/Jun 2025 as unseen months, in gate 1.4's blend world.
scale2 multiplies the stage-2 output before the LOBT window, so no model changes. This script rebuilds gate14's and gate15's matrices with their
code and arguments (float64), loads every LightGBM model from gate14's store (--fit_cache, the same keys, so nothing is refitted), and windows the
same stage-2 predictions at 0.5 and 0.75. The models without the monthly AIU columns (gate15's fit_nm, not stored) are refitted only when 1.5's
Feb/Jun part passed, since only then does the final use them on Feb/Jun rows. Every 0.5-side prediction must equal the per-row prediction that
gate14 and gate15 saved, bit for bit, or it stops.
Decision on the three-chain average (per-row mean of the chains' windowed predictions), all F rows, day-block bootstrap (UTC day of MVT, 2,000
draws, seed 0): in the corrector form the final uses on Feb/Jun rows (L if gate 1.4 passed, T if it failed; L_nm or T_nm if 1.5's Feb/Jun part
also passed), use 0.75 on Feb/Jun rows iff form(0.75) - form(0.5) <= -20 with P(<0) >= 0.9, else 0.5. Jan/Jul rows take 0.75 (server, v36).
Information: every form at both scales, per month; Jan/Jul whole month held out (LV2, LV4) and in-month (I_V, U_V).
    python -u research/f2/scale2_gate.py --ctx f2ctx_v3 --ctx_sha <16> --threads 2 [--out research/f2/out] [--fit_cache research/f2/fitcache_g14]
Writes <out>/scale2.json and <out>/scale2_<chain>_{V,F}.parquet (s1 and the unscaled stage-2 prediction of every form per row, for any scale).
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
import src.ctx_corr2 as c2  # noqa: E402
from src.load import CACHE  # noqa: E402
from src.lobt_window import feats_d_lobt, mu_of, project  # noqa: E402
import m3_corr as m3  # noqa: E402
from gate15 import NM, fit_nm  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
T0 = time.time()
SCALES = {"s50": 0.5, "s75": 0.75}


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def windower(b: pd.DataFrame, feats: str):
    """m3_corr.windowed with the frame's window inputs read once: s -> project(s, ...)[0]."""
    _, mu = mu_of(b)
    p_art, g, dsch = b["p_art"].to_numpy(dtype=float), b["g"].to_numpy().astype(bool), b["dsch"].to_numpy(dtype=float)
    d = feats_d_lobt(feats, b["MVT_ID_mvt"].to_numpy())
    return lambda s: project(s, mu, p_art, g, dsch, d)[0]


def chain(name, vbase, fbase, a, A, ctx, keep, names, do_nm, out):
    V = pd.read_parquet(CACHE / f"val_{vbase}_s0.parquet")
    F = pd.read_parquet(CACHE / f"fj_{fbase}_s0.parquet")
    T = ac.design(V, A)
    T = T[np.isfinite(T["y"].to_numpy())]
    RV, RF = cc.Rows(V, A, a, ctx), cc.Rows(F, A, a, ctx)
    assert set(np.unique(RV.day)) == set(T["day"].unique())
    yV, yF = V["y"].to_numpy(dtype=float), F["y"].to_numpy(dtype=float)
    keepV, keepF = np.isfinite(yV), np.isfinite(yF)
    fo, fT, models = cc.stage1_folds(T, a)
    oof = np.zeros(len(T))
    for j, m in enumerate(models):
        oof[fT == j] = ac.predict(m, T[fT == j])
    T_pos = T["pos"].to_numpy()
    d1V = np.zeros(len(V))
    d1V[T_pos] = oof
    d1F = np.zeros(len(F))
    if len(RF.X):
        d1F[RF.X["pos"].to_numpy()] = np.mean([ac.predict(m, RF.X) for m in models], axis=0)
    s1V = np.clip(V["pred"].to_numpy(dtype=float) + d1V, 0, None)
    s1F = np.clip(F["pred"].to_numpy(dtype=float) + d1F, 0, None)
    MV, MF = RV.design2(s1V, d1V), RF.design2(s1F, d1F)
    appV, appF = s1V < a.max_pred, s1F < a.max_pred
    tV, tF = yV - s1V, yF - s1F
    fold_row = pd.Series(RV.day).map(fo).to_numpy()
    monF = pd.Series(RF.day).str[5:7].astype(int).to_numpy()
    monV = pd.Series(RV.day).str[5:7].astype(int).to_numpy()
    log(f"{name}: V {len(V):,} (matched {len(T):,}), F {len(F):,} (matched {len(RF.X):,}); nested stage 1 next")
    nested = cc.nested_stage1(T, fT, a)
    fitA = lambda M, t: cc.fit2(M, t, a.seed, a.rounds2, a.threads)  # noqa: E731  (gate14's fit and gate15's fitA: stored models)
    fitN = lambda M, t: fit_nm(M, t, names, a)  # noqa: E731  (gate15's fit: refitted)
    allF = keepF & appF
    trV = keepV & appV
    xV, xF = {}, {}
    sets = [("", fitA, slice(None))] + ([("_nm", fitN, keep)] if do_nm else [])
    for sfx, fit, cols in sets:
        MVc, MFc = MV[:, cols], MF[:, cols]
        xI, xU = np.zeros(len(V)), np.zeros(len(V))
        S = []
        for k in range(a.folds):
            dm = np.zeros(len(T))
            for j in range(a.folds):
                if j != k:
                    dm[fT == j] = nested[(j, k)]
            delta = RV.stage1(dm, T_pos)
            s1 = np.clip(V["pred"].to_numpy(dtype=float) + delta, 0, None)
            tr = (fold_row != k) & keepV & (s1 < a.max_pred)
            Mk = RV.design2(s1, delta)[:, cols]
            sel = (fold_row == k) & appV
            mI = fit(Mk[tr], (yV - s1)[tr])
            S.append(mI)
            xI[sel] = mI.predict(MVc[sel])
            mU = fit(np.vstack([Mk[tr], MFc[allF]]), np.concatenate([(yV - s1)[tr], tF[allF]]))
            xU[sel] = mU.predict(MVc[sel])
            del Mk
        xT = np.zeros(len(F))
        xT[appF] = np.mean([m.predict(MFc[appF]) for m in S], axis=0)
        m1 = fit(MVc[trV], tV[trV])
        x1 = np.zeros(len(F))
        x1[appF] = m1.predict(MFc[appF])
        xL = x1.copy()
        for hold, other in ((2, 6), (6, 2)):
            tro = allF & (monF == other)
            m = fit(np.vstack([MVc[trV], MFc[tro]]), np.concatenate([tV[trV], tF[tro]]))
            sel = appF & (monF == hold)
            xL[sel] = m.predict(MFc[sel])
        xLV2, xLV4 = np.zeros(len(V)), np.zeros(len(V))
        for hold, other in ((1, 7), (7, 1)):
            tro, sel = trV & (monV == other), appV & (monV == hold)
            xLV2[sel] = fit(MVc[tro], tV[tro]).predict(MVc[sel])
            xLV4[sel] = fit(np.vstack([MVc[tro], MFc[allF]]), np.concatenate([tV[tro], tF[allF]])).predict(MVc[sel])
        xV.update({f"I_V{sfx}": xI, f"U_V{sfx}": xU, f"LV2{sfx}": xLV2, f"LV4{sfx}": xLV4})
        xF.update({f"T{sfx}": xT, f"T1{sfx}": x1, f"L{sfx}": xL})
        log(f"{name}: stage-2 predictions{sfx or ' (all columns)'} done")
    wV, wF = windower(V, a.feats), windower(F, a.feats)
    sv = out / f"gate15_{name}_V.parquet" if (out / f"gate15_{name}_V.parquet").exists() else out / f"gate14_{name}_V.parquet"
    sf = out / f"gate15_{name}_F.parquet" if (out / f"gate15_{name}_F.parquet").exists() else out / f"gate14_{name}_F.parquet"
    gV, gF = pd.read_parquet(sv), pd.read_parquet(sf)
    assert np.array_equal(gV["MVT_ID_mvt"].to_numpy(), V["MVT_ID_mvt"].to_numpy()) and np.array_equal(gF["MVT_ID_mvt"].to_numpy(), F["MVT_ID_mvt"].to_numpy())
    PV = pd.DataFrame({"MVT_ID_mvt": V["MVT_ID_mvt"].to_numpy(), "y": yV, "day": RV.day, "s1": s1V, "s1w": wV(s1V)})
    PF = pd.DataFrame({"MVT_ID_mvt": F["MVT_ID_mvt"].to_numpy(), "y": yF, "day": RF.day, "s1": s1F, "s1w": wF(s1F)})
    checked, missing = [], []
    for P, g, xs, w, s1 in ((PV, gV, xV, wV, s1V), (PF, gF, xF, wF, s1F)):
        if not np.array_equal(P["s1w"].to_numpy(), g["s1w"].to_numpy()):
            raise SystemExit(f"{name}: stage-1 window differs from the gate files ({sv.name}, {sf.name})")
        for form, x in xs.items():
            P[f"x_{form}"] = x
            for tag, s in SCALES.items():
                P[f"{form}_{tag}"] = w(np.clip(s1 + s * x, 0, None))
            if form in g.columns:
                if not np.array_equal(P[f"{form}_s50"].to_numpy(), g[form].to_numpy()):
                    d = np.max(np.abs(P[f"{form}_s50"].to_numpy() - g[form].to_numpy()))
                    raise SystemExit(f"{name}: {form} at scale2 0.5 differs from the gate file by up to {d:.6f} s; the stored models were not reproduced")
                checked.append(form)
            else:
                missing.append(form)
    log(f"{name}: 0.5 side equal to the gate files bit for bit for {checked}; not in the gate files: {missing}")
    PV.to_parquet(out / f"scale2_{name}_V.parquet", index=False)
    PF.to_parquet(out / f"scale2_{name}_F.parquet", index=False)
    return PV, PF, checked, missing


def contrasts(PV, PF, formsV, formsF) -> dict:
    se = lambda df, k: ((df[k] - df["y"]) ** 2).to_numpy()  # noqa: E731
    r = {}
    for P, forms, months in ((PF, formsF, ("02", "06")), (PV, formsV, ("01", "07"))):
        day = P["day"].to_numpy()
        mon = P["day"].str[5:7].to_numpy()
        for f in forms:
            d = se(P, f"{f}_s75") - se(P, f"{f}_s50")
            r[f"{f}_s75_minus_s50"] = {"all": m3.boot(d, day), **{mm: m3.boot(d[mon == mm], day[mon == mm]) for mm in months}}
            for tag in SCALES:
                r[f"{f}_{tag}_vs_stage1"] = m3.boot(se(P, f"{f}_{tag}") - se(P, "s1w"), day)
    return r


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ctx", required=True)
    pa.add_argument("--ctx_sha", required=True)
    pa.add_argument("--threads", type=int, default=2)
    pa.add_argument("--out", default="research/f2/out")
    pa.add_argument("--fit_cache", default="research/f2/fitcache_g14")
    pa.add_argument("--nm", default="auto", choices=["auto", "yes", "no"], help="auto: the _nm forms only when gate15.json says pass_feb_jun")
    g = pa.parse_args()
    out = Path(g.out)
    g14 = json.loads((out / "gate14.json").read_text(encoding="utf-8"))
    if g14["ctx"] != g.ctx or g14["ctx_sha"] != g.ctx_sha:
        raise SystemExit(f"gate14 ran on {g14['ctx']} {g14['ctx_sha']}, not {g.ctx} {g.ctx_sha}")
    p15 = out / "gate15.json"
    g15 = json.loads(p15.read_text(encoding="utf-8")) if p15.exists() else None
    feb_jun_nm = bool(g15 and g15.get("pass_feb_jun"))
    do_nm = g.nm == "yes" or (g.nm == "auto" and feb_jun_nm)
    a = SimpleNamespace(adsb="adsb_v1nd", adsb_sha="0d03ff06267e0dee", geo="geo_v1nd", geo_sha="43310b9bbeefe81b", feats="feats_v25", folds=10,
                        rounds=300, rounds2=400, max_pred=3600.0, threads=g.threads, seed=0, scale2=0.5)
    ac.below_normal()
    m3.setup()
    if g.fit_cache:
        c2.fit_cache(Path(g.fit_cache))
    X2 = list(cc.X2)
    keep = [i for i, c in enumerate(X2) if c not in set(NM)]
    names = [X2[i] for i in keep]
    A = cc.stage1_inputs(a)
    ctx = cc.checked(g.ctx, g.ctx_sha).set_index("MVT_ID_mvt")
    formsV = ["I_V", "U_V", "LV2", "LV4"] + (["I_V_nm", "U_V_nm", "LV2_nm", "LV4_nm"] if do_nm else [])
    formsF = ["T", "T1", "L"] + (["T_nm", "T1_nm", "L_nm"] if do_nm else [])
    rep = {"registered": "private note 1.7, 2026-10-09 09:58 KST", "ctx": g.ctx, "ctx_sha": g.ctx_sha, "gate14_pass": bool(g14["pass"]),
           "gate15_present": g15 is not None, "gate15_pass_feb_jun": feb_jun_nm, "nm_forms": do_nm, "chains": {}}
    PVs, PFs = [], []
    for name, spec in g14["chains"].items():
        PV, PF, checked, missing = chain(name, spec["vbase"], spec["fbase"], a, A, ctx, keep, names, do_nm, out)
        rep["chains"][name] = {"vbase": spec["vbase"], "fbase": spec["fbase"], "equal_to_gate_files": checked, "not_in_gate_files": missing,
                               **contrasts(PV, PF, formsV, formsF)}
        PVs.append(PV.set_index("MVT_ID_mvt"))
        PFs.append(PF.set_index("MVT_ID_mvt"))
    PV, PF = PVs[0][["y", "day"]].copy(), PFs[0][["y", "day"]].copy()
    for P, Ps, forms in ((PV, PVs, formsV), (PF, PFs, formsF)):
        P["s1w"] = np.mean([p["s1w"].reindex(P.index).to_numpy() for p in Ps], axis=0)
        for f in forms:
            for tag in SCALES:
                P[f"{f}_{tag}"] = np.mean([p[f"{f}_{tag}"].reindex(P.index).to_numpy() for p in Ps], axis=0)
    avg = contrasts(PV.reset_index(), PF.reset_index(), formsV, formsF)
    rep["three_chain_average"] = avg
    form = ("L" if rep["gate14_pass"] else "T") + ("_nm" if feb_jun_nm else "")
    if form not in formsF:
        raise SystemExit(f"form {form} was not computed")
    d = avg[f"{form}_s75_minus_s50"]["all"]
    use75 = bool(d["dmse"] <= -20 and d["p_lt0"] >= 0.9)
    rep["decision"] = {"form": form, "delta_s75_minus_s50": d, "feb_jun_scale2": 0.75 if use75 else 0.5, "jan_jul_scale2": 0.75,
                       "rule": "use 0.75 on Feb/Jun rows iff form(0.75) - form(0.5) <= -20 with P(<0) >= 0.9 (note 1.7)",
                       "note": None if g15 is not None else "gate15.json missing: the form ignores 1.5's Feb/Jun part"}
    rep["seconds"] = round(time.time() - T0)
    (out / "scale2.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"SCALE2 Feb/Jun rows: {rep['decision']['feb_jun_scale2']} ({form}(0.75) - {form}(0.5) {d}); T {avg['T_s75_minus_s50']['all']}, "
        f"L {avg['L_s75_minus_s50']['all']}, LV2 {avg['LV2_s75_minus_s50']['all']}, LV4 {avg['LV4_s75_minus_s50']['all']}, I_V {avg['I_V_s75_minus_s50']['all']}")


if __name__ == "__main__":
    main()
