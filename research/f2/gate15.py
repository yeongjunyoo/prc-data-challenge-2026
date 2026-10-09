# SPDX-License-Identifier: GPL-3.0-only
"""F2 gate 1.5 (private notes research/2026-10-08-f2-four-months.md 1.5, registered 2026-10-08 23:23 KST before any blend-world or server number of
this variant): the A3 corrector's stage 2 without the four monthly AIU columns (aiu_m_txo, aiu_m_add, aiu_m_valid, aiu_m_gap), measured in the
form that would go into the final file, in gate 1.4's blend world and with gate 1.4's code.

Runs after research/f2/gate14.py (reads research/f2/out/gate14.json and gate14_<chain>_{V,F}.parquet; 1.4's numbers are not recomputed).
Stage 1 is refitted with gate14's code and arguments through the same model store (--fit_cache, research/f2/fitcache_g14 by default), so it is
gate14's stage 1; the per-row stage-1 window (s1w) is compared with gate14's files and must be identical. Stage 2 without the four columns
("nm") is fitted on the same rows and seeds with LightGBM's parameters from src/ctx_corr.fit2 (feature subset in X2 order, as --drop2 does):
  I_V_nm : val recipe on V          U_V_nm : val recipe + every F row, on V
  T_nm   : the ten S_k_nm averaged, on F (the final-file form)          T1_nm : one model on all V rows, on F
  L_nm   : T1_nm's recipe + the F rows of the other F month, on the held-out F month
Information only (added 2026-10-08 23:54 KST, `date`, before any blend-world number, not part of the decision): LV2, LV4 (+ _nm) = Jan/Jul 2025 with the
evaluated month held out whole (stage 2 on the other Jan/Jul month, and on it plus every F row), the leak-free local view of Jan/Jul.
Decision on the three-chain average (per-row mean of the chains' windowed predictions, as tools/chainavg3.py with 1/3 each), day-block bootstrap
(UTC day of MVT, 2,000 draws, seed 0). Feb/Jun part, computed here:
  gate 1.4 passed (four-month corrector + nm):  mean over F of (L_nm - L) <= -20 with P(<0) >= 0.9
  gate 1.4 failed (514's two-month corrector + nm):  mean over F of (T_nm - T) <= -20 with P(<0) >= 0.9
Jan/Jul part, in both branches: the server read of 560 (514 with the same drop) against v35 (514), adopt only if its server delta MSE <= 0
(amended 2026-10-08 23:55 KST, `date`, before any 1.5 number existed, coordinator's decision: U_V_nm - I_V is an in-month comparison where
the monthly columns act as an airport-month label the 2026 rows do not have, so it is reported only, as are LV2/LV4). Gate 1.4's own
U_V - I_V <= +25 still applies to the four-month part. gate15.json carries pass_feb_jun; the final decision adds the server sign.
    python -u research/f2/gate15.py --ctx f2ctx_v3 --ctx_sha <16> --threads 2
Writes research/f2/out/gate15.json and gate15_<chain>_{V,F}.parquet.
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
import m3_corr as m3  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
T0 = time.time()
OUT = Path("research/f2/out")
NM = ["aiu_m_txo", "aiu_m_add", "aiu_m_valid", "aiu_m_gap"]


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def fit_nm(M, t, names, a):
    import lightgbm as lgb
    params = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=a.threads, seed=a.seed, deterministic=True, force_col_wise=True)
    ds = lgb.Dataset(M, np.clip(t, -cc.CLIP, cc.CLIP), feature_name=names, categorical_feature=[c for c in cc.CAT2 if c in names], free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=a.rounds2)


def chain(name, vbase, fbase, a, A, ctx, keep, names):
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
    MV_all, MF_all = RV.design2(s1V, d1V), RF.design2(s1F, d1F)  # float64 as src/ctx_corr.py (a float32 cast changes LightGBM's bins)
    MV, MF = MV_all[:, keep], MF_all[:, keep]
    appV, appF = s1V < a.max_pred, s1F < a.max_pred
    tV, tF = yV - s1V, yF - s1F
    fold_row = pd.Series(RV.day).map(fo).to_numpy()
    monF = pd.Series(RF.day).str[5:7].astype(int).to_numpy()
    log(f"{name}: V {len(V):,}, F {len(F):,}; nested stage 1 next")
    nested = cc.nested_stage1(T, fT, a)
    fit = lambda M, t: fit_nm(M, t, names, a)  # noqa: E731
    allF = keepF & appF
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
        Mk = RV.design2(s1, delta)[:, keep]
        sel = (fold_row == k) & appV
        mI = fit(Mk[tr], (yV - s1)[tr])
        S.append(mI)
        xI[sel] = mI.predict(MV[sel])
        mU = fit(np.vstack([Mk[tr], MF[allF]]), np.concatenate([(yV - s1)[tr], tF[allF]]))
        xU[sel] = mU.predict(MV[sel])
        del Mk
        log(f"{name}: fold {k} done")
    xT = np.zeros(len(F))
    xT[appF] = np.mean([m.predict(MF[appF]) for m in S], axis=0)
    trV = keepV & appV
    m1 = fit(MV[trV], tV[trV])
    x1 = np.zeros(len(F))
    x1[appF] = m1.predict(MF[appF])
    xL = x1.copy()
    for hold, other in ((2, 6), (6, 2)):
        tro = allF & (monF == other)
        m = fit(np.vstack([MV[trV], MF[tro]]), np.concatenate([tV[trV], tF[tro]]))
        sel = appF & (monF == hold)
        xL[sel] = m.predict(MF[sel])
    # information only (not in 1.5's decision): Jan/Jul with the evaluated month held out whole, as the 2026 months are for the final file.
    # LV2: stage 2 on the other Jan/Jul 2025 month; LV4: the other month + every F row; _nm without the four monthly columns.
    monV = pd.Series(RV.day).str[5:7].astype(int).to_numpy()
    fitA = lambda M, t: cc.fit2(M, t, a.seed, a.rounds2, a.threads)  # noqa: E731
    xLV2, xLV2n, xLV4, xLV4n = (np.zeros(len(V)) for _ in range(4))
    for hold, other in ((1, 7), (7, 1)):
        tro, sel = trV & (monV == other), appV & (monV == hold)
        xLV2[sel] = fitA(MV_all[tro], tV[tro]).predict(MV_all[sel])
        xLV2n[sel] = fit(MV[tro], tV[tro]).predict(MV[sel])
        xLV4[sel] = fitA(np.vstack([MV_all[tro], MF_all[allF]]), np.concatenate([tV[tro], tF[allF]])).predict(MV_all[sel])
        xLV4n[sel] = fit(np.vstack([MV[tro], MF[allF]]), np.concatenate([tV[tro], tF[allF]])).predict(MV[sel])
    log(f"{name}: whole-month Jan/Jul fits done")
    w = lambda b, s1, x: m3.windowed(b, np.clip(s1 + a.scale2 * x, 0, None), a.feats)  # noqa: E731
    PV = pd.DataFrame({"MVT_ID_mvt": V["MVT_ID_mvt"].to_numpy(), "s1w_nm": m3.windowed(V, s1V, a.feats), "I_V_nm": w(V, s1V, xI), "U_V_nm": w(V, s1V, xU),
                       "LV2": w(V, s1V, xLV2), "LV2_nm": w(V, s1V, xLV2n), "LV4": w(V, s1V, xLV4), "LV4_nm": w(V, s1V, xLV4n)})
    PF = pd.DataFrame({"MVT_ID_mvt": F["MVT_ID_mvt"].to_numpy(), "s1w_nm": m3.windowed(F, s1F, a.feats), "T_nm": w(F, s1F, xT), "T1_nm": w(F, s1F, x1),
                       "L_nm": w(F, s1F, xL)})
    gV = pd.read_parquet(OUT / f"gate14_{name}_V.parquet")
    gF = pd.read_parquet(OUT / f"gate14_{name}_F.parquet")
    assert np.array_equal(gV["MVT_ID_mvt"].to_numpy(), PV["MVT_ID_mvt"].to_numpy()) and np.array_equal(gF["MVT_ID_mvt"].to_numpy(), PF["MVT_ID_mvt"].to_numpy())
    same1 = bool(np.array_equal(gV["s1w"].to_numpy(), PV["s1w_nm"].to_numpy()) and np.array_equal(gF["s1w"].to_numpy(), PF["s1w_nm"].to_numpy()))
    if not same1:
        raise SystemExit(f"{name}: stage 1 differs from gate14's (max |d| V {np.max(np.abs(gV['s1w'] - PV['s1w_nm'])):.6f}, "
                         f"F {np.max(np.abs(gF['s1w'] - PF['s1w_nm'])):.6f})")
    PV = pd.concat([gV, PV.drop(columns=["MVT_ID_mvt", "s1w_nm"])], axis=1)
    PF = pd.concat([gF, PF.drop(columns=["MVT_ID_mvt", "s1w_nm"])], axis=1)
    PV.to_parquet(OUT / f"gate15_{name}_V.parquet", index=False)
    PF.to_parquet(OUT / f"gate15_{name}_F.parquet", index=False)
    return PV, PF


def contrasts(PV, PF) -> dict:
    se = lambda p, y: ((p - y) ** 2).to_numpy()  # noqa: E731
    r = {}
    for new, old in (("I_V_nm", "I_V"), ("U_V_nm", "U_V"), ("U_V_nm", "I_V"), ("LV2_nm", "LV2"), ("LV4_nm", "LV4"), ("LV4", "LV2"), ("LV4_nm", "LV2")):
        r[f"{new}_minus_{old}"] = m3.boot(se(PV[new], PV["y"]) - se(PV[old], PV["y"]), PV["day"].to_numpy())
    monv = PV["day"].str[5:7].to_numpy()
    for mm in ("01", "07"):
        s = monv == mm
        for new, old in (("LV2_nm", "LV2"), ("LV4_nm", "LV4"), ("LV4", "LV2")):
            r[f"{new}_minus_{old}_{mm}"] = m3.boot((se(PV[new], PV["y"]) - se(PV[old], PV["y"]))[s], PV["day"].to_numpy()[s])
    for new, old in (("T_nm", "T"), ("T1_nm", "T1"), ("L_nm", "L"), ("L_nm", "T")):
        r[f"{new}_minus_{old}"] = m3.boot(se(PF[new], PF["y"]) - se(PF[old], PF["y"]), PF["day"].to_numpy())
    mon = PF["day"].str[5:7].to_numpy()
    for mm in ("02", "06"):
        s = mon == mm
        for new, old in (("T_nm", "T"), ("L_nm", "L")):
            r[f"{new}_minus_{old}_{mm}"] = m3.boot((se(PF[new], PF["y"]) - se(PF[old], PF["y"]))[s], PF["day"].to_numpy()[s])
    return r


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ctx", required=True)
    pa.add_argument("--ctx_sha", required=True)
    pa.add_argument("--threads", type=int, default=2)
    pa.add_argument("--out", default="research/f2/out", help="output folder (a smoke run uses another one)")
    pa.add_argument("--fit_cache", default="research/f2/fitcache_g14", help="gate14's model store ('' = none)")
    g = pa.parse_args()
    global OUT
    OUT = Path(g.out)
    g14 = json.loads((OUT / "gate14.json").read_text(encoding="utf-8"))
    if g14["ctx"] != g.ctx or g14["ctx_sha"] != g.ctx_sha:
        raise SystemExit(f"gate14 ran on {g14['ctx']} {g14['ctx_sha']}, not {g.ctx} {g.ctx_sha}")
    a = SimpleNamespace(adsb="adsb_v1nd", adsb_sha="0d03ff06267e0dee", geo="geo_v1nd", geo_sha="43310b9bbeefe81b", feats="feats_v25", folds=10,
                        rounds=300, rounds2=400, max_pred=3600.0, threads=g.threads, seed=0, scale2=0.5)
    ac.below_normal()
    m3.setup()
    if g.fit_cache:
        c2.fit_cache(Path(g.fit_cache))
    X2 = list(cc.X2)
    missing = set(NM) - set(X2)
    if missing:
        raise SystemExit(f"not stage-2 columns: {sorted(missing)}")
    keep = [i for i, c in enumerate(X2) if c not in set(NM)]
    names = [X2[i] for i in keep]
    OUT.mkdir(parents=True, exist_ok=True)
    A = cc.stage1_inputs(a)
    ctx = cc.checked(g.ctx, g.ctx_sha).set_index("MVT_ID_mvt")
    rep = {"registered": "private note 1.5, 2026-10-08 23:23 KST", "ctx": g.ctx, "ctx_sha": g.ctx_sha, "dropped": NM, "stage2_columns": len(names),
           "gate14_pass": bool(g14["pass"]), "chains": {}}
    PVs, PFs = [], []
    for name, spec in g14["chains"].items():
        PV, PF = chain(name, spec["vbase"], spec["fbase"], a, A, ctx, keep, names)
        rep["chains"][name] = {"vbase": spec["vbase"], "fbase": spec["fbase"], **contrasts(PV, PF)}
        log(f"{name}: T_nm-T {rep['chains'][name]['T_nm_minus_T']}, L_nm-L {rep['chains'][name]['L_nm_minus_L']}, "
            f"U_V_nm-I_V {rep['chains'][name]['U_V_nm_minus_I_V']}")
        PVs.append(PV.set_index("MVT_ID_mvt"))
        PFs.append(PF.set_index("MVT_ID_mvt"))
    PV, PF = PVs[0].copy(), PFs[0].copy()
    for k in ("s1w", "I_V", "U_V", "I_V_nm", "U_V_nm", "LV2", "LV2_nm", "LV4", "LV4_nm"):
        PV[k] = np.mean([p[k].reindex(PV.index).to_numpy() for p in PVs], axis=0)
    for k in ("s1w", "T", "T1", "L", "T_nm", "T1_nm", "L_nm"):
        PF[k] = np.mean([p[k].reindex(PF.index).to_numpy() for p in PFs], axis=0)
    avg = contrasts(PV.reset_index(), PF.reset_index())
    rep["three_chain_average"] = avg
    if rep["gate14_pass"]:
        rep["form"] = "four-month corrector without the monthly AIU columns"
        fj = avg["L_nm_minus_L"]
    else:
        rep["form"] = "514's two-month corrector without the monthly AIU columns"
        fj = avg["T_nm_minus_T"]
    rep["pass_feb_jun"] = bool(fj["dmse"] <= -20 and fj["p_lt0"] >= 0.9)
    rep["jan_jul_rule"] = "server read of 560 against v35 (514): adopt only if its server delta MSE <= 0 (amendment 2026-10-08 23:55 KST)"
    rep["info_only"] = ["U_V_nm_minus_I_V", "I_V_nm_minus_I_V", "U_V_nm_minus_U_V", "LV2_nm_minus_LV2", "LV4_nm_minus_LV4", "LV4_minus_LV2", "LV4_nm_minus_LV2"]
    rep["seconds"] = round(time.time() - T0)
    (OUT / "gate15.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"GATE 1.5 Feb/Jun part {'PASS' if rep['pass_feb_jun'] else 'FAIL'} ({rep['form']}; Jan/Jul part = 560's server sign): T_nm-T {avg['T_nm_minus_T']}, L_nm-L {avg['L_nm_minus_L']}, "
        f"U_V_nm-I_V {avg['U_V_nm_minus_I_V']}")


if __name__ == "__main__":
    main()
