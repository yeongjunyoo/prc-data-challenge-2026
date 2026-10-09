# SPDX-License-Identifier: GPL-3.0-only
"""F2 gate 1.4 (vault research/2026-10-08-f2-four-months.md 1.4, registered 2026-10-08 22:48 KST before any blend-world number):
does a corrector whose stage 2 also learns Feb/Jun 2025 help months it never saw, in 514's three-chain blend world?

Per chain c (A, B, C): V = cache/val_<vbase_c>_s0.parquet (514's seed-averaged val base, Jan + Jul 2025), F = cache/fj_<fbase_c>_s0.parquet
(leave-Feb/Jun-out seed-0 blend of the same families, blend_fixed full weights; A without TabM). Corrector = src/ctx_corr3.py's code with 514's
arguments (adsb_v1nd, geo_v1nd, the context table holding Jan/Feb/Jun/Jul 2025 rows, folds 10, rounds 300/400, max_pred 3600, scale2 0.5,
LOBT window last). Variants (all with stage 1 fitted on V as in val/full):
  I_V : val recipe (S_k on the other day folds, pairwise-nested stage-1 inputs), on V
  U_V : I_V with every F row added to each S_k's training rows, on V
  T   : the ten S_k averaged, on F (= the full recipe; reference)
  T1  : one stage-2 model on all V rows (stage-1 out-of-fold inputs), on F
  L   : T1's recipe plus the F rows of the OTHER F month, applied to the held-out F month (Feb from Jan/Jul + Jun, Jun from Jan/Jul + Feb)
Decision on the three-chain average (per-row mean of the chains' windowed predictions, as tools/chainavg3.py with 1/3 each):
  pass  iff  mean over F rows of (L - T1) <= -50 with day-bootstrap P(<0) >= 0.9   and   mean over V rows of (U_V - I_V) <= +25.
Per-chain numbers are reported only. Day-block bootstrap: UTC day of MVT, 2,000 draws, seed 0.
    python -u research/f2/gate14.py --ctx f2ctx_v3 --ctx_sha <16> --threads 2
      [--chains A:m6r27wincbrmfw4sv434tmc:f2wincbrmfw4,B:m6r27skcbrmfw4sv434:f2skcbrmfw4,C:m6r27apcbrmfw4sv114:f2apcbrmfw4]
Writes research/f2/out/gate14.json (and gate14_<chain>.parquet with the per-row windowed predictions of each variant).
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
OUT = Path("research/f2/out")
DEFAULT = "A:m6r27wincbrmfw4sv434tmc:f2wincbrmfw4,B:m6r27skcbrmfw4sv434:f2skcbrmfw4,C:m6r27apcbrmfw4sv114:f2apcbrmfw4"


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def chain(name, vbase, fbase, a, A, ctx):
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
    MV, MF = RV.design2(s1V, d1V), RF.design2(s1F, d1F)  # float64 as src/ctx_corr.py (a float32 cast changes LightGBM's bins)
    appV, appF = s1V < a.max_pred, s1F < a.max_pred
    tV, tF = yV - s1V, yF - s1F
    fold_row = pd.Series(RV.day).map(fo).to_numpy()
    monF = pd.Series(RF.day).str[5:7].astype(int).to_numpy()
    log(f"{name}: V {len(V):,} (matched {len(T):,}), F {len(F):,} (matched {len(RF.X):,}); nested stage 1 next")
    nested = cc.nested_stage1(T, fT, a)
    fit = lambda M, t: cc.fit2(M, t, a.seed, a.rounds2, a.threads)  # noqa: E731
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
        Mk = RV.design2(s1, delta)
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
    w = lambda b, s1, x: m3.windowed(b, np.clip(s1 + a.scale2 * x, 0, None), a.feats)  # noqa: E731
    PV = pd.DataFrame({"MVT_ID_mvt": V["MVT_ID_mvt"].to_numpy(), "y": yV, "day": RV.day, "s1w": m3.windowed(V, s1V, a.feats),
                       "I_V": w(V, s1V, xI), "U_V": w(V, s1V, xU)})
    PF = pd.DataFrame({"MVT_ID_mvt": F["MVT_ID_mvt"].to_numpy(), "y": yF, "day": RF.day, "s1w": m3.windowed(F, s1F, a.feats),
                       "T": w(F, s1F, xT), "T1": w(F, s1F, x1), "L": w(F, s1F, xL)})
    PV.to_parquet(OUT / f"gate14_{name}_V.parquet", index=False)
    PF.to_parquet(OUT / f"gate14_{name}_F.parquet", index=False)
    return PV, PF


def contrasts(PV, PF) -> dict:
    se = lambda p, y: (p - y) ** 2  # noqa: E731
    r = {}
    for k in ("I_V", "U_V"):
        r[f"{k}_vs_stage1"] = m3.boot(se(PV[k], PV["y"]).to_numpy() - se(PV["s1w"], PV["y"]).to_numpy(), PV["day"].to_numpy())
    for k in ("T", "T1", "L"):
        r[f"{k}_vs_stage1"] = m3.boot(se(PF[k], PF["y"]).to_numpy() - se(PF["s1w"], PF["y"]).to_numpy(), PF["day"].to_numpy())
    r["L_minus_T1"] = m3.boot(se(PF["L"], PF["y"]).to_numpy() - se(PF["T1"], PF["y"]).to_numpy(), PF["day"].to_numpy())
    mon = PF["day"].str[5:7].to_numpy()
    for mm in ("02", "06"):
        s = mon == mm
        r[f"L_minus_T1_{mm}"] = m3.boot((se(PF["L"], PF["y"]) - se(PF["T1"], PF["y"])).to_numpy()[s], PF["day"].to_numpy()[s])
    r["U_V_minus_I_V"] = m3.boot(se(PV["U_V"], PV["y"]).to_numpy() - se(PV["I_V"], PV["y"]).to_numpy(), PV["day"].to_numpy())
    return r


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ctx", required=True)
    pa.add_argument("--ctx_sha", required=True)
    pa.add_argument("--chains", default=DEFAULT)
    pa.add_argument("--threads", type=int, default=2)
    pa.add_argument("--out", default="research/f2/out", help="output folder (a smoke run uses another one)")
    pa.add_argument("--fit_cache", default="research/f2/fitcache_g14", help="model store (src/ctx_corr2.fit_cache) that gate15.py reuses for stage 1")
    g = pa.parse_args()
    global OUT
    OUT = Path(g.out)
    a = SimpleNamespace(adsb="adsb_v1nd", adsb_sha="0d03ff06267e0dee", geo="geo_v1nd", geo_sha="43310b9bbeefe81b", feats="feats_v25", folds=10,
                        rounds=300, rounds2=400, max_pred=3600.0, threads=g.threads, seed=0, scale2=0.5)
    ac.below_normal()
    m3.setup()
    if g.fit_cache:
        import src.ctx_corr2 as c2
        c2.fit_cache(Path(g.fit_cache))
    OUT.mkdir(parents=True, exist_ok=True)
    A = cc.stage1_inputs(a)
    ctx = cc.checked(g.ctx, g.ctx_sha).set_index("MVT_ID_mvt")
    rep = {"registered": "vault note 1.4, 2026-10-08 22:48 KST", "ctx": g.ctx, "ctx_sha": g.ctx_sha, "chains": {}}
    PVs, PFs = [], []
    for spec in g.chains.split(","):
        name, vb, fb = spec.split(":")
        PV, PF = chain(name, vb, fb, a, A, ctx)
        rep["chains"][name] = {"vbase": vb, "fbase": fb, **contrasts(PV, PF)}
        log(f"{name}: L-T1 {rep['chains'][name]['L_minus_T1']}, U_V-I_V {rep['chains'][name]['U_V_minus_I_V']}")
        PVs.append(PV.set_index("MVT_ID_mvt"))
        PFs.append(PF.set_index("MVT_ID_mvt"))
    PV, PF = PVs[0].copy(), PFs[0].copy()
    for k in ("s1w", "I_V", "U_V"):
        PV[k] = np.mean([p[k].reindex(PV.index).to_numpy() for p in PVs], axis=0)
    for k in ("s1w", "T", "T1", "L"):
        PF[k] = np.mean([p[k].reindex(PF.index).to_numpy() for p in PFs], axis=0)
    avg = contrasts(PV.reset_index(), PF.reset_index())
    rep["three_chain_average"] = avg
    lt, uv = avg["L_minus_T1"], avg["U_V_minus_I_V"]
    rep["pass"] = bool(lt["dmse"] <= -50 and lt["p_lt0"] >= 0.9 and uv["dmse"] <= 25)
    rep["seconds"] = round(time.time() - T0)
    (OUT / "gate14.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"GATE 1.4 {'PASS' if rep['pass'] else 'FAIL'}: L-T1 {lt}, U_V-I_V {uv}")


if __name__ == "__main__":
    main()
