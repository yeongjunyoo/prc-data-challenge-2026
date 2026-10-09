# SPDX-License-Identifier: GPL-3.0-only
"""F2 1.6 (vault research/2026-10-08-f2-four-months.md 1.6, registered 2026-10-09 09:05 KST before any number): does the stage-2 corrector keep
helping unseen months (Feb/Jun 2025) as its training months grow past four? LightGBM-only world of 4.3, corrector code = src/ctx_corr3.py's
(imported, float64 design matrices as src/ctx_corr.py, 514's arguments: adsb_v1nd, geo_v1nd, folds 10, rounds 300/400, max_pred 3600, scale2 0.5,
LOBT window last), context table with every month used (--ctx).
Bases: V = cache/val_m6r27skwing_s0 (Jan/Jul), F = cache/fj_f2skwing_s0 (Feb/Jun), added months from leave-pair-out legs
cache/lmo<MMMM>_f2skwing_s0 (research/f2/model3_lmo.py + g_band_lmo.py, the m6r27skwin / skwing commands with the setting changed).
Stage 1: ten day-fold models on V's matched rows (V rows get their out-of-fold value; other rows the mean of the ten, 0 when unmatched).
Stage 2 per step S (T1 form): one model on every training row of S (V rows with out-of-fold stage-1 inputs; added rows), applied to F.
Steps: S2 = {1, 7}; S4 = S2 + {3, 5}; S6 = S4 + {12, 8}; S8 = S6 + {4, 10}. Delta MSE over all F rows (windowed), day-block bootstrap
(UTC day of MVT, 2,000 draws, seed 0): D4 = S4 - S2, D6 = S6 - S4, D8 = S8 - S6, S4/S6/S8 - S2, per month too.
Reading rule (registered): past four months is worth it iff D6 <= -50 with P(D6 < 0) >= 0.9; then eight months if also D8 <= -50 with
P >= 0.9, else six; otherwise stop at four.
Information only: Jan/Jul with the evaluated month held out whole (model on S minus that month, applied to it).
    python -u research/f2/m3_curve.py --ctx f2ctx8_v3 --ctx_sha <16> --threads 2
Writes research/f2/out/m3_curve.json and m3_curve_rows.parquet.
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
ADDED = [("lmo0305", (3, 5)), ("lmo0812", (12, 8)), ("lmo0410", (4, 10))]
STEPS = {"S2": [], "S4": ["lmo0305"], "S6": ["lmo0305", "lmo0812"], "S8": ["lmo0305", "lmo0812", "lmo0410"]}


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


class Part:
    """One base file's rows: stage-1 value, stage-1-only prediction, stage-2 matrix, target, masks."""

    def __init__(self, B: pd.DataFrame, A, a, ctx, models, oof_rows=None):
        self.B = B
        self.R = cc.Rows(B, A, a, ctx)
        self.y = B["y"].to_numpy(dtype=float)
        d1 = np.zeros(len(B))
        if oof_rows is not None:
            d1 = oof_rows
        elif len(self.R.X):
            d1[self.R.X["pos"].to_numpy()] = np.mean([ac.predict(m, self.R.X) for m in models], axis=0)
        self.d1 = d1
        self.s1 = np.clip(B["pred"].to_numpy(dtype=float) + d1, 0, None)
        self.M = self.R.design2(self.s1, self.d1)
        self.app = self.s1 < a.max_pred
        self.keep = np.isfinite(self.y)
        self.t = self.y - self.s1
        self.mon = pd.Series(self.R.day).str[5:7].astype(int).to_numpy()
        self.day = self.R.day


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--ctx", required=True)
    pa.add_argument("--ctx_sha", required=True)
    pa.add_argument("--threads", type=int, default=2)
    g = pa.parse_args()
    a = SimpleNamespace(adsb="adsb_v1nd", adsb_sha="0d03ff06267e0dee", geo="geo_v1nd", geo_sha="43310b9bbeefe81b", feats="feats_v25", folds=10,
                        rounds=300, rounds2=400, max_pred=3600.0, threads=g.threads, seed=0, scale2=0.5)
    ac.below_normal()
    m3.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    A = cc.stage1_inputs(a)
    ctx = cc.checked(g.ctx, g.ctx_sha).set_index("MVT_ID_mvt")
    V = pd.read_parquet(CACHE / "val_m6r27skwing_s0.parquet")
    T = ac.design(V, A)
    T = T[np.isfinite(T["y"].to_numpy())]
    fo, fT, models = cc.stage1_folds(T, a)
    oof = np.zeros(len(T))
    for j, m in enumerate(models):
        oof[fT == j] = ac.predict(m, T[fT == j])
    d1V = np.zeros(len(V))
    d1V[T["pos"].to_numpy()] = oof
    PV = Part(V, A, a, ctx, models, oof_rows=d1V)
    assert set(np.unique(PV.R.day)) == set(T["day"].unique())
    PF = Part(pd.read_parquet(CACHE / "fj_f2skwing_s0.parquet"), A, a, ctx, models)
    PA = {}
    for name, months in ADDED:
        B = pd.read_parquet(CACHE / f"{name}_f2skwing_s0.parquet")
        PA[name] = Part(B, A, a, ctx, models)
        # the leg's rows are its two training files (src); a few rows sit on the next or previous UTC day of MVT across a month boundary
        mc = pd.Series(PA[name].mon).value_counts()
        share = float(mc.reindex(list(months)).fillna(0).sum() / len(PA[name].mon))
        assert share >= 0.99, f"{name}: only {share:.4f} of the rows have a UTC MVT day in {months}: {mc.to_dict()}"
        log(f"{name}: {len(B):,} rows, UTC MVT months {mc.to_dict()}, matched {len(PA[name].R.X):,}")
    log(f"V {len(V):,} (matched {len(T):,}), F {len(PF.B):,} (matched {len(PF.R.X):,})")
    fit = lambda M, t: cc.fit2(M, t, a.seed, a.rounds2, a.threads)  # noqa: E731
    w = lambda P, x: m3.windowed(P.B, np.clip(P.s1 + a.scale2 * x, 0, None), a.feats)  # noqa: E731
    rowsF = pd.DataFrame({"MVT_ID_mvt": PF.B["MVT_ID_mvt"].to_numpy(), "y": PF.y, "day": PF.day, "s1w": m3.windowed(PF.B, PF.s1, a.feats)})
    rowsV = pd.DataFrame({"MVT_ID_mvt": V["MVT_ID_mvt"].to_numpy(), "y": PV.y, "day": PV.day, "s1w": m3.windowed(V, PV.s1, a.feats)})
    trV = PV.keep & PV.app
    for step, names in STEPS.items():
        Ms = [PV.M[trV]] + [PA[n].M[PA[n].keep & PA[n].app] for n in names]
        ts = [PV.t[trV]] + [PA[n].t[PA[n].keep & PA[n].app] for n in names]
        m = fit(np.vstack(Ms), np.concatenate(ts))
        x = np.zeros(len(PF.B))
        x[PF.app] = m.predict(PF.M[PF.app])
        rowsF[step] = w(PF, x)
        xj = np.zeros(len(V))
        for hold in (1, 7):
            tro = trV & (PV.mon != hold)
            mj = fit(np.vstack([PV.M[tro]] + Ms[1:]), np.concatenate([PV.t[tro]] + ts[1:]))
            sel = PV.app & (PV.mon == hold)
            xj[sel] = mj.predict(PV.M[sel])
        rowsV[step] = w(PV, xj)
        log(f"{step}: {sum(len(t) for t in ts):,} training rows; F stage-2 dMSE vs stage 1 {np.mean((rowsF[step] - PF.y) ** 2) - np.mean((rowsF['s1w'] - PF.y) ** 2):+.1f}")
    se = lambda df, k: ((df[k] - df["y"]) ** 2).to_numpy()  # noqa: E731
    rep = {"registered": "vault note 1.6, 2026-10-09 09:05 KST", "ctx": g.ctx, "ctx_sha": g.ctx_sha, "steps": {k: ["lmo0107 (V)"] + v for k, v in STEPS.items()},
           "F": {}, "JanJul_whole_month": {}}
    monF, monV = rowsF["day"].str[5:7].to_numpy(), rowsV["day"].str[5:7].to_numpy()
    pairs = [("S2", "s1w"), ("S4", "S2"), ("S6", "S4"), ("S8", "S6"), ("S6", "S2"), ("S8", "S2"), ("S8", "S4")]
    for new, old in pairs:
        d = se(rowsF, new) - se(rowsF, old)
        rep["F"][f"{new}_minus_{old}"] = {"all": m3.boot(d, rowsF["day"].to_numpy()),
                                          **{mm: m3.boot(d[monF == mm], rowsF["day"].to_numpy()[monF == mm]) for mm in ("02", "06")}}
        dv = se(rowsV, new) - se(rowsV, old)
        rep["JanJul_whole_month"][f"{new}_minus_{old}"] = {"all": m3.boot(dv, rowsV["day"].to_numpy()),
                                                           **{mm: m3.boot(dv[monV == mm], rowsV["day"].to_numpy()[monV == mm]) for mm in ("01", "07")}}
    d6, d8 = rep["F"]["S6_minus_S4"]["all"], rep["F"]["S8_minus_S6"]["all"]
    more = bool(d6["dmse"] <= -50 and d6["p_lt0"] >= 0.9)
    rep["reading"] = ("eight months" if (more and d8["dmse"] <= -50 and d8["p_lt0"] >= 0.9) else "six months") if more else "stop at four months"
    rep["seconds"] = round(time.time() - T0)
    (OUT / "m3_curve.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    rowsF.to_parquet(OUT / "m3_curve_rowsF.parquet", index=False)
    rowsV.to_parquet(OUT / "m3_curve_rowsV.parquet", index=False)
    log(f"CURVE: D4 {rep['F']['S4_minus_S2']['all']}, D6 {d6}, D8 {d8} -> {rep['reading']}")


if __name__ == "__main__":
    main()
