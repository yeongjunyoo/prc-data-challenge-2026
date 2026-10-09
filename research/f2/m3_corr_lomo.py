# SPDX-License-Identifier: GPL-3.0-only
"""F2 M3 follow-up, NOT pre-registered (written 2026-10-08 22:4x after research/f2/m3_corr.py's numbers; vault note section 4.3).
Why: m3_corr's U_F and I_F fit stage 2 on other days of the same month. aiu_m_* is constant within an airport-month, so with day folds the
model can learn each airport-month's mean residual from that month's other days: a leak the final file cannot have (no Feb/Jun 2026 label).
This check holds out the whole month instead: stage 2 = one model (the T1 recipe of m3_corr) trained on all V rows plus the F rows of the
OTHER F month, applied to the held-out F month; compared on the same rows with T1 (V rows only). Also the symmetric check on V: Jan rows from a
model on Jul + F, against Jul only (and Jul from Jan + F). Day-block bootstrap as in m3_corr.
    python -u research/f2/m3_corr_lomo.py --vbase m6r27skwing --fbase f2skwing --seed 0 --ctx f2ctx_v3 --ctx_sha <16> --tag f2m3 --threads 4
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


def main():
    pa = argparse.ArgumentParser()
    for k in ("--vbase", "--fbase", "--ctx", "--ctx_sha", "--tag"):
        pa.add_argument(k, required=True)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--threads", type=int, default=4)
    g = pa.parse_args()
    a = SimpleNamespace(adsb="adsb_v1nd", adsb_sha="0d03ff06267e0dee", geo="geo_v1nd", geo_sha="43310b9bbeefe81b", feats="feats_v25", folds=10,
                        rounds=300, rounds2=400, max_pred=3600.0, threads=g.threads, seed=g.seed, scale2=0.5, ctx=g.ctx, ctx_sha=g.ctx_sha)
    ac.below_normal()
    m3.setup()
    A = cc.stage1_inputs(a)
    ctx = cc.checked(a.ctx, a.ctx_sha).set_index("MVT_ID_mvt")
    V = pd.read_parquet(CACHE / f"val_{g.vbase}_s{g.seed}.parquet")
    F = pd.read_parquet(CACHE / f"fj_{g.fbase}_s{g.seed}.parquet")
    T = ac.design(V, A)
    T = T[np.isfinite(T["y"].to_numpy())]
    RV, RF = cc.Rows(V, A, a, ctx), cc.Rows(F, A, a, ctx)
    fo, fT, models = cc.stage1_folds(T, a)
    oof = np.zeros(len(T))
    for j, m in enumerate(models):
        oof[fT == j] = ac.predict(m, T[fT == j])
    d1V = np.zeros(len(V))
    d1V[T["pos"].to_numpy()] = oof
    d1F = np.zeros(len(F))
    yV, yF = V["y"].to_numpy(dtype=float), F["y"].to_numpy(dtype=float)
    s1V = np.clip(V["pred"].to_numpy(dtype=float) + d1V, 0, None)
    s1F = np.clip(F["pred"].to_numpy(dtype=float), 0, None)
    MV, MF = RV.design2(s1V, d1V), RF.design2(s1F, d1F)
    appV, appF = s1V < a.max_pred, s1F < a.max_pred
    monV = pd.Series(RV.day).str[5:7].astype(int).to_numpy()
    monF = pd.Series(RF.day).str[5:7].astype(int).to_numpy()
    tV, tF = yV - s1V, yF - s1F
    fit = lambda M, t: cc.fit2(M, t, a.seed, a.rounds2, a.threads)  # noqa: E731
    se = lambda p, y: (p - y) ** 2  # noqa: E731
    out = {}
    base_F = se(m3.windowed(F, s1F, a.feats), yF)
    base_V = se(m3.windowed(V, s1V, a.feats), yV)
    trV = appV
    m1 = fit(MV[trV], tV[trV])
    x1 = np.zeros(len(F))
    x1[appF] = m1.predict(MF[appF])
    p_T1 = se(m3.windowed(F, np.clip(s1F + a.scale2 * x1, 0, None), a.feats), yF)
    for hold, other in ((2, 6), (6, 2)):
        tr_o = appF & (monF == other)
        m = fit(np.vstack([MV[trV], MF[tr_o]]), np.concatenate([tV[trV], tF[tr_o]]))
        x = x1.copy()
        sel = appF & (monF == hold)
        x[sel] = m.predict(MF[sel])
        p = se(m3.windowed(F, np.clip(s1F + a.scale2 * x, 0, None), a.feats), yF)
        rows = monF == hold
        out[f"F{hold}_L_minus_T1"] = m3.boot((p - p_T1)[rows], RF.day[rows])
        out[f"F{hold}_T1_vs_none"] = m3.boot((p_T1 - base_F)[rows], RF.day[rows])
        out[f"F{hold}_L_vs_none"] = m3.boot((p - base_F)[rows], RF.day[rows])
        out[f"F{hold}_L_minus_T1_by_airport"] = {k: round(float((p - p_T1)[rows & (RF.ap == k)].sum() / rows.sum()), 2) for k in sorted(set(RF.ap))}
        print(hold, json.dumps({k: v for k, v in out.items() if k.startswith(f"F{hold}_") and "airport" not in k}), flush=True)
    for hold, other in ((1, 7), (7, 1)):
        tr_o = appV & (monV == other)
        mo = fit(MV[tr_o], tV[tr_o])
        mf = fit(np.vstack([MV[tr_o], MF[appF]]), np.concatenate([tV[tr_o], tF[appF]]))
        sel = appV & (monV == hold)
        xo, xf = np.zeros(len(V)), np.zeros(len(V))
        xo[sel], xf[sel] = mo.predict(MV[sel]), mf.predict(MV[sel])
        po = se(m3.windowed(V, np.clip(s1V + a.scale2 * xo, 0, None), a.feats), yV)
        pf = se(m3.windowed(V, np.clip(s1V + a.scale2 * xf, 0, None), a.feats), yV)
        rows = monV == hold
        out[f"V{hold}_other_month_vs_none"] = m3.boot((po - base_V)[rows], RV.day[rows])
        out[f"V{hold}_other_plus_F_vs_none"] = m3.boot((pf - base_V)[rows], RV.day[rows])
        out[f"V{hold}_plusF_minus_other"] = m3.boot((pf - po)[rows], RV.day[rows])
        print(hold, json.dumps({k: v for k, v in out.items() if k.startswith(f"V{hold}_")}), flush=True)
    out["seconds"] = round(time.time() - T0)
    Path("research/f2/out").mkdir(parents=True, exist_ok=True)
    (Path("research/f2/out") / f"m3_corr_lomo_{g.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
