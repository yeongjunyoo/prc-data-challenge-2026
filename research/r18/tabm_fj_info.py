# SPDX-License-Identifier: GPL-3.0-only
"""R18 information only (not registered, computed 2026-10-09 21:5x after M1-B): does the deployed TabM recipe (MSE, seed 0, PC CUDA r18tmc,
bit-identical to cache/val_m6r27tmcwin_s0 on val) help F2's chain-A fj blend when added with blend_add's full rule (weight fitted on all Jan/Jul
2025 val ruler days)? dMSE = (blend + TabM) - (blend without TabM) on the 326,873 Feb/Jun 2025 rows; UTC day bootstrap 2,000, seed 0. Also seed 1.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m1b_fj as m  # noqa: E402
import src.blend_oof as bo  # noqa: E402

R18 = Path(__file__).resolve().parent
V = pd.read_parquet(m.C / f"val_{m.VBASE}_s0.parquet")
F = pd.read_parquet(R18 / "mac_fj" / "cache" / "fj_f2wincbrmfw4_s0.parquet")
yv, yf = V["y"].to_numpy(dtype=float), F["y"].to_numpy(dtype=float)
fit_ok = ~V["g"].to_numpy().astype(bool) & np.isfinite(yv)
A = pd.read_parquet(m.C / "all.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "t_mvt", "ADEP_mvt"])
A = A[A["PHASE_mvt"] == "DEP"].set_index("MVT_ID_mvt").reindex(F["MVT_ID_mvt"].to_numpy())
day = (A["t_mvt"].to_numpy() // 86400).astype(np.int64)
apv = A["ADEP_mvt"].astype(str).to_numpy()
mon = pd.to_datetime(day * 86400, unit="s").month
e0 = (F["pred"].to_numpy(dtype=float) - yf) ** 2
rep = {"base": "fj_f2wincbrmfw4_s0 (F2, chain A, no TabM)", "base_rmse": round(float(np.sqrt(e0.mean())), 3)}
for seed in (0, 1):
    def leg(setting, ids, seed=seed):
        d = pd.read_parquet(R18 / "cache" / f"{setting}_r18tmc_s{seed}.parquet", columns=["MVT_ID_mvt", "mu_tabm"]).set_index("MVT_ID_mvt")["mu_tabm"]
        return d.reindex(ids).to_numpy(dtype=float)
    MUv = np.column_stack([V["mu"].to_numpy(dtype=float), leg("val", V["MVT_ID_mvt"].to_numpy())])
    Av, offv = bo.design(V, MUv)
    w = bo.simplex_ls(Av[fit_ok], (yv - offv)[fit_ok])
    MUf = np.column_stack([F["mu"].to_numpy(dtype=float), leg("fj", F["MVT_ID_mvt"].to_numpy())])
    pf, _ = bo.mix(F, MUf, np.tile(w, (len(F), 1)))
    dd = (pf - yf) ** 2 - e0
    rep[f"seed{seed}"] = {"w": round(float(w[1]), 4), "rmse": round(float(np.sqrt(((pf - yf) ** 2).mean())), 3), "d_all": m.boot(dd, day),
                          "by_month": {int(k): round(float(dd[mon == k].sum() / len(dd)), 2) for k in np.unique(mon)},
                          "by_airport": {k: round(float(dd[apv == k].sum() / len(dd)), 2) for k in sorted(set(apv))}}
print(json.dumps(rep, ensure_ascii=False))
(R18 / "screens" / "tabm_fj_info.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
