# SPDX-License-Identifier: GPL-3.0-only
"""R17-A L4 seed averages of the refit learners: research/r17_a/avg_r17a.py (sha16 b5aacb51c0b84a7c, itself a copy of 514's
research/r10_b/s20_avg_legs.py 94154b858cc18e71) with two changes, nothing else: the settings to average are an argument (L4 fits val and fwd
on the public root, never full; vault research/2026-10-08-r17-a-adsb-fill.md section 8), and the column table knows the L4 learners exactly as
s20_avg_legs.py knew 514's: CatBoost legs average mu_cat, RealMLP legs mu_rmlp, on all rows (no G-row exception, which is for model3 legs).
Other columns are seed 0's; es_valid_ids of the new tag = seed 0's.
    python research/r17_a/avg_l4.py <leg tag> <new tag> <seeds, e.g. 0,1,2> <settings, e.g. val or val,fwd>      (root = this file's root)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COLS = {"m6r28cbwin": ["mu_cat"], "m6r28cb": ["mu_cat"], "m6r28rmwin": ["mu_rmlp"], "m6r28rm": ["mu_rmlp"]}
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
leg, new, seeds, settings = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")], sys.argv[4].split(",")
assert seeds[0] == 0 and new != leg and new not in COLS and leg in COLS and set(settings) <= {"val", "fwd", "full"}
for st in settings:
    out = ROOT / "cache" / f"{st}_{new}_s0.parquet"
    if out.exists():
        raise SystemExit(f"exists: {out}")
    b0 = pd.read_parquet(ROOT / "cache" / f"{st}_{leg}_s0.parquet")
    ids = b0["MVT_ID_mvt"].to_numpy()
    res = b0.copy()
    for c in COLS[leg]:
        vals = [b0[c].to_numpy(dtype=float)]
        for k in seeds[1:]:
            bk = pd.read_parquet(ROOT / "cache" / f"{st}_{leg}_s{k}.parquet").set_index("MVT_ID_mvt").reindex(ids)
            if bk[c].isna().any():
                raise SystemExit(f"{st} seed {k}: {c} has rows missing")
            vals.append(bk[c].to_numpy(dtype=float))
        res[c] = np.mean(vals, axis=0)
        print(f"{st} {leg} {c}: seeds {seeds}, RMS change vs seed 0 {np.sqrt(np.mean((res[c].to_numpy() - b0[c].to_numpy()) ** 2)):.3f}", flush=True)
    res.to_parquet(out, index=False)
    if st != "full":
        src = ROOT / "preds" / f"es_valid_ids_{leg}_{st}_s0.parquet"
        dst = ROOT / "preds" / f"es_valid_ids_{new}_{st}_s0.parquet"
        if dst.exists():
            raise SystemExit(f"exists: {dst}")
        shutil.copyfile(src, dst)
    print(f"wrote {out.name}", flush=True)
