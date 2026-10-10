# SPDX-License-Identifier: GPL-3.0-only
"""R17-A copy of research/r10_b/s20_avg_legs.py (sha16 94154b858cc18e71) for the Mac sandbox ~/nova-adsb/prcm/r17a_sbx (vault
research/2026-10-08-r17-a-adsb-fill.md section 1). Changes, nothing else: ROOT is the sandbox (this file's location), the seed files are the
sandbox's own cache/<setting>_<seed-file tag>_s<k>.parquet (this lane trains every seed here), and the column table knows the feats_v28 legs
(m6r28skwin, m6r28sk and their G-band legs m6r28skwing, m6r28skg). Original text follows.
R10-B: seed-averaged base legs as new cache tags (pre-registration vault note 7.7). Seed 0 is the existing cache file; seeds >= 1 come from
research/r10_b/mac/ (Mac runs in ~/nova-adsb/prcm/r10b_run, original commands with --seed k). Only the leg's own learned columns are averaged:
  rmlp legs (m6r27rmwin, m6r27rm): mu_rmlp;  cat legs (m6r27cbwin, m6r27cb): mu_cat;
  model3 legs (m6r27skwin, m6r27sk, and their G-band legs m6r27skwing, m6r27skg): p_art, mu, pred, pred_noG on non-G rows (G rows stay seed 0,
  so the G band and every G step see seed 0). src/g_band.py changes only the G rows' pred (checked 10-07 07:5x on all three settings), so the
  G-band leg's non-G rows are the model3 leg's: run on the G-band leg with the model3 seed files (4th argument).
Other columns are seed 0's. es_valid_ids of the new tag = seed 0's (early-stopping rows depend on days, not seeds: the Mac's seed-1 es ids of m6r27skwin, m6r27cbwin,
m6r27rmwin, val and fwd, are the seed-0 rows in the same order, checked 10-07 08:0x with research/r10_b/mac/es/).
New names only: refuses if the target cache or preds file exists.
    python avg_r17a.py <leg tag> <new tag> <seeds, e.g. 0,1,2,3> [<seed-file tag, default the leg tag>]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAC = ROOT / "cache"
M3 = ["p_art", "mu", "pred", "pred_noG"]
COLS = {"m6r28skwin": M3, "m6r28sk": M3, "m6r28skwing": M3, "m6r28skg": M3}
G_ROWS_SEED0 = {"m6r28skwin", "m6r28sk", "m6r28skwing", "m6r28skg"}
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
leg, new, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
sfile = sys.argv[4] if len(sys.argv) > 4 else leg
assert seeds[0] == 0 and new != leg and new not in COLS
for st in ("val", "fwd", "full"):
    out = ROOT / "cache" / f"{st}_{new}_s0.parquet"
    if out.exists():
        raise SystemExit(f"exists: {out}")
    b0 = pd.read_parquet(ROOT / "cache" / f"{st}_{leg}_s0.parquet")
    ids = b0["MVT_ID_mvt"].to_numpy()
    g = b0["g"].to_numpy().astype(bool) if leg in G_ROWS_SEED0 else np.zeros(len(b0), bool)
    res = b0.copy()
    for c in COLS[leg]:
        if c not in b0.columns:
            continue
        vals = [b0[c].to_numpy(dtype=float)]
        for k in seeds[1:]:
            bk = pd.read_parquet(MAC / f"{st}_{sfile}_s{k}.parquet").set_index("MVT_ID_mvt").reindex(ids)
            if bk[c].isna().any():
                raise SystemExit(f"{st} seed {k}: {c} has rows missing")
            vals.append(bk[c].to_numpy(dtype=float))
        m = np.mean(vals, axis=0)
        res[c] = np.where(g, b0[c].to_numpy(dtype=float), m)
        print(f"{st} {leg} {c}: seeds {seeds}, non-G RMS change vs seed 0 {np.sqrt(np.mean((res[c].to_numpy() - b0[c].to_numpy())[~g] ** 2)):.3f}", flush=True)
    res.to_parquet(out, index=False)
    if st != "full":
        src = ROOT / "preds" / f"es_valid_ids_{leg}_{st}_s0.parquet"
        dst = ROOT / "preds" / f"es_valid_ids_{new}_{st}_s0.parquet"
        if dst.exists():
            raise SystemExit(f"exists: {dst}")
        shutil.copyfile(src, dst)
    print(f"wrote {out.name}", flush=True)
