"""F3 final root: research/r10_b/s20_avg_legs.py for the full setting only, inside the root (cwd), with every seed file in cache/.
Same rule as s20: only the leg's own learned columns are averaged (rmlp: mu_rmlp; cat: mu_cat; model3 and G-band legs: p_art, mu, pred,
pred_noG on non-G rows, G rows stay seed 0); other columns are seed 0's. Seed 0 is cache/full_<leg>_s0.parquet; seeds >= 1 are
cache/full_<seed-file tag>_s<k>.parquet (for a G-band leg the model3 leg's seed files, as R10-B ran it: src/g_band.py changes only G rows).
val and fwd are not rebuilt (their averages are today's read-only copies). Refuses if the target exists.
    python -X utf8 research/f3/avg_legs_final.py <leg tag> <new tag> <seeds, e.g. 0,1,2,3> [<seed-file tag>]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
M3 = ["p_art", "mu", "pred", "pred_noG"]
COLS = {"m6r27rmwin": ["mu_rmlp"], "m6r27rm": ["mu_rmlp"], "m6r27cbwin": ["mu_cat"], "m6r27cb": ["mu_cat"],
        "m6r27skwin": M3, "m6r27sk": M3, "m6r27skwing": M3, "m6r27skg": M3}
G_ROWS_SEED0 = {"m6r27skwin", "m6r27sk", "m6r27skwing", "m6r27skg"}
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
leg, new, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
sfile = sys.argv[4] if len(sys.argv) > 4 else leg
assert seeds[0] == 0 and new != leg and new not in COLS
st = "full"
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
        bk = pd.read_parquet(ROOT / "cache" / f"{st}_{sfile}_s{k}.parquet").set_index("MVT_ID_mvt").reindex(ids)
        if bk[c].isna().any():
            raise SystemExit(f"{st} seed {k}: {c} has rows missing")
        vals.append(bk[c].to_numpy(dtype=float))
    m = np.mean(vals, axis=0)
    res[c] = np.where(g, b0[c].to_numpy(dtype=float), m)
    print(f"{st} {leg} {c}: seeds {seeds}, non-G RMS change vs seed 0 {np.sqrt(np.mean((res[c].to_numpy() - b0[c].to_numpy())[~g] ** 2)):.3f}", flush=True)
res.to_parquet(out, index=False)
print(f"wrote {out.name}", flush=True)
