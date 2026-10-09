# SPDX-License-Identifier: GPL-3.0-only
"""F2 check: do the 2025 training departures already carry the same paired +-6 s offsets (MVT vs BLOCK) as the final file? Inputs only."""
import glob, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
cols = ["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
fs = [f for f in sorted(glob.glob("data/training_*.parquet")) if f[-25:-23] in ("02", "06", "01", "07")]
tr = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/training_*.parquet"))], ignore_index=True)
d = tr[tr["PHASE_mvt"] == "DEP"].copy()
ep = lambda t: (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
off = lambda s: ((s + 30) % 60) - 30
d["om"], d["ob"] = off(ep(d["MVT_TIME_UTC_mvt"])), off(ep(d["BLOCK_TIME_UTC_mvt"]))
d["ap"] = d["ADEP_mvt"].astype(str)
rows = []
for ap, g in d.groupby("ap"):
    near = (np.abs(g["om"]) <= 6.5).mean()
    nb = (np.abs(g["ob"]) <= 6.5).mean()
    c = np.corrcoef(g["om"], g["ob"])[0, 1]
    s01 = (np.abs(g["om"] + g["ob"]) <= 1).mean()
    rows.append((ap, len(g), round(near, 4), round(nb, 4), round(c, 3), round(s01, 4), round(float(g["om"].std()), 2)))
print(pd.DataFrame(rows, columns=["ap", "dep_rows_2025", "mvt_within_6.5s_of_minute", "block_within_6.5s", "corr(off_mvt, off_block)", "share |sum|<=1s", "sd_off_mvt"]).to_string(index=False))
