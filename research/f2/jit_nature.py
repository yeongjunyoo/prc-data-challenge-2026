# SPDX-License-Identifier: GPL-3.0-only
"""F2: nature of the final_ranking timestamp change, per airport (inputs only): share changed, delta distribution, within-minute seconds."""
import sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
cols = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "ADEP_mvt", "ADES_mvt"]
o = pd.read_parquet("data/ranking.parquet", columns=cols)
n = pd.read_parquet("data/final_phase/final_ranking.parquet", columns=cols)
n = n[n["MVT_ID_mvt"].isin(o["MVT_ID_mvt"])]
m = o.merge(n, on="MVT_ID_mvt", suffixes=("_o", "_n"))
m["ap"] = np.where(m["PHASE_mvt_o"] == "DEP", m["ADEP_mvt_o"].astype(str), m["ADES_mvt_o"].astype(str))
sec = lambda s: (s.astype("int64") // 10**9) % 60 if s.dtype.kind == "M" else s
def secs(t):
    return (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()  # epoch seconds, unit- and tz-safe
m["e"] = secs(m["MVT_TIME_UTC_mvt_n"]) - secs(m["MVT_TIME_UTC_mvt_o"])
so = np.mod(secs(m["MVT_TIME_UTC_mvt_o"]), 60); sn = np.mod(secs(m["MVT_TIME_UTC_mvt_n"]), 60)
near = lambda s: (np.minimum(s, 60 - s) <= 6)
m["near_o"], m["near_n"] = near(so), near(sn)
g = m.groupby(["PHASE_mvt_o", "ap"])
out = g.agg(rows=("e", "size"), changed=("e", lambda x: float((x != 0).mean())), e_sd=("e", "std"), e_min=("e", "min"), e_max=("e", "max"),
            near6_old=("near_o", "mean"), near6_new=("near_n", "mean"))
print(out.round(3).to_string())
arr = m[m["PHASE_mvt_o"] == "ARR"].copy()
eb = secs(arr["BLOCK_TIME_UTC_mvt_n"]) - secs(arr["BLOCK_TIME_UTC_mvt_o"])
print("ARR block shift: changed", round(float((eb != 0).mean()), 3), "corr(e_mvt, e_block)", round(float(np.corrcoef(arr["e"], eb)[0, 1]), 3))
bo = np.abs(secs(arr["BLOCK_TIME_UTC_mvt_o"]) - secs(arr["SCHED_TIME_UTC_mvt_o"])) <= 6.5
bn = np.abs(secs(arr["BLOCK_TIME_UTC_mvt_n"]) - secs(arr["SCHED_TIME_UTC_mvt_n"])) <= 6.5
t = pd.DataFrame({"ap": arr["ap"], "old": bo, "new": bn}).groupby("ap")[["old", "new"]].mean()
print("ARR copy share |BLOCK-SCHED|<=6.5 old vs new\n", t.round(4).to_string())
print("copy agreement", round(float((bo == bn).mean()), 5), "old copies kept", round(float(bn[bo].mean()), 4))
