# SPDX-License-Identifier: GPL-3.0-only
"""Jitter anatomy on Jan/Jul rows (old vs final), and ARR taxi-in consistency (read-only, low memory)."""
import gc
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import os
D = os.environ.get("PRC_DATA", "data")  # folder with ranking.parquet and final_phase/
R0 = D + "/ranking.parquet"
R1 = D + "/final_phase/final_ranking.parquet"
cols = ["PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
t0 = pq.read_table(R0, columns=cols)
n = t0.num_rows
t1all = pq.read_table(R1, columns=cols)
t1 = t1all.slice(0, n)
dep = pc.equal(t0["PHASE_mvt"], "DEP").to_numpy(zero_copy_only=False)
def secs(col): return pc.cast(col, pa.int64()).to_numpy(zero_copy_only=False).astype("float64") / 1e6
m0, m1 = secs(t0["MVT_TIME_UTC_mvt"]), secs(t1["MVT_TIME_UTC_mvt"])
dm = (m1 - m0)
for lab, msk in [("DEP", dep), ("ARR", ~dep)]:
    d = dm[msk]
    u, c = np.unique(d, return_counts=True)
    print(f"MVT_TIME delta {lab}: n={len(d)} changed={int((d!=0).sum())} share_changed={(d!=0).mean():.4f} min={d.min():.0f} max={d.max():.0f} mean={d.mean():.3f} std={d.std():.3f}")
    print("   delta histogram:", dict(zip(u.astype(int).tolist(), c.tolist())))
arr = ~dep
b0 = pc.cast(t0["BLOCK_TIME_UTC_mvt"], pa.int64()).to_numpy(zero_copy_only=False)
b1 = pc.cast(t1["BLOCK_TIME_UTC_mvt"], pa.int64()).to_numpy(zero_copy_only=False)
db = (b1[arr].astype("float64") - b0[arr].astype("float64")) / 1e6
u, c = np.unique(db, return_counts=True)
print(f"ARR BLOCK delta: n={len(db)} changed={int((db!=0).sum())} min={db.min():.0f} max={db.max():.0f} mean={db.mean():.3f} std={db.std():.3f}")
print("   hist:", dict(zip(u.astype(int).tolist(), c.tolist())))
x0 = t0["TAXITIME_SEC_mvt"].to_numpy(zero_copy_only=False)[arr].astype("float64")
x1 = t1["TAXITIME_SEC_mvt"].to_numpy(zero_copy_only=False)[arr].astype("float64")
# ARR taxi-in = BLOCK(in-block) - MVT(landing)
c0 = (b0[arr].astype("float64") / 1e6 - m0[arr]); c1 = (b1[arr].astype("float64") / 1e6 - m1[arr])
print("ARR TAXI == BLOCK-MVT old:", float(np.mean(np.isclose(x0, c0))), " new:", float(np.mean(np.isclose(x1, c1))))
dx = x1 - x0
print(f"ARR TAXI delta: changed={int((dx!=0).sum())} min={dx.min():.0f} max={dx.max():.0f} mean={dx.mean():.3f} std={dx.std():.3f}")
print("corr(MVT delta, BLOCK delta) ARR:", float(np.corrcoef(dm[arr], db)[0, 1]))
# new months: seconds-field check of MVT_TIME (any obvious structure)
m_all = secs(t1all["MVT_TIME_UTC_mvt"])
ph = pc.equal(t1all["PHASE_mvt"], "DEP").to_numpy(zero_copy_only=False)
newpart = np.arange(len(m_all)) >= n
for lab, msk in [("old Jan/Jul (orig)", None), ("final Jan/Jul", (~newpart)), ("final Feb/Jun", newpart)]:
    s = (m0 % 60) if msk is None else (m_all[msk] % 60)
    print(f"{lab}: MVT seconds field share of :00 = {np.mean(s==0):.4f}  n={len(s)}")
