# SPDX-License-Identifier: GPL-3.0-only
"""Null share per month for DEP rows in final_ranking (read-only, column at a time)."""
import gc
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import os
R1 = os.environ.get("PRC_DATA", "data") + "/final_phase/final_ranking.parquet"
base = pq.read_table(R1, columns=["PHASE_mvt", "MVT_TIME_UTC_mvt"])
dep = pc.equal(base["PHASE_mvt"], "DEP").to_numpy(zero_copy_only=False)
month = np.array(pc.strftime(base["MVT_TIME_UTC_mvt"], format="%m").to_pylist())[dep]
del base; gc.collect()
months = ["01", "02", "06", "07"]
cols = ["FLIGHT_ID_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt"]
print(f"{'col':24s} " + " ".join(f"{m:>8s}" for m in months))
print(f"{'DEP rows':24s} " + " ".join(f"{int((month==m).sum()):>8d}" for m in months))
for c in cols:
    a = pq.read_table(R1, columns=[c])[c].combine_chunks()
    isn = a.is_null().to_numpy(zero_copy_only=False)[dep]
    print(f"{c:24s} " + " ".join(f"{isn[month==m].mean()*100:7.2f}%" for m in months))
    del a, isn; gc.collect()
