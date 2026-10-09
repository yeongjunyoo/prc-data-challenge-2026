# SPDX-License-Identifier: GPL-3.0-only
"""Content checks with low memory: months, phases, id overlap, blanking, submitting alignment (read-only)."""
import gc
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import os
D = os.environ.get("PRC_DATA", "data")  # folder with ranking.parquet and final_phase/
F = D + "/final_phase"
R0, S0 = D + "/ranking.parquet", D + "/submitting.parquet"
R1, S1 = F + "/final_ranking.parquet", F + "/final_submitting.parquet"
cols = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "ADES_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
info = {}
for name, path in [("ranking", R0), ("final_ranking", R1)]:
    t = pq.read_table(path, columns=cols, read_dictionary=["PHASE_mvt", "ADEP_mvt", "ADES_mvt"])
    month = pc.strftime(t["MVT_TIME_UTC_mvt"], format="%Y-%m")
    phase = pc.cast(t["PHASE_mvt"], pa.string())
    g = pa.table({"month": month, "phase": phase}).group_by(["month", "phase"]).aggregate([([], "count_all")]).sort_by([("month", "ascending"), ("phase", "ascending")])
    print(f"== {name}: rows={t.num_rows} id nulls={t['MVT_ID_mvt'].null_count}")
    for r in g.to_pylist(): print("  ", r["month"], r["phase"], r["count_all"])
    isdep = pc.equal(phase, "DEP")
    dep = t.filter(isdep); arr = t.filter(pc.invert(isdep))
    print(f"   DEP rows={dep.num_rows} BLOCK null={dep['BLOCK_TIME_UTC_mvt'].null_count} TAXI null={dep['TAXITIME_SEC_mvt'].null_count}")
    print(f"   ARR rows={arr.num_rows} BLOCK null={arr['BLOCK_TIME_UTC_mvt'].null_count} TAXI null={arr['TAXITIME_SEC_mvt'].null_count}")
    mm = pc.min_max(t["MVT_TIME_UTC_mvt"]); print("   MVT_TIME min/max:", mm["min"], mm["max"])
    dmonth = month.filter(isdep)
    gg = pa.table({"apt": pc.cast(dep["ADEP_mvt"], pa.string()), "month": dmonth}).group_by(["apt", "month"]).aggregate([([], "count_all")]).sort_by([("apt", "ascending"), ("month", "ascending")])
    byapt = {}
    for r in gg.to_pylist(): byapt.setdefault(r["apt"], {})[r["month"]] = r["count_all"]
    print("   DEP by ADEP x month:")
    for a, d in byapt.items(): print("     ", a, d)
    info[name] = {"ids": t["MVT_ID_mvt"].to_numpy(), "dep_ids": dep["MVT_ID_mvt"].to_numpy(), "dep_month": np.array(dmonth.to_pylist()), "month": np.array(month.to_pylist())}
    del t, dep, arr, month, phase, g, gg; gc.collect()
i0, i1 = info["ranking"]["ids"], info["final_ranking"]["ids"]
in0 = np.isin(i1, i0)
print(f"\nranking ids found in final_ranking: {np.isin(i0, i1).sum()} of {len(i0)}; final-only ids: {(~in0).sum()}")
fm = info["final_ranking"]["month"]
u, c = np.unique(fm[~in0], return_counts=True); print("final-only rows by month:", dict(zip(u.tolist(), c.tolist())))
u, c = np.unique(fm[in0], return_counts=True); print("final rows with old ids by month:", dict(zip(u.tolist(), c.tolist())))
s0 = pq.read_table(S0); s1 = pq.read_table(S1)
a0, a1 = s0["MVT_ID_mvt"].to_numpy(), s1["MVT_ID_mvt"].to_numpy()
print(f"\nsubmitting rows={len(a0)} unique={len(np.unique(a0))} TAXI non-null={len(a0)-s0['TAXITIME_SEC_mvt'].null_count}")
print(f"final_submitting rows={len(a1)} unique={len(np.unique(a1))} TAXI non-null={len(a1)-s1['TAXITIME_SEC_mvt'].null_count}")
print("old submitting ids found in final_submitting:", int(np.isin(a0, a1).sum()), "of", len(a0), "; final-only:", int((~np.isin(a1, a0)).sum()))
d1 = info["final_ranking"]["dep_ids"]; d0 = info["ranking"]["dep_ids"]
print("final_submitting ids == final_ranking DEP ids (set):", set(a1.tolist()) == set(d1.tolist()))
print("final_submitting order == final_ranking DEP order:", np.array_equal(a1, d1))
print("old submitting order == ranking DEP order:", np.array_equal(a0, d0))
print("final_submitting sorted asc:", bool((np.diff(a1) > 0).all()), "| old submitting sorted asc:", bool((np.diff(a0) > 0).all()))
print("final_submitting first len(old) ids == old submitting ids, same order:", bool(np.array_equal(a1[:len(a0)], a0)))
print("first 3 final_submitting ids:", a1[:3].tolist(), "last 3:", a1[-3:].tolist())
print("id ranges: old sub", a0.min(), a0.max(), "| final sub", a1.min(), a1.max())
dmon = info["final_ranking"]["dep_month"]
newmask = ~np.isin(a1, a0)
u, c = np.unique(dmon[newmask], return_counts=True); print("final_submitting new ids by month:", dict(zip(u.tolist(), c.tolist())))
u, c = np.unique(dmon[~newmask], return_counts=True); print("final_submitting old ids by month:", dict(zip(u.tolist(), c.tolist())))
print("final_submitting TAXI dtype", s1.schema.field("TAXITIME_SEC_mvt").type, "first values", s1["TAXITIME_SEC_mvt"][:3].to_pylist())
print("MVT_ID all integral (final sub):", bool(np.all(np.floor(a1) == a1)))
