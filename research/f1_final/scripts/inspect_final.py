# SPDX-License-Identifier: GPL-3.0-only
"""Inspect final-phase files vs current ranking/submitting (read-only)."""
import sys, json
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import pandas as pd
import os
D = os.environ.get("PRC_DATA", "data")  # folder with ranking.parquet and final_phase/
F = D + "/final_phase"
files = {
    "ranking": D + "/ranking.parquet",
    "submitting": D + "/submitting.parquet",
    "final_ranking": F + r"\final_ranking.parquet",
    "final_submitting": F + r"\final_submitting.parquet",
    "training_2025-01": D + "/training_2025-01-01_2025-02-01.parquet",
}
sch = {}
for k, p in files.items():
    m = pq.ParquetFile(p)
    s = m.schema_arrow
    sch[k] = s
    print(f"== {k}: rows={m.metadata.num_rows} row_groups={m.metadata.num_row_groups} cols={len(s)} created_by={m.metadata.created_by}")
    md = s.metadata or {}
    for mk in md:
        v = md[mk]
        print("   meta key", mk.decode(errors='replace'), "len", len(v))
print()
for a, b in [("ranking", "final_ranking"), ("submitting", "final_submitting"), ("training_2025-01", "final_ranking")]:
    sa, sb = sch[a], sch[b]
    na, nb = [f.name for f in sa], [f.name for f in sb]
    print(f"-- schema {a} vs {b}: same names&order={na==nb}")
    for f in sa:
        if f.name not in nb: print("   only in", a, f.name, f.type)
        elif sb.field(f.name).type != f.type: print("   type differs", f.name, f.type, "->", sb.field(f.name).type)
    for f in sb:
        if f.name not in na: print("   only in", b, f.name, f.type)
print()
print("final_ranking columns:", [f"{f.name}:{f.type}" for f in sch["final_ranking"]])
print("final_submitting columns:", [f"{f.name}:{f.type}" for f in sch["final_submitting"]])
