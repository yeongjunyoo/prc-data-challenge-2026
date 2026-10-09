# SPDX-License-Identifier: GPL-3.0-only
"""Column-by-column comparison of Jan/Jul rows: ranking.parquet vs final_ranking.parquet (read-only, low memory)."""
import gc
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import os
D = os.environ.get("PRC_DATA", "data")  # folder with ranking.parquet and final_phase/
R0 = D + "/ranking.parquet"
R1 = D + "/final_phase/final_ranking.parquet"
f0, f1 = pq.ParquetFile(R0), pq.ParquetFile(R1)
print("final_ranking row groups:", [f1.metadata.row_group(i).num_rows for i in range(f1.metadata.num_row_groups)])
id0 = pq.read_table(R0, columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()
id1 = pq.read_table(R1, columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()
n = len(id0)
same_prefix = np.array_equal(id1[:n], id0)
print("final_ranking first", n, "ids == ranking ids in same order:", same_prefix)
if same_prefix:
    take1 = None
else:
    o1 = np.argsort(id1, kind="stable"); pos = np.searchsorted(id1, id0, sorter=o1); take1 = o1[pos]
    print("matched by id; all matched:", bool(np.array_equal(id1[take1], id0)))
del id1; gc.collect()
names = [f.name for f in f0.schema_arrow]
for c in names:
    a = pq.read_table(R0, columns=[c])[c].combine_chunks()
    b = pq.read_table(R1, columns=[c])[c]
    b = (b.slice(0, n) if take1 is None else b.take(pa.array(take1))).combine_chunks()
    an, bn = a.is_null(), b.is_null()
    eq = pc.equal(a, b)
    both_null = pc.and_(an, bn)
    same = pc.or_(pc.fill_null(eq, False), both_null)
    ndiff = n - pc.sum(pc.cast(same, pa.int64())).as_py()
    nn_a, nn_b = a.null_count, b.null_count
    line = f"{c:24s} diff_rows={ndiff:>7d} nulls old={nn_a:>7d} new={nn_b:>7d}"
    if ndiff:
        mask = pc.invert(same)
        ia = np.flatnonzero(mask.to_numpy(zero_copy_only=False))[:3]
        ex = [(float(id0[i]), a[int(i)].as_py(), b[int(i)].as_py()) for i in ia]
        line += f" examples={ex}"
        if pa.types.is_timestamp(a.type):
            d = pc.subtract(pc.cast(b, pa.int64()), pc.cast(a, pa.int64())).filter(mask).drop_null()
            if len(d):
                dd = np.asarray(d.to_numpy()) / 1e6
                line += f" delta_sec min={dd.min():.0f} median={np.median(dd):.0f} max={dd.max():.0f} n={len(dd)}"
    print(line)
    del a, b, an, bn, eq, both_null, same; gc.collect()
