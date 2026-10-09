# SPDX-License-Identifier: GPL-3.0-only
"""F3: a hybrid ranking file = the old ranking.parquet rows (2026 Jan/Jul with the old timestamp draw) followed by the final file's Feb/Jun
rows. Built only to test that every Jan/Jul difference between the final root's feature tables and today's comes from the re-drawn
timestamps: the features rebuilt from this file must equal today's on the Jan/Jul rows except the columns computed over all rows.
    python -X utf8 research/f3/mk_hybrid.py <old ranking> <final ranking> <out>
"""
import sys

import pyarrow as pa
import pyarrow.parquet as pq

old, fin, out = sys.argv[1:4]
a = pq.read_table(old)
b = pq.read_table(fin)
assert a.schema.equals(b.schema, check_metadata=False)
assert (a.column("MVT_ID_mvt").to_numpy() == b.slice(0, a.num_rows).column("MVT_ID_mvt").to_numpy()).all()
t = pa.concat_tables([a, b.slice(a.num_rows).cast(a.schema)])
pq.write_table(t, out)
print(t.num_rows, a.num_rows, b.num_rows - a.num_rows)
