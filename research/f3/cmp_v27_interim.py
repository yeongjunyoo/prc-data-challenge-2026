# SPDX-License-Identifier: GPL-3.0-only
"""F3 fitwrap path: the 2025 rows of the interim and the real feats_v27 must be byte-identical (float bits and categorical codes and
categories included) for the fits stored on the interim table to load on the real one. Reports the columns that differ; exit 0 when
identical, 1 otherwise. Report: research/f3/out/v27_interim_vs_real.json.
    python -X utf8 research/f3/cmp_v27_interim.py <interim feats_v27> <real feats_v27>
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

a, b = pd.read_parquet(sys.argv[1]), pd.read_parquet(sys.argv[2])
rep = {"columns_equal": list(a.columns) == list(b.columns)}
A, B = a[a["src"].to_numpy() != 0].reset_index(drop=True), b[b["src"].to_numpy() != 0].reset_index(drop=True)
rep["rows_2025"] = [len(A), len(B)]
rep["ids_equal"] = bool(len(A) == len(B) and (A["MVT_ID_mvt"].to_numpy() == B["MVT_ID_mvt"].to_numpy()).all())
diff = {}
for c in a.columns:
    if c not in b.columns:
        diff[c] = "missing in real"
        continue
    if isinstance(a[c].dtype, pd.CategoricalDtype) or isinstance(b[c].dtype, pd.CategoricalDtype):
        if not (isinstance(a[c].dtype, pd.CategoricalDtype) and isinstance(b[c].dtype, pd.CategoricalDtype)
                and list(a[c].cat.categories) == list(b[c].cat.categories)):
            diff[c] = "categories differ"
            continue
        d = A[c].cat.codes.to_numpy() != B[c].cat.codes.to_numpy()
    else:
        x, y = A[c].to_numpy(), B[c].to_numpy()
        if x.dtype != y.dtype:
            diff[c] = f"dtype {x.dtype} vs {y.dtype}"
            continue
        d = (x.view(f"u{x.itemsize}") != y.view(f"u{y.itemsize}")) if x.dtype.kind in "fiub" else (x != y)
    if d.any():
        diff[c] = int(d.sum())
rep["columns_differing_2025"] = diff
rep["identical_2025"] = bool(rep["columns_equal"] and rep["ids_equal"] and not diff)
open("research/f3/out/v27_interim_vs_real.json", "w", encoding="utf-8").write(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
sys.exit(0 if rep["identical_2025"] else 1)
