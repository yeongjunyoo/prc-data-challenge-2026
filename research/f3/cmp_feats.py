# SPDX-License-Identifier: GPL-3.0-only
"""F3 final root: compare a rebuilt feature table with today's table of the same name on the rows both hold.
Per column: rows that differ (NaN-aware, exact), for categoricals both the string values and the integer codes, max |diff| for numbers.
Also the dtype and column order. Two row groups: training rows (src != 0, 2025, must be bit-identical apart from columns computed over
all rows) and the test rows whose MVT_ID is in both (2026 Jan/Jul; differences there should come only from the MVT re-draw).
    python -X utf8 research/f3/cmp_feats.py <new parquet> <old parquet> <out json> [--delta_mvt <final_ranking> <old ranking>]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd


def cmp_col(a: pd.Series, b: pd.Series) -> dict:
    r = {"dtype_new": str(a.dtype), "dtype_old": str(b.dtype)}
    if isinstance(a.dtype, pd.CategoricalDtype) or isinstance(b.dtype, pd.CategoricalDtype):
        sa, sb = a.astype(object).where(a.notna(), None), b.astype(object).where(b.notna(), None)
        diff = ~((sa.to_numpy() == sb.to_numpy()) | (a.isna().to_numpy() & b.isna().to_numpy()))
        r["diff_values"] = int(diff.sum())
        if isinstance(a.dtype, pd.CategoricalDtype) and isinstance(b.dtype, pd.CategoricalDtype):
            r["diff_codes"] = int((a.cat.codes.to_numpy() != b.cat.codes.to_numpy()).sum())
            r["n_cats_new"], r["n_cats_old"] = len(a.cat.categories), len(b.cat.categories)
        return r
    x, y = a.to_numpy(dtype="float64", na_value=np.nan), b.to_numpy(dtype="float64", na_value=np.nan)
    nan_x, nan_y = np.isnan(x), np.isnan(y)
    same = (x == y) | (nan_x & nan_y)
    r["diff"] = int((~same).sum())
    r["nan_mismatch"] = int((nan_x != nan_y).sum())
    both = ~nan_x & ~nan_y
    d = np.abs(x[both] - y[both])
    r["max_abs"] = float(d.max()) if d.size else 0.0
    if r["diff"]:
        r["mean_abs_on_diff"] = float(d[d > 0].mean()) if (d > 0).any() else 0.0
    return r


def main():
    new_p, old_p, out_p = sys.argv[1:4]
    N, O = pd.read_parquet(new_p), pd.read_parquet(old_p)
    res = {"new": new_p, "old": old_p, "rows_new": len(N), "rows_old": len(O), "same_column_order": list(N.columns) == list(O.columns),
           "cols_only_new": [c for c in N.columns if c not in O.columns], "cols_only_old": [c for c in O.columns if c not in N.columns]}
    cols = [c for c in O.columns if c in N.columns]
    for grp, mn, mo in [("train", N["src"].to_numpy() != 0, O["src"].to_numpy() != 0), ("test_common", N["src"].to_numpy() == 0, O["src"].to_numpy() == 0)]:
        a, b = N[mn].reset_index(drop=True), O[mo].reset_index(drop=True)
        if grp == "train":
            if len(a) != len(b) or not (a["MVT_ID_mvt"].to_numpy() == b["MVT_ID_mvt"].to_numpy()).all():
                res[grp] = {"error": "training rows differ in count or order", "n_new": len(a), "n_old": len(b)}
                continue
        else:
            common = np.intersect1d(a["MVT_ID_mvt"].to_numpy(), b["MVT_ID_mvt"].to_numpy())
            a = a.set_index("MVT_ID_mvt").loc[common].reset_index()
            b = b.set_index("MVT_ID_mvt").loc[common].reset_index()
        g = {"rows": len(a)}
        for c in cols:
            r = cmp_col(a[c], b[c])
            if r.get("diff") or r.get("diff_values") or r.get("diff_codes") or r["dtype_new"] != r["dtype_old"]:
                g[c] = r
        g["identical_columns"] = len(cols) - (len(g) - 1)
        res[grp] = g
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("train", "test_common")}))
    for grp in ("train", "test_common"):
        g = res.get(grp, {})
        print(f"== {grp}: rows {g.get('rows')}, identical columns {g.get('identical_columns')} of {len(cols)}")
        for c, r in g.items():
            if c in ("rows", "identical_columns"):
                continue
            print(f"   {c}: {r}")


if __name__ == "__main__":
    main()
