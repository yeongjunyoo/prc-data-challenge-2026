# SPDX-License-Identifier: GPL-3.0-only
"""F3 final root: compare two tables keyed by MVT_ID_mvt on the ids both hold, split into the 2025 rows (must be bit-identical) and the
2026 rows (should differ only where the MVT re-draw explains it). The year comes from --year_col (a 'YYYY-MM' part or a 'YYYY-MM-DD' day
string, or a src int column where 0 = 2026). Also checks that the 2025 rows keep the same order in both files.
    python -X utf8 research/f3/cmp_tab.py <new parquet> <old parquet> <out json> --year_col part
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def year_of(s: pd.Series) -> np.ndarray:
    if s.dtype.kind in "iuf":
        return np.where(s.to_numpy() == 0, 2026, 2025)
    return s.astype(str).str[:4].astype(int).to_numpy()


def diff_col(a: pd.Series, b: pd.Series) -> dict | None:
    if a.dtype.kind in "OUSb" or isinstance(a.dtype, pd.CategoricalDtype) or b.dtype.kind in "OUS":
        sa, sb = a.astype(object).to_numpy(), b.astype(object).to_numpy()
        na, nb = pd.isna(a).to_numpy(), pd.isna(b).to_numpy()
        d = ~((sa == sb) | (na & nb))
        return {"diff": int(d.sum())} if d.any() or str(a.dtype) != str(b.dtype) else None
    x, y = a.to_numpy(dtype="float64", na_value=np.nan), b.to_numpy(dtype="float64", na_value=np.nan)
    same = (x == y) | (np.isnan(x) & np.isnan(y))
    if same.all() and str(a.dtype) == str(b.dtype):
        return None
    both = ~np.isnan(x) & ~np.isnan(y)
    dd = np.abs(x[both] - y[both])
    return {"diff": int((~same).sum()), "nan_mismatch": int((np.isnan(x) != np.isnan(y)).sum()), "max_abs": float(dd.max()) if dd.size else 0.0,
            "median_abs_on_diff": float(np.median(dd[dd > 0])) if (dd > 0).any() else 0.0, "dtype": [str(a.dtype), str(b.dtype)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("old")
    ap.add_argument("out")
    ap.add_argument("--year_col", default="part")
    a = ap.parse_args()
    N, O = pd.read_parquet(a.new), pd.read_parquet(a.old)
    res = {"new": a.new, "old": a.old, "rows_new": len(N), "rows_old": len(O), "same_columns": list(N.columns) == list(O.columns),
           "cols_only_new": [c for c in N.columns if c not in O.columns], "cols_only_old": [c for c in O.columns if c not in N.columns]}
    yN, yO = year_of(N[a.year_col]), year_of(O[a.year_col])
    n25, o25 = N[yN == 2025].reset_index(drop=True), O[yO == 2025].reset_index(drop=True)
    res["rows_2025"] = [len(n25), len(o25)]
    res["order_2025_same"] = bool(len(n25) == len(o25) and (n25["MVT_ID_mvt"].to_numpy() == o25["MVT_ID_mvt"].to_numpy()).all())
    cols = [c for c in O.columns if c in N.columns and c != "MVT_ID_mvt"]
    for grp, A, B in [("2025", n25, o25), ("2026_common", N[yN == 2026], O[yO == 2026])]:
        common = np.intersect1d(A["MVT_ID_mvt"].to_numpy(), B["MVT_ID_mvt"].to_numpy())
        A2 = A.set_index("MVT_ID_mvt").loc[common]
        B2 = B.set_index("MVT_ID_mvt").loc[common]
        g = {"rows_common": int(len(common)), "rows_only_new": int(len(A) - len(common)), "rows_only_old": int(len(B) - len(common))}
        for c in cols:
            r = diff_col(A2[c], B2[c])
            if r:
                g[c] = r
        res[grp] = g
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("2025", "2026_common")}))
    for grp in ("2025", "2026_common"):
        print(f"== {grp}:", json.dumps(res[grp])[:3000])


if __name__ == "__main__":
    main()
