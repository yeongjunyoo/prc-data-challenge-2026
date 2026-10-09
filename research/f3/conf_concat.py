# SPDX-License-Identifier: GPL-3.0-only
"""F3 confirmation: one feature table that lets a single m6r25 val fit predict the val rows several times.
Rows: today's feats_v25 (old ranking) without its test rows (src 0 is neither fit nor eval in the val setting), plus copies of the 2025
Jan/Jul rows (src 1, 7) with MVT_ID + k * 1e10:
  k = 1, 2, 3   the Jan/Jul rows of feats_v25 rebuilt from re-draw k (research/f3/redraw.py, every column as the build makes it)
  k = 4         original rows with only sec_mvt taken from draw 1
  k = 5         original rows with only d_aobt_mode_dev taken from draw 1
Training rows are byte-for-byte today's, so the fitted models are the same for every copy (one fit, one prediction pass).
Categoricals keep today's category lists (re-drawn rows carry no new category values; checked).
    python -X utf8 research/f3/conf_concat.py <today's feats_v25> <draw root pattern with {k}> <out parquet>
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def main():
    base_p, pat, out_p = sys.argv[1:4]
    B = pd.read_parquet(base_p)
    B = B[B["src"].to_numpy() != 0].reset_index(drop=True)
    ev = np.isin(B["src"].to_numpy(), [1, 7])
    orig = B[ev].reset_index(drop=True)
    parts = [B]
    d1 = None
    for k in (1, 2, 3):
        D = pd.read_parquet(pat.format(k=k))
        D = D[np.isin(D["src"].to_numpy(), [1, 7])].set_index("MVT_ID_mvt").loc[orig["MVT_ID_mvt"].to_numpy()].reset_index()
        assert list(D.columns) == list(B.columns)
        for c in B.columns:
            if isinstance(B[c].dtype, pd.CategoricalDtype):
                vals = D[c].astype(object)
                bad = ~vals.isna() & ~vals.isin(B[c].cat.categories)
                assert not bad.any(), f"draw {k} {c}: new category values {vals[bad].unique()[:5]}"
                D[c] = pd.Categorical(vals, categories=B[c].cat.categories)
            elif D[c].dtype != B[c].dtype:
                D[c] = D[c].astype(B[c].dtype)
        if k == 1:
            d1 = D.copy()
        D["MVT_ID_mvt"] = D["MVT_ID_mvt"] + k * 1e10
        parts.append(D)
    for k, col in ((4, "sec_mvt"), (5, "d_aobt_mode_dev")):
        C = orig.copy()
        C[col] = d1[col].to_numpy()
        C["MVT_ID_mvt"] = C["MVT_ID_mvt"] + k * 1e10
        parts.append(C)
    X = pd.concat(parts, ignore_index=True)
    for c in B.columns:
        if isinstance(B[c].dtype, pd.CategoricalDtype):
            assert isinstance(X[c].dtype, pd.CategoricalDtype) and list(X[c].cat.categories) == list(B[c].cat.categories), c
        else:
            assert X[c].dtype == B[c].dtype, (c, X[c].dtype, B[c].dtype)
    assert X["MVT_ID_mvt"].is_unique
    X.to_parquet(out_p, index=False)
    ch = {c: int((~((orig[c].to_numpy() == d1[c].to_numpy()) | (pd.isna(orig[c]).to_numpy() & pd.isna(d1[c]).to_numpy()))).sum())
          for c in B.columns if c not in ("MVT_ID_mvt",) and not isinstance(B[c].dtype, pd.CategoricalDtype)}
    print("rows", len(X), "eval copies", (len(X) - len(B)) // len(orig), "| draw-1 rows changed per column:",
          {c: v for c, v in sorted(ch.items(), key=lambda x: -x[1]) if v})


if __name__ == "__main__":
    main()
