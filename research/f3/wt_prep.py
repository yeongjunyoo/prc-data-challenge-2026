# SPDX-License-Identifier: GPL-3.0-only
"""F3 fitwrap test inputs (research/f3/wt_run.sh), from the final root's feats_v25:
  <W>/r4/cache/feats_v25.parquet  one training value changed: column c_dep_m15 of the first 2025 row (src != 0) with a finite value, +1.
                                  Everything else byte-identical. The strict run on it must stop at the first fit that uses that row.
  <W>/r5/cache/feats_v25.parquet  the 2026 Jan/Jul rows (the ids of today's feats_v25 test rows) carry the hybrid table's values (old
                                  timestamp draw, features computed with the final Feb/Jun rows present; categoricals mapped by value into
                                  the final categories). Every other row byte-identical, so the training matrix is the final one and the
                                  stored fits must load; the replaced ids go to <W>/r5/replaced_ids.parquet (real re-draw move check).
Both files are read back and checked against the final table as model3 reads it (categories and codes included).
    python -X utf8 research/f3/wt_prep.py <W> <final feats_v25> <hybrid feats_v25> <today's feats_v25>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W, fin, hyb, today = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]


def differing_rows(a: pd.Series, b: pd.Series) -> np.ndarray:
    if isinstance(a.dtype, pd.CategoricalDtype):
        assert isinstance(b.dtype, pd.CategoricalDtype) and list(a.cat.categories) == list(b.cat.categories), a.name
        return a.cat.codes.to_numpy() != b.cat.codes.to_numpy()
    x, y = a.to_numpy(), b.to_numpy()
    assert x.dtype == y.dtype, a.name
    if x.dtype.kind == "f":
        return ~((x == y) | (np.isnan(x) & np.isnan(y)))
    return x != y


def check(path: Path, F: pd.DataFrame, allowed_rows: np.ndarray) -> dict:
    R = pd.read_parquet(path)
    assert list(R.columns) == list(F.columns)
    out = {}
    for c in F.columns:
        d = differing_rows(R[c], F[c])
        assert not (d & ~allowed_rows).any(), f"{path}: column {c} differs outside the allowed rows"
        if d.any():
            out[c] = int(d.sum())
    return out


F = pd.read_parquet(fin)
rep = {}
# r4
C = "c_dep_m15"
v = F[C].to_numpy()
i = int(np.where((F["src"].to_numpy() != 0) & np.isfinite(v))[0][0])
P4 = F.copy()
vv = P4[C].to_numpy().copy()
vv[i] = vv[i] + np.float32(1)
P4[C] = vv
P4.to_parquet(W / "r4" / "cache" / "feats_v25.parquet", index=False)
del P4
allow = np.zeros(len(F), bool)
allow[i] = True
rep["r4"] = {"column": C, "row": i, "id": float(F["MVT_ID_mvt"].iloc[i]), "old": float(v[i]), "new": float(vv[i]),
             "read_back_differing": check(W / "r4" / "cache" / "feats_v25.parquet", F, allow)}
assert rep["r4"]["read_back_differing"] == {C: 1}, rep["r4"]
# r5
ids = pd.read_parquet(today, columns=["MVT_ID_mvt", "src"])
ids = ids.loc[ids["src"] == 0, "MVT_ID_mvt"].to_numpy()
pos = np.where(np.isin(F["MVT_ID_mvt"].to_numpy(), ids))[0]
assert len(pos) == len(ids) and (F["src"].to_numpy()[pos] == 0).all()
H = pd.read_parquet(hyb).set_index("MVT_ID_mvt")
Hj = H.reindex(F["MVT_ID_mvt"].to_numpy()[pos])
del H
assert list(Hj.columns) == [c for c in F.columns if c != "MVT_ID_mvt"]
P5 = F.copy()
unmapped = {}
for c in F.columns:
    if c == "MVT_ID_mvt":
        continue
    if isinstance(F[c].dtype, pd.CategoricalDtype):
        new = pd.Categorical(Hj[c].astype(object).to_numpy(), categories=F[c].cat.categories)
        bad = int((pd.isna(new) & Hj[c].notna().to_numpy()).sum())
        if bad:
            unmapped[c] = bad
        codes = P5[c].cat.codes.to_numpy().copy()
        codes[pos] = new.codes
        P5[c] = pd.Categorical.from_codes(codes, dtype=F[c].dtype)
    else:
        col = P5[c].to_numpy().copy()
        col[pos] = Hj[c].to_numpy().astype(col.dtype)
        P5[c] = col
assert not unmapped, unmapped
P5.to_parquet(W / "r5" / "cache" / "feats_v25.parquet", index=False)
del P5, Hj
pd.DataFrame({"MVT_ID_mvt": F["MVT_ID_mvt"].to_numpy()[pos]}).to_parquet(W / "r5" / "replaced_ids.parquet", index=False)
allow = np.zeros(len(F), bool)
allow[pos] = True
rep["r5"] = {"rows_replaced": int(len(pos)), "read_back_rows_differing_by_column": check(W / "r5" / "cache" / "feats_v25.parquet", F, allow)}
(W / "out").mkdir(exist_ok=True)
(W / "out" / "wt_prep.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
print(json.dumps(rep, indent=1))
