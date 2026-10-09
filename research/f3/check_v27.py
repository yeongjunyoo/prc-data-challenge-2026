# SPDX-License-Identifier: GPL-3.0-only
"""F3 final root: the pass/fail check of cache/feats_v27.parquet before the LightGBM legs start (criteria fixed 2026-10-08 23:0x KST,
before the table exists). Pass when all hold:
  1. columns, order and dtypes equal today's feats_v27 (sha16 c05acdc9987ff9c7); src 0 rows = 670,790 = the final DEP rows, ids unique;
  2. 2025 rows: same ids in the same order; every column byte-equal except the seven computed over all rows (stand, actype, op, ades_reg,
     ades: values equal, codes may move; day_daobt_rel; d_aobt_mode_dev);
  3. the non-ab_ columns equal the root's feats_v25 on every row (feats_adsb only appends);
  4. 2026 Jan/Jul ids in both versions: ab_est_mvt, ab_est_first, ab_est_mvt_last move by +ΔMVT and ab_dt_to by -ΔMVT (ΔMVT from d_sch),
     every other ab_ column is unchanged, except on at most 200 rows whose ADS-B match changed;
  5. ab_matched fill on Feb/Jun 2026 rows is not lower than half of the Jan/Jul 2026 fill (the new days are really there).
Exit 0 on pass, 1 on fail; report research/f3/out/check_v27.json.
    python -X utf8 research/f3/check_v27.py <today's feats_v27.parquet>
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

ALLROW = {"stand", "actype", "op", "ades_reg", "ades", "day_daobt_rel", "d_aobt_mode_dev"}
PLUS = ["ab_est_mvt", "ab_est_first", "ab_est_mvt_last"]
MINUS = ["ab_dt_to"]


def eq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        a, b = a.astype(float), b.astype(float)
        return (a == b) | (np.isnan(a) & np.isnan(b))
    return a == b


def main():
    old_p = sys.argv[1]
    N = pd.read_parquet("cache/feats_v27.parquet")
    O = pd.read_parquet(old_p)
    V = pd.read_parquet("cache/feats_v25.parquet")
    sub = pd.read_parquet("data/submitting.parquet", columns=["MVT_ID_mvt"])
    rep, ok = {}, True
    c1 = (list(N.columns) == list(O.columns) and all(str(N[c].dtype) == str(O[c].dtype) for c in N.columns)
          and int((N["src"] == 0).sum()) == 670790 and N["MVT_ID_mvt"].is_unique
          and set(N.loc[N["src"] == 0, "MVT_ID_mvt"]) == set(sub["MVT_ID_mvt"]))
    rep["1_schema_rows"] = bool(c1)
    ok &= c1
    n25, o25 = N[N["src"] != 0].reset_index(drop=True), O[O["src"] != 0].reset_index(drop=True)
    same_ids = len(n25) == len(o25) and (n25["MVT_ID_mvt"].to_numpy() == o25["MVT_ID_mvt"].to_numpy()).all()
    bad = {}
    for c in N.columns:
        if isinstance(N[c].dtype, pd.CategoricalDtype):
            d = ~eq(n25[c].astype(object).to_numpy(), o25[c].astype(object).to_numpy()) & ~(n25[c].isna().to_numpy() & o25[c].isna().to_numpy())
        else:
            d = ~eq(n25[c].to_numpy(), o25[c].to_numpy())
        if d.any() and c not in ALLROW:
            bad[c] = int(d.sum())
    c2 = bool(same_ids) and not bad
    rep["2_rows_2025"] = {"same_ids_order": bool(same_ids), "unexpected_columns": bad, "pass": c2}
    ok &= c2
    vcols = [c for c in V.columns]
    c3 = list(N.columns[:len(vcols)]) == vcols and all(
        (N[c].astype(object).equals(V[c].astype(object)) if isinstance(V[c].dtype, pd.CategoricalDtype) else bool(eq(N[c].to_numpy(), V[c].to_numpy()).all()))
        for c in vcols)
    rep["3_v25_part_equal"] = bool(c3)
    ok &= c3
    nt, ot = N[N["src"] == 0].set_index("MVT_ID_mvt"), O[O["src"] == 0].set_index("MVT_ID_mvt")
    common = nt.index.intersection(ot.index)
    a, b = nt.loc[common], ot.loc[common]
    dm = a["d_sch"].to_numpy(float) - b["d_sch"].to_numpy(float)
    abc = [c for c in N.columns if c.startswith("ab_")]
    rowbad = np.zeros(len(common), bool)
    per = {}
    for c in abc:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        exp = y + dm if c in PLUS else y - dm if c in MINUS else y
        good = (np.abs(x - exp) < 0.5) | (np.isnan(x) & np.isnan(y))
        per[c] = int((~good).sum())
        rowbad |= ~good
    c4 = int(rowbad.sum()) <= 200
    rep["4_jan_jul_ab"] = {"common_ids": int(len(common)), "rows_not_as_expected": int(rowbad.sum()), "per_column": {k: v for k, v in per.items() if v},
                           "pass": c4}
    ok &= c4
    t = N[N["src"] == 0]
    mo = pd.to_datetime(pd.read_parquet("cache/all.parquet", columns=["MVT_ID_mvt", "t_mvt"]).set_index("MVT_ID_mvt")["t_mvt"].reindex(t["MVT_ID_mvt"]).to_numpy(),
                        unit="s").month
    fill = {int(m): round(float(t["ab_matched"].notna().to_numpy()[mo == m].mean()), 4) for m in (1, 2, 6, 7)}
    match = {int(m): round(float((t["ab_matched"].to_numpy()[mo == m] == 1).mean()), 4) for m in (1, 2, 6, 7)}
    c5 = min(fill[2], fill[6]) >= 0.5 * min(fill[1], fill[7])
    rep["5_feb_jun_fill"] = {"day_data_fill": fill, "matched_share": match, "pass": bool(c5)}
    ok &= c5
    rep["pass"] = bool(ok)
    with open("research/f3/out/check_v27.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
