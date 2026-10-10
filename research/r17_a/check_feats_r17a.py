# SPDX-License-Identifier: GPL-3.0-only
"""R17-A check of the new feature version (private notes research/2026-10-08-r17-a-adsb-fill.md section 1): feats_v28 must equal feats_v27 on every
non-ab_ column (row order and values, NaN where NaN) and on the ab_* columns of every row that already had ADS-B day data in feats_v27; rows that
had none in feats_v27 are the 187 new 2025 days. adsb_v3 must equal adsb_v2 on adsb_v2's rows. Reports the day-data share of the leg training
rows per setting (val: 2025 outside months 1 and 7; fwd: 2025 months 1..6; full: all of 2025) before and after. Column by column (low memory).
    python research/r17_a/check_feats_r17a.py [<new feats, default feats_v28> <new adsb table, default adsb_v3>]   (sandbox root; exits 1 on any
    difference; `feats_v27 adsb_v2` compares the old files with themselves, the self-test of this script)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "cache"


def same(a, b) -> bool:
    """Equal values in the same places, a missing value (NaN, None, null) equal to a missing value (string columns hold NaN for missing)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    na, nb = pd.isna(a), pd.isna(b)
    if not np.array_equal(na, nb):
        return False
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        return bool(np.array_equal(a[~na].astype(float), b[~nb].astype(float)))
    return bool((a[~na] == b[~nb]).all())


def main() -> int:
    new_f, new_a = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("feats_v28", "adsb_v3")
    f27, f28 = pq.ParquetFile(C / "feats_v27.parquet"), pq.ParquetFile(C / f"{new_f}.parquet")
    n27, n28 = f27.schema_arrow.names, f28.schema_arrow.names
    bad = []
    if n27 != n28:
        print("column lists differ", set(n27) ^ set(n28))
        return 1
    src = f27.read(columns=["src"]).column("src").to_numpy()
    m27 = f27.read(columns=["ab_matched"]).column("ab_matched").to_numpy(zero_copy_only=False).astype(float)
    m28 = f28.read(columns=["ab_matched"]).column("ab_matched").to_numpy(zero_copy_only=False).astype(float)
    had = ~np.isnan(m27)
    for c in n27:
        a = f27.read(columns=[c]).column(c)
        b = f28.read(columns=[c]).column(c)
        if a.type != b.type:
            bad.append(c)
            continue
        av, bv = a.to_numpy(zero_copy_only=False), b.to_numpy(zero_copy_only=False)
        if not (same(av[had], bv[had]) if c.startswith("ab_") else same(av, bv)):
            bad.append(c)
    rep = {"columns": len(n27), "differing_columns": bad, "rows": int(len(src))}
    sets = {"val": (src != 0) & ~np.isin(src, (1, 7)), "fwd": np.isin(src, (1, 2, 3, 4, 5, 6)), "full": src != 0, "test": src == 0}
    rep["day_data_share"] = {k: {"feats_v27": round(float((~np.isnan(m27[s])).mean()), 4), "feats_v28": round(float((~np.isnan(m28[s])).mean()), 4),
                                 "rows": int(s.sum())} for k, s in sets.items()}
    rep["new_rows_with_day_data"] = int((~had & ~np.isnan(m28)).sum())
    rep["matched_share_train_2025"] = {"feats_v27": round(float(np.nanmean(m27[src != 0])), 4), "feats_v28": round(float(np.nanmean(m28[src != 0])), 4)}
    v2 = pd.read_parquet(C / "adsb_v2.parquet")
    v3 = pd.read_parquet(C / f"{new_a}.parquet").set_index("MVT_ID_mvt").reindex(v2["MVT_ID_mvt"].to_numpy())
    v2 = v2.set_index("MVT_ID_mvt")
    diff_v = [c for c in v2.columns if not same(v2[c].to_numpy(), v3[c].to_numpy())] if list(v2.columns) == list(v3.columns) else ["columns"]
    rep["adsb_v3_vs_v2_on_v2_rows"] = {"rows": int(len(v2)), "differing_columns": diff_v}
    print(json.dumps(rep, indent=1))
    rep["compared"] = {"feats": new_f, "adsb": new_a}
    (ROOT / "research" / "r17_a" / f"check_feats_r17a_{new_f}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 1 if (bad or diff_v) else 0


if __name__ == "__main__":
    sys.exit(main())
