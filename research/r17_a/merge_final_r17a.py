# SPDX-License-Identifier: GPL-3.0-only
"""R17-A: one per-departure ADS-B table for the final phase from a public-phase table and the final-test table (private notes
research/2026-10-08-r17-a-adsb-fill.md section 4.3). Keeps the old table's 2025 rows (day < 2026-01-01: the training and ruler days, unchanged)
and appends every row of the final table (all four test months, matched on final_ranking's own MVT time), so the 2026-01 and 2026-07 rows of the
old table (matched on ranking.parquet) are replaced, not mixed. Same columns as both inputs (build_feats.py / build_feats2.py KEEP), MVT_ID_mvt
unique, rows ordered by day then MVT_ID_mvt as the builders write them. Reads labels nowhere.
  adsb_v2-style (the legs' ab_* input):      python merge_final_r17a.py <root>/cache/adsb_v2.parquet  final/adsb_final.parquet   <out>.parquet
  adsb_v1-style (the corrector's stage 1):    python merge_final_r17a.py <root>/cache/adsb_v1.parquet  final/adsb_v1final.parquet <out>.parquet
Writes <out>.parquet and <out>.json (inputs and their sha16, rows by month, sha16 of the output).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main() -> int:
    old_p, fin_p, out_p = (Path(x) for x in sys.argv[1:4])
    if out_p.exists():
        raise SystemExit(f"exists, refusing: {out_p}")
    old, fin = pd.read_parquet(old_p), pd.read_parquet(fin_p)
    if list(old.columns) != list(fin.columns):
        raise SystemExit(f"columns differ: {set(old.columns) ^ set(fin.columns)}")
    keep = old[old["day"] < "2026-01-01"]
    assert (fin["day"] >= "2026-01-01").all(), "final table holds pre-2026 rows"
    for c in old.columns:  # the builders' missing-string forms differ (None vs NaN); align dtypes on the final table's
        if keep[c].dtype != fin[c].dtype and fin[c].dtype == object:
            keep = keep.assign(**{c: keep[c].astype(object)})
    t = pd.concat([keep, fin], ignore_index=True).sort_values(["day", "MVT_ID_mvt"], kind="stable").reset_index(drop=True)
    assert t["MVT_ID_mvt"].is_unique, "MVT_ID_mvt repeats across the two tables"
    t.to_parquet(out_p, index=False)
    meta = {"script": "research/r17_a/merge_final_r17a.py", "old": str(old_p), "old_sha16": sha16(old_p), "old_rows_kept_2025": int(len(keep)),
            "old_rows_dropped_2026": int(len(old) - len(keep)), "final": str(fin_p), "final_sha16": sha16(fin_p), "final_rows": int(len(fin)),
            "rows": int(len(t)), "rows_by_month": t["day"].str[:7].value_counts().sort_index().to_dict(), "sha16": sha16(out_p)}
    out_p.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
