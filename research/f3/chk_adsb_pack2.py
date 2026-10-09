# SPDX-License-Identifier: GPL-3.0-only
"""F3 final root, R17-A's pack 2 (research/r17_a/final_pack2.DONE; 2026-10-09 02:1x KST: adsb_v2_merged is the legs'
adsb_v2, adsb_v1nd_merged the corrector's adsb_v1nd). Checks, then places them:
  - sha16 of final/adsb_v2_merged_v2.parquet and final/adsb_v1nd_merged_v2.parquet equal the manifest's pack2 files (54ba81f368669d1d,
    07df03ca69833082, as passed on);
  - columns and types equal today's adsb_v2 (sha16 8ee80c46165c4ef9) and adsb_v1nd (0d03ff06267e0dee);
  - the 2025 rows (day < 2026-01-01) carry exactly today's 2025 rows: same ids, every value bit-equal per id (else every fit stored on the
    interim table misses, and the corrector's fit cache too);
  - 2026 rows: count and match rate by month, and the rows of 2026-06-11 (a short source day, 89 events: must be absent, so that
    src.feats_adsb reads the day as missing, NaN, not as unmatched, 0).
On pass copies them to cache/adsb_v2.parquet and cache/adsb_v1nd.parquet and exits 0. Report: research/f3/out/adsb_pack2.json.
    python -X utf8 research/f3/chk_adsb_pack2.py <today's cache dir> <R17-A final dir>
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WANT_TODAY = {"adsb_v2": "8ee80c46165c4ef9", "adsb_v1nd": "0d03ff06267e0dee"}
PACK2 = (("adsb_v2", "adsb_v2_merged_v2.parquet"), ("adsb_v1nd", "adsb_v1nd_merged_v2.parquet"))  # sha16 54ba81f368669d1d, 07df03ca69833082


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def col_diff(a: pd.Series, b: pd.Series) -> int:
    x, y = a.to_numpy(), b.to_numpy()
    if x.dtype != y.dtype:
        return -1
    if x.dtype.kind in "fiub":
        return int((x.view(f"u{x.itemsize}") != y.view(f"u{y.itemsize}")).sum())
    if x.dtype.kind in "mM":
        return int((x.view("i8") != y.view("i8")).sum())
    na = pd.isna(a).to_numpy() & pd.isna(b).to_numpy()
    return int((~((x == y) | na)).sum())


def main():
    today, fin = Path(sys.argv[1]), Path(sys.argv[2])
    man = json.loads((fin / "manifest.json").read_text(encoding="utf-8"))
    rep, ok = {"manifest_written_kst": man.get("written_kst"), "days_under_1000_events": man.get("days_under_1000_events")}, True
    for name, src in PACK2:
        r = {}
        p_new, p_old = fin / src, today / f"{name}.parquet"
        r["sha16"] = sha16(p_new)
        r["sha16_manifest"] = man.get("pack2", {}).get("files", {}).get(src)
        r["sha16_today"] = sha16(p_old)
        r["sha_ok"] = r["sha16"] == r["sha16_manifest"] and r["sha16_today"] == WANT_TODAY[name]
        so, sn = pq.read_schema(p_old), pq.read_schema(p_new)
        r["schema_equal"] = [(f.name, str(f.type)) for f in so] == [(f.name, str(f.type)) for f in sn]
        O, N = pd.read_parquet(p_old), pd.read_parquet(p_new)
        o25 = O[O["day"].astype(str) < "2026-01-01"].sort_values("MVT_ID_mvt", kind="stable").reset_index(drop=True)
        n25 = N[N["day"].astype(str) < "2026-01-01"].sort_values("MVT_ID_mvt", kind="stable").reset_index(drop=True)
        r["rows_2025"] = [len(o25), len(n25)]
        same_ids = len(o25) == len(n25) and bool((o25["MVT_ID_mvt"].to_numpy() == n25["MVT_ID_mvt"].to_numpy()).all())
        diffs = {c: col_diff(o25[c], n25[c]) for c in O.columns} if same_ids else {}
        r["rows_2025_identical"] = bool(same_ids and not any(diffs.values()))
        r["rows_2025_columns_differing"] = {c: v for c, v in diffs.items() if v}
        n26 = N[N["day"].astype(str) >= "2026-01-01"]
        mo = n26["day"].astype(str).str[:7]
        r["rows_2026_by_month"] = {k: int(v) for k, v in mo.value_counts().sort_index().items()}
        if "matched" in N.columns:
            r["match_rate_2026_by_month"] = {k: round(float(v), 4) for k, v in n26.groupby(mo)["matched"].mean().sort_index().items()}
        r["rows_2026_06_11"] = int((n26["day"].astype(str) == "2026-06-11").sum())
        r["pass"] = bool(r["sha_ok"] and r["schema_equal"] and r["rows_2025_identical"] and r["rows_2026_06_11"] == 0)
        ok &= r["pass"]
        rep[name] = r
    rep["pass"] = bool(ok)
    if ok:
        for name, src in PACK2:
            dst = Path("cache") / f"{name}.parquet"
            assert not dst.exists(), dst
            shutil.copyfile(fin / src, dst)
            assert sha16(dst) == rep[name]["sha16"]
        rep["placed"] = ["cache/adsb_v2.parquet", "cache/adsb_v1nd.parquet"]
    Path("research/f3/out/adsb_pack2.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
