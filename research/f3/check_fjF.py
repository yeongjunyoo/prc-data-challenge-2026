# SPDX-License-Identifier: GPL-3.0-only
"""F3 final round c (10-10 08:2x, F2's lever 2): check F2's tables before the four-month corrector reads them. cache/adsb_v1ndfjF and
cache/geo_v1ndfjF (cloned from f2_run, sha16 e13437a2c55a77c5 and 4fd677dc9060ce22, built by f2_run research/f2/fj_tables.py) must be this root's
adsb_v1nd and geo_v1nd (07df03ca69833082, 9dfa71b64e1099de) with every row unchanged (same columns and dtypes, every cell equal, a missing value
equal to a missing value) plus 326,873 rows, all of them February or June 2025 movements (cache/all.parquet t_mvt, seconds). Exit 1 otherwise.
Report research/f3/out/check_fjF.json.
    python -X utf8 research/f3/check_fjF.py        (cwd = the final root)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

C = Path("cache")
WANT = {"adsb_v1nd": "07df03ca69833082", "geo_v1nd": "9dfa71b64e1099de", "adsb_v1ndfjF": "e13437a2c55a77c5", "geo_v1ndfjF": "4fd677dc9060ce22"}
EXTRA = 326873


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def same(a: pd.Series, b: pd.Series) -> bool:
    a, b = a.to_numpy(), b.to_numpy()
    na, nb = pd.isna(a), pd.isna(b)
    if not np.array_equal(na, nb):
        return False
    return bool((a[~na] == b[~nb]).all())


def main() -> int:
    t0 = time.time()
    rep, fails = {"written": time.strftime("%Y-%m-%d %H:%M:%S")}, []
    for k, w in WANT.items():
        got = sha16(C / f"{k}.parquet")
        rep[f"{k}_sha16"] = got
        if got != w:
            fails.append(f"{k}: sha16 {got} != {w}")
    tm = pd.read_parquet(C / "all.parquet", columns=["MVT_ID_mvt", "t_mvt"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")["t_mvt"]
    for old, new in (("adsb_v1nd", "adsb_v1ndfjF"), ("geo_v1nd", "geo_v1ndfjF")):
        a, b = pd.read_parquet(C / f"{old}.parquet"), pd.read_parquet(C / f"{new}.parquet")
        r = {"rows_old": int(len(a)), "rows_new": int(len(b))}
        if list(a.columns) != list(b.columns):
            fails.append(f"{new}: columns differ from {old}")
        elif [str(x) for x in a.dtypes] != [str(x) for x in b.dtypes]:
            fails.append(f"{new}: dtypes differ from {old}")
        if not (a["MVT_ID_mvt"].is_unique and b["MVT_ID_mvt"].is_unique):
            fails.append(f"{old} or {new}: MVT_ID_mvt repeats")
        inb = a["MVT_ID_mvt"].isin(b["MVT_ID_mvt"])
        r["old_rows_missing_in_new"] = int((~inb).sum())
        if (~inb).any():
            fails.append(f"{new}: {int((~inb).sum())} rows of {old} are missing")
        bo = b.set_index("MVT_ID_mvt").reindex(a["MVT_ID_mvt"])
        ao = a.set_index("MVT_ID_mvt")
        diff = [c for c in ao.columns if not same(ao[c], bo[c])]
        r["columns_differing_on_old_rows"] = diff
        if diff:
            fails.append(f"{new}: old rows differ in {diff}")
        ex = b[~b["MVT_ID_mvt"].isin(a["MVT_ID_mvt"])]
        r["extra_rows"] = int(len(ex))
        months = pd.Series(pd.to_datetime(tm.reindex(ex["MVT_ID_mvt"]).to_numpy(), unit="s", utc=True).strftime("%Y-%m")).value_counts().to_dict()
        r["extra_rows_by_month"] = {str(k): int(v) for k, v in sorted(months.items())}
        if len(ex) != EXTRA:
            fails.append(f"{new}: {len(ex)} extra rows, expected {EXTRA}")
        if set(r["extra_rows_by_month"]) - {"2025-02", "2025-06"}:
            fails.append(f"{new}: extra rows outside Feb/Jun 2025: {r['extra_rows_by_month']}")
        if "matched" in ex.columns:
            r["extra_rows_matched"] = int(pd.to_numeric(ex["matched"], errors="coerce").fillna(0).astype(bool).sum())
        rep[new] = r
    rep["fails"], rep["pass"], rep["seconds"] = fails, not fails, round(time.time() - t0, 1)
    Path("research/f3/out/check_fjF.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
