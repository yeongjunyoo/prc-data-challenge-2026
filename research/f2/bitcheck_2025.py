# SPDX-License-Identifier: GPL-3.0-only
"""F2: are the corrector's 2025 rows byte copies in F3's final tables? (coordinator 10-09 02:1x; the stage-1 fit-cache key is a digest of
the matched rows' values, research/f2 reads src/ctx_corr2.fit_cache, and src/adsb_corr.design looks rows up by MVT_ID, so the key is untouched
exactly when the set of 2025 departure ids in a table and every column's bytes on those rows are the same.)
For each table: MVT_ID unique in both files; the set (and order) of ids that are 2025 departures (today's feats_v25, src != 0); same columns
and Arrow types; per column, the bytes of the 2025 rows aligned by id (numpy tobytes for numeric columns, so NaN payloads and -0.0 count;
value equality for other columns). Read-only on both roots.
    python -X utf8 research/f2/bitcheck_2025.py <final root> adsb_v1nd,geo_v1nd
Writes research/f2/out/bitcheck_2025_<root name>.json and prints one verdict line per table.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
HERE = Path(__file__).resolve().parents[2]


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def column(p: Path, c: str) -> np.ndarray:
    return pq.read_table(p, columns=[c]).column(0).to_numpy()


def main() -> int:
    root, tables = Path(sys.argv[1]), [t for t in sys.argv[2].split(",") if t]
    f = pq.read_table(HERE / "cache" / "feats_v25.parquet", columns=["MVT_ID_mvt", "src"]).to_pandas()
    y25 = np.unique(f.loc[f["src"] != 0, "MVT_ID_mvt"].to_numpy(dtype=np.float64))
    rep, bad_any = {"root": str(root), "tables": {}}, False
    for t in tables:
        po, pn = HERE / "cache" / f"{t}.parquet", root / "cache" / f"{t}.parquet"
        so, sn = pq.read_schema(po), pq.read_schema(pn)
        r = {"sha_old": sha16(po), "sha_new": sha16(pn)}
        io, inn = column(po, "MVT_ID_mvt").astype(np.float64), column(pn, "MVT_ID_mvt").astype(np.float64)
        r["rows_old"], r["rows_new"] = int(len(io)), int(len(inn))
        r["unique_old"], r["unique_new"] = bool(len(np.unique(io)) == len(io)), bool(len(np.unique(inn)) == len(inn))
        ko, kn = np.isin(io, y25), np.isin(inn, y25)
        ids_o, ids_n = io[ko], inn[kn]
        r["rows2025_old"], r["rows2025_new"] = int(ko.sum()), int(kn.sum())
        r["same_2025_id_set"] = bool(np.array_equal(np.sort(ids_o), np.sort(ids_n)))
        r["same_2025_order"] = bool(len(ids_o) == len(ids_n) and np.array_equal(ids_o, ids_n))
        r["columns_same"] = list(so.names) == list(sn.names)
        diffs = {}
        if r["same_2025_id_set"] and r["unique_new"]:
            pos_o = np.nonzero(ko)[0]
            pos_n = pd.Series(np.nonzero(kn)[0], index=ids_n).reindex(ids_o).to_numpy()
            for c in so.names:
                if c not in sn.names:
                    diffs[c] = "missing in new"
                    continue
                if str(so.field(c).type) != str(sn.field(c).type):
                    diffs[c] = f"type {so.field(c).type} -> {sn.field(c).type}"
                    continue
                a, b = column(po, c)[pos_o], column(pn, c)[pos_n]
                if a.dtype.kind in "fiub" and a.dtype == b.dtype:
                    if a.tobytes() != b.tobytes():
                        w = a.dtype.itemsize
                        rows = np.frombuffer(a.tobytes(), dtype=np.uint8).reshape(-1, w) != np.frombuffer(b.tobytes(), dtype=np.uint8).reshape(-1, w)
                        diffs[c] = f"bytes differ on {int(rows.any(axis=1).sum())} rows"
                elif not pd.Series(a).equals(pd.Series(b)):
                    diffs[c] = f"values differ ({a.dtype})"
        r["diffs_2025"] = diffs
        r["byte_copy_2025"] = bool(r["same_2025_id_set"] and r["unique_new"] and r["unique_old"] and r["columns_same"] and not diffs)
        bad_any |= not r["byte_copy_2025"]
        rep["tables"][t] = r
        print(f"{t}: {'BYTE COPY' if r['byte_copy_2025'] else 'DIFFERS'} on 2025 rows | sha {r['sha_old']} -> {r['sha_new']} | rows {r['rows_old']:,} -> "
              f"{r['rows_new']:,}, 2025 rows {r['rows2025_old']:,} -> {r['rows2025_new']:,}, same id set {r['same_2025_id_set']}, same order "
              f"{r['same_2025_order']}, unique {r['unique_old']}/{r['unique_new']}, columns same {r['columns_same']}, diffs {diffs}", flush=True)
    out = HERE / "research" / "f2" / "out" / f"bitcheck_2025_{root.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if bad_any else 0


if __name__ == "__main__":
    sys.exit(main())
