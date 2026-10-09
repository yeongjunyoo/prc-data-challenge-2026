# SPDX-License-Identifier: GPL-3.0-only
"""F2 regression check of F3's final cache root against prc-2026 before the final legs start (10-08 22:5x, F2 role 4).

For each table: the rows of 2025 (src 1..12, or the 2025 parts of a context table, or the old table's MVT_IDs for keyed tables) must equal
today's table column by column; the final rows (src 0) must be exactly the final_ranking DEP ids. Columns that differ on 2025 rows are listed
with the share of rows and the max abs diff. Memory-light: one column at a time from both files (pyarrow).
Copied read-only inputs (iters files, val/fwd caches, fit caches) can be checked byte for byte with --copies.
    python -X utf8 research/f2/regress_check.py --root <final root> [--tables feats_v25,ctx_v3] [--copies cache/iters_x.json,...]
Writes research/f2/out/regress_<root name>.json and prints a status line per table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
HERE = Path(__file__).resolve().parents[2]  # prc-2026
OUT = HERE / "research" / "f2" / "out"


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def col(path: Path, c: str) -> np.ndarray | pd.Series:
    t = pq.read_table(path, columns=[c]).column(c)
    s = t.to_pandas()
    return s


def table(po: Path, pn: Path, ids_final: np.ndarray) -> dict:
    so, sn = pq.read_schema(po), pq.read_schema(pn)
    co, cn = list(so.names), list(sn.names)
    r = {"sha_old": sha16(po), "sha_new": sha16(pn), "columns_same_order": co == cn, "cols_old_only": sorted(set(co) - set(cn)),
         "cols_new_only": sorted(set(cn) - set(co)), "dtype_changes": {c: [str(so.field(c).type), str(sn.field(c).type)] for c in co if c in cn and str(so.field(c).type) != str(sn.field(c).type)}}
    io, inn = col(po, "MVT_ID_mvt").to_numpy(dtype=float), col(pn, "MVT_ID_mvt").to_numpy(dtype=float)
    if "src" in co:
        ko = col(po, "src").to_numpy() != 0
        kn_src = col(pn, "src").to_numpy()
        te = np.sort(inn[kn_src == 0])
        r["test_rows_new"] = int(len(te))
        r["test_ids_equal_final_dep"] = bool(len(te) == len(ids_final) and np.array_equal(te, ids_final))
    elif "part" in co:
        po_part, pn_part = col(po, "part").astype(str).to_numpy(), col(pn, "part").astype(str).to_numpy()
        ko = pd.Series(po_part).str.startswith("2025").to_numpy()
        r["parts_new"] = sorted(set(pn_part.tolist()))
    else:  # keyed tables (ADS-B, geo): compare only rows whose MVT_ID is a 2025 departure (today's feats_v25 src != 0)
        f = pq.read_table(HERE / "cache" / "feats_v25.parquet", columns=["MVT_ID_mvt", "src"]).to_pandas()
        y2025 = set(f.loc[f["src"] != 0, "MVT_ID_mvt"].to_numpy(dtype=float).tolist())
        ko = np.array([x in y2025 for x in io])
        r["rows_old_total"], r["rows_new_total"] = int(len(io)), int(len(inn))
        r["rows_new_not_in_old"] = int(len(set(inn.tolist()) - set(io.tolist())))
    key_o = pd.Series(np.arange(len(io)), index=io)[ko]
    key_n = pd.Series(np.arange(len(inn)), index=inn)
    if not key_n.index.is_unique or not key_o.index.is_unique:
        r["status"] = "FAIL"
        r["why"] = "MVT_ID not unique"
        return r
    common = key_o.index.intersection(key_n.index)
    r["rows2025_old"], r["rows2025_missing_in_new"] = int(len(key_o)), int(len(key_o.index.difference(key_n.index)))
    po_ix, pn_ix = key_o.reindex(common).to_numpy(), key_n.reindex(common).to_numpy()
    diff = {}
    for c in [c for c in co if c in cn and c != "MVT_ID_mvt"]:
        a, b = col(po, c), col(pn, c)
        if not (pd.api.types.is_numeric_dtype(a.dtype) and pd.api.types.is_numeric_dtype(b.dtype)) or isinstance(a.dtype, pd.CategoricalDtype)                 or isinstance(b.dtype, pd.CategoricalDtype):
            ao, bo = a.astype(object).to_numpy()[po_ix], b.astype(object).to_numpy()[pn_ix]
            na_a, na_b = pd.isna(ao), pd.isna(bo)
            ne = (na_a != na_b) | (~na_a & ~na_b & (pd.Series(ao).astype(str).to_numpy() != pd.Series(bo).astype(str).to_numpy()))
            if ne.any():
                diff[c] = {"share": round(float(ne.mean()), 6), "rows": int(ne.sum()), "kind": "label", "dtypes": [str(a.dtype), str(b.dtype)]}
            continue
        av, bv = a.to_numpy(dtype=float, na_value=np.nan)[po_ix], b.to_numpy(dtype=float, na_value=np.nan)[pn_ix]
        both_nan = np.isnan(av) & np.isnan(bv)
        ne = ~both_nan & ~(av == bv)
        if ne.any():
            dd = np.abs(av - bv)[ne]
            diff[c] = {"share": round(float(ne.mean()), 6), "rows": int(ne.sum()), "max_abs": float(np.nanmax(dd)) if np.isfinite(dd).any() else None,
                       "nan_mismatch": int((np.isnan(av) ^ np.isnan(bv)).sum())}
    r["diff_2025"] = diff
    ok = r["rows2025_missing_in_new"] == 0 and not r["cols_old_only"] and r.get("test_ids_equal_final_dep", True)
    r["status"] = ("PASS" if not diff else "PASS_WITH_DIFFS") if ok else "FAIL"
    return r


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--root", required=True)
    pa.add_argument("--tables", default="feats_v25,feats_v27,ctx_v3,adsb_v1nd,geo_v1nd")
    pa.add_argument("--copies", default="")
    pa.add_argument("--out", default="", help="suffix of the output json (default: the root's name)")
    g = pa.parse_args()
    root = Path(g.root)
    d = pq.read_table(root / "data" / "ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt"]).to_pandas()
    ids_final = np.sort(d.loc[d["PHASE_mvt"] == "DEP", "MVT_ID_mvt"].to_numpy(dtype=float))
    rep = {"root": str(root), "final_dep_rows": int(len(ids_final)), "data_ranking_sha16": sha16(root / "data" / "ranking.parquet"), "tables": {}, "copies": {}}
    for t in [x for x in g.tables.split(",") if x]:
        po, pn = HERE / "cache" / f"{t}.parquet", root / "cache" / f"{t}.parquet"
        if not pn.exists():
            rep["tables"][t] = {"status": "MISSING"}
            print(f"{t}: MISSING in root", flush=True)
            continue
        r = table(po, pn, ids_final)
        rep["tables"][t] = r
        print(f"{t}: {r['status']}  sha {r['sha_old']} -> {r['sha_new']}  missing2025 {r.get('rows2025_missing_in_new')}  "
              f"diff cols {[(k, v.get('rows', v.get('share'))) for k, v in r.get('diff_2025', {}).items()]}"
              + (f"  test ids ok {r.get('test_ids_equal_final_dep')} ({r.get('test_rows_new')})" if "test_rows_new" in r else "")
              + (f"  parts {r.get('parts_new')}" if "parts_new" in r else ""), flush=True)
    for rel in [x for x in g.copies.split(",") if x]:
        po, pn = HERE / rel, root / rel
        e = {"old": sha16(po) if po.exists() else None, "new": sha16(pn) if pn.exists() else None}
        e["equal"] = e["old"] is not None and e["old"] == e["new"]
        rep["copies"][rel] = e
        print(f"copy {rel}: {'OK' if e['equal'] else 'DIFF'}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"regress_{root.name}{('_' + g.out) if g.out else ''}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
