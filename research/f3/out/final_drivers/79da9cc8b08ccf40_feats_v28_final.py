"""F3 L3 final (coordinator 10-10 03:5x): the final root's feats_v28, built the way the final feats_v27 was, with R17-A's filled 2025 days.
  1. cache/adsb_v3.parquet = r17a_sbx's adsb_v3 rows with day < 2026-01-01 (2025: R17-A's 427-day table minus its 2026 days) plus this root's
     adsb_v2 rows with day >= 2026-01-01 (R17-A pack 2, the four test months); the merge rule of research/r17_a/merge_final_r17a.py (dtype
     alignment on the 2026 table, sort by day then MVT_ID_mvt, MVT_ID_mvt unique). Refuses if it exists.
  2. python -m src.feats_adsb feats_v28 --adsb adsb_v3 --mode raw (unchanged; this root's feats_v25 and all.parquet).
  3. Checks, column by column (exit 1 on any failure), report research/f3/out/feats_v28_final.json:
     a. against this root's feats_v27 (the final one, 8fc36fd4314708a8): same columns, rows and order; every non-ab_ cell equal; the ab_ cells
        equal on every 2026 row (src == 0) and on every 2025 row that had ADS-B day data in feats_v27; the rows that differ are 2025 rows of the
        days R17-A filled (listed by month);
     b. against r17a_sbx's feats_v28 (the public-phase one, b870eefe01d5c95e): its 2025 rows (src != 0) equal this root's in every column,
        except columns whose 2025 cells this root's base table already has differently (restated 10-10 04:3x after the first run: the final
        feats_v25 differs from the public one on 2025 rows in day_daobt_rel and d_aobt_mode_dev, columns computed over all rows including the
        test months, so the final feats_v27 has the same differences). Such a column must not be an ab_ column, and in it this root's feats_v28
        must equal this root's feats_v25 and r17a's feats_v28 must equal r17a's feats_v25 on the 2025 rows (the difference is wholly inherited).
    python -X utf8 research/f3/feats_v28_final.py        (cwd = the final root)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

R17 = Path.home() / "nova-adsb" / "prcm" / "r17a_sbx" / "cache"
C = Path("cache")
WANT = {"feats_v27": "8fc36fd4314708a8", "r17a_feats_v28": "b870eefe01d5c95e", "r17a_adsb_v3": "816f0ffd9b7ae6a4"}


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def same(a, b) -> bool:
    """research/r17_a/check_feats_r17a.py's rule: equal values in the same places, a missing value equal to a missing value."""
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
    t0 = time.time()
    rep = {"what": "final root feats_v28 (research/f3/feats_v28_final.py)", "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    for k, p in (("feats_v27", C / "feats_v27.parquet"), ("r17a_feats_v28", R17 / "feats_v28.parquet"), ("r17a_adsb_v3", R17 / "adsb_v3.parquet")):
        got = sha16(p)
        rep[f"{k}_sha16"] = got
        if got != WANT[k]:
            raise SystemExit(f"{p}: sha16 {got} != {WANT[k]}")
    out = C / "adsb_v3.parquet"
    if out.exists() or (C / "feats_v28.parquet").exists():
        raise SystemExit("cache/adsb_v3.parquet or cache/feats_v28.parquet exists")
    old, fin = pd.read_parquet(R17 / "adsb_v3.parquet"), pd.read_parquet(C / "adsb_v2.parquet")
    if list(old.columns) != list(fin.columns):
        raise SystemExit(f"columns differ: {set(old.columns) ^ set(fin.columns)}")
    keep, new = old[old["day"] < "2026-01-01"], fin[fin["day"] >= "2026-01-01"]
    for c in old.columns:
        if keep[c].dtype != new[c].dtype and new[c].dtype == object:
            keep = keep.assign(**{c: keep[c].astype(object)})
    t = pd.concat([keep, new], ignore_index=True).sort_values(["day", "MVT_ID_mvt"], kind="stable").reset_index(drop=True)
    assert t["MVT_ID_mvt"].is_unique, "MVT_ID_mvt repeats"
    t.to_parquet(out, index=False)
    rep["adsb_v3"] = {"rows_2025_from_r17a": int(len(keep)), "rows_2026_from_final_adsb_v2": int(len(new)), "rows": int(len(t)),
                      "days_2025": int(keep["day"].nunique()), "days_2026": int(new["day"].nunique()),
                      "rows_by_month": t["day"].str[:7].value_counts().sort_index().to_dict(), "sha16": sha16(out),
                      "final_adsb_v2_sha16": sha16(C / "adsb_v2.parquet")}
    del old, fin, keep, new, t
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "src.feats_adsb", "feats_v28", "--adsb", "adsb_v3", "--mode", "raw"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("src.feats_adsb failed: " + r.stdout[-1500:] + r.stderr[-1500:])
    rep["feats_v28_sha16"] = sha16(C / "feats_v28.parquet")
    # a. against the final feats_v27
    f27, f28 = pq.ParquetFile(C / "feats_v27.parquet"), pq.ParquetFile(C / "feats_v28.parquet")
    n27, n28 = f27.schema_arrow.names, f28.schema_arrow.names
    fails = []
    if n27 != n28:
        fails.append(f"column lists differ: {sorted(set(n27) ^ set(n28))}")
    src = f27.read(columns=["src"]).column("src").to_numpy()
    if not np.array_equal(src, f28.read(columns=["src"]).column("src").to_numpy()) or not same(
            f27.read(columns=["MVT_ID_mvt"]).column(0).to_numpy(), f28.read(columns=["MVT_ID_mvt"]).column(0).to_numpy()):
        fails.append("rows or row order differ from feats_v27")
    had = ~np.isnan(f27.read(columns=["ab_matched"]).column(0).to_numpy(zero_copy_only=False).astype(float))
    test = src == 0
    changed = np.zeros(len(src), bool)
    rep["a_columns_differing_where_they_must_not"] = []
    for c in n27:
        a, b = f27.read(columns=[c]).column(c), f28.read(columns=[c]).column(c)
        if a.type != b.type:
            fails.append(f"{c}: type {a.type} vs {b.type}")
            continue
        av, bv = a.to_numpy(zero_copy_only=False), b.to_numpy(zero_copy_only=False)
        must = (had | test) if c.startswith("ab_") else np.ones(len(src), bool)
        if not same(av[must], bv[must]):
            rep["a_columns_differing_where_they_must_not"].append(c)
        if c.startswith("ab_"):
            na, nb = pd.isna(av), pd.isna(bv)
            ne = na != nb
            both = ~na & ~nb
            if both.any():
                ne[both] |= av[both].astype(float) != bv[both].astype(float)
            changed |= ne
    if rep["a_columns_differing_where_they_must_not"]:
        fails.append(f"cells differ outside the 2025 rows without day data: {rep['a_columns_differing_where_they_must_not']}")
    if (changed & test).any() or (changed & had).any():
        fails.append("changed rows include 2026 rows or rows that had day data")
    tm = pd.read_parquet(C / "all.parquet", columns=["MVT_ID_mvt", "t_mvt"]).set_index("MVT_ID_mvt")["t_mvt"]
    ids = f27.read(columns=["MVT_ID_mvt"]).column(0).to_numpy()
    days = pd.to_datetime(tm.reindex(ids[changed]).to_numpy(), unit="s", utc=True).strftime("%Y-%m-%d")  # all.parquet t_mvt is seconds (src/feats_adsb.py)
    rep["a_changed_rows"] = int(changed.sum())
    rep["a_changed_rows_by_month"] = pd.Series(days).str[:7].value_counts().sort_index().to_dict()
    rep["a_changed_days"] = int(pd.Series(days).nunique())
    rep["a_rows_2025_without_day_data_v27"] = int((~had & ~test).sum())
    rep["a_rows_2025_with_day_data_v28"] = int((~test & ~np.isnan(f28.read(columns=["ab_matched"]).column(0).to_numpy(zero_copy_only=False).astype(float))).sum())
    # b. 2025 rows against r17a_sbx's feats_v28
    g = pq.ParquetFile(R17 / "feats_v28.parquet")
    if g.schema_arrow.names != n28:
        fails.append("r17a feats_v28 columns differ")
    gsrc = g.read(columns=["src"]).column("src").to_numpy()
    gid = g.read(columns=["MVT_ID_mvt"]).column(0).to_numpy()[gsrc != 0]
    if not np.array_equal(gid, ids[~test]):
        fails.append("2025 rows: ids or order differ from r17a feats_v28")
    else:
        rep["b_columns_differing"] = []
        for c in n28:
            a, b = f28.read(columns=[c]).column(c), g.read(columns=[c]).column(c)
            if a.type != b.type or not same(a.to_numpy(zero_copy_only=False)[~test], b.to_numpy(zero_copy_only=False)[gsrc != 0]):
                rep["b_columns_differing"].append(c)
        # a column may differ only where both tables carry their own base table's (feats_v25's) 2025 cells unchanged
        f25, g25 = pq.ParquetFile(C / "feats_v25.parquet"), pq.ParquetFile(R17 / "feats_v25.parquet")
        rep["feats_v25_sha16"], rep["r17a_feats_v25_sha16"] = sha16(C / "feats_v25.parquet"), sha16(R17 / "feats_v25.parquet")
        s25, gs25 = f25.read(columns=["src"]).column(0).to_numpy(), g25.read(columns=["src"]).column(0).to_numpy()
        if not (np.array_equal(f25.read(columns=["MVT_ID_mvt"]).column(0).to_numpy()[s25 != 0], ids[~test])
                and np.array_equal(g25.read(columns=["MVT_ID_mvt"]).column(0).to_numpy()[gs25 != 0], gid)):
            fails.append("2025 rows of a feats_v25 differ in ids or order")
        else:
            rep["b_inherited_from_feats_v25"], bad = {}, []
            for c in rep["b_columns_differing"]:
                if c.startswith("ab_") or c not in f25.schema_arrow.names or c not in g25.schema_arrow.names:
                    bad.append(c)
                    continue
                own = same(f28.read(columns=[c]).column(c).to_numpy(zero_copy_only=False)[~test], f25.read(columns=[c]).column(c).to_numpy(zero_copy_only=False)[s25 != 0])
                pub = same(g.read(columns=[c]).column(c).to_numpy(zero_copy_only=False)[gsrc != 0], g25.read(columns=[c]).column(c).to_numpy(zero_copy_only=False)[gs25 != 0])
                x = pd.Series(f28.read(columns=[c]).column(c).to_numpy(zero_copy_only=False)[~test])
                y = pd.Series(g.read(columns=[c]).column(c).to_numpy(zero_copy_only=False)[gsrc != 0])
                rep["b_inherited_from_feats_v25"][c] = {"this_v28_equals_this_v25": own, "r17a_v28_equals_r17a_v25": pub,
                                                        "rows_differing": int((~((x == y) | (x.isna() & y.isna()))).sum())}
                if not (own and pub):
                    bad.append(c)
            if bad:
                fails.append(f"2025 rows differ from r17a feats_v28 in {bad}, not inherited from the feats_v25 base tables")
        rep["b_rows_compared"] = int((~test).sum())
    rep["fails"] = fails
    rep["pass"] = not fails
    rep["seconds"] = round(time.time() - t0, 1)
    Path("research/f3/out/feats_v28_final.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
