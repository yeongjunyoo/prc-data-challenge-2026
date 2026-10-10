"""F2 lever 2, step 2 (vault research/2026-10-08-f2-four-months.md 1.9, registered 06:10:55 before any number): the corrector's stage-1 tables
with the Feb/Jun 2025 rows of the gate's fj bases (326,873 MVT_IDs). Inputs read only; the code paths are R17-A's and F3's, unchanged.
  adsb: R17-A's adsb_v3 (research/adsb/build_feats2.py, adsb_match.match_day, prc_all) rows of the fj MVT_IDs, build_nd.py's four columns emptied
        (pa.nulls float64). Regression first: adsb_v3 with the four columns emptied equals today's adsb_v1nd on all its rows (step 1 checked
        the same at 06:23; repeated here on the bytes this script reads).
  geo : research/f3/geo_final.py's route (copy research/f2/geo_final_f3.py, sha16 checked against final_run's): match_day on every departure of
        the day (prc_all, STAND_mvt from cache/all.parquet's 2025 departures), geo_cols_filled against the frozen research/obs2/stand_map_filled.parquet,
        to_gap emptied. Regression first: the same route on the 62 days of 2025-01 and 2025-07 must give today's geo_v1nd 2025 rows (same ids;
        st_known, seg_code, to_gap bit-equal; distance columns within 1e-6 m, F3's tolerance).
Outputs in f2_run/cache (refuses to overwrite):
  adsb_v1ndfj, geo_v1ndfj   = f2_run's adsb_v1nd (0d03ff06267e0dee) / geo_v1nd (43310b9bbeefe81b), every row unchanged, plus the new rows (gate tables)
  adsb_v1ndfjF, geo_v1ndfjF = final_run's adsb_v1nd (07df03ca69833082) / geo_v1nd (9dfa71b64e1099de) plus the same new rows (for F3's final rebuild)
and research/f2/out/fj_tables.json (sha16 of inputs and outputs, regressions, coverage by month x airport, seconds).
    PYTHONDONTWRITEBYTECODE=1 python -X utf8 -u research/f2/fj_tables.py      (from ~/nova-adsb/prcm/f2_run, 1 thread)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

T0 = time.time()
P = Path.home() / "nova-adsb" / "prcm"
RUN, FIN, SBX = P / "f2_run", P / "final_run", P / "r17a_sbx"
EV = Path.home() / "nova-adsb" / "r17a" / "events_all"
sys.path.insert(0, str(SBX / "research" / "adsb"))  # the copy that built adsb_v3 (f2_run has no research/adsb)
sys.path.insert(0, str(RUN / "research" / "f2"))
from adsb_match import match_day  # noqa: E402
import geo_final_f3 as gf  # noqa: E402

ND = ["ev_n_pts", "ev_to_gap", "ev_max_gap_after_fm", "ev_share_mlat"]
WANT = {"f2_run adsb_v1nd": "0d03ff06267e0dee", "f2_run geo_v1nd": "43310b9bbeefe81b", "final_run adsb_v1nd": "07df03ca69833082",
        "final_run geo_v1nd": "9dfa71b64e1099de", "adsb_v3": "816f0ffd9b7ae6a4"}
SM: Path = Path()
PRC_COLS = ["MVT_ID_mvt", "day", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt"]


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def empty_nd(t: pa.Table) -> pa.Table:
    for c in ND:  # as research/adsb/gprobe/build_nd.py
        i = t.schema.get_field_index(c)
        assert i >= 0, c
        t = t.set_column(i, pa.field(c, pa.float64()), pa.nulls(t.num_rows, pa.float64()))
    return t


def cols_equal(x: pa.Table, y: pa.Table) -> dict:
    """Column-wise equality (floats NaN-aware, null counts compared); {} when equal."""
    diff = {}
    if x.schema.names != y.schema.names:
        return {"columns": [x.schema.names, y.schema.names]}
    for c in x.schema.names:
        a, b = x.column(c), y.column(c)
        if a.type != b.type:
            diff[c] = f"type {a.type} vs {b.type}"
            continue
        if a.null_count != b.null_count:
            diff[c] = f"nulls {a.null_count} vs {b.null_count}"
            continue
        if pa.types.is_floating(a.type) or pa.types.is_integer(a.type) or pa.types.is_boolean(a.type):
            xa, ya = a.to_numpy(zero_copy_only=False).astype(float), b.to_numpy(zero_copy_only=False).astype(float)
            n = int((~((xa == ya) | (np.isnan(xa) & np.isnan(ya)))).sum())
        else:
            n = int(sum(1 for p, q in zip(a.to_pylist(), b.to_pylist()) if p != q))
        if n:
            diff[c] = n
    return diff


def write_new(t: pa.Table, name: str) -> str:
    p = RUN / "cache" / f"{name}.parquet"
    assert not p.exists(), f"{p} exists, refusing to overwrite"
    pq.write_table(t, p)
    return sha16(p)


def main():
    rep = {"what": __doc__.split("\n")[0], "registered": "vault note 1.9, 2026-10-10 06:10:55 KST"}
    paths = {"f2_run adsb_v1nd": RUN / "cache/adsb_v1nd.parquet", "f2_run geo_v1nd": RUN / "cache/geo_v1nd.parquet",
             "final_run adsb_v1nd": FIN / "cache/adsb_v1nd.parquet", "final_run geo_v1nd": FIN / "cache/geo_v1nd.parquet",
             "adsb_v3": SBX / "cache/adsb_v3.parquet"}
    got = {k: sha16(v) for k, v in paths.items()}
    rep["inputs_sha16"] = got
    bad = {k: (got[k], WANT[k]) for k in WANT if got[k] != WANT[k]}
    assert not bad, f"input sha16 differs: {bad}"
    code = {"adsb_match.py r17a_sbx (imported)": sha16(SBX / "research/adsb/adsb_match.py"),
            "geo_final_f3.py (this copy)": sha16(RUN / "research/f2/geo_final_f3.py"), "geo_final.py final_run": sha16(FIN / "research/f3/geo_final.py"),
            "prc_all r17a_sbx": sha16(SBX / "research/adsb/prc_all.parquet")}
    sms = [r / "research/obs2/stand_map_filled.parquet" for r in (RUN, FIN, SBX) if (r / "research/obs2/stand_map_filled.parquet").exists()]
    assert sms, "no stand_map_filled.parquet"
    code["stand_map_filled"] = {str(x): sha16(x) for x in sms}
    rep["code_sha16"] = code
    if (FIN / "research/adsb/adsb_match.py").exists():
        code["adsb_match.py final_run"] = sha16(FIN / "research/adsb/adsb_match.py")
        assert code["adsb_match.py final_run"] == code["adsb_match.py r17a_sbx (imported)"], "adsb_match.py differs from F3's root"
    assert code["geo_final_f3.py (this copy)"] == code["geo_final.py final_run"], "geo_final copy differs from F3's"
    assert len(set(code["stand_map_filled"].values())) == 1, "stand maps differ between roots"
    global SM
    SM = sms[0]
    rep["stand_source"] = {"file": str(RUN / "cache/all.parquet"), "sha16": sha16(RUN / "cache/all.parquet")}  # src/load.py's table of the raw files
    fj_ids = pq.read_table(RUN / "cache/fj_f2wincbrmfw4_s0.parquet", columns=["MVT_ID_mvt"]).column(0).to_numpy()
    assert len(np.unique(fj_ids)) == len(fj_ids) == 326873
    # ---- adsb ----
    old = pq.read_table(paths["f2_run adsb_v1nd"])
    fin = pq.read_table(paths["final_run adsb_v1nd"])
    assert old.schema.equals(fin.schema, check_metadata=False), "the two roots' adsb_v1nd schemas differ"
    v3old = pq.read_table(paths["adsb_v3"], filters=[("MVT_ID_mvt", "in", pa.array(old.column("MVT_ID_mvt").to_numpy()))])
    pos = pd.Index(v3old.column("MVT_ID_mvt").to_numpy()).get_indexer(old.column("MVT_ID_mvt").to_numpy())
    assert (pos >= 0).all()
    reg = cols_equal(old, empty_nd(v3old.take(pa.array(pos))).replace_schema_metadata(old.schema.metadata))
    rep["adsb_regression"] = {"rows": old.num_rows, "differences": reg, "pass": not reg}
    log(f"adsb regression on {old.num_rows:,} rows: {'pass' if not reg else reg}")
    assert not reg
    new = empty_nd(pq.read_table(paths["adsb_v3"], filters=[("MVT_ID_mvt", "in", pa.array(fj_ids))]))
    new = new.cast(old.schema.remove_metadata()).replace_schema_metadata(old.schema.metadata)
    nid = new.column("MVT_ID_mvt").to_numpy()
    assert new.num_rows == len(fj_ids) and set(nid) == set(fj_ids)
    assert not np.isin(nid, old.column("MVT_ID_mvt").to_numpy()).any() and not np.isin(nid, fin.column("MVT_ID_mvt").to_numpy()).any()
    tg = pa.concat_tables([old, new])
    tf = pa.concat_tables([fin, new.replace_schema_metadata(fin.schema.metadata)])
    out = {"adsb_v1ndfj": write_new(tg, "adsb_v1ndfj"), "adsb_v1ndfjF": write_new(tf, "adsb_v1ndfjF")}
    for name, base, n0 in (("adsb_v1ndfj", old, old.num_rows), ("adsb_v1ndfjF", fin, fin.num_rows)):
        back = pq.read_table(RUN / "cache" / f"{name}.parquet")
        assert len(np.unique(back.column("MVT_ID_mvt").to_numpy())) == back.num_rows
        d0 = cols_equal(base, back.slice(0, n0))
        d1 = cols_equal(new, back.slice(n0))
        assert not d0 and not d1, (name, d0, d1)
    nd = new.select(["day", "ADEP_mvt", "matched"]).to_pandas()
    nd["ym"] = nd["day"].astype(str).str[:7]
    rep["new_adsb_rows"] = {"rows": int(new.num_rows), "matched": int(nd["matched"].sum()),
                            "by_month": {m: [int(len(g)), int(g["matched"].sum())] for m, g in nd.groupby("ym")},
                            "matched_pct_by_month_airport": {f"{m} {a}": round(100 * float(g["matched"].mean()), 1) for (m, a), g in nd.groupby(["ym", "ADEP_mvt"])}}
    log(f"adsb tables written {out}; new rows {rep['new_adsb_rows']['by_month']}")
    # ---- geo ----
    prc = pq.read_table(SBX / "research/adsb/prc_all.parquet", columns=PRC_COLS).to_pandas()
    st = pq.read_table(RUN / "cache/all.parquet", columns=["MVT_ID_mvt", "src", "PHASE_mvt", "STAND_mvt"]).to_pandas()
    st = st[(st["src"].to_numpy() > 0) & (st["PHASE_mvt"].astype(str).to_numpy() == "DEP")]
    assert st["MVT_ID_mvt"].is_unique
    st = pd.Series(st["STAND_mvt"].astype(object).to_numpy(), index=st["MVT_ID_mvt"].to_numpy(dtype=float))  # strings or NaN, as the raw files
    prc["STAND_mvt"] = st.reindex(prc["MVT_ID_mvt"].to_numpy(dtype=float)).to_numpy()
    rep["stand_missing_on_fj_rows"] = int(pd.isna(prc.loc[prc["MVT_ID_mvt"].isin(set(fj_ids)), "STAND_mvt"]).sum())
    sm = pd.read_parquet(SM)

    def obs(days):
        parts = []
        for d in days:
            ev = pd.read_parquet(EV / d / "events.parquet")
            m = match_day(prc[prc["day"] == d].copy(), ev)
            m["stand_id"] = m["STAND_mvt"].fillna("").astype(str).str.strip()  # as research/f3/geo_final.py
            parts.append(m[gf.OBS])
        return pd.concat(parts, ignore_index=True)

    g_old = pq.read_table(paths["f2_run geo_v1nd"])
    g_fin = pq.read_table(paths["final_run geo_v1nd"])
    days_jj = sorted(x.name for x in EV.iterdir() if x.name[:7] in ("2025-01", "2025-07"))
    assert len(days_jj) == 62, len(days_jj)
    O = obs(days_jj)
    G = gf.geo_cols_filled(O, sm)
    G["to_gap"] = np.nan
    ref = g_old.to_pandas().set_index("MVT_ID_mvt")
    ids25 = old.select(["MVT_ID_mvt", "day"]).to_pandas()
    ids25 = set(ids25.loc[ids25["day"].astype(str) < "2026", "MVT_ID_mvt"])
    Gi = G.set_index("MVT_ID_mvt")
    greg = {"rows_recomputed": int(len(G)), "rows_2025_today": int(len(ids25)), "same_ids": bool(set(Gi.index) == ids25)}
    if greg["same_ids"]:
        R0 = ref.loc[Gi.index]
        for c in gf.NEW:
            x, y = Gi[c].to_numpy(float), R0[c].to_numpy(float)
            nan_ok = bool((np.isnan(x) == np.isnan(y)).all())
            mx = float(np.nanmax(np.abs(x - y))) if (~np.isnan(x)).any() else 0.0
            greg[c] = {"nan_pattern_equal": nan_ok, "bits_equal": bool(np.array_equal(x, y, equal_nan=True)), "max_abs": mx}
        greg["pass"] = all(greg[c]["nan_pattern_equal"] and greg[c]["max_abs"] <= 1e-6 for c in gf.NEW) and all(
            greg[c]["bits_equal"] for c in ("st_known", "seg_code", "to_gap"))
    else:
        greg["pass"] = False
    rep["geo_regression"] = greg
    log(f"geo regression on {len(days_jj)} days: {greg}")
    assert greg["pass"]
    days_fj = sorted(set(new.column("day").to_pylist()))
    O1 = obs(days_fj)
    O1 = O1[O1["MVT_ID_mvt"].isin(set(fj_ids))]
    assert O1["MVT_ID_mvt"].is_unique and set(O1["MVT_ID_mvt"]) == set(fj_ids)
    G1 = gf.geo_cols_filled(O1, sm)
    G1["to_gap"] = np.nan
    order = pd.Index(G1["MVT_ID_mvt"].to_numpy()).get_indexer(nid)  # the new adsb rows' order
    G1 = G1.iloc[order].reset_index(drop=True)
    g_new = pa.Table.from_pandas(G1[g_old.schema.names], preserve_index=False).cast(g_old.schema.remove_metadata()).replace_schema_metadata(g_old.schema.metadata)
    out["geo_v1ndfj"] = write_new(pa.concat_tables([g_old, g_new]), "geo_v1ndfj")
    out["geo_v1ndfjF"] = write_new(pa.concat_tables([g_fin, g_new.replace_schema_metadata(g_fin.schema.metadata)]), "geo_v1ndfjF")
    for name, base in (("geo_v1ndfj", g_old), ("geo_v1ndfjF", g_fin)):
        back = pq.read_table(RUN / "cache" / f"{name}.parquet")
        assert len(np.unique(back.column("MVT_ID_mvt").to_numpy())) == back.num_rows
        d0, d1 = cols_equal(base, back.slice(0, base.num_rows)), cols_equal(g_new, back.slice(base.num_rows))
        assert not d0 and not d1, (name, d0, d1)
    rep["new_geo_rows"] = {"rows": int(len(G1)), "filled": {c: round(float(G1[c].notna().mean()), 4) for c in gf.NEW}}
    rep["outputs_sha16"] = out
    rep["rows"] = {"adsb_v1ndfj": tg.num_rows, "adsb_v1ndfjF": tf.num_rows, "geo_v1ndfj": g_old.num_rows + g_new.num_rows,
                   "geo_v1ndfjF": g_fin.num_rows + g_new.num_rows}
    rep["seconds"] = round(time.time() - T0)
    o = RUN / "research/f2/out/fj_tables.json"
    o.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"TABLES done {out} -> {o} sha16 {sha16(o)}")


if __name__ == "__main__":
    main()
