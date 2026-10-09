# SPDX-License-Identifier: GPL-3.0-only
"""F3 final root: cache/geo_v1nd.parquet for the final rebuild (stand-geometry columns of the corrector, research/obs2 chain).
2025 rows: today's geo_v1nd rows (byte-identical values, same order). 2026 rows (all four months): observation rows made as
research/obs2/obs_rows.py makes them (research/adsb/adsb_match.match_day, unchanged, on R17-A's final package: prc_final.parquet +
prc_final_stand.parquet and events/<day>/events.parquet), then the six columns exactly as research/obs2/corr_plus.geo_cols("filled")
computes them against the FROZEN filled stand map (research/obs2/stand_map_filled.parquet, 10-05; nothing re-estimated), then to_gap
emptied as research/adsb/gprobe/build_nd.py does for geo_v1nd. The pack-2 no-data days (manifest pack2.no_data_days, 2026-06-11) are left
out, as R17-A's tables leave them out (a day without ADS-B: no row, NaN downstream).
Regression check first: the copy of geo_cols below, run on today's research/obs2/obs_geo.parquet, must give today's adsb_geo_v1: same ids,
discrete columns bit-equal, distance columns within 1e-6 m (float rounding of this machine's trigonometry; tolerance set 02:1x after the
first, exact check showed differences of at most 1.8e-12 m). The 2025 rows of geo_v1nd are today's bytes, not recomputed.
    python -X utf8 research/f3/geo_final.py <today's cache dir> <today's obs_geo.parquet> <R17-A final dir>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "research" / "adsb"))
from adsb_match import match_day  # noqa: E402

NEW = ["d_stand_first", "st_known", "d_stand_stop", "d_any_stand", "seg_code", "to_gap"]
OBS = ["MVT_ID_mvt", "day", "ADEP_mvt", "stand_id", "matched", "fm_ok", "ev_first_lat", "ev_first_lon", "ev_stand_lat", "ev_stand_lon",
       "ev_seg_reason", "ev_to_gap"]


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def geo_cols_filled(g: pd.DataFrame, sm: pd.DataFrame) -> pd.DataFrame:
    """research/obs2/corr_plus.geo_cols('filled') with the observation frame passed in (same statements, same order)."""
    g = g[OBS].copy()
    g = g.join(sm.set_index(["ADEP_mvt", "stand_id"])[["lat", "lon"]].rename(columns={"lat": "st_lat", "lon": "st_lon"}), on=["ADEP_mvt", "stand_id"])
    k0 = np.cos(np.radians(g.st_lat.fillna(0)))
    g["d_stand_first"] = np.hypot((g.ev_first_lon - g.st_lon) * 111200 * k0, (g.ev_first_lat - g.st_lat) * 111200)
    g["d_any_stand"] = np.nan
    for ap, s in sm.groupby("ADEP_mvt"):
        la0 = s.lat.mean()
        c = np.cos(np.radians(la0))
        tree = cKDTree(np.c_[s.lon * 111200 * c, s.lat * 111200])
        sel = (g.ADEP_mvt == ap) & g.ev_first_lat.notna()
        d, _ = tree.query(np.c_[g.loc[sel, "ev_first_lon"] * 111200 * c, g.loc[sel, "ev_first_lat"] * 111200])
        g.loc[sel, "d_any_stand"] = d
    g["st_known"] = g.st_lat.notna().astype(float)
    kk = np.cos(np.radians(g.st_lat.fillna(0)))
    g["d_stand_stop"] = np.hypot((g.ev_stand_lon - g.st_lon) * 111200 * kk, (g.ev_stand_lat - g.st_lat) * 111200)
    g["seg_code"] = g.ev_seg_reason.map({"trace_start": 0.0, "gap_cut": 1.0, "landing": 2.0})
    g["to_gap"] = g.ev_to_gap
    return g.set_index("MVT_ID_mvt")[NEW].astype("float64").reset_index()


def main():
    today, obs_geo, fin = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rep = {}
    sm = pd.read_parquet(ROOT / "research" / "obs2" / "stand_map_filled.parquet")
    rep["stand_map_filled_sha16"] = sha16(ROOT / "research" / "obs2" / "stand_map_filled.parquet")
    # 1) regression: today's obs_geo -> today's adsb_geo_v1
    G0 = geo_cols_filled(pd.read_parquet(obs_geo), sm)
    T0 = pd.read_parquet(today / "adsb_geo_v1.parquet")
    # discrete columns bit-equal, distance columns equal up to float rounding (first run 02:12: d_stand_first, d_stand_stop, d_any_stand
    # differ on 28,827, 5,135 and 36,093 of 689,260 rows by at most 1.8e-12 m, the last bits of the trigonometry on this machine)
    ids_ok = len(G0) == len(T0) and bool((G0["MVT_ID_mvt"].to_numpy() == T0["MVT_ID_mvt"].to_numpy()).all())
    reg = {}
    for c in NEW:
        x, y = G0[c].to_numpy(float), T0[c].to_numpy(float)
        reg[c] = {"nan_pattern_equal": bool((np.isnan(x) == np.isnan(y)).all()), "bits_equal": bool(np.array_equal(x, y, equal_nan=True)),
                  "max_abs": float(np.nanmax(np.abs(x - y))) if (~np.isnan(x)).any() else 0.0}
    same = ids_ok and all(r["nan_pattern_equal"] and r["max_abs"] <= 1e-6 for r in reg.values()) and all(
        reg[c]["bits_equal"] for c in ("st_known", "seg_code", "to_gap"))
    rep["regression_obs_geo_to_adsb_geo_v1"] = {"pass": bool(same), "columns": reg}
    assert same, "copy of geo_cols does not reproduce today's adsb_geo_v1"
    # 2) final observation rows (obs_rows.py logic on R17-A's inputs)
    prc = pd.read_parquet(fin / "prc_final.parquet")
    st = pd.read_parquet(fin / "prc_final_stand.parquet")
    if "STAND_mvt" not in prc.columns:
        prc = prc.merge(st[["MVT_ID_mvt", "STAND_mvt"]], on="MVT_ID_mvt", how="left")
    # pack 2 (R17-A 02:07, 02:1x): a short source day reads as a day without ADS-B, so its rows are dropped afterwards
    no_data = set(json.loads((fin / "manifest.json").read_text(encoding="utf-8")).get("pack2", {}).get("no_data_days", []))
    rep["no_data_days_dropped"] = sorted(no_data)
    parts = []
    for dd in sorted((fin / "events").iterdir()):
        if not (dd / "events.parquet").exists() or dd.name in no_data:
            continue
        ev = pd.read_parquet(dd / "events.parquet")
        m = match_day(prc[prc["day"] == dd.name].copy(), ev)
        m["stand_id"] = m["STAND_mvt"].fillna("").astype(str).str.strip()
        parts.append(m[OBS])
    O = pd.concat(parts, ignore_index=True)
    assert O["MVT_ID_mvt"].is_unique
    rep["obs_rows_2026"] = int(len(O))
    rep["obs_by_month"] = O["day"].str[:7].value_counts().sort_index().to_dict()
    G1 = geo_cols_filled(O, sm)
    # 3) Jan/Jul 2026 against today's geo (ids in both)
    T26 = T0.set_index("MVT_ID_mvt")
    N26 = G1.set_index("MVT_ID_mvt")
    common = T26.index.intersection(N26.index)
    rep["jan_jul_common"] = int(len(common))
    rep["jan_jul_rows_differing"] = {c: int((~((T26.loc[common, c].to_numpy() == N26.loc[common, c].to_numpy())
                                                | (np.isnan(T26.loc[common, c].to_numpy()) & np.isnan(N26.loc[common, c].to_numpy())))).sum()) for c in NEW}
    # 4) geo_v1nd = today's 2025 rows + new 2026 rows with to_gap emptied
    old = pd.read_parquet(today / "geo_v1nd.parquet")
    assert sha16(today / "geo_v1nd.parquet") == "43310b9bbeefe81b"
    days25 = set(pd.read_parquet(today / "adsb_v1nd.parquet", columns=["MVT_ID_mvt", "day"]).query("day < '2026'")["MVT_ID_mvt"])
    keep = old["MVT_ID_mvt"].isin(days25).to_numpy()
    G1["to_gap"] = np.nan
    out = pd.concat([old[keep], G1[old.columns]], ignore_index=True)
    assert out["MVT_ID_mvt"].is_unique and list(out.columns) == list(old.columns)
    p = Path("cache/geo_v1nd.parquet")
    assert not p.exists()
    out.to_parquet(p, index=False)
    rep.update({"rows": int(len(out)), "rows_2025": int(keep.sum()), "rows_2026": int(len(G1)), "sha16": sha16(p),
                "fill_2026": {c: round(float(G1[c].notna().mean()), 4) for c in NEW}})
    Path("research/f3/out/geo_final.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
