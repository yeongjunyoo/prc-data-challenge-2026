# SPDX-License-Identifier: GPL-3.0-only
"""OSM (ODbL) aeroway=parking_position refs -> PRC STAND_mvt, validated against the empirical ADS-B stand map, then a filled map.

Inputs: research/obs2/osm/<ICAO>.json (Overpass, fetched 2026-10-05; data (c) OpenStreetMap contributors, ODbL 1.0),
        research/obs2/stand_map.parquet, research/obs2/obs_geo.parquet.
Output: research/obs2/stand_map_filled.parquet (ADEP_mvt, stand_id, lat, lon, src in {adsb, osm}) and a printed report.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def norm(s: str) -> str:
    s = re.sub(r"\s+", "", str(s).upper())
    m = re.match(r"^([A-Z]*)0*(\d+)([A-Z]*)$", s)
    return f"{m.group(1)}{int(m.group(2))}{m.group(3)}" if m else s


def osm_table() -> pd.DataFrame:
    rows = []
    for f in sorted((HERE / "osm").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for e in d["elements"]:
            t = e.get("tags", {})
            if t.get("aeroway") != "parking_position" or not t.get("ref"):
                continue
            la, lo = (e["lat"], e["lon"]) if e["type"] == "node" else (e["center"]["lat"], e["center"]["lon"])
            for r in re.split(r"[;,]", t["ref"]):
                rows.append((f.stem, norm(r), la, lo))
    o = pd.DataFrame(rows, columns=["ADEP_mvt", "key", "lat", "lon"])
    return o.groupby(["ADEP_mvt", "key"], as_index=False)[["lat", "lon"]].median()


def main():
    o = osm_table()
    sm = pd.read_parquet(HERE / "stand_map.parquet")
    sm["key"] = sm.stand_id.map(norm)
    g = pd.read_parquet(HERE / "obs_geo.parquet", columns=["day", "ADEP_mvt", "stand_id", "matched", "fm_ok", "st_lat"])
    g["key"] = g.stand_id.map(norm)
    v = sm[sm.ok].merge(o, on=["ADEP_mvt", "key"], suffixes=("", "_osm"))
    k = np.cos(np.radians(v.lat))
    v["err"] = np.hypot((v.lon - v.lon_osm) * 111200 * k, (v.lat - v.lat_osm) * 111200)
    print("validation (stands in both):")
    print(v.groupby("ADEP_mvt").err.describe(percentiles=[.5, .9]).round(0).to_string())
    T = g[g.day.str[:4] == "2026"]
    T = T.merge(o[["ADEP_mvt", "key"]].assign(in_osm=True), on=["ADEP_mvt", "key"], how="left")
    T["in_osm"] = T.in_osm.fillna(False).astype(bool)
    rep = T.groupby("ADEP_mvt").agg(rows=("key", "size"), osm_match=("in_osm", "mean"), adsb_map=("st_lat", lambda s: s.notna().mean()))
    rep["either"] = T.assign(e=T.in_osm | T.st_lat.notna()).groupby("ADEP_mvt").e.mean()
    m0 = T[T.matched & ~T.fm_ok]
    rep["m0_either"] = m0.assign(e=m0.in_osm | m0.st_lat.notna()).groupby("ADEP_mvt").e.mean()
    rep["m0_adsb"] = m0.groupby("ADEP_mvt").st_lat.apply(lambda s: s.notna().mean())
    print(rep.round(3).to_string())
    # filled map: empirical where ok, else OSM, but only for airports where OSM agrees (median error <= 60 m)
    good = v.groupby("ADEP_mvt").err.median()
    good = set(good[good <= 60].index)
    e = sm[sm.ok][["ADEP_mvt", "stand_id", "key", "lat", "lon"]].assign(src="adsb")
    keys = g[["ADEP_mvt", "stand_id", "key"]].drop_duplicates()
    oo = keys.merge(o, on=["ADEP_mvt", "key"])
    oo = oo[oo.ADEP_mvt.isin(good)]
    oo = oo[~oo.set_index(["ADEP_mvt", "stand_id"]).index.isin(e.set_index(["ADEP_mvt", "stand_id"]).index)].assign(src="osm")
    f = pd.concat([e, oo[["ADEP_mvt", "stand_id", "key", "lat", "lon", "src"]]], ignore_index=True)
    f.to_parquet(HERE / "stand_map_filled.parquet", index=False)
    print("airports where OSM agrees (median err <= 60 m):", sorted(good), "| filled stands:", f.src.value_counts().to_dict())


if __name__ == "__main__":
    main()
