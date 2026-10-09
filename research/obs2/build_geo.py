# SPDX-License-Identifier: GPL-3.0-only
"""Freeze the six stand-geometry columns (research/obs2/corr_plus.geo_cols, filled stand map) as cache/<name>.parquet + .json
for src/adsb_geo.py. One row per PRC departure of the 124 route A days (689,260), keyed by MVT_ID_mvt. No labels are read.
Chain: obs_rows.py -> coverage.py -> osm_stands.py (OSM Overpass 2026-10-05, ODbL) -> this file.
Usage: .venv/Scripts/python.exe research/obs2/build_geo.py --name adsb_geo_v1
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from corr_plus import NEW, geo_cols  # noqa: E402


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="adsb_geo_v1")
    a = ap.parse_args()
    G = geo_cols("filled")[NEW].astype("float64").reset_index()
    assert len(G) == 689260 and G.MVT_ID_mvt.is_unique
    out = ROOT / "cache" / f"{a.name}.parquet"
    G.to_parquet(out, index=False)
    inputs = ["research/obs2/obs_rows.py", "research/obs2/coverage.py", "research/obs2/osm_stands.py", "research/obs2/corr_plus.py",
              "research/obs2/build_geo.py", "research/obs2/obs_geo.parquet", "research/obs2/stand_map.parquet",
              "research/obs2/stand_map_filled.parquet"] + sorted(p.relative_to(ROOT).as_posix() for p in (HERE / "osm").glob("*.json"))
    meta = dict(name=a.name, file=f"cache/{a.name}.parquet", sha16=sha16(out), rows=int(len(G)), columns=NEW,
                filled={c: int(G[c].notna().sum()) for c in NEW},
                source="adsb.lol globe_history (ODbL 1.0, via cache/adsb_v1 route A events) for first points and stand stops; "
                       "OpenStreetMap aeroway=parking_position via Overpass API, fetched 2026-10-05 (ODbL 1.0, (c) OpenStreetMap contributors) "
                       "for stands without an ADS-B position (LFPG excluded: OSM and ADS-B positions disagree)",
                licence="ODbL 1.0 (adsb.lol, OpenStreetMap)", labels_used=False,
                inputs={p: sha16(ROOT / p) for p in inputs},
                written_kst=dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST"))
    (ROOT / "cache" / f"{a.name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(meta["sha16"], meta["rows"], meta["filled"])


if __name__ == "__main__":
    main()
