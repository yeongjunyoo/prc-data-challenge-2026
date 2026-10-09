# SPDX-License-Identifier: GPL-3.0-only
"""Freeze the observation lane's six stand-geometry columns (the 321 line's extra ADS-B corrector columns) into cache/geo_v1.parquet.

Calls research/obs2/corr_plus.py geo_cols("filled") read-only (that lane's files are not modified) and keeps MVT_ID_mvt plus its NEW columns,
so src/ctx_corr.py --geo can check one sha16 the way src/adsb_corr.py checks cache/adsb_v1.parquet. The json records the shas of the
inputs the observation note lists (corr_plus.py, obs_geo.parquet, stand_map_filled.parquet).
    .venv/Scripts/python.exe -X utf8 research/ctx_corr/freeze_geo.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OBS = ROOT / "research" / "obs2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OBS))
from corr_plus import NEW, geo_cols  # noqa: E402


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main():
    want = {"corr_plus.py": "9d1a4acb4f107595", "obs_geo.parquet": "18f4db0ff9212aba", "stand_map_filled.parquet": "419317298abf4971"}
    got = {k: sha16(OBS / k) for k in want}
    if got != want:
        raise SystemExit(f"observation inputs differ from the note's shas: {got}")
    G = geo_cols("filled")[NEW].reset_index()
    assert G["MVT_ID_mvt"].is_unique
    p = ROOT / "cache" / "geo_v1.parquet"
    G.to_parquet(p, index=False)
    meta = {"name": "geo_v1", "file": "cache/geo_v1.parquet", "sha16": sha16(p), "rows": int(len(G)), "columns": NEW,
            "source": "research/obs2/corr_plus.py geo_cols('filled') (observation lane, 10-05; OSM parking_position ODbL 1.0 + adsb.lol ODbL 1.0)",
            "inputs_sha16": got, "fill": {c: round(float(G[c].notna().mean()), 4) for c in NEW}}
    (ROOT / "cache" / "geo_v1.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
