# SPDX-License-Identifier: GPL-3.0-only
"""Inputs of the 361 line (10-06): the geometry corrector without the reception-density and gap inputs.

Copies cache/adsb_v1.parquet (sha 2bb0952d93192201) and cache/geo_v1.parquet (sha c088c658eb4b5cd3) and empties (all null) the
columns that describe how densely and with what gaps an aircraft was heard, the inputs that moved most between the 2025 ruler and the
2026 test (research/adsb/gprobe/g4_shift.py) and whose removal cost nothing on the ruler or January <-> July (g5_variants.py V5):
  adsb_v1nd: ev_n_pts, ev_to_gap, ev_max_gap_after_fm, ev_share_mlat
  geo_v1nd : to_gap (the same value as ev_to_gap)
Every other column and every row stay byte-for-byte the same values. A column with no values gives LightGBM nothing to split, so the
corrector (src/adsb_corr.py, src/ctx_corr.py, unchanged) runs as if the column were not there. Inputs only; no label is read.

  .venv/Scripts/python.exe -X utf8 research/adsb/gprobe/build_nd.py
Writes cache/adsb_v1nd.parquet, cache/geo_v1nd.parquet and cache/adsb_v1nd.json (sources, emptied columns, sha16 of each output).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
C = ROOT / "cache"
JOBS = [("adsb_v1", "2bb0952d93192201", "adsb_v1nd", ["ev_n_pts", "ev_to_gap", "ev_max_gap_after_fm", "ev_share_mlat"]),
        ("geo_v1", "c088c658eb4b5cd3", "geo_v1nd", ["to_gap"])]


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    rep = {}
    for src, want, dst, cols in JOBS:
        p = C / f"{src}.parquet"
        got = sha16(p)
        assert got == want, (src, got, want)
        t = pq.read_table(p)
        for c in cols:
            i = t.schema.get_field_index(c)
            assert i >= 0, (src, c)
            t = t.set_column(i, pa.field(c, pa.float64()), pa.nulls(t.num_rows, pa.float64()))
        out = C / f"{dst}.parquet"
        pq.write_table(t, out)
        rep[dst] = {"source": f"cache/{src}.parquet", "source_sha": want, "emptied": cols, "rows": t.num_rows, "sha16": sha16(out)}
        print(dst, rep[dst])
    (C / "adsb_v1nd.json").write_text(json.dumps({"script": "research/adsb/gprobe/build_nd.py", **rep}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
