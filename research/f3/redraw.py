# SPDX-License-Identifier: GPL-3.0-only
"""F3 confirmation sampler: re-draw the organizer's timestamp jitter on 2025 rows the way the final file re-drew it on 2026 Jan/Jul.

The final file is a fresh draw of the jitter every file already carries (research/f3/jit_*.py): new = true + n, old = true + o. The change
d = n - o therefore depends on where the old value sits within its minute (at whole-minute airports the old second gives o). So d is drawn
from the empirical P(d | phase, airport, old second of MVT) of the 2026 Jan/Jul pairs (ranking.parquet vs final_ranking.parquet head),
falling back to P(d | phase, old second) for a cell with fewer than 30 pairs. ARR rows draw the (dM, dB) pair jointly from the ARR cell, so
the ARR BLOCK moves opposite to MVT as in the final file, and TAXITIME = BLOCK - MVT is recomputed. DEP rows draw dM from the DEP cell; the
hidden DEP BLOCK change for the second truth variant is dB = -dM + eta, eta = (dB + dM) of an ARR pair drawn from the same airport and old
second (the anti-correlated jitter the training labels show: research/f3/out/jit_redraw.json).
Writes, for each draw k: <out>/draw<k>/training_2025-01-01_2025-02-01.parquet and training_2025-07-01_2025-08-01.parquet (same schema,
only MVT_TIME, ARR BLOCK_TIME and ARR TAXITIME changed) and <out>/draw<k>/deltas.parquet (MVT_ID, phase, dM, dB, y_old, y_b).
    python -X utf8 research/f3/redraw.py --old data/ranking.parquet --new data/final_phase/final_ranking.parquet --data data --out <dir> --draws 1,2,3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

MONTHS = ["training_2025-01-01_2025-02-01.parquet", "training_2025-07-01_2025-08-01.parquet"]
MIN_CELL = 30


def secs(col: pa.ChunkedArray) -> np.ndarray:
    return np.asarray(col.cast(pa.int64()).to_numpy(zero_copy_only=False), dtype="float64") / 1e6


def pairs(old: str, new: str) -> pd.DataFrame:
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]
    a = pq.read_table(old, columns=cols)
    b = pq.read_table(new, columns=cols).slice(0, a.num_rows)
    assert (np.asarray(a.column("MVT_ID_mvt").to_numpy()) == np.asarray(b.column("MVT_ID_mvt").to_numpy())).all()
    ph = a.column("PHASE_mvt").to_pandas().astype(str).to_numpy()
    ap = np.where(ph == "DEP", a.column("ADEP_mvt").to_pandas().astype(str), a.column("ADES_mvt").to_pandas().astype(str))
    m0, m1 = secs(a.column("MVT_TIME_UTC_mvt")), secs(b.column("MVT_TIME_UTC_mvt"))
    b0, b1 = secs(a.column("BLOCK_TIME_UTC_mvt")), secs(b.column("BLOCK_TIME_UTC_mvt"))
    return pd.DataFrame({"ph": ph, "ap": ap, "s": np.mod(m0, 60).astype(int), "dM": m1 - m0, "dB": b1 - b0})


class Pool:
    def __init__(self, P: pd.DataFrame, ph: str):
        q = P[P["ph"] == ph].reset_index(drop=True)
        self.dM, self.dB = q["dM"].to_numpy(), q["dB"].to_numpy()
        self.cell = {k: g.index.to_numpy() for k, g in q.groupby(["ap", "s"])}
        self.sec = {k: g.index.to_numpy() for k, g in q.groupby("s")}

    def draw(self, ap: np.ndarray, s: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
        idx = np.empty(len(ap), dtype=np.int64)
        fb = 0
        key = pd.MultiIndex.from_arrays([ap, s])
        for (a, ss), rows in pd.Series(np.arange(len(ap))).groupby(key):
            pool = self.cell.get((a, ss))
            if pool is None or len(pool) < MIN_CELL:
                pool = self.sec[ss]
                fb += len(rows)
            idx[rows.to_numpy()] = pool[rng.integers(0, len(pool), len(rows))]
        return self.dM[idx], self.dB[idx], fb


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--old", required=True)
    ap_.add_argument("--new", required=True)
    ap_.add_argument("--data", required=True)
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--draws", default="1,2,3")
    a = ap_.parse_args()
    P = pairs(a.old, a.new)
    pd_, pa_ = Pool(P, "DEP"), Pool(P, "ARR")
    rep = {}
    for k in [int(x) for x in a.draws.split(",")]:
        rng = np.random.default_rng(20261008 + k)
        od = Path(a.out) / f"draw{k}"
        od.mkdir(parents=True, exist_ok=False)
        side = []
        for name in MONTHS:
            t = pq.read_table(Path(a.data) / name)
            ph = t.column("PHASE_mvt").to_pandas().astype(str).to_numpy()
            ap = np.where(ph == "DEP", t.column("ADEP_mvt").to_pandas().astype(str), t.column("ADES_mvt").to_pandas().astype(str))
            m0, b0 = secs(t.column("MVT_TIME_UTC_mvt")), secs(t.column("BLOCK_TIME_UTC_mvt"))
            s = np.mod(m0, 60).astype(int)
            dM, dB = np.zeros(len(ph)), np.zeros(len(ph))
            dep, arr = ph == "DEP", ph == "ARR"
            dM[dep], _, fb_d = pd_.draw(ap[dep], s[dep], rng)
            eM, eB, _ = pa_.draw(ap[dep], s[dep], rng)  # eta for the DEP hidden BLOCK (variant b)
            dB[dep] = -dM[dep] + (eB + eM)
            dM[arr], dB[arr], fb_a = pa_.draw(ap[arr], s[arr], rng)
            m1 = m0 + dM
            b1 = np.where(arr, b0 + dB, b0)  # DEP BLOCK (the label side) is left as is in the input file
            y0 = np.asarray(t.column("TAXITIME_SEC_mvt").to_numpy(zero_copy_only=False), dtype="float64")
            tx1 = np.where(arr, b1 - m1, y0)
            assert np.all(np.isfinite(m1))
            us = lambda v: pa.array(np.round(v * 1e6).astype(np.int64), type=pa.int64()).cast(pa.timestamp("us", tz="UTC"))  # noqa: E731
            t2 = t.set_column(t.schema.get_field_index("MVT_TIME_UTC_mvt"), t.schema.field("MVT_TIME_UTC_mvt"), us(m1))
            bmask = np.isfinite(b1)
            barr = pa.array(np.where(bmask, np.round(np.nan_to_num(b1) * 1e6), 0).astype(np.int64), type=pa.int64(), mask=~bmask).cast(pa.timestamp("us", tz="UTC"))
            t2 = t2.set_column(t2.schema.get_field_index("BLOCK_TIME_UTC_mvt"), t2.schema.field("BLOCK_TIME_UTC_mvt"), barr)
            tmask = np.isfinite(tx1)
            tarr = pa.array(np.where(tmask, np.nan_to_num(tx1), 0).astype(np.int32), type=pa.int32(), mask=~tmask)
            t2 = t2.set_column(t2.schema.get_field_index("TAXITIME_SEC_mvt"), t2.schema.field("TAXITIME_SEC_mvt"), tarr)
            assert t2.schema.equals(t.schema, check_metadata=False)
            pq.write_table(t2, od / name)
            ids = np.asarray(t.column("MVT_ID_mvt").to_numpy(zero_copy_only=False), dtype="float64")
            side.append(pd.DataFrame({"MVT_ID_mvt": ids, "phase": ph, "dM": dM, "dB": dB, "y_old": y0, "y_b": np.where(dep, y0 + dM - dB, y0)}))
            rep[f"draw{k}|{name[9:16]}"] = {"rows": int(len(ph)), "dep_fallback_rows": int(fb_d), "arr_fallback_rows": int(fb_a),
                                            "dM_sd": round(float(dM.std()), 3), "dM_changed": round(float((dM != 0).mean()), 4),
                                            "dep_yb_minus_y_sd": round(float((dM - dB)[dep].std()), 3),
                                            "arr_taxi_change_sd": round(float((tx1 - y0)[arr].std()), 3)}
        pd.concat(side, ignore_index=True).to_parquet(od / "deltas.parquet", index=False)
    (Path(a.out) / "redraw_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
