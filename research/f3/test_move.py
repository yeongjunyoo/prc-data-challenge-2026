# SPDX-License-Identifier: GPL-3.0-only
"""F3 L4 final (10-10 10:2x: gate 7.8's test move is F3's, on the final root after full and before staging): how a final-root
candidate's test predictions move against a reference on the same draw (600 against 596, 602c against 598c), float predictions, read-only.
The same summary as R17-A's gate line "시험 이동" (MSE, rows moved more than 0.5 s, mean), by month and for Jan/Jul and Feb/Jun.
    python -X utf8 research/f3/test_move.py <candidate test parquet> <reference test parquet> <final_ranking parquet> <out json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def summary(m: np.ndarray) -> dict:
    return {"rows": int(len(m)), "mse": round(float(np.mean(m ** 2)), 2), "mean": round(float(np.mean(m)), 3),
            "rows_moved_over_0_5s": int((np.abs(m) > 0.5).sum()), "max_abs": round(float(np.abs(m).max()), 2)}


def main():
    cp, rp, kp, outp = sys.argv[1:5]
    c = pd.read_parquet(cp, columns=["MVT_ID_mvt", "pred"]).set_index("MVT_ID_mvt")["pred"].astype(float)
    r = pd.read_parquet(rp, columns=["MVT_ID_mvt", "pred"]).set_index("MVT_ID_mvt")["pred"].astype(float)
    assert c.index.is_unique and r.index.is_unique and set(c.index) == set(r.index), "candidate and reference rows differ"
    r = r.reindex(c.index)
    k = pd.read_parquet(kp, columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"])
    month = k[k["PHASE_mvt"] == "DEP"].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")["MVT_TIME_UTC_mvt"].reindex(c.index).dt.month.to_numpy()
    m = (c - r).to_numpy()
    jj = np.isin(month, [1, 7])
    rep = {"candidate": cp, "reference": rp, "all": summary(m), "jan_jul": summary(m[jj]), "feb_jun": summary(m[~jj]),
           "by_month": {int(x): summary(m[month == x]) for x in (1, 2, 6, 7)}}
    Path(outp).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
