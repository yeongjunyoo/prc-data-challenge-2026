# SPDX-License-Identifier: GPL-3.0-only
"""F3 step 1d: is the final file a fresh draw of the same jitter (new = true + n, old = true + o, n and o iid) rather than old + noise?
Tests: (1) symmetry of (old second, new second) for the same Jan/Jul rows; (2) in the training files, at airports where MVT and BLOCK sit
within 6 s of a whole minute, does the BLOCK offset mirror the MVT offset (BLOCK jitter = -MVT jitter), and is the taxi second part
= 2 x MVT offset; (3) the same in the old and new ARR rows.
    python -X utf8 research/f3/jit_redraw.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("research/f3/out")
MINRES = ["EDDM", "EGLL", "LFPG", "LIRF", "LTFM"]


def to_s(s: pd.Series) -> np.ndarray:
    out = np.full(len(s), np.nan)
    m = s.notna().to_numpy()
    out[m] = (s[m].dt.tz_convert("UTC").dt.tz_localize(None) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    return out


def off(v: np.ndarray) -> np.ndarray:
    """signed offset from the nearest whole minute, in [-30, 30)"""
    return np.mod(v + 30, 60) - 30


def main():
    res = {}
    r = pd.read_parquet("data/ranking.parquet")
    f = pd.read_parquet("data/final_phase/final_ranking.parquet").iloc[:len(r)].reset_index(drop=True)
    m0, m1 = to_s(r["MVT_TIME_UTC_mvt"]), to_s(f["MVT_TIME_UTC_mvt"])
    d = m1 - m0
    s0, s1 = np.mod(m0, 60).astype(int), np.mod(m1, 60).astype(int)
    ct = pd.crosstab(s0, s1).to_numpy().astype(float)
    res["symmetry_old_new_second"] = {"rel_asym": float(np.abs(ct - ct.T).sum() / ct.sum()),
                                      "expected_noise_rel": float(np.sqrt(ct + ct.T).sum() / ct.sum())}
    e_old = pd.Series(d).groupby(s0).mean()
    e_new = pd.Series(d).groupby(s1).mean()
    res["E[d|old sec] + E[d|new sec] (should be ~0 by exchangeability)"] = {int(k): round(float(e_old[k] + e_new[k]), 3) for k in [0, 3, 5, 6, 7, 30, 53, 54, 55, 57]}
    # training: mirrored offsets
    tr = pd.concat([pd.read_parquet(p, columns=["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
                    for p in sorted(Path("data").glob("training_*.parquet"))], ignore_index=True)
    for nm, fr in [("training", tr), ("ranking_old", r), ("final_head", f)]:
        ph = fr["PHASE_mvt"].to_numpy()
        ap = np.where(ph == "DEP", fr["ADEP_mvt"], fr["ADES_mvt"])
        om, ob = off(to_s(fr["MVT_TIME_UTC_mvt"])), off(to_s(fr["BLOCK_TIME_UTC_mvt"]))
        tx = fr["TAXITIME_SEC_mvt"].to_numpy(float)
        for p in ["DEP", "ARR"]:
            m = (ph == p) & np.isin(ap, MINRES) & np.isfinite(ob) & (np.abs(om) <= 6) & (np.abs(ob) <= 6)
            if m.sum() == 0:
                continue
            ssum = om[m] + ob[m]
            taxi_sec = off(tx[m])
            sgn = 1 if p == "DEP" else -1
            res[f"{nm}|{p}|minres airports, both near minute"] = {
                "n": int(m.sum()), "corr(offM, offB)": round(float(np.corrcoef(om[m], ob[m])[0, 1]), 4),
                "offM+offB hist": {int(k): int(v) for k, v in zip(*np.unique(ssum, return_counts=True)) if abs(k) <= 3},
                "share |offM+offB|<=1": round(float((np.abs(ssum) <= 1).mean()), 4),
                "corr(taxi second part, 2*offM*sign)": round(float(np.corrcoef(taxi_sec, sgn * 2 * om[m])[0, 1]), 4),
                "share taxi_sec == sign*2*offM +-1": round(float((np.abs(taxi_sec - sgn * 2 * om[m]) <= 1).mean()), 4)}
    (OUT / "jit_redraw.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, v in res.items():
        print(k, json.dumps(v, ensure_ascii=False))


if __name__ == "__main__":
    main()
