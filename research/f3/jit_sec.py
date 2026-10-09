# SPDX-License-Identifier: GPL-3.0-only
"""F3 step 1c: per airport and phase, the share of MVT (and ARR BLOCK) values within 6 s of a whole minute, and the second-of-minute
support, in training, the old ranking, the final Jan/Jul head and the final Feb/Jun tail. Also the flips of round((MVT - AOBT_3)/60) and
of round((MVT - SCHED)/60) between the old and the new values of the same Jan/Jul rows.
    python -X utf8 research/f3/jit_sec.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("research/f3/out")


def to_s(s: pd.Series) -> np.ndarray:
    out = np.full(len(s), np.nan)
    m = s.notna().to_numpy()
    out[m] = (s[m].dt.tz_convert("UTC").dt.tz_localize(None) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    return out


def near_min(v: np.ndarray) -> np.ndarray:
    s = np.mod(v, 60)
    return (s <= 6) | (s >= 54)


def table(fr: pd.DataFrame, col: str) -> dict:
    ph = fr["PHASE_mvt"].to_numpy()
    ap = np.where(ph == "DEP", fr["ADEP_mvt"], fr["ADES_mvt"])
    v = to_s(fr[col])
    res = {}
    for p in ["DEP", "ARR"]:
        for a in sorted(set(ap)):
            m = (ph == p) & (ap == a) & np.isfinite(v)
            if m.sum() == 0:
                continue
            s = np.mod(v[m], 60).astype(int)
            res[f"{p}|{a}"] = {"n": int(m.sum()), "near_min": round(float(near_min(v[m]).mean()), 4),
                               "sec_7_53": int(((s >= 7) & (s <= 53)).sum()), "sec_le_12_or_ge_48": round(float(((s <= 12) | (s >= 48)).mean()), 4)}
    return res


def main():
    tr = pd.concat([pd.read_parquet(p, columns=["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"])
                    for p in sorted(Path("data").glob("training_*.parquet"))], ignore_index=True)
    r = pd.read_parquet("data/ranking.parquet")
    f = pd.read_parquet("data/final_phase/final_ranking.parquet")
    head, tail = f.iloc[:len(r)].reset_index(drop=True), f.iloc[len(r):].reset_index(drop=True)
    out = {}
    for col in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]:
        for nm, fr in [("training", tr), ("ranking_old", r), ("final_head", head), ("final_tail", tail)]:
            out[f"{col}|{nm}"] = table(fr, col)
    # flips of rounded minute differences, old vs new, same rows
    dep = r["PHASE_mvt"].to_numpy() == "DEP"
    flips = {}
    for ref in ["AOBT_3_flt", "SCHED_TIME_UTC_mvt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt"]:
        a0 = to_s(r["MVT_TIME_UTC_mvt"]) - to_s(r[ref])
        a1 = to_s(head["MVT_TIME_UTC_mvt"]) - to_s(head[ref])
        for p, m in [("DEP", dep), ("ARR", ~dep)]:
            ok = m & np.isfinite(a0)
            k0, k1 = np.round(a0[ok] / 60), np.round(a1[ok] / 60)
            nm0 = near_min(to_s(r["MVT_TIME_UTC_mvt"])[ok])
            flips[f"{p}|MVT-{ref}"] = {"n": int(ok.sum()), "flip": int((k0 != k1).sum()), "flip_rate": round(float((k0 != k1).mean()), 5),
                                      "flip_rate_near_min_rows": round(float((k0 != k1)[nm0].mean()), 5),
                                      "flip_rate_other_rows": round(float((k0 != k1)[~nm0].mean()), 5)}
    out["round_minute_flips_old_vs_new"] = flips
    (OUT / "jit_sec.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for col in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]:
        print(f"== {col}: near-minute share (|sec| <= 6) by phase|airport: training / old / final Jan-Jul / final Feb-Jun ; sec 7..53 count")
        for k in out[f"{col}|training"]:
            vals = [out[f"{col}|{nm}"].get(k) for nm in ["training", "ranking_old", "final_head", "final_tail"]]
            print(f"  {k:10s} " + "  ".join(f"{v['near_min']:.4f} ({v['sec_7_53']:6d})" if v else "-" for v in vals))
    print("== flips of round(diff/60), old vs new Jan/Jul rows")
    for k, v in flips.items():
        print(" ", k, v)


if __name__ == "__main__":
    main()
