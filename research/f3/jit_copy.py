# SPDX-License-Identifier: GPL-3.0-only
"""F3 step 1b: exact-equality ("copy") rates between the re-jittered columns (MVT, BLOCK) and the untouched timestamp columns, in the
training files, the old ranking file and the final file (Jan/Jul head and Feb/Jun tail), per phase. Also the rate within 12 s and 24 s.
    python -X utf8 research/f3/jit_copy.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("research/f3/out")
JIT = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]
FIX = ["SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt"]


def to_s(s: pd.Series) -> np.ndarray:
    out = np.full(len(s), np.nan)
    m = s.notna().to_numpy()
    out[m] = (s[m].dt.tz_convert("UTC").dt.tz_localize(None) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    return out


def rates(fr: pd.DataFrame) -> dict:
    res = {}
    ph = fr["PHASE_mvt"].to_numpy()
    tv = {c: to_s(fr[c]) for c in JIT + FIX}
    for p in ["DEP", "ARR"]:
        m = ph == p
        n = int(m.sum())
        for a in JIT:
            for b in FIX + ([x for x in JIT if x != a] if a == "MVT_TIME_UTC_mvt" else []):
                d = tv[a][m] - tv[b][m]
                ok = np.isfinite(d)
                if ok.sum() == 0:
                    continue
                dd = d[ok]
                res[f"{p}|{a[:5]}-{b[:8]}"] = {"n": n, "both": int(ok.sum()), "eq": int((dd == 0).sum()), "le12": int((np.abs(dd) <= 12).sum()),
                                               "le24": int((np.abs(dd) <= 24).sum()), "mod60_0": int((np.mod(dd, 60) == 0).sum())}
    return res


def main():
    out = {}
    tr = pd.concat([pd.read_parquet(p) for p in sorted(Path("data").glob("training_*.parquet"))], ignore_index=True)
    out["training"] = rates(tr)
    r = pd.read_parquet("data/ranking.parquet")
    out["ranking_old"] = rates(r)
    f = pd.read_parquet("data/final_phase/final_ranking.parquet")
    out["final_head"] = rates(f.iloc[:len(r)])
    out["final_tail"] = rates(f.iloc[len(r):])
    (OUT / "jit_copy.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    keys = sorted(set().union(*[set(v) for v in out.values()]))
    print(f"{'pair':28s} " + " ".join(f"{k:>34s}" for k in out))
    for k in keys:
        row = []
        for s in out:
            v = out[s].get(k)
            row.append(f"{v['eq'] / v['both'] * 100:7.3f}% {v['le12'] / v['both'] * 100:7.3f}% {v['mod60_0'] / v['both'] * 100:7.3f}% ({v['both']:7d})" if v else " " * 34)
        print(f"{k:28s} " + " ".join(row))


if __name__ == "__main__":
    main()
