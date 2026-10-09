# SPDX-License-Identifier: GPL-3.0-only
"""F3: is the organizer's 2026 re-draw the same process as the original jitter? If it is, the old and the new versions of the same rows are
exchangeable, so every feature's distribution (not only each timestamp's) must match between them, including cross-row ones such as
runway gaps, counts and orders. Compares, for the same ids:
  2026 Jan/Jul: today's feats_v25 (old draw) vs the final root's feats_v25 (organizer's new draw)
  2025 Jan/Jul: today's feats_v25 (original draw) vs the sampler's draws 1..3
per feature: the two-sample KS statistic and, for gap-like features, the counts in small bins; split by airport group (whole-minute
airports, second-resolution airports).
    python -X utf8 research/f3/exch_check.py <today's feats_v25> <final feats_v25> <conf dir> <out json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MINRES = ["EDDM", "EGLL", "LFPG", "LIRF", "LTFM"]
GAPS = {"rwy_gap_prev": [0, 1, 6, 11, 21, 31, 46, 61, 91, 121], "rwy_gap_next": [0, 1, 6, 11, 21, 31, 46, 61, 91, 121],
        "st_prev_dep_gap": [0, 61, 301, 1801], "dup_gap_prev": [0, 1, 6, 31, 61, 301], "c_dep_m5": [0, 1, 2, 3, 4], "c_dep_rwy_m10": [0, 1, 2, 3, 4]}


def ks(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    v = np.union1d(a, b)
    return float(np.max(np.abs(np.searchsorted(a, v, side="right") / len(a) - np.searchsorted(b, v, side="right") / len(b))))


def bins(x: np.ndarray, edges: list) -> list:
    x = x[np.isfinite(x)]
    e = edges + [np.inf]
    return [int(((x >= e[i]) & (x < e[i + 1])).sum()) for i in range(len(edges))]


def compare(A: pd.DataFrame, B: pd.DataFrame, ap: np.ndarray) -> dict:
    out = {}
    num = [c for c in A.columns if not isinstance(A[c].dtype, pd.CategoricalDtype) and c not in ("MVT_ID_mvt", "src", "y")]
    for grp, m in (("minute", np.isin(ap, MINRES)), ("second", ~np.isin(ap, MINRES))):
        r = {}
        for c in num:
            a, b = A[c].to_numpy(float)[m], B[c].to_numpy(float)[m]
            if np.array_equal(a, b, equal_nan=True):
                continue
            r[c] = {"ks": round(ks(a, b), 5), "mean": [round(float(np.nanmean(a)), 3), round(float(np.nanmean(b)), 3)]}
            if c in GAPS:
                r[c]["bins"] = [bins(a, GAPS[c]), bins(b, GAPS[c])]
        out[grp] = r
    return out


def main():
    today, final, conf, outp = sys.argv[1:5]
    T = pd.read_parquet(today)
    Fn = pd.read_parquet(final)
    rep = {"n_ks_crit_note": "two-sample KS 5% critical value about 1.36*sqrt(2/n): 0.0046 for n = 180,000"}
    o26 = T[T["src"] == 0].set_index("MVT_ID_mvt")
    n26 = Fn[Fn["src"] == 0].set_index("MVT_ID_mvt").reindex(o26.index)
    rep["2026_old_vs_new"] = compare(o26.reset_index(), n26.reset_index(), o26["ap"].astype(str).to_numpy())
    o25 = T[np.isin(T["src"].to_numpy(), [1, 7])].set_index("MVT_ID_mvt")
    for k in (1, 2, 3):
        D = pd.read_parquet(Path(conf) / f"xr{k}" / "cache" / "feats_v25.parquet")
        D = D[np.isin(D["src"].to_numpy(), [1, 7])].set_index("MVT_ID_mvt").reindex(o25.index)
        rep[f"2025_orig_vs_draw{k}"] = compare(o25.reset_index(), D.reset_index(), o25["ap"].astype(str).to_numpy())
    Path(outp).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    for key in ("2026_old_vs_new", "2025_orig_vs_draw1", "2025_orig_vs_draw2"):
        for grp in ("minute", "second"):
            r = rep[key][grp]
            top = sorted(r.items(), key=lambda kv: -kv[1]["ks"])[:8]
            print(key, grp, [(c, v["ks"]) for c, v in top])
            for c in ("rwy_gap_prev", "dup_gap_prev", "c_dep_m5"):
                if c in r and "bins" in r[c]:
                    print("   ", c, r[c]["bins"])


if __name__ == "__main__":
    main()
