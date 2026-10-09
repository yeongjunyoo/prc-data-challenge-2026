# SPDX-License-Identifier: GPL-3.0-only
"""F2 information for the coordinator's row-split question (asked 10-09 21:4x after chain A's gate14 numbers were seen; not a gate). Gate 1.4's
L - T1 is measured on Feb/Jun 2025 rows, none of which has an ADS-B match (stage 1 = 0 there), while about 72% of the Feb/Jun 2026 rows of the
final have one (research/f2/f2_matchshare.py). The only 2025 rows with matches are Jan/Jul, so the one whole-month-out view of "the four-month
corrector on matched rows of a month stage 2 never saw" is gate 1.5's LV4 - LV2 (stage 2 on the other Jan/Jul month, against it plus every
Feb/Jun 2025 row, the evaluated month held out whole), split by match. Reads research/f2/out/gate15_<chain>_V.parquet and the matched ids of
cache/adsb_v1nd.parquet; fits nothing. Per chain and for the three-chain average (per-row mean of the chains' windowed predictions, 1/3 each):
LV4 - LV2, LV4_nm - LV2_nm and U_V - I_V (gate 1.4's in-month guard) on all, matched and unmatched rows, and per month; mean over the subset's
own rows, day-block bootstrap (research/f2/m3_corr.boot copied: UTC day of MVT, 2,000 draws, seed 0).
    python -u research/f2/lv_matched.py [--dir research/f2/out] [--adsb cache/adsb_v1nd.parquet] [--out research/f2/out/lv_matched.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
CONTRASTS = (("LV4_minus_LV2", "LV4", "LV2"), ("LV4nm_minus_LV2nm", "LV4_nm", "LV2_nm"), ("U_V_minus_I_V", "U_V", "I_V"))
COLS = ["I_V", "U_V", "LV2", "LV4", "LV2_nm", "LV4_nm"]


def boot(d: np.ndarray, day: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """research/f2/m3_corr.boot verbatim: day-block bootstrap of mean(d): point, sd, P(<0)."""
    codes, days = pd.factorize(day)
    s = np.bincount(codes, weights=d)
    n = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    est = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return {"dmse": round(float(d.mean()), 2), "sd": round(float(est.std()), 2), "p_lt0": round(float((est < 0).mean()), 4), "days": int(len(s))}


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--dir", default="research/f2/out")
    pa.add_argument("--adsb", default="cache/adsb_v1nd.parquet")
    pa.add_argument("--out", default="research/f2/out/lv_matched.json")
    g = pa.parse_args()
    d = Path(g.dir)
    ids_m = pq.read_table(g.adsb, columns=["MVT_ID_mvt"], filters=[("matched", "=", True)]).column(0).to_numpy()
    P = {ch: pd.read_parquet(d / f"gate15_{ch}_V.parquet").set_index("MVT_ID_mvt") for ch in "ABC"}
    idx = P["A"].index
    for ch in "BC":
        assert P[ch].index.is_unique and len(P[ch]) == len(idx) and P[ch].index.isin(idx).all(), f"chain {ch}: row set differs from A's"
        P[ch] = P[ch].reindex(idx)
        assert np.array_equal(P[ch]["y"].to_numpy(float), P["A"]["y"].to_numpy(float), equal_nan=True), f"chain {ch}: y differs"
    avg = P["A"][["y", "day"]].copy()
    for c in COLS:
        avg[c] = np.mean([P[ch][c].to_numpy(float) for ch in "ABC"], axis=0)
    P["avg"] = avg
    matched = np.asarray(idx.isin(ids_m))
    day = avg["day"].astype(str).to_numpy()
    mon = np.array([x[5:7] for x in day])
    y = avg["y"].to_numpy(float)
    rep = {"what": __doc__.split("\n")[0], "rows": int(len(idx)), "matched_rows": int(matched.sum()),
           "matched_share_by_month": {mm: round(float(matched[mon == mm].mean()), 4) for mm in ("01", "07")}, "by": {}}
    for ch, df in P.items():
        r = {}
        for name, new, old in CONTRASTS:
            dd = (df[new].to_numpy(float) - y) ** 2 - (df[old].to_numpy(float) - y) ** 2
            ok = np.isfinite(dd)
            sub = {}
            for sname, s in (("all", ok), ("matched", ok & matched), ("unmatched", ok & ~matched)):
                sub[sname] = boot(dd[s], day[s])
                for mm in ("01", "07"):
                    t = s & (mon == mm)
                    sub[f"{sname}_{mm}"] = boot(dd[t], day[t])
            r[name] = sub
        rep["by"][ch] = r
        print(f"{ch:3s} " + " | ".join(f"{name}: all {r[name]['all']['dmse']:+.1f} (P {r[name]['all']['p_lt0']:.3f}), matched {r[name]['matched']['dmse']:+.1f} "
                                       f"(sd {r[name]['matched']['sd']:.1f}, P {r[name]['matched']['p_lt0']:.3f}), unmatched {r[name]['unmatched']['dmse']:+.1f} "
                                       f"(sd {r[name]['unmatched']['sd']:.1f}, P {r[name]['unmatched']['p_lt0']:.3f})" for name, _, _ in CONTRASTS))
    out = Path(g.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"LVMATCHED written {out}: {rep['rows']:,} rows, matched {rep['matched_rows']:,} {rep['matched_share_by_month']}")


if __name__ == "__main__":
    main()
