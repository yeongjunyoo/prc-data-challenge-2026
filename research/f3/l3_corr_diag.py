# SPDX-License-Identifier: GPL-3.0-only
"""F3 L3 final, consistency expectation (a), written down before the rebuild (private notes): why the L3-vs-L move on the new draw (596 - 592, Jan/Jul) correlates only 0.50 per row
with the same move on the old draw (570 - 531), against the expected 0.9 or more. Read-only ('pred' column, Jan/Jul rows = the public test ids):
  new = the final root's files (L3 legs refit there on feats_v28; 514's legs and blends from the s50 build), old = the public phase (L3: r17a_sbx;
  514: the live root's public files).
  1. For each leg or blend pair (L3 tag, 514 tag): MSE of (L3 - L) on each draw and the row correlation between draws; each family's own move
     between draws (final fit on the new draw minus public fit on the old draw) and the correlation of the two families' moves.
  2. A pure refit difference inside one family (m6r28sk and m6r28skwin seed 0 - seed 1) on each draw and its correlation between draws: how much
     of a leg difference is a fit's own pattern rather than a property of the rows.
  3. The final files: 596 - 592 against 570 - 531 (research/f3/out/cmp_ff_596.json's inputs): regression slope, R2 and the shared variance
     cov(new, old) under "shared part plus independent fit noise of equal size in each draw".
Report <out json>.
    python -X utf8 research/f3/l3_corr_diag.py <final cache dir> <r17a cache dir> <public 514 cache dir> <596 test> <592 test> <570 test> <531 test> <out json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load(p: Path, ids: np.ndarray) -> np.ndarray | None:
    if not p.exists():
        return None
    return pd.read_parquet(p, columns=["MVT_ID_mvt", "pred"]).set_index("MVT_ID_mvt")["pred"].astype(float).reindex(ids).to_numpy()


def stats(new: np.ndarray, old: np.ndarray) -> dict:
    ok = np.isfinite(new) & np.isfinite(old)
    n, o = new[ok], old[ok]
    return {"rows": int(ok.sum()), "mse_new": round(float(np.mean(n ** 2)), 2), "mse_old": round(float(np.mean(o ** 2)), 2),
            "corr_new_old": round(float(np.corrcoef(n, o)[0, 1]), 4), "cov_new_old": round(float(np.mean((n - n.mean()) * (o - o.mean()))), 2)}


def main():
    fin, r17, pub = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    t596, t592, t570, t531, outp = sys.argv[4:9]
    ids = pd.read_parquet(t570, columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()
    rep = {"jan_jul_rows": int(len(ids)), "dirs": {"final": str(fin), "r17a": str(r17), "public_514": str(pub)}, "pairs": {}, "refit_within_family": {}}
    for a, b in (("m6r28sk_s0", "m6r27sk_s0"), ("m6r28skwin_s0", "m6r27skwin_s0"), ("m6r28skap_s0", "m6r27skap_s0"),
                 ("m6r28skgx4_s0", "m6r27skgx4_s0"), ("m6r28skwingx4_s0", "m6r27skwingx4_s0"), ("m6r28skapg_s0", "m6r27skapg_s0"),
                 ("m6r28skcbrmfw4sv434_s0", "m6r27skcbrmfw4sv434_s0"), ("m6r28apcbrmfw4sv114_s0", "m6r27apcbrmfw4sv114_s0")):
        fa, fb = load(fin / f"full_{a}.parquet", ids), load(fin / f"full_{b}.parquet", ids)
        pa, pb = load(r17 / f"full_{a}.parquet", ids), load(pub / f"full_{b}.parquet", ids)
        if any(v is None for v in (fa, fb, pa, pb)):
            rep["pairs"][f"{a} - {b}"] = "missing file"
            continue
        mL3, mL = fa - pa, fb - pb
        rep["pairs"][f"{a} - {b}"] = {**stats(fa - fb, pa - pb), "own_move_L3_mse": round(float(np.mean(mL3 ** 2)), 2),
                                      "own_move_L_mse": round(float(np.mean(mL ** 2)), 2),
                                      "corr_own_moves_L3_vs_L": round(float(np.corrcoef(mL3, mL)[0, 1]), 4)}
    for fam in ("m6r28sk", "m6r28skwin"):
        f0, f1 = load(fin / f"full_{fam}_s0.parquet", ids), load(fin / f"full_{fam}_s1.parquet", ids)
        p0, p1 = load(r17 / f"full_{fam}_s0.parquet", ids), load(r17 / f"full_{fam}_s1.parquet", ids)
        if any(v is None for v in (f0, f1, p0, p1)):
            rep["refit_within_family"][fam] = "missing file"
            continue
        rep["refit_within_family"][f"{fam}: seed 0 - seed 1"] = {**stats(f0 - f1, p0 - p1),
                                                                 "own_move_s0_mse": round(float(np.mean((f0 - p0) ** 2)), 2),
                                                                 "corr_own_moves_s0_vs_s1": round(float(np.corrcoef(f0 - p0, f1 - p1)[0, 1]), 4)}
    a, b, c, d = (load(Path(x), ids) for x in (t596, t592, t570, t531))
    new, old = a - b, c - d
    s = stats(new, old)
    slope = s["cov_new_old"] / float(np.var(old))
    s.update({"slope_new_on_old": round(slope, 4), "r2": round(s["corr_new_old"] ** 2, 4),
              "shared_variance_cov": s["cov_new_old"],
              "non_replicating_variance_new (mse_new - cov)": round(s["mse_new"] - s["cov_new_old"], 2),
              "share_non_replicating_new": round(1 - s["cov_new_old"] / s["mse_new"], 3)})
    rep["final_files"] = s
    Path(outp).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
