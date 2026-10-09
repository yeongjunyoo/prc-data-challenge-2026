# SPDX-License-Identifier: GPL-3.0-only
"""F3 confirmation: ΔMSE of the m6r25 val predictions on re-drawn inputs against the original inputs, same fitted models.
Copies k = 1..3 are full re-draws, k = 4 changes only sec_mvt (draw 1), k = 5 only d_aobt_mode_dev (draw 1) (research/f3/conf_concat.py).
Truth: variant a = today's label; variant b = the label moved as if the hidden DEP BLOCK were re-drawn opposite to MVT (y + dM - dB of
the draw; k = 4, 5 use draw 1's). ΔMSE per segment is summed over the segment and divided by all val rows (the R16-B gate's convention):
all, body_nonG (y <= 3600, not G), G rows, y > 3600. Day-block bootstrap (UTC day of the row, 2,000 draws, seed 0) gives P(Δ < 0).
    python -X utf8 research/f3/conf_eval.py <val parquet> <redraw dir> <out json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    vp, rd, outp = sys.argv[1:4]
    V = pd.read_parquet(vp)
    k = np.floor(V["MVT_ID_mvt"].to_numpy() / 1e10).astype(int)
    base = V[k == 0].set_index("MVT_ID_mvt")
    ids = base.index.to_numpy()
    y0 = base["y"].to_numpy(float)
    g = base["g"].to_numpy().astype(bool)
    p0 = base["pred"].to_numpy(float)
    t = pd.read_parquet(Path(vp).parent / "all.parquet", columns=["MVT_ID_mvt", "t_mvt"]).set_index("MVT_ID_mvt")["t_mvt"].reindex(ids)
    day = np.floor(t.to_numpy() / 86400).astype(np.int64)
    ud, di = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(ud), len(ud)), minlength=len(ud)) for _ in range(2000)])  # day multiplicities
    n = len(ids)
    res = {"rows": int(n), "mse_orig": float(np.mean((p0 - y0) ** 2)), "copies": {}}
    deltas = {j: pd.read_parquet(Path(rd) / f"draw{j}" / "deltas.parquet").set_index("MVT_ID_mvt").reindex(ids) for j in (1, 2, 3)}
    for kk in sorted(set(k) - {0}):
        C = V[k == kk].copy()
        C["MVT_ID_mvt"] = C["MVT_ID_mvt"] - kk * 1e10
        C = C.set_index("MVT_ID_mvt").reindex(ids)
        p = C["pred"].to_numpy(float)
        dj = deltas[kk if kk <= 3 else 1]
        yb = dj["y_b"].to_numpy(float)
        assert np.allclose(dj["y_old"].to_numpy(float), y0, equal_nan=True)
        r = {"pred_move_rms": float(np.sqrt(np.mean((p - p0) ** 2))), "pred_move_max": float(np.max(np.abs(p - p0))),
             "rows_moved_gt1s": int((np.abs(p - p0) > 1).sum()), "p_art_move_max": float(np.max(np.abs(C["p_art"].to_numpy(float) - base["p_art"].to_numpy(float))))}
        for var, y in (("a", y0), ("b", yb)):
            if var == "b":
                r["b_label_change_sd"] = float(np.std(yb - y0))
                r["b_mse_orig_pred"] = float(np.mean((p0 - yb) ** 2) - np.mean((p0 - y0) ** 2))  # cost of the label change alone
            d = (p - y) ** 2 - (p0 - y0) ** 2  # vs original inputs and original label
            d_same = (p - y) ** 2 - (p0 - y) ** 2  # input effect alone, same label
            segs = {"all": np.ones(n, bool), "body_nonG": (y0 <= 3600) & ~g, "G": g, "y_gt3600": (y0 > 3600) & ~g}
            for sn, m in segs.items():
                dd = np.where(m, d, 0.0)
                ds = np.where(m, d_same, 0.0)
                daysum = np.bincount(di, weights=dd, minlength=len(ud))
                boot = (W @ daysum) / n
                r[f"{var}|{sn}"] = {"dmse": round(float(dd.sum() / n), 3), "dmse_inputs_only": round(float(ds.sum() / n), 3),
                                    "p_below0": round(float((boot < 0).mean()), 4), "boot_sd": round(float(boot.std()), 3)}
        res["copies"][int(kk)] = r
    Path(outp).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
