# SPDX-License-Identifier: GPL-3.0-only
"""F3 confirmation, split: ΔMSE of the re-drawn copies by airport group. The sampler draws d from P(d | phase, airport, old second). At the
five whole-minute airports (MVT within 6 s of a whole minute: EDDM, EGLL, LFPG, LIRF, LTFM) the old second gives the old jitter o, so the
draw is old - o + n = true + n: faithful. At the second-resolution airports (EDDF, EHAM, LEBL, LEMD, LSZH) o is not visible, the drawn d is
independent of o and the new value is true + o + d: noisier than any real draw (variance 12 + 24 against 12). If the ΔMSE sits at the
second-resolution airports, it measures the sampler's extra noise, not the organizer's re-draw.
Also the per-row move against the old residual and the share of ΔMSE from the 1% largest |Δ se| rows.
    python -X utf8 research/f3/conf_split.py <val parquet> <redraw dir> <feats parquet for ap> <out json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MINRES = {"EDDM", "EGLL", "LFPG", "LIRF", "LTFM"}


def main():
    vp, rd, fp, outp = sys.argv[1:5]
    V = pd.read_parquet(vp)
    k = np.floor(V["MVT_ID_mvt"].to_numpy() / 1e10).astype(int)
    base = V[k == 0].set_index("MVT_ID_mvt")
    ids = base.index.to_numpy()
    ap = pd.read_parquet(fp, columns=["MVT_ID_mvt", "ap"]).set_index("MVT_ID_mvt")["ap"].astype(str).reindex(ids).to_numpy()
    grp = np.where(np.isin(ap, list(MINRES)), "minute", "second")
    y0 = base["y"].to_numpy(float)
    g = base["g"].to_numpy().astype(bool)
    p0 = base["pred"].to_numpy(float)
    n = len(ids)
    res = {"rows": int(n), "rows_minute_airports": int((grp == "minute").sum()), "rows_second_airports": int((grp == "second").sum()), "copies": {}}
    for kk in (1, 2, 3, 4, 5):
        C = V[k == kk].copy()
        C["MVT_ID_mvt"] = C["MVT_ID_mvt"] - kk * 1e10
        p = C.set_index("MVT_ID_mvt").reindex(ids)["pred"].to_numpy(float)
        D = pd.read_parquet(Path(rd) / f"draw{kk if kk <= 3 else 1}" / "deltas.parquet").set_index("MVT_ID_mvt").reindex(ids)
        yb = D["y_b"].to_numpy(float)
        r = {}
        for var, y in (("a", y0), ("b", yb)):
            d = (p - y) ** 2 - (p0 - y0) ** 2
            for gname in ("minute", "second"):
                for seg, m in (("all", np.ones(n, bool)), ("body_nonG", (y0 <= 3600) & ~g), ("y_gt3600", (y0 > 3600) & ~g), ("G", g)):
                    mm = m & (grp == gname)
                    r[f"{var}|{gname}|{seg}"] = round(float(d[mm].sum() / n), 2)
            top = np.argsort(-np.abs(d))[: n // 100]
            r[f"{var}|share_from_top1pct_rows"] = round(float(d[top].sum() / d.sum()), 3) if d.sum() != 0 else None
        mv = p - p0
        for gname in ("minute", "second"):
            m = grp == gname
            r[f"move_rms|{gname}"] = round(float(np.sqrt(np.mean(mv[m] ** 2))), 2)
            r[f"move_vs_resid_corr|{gname}"] = round(float(np.corrcoef(mv[m], (y0 - p0)[m])[0, 1]), 4)
        res["copies"][kk] = r
    Path(outp).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
