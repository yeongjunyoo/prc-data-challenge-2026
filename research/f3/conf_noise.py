# SPDX-License-Identifier: GPL-3.0-only
"""F3 confirmation: is "the old draw scores better than the three new draws" larger than the draw-to-draw noise of the val MSE?
Under the re-draw model with the label moved too (variant b), the old draw (k = 0) and the new draws (k = 1..3) are exchangeable per row,
so the four squared errors e_k^2 of a row are iid. The noise of one draw's MSE is then sqrt(sum_i var_i(e^2)) / n, with var_i estimated per
row from its four values (unbiased, 4 values). The day-block bootstrap of the earlier report does not contain this term (it keeps the
old draw fixed). Reports the observed (mean of new) - old and its z against that noise, overall and by airport group and segment.
    python -X utf8 research/f3/conf_noise.py <conf dir> <feats parquet for ap>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MINRES = ["EDDM", "EGLL", "LFPG", "LIRF", "LTFM"]


def main():
    C = Path(sys.argv[1])
    V = pd.read_parquet(C / "xconf" / "cache" / "val_m6r25_s0.parquet")
    k = np.floor(V["MVT_ID_mvt"].to_numpy() / 1e10).astype(int)
    b = V[k == 0].set_index("MVT_ID_mvt")
    ids = b.index.to_numpy()
    ap = pd.read_parquet(sys.argv[2], columns=["MVT_ID_mvt", "ap"]).set_index("MVT_ID_mvt")["ap"].astype(str).reindex(ids).to_numpy()
    y0, g = b["y"].to_numpy(float), b["g"].to_numpy().astype(bool)
    P = [b["pred"].to_numpy(float)]
    Y = [y0]
    for kk in (1, 2, 3):
        c = V[k == kk].copy()
        c["MVT_ID_mvt"] -= kk * 1e10
        P.append(c.set_index("MVT_ID_mvt").reindex(ids)["pred"].to_numpy(float))
        Y.append(pd.read_parquet(C / "redraw" / f"draw{kk}" / "deltas.parquet").set_index("MVT_ID_mvt").reindex(ids)["y_b"].to_numpy(float))
    n = len(ids)
    rep = {}
    for var in ("b", "a"):
        E2 = np.stack([(P[j] - (Y[j] if var == "b" else y0)) ** 2 for j in range(4)])
        vi = E2.var(axis=0, ddof=1)
        for gname, gm in (("all", np.ones(n, bool)), ("minute", np.isin(ap, MINRES)), ("second", ~np.isin(ap, MINRES))):
            for seg, sm in (("all", np.ones(n, bool)), ("body_nonG", (y0 <= 3600) & ~g), ("y_gt3600", (y0 > 3600) & ~g), ("G", g)):
                m = gm & sm
                obs = float((E2[1:, m].mean(axis=0).sum() - E2[0, m].sum()) / n)
                sd1 = float(np.sqrt(vi[m].sum()) / n)
                rep[f"{var}|{gname}|{seg}"] = {"new_minus_old": round(obs, 2), "sd_one_draw": round(sd1, 2), "z": round(obs / (sd1 * np.sqrt(1 + 1 / 3)), 2) if sd1 > 0 else None}
    (C / "conf_noise.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    for kk, v in rep.items():
        print(kk, v)


if __name__ == "__main__":
    main()
