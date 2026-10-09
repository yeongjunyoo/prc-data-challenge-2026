# SPDX-License-Identifier: GPL-3.0-only
"""F3 confirmation, the rows that carry the ΔMSE: for draw 1, the 25 rows with the largest increase of squared error (variant b) and the
features that changed on them (original feats_v25 row vs draw-1 row), plus how often each feature changed on the top 1% rows against all
rows. Also the same top rows' p_art and mu moves.
    python -X utf8 research/f3/conf_top.py <conf dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    C = Path(sys.argv[1])
    V = pd.read_parquet(C / "xconf" / "cache" / "val_m6r25_s0.parquet")
    k = np.floor(V["MVT_ID_mvt"].to_numpy() / 1e10).astype(int)
    b = V[k == 0].set_index("MVT_ID_mvt")
    c = V[k == 1].copy()
    c["MVT_ID_mvt"] -= 1e10
    c = c.set_index("MVT_ID_mvt").reindex(b.index)
    D = pd.read_parquet(C / "redraw" / "draw1" / "deltas.parquet").set_index("MVT_ID_mvt").reindex(b.index)
    y0, yb = b["y"].to_numpy(float), D["y_b"].to_numpy(float)
    d = (c["pred"].to_numpy() - yb) ** 2 - (b["pred"].to_numpy() - y0) ** 2
    F0 = pd.read_parquet(C.parent / "cache" / "feats_v25.parquet")
    F0 = F0[np.isin(F0["src"].to_numpy(), [1, 7])].set_index("MVT_ID_mvt").reindex(b.index)
    F1 = pd.read_parquet(C / "xr1" / "cache" / "feats_v25.parquet")
    F1 = F1[np.isin(F1["src"].to_numpy(), [1, 7])].set_index("MVT_ID_mvt").reindex(b.index)
    num = [x for x in F0.columns if not isinstance(F0[x].dtype, pd.CategoricalDtype) and x not in ("src", "y")]
    ch = pd.DataFrame({x: ~((F0[x].to_numpy(float) == F1[x].to_numpy(float)) | (np.isnan(F0[x].to_numpy(float)) & np.isnan(F1[x].to_numpy(float)))) for x in num},
                      index=b.index)
    n = len(d)
    top = np.argsort(-d)[: n // 100]
    rate_all, rate_top = ch.mean(), ch.iloc[top].mean()
    lift = (rate_top / rate_all.replace(0, np.nan)).sort_values(ascending=False)
    rows = []
    for i in np.argsort(-d)[:25]:
        r = {"id": float(b.index[i]), "ap": str(F0["ap"].iloc[i]), "y": y0[i], "y_b": yb[i], "dse": round(float(d[i])), "pred0": round(float(b["pred"].iloc[i]), 1),
             "pred1": round(float(c["pred"].iloc[i]), 1), "p_art0": round(float(b["p_art"].iloc[i]), 3), "p_art1": round(float(c["p_art"].iloc[i]), 3),
             "mu0": round(float(b["mu"].iloc[i]), 1), "mu1": round(float(c["mu"].iloc[i]), 1), "dsch0": float(F0["d_sch"].iloc[i]), "g": int(b["g"].iloc[i]),
             "changed": {x: [float(F0[x].iloc[i]), float(F1[x].iloc[i])] for x in num if ch[x].iloc[i] and x not in ("day_daobt_rel",)}}
        rows.append(r)
    out = {"dmse_b": float(d.sum() / n), "top1pct_share": float(d[top].sum() / d.sum()),
           "top1pct_p_art_move_mean_abs": float(np.abs(c["p_art"].to_numpy() - b["p_art"].to_numpy())[top].mean()),
           "all_p_art_move_mean_abs": float(np.abs(c["p_art"].to_numpy() - b["p_art"].to_numpy()).mean()),
           "top1pct_mu_move_mean_abs": float(np.abs(c["mu"].to_numpy() - b["mu"].to_numpy())[top].mean()),
           "feature_change_lift_top1pct": {x: [round(float(rate_top[x]), 3), round(float(rate_all[x]), 3)] for x in lift.index[:20]},
           "top25": rows}
    (C / "conf_top.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({kk: v for kk, v in out.items() if kk != "top25"}, indent=1))
    for r in rows[:12]:
        print(json.dumps(r, default=str)[:700])


if __name__ == "__main__":
    main()
