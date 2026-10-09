# SPDX-License-Identifier: GPL-3.0-only
"""F2 gate 1.9, information only (after the decision): Lm - L0 by month x airport x matched, per chain and three-chain average, from
research/f2/out/gate19_<chain>_F.parquet (no fit). Cell value = own-row mean of se(Lm) - se(L0); share = the cell's sum over all F rows
(326,873), so the shares add up to the all-row contrast. Writes research/f2/out/gate19_ap.json.
    python -X utf8 -u research/f2/gate19_ap.py      (from <mac-workdir>/prcm/f2_run)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("research/f2/out")
P = {c: pd.read_parquet(OUT / f"gate19_{c}_F.parquet").set_index("MVT_ID_mvt") for c in "ABC"}
ids = P["A"].index
for c in "BC":
    P[c] = P[c].reindex(ids)
avg = P["A"][["y", "day", "matched"]].copy()
for k in ("L0", "Lm", "T"):
    avg[k] = np.mean([P[c][k].to_numpy() for c in "ABC"], axis=0)
P["avg"] = avg
ap = pd.read_parquet("cache/feats_v25.parquet", columns=["MVT_ID_mvt", "ap"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")["ap"].astype(str)
N = len(ids)
rep = {}
for c, d in P.items():
    t = pd.DataFrame({"ap": ap.reindex(ids).to_numpy(), "mon": d["day"].astype(str).str[5:7].to_numpy(), "m": d["matched"].to_numpy().astype(bool),
                      "dd": (d["Lm"] - d["y"]) ** 2 - (d["L0"] - d["y"]) ** 2})
    g = t.groupby(["mon", "ap", "m"])["dd"].agg(["size", "mean", "sum"])
    g["share"] = g["sum"] / N
    rows = [{"cell": f"{mo} {a} {'matched' if mm else 'unmatched'}", "rows": int(r["size"]), "mean": round(float(r["mean"]), 1), "share": round(float(r["share"]), 2)}
            for (mo, a, mm), r in g.iterrows()]
    rows.sort(key=lambda x: x["share"])
    rep[c] = {"all": round(float(t["dd"].mean()), 2), "cells": rows}
    print(c, "all", rep[c]["all"], "| most negative", rows[:4], "| most positive", rows[-4:], flush=True)
(OUT / "gate19_ap.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
print("AP written", OUT / "gate19_ap.json")
