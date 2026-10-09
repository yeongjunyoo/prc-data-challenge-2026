# SPDX-License-Identifier: GPL-3.0-only
"""F2 (10-09 22:xx, for our row-split question): share of context rows whose id has an ADS-B match in the corrector's table
(adsb_v1nd, matched == True, the rows stage 1 corrects), by context part, in final_run (the final's tables) and in f2_run (gate 1.4's world).
Reads id columns only, one thread, writes nothing.
    python -u research/f2/f2_matchshare.py
"""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

P = Path(__import__("os").environ.get("PRC_WORK", "."))
for root, ctx in (("final_run", "ctx_v3"), ("f2_run", "f2ctx_v3")):
    c = P / root / "cache"
    a = pq.read_table(c / "adsb_v1nd.parquet", columns=["MVT_ID_mvt"], filters=[("matched", "=", True)]).to_pandas()
    x = pd.read_parquet(c / f"{ctx}.parquet", columns=["MVT_ID_mvt", "part"])
    x["m"] = x["MVT_ID_mvt"].isin(a["MVT_ID_mvt"].to_numpy())
    g = x.groupby(x["part"].astype(str))["m"].agg(["sum", "size", "mean"]).round(4)
    print(f"{root}/{ctx}: {len(a):,} matched ADS-B ids")
    print(g.to_string())
