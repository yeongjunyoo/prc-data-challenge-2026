# SPDX-License-Identifier: GPL-3.0-only
"""로컬 자 표: cache/val_<tag>_s<seed>.parquet 의 방법별 RMSE, 시드 간 표준편차, 시드 평균 예측의 RMSE.

사용: python src/final_table.py <tag> [<tag> ...]
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
METHODS = ["pred", "pred_noG", "direct", "direct_G", "blend"]


def rm(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - y) ** 2)))


for tag in sys.argv[1:]:
    files = sorted(glob.glob(str(ROOT / "cache" / f"val_{tag}_s*.parquet")))
    if not files:
        continue
    dfs = [pd.read_parquet(f) for f in files]
    y = dfs[0]["y"].to_numpy()
    dsch = dfs[0]["dsch"].to_numpy()
    noise = (y > 50000) & (dsch < 20000)
    for d in dfs:
        if "direct_G" in d:
            d["blend"] = 0.5 * d["pred"] + 0.5 * d["direct_G"]
    for mth in METHODS:
        if mth not in dfs[0]:
            continue
        r = [rm(d[mth], y) for d in dfs]
        avg = np.mean([d[mth].to_numpy() for d in dfs], axis=0)
        sd = np.std(r, ddof=1) if len(r) > 1 else float("nan")
        print(f"{tag:6s} {mth:9s} seeds={len(r)} RMSE mean {np.mean(r):7.2f} sd {sd:5.2f} | seed-avg {rm(avg, y):7.2f} | "
              f"without {int(noise.sum())} no-signal rows {rm(avg[~noise], y[~noise]):7.2f} | y<3600 rows {rm(avg[y < 3600], y[y < 3600]):6.2f}")
