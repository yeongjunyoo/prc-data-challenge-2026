# SPDX-License-Identifier: GPL-3.0-only
# 2025년 1월, 7월을 시험 기간처럼 떼어 두 기준을 잰다: 공항별 중앙값, 이륙 시각 - AOBT_3(운항 계획 쪽 실제 출발).
import glob
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
        "AOBT_3_flt", "EOBT_1_flt", "SCHED_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt"]
df = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob("data/training_*.parquet"))], ignore_index=True)
df = df[df["PHASE_mvt"] == "DEP"].copy()
df["month"] = df["MVT_TIME_UTC_mvt"].dt.month
y = df["TAXITIME_SEC_mvt"].astype(float)
print("DEP rows", len(df), "target mean", round(y.mean(), 1), "sd", round(y.std(), 1))

hold = df["month"].isin([1, 7])
med = df[~hold].groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].median()
pred_med = df.loc[hold, "ADEP_mvt"].map(med).astype(float)
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))
print("holdout rows", int(hold.sum()), "| airport median RMSE", round(rmse(pred_med, y[hold]), 2))

proxy = (df["MVT_TIME_UTC_mvt"] - df["AOBT_3_flt"]).dt.total_seconds()
gap = (df["BLOCK_TIME_UTC_mvt"] - df["AOBT_3_flt"]).dt.total_seconds()
ok = proxy.notna()
print("AOBT_3 present", round(ok.mean(), 4))
print("BLOCK - AOBT_3 seconds quantiles:", gap[ok].quantile([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(1).to_dict())
print("|BLOCK - AOBT_3| <= 60s share:", round((gap[ok].abs() <= 60).mean(), 3), "| <= 300s:", round((gap[ok].abs() <= 300).mean(), 3))
h = hold & ok
print("holdout with AOBT_3: proxy RMSE", round(rmse(proxy[h], y[h]), 2), "| median RMSE on same rows", round(rmse(df.loc[h, 'ADEP_mvt'].map(med), y[h]), 2))
fill = proxy.where(ok, df["ADEP_mvt"].map(med)).clip(lower=0)
print("holdout all rows: proxy+median fill RMSE", round(rmse(fill[hold], y[hold]), 2))
for ap, g in df[h].groupby("ADEP_mvt"):
    print("  ", ap, "n", len(g), "proxy RMSE", round(rmse(proxy[g.index], y[g.index]), 1), "median gap", round(gap[g.index].median(), 1))
