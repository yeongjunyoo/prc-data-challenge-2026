# SPDX-License-Identifier: GPL-3.0-only
"""F2 deliverable 2: 2025 training coverage by month and airport (departures), ranking file months."""
import glob, sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
cols = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt", "FLIGHT_ID_mvt"]
tr = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/training_*.parquet"))], ignore_index=True)
dep = tr[tr["PHASE_mvt"] == "DEP"].copy()
dep["month"] = dep["MVT_TIME_UTC_mvt"].dt.month
dep["day"] = dep["MVT_TIME_UTC_mvt"].dt.day
print("2025 DEP rows", len(dep), "months", sorted(dep["month"].unique()))
g = dep.groupby("month").agg(rows=("MVT_ID_mvt", "size"), days=("day", "nunique"), y_mean=("TAXITIME_SEC_mvt", "mean"),
                             y_med=("TAXITIME_SEC_mvt", "median"), nm_unmatched=("FLIGHT_ID_mvt", lambda s: int(s.isna().sum())))
print(g.to_string())
ap = dep.pivot_table(index="ADEP_mvt", columns="month", values="MVT_ID_mvt", aggfunc="size", observed=True)
print(ap.to_string())
# days per month per airport (missing days?)
dd = dep.groupby(["ADEP_mvt", "month"], observed=True)["day"].nunique().unstack()
print("days present per airport-month\n", dd.to_string())
rk = pd.read_parquet("data/ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt"])
rk["month"] = rk["MVT_TIME_UTC_mvt"].dt.month
print("ranking", rk.groupby(["PHASE_mvt", "month"], observed=True).size().to_string())
