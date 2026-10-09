# SPDX-License-Identifier: GPL-3.0-only
"""F2 deliverable 2: G rows (LIRF NM-unmatched) and copy (art) share by month; big-label share by month."""
import glob, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
cols = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt", "FLIGHT_ID_mvt"]
tr = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/training_*.parquet"))], ignore_index=True)
dep = tr[tr["PHASE_mvt"] == "DEP"].copy()
dep["month"] = dep["MVT_TIME_UTC_mvt"].dt.month
y = dep["TAXITIME_SEC_mvt"].astype(float)
dsch = (dep["MVT_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
dep["art"] = (np.abs(y - dsch) <= 60).astype(int)
dep["big"] = (y > 3600).astype(int)
dep["G"] = ((dep["ADEP_mvt"] == "LIRF") & dep["FLIGHT_ID_mvt"].isna()).astype(int)
print("G rows by month:", dep.groupby("month")["G"].sum().to_dict())
print("y>3600 share by month (%):", (dep.groupby("month")["big"].mean() * 100).round(3).to_dict())
print("y>3600 SSE share proxy: sum(y^2 | y>3600)/sum(y^2)")
dep["y2"] = y ** 2
s = dep.groupby("month").apply(lambda d: d.loc[d.big == 1, "y2"].sum() / d["y2"].sum()).round(3)
print(s.to_dict())
art = dep.pivot_table(index="ADEP_mvt", columns="month", values="art", aggfunc="mean", observed=True) * 100
print("art (copy) share % by airport-month\n", art.round(1).to_string())
sd = dep.pivot_table(index="ADEP_mvt", columns="month", values="TAXITIME_SEC_mvt", aggfunc="mean", observed=True)
print("mean label by airport-month\n", sd.round(0).to_string())
