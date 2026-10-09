# SPDX-License-Identifier: GPL-3.0-only
"""꼬리 6단계: 중복 행과 재기록, 같은 편명이나 콜사인의 도착(회항, 회전 운항)이 꼬리와 관련 있나."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.load import CACHE  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
cols = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "t_mvt", "t_blk", "t_sch", "y", "src",
        "CALLSIGN_flt", "STAND_mvt"]
df = pd.read_parquet(CACHE / "all.parquet", columns=cols)
df["fl"] = df["FLIGHT_mvt"].astype(str)
dep = df[df["PHASE_mvt"] == "DEP"].copy()
arr = df[df["PHASE_mvt"] == "ARR"].copy()
tr = dep[dep["src"] > 0].copy()
tr["tail"] = tr["y"] >= 3600
tr["huge"] = tr["y"] >= 40000
tr["dsch"] = tr["t_mvt"] - tr["t_sch"]

# 1) 같은 출발 공항, 같은 편명, 같은 SCHED 의 중복 DEP
k = tr["ADEP_mvt"].astype(str) + "|" + tr["fl"] + "|" + tr["t_sch"].astype(str)
dupk = k.duplicated(keep=False)
print("DEP duplicates on (ADEP, flight, SCHED):", int(dupk.sum()), "tail among them", int(tr.loc[dupk, "tail"].sum()))
# 같은 출발 공항, 같은 편명, SCHED 36시간 안에 두 번 이상
tr = tr.sort_values(["ADEP_mvt", "fl", "t_sch"])
same = (tr["ADEP_mvt"].astype(str) == tr["ADEP_mvt"].astype(str).shift()) & (tr["fl"] == tr["fl"].shift()) & (tr["fl"] != "nan")
gap = tr["t_sch"] - tr["t_sch"].shift()
tr["prev_same_fl_gap"] = np.where(same, gap, np.nan)
same2 = (tr["ADEP_mvt"].astype(str) == tr["ADEP_mvt"].astype(str).shift(-1)) & (tr["fl"] == tr["fl"].shift(-1)) & (tr["fl"] != "nan")
tr["next_same_fl_gap"] = np.where(same2, tr["t_sch"].shift(-1) - tr["t_sch"], np.nan)
near = (tr["prev_same_fl_gap"] < 36 * 3600) | (tr["next_same_fl_gap"] < 36 * 3600)
print("DEP with same flight number from same airport within 36h of SCHED:", int(near.sum()), "| tail share among them",
      round(tr.loc[near, "tail"].mean(), 4), "vs overall", round(tr["tail"].mean(), 4), "| huge among them", int(tr.loc[near, "huge"].sum()),
      "of", int(tr["huge"].sum()))
# 2) 회항: 같은 공항에서 출발 후 같은 공항으로 도착(ADEP==ADES 이거나 같은 편명이 같은 공항에 도착)
ret = arr[arr["ADEP_mvt"].astype(str) == arr["ADES_mvt"].astype(str)]
print("ARR rows with ADEP==ADES (return to origin):", len(ret), "by airport", ret.groupby("ADES_mvt", observed=True).size().to_dict())
a2 = arr[["ADES_mvt", "fl", "t_mvt"]].rename(columns={"ADES_mvt": "ap", "t_mvt": "t_arr"})
a2["ap"] = a2["ap"].astype(str)
d2 = tr[["ADEP_mvt", "fl", "t_mvt", "t_blk", "y", "tail", "huge", "MVT_ID_mvt"]].rename(columns={"ADEP_mvt": "ap"})
d2["ap"] = d2["ap"].astype(str)
m = d2.merge(a2, on=["ap", "fl"], how="inner")
m = m[(m["fl"] != "nan") & (m["t_arr"] > m["t_blk"] - 3600) & (m["t_arr"] < m["t_mvt"])]
print("DEP whose same flight number ARRIVED at the same airport between its off-block-1h and takeoff:", m["MVT_ID_mvt"].nunique(),
      "tail", int(m.drop_duplicates("MVT_ID_mvt")["tail"].sum()), "huge", int(m.drop_duplicates("MVT_ID_mvt")["huge"].sum()))
# 시험에서 볼 수 있는 쪽: 이륙 전 같은 편명이 같은 공항에 도착한 기록이 있나(BLOCK 없이 SCHED-6h ~ 이륙)
d3 = tr[["ADEP_mvt", "fl", "t_mvt", "t_sch", "y", "tail", "huge", "MVT_ID_mvt"]].rename(columns={"ADEP_mvt": "ap"})
d3["ap"] = d3["ap"].astype(str)
m3 = d3.merge(a2, on=["ap", "fl"], how="inner")
m3 = m3[(m3["fl"] != "nan") & (m3["t_arr"] > m3["t_sch"] - 6 * 3600) & (m3["t_arr"] < m3["t_mvt"])]
u = m3.drop_duplicates("MVT_ID_mvt")
print("visible version (same flight number ARR at same airport in [SCHED-6h, takeoff)):", len(u), "tail", int(u["tail"].sum()), "huge", int(u["huge"].sum()),
      "| y mean", round(u["y"].mean(), 0))
