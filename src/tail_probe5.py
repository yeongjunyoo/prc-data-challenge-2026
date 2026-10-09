# SPDX-License-Identifier: GPL-3.0-only
"""꼬리 5단계: LIRF 의 NM 미매칭 출발에서 SCHED 대체(art)와 정상 행을 무엇이 가르나."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.load import load_all  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 20)

df = load_all()
dep = df[df["PHASE_mvt"] == "DEP"].copy()
arr = df[df["PHASE_mvt"] == "ARR"]
del df
dep["dsch"] = dep["t_mvt"] - dep["t_sch"]
L = dep[(dep["ADEP_mvt"] == "LIRF")].copy()
L["art"] = (L["y"] - L["dsch"]).abs() <= 60
L["kind"] = np.select([L["art"], L["y"] >= 80000, L["y"] < 3600], ["art", "shift", "normal"], "longtaxi")
N = L[L["FLIGHT_ID_mvt"].isna() & (L["src"] > 0)].copy()
print("LIRF noNM train kinds:", N["kind"].value_counts().to_dict())
N["blk_m_sch"] = N["t_blk"] - N["t_sch"]
print("non-art noNM: BLK-SCHED quantiles", N.loc[~N["art"], "blk_m_sch"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(0).tolist())
print("non-art noNM: y quantiles", N.loc[~N["art"], "y"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(0).tolist())
for c in ["AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "FLIGHT_RULE_mvt"]:
    t = N.groupby(N[c].astype(str))["art"].agg(["size", "sum", "mean"]).round(2)
    print(c, t.sort_values("sum", ascending=False).head(12).to_dict("index"))
N["stand1"] = N["STAND_mvt"].astype(str).str[:1]
t = N.groupby(N["stand1"])["art"].agg(["size", "sum", "mean"]).round(2); print("stand first char", t.to_dict("index"))
N["pfx"] = N["FLIGHT_mvt"].astype(str).str[:3]
t = N.groupby(N["pfx"])["art"].agg(["size", "sum", "mean"]).round(2)
print("prefix", t.sort_values("sum", ascending=False).head(20).to_dict("index"))
N["sec_mvt"] = N["t_mvt"] % 60
N["hour"] = ((N["t_mvt"] % 86400) // 3600)
N["dsch_bin"] = pd.cut(N["dsch"], [-1e9, 1800, 3600, 7200, 14400, 28800, 1e9])
t = N.groupby(N["dsch_bin"])["art"].agg(["size", "sum", "mean"]).round(2); print("dsch bin", t.to_dict("index"))
t = N.groupby(N["hour"])["art"].agg(["size", "sum", "mean"]).round(2); print("utc hour", t["mean"].to_dict())
N["month"] = N["src"]
t = N.groupby(N["month"])["art"].agg(["size", "sum", "mean"]).round(2); print("month", t.to_dict("index"))

# 같은 편명(앞 3자 + 숫자 부분) 의 다른 출발 행이 가까이 있는가
def core(s):
    s = s.astype(str)
    return s.str.replace(r"^([A-Z]{3})\1?", "", regex=True)

dep["fl"] = dep["FLIGHT_mvt"].astype(str)
L2 = dep[dep["ADEP_mvt"] == "LIRF"][["MVT_ID_mvt", "fl", "ADES_mvt", "t_mvt", "t_sch", "FLIGHT_ID_mvt", "src"]].copy()
# 같은 목적지, 같은 SCHED 날짜에서 다른 출발 행(편명 다름 포함)
L2["sday"] = (L2["t_sch"] // 86400).astype("int64")
key = L2["ADES_mvt"].astype(str) + "|" + L2["sday"].astype(str)
L2["n_same_ades_sday"] = key.map(key.value_counts())
N = N.join(L2[["n_same_ades_sday"]])
t = N.groupby(N["n_same_ades_sday"].clip(upper=4))["art"].agg(["size", "sum", "mean"]).round(2)
print("same ADES same sched-day DEP count", t.to_dict("index"))

cols = ["src", "FLIGHT_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "y", "dsch", "blk_m_sch", "kind"]
S = N.copy()
for c in ["t_mvt", "t_sch", "t_blk"]:
    S[c] = pd.to_datetime(S[c], unit="s").dt.strftime("%m-%d %H:%M:%S")
print(S[S["kind"] == "art"].sample(15, random_state=2)[cols + ["t_mvt", "t_sch", "t_blk"]].to_string())
print(S[S["kind"] == "normal"].sample(15, random_state=2)[cols + ["t_mvt", "t_sch", "t_blk"]].to_string())
