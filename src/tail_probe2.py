# SPDX-License-Identifier: GPL-3.0-only
"""꼬리의 정체 2단계: BLOCK 이 SCHED 와 같은 행(기록 대체), 초 단위 반올림, 하루 어긋남, NM 미매칭 행."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.load import load_all  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

df = load_all()
dep = df[df["PHASE_mvt"] == "DEP"].copy()
del df
tr = dep[dep["src"] > 0].copy()
rk = dep[dep["src"] == 0].copy()

tr["blk_sec0"] = (tr["t_blk"] % 60 == 0)
tr["mvt_sec0"] = (tr["t_mvt"] % 60 == 0)
tr["sch_sec0"] = (tr["t_sch"] % 60 == 0)
tr["blk_eq_sch"] = (tr["t_blk"] == tr["t_sch"])
tr["blk_near_sch"] = ((tr["t_blk"] - tr["t_sch"]).abs() <= 60)
tr["blk_eq_aobt"] = (tr["t_blk"] == tr["t_aobt3"])
tr["nm"] = tr["FLIGHT_ID_mvt"].notna()
tr["tail"] = tr["y"] >= 3600
rk["mvt_sec0"] = (rk["t_mvt"] % 60 == 0)
rk["sch_sec0"] = (rk["t_sch"] % 60 == 0)
rk["nm"] = rk["FLIGHT_ID_mvt"].notna()

a = tr.groupby("ADEP_mvt", observed=True).agg(
    n=("y", "size"), blk_sec0=("blk_sec0", "mean"), mvt_sec0=("mvt_sec0", "mean"), sch_sec0=("sch_sec0", "mean"),
    blk_eq_sch=("blk_eq_sch", "mean"), blk_near_sch=("blk_near_sch", "mean"), blk_eq_aobt=("blk_eq_aobt", "mean"),
    nm=("nm", "mean"), tail=("tail", "mean"))
b = rk.groupby("ADEP_mvt", observed=True).agg(n_rk=("y", "size"), mvt_sec0_rk=("mvt_sec0", "mean"),
                                            sch_sec0_rk=("sch_sec0", "mean"), nm_rk=("nm", "mean"))
print(a.join(b).round(4).to_string())

print("\nshare of tail rows with BLK==SCHED", round(tr.loc[tr["tail"], "blk_eq_sch"].mean(), 4),
      "| with |BLK-SCHED|<=60", round(tr.loc[tr["tail"], "blk_near_sch"].mean(), 4))
print("BLK==SCHED rows: y quantiles", tr.loc[tr["blk_eq_sch"], "y"].quantile([0.1, 0.5, 0.9, 0.99]).to_dict())
for ap in ["LIRF", "EGLL", "LFPG", "LTFM"]:
    s = tr[tr["ADEP_mvt"] == ap]
    t = s[s["tail"]]
    print(ap, "tail", len(t), "BLK==SCHED", int(t["blk_eq_sch"].sum()), "near", int(t["blk_near_sch"].sum()),
          "no NM", int((~t["nm"]).sum()), "| blk_sec0 in tail", round(t["blk_sec0"].mean(), 3), "overall", round(s["blk_sec0"].mean(), 3))

# 기록 대체 가설: LIRF 에서 BLK==SCHED 이면 y == MVT - SCHED
s = tr[tr["ADEP_mvt"] == "LIRF"]
x = s["t_mvt"] - s["t_sch"]
print("\nLIRF: y vs MVT-SCHED for BLK==SCHED rows, frac of all LIRF rows", round(s["blk_eq_sch"].mean(), 4))
for lo, hi in [(-1e9, 0), (0, 600), (600, 1200), (1200, 1800), (1800, 3600), (3600, 7200), (7200, 1e9)]:
    m = (x > lo) & (x <= hi)
    print(f"  MVT-SCHED in ({lo},{hi}] n={int(m.sum())} BLK==SCHED share {s.loc[m, 'blk_eq_sch'].mean():.3f} "
          f"near {s.loc[m, 'blk_near_sch'].mean():.3f} y mean {s.loc[m, 'y'].mean():.0f} no-NM share {(~s.loc[m, 'nm']).mean():.3f}")
# 같은 표를 전 공항으로
x = tr["t_mvt"] - tr["t_sch"]
print("\nALL airports by MVT-SCHED:")
for lo, hi in [(-1e9, 0), (0, 600), (600, 1200), (1200, 1800), (1800, 3600), (3600, 7200), (7200, 1e9)]:
    m = (x > lo) & (x <= hi)
    print(f"  ({lo},{hi}] n={int(m.sum())} BLK==SCHED {tr.loc[m, 'blk_eq_sch'].mean():.3f} near {tr.loc[m, 'blk_near_sch'].mean():.3f} "
          f"y mean {tr.loc[m, 'y'].mean():.0f} tail share {tr.loc[m, 'tail'].mean():.3f}")

# 하루 어긋남 행 전부
big = tr[tr["y"] >= 40000].copy()
cols = ["src", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt", "y", "t_mvt", "t_blk", "t_sch", "t_aobt3", "t_eobt1",
        "FLIGHT_mvt", "CALLSIGN_flt", "AIRCRAFT_TYPE_mvt"]
big["mvt_m_sch"] = big["t_mvt"] - big["t_sch"]
big["blk_m_sch"] = big["t_blk"] - big["t_sch"]
for c in ["t_mvt", "t_blk", "t_sch", "t_aobt3", "t_eobt1"]:
    big[c] = pd.to_datetime(big[c], unit="s").dt.strftime("%m-%d %H:%M:%S")
print("\ny>=40000 rows:", len(big))
print(big[cols + ["mvt_m_sch", "blk_m_sch"]].sort_values("y").to_string())

# 음수 y
neg = tr[tr["y"] < 0]
print("\nnegative y rows", len(neg), neg.groupby("ADEP_mvt", observed=True).size().to_dict(), "min", neg["y"].min())
