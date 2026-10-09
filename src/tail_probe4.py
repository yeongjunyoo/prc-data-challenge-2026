# SPDX-License-Identifier: GPL-3.0-only
"""꼬리 4단계: SCHED 대체(BLOCK=SCHED)가 날짜에 몰리는가, 도착편에도 같은 대체가 있는가(시험에서 도착은 BLOCK 이 보인다)."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.load import load_all  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)

df = load_all()
df["day"] = (df["t_mvt"] // 86400).astype("int64")
df["eq_sch"] = ((df["t_blk"] - df["t_sch"]).abs() <= 6)
df["dsch"] = np.where(df["PHASE_mvt"] == "DEP", df["t_mvt"] - df["t_sch"], df["t_blk"] - df["t_sch"])
dep = df[(df["PHASE_mvt"] == "DEP") & (df["src"] > 0)]
arr = df[df["PHASE_mvt"] == "ARR"]
# 도착 BLOCK=SCHED 대체 비율(학습과 시험)
for ap in ["LIRF", "EGLL", "LFPG", "LTFM", "EDDF"]:
    a_tr = arr[(arr["ADES_mvt"] == ap) & (arr["src"] > 0)]
    a_te = arr[(arr["ADES_mvt"] == ap) & (arr["src"] == 0)]
    d_tr = dep[dep["ADEP_mvt"] == ap]
    big = d_tr["dsch"].abs() > 1800
    print(f"{ap}: ARR eq_sch train {a_tr['eq_sch'].mean():.4f} test {a_te['eq_sch'].mean():.4f} | "
          f"ARR eq_sch where |inblk-sch|... DEP eq_sch(train) {d_tr['eq_sch'].mean():.4f} | DEP eq_sch among |MVT-SCHED|>1800: {d_tr.loc[big, 'eq_sch'].mean():.4f}")
    # 도착: 대체라면 in-block = SCHED 인데 착륙(MVT)이 SCHED 와 멀다
    a_big_tr = a_tr[(a_tr["t_mvt"] - a_tr["t_sch"]).abs() > 1800]
    a_big_te = a_te[(a_te["t_mvt"] - a_te["t_sch"]).abs() > 1800]
    print(f"     ARR eq_sch among |landing-SCHED|>1800: train {a_big_tr['eq_sch'].mean():.4f} (n={len(a_big_tr)}) "
          f"test {a_big_te['eq_sch'].mean():.4f} (n={len(a_big_te)})")

# LIRF 날짜별 몰림
d = dep[dep["ADEP_mvt"] == "LIRF"].copy()
d["art"] = d["eq_sch"] & (d["dsch"] > 1800)
d["late"] = d["dsch"] > 1800
g = d.groupby("day").agg(n=("art", "size"), art=("art", "sum"), late=("late", "sum"), nonm=("FLIGHT_ID_mvt", lambda s: s.isna().sum()))
a = arr[(arr["ADES_mvt"] == "LIRF") & (arr["src"] > 0)].copy()
a["art"] = a["eq_sch"] & ((a["t_mvt"] - a["t_sch"]).abs() > 1800)
ga = a.groupby("day").agg(arr_art=("art", "sum"), arr_n=("art", "size"))
g = g.join(ga)
print("\nLIRF days: corr(dep art, arr art)", round(g["art"].corr(g["arr_art"]), 3), "| corr(dep art, nonm)", round(g["art"].corr(g["nonm"]), 3),
      "| corr(dep art, late)", round(g["art"].corr(g["late"]), 3))
print("LIRF dep art per day quantiles", g["art"].quantile([0.5, 0.9, 0.99, 1]).to_dict(), "days with >=5:", int((g["art"] >= 5).sum()))
print(g.sort_values("art", ascending=False).head(12).to_string())
# LIRF 대체 행: NM 매칭 여부
print("\nLIRF art rows by NM:", d.groupby(d["FLIGHT_ID_mvt"].notna())["art"].agg(["size", "sum", "mean"]).to_string())
# 시험 LIRF 미매칭 행의 MVT-SCHED 분포 vs 학습
te = df[(df["PHASE_mvt"] == "DEP") & (df["src"] == 0) & (df["ADEP_mvt"] == "LIRF")]
for nm, s in [("train noNM", d[d["FLIGHT_ID_mvt"].isna()]["dsch"]), ("test noNM", te[te["FLIGHT_ID_mvt"].isna()]["dsch"]),
              ("train NM", d[d["FLIGHT_ID_mvt"].notna()]["dsch"]), ("test NM", te[te["FLIGHT_ID_mvt"].notna()]["dsch"])]:
    print(f"{nm:11s} n={len(s)} MVT-SCHED q: {s.quantile([0.1, 0.5, 0.75, 0.9, 0.95, 0.99]).round(0).tolist()} >3600: {(s > 3600).sum()} >7200: {(s > 7200).sum()}")
