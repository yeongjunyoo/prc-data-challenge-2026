# SPDX-License-Identifier: GPL-3.0-only
"""꼬리의 정체 1단계: 검증 MSE 를 정답 구간별, 공항별로 나누고 꼬리 행의 시간차 분포를 본다."""
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
hold = tr["src"].isin([1, 7])
y = tr["y"]
med = tr[~hold].groupby("ADEP_mvt", observed=True)["y"].median()
tr["p_med"] = tr["ADEP_mvt"].map(med).astype(float)
h = tr[hold].copy()
n = len(h)
se = (h["p_med"] - h["y"]) ** 2
print("holdout rows", n, "median RMSE", round(np.sqrt(se.mean()), 2))

bins = [-1e9, 0, 600, 1200, 1800, 3600, 7200, 86400 * 0.9, 1e12]
h["ybin"] = pd.cut(h["y"], bins)
g = h.assign(se=se).groupby("ybin", observed=True).agg(n=("y", "size"), y_mean=("y", "mean"), se_sum=("se", "sum"))
g["share_rows"] = (g["n"] / n).round(5)
g["share_mse"] = (g["se_sum"] / se.sum()).round(4)
g["rmse_if_removed"] = [round(np.sqrt((se.sum() - s) / n), 1) for s in g["se_sum"]]
print(g.drop(columns="se_sum"))

ga = h.assign(se=se).groupby("ADEP_mvt", observed=True).agg(n=("y", "size"), y_med=("y", "median"), y_mean=("y", "mean"),
                                                          se_sum=("se", "sum"), n_tail=("y", lambda s: int((s >= 3600).sum())))
ga["share_mse"] = (ga["se_sum"] / se.sum()).round(4)
ga["rmse"] = np.sqrt(ga["se_sum"] / ga["n"]).round(1)
print(ga.drop(columns="se_sum"))

# 꼬리 전체(2025 전 달) 의 시간차들
tail = tr[tr["y"] >= 3600]
print("\ntrain tail rows (y>=3600)", len(tail), "share", round(len(tail) / len(tr), 5), "max", tail["y"].max())
print("y quantiles over all train DEP:", tr["y"].quantile([0.5, 0.9, 0.99, 0.999, 0.9999]).to_dict())
print("negative y", int((tr["y"] < 0).sum()), "zero y", int((tr["y"] == 0).sum()))
for name, a in [("MVT-AOBT3", "t_aobt3"), ("MVT-EOBT1", "t_eobt1"), ("MVT-SCHED", "t_sch"), ("MVT-IOBT", "t_iobt"),
                ("MVT-LOBT", "t_lobt")]:
    d_all = tr["t_mvt"] - tr[a]
    d_t = tail["t_mvt"] - tail[a]
    print(f"{name:10s} present all {d_all.notna().mean():.3f} tail {d_t.notna().mean():.3f} | "
          f"tail q: {d_t.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(0).tolist()} | all q: {d_all.quantile([0.05, 0.5, 0.95]).round(0).tolist()}")
for name, a in [("BLK-AOBT3", "t_aobt3"), ("BLK-EOBT1", "t_eobt1"), ("BLK-SCHED", "t_sch"), ("BLK-IOBT", "t_iobt"), ("BLK-LOBT", "t_lobt")]:
    d_all = tr["t_blk"] - tr[a]
    d_t = tail["t_blk"] - tail[a]
    print(f"{name:10s} tail q: {d_t.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(0).tolist()} | all q: {d_all.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(0).tolist()}")

# 하루 단위 어긋남
big = tr[tr["y"] >= 20000]
print("\ny>=20000 rows", len(big), "y mod 86400 near 0 (<3600):", int(((big["y"] % 86400) < 3600).sum()))
print(big["y"].describe())
print("y>=20000 by airport", big.groupby("ADEP_mvt", observed=True).size().to_dict())
print("tail (>=3600) by airport", tail.groupby("ADEP_mvt", observed=True).size().to_dict())
print("tail by month", tail.groupby("src").size().to_dict())

# 중복: 같은 FLIGHT_ID 로 DEP 가 여러 줄
fid = tr["FLIGHT_ID_mvt"]
dup = tr[fid.notna() & fid.duplicated(keep=False)]
print("\nDEP rows sharing FLIGHT_ID", len(dup), "of which tail", int((dup["y"] >= 3600).sum()))
print("tail rows with FLIGHT_ID present", round(tail["FLIGHT_ID_mvt"].notna().mean(), 3), "all", round(fid.notna().mean(), 3))
cols = ["ADEP_mvt", "ADES_mvt", "ADES_flt", "ADES_FILED_flt", "STAND_mvt", "RUNWAY_mvt", "y", "t_mvt", "t_blk", "t_sch", "t_aobt3",
        "t_eobt1", "t_iobt", "t_lobt", "CALLSIGN_flt", "FLIGHT_mvt", "MARKET_SEGMENT_flt"]
s = tail.sample(25, random_state=0)[cols].copy()
for c in ["t_mvt", "t_blk", "t_sch", "t_aobt3", "t_eobt1", "t_iobt", "t_lobt"]:
    s[c] = pd.to_datetime(s[c], unit="s").dt.strftime("%m-%d %H:%M")
print(s.to_string())
