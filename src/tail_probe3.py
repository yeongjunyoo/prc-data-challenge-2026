# SPDX-License-Identifier: GPL-3.0-only
"""꼬리의 정체 3단계: 어느 시각이 BLOCK 자리를 대신하나(앵커), 현지 날짜 어긋남 규칙, 그리고 시험에서 보이는 칸으로 가르는 정도."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.load import load_all  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
TZ = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London", "EHAM": "Europe/Amsterdam",
      "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid", "LFPG": "Europe/Paris", "LIRF": "Europe/Rome",
      "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul"}


def local_day(t: pd.Series, ap: pd.Series) -> np.ndarray:
    out = np.full(len(t), np.nan)
    for a, z in TZ.items():
        m = (ap == a).to_numpy() & t.notna().to_numpy()
        if m.any():
            ts = pd.to_datetime(t[m], unit="s", utc=True).dt.tz_convert(z)
            out[m] = (ts.dt.tz_localize(None).dt.normalize() - pd.Timestamp("2025-01-01")).dt.days.to_numpy()
    return out


df = load_all()
dep = df[df["PHASE_mvt"] == "DEP"].copy()
del df
dep["ld_mvt"] = local_day(dep["t_mvt"], dep["ADEP_mvt"])
dep["ld_sch"] = local_day(dep["t_sch"], dep["ADEP_mvt"])
dep["dd"] = dep["ld_mvt"] - dep["ld_sch"]
tr = dep[dep["src"] > 0].copy()
rk = dep[dep["src"] == 0]

y = tr["y"]
anch = {"AOBT3": "t_aobt3", "SCHED": "t_sch", "EOBT1": "t_eobt1", "IOBT": "t_iobt", "LOBT": "t_lobt"}
for k, c in anch.items():
    tr["e_" + k] = (tr["t_blk"] - tr[c]).abs()
ecols = ["e_" + k for k in anch]
E = tr[ecols].fillna(1e12)
tr["best"] = E.idxmin(axis=1).str[2:]
tr["best_err"] = E.min(axis=1)
tr["ybin"] = pd.cut(y, [-1e9, 1800, 3600, 7200, 80000, 1e12])
print("nearest anchor to BLOCK, share by y bin (only where best_err <= 120 s else 'none'):")
tr["best2"] = np.where(tr["best_err"] <= 120, tr["best"], "none")
print(pd.crosstab(tr["ybin"], tr["best2"], normalize="index").round(3))
print(pd.crosstab(tr["ybin"], tr["best2"]))

# 날짜 어긋남(현지): MVT 의 현지 날짜가 SCHED 의 현지 날짜보다 늦은 행
print("\nrows by local day diff (MVT - SCHED), train:", tr["dd"].value_counts().sort_index().to_dict())
print("rows by local day diff, ranking:", rk["dd"].value_counts().sort_index().to_dict())
x = tr[tr["dd"] >= 1].copy()
x["kind"] = np.select([x["y"] >= 80000, (x["y"] - (x["t_mvt"] - x["t_sch"])).abs() <= 60, x["y"] < 20000],
                      ["shift+1d", "=MVT-SCHED", "normal"], "other")
print(pd.crosstab(x["ADEP_mvt"], x["kind"]))
print("dd>=1 & NM present:", pd.crosstab(x["FLIGHT_ID_mvt"].notna(), x["kind"]))
print("ranking dd>=1 by airport", rk[rk["dd"] >= 1].groupby("ADEP_mvt", observed=True).size().to_dict())
print("ranking dd>=1 & no NM by airport", rk[(rk["dd"] >= 1) & rk["FLIGHT_ID_mvt"].isna()].groupby("ADEP_mvt", observed=True).size().to_dict())

# 꼬리를 보이는 칸으로 가르기: 꼬리(y>=3600) 판별 AUC 를 칸 하나씩
from sklearn.metrics import roc_auc_score  # noqa: E402
t = (tr["y"] >= 3600).astype(int)
feats = {"MVT-SCHED": tr["t_mvt"] - tr["t_sch"], "MVT-AOBT3": tr["t_mvt"] - tr["t_aobt3"], "MVT-LOBT": tr["t_mvt"] - tr["t_lobt"],
         "MVT-IOBT": tr["t_mvt"] - tr["t_iobt"], "MVT-EOBT1": tr["t_mvt"] - tr["t_eobt1"], "AOBT3-LOBT": tr["t_aobt3"] - tr["t_lobt"],
         "noNM": tr["FLIGHT_ID_mvt"].isna().astype(float)}
for k, v in feats.items():
    vv = v.fillna(v.median())
    print(f"AUC tail vs {k}: {roc_auc_score(t, vv):.3f}")

# 꼬리 행에서 y 가 어느 보이는 차와 60초 안에서 맞는가(보이는 정보로 정답이 '정확히' 나오는 몫)
tail = tr[tr["y"] >= 3600]
hit = pd.DataFrame({k: ((tail["y"] - (tail["t_mvt"] - tail[c])).abs() <= 60) for k, c in anch.items()})
hit["shift1d_sched"] = ((tail["y"] - 86400 * tail["dd"] - 0).abs() < 3600) & (tail["dd"] >= 1)
print("\ntail rows", len(tail), "share where y within 60 s of MVT-anchor:", hit.mean().round(3).to_dict(),
      "| any:", round(hit.any(axis=1).mean(), 3))
print(tail.assign(anyhit=hit.any(axis=1)).groupby("ADEP_mvt", observed=True)["anyhit"].agg(["size", "mean"]).round(3).T)
