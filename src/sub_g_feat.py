# SPDX-License-Identifier: GPL-3.0-only
"""G 그룹 GBM 에 스탠드 순서와 교통량 피처를 더하면 나아지는가(12겹 달 교차검증, 로컬 1, 7월 둘 다)."""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src import sub_g  # noqa: E402
from src.load import CACHE  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
BASE = ["d_sch", "dd", "hour_mvt", "hour_sch", "garb", "fl_len", "stand_d"]
EXTRA = ["st_next_dep_sch_gap", "st_arr_between_sch_mvt", "st_next_arr_gap", "st_prev_arr_gap", "c_arr_m60", "c_dep_p60"]
F = pd.read_parquet(CACHE / "feats.parquet")
G = F[(F["ap"] == "LIRF") & (F["nm"] == 0) & (F["src"] > 0)].copy()
del F
G["pfx"] = G["prefix"].astype(str).fillna("nan")
G["stand_d"] = pd.to_numeric(G["stand"].astype(str).fillna("nan").str[5:6], errors="coerce")
y = G["y"].to_numpy()
dsch = G["d_sch"].to_numpy().astype(float)
dd = G["dd"].to_numpy().astype(float)
k = sub_g.labels(y, dsch)
c = sub_g.cell(dsch, dd)
m = G["src"].to_numpy()
pfx = G["pfx"].to_numpy()
N = sub_g.N_VALID


def run(feat, setting):
    X = G[feat].astype("float32").copy()
    X["pfx"] = G["pfx"].astype("category")
    folds = [[mo] for mo in range(1, 13)] if setting == "cv12" else [[1, 7]]
    P1 = np.full((len(G), 3), np.nan)
    P2 = np.full((len(G), 3), np.nan)
    tau = np.full(len(G), np.nan)
    for f in folds:
        te_ = np.isin(m, f)
        tr_ = ~te_ if setting == "cv12" else ~np.isin(m, [1, 7])
        P1[te_] = sub_g.pfx_adjust(sub_g.predict_lookup(sub_g.fit_lookup(c[tr_], k[tr_]), c[te_]), pfx[te_], pfx[tr_], k[tr_])
        P2[te_] = np.mean([sub_g.gbm_probs(X[tr_], k[tr_], X[te_], seed=s) for s in range(3)], axis=0)
        tau[te_] = y[tr_ & (k == 0)].mean()
    ev = np.isin(m, sum(folds, []))
    scale = 2 / 12 if setting == "cv12" else 1.0
    out = {}
    for name, P in [("gbm", P2), ("avg", 0.5 * P1 + 0.5 * P2)]:
        pr = sub_g.expected(P[ev], dsch[ev], dd[ev], tau[ev])
        out[name] = round(float(((pr - y[ev]) ** 2).sum() * scale / N))
    return out


for setting in ["cv12", "local"]:
    print(setting, "base", run(BASE, setting), "| +extra", run(BASE + EXTRA, setting))
