# SPDX-License-Identifier: GPL-3.0-only
"""피처 v2.5 는 편명(FLIGHT_mvt) 파생 칸(prefix, fl_len)을 뺀다. G 그룹 모형의 항공사 신호를 목적지(ADES)로 바꿔도 되는지 잰다.

비교(12겹 달 교차검증 두 달 환산, 로컬 1, 7월): 접두사 보정 대 목적지 보정 대 보정 없음, GBM 의 범주 칸도 같은 식.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src import sub_g  # noqa: E402
from src.load import CACHE  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
F = pd.read_parquet(CACHE / "feats_v24.parquet", columns=["src", "y", "ap", "nm", "d_sch", "dd", "prefix", "ades", "hour_mvt",
                                                          "hour_sch", "garb", "fl_len", "stand"])
G = F[(F["ap"] == "LIRF") & (F["nm"] == 0) & (F["src"] > 0)].copy()
del F
G["pfx"] = G["prefix"].astype(str).fillna("nan")
G["ad"] = G["ades"].astype(str).fillna("nan")
G["stand_d"] = pd.to_numeric(G["stand"].astype(str).fillna("nan").str[5:6], errors="coerce")
y = G["y"].to_numpy()
dsch = G["d_sch"].to_numpy().astype(float)
dd = G["dd"].to_numpy().astype(float)
k = sub_g.labels(y, dsch)
c = sub_g.cell(dsch, dd)
m = G["src"].to_numpy()
N = sub_g.N_VALID


def run(key, gbm_feats, setting):
    folds = [[mo] for mo in range(1, 13)] if setting == "cv12" else [[1, 7]]
    out = {}
    P1 = np.full((len(G), 3), np.nan)
    P2 = np.full((len(G), 3), np.nan)
    tau = np.full(len(G), np.nan)
    for f in folds:
        te_ = np.isin(m, f)
        tr_ = ~te_ if setting == "cv12" else ~np.isin(m, [1, 7])
        P = sub_g.predict_lookup(sub_g.fit_lookup(c[tr_], k[tr_]), c[te_])
        if key is not None:
            kv = G[key].to_numpy()
            P = sub_g.pfx_adjust(P, kv[te_], kv[tr_], k[tr_])
        P1[te_] = P
        X = G[gbm_feats].astype("float32").copy()
        if key is not None:
            X[key] = G[key].astype("category")
        cat = [key] if key is not None else []
        import lightgbm as lgb
        ps = []
        for s in range(3):
            d = lgb.Dataset(X[tr_], k[tr_], categorical_feature=cat, free_raw_data=True)
            mdl = lgb.train(dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=6, min_data_in_leaf=15,
                                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, cat_smooth=10,
                                 min_data_per_group=10, verbose=-1, num_threads=2, seed=s), d, num_boost_round=200)
            ps.append(mdl.predict(X[te_]))
        P2[te_] = np.mean(ps, axis=0)
        tau[te_] = y[tr_ & (k == 0)].mean()
    ev = np.isin(m, sum(folds, []))
    scale = 2 / 12 if setting == "cv12" else 1.0
    for name, P in [("lookup", P1), ("gbm", P2), ("avg", 0.5 * P1 + 0.5 * P2)]:
        pr = sub_g.expected(P[ev], dsch[ev], dd[ev], tau[ev])
        out[name] = round(float(((pr - y[ev]) ** 2).sum() * scale / N))
    return out


base_old = ["d_sch", "dd", "hour_mvt", "hour_sch", "garb", "fl_len", "stand_d"]
base_new = ["d_sch", "dd", "hour_mvt", "hour_sch", "garb", "stand_d"]
for setting in ["cv12", "local"]:
    print(setting, "| v24 (prefix, fl_len):", run("pfx", base_old, setting), "| v25 ADES:", run("ad", base_new, setting),
          "| v25 none:", run(None, base_new, setting))
