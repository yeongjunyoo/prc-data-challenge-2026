# SPDX-License-Identifier: GPL-3.0-only
"""LIRF NM 미매칭 출발만 떼어 BLOCK=SCHED(art) 를 얼마나 가를 수 있는지 잰다(달 단위 교차검증).

비교: (a) 그룹 하나의 평균 비율, (b) 접두사 비율, (c) 작은 LightGBM. 지표는 logloss, AUC, 그리고
예측 p*(이륙-SCHED)+(1-p)*tau 로 이 행들의 제곱오차 합(전체 검증 MSE 로 환산).
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, ".")
from src.feats import build  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
F = build()
G = F[(F["ap"] == "LIRF") & (F["nm"] == 0) & (F["src"] > 0)].copy()
del F
y = G["y"].to_numpy()
dsch = G["d_sch"].to_numpy().astype(float)
art = (np.abs(y - dsch) <= 60).astype(int)
shift = ((y > 60000) & (art == 0)).astype(int)
m = G["src"].to_numpy()
G["stand_d"] = G["stand"].astype(str).str[5:6]
G["pfx"] = G["prefix"].astype(str)
print("rows", len(G), "art", art.sum(), "shift", shift.sum())
N_VALID = 344419  # 로컬 자 행 수(MSE 환산용)

feat = ["d_sch", "dd", "hour_mvt", "hour_sch", "dow", "garb", "fl_len", "c_dep_m60", "c_arr_m60", "nb_dsch_m60", "nb_daobt_m60",
        "taxiing_now", "st_prev_arr_gap", "st_next_arr_gap", "st_prev_dep_gap", "st_next_dep_sch_gap", "c_takeoff_in_taxi_sch",
        "nb_taxiin_m60", "c_day"]
cat = ["pfx", "stand_d", "ades_reg", "rwy"]
X = G[feat].astype("float32").copy()
for c in cat:
    X[c] = G[c].astype(str).astype("category")
p_grp = np.zeros(len(G))
p_pfx = np.zeros(len(G))
p_gbm = np.zeros(len(G))
tau = np.zeros(len(G))
for mo in range(1, 13):
    te_ = m == mo
    tr_ = ~te_
    p_grp[te_] = art[tr_].mean()
    tab = pd.DataFrame({"p": G["pfx"].to_numpy()[tr_], "a": art[tr_]}).groupby("p")["a"].agg(["sum", "count"])
    prior = art[tr_].mean()
    r = (tab["sum"] + 10 * prior) / (tab["count"] + 10)
    p_pfx[te_] = pd.Series(G["pfx"].to_numpy()[te_]).map(r).fillna(prior).to_numpy()
    tau[te_] = np.mean(y[tr_ & (art == 0) & (shift == 0)])
    d = lgb.Dataset(X[tr_], art[tr_], categorical_feature=cat)
    mdl = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=8, min_data_in_leaf=20, feature_fraction=0.8,
                         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, cat_smooth=10, min_data_per_group=10,
                         verbose=-1, num_threads=4, seed=0), d, num_boost_round=300)
    p_gbm[te_] = mdl.predict(X[te_])
for name, p in [("group", p_grp), ("prefix", p_pfx), ("gbm", p_gbm)]:
    pr = p * dsch + (1 - p) * tau
    sse = ((pr - y) ** 2).sum() / 12 * 2  # 두 달치로 환산
    print(f"{name:7s} logloss {log_loss(art, np.clip(p, 1e-4, 1 - 1e-4)):.4f} AUC {roc_auc_score(art, p):.3f} "
          f"| SSE/2mo -> MSE contrib {sse / N_VALID:.0f} | rows d_sch>14400 SSE share {((pr - y) ** 2)[dsch > 14400].sum() / ((pr - y) ** 2).sum():.2f}")
or_ = art * dsch + (1 - art) * tau
print("oracle art (shift rows still missed): MSE contrib", round(((or_ - y) ** 2).sum() / 12 * 2 / N_VALID), "| shift rows SSE share",
      round(((or_ - y) ** 2)[shift == 1].sum() / ((or_ - y) ** 2).sum(), 2))
