# SPDX-License-Identifier: GPL-3.0-only
"""G 그룹 모형의 작은 격자: 평활 강도(alpha, kappa), GBM 반복 수, 두 모형 섞는 비율. 기준은 12겹 달 교차검증(두 달 환산 MSE 기여)."""
import itertools
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src import sub_g  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
G = sub_g.load_g()
G["pfx"] = G["pfx"].fillna("nan")
T = G[G["src"] > 0].copy()
y = T["y"].to_numpy()
dsch = T["d_sch"].to_numpy().astype(float)
dd = T["dd"].to_numpy().astype(float)
k = sub_g.labels(y, dsch)
c = sub_g.cell(dsch, dd)
m = T["src"].to_numpy()
pfx = T["pfx"].to_numpy()
feat = ["d_sch", "dd", "hour_mvt", "hour_sch", "garb", "fl_len", "stand_d"]
X = T[feat].astype("float32").copy()
X["pfx"] = T["pfx"].astype("category")
N = sub_g.N_VALID

# GBM 확률은 설정마다 한 번만 계산(시드 3개 평균)
gbm_P = {}
for rounds in [100, 200, 400]:
    P = np.zeros((len(T), 3))
    for mo in range(1, 13):
        te_ = m == mo
        tr_ = ~te_
        import lightgbm as lgb
        ps = []
        for s in range(3):
            d = lgb.Dataset(X[tr_], k[tr_], categorical_feature=["pfx"], free_raw_data=True)
            mdl = lgb.train(dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=6, min_data_in_leaf=15,
                                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, cat_smooth=10,
                                 min_data_per_group=10, verbose=-1, num_threads=2, seed=s), d, num_boost_round=rounds)
            ps.append(mdl.predict(X[te_]))
        P[te_] = np.mean(ps, axis=0)
    gbm_P[rounds] = P

rows = []
for alpha, kappa in itertools.product([1.0, 2.0, 5.0], [3.0, 5.0, 10.0, 20.0]):
    P1 = np.zeros((len(T), 3))
    tau = np.zeros(len(T))
    for mo in range(1, 13):
        te_ = m == mo
        tr_ = ~te_
        tab = sub_g.fit_lookup(c[tr_], k[tr_], alpha=alpha)
        P1[te_] = sub_g.pfx_adjust(sub_g.predict_lookup(tab, c[te_]), pfx[te_], pfx[tr_], k[tr_], kappa=kappa)
        tau[te_] = y[tr_ & (k == 0)].mean()
    for rounds, wgt in itertools.product([100, 200, 400], [0.0, 0.25, 0.5, 0.75, 1.0]):
        P = wgt * P1 + (1 - wgt) * gbm_P[rounds]
        pr = sub_g.expected(P, dsch, dd, tau)
        se = (pr - y) ** 2
        loc = np.isin(m, [1, 7])
        rows.append(dict(alpha=alpha, kappa=kappa, rounds=rounds, w_lookup=wgt, cv12=se.sum() * 2 / 12 / N, local_1_7=se[loc].sum() / N))
R = pd.DataFrame(rows).sort_values("cv12")
print(R.head(15).round(0).to_string())
print("current (alpha 2, kappa 5, rounds 200, w 0.5):")
print(R[(R.alpha == 2) & (R.kappa == 5) & (R.rounds == 200) & (R.w_lookup == 0.5)].round(0).to_string())
