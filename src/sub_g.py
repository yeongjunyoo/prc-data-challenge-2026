# SPDX-License-Identifier: GPL-3.0-only
"""G 그룹(LIRF, NM 미매칭) 출발의 기대값 모형. 세 가지 기록 유형의 확률을 낸다.

  art   : BLOCK 이 SCHED 로 적힘        -> y = 이륙 - SCHED
  shift : BLOCK 날짜가 SCHED 현지 날짜   -> y = 86400*dd + 택시
  normal: 제대로 적힘                    -> y = 택시
E[y] = pa*d_sch + ps*(86400*max(dd,1) + tau) + pn*tau_n

비교(달 단위 12겹 교차검증, 그리고 로컬 자 그대로 1, 7월 제외 학습 -> 1, 7월):
  lookup : (이륙-SCHED 구간 x 날짜 바뀜) 칸의 평활 빈도
  lookup+pfx : 위 + 접두사(항공사) art 비율로 art 몫 조정
  gbm    : 작은 LightGBM 다중분류
"""
from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.load import CACHE  # noqa: E402

N_VALID = 344419
BINS = [-1e12, 3600, 14400, 57600, 1e12]


def labels(y, dsch):
    art = np.abs(y - dsch) <= 60
    shift = (y > 60000) & ~art
    return np.where(art, 1, np.where(shift, 2, 0))


def cell(dsch, dd):
    b = np.digitize(dsch, BINS[1:-1])  # 0..3
    return b * 2 + (dd >= 1).astype(int)


def fit_lookup(c_tr, k_tr, alpha=2.0):
    """칸별 세 유형 빈도. 전체 빈도를 사전분포로 alpha 만큼 섞는다. 날짜가 안 바뀐 칸의 shift 는 0."""
    glob_ = np.bincount(k_tr, minlength=3) / len(k_tr)
    tab = {}
    for c in range(8):
        m = c_tr == c
        cnt = np.bincount(k_tr[m], minlength=3).astype(float)
        prior = glob_.copy()
        if c % 2 == 0:
            prior[2] = 0.0
            prior = prior / prior.sum()
        p = (cnt + alpha * prior) / (cnt.sum() + alpha)
        tab[c] = p
    return tab


def predict_lookup(tab, c_te):
    return np.vstack([tab[c] for c in c_te])


def pfx_adjust(P, pfx_te, pfx_tr, k_tr, kappa=5.0):
    """접두사의 art 비율(평활)과 전체 art 비율의 로짓 차를 art 몫에 더한다. 나머지 둘은 비율을 유지."""
    base = (k_tr == 1).mean()
    d = pd.DataFrame({"p": pfx_tr, "a": (k_tr == 1).astype(float)}).groupby("p")["a"].agg(["sum", "count"])
    rate = (d["sum"] + kappa * base) / (d["count"] + kappa)
    r = pd.Series(pfx_te).map(rate).fillna(base).to_numpy()
    lo = np.log(r / (1 - r)) - np.log(base / (1 - base))
    pa = P[:, 1]
    la = np.log(np.clip(pa, 1e-6, 1 - 1e-6) / np.clip(1 - pa, 1e-6, 1)) + lo
    pa2 = 1 / (1 + np.exp(-la))
    rest = P[:, [0, 2]]
    rest = rest / np.clip(rest.sum(axis=1, keepdims=True), 1e-12, None) * (1 - pa2)[:, None]
    return np.column_stack([rest[:, 0], pa2, rest[:, 1]])


def expected(P, dsch, dd, tau_n, tau_s=1000.0):
    return P[:, 1] * dsch + P[:, 2] * (86400 * np.maximum(dd, 1) + tau_s) + P[:, 0] * tau_n


def gbm_probs(X_tr, k_tr, X_te, seed=0):
    d = lgb.Dataset(X_tr, k_tr, categorical_feature=["pfx"], free_raw_data=True)
    m = lgb.train(dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=6, min_data_in_leaf=15,
                       feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, cat_smooth=10,
                       min_data_per_group=10, verbose=-1, num_threads=4, seed=seed), d, num_boost_round=200)
    return m.predict(X_te)


def load_g():
    cols = ["MVT_ID_mvt", "src", "y", "ap", "nm", "d_sch", "dd", "prefix", "hour_mvt", "hour_sch", "garb", "fl_len", "stand"]
    F = pd.read_parquet(CACHE / "feats.parquet", columns=cols)
    G = F[(F["ap"] == "LIRF") & (F["nm"] == 0)].copy()
    G["pfx"] = G["prefix"].astype(str)
    G["stand_d"] = pd.to_numeric(G["stand"].astype(str).str[5:6], errors="coerce")
    return G


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    G = load_g()
    T = G[G["src"] > 0].copy()
    y = T["y"].to_numpy()
    dsch = T["d_sch"].to_numpy().astype(float)
    dd = T["dd"].to_numpy().astype(float)
    k = labels(y, dsch)
    c = cell(dsch, dd)
    m = T["src"].to_numpy()
    pfx = T["pfx"].to_numpy()
    feat = ["d_sch", "dd", "hour_mvt", "hour_sch", "garb", "fl_len", "stand_d"]
    X = T[feat].astype("float32").copy()
    X["pfx"] = T["pfx"].astype("category")
    print("G train rows", len(T), "kinds normal/art/shift", np.bincount(k, minlength=3).tolist())
    for setting in ["cv12", "local"]:
        folds = [[mo] for mo in range(1, 13)] if setting == "cv12" else [[1, 7]]
        res = {n: np.full(len(T), np.nan) for n in ["const", "lookup", "lookup_pfx", "gbm", "oracle"]}
        for f in folds:
            te_ = np.isin(m, f)
            tr_ = ~te_ if setting == "cv12" else ~np.isin(m, [1, 7])
            tau_n = y[tr_ & (k == 0)].mean()
            glob_ = np.bincount(k[tr_], minlength=3) / tr_.sum()
            res["const"][te_] = expected(np.tile(glob_, (te_.sum(), 1)), dsch[te_], dd[te_], tau_n)
            tab = fit_lookup(c[tr_], k[tr_])
            P = predict_lookup(tab, c[te_])
            res["lookup"][te_] = expected(P, dsch[te_], dd[te_], tau_n)
            P2 = pfx_adjust(P, pfx[te_], pfx[tr_], k[tr_])
            res["lookup_pfx"][te_] = expected(P2, dsch[te_], dd[te_], tau_n)
            P3 = gbm_probs(X[tr_], k[tr_], X[te_])
            res["gbm"][te_] = expected(P3, dsch[te_], dd[te_], tau_n)
            oh = np.eye(3)[k[te_]]
            res["oracle"][te_] = expected(oh, dsch[te_], dd[te_], tau_n)
        ev = np.isin(m, sum(folds, []))
        scale = (2 / 12) if setting == "cv12" else 1.0
        out = []
        for n, p in res.items():
            se = (p[ev] - y[ev]) ** 2
            out.append(f"{n}: MSE contrib {se.sum() * scale / N_VALID:,.0f}")
        print(f"[{setting}] " + " | ".join(out))


if __name__ == "__main__":
    main()
