# SPDX-License-Identifier: GPL-3.0-only
"""1차 모델 비교(로컬 자: 2025-01, 2025-07 출발).

A: LightGBM L2 로 정답을 바로.
B: 혼합. p = P(BLOCK 이 SCHED 와 60초 안 | x) 분류기, 나머지 행으로 L2 회귀 mu. 예측 = p*(이륙-SCHED) + (1-p)*mu.
검증 예측은 cache/val_<이름>.parquet 로 남긴다.
"""
from __future__ import annotations

import argparse
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.common import breakdown, feature_cols, rmse, split  # noqa: E402
from src.feats import build  # noqa: E402
from src.load import CACHE  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

BASE = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=6, verbose=-1, cat_smooth=20.0,
            min_data_per_group=100, max_cat_threshold=64)


def fit(X, y, Xv, yv, params, rounds, w=None):
    dtr = lgb.Dataset(X, y, weight=w, free_raw_data=True)
    dva = lgb.Dataset(Xv, yv, reference=dtr, free_raw_data=True)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    return m


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--mode", default="AB")
    ap_.add_argument("--seed", type=int, default=0)
    ap_.add_argument("--rounds", type=int, default=4000)
    ap_.add_argument("--tag", default="")
    args = ap_.parse_args()
    F = build()
    tr, va, te = split(F)
    cols = feature_cols(F)
    y = F["y"].to_numpy()
    dsch = F["d_sch"].to_numpy().astype(float)
    X = F[cols]
    apv = F["ap"].astype(str).to_numpy()
    print("features", len(cols), "train", int(tr.sum()), "valid", int(va.sum()))
    out = {}
    if "A" in args.mode:
        t0 = time.time()
        p = dict(BASE, objective="regression", seed=args.seed)
        m = fit(X[tr], y[tr], X[va], y[va], p, args.rounds)
        pa = m.predict(X[va], num_iteration=m.best_iteration)
        print(f"A best_iter {m.best_iteration} time {time.time() - t0:.0f}s")
        print(breakdown(pa, y[va], apv[va], "A direct L2"))
        out["A"] = pa
        imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
        print("A top gain:", (imp / imp.sum()).head(20).round(3).to_dict())
    if "B" in args.mode:
        t0 = time.time()
        art = (np.abs(y - dsch) <= 60).astype(float)
        pc = dict(BASE, objective="binary", seed=args.seed)
        mc = fit(X[tr], art[tr], X[va], art[va], pc, args.rounds)
        pr = mc.predict(X[va], num_iteration=mc.best_iteration)
        print(f"B clf best_iter {mc.best_iteration} time {time.time() - t0:.0f}s, valid art rate {art[va].mean():.4f} mean p {pr.mean():.4f}")
        t0 = time.time()
        trr = tr & (art == 0)
        var = va & (art == 0)
        pg = dict(BASE, objective="regression", seed=args.seed)
        mr = fit(X[trr], y[trr], X[var], y[var], pg, args.rounds)
        mu = mr.predict(X[va], num_iteration=mr.best_iteration)
        print(f"B reg best_iter {mr.best_iteration} time {time.time() - t0:.0f}s")
        pb = pr * dsch[va] + (1 - pr) * mu
        print(breakdown(pb, y[va], apv[va], "B mixture"))
        out["B"] = pb
        out["B_p"] = pr
        out["B_mu"] = mu
        # 분류가 완벽하다면(오라클 p) 얼마인가: 천장 확인
        po = art[va] * dsch[va] + (1 - art[va]) * mu
        print(f"  oracle-p ceiling RMSE {rmse(po, y[va]):.2f}")
    if out:
        pd.DataFrame({"MVT_ID_mvt": F.loc[va, "MVT_ID_mvt"].to_numpy(), "y": y[va], **out}).to_parquet(
            CACHE / f"val_model1{args.tag}_s{args.seed}.parquet", index=False)


if __name__ == "__main__":
    main()
