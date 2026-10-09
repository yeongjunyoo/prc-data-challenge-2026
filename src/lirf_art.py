# SPDX-License-Identifier: GPL-3.0-only
"""LIRF NM 매칭 출발 전용 art(BLOCK=SCHED) 분류기. 전역 혼합의 p_art 를 LIRF NM 행에서만 바꾼다.

pred_L = p_L * d_sch + (1 - p_L) * mu   (mu 는 같은 설정의 model3 결과 파일에서 그대로)
피처와 그룹 통계는 model3 와 같은 함수(build, add_te, feature_cols, to_matrix)로 만든다.

--setting val : 1, 7월 제외 학습 -> 1, 7월. cache/val_<base>_s<seed>.parquet 의 mu 를 쓰고 결과를 cache/val_<out>_s<seed>.parquet 로.
--setting full: 2025 전체 학습 -> ranking. cache/full_<base>_s<seed>.parquet 의 mu.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.common import breakdown, feature_cols, rmse  # noqa: E402
from src.feats import CAT_FEATS, build  # noqa: E402
from src.load import CACHE  # noqa: E402
from src.model3 import to_matrix  # noqa: E402
from src.te import add_te  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", default="val", choices=["val", "full"])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--base", default="m3w")
    pa.add_argument("--out", default="m3wL")
    pa.add_argument("--wpow", type=float, default=1.0)
    pa.add_argument("--leaves", type=int, default=31)
    pa.add_argument("--mdl", type=int, default=50)
    pa.add_argument("--lr", type=float, default=0.03)
    pa.add_argument("--threads", type=int, default=4)
    args = pa.parse_args()
    t0 = time.time()
    F = build()
    va = F["src"].isin([1, 7]).to_numpy()
    te = (F["src"] == 0).to_numpy()
    fit_rows, eval_rows = (~va & ~te, va) if args.setting == "val" else (~te, te)
    T = add_te(F, fit_rows)
    F = pd.concat([F, T], axis=1)
    del T
    cols = feature_cols(F)
    cat_idx = [j for j, c in enumerate(cols) if c in CAT_FEATS]
    L = ((F["ap"] == "LIRF") & (F["nm"] == 1)).to_numpy()
    y = F["y"].to_numpy().astype(float)
    dsch = F["d_sch"].to_numpy().astype(float)
    art = (np.abs(y - dsch) <= 60).astype(float)
    ids = F["MVT_ID_mvt"].to_numpy()
    lf, le = fit_rows & L, eval_rows & L
    Xf = to_matrix(F, cols, lf)
    Xe = to_matrix(F, cols, le)
    del F
    w = lambda m: (np.clip(np.abs(dsch[m] - 1000.0), 300.0, 60000.0) / 1000.0) ** args.wpow
    params = dict(objective="binary", learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=args.mdl,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=args.threads,
                  verbose=-1, cat_smooth=20.0, min_data_per_group=30, max_cat_threshold=64, seed=args.seed,
                  feature_pre_filter=False)
    dtr = lgb.Dataset(Xf, art[lf], weight=w(lf), categorical_feature=cat_idx, params=params)
    iters_path = CACHE / f"iters_{args.out}.json"
    if args.setting == "val":
        dva = lgb.Dataset(Xe, art[le], weight=w(le), reference=dtr, categorical_feature=cat_idx, params=params)
        m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        it = m.best_iteration
        if args.seed == 0:
            iters_path.write_text(json.dumps({"clf": it}))
        p_L = m.predict(Xe, num_iteration=it)
    else:
        it = int(json.loads(iters_path.read_text())["clf"] * 1.15)
        m = lgb.train(params, dtr, num_boost_round=it)
        p_L = m.predict(Xe)
    print(f"[{time.time() - t0:.0f}s] {args.setting} LIRF-NM rows fit {int(lf.sum())} eval {int(le.sum())} iters {it}")
    prev = pd.read_parquet(CACHE / f"{args.setting}_{args.base}_s{args.seed}.parquet")
    pos = pd.Series(np.arange(len(prev)), index=prev["MVT_ID_mvt"].to_numpy())
    idx = pos.reindex(ids[le]).to_numpy()
    assert not np.isnan(idx).any()
    idx = idx.astype(int)
    new = prev.copy()
    mu = prev["mu"].to_numpy()[idx]
    ds = prev["dsch"].to_numpy()[idx]
    p_L = np.where(ds < 0, 0.0, p_L)
    new.loc[new.index[idx], "p_art"] = p_L
    new.loc[new.index[idx], "pred"] = p_L * ds + (1 - p_L) * mu
    new["pred"] = np.clip(new["pred"].to_numpy(), 0, None)
    if args.setting == "val":
        yv = new["y"].to_numpy()
        old = np.clip(prev["pred"].to_numpy(), 0, None)
        n = len(yv)
        print(f"  holdout RMSE base {rmse(old, yv):.2f} -> with LIRF art model {rmse(new['pred'], yv):.2f}")
        print(f"  LIRF NM rows MSE contrib {((old[idx] - yv[idx]) ** 2).sum() / n:,.0f} -> {((new['pred'].to_numpy()[idx] - yv[idx]) ** 2).sum() / n:,.0f}")
        a = art[le]
        print(f"  mean p on art rows {prev['p_art'].to_numpy()[idx][a == 1].mean():.3f} -> {p_L[a == 1].mean():.3f} | "
              f"on non-art rows {prev['p_art'].to_numpy()[idx][a == 0].mean():.3f} -> {p_L[a == 0].mean():.3f}")
        print(breakdown(new["pred"].to_numpy(), yv, pd.read_parquet(CACHE / "feats.parquet", columns=["MVT_ID_mvt", "ap"]).set_index("MVT_ID_mvt")["ap"].reindex(new["MVT_ID_mvt"]).astype(str).to_numpy(), args.out))
    out = CACHE / f"{args.setting}_{args.out}_s{args.seed}.parquet"
    new.to_parquet(out, index=False)
    print(f"  saved {out.name}")


if __name__ == "__main__":
    main()
