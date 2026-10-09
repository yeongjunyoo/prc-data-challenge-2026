# SPDX-License-Identifier: GPL-3.0-only
"""2차 모델: 피처 + 학습 정답 그룹 통계(OOF) + 혼합.

예측 = p_art * (이륙-SCHED) + (1-p_art) * (mu + p_shift * 86400 * dd)
  p_art   : BLOCK 이 SCHED 와 60초 안일 확률(LightGBM 이진 분류)
  mu      : art 아닌 행으로 학습한 L2 회귀(하루 어긋남 행은 학습에서 뺀다)
  p_shift : 현지 날짜가 바뀐(dd>=1) art 아닌 행 중 BLOCK 날짜가 SCHED 날짜로 적힌 비율(그룹 빈도, 평활)
direct  : 같은 피처로 정답을 바로 맞히는 L2 회귀(비교용, 성격이 다른 차선)

--setting val : 2025 의 1, 7월 제외로 학습, 1, 7월로 잰다(로컬 자). 반복 수를 cache/iters_<tag>.json 에 남긴다.
--setting full: 2025 전체로 학습, ranking 출발 예측. 반복 수는 val 의 1.1 배.
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
from src.te import add_te  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

BASE = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=6, verbose=-1, cat_smooth=20.0,
            min_data_per_group=50, max_cat_threshold=64)


def matrix(F: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = np.empty((len(F), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        s = F[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            v = s.cat.codes.to_numpy().astype(np.float32)
            v[v < 0] = np.nan
            X[:, j] = v
        else:
            X[:, j] = s.to_numpy(dtype=np.float32, na_value=np.nan)
    return X


def train_one(X, y, rows, vrows, params, rounds, cat_idx, fixed_rounds=None, w=None):
    dtr = lgb.Dataset(X[rows], y[rows], weight=None if w is None else w[rows], categorical_feature=cat_idx,
                      free_raw_data=True)
    if fixed_rounds is not None:
        m = lgb.train(params, dtr, num_boost_round=fixed_rounds)
        return m, fixed_rounds
    dva = lgb.Dataset(X[vrows], y[vrows], weight=None if w is None else w[vrows], reference=dtr,
                      categorical_feature=cat_idx, free_raw_data=True)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    return m, m.best_iteration


def shift_rate(F: pd.DataFrame, fit_rows: np.ndarray, art: np.ndarray, shift: np.ndarray, k: float = 5.0) -> np.ndarray:
    """dd>=1 이고 art 가 아닌 행에서 (공항, NM 매칭) 그룹별 하루 어긋남 비율. 나머지 행은 0."""
    dd = F["dd"].to_numpy()
    grp = F["ap"].astype(str).to_numpy() + "|" + F["nm"].astype(str).to_numpy()
    elig = (dd >= 1)
    base = fit_rows & elig & (art == 0)
    d = pd.DataFrame({"g": grp[base], "s": shift[base].astype(float)})
    st = d.groupby("g")["s"].agg(["sum", "count"])
    prior = float(d["s"].mean()) if len(d) else 0.0
    rate = (st["sum"] + prior * k) / (st["count"] + k)
    out = pd.Series(grp).map(rate).fillna(prior).to_numpy()
    return np.where(elig, out, 0.0)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", default="val", choices=["val", "full"])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--rounds", type=int, default=5000)
    pa.add_argument("--tag", default="m2")
    pa.add_argument("--no_te", action="store_true")
    pa.add_argument("--direct", action="store_true", help="직접 회귀도 학습")
    pa.add_argument("--lr", type=float, default=0.05)
    pa.add_argument("--leaves", type=int, default=127)
    pa.add_argument("--mdl", type=int, default=200)
    pa.add_argument("--wpow", type=float, default=0.0)
    pa.add_argument("--wcap", type=float, default=60000.0)
    args = pa.parse_args()
    t00 = time.time()
    F = build()
    va = F["src"].isin([1, 7]).to_numpy()
    te = (F["src"] == 0).to_numpy()
    if args.setting == "val":
        fit_rows = ~va & ~te
        eval_rows = va
    else:
        fit_rows = ~te
        eval_rows = te
    if not args.no_te:
        T = add_te(F, fit_rows)
        F = pd.concat([F, T], axis=1)
        del T
    cols = feature_cols(F)
    cat_idx = [j for j, c in enumerate(cols) if c in CAT_FEATS]
    y = F["y"].to_numpy().astype(float)
    dsch = F["d_sch"].to_numpy().astype(float)
    dd = F["dd"].to_numpy().astype(float)
    apv = F["ap"].astype(str).to_numpy()
    mvt_id = F["MVT_ID_mvt"].to_numpy()
    art = (np.abs(y - dsch) <= 60).astype(float)
    shift = ((y > 60000) & (art == 0) & (dd >= 1)).astype(float)
    p_shift = shift_rate(F, fit_rows, art, shift)
    X = matrix(F, cols)
    del F
    print(f"[{args.tag} {args.setting} seed{args.seed}] features {len(cols)} fit {int(fit_rows.sum())} eval {int(eval_rows.sum())} "
          f"prep {time.time() - t00:.0f}s")
    params = dict(BASE, seed=args.seed, learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=args.mdl)
    iters_path = CACHE / f"iters_{args.tag}.json"
    fixed = None
    if args.setting == "full":
        fixed = json.loads(iters_path.read_text())
    # 1) art 분류. 가중치: 잘못 가를 때 제곱오차가 (이륙-SCHED - 1000)^2 에 비례하므로 그 크기로 무게를 준다(--wpow)
    t0 = time.time()
    wc = None
    if args.wpow > 0:
        wc = (np.clip(np.abs(dsch - 1000.0), 300.0, args.wcap) / 1000.0) ** args.wpow
    m_c, it_c = train_one(X, art, fit_rows, eval_rows, dict(params, objective="binary"), args.rounds, cat_idx,
                          None if fixed is None else int(fixed["clf"] * 1.1), w=wc)
    p_art = m_c.predict(X[eval_rows], num_iteration=it_c if fixed is None else None)
    print(f"  clf iters {it_c} {time.time() - t0:.0f}s")
    # 2) 나머지 회귀
    t0 = time.time()
    rr = fit_rows & (art == 0) & (shift == 0)
    vr = eval_rows & (art == 0) & (shift == 0)
    m_r, it_r = train_one(X, y, rr, vr, dict(params, objective="regression"), args.rounds, cat_idx,
                          None if fixed is None else int(fixed["reg"] * 1.1))
    mu = m_r.predict(X[eval_rows], num_iteration=it_r if fixed is None else None)
    print(f"  reg iters {it_r} {time.time() - t0:.0f}s")
    ps = p_shift[eval_rows]
    pred = p_art * dsch[eval_rows] + (1 - p_art) * (mu + ps * 86400 * np.maximum(dd[eval_rows], 1))
    res = {"MVT_ID_mvt": mvt_id[eval_rows], "pred": pred, "p_art": p_art, "mu": mu, "p_shift": ps,
           "dsch": dsch[eval_rows]}
    iters = {"clf": it_c, "reg": it_r}
    if args.direct:
        t0 = time.time()
        m_d, it_d = train_one(X, y, fit_rows, eval_rows, dict(params, objective="regression"), args.rounds, cat_idx,
                              None if fixed is None else int(fixed["direct"] * 1.1))
        res["direct"] = m_d.predict(X[eval_rows], num_iteration=it_d if fixed is None else None)
        iters["direct"] = it_d
        print(f"  direct iters {it_d} {time.time() - t0:.0f}s")
    if args.setting == "val":
        yv = y[eval_rows]
        res["y"] = yv
        print(breakdown(pred, yv, apv[eval_rows], f"{args.tag} mixture"))
        if args.direct:
            print(breakdown(res["direct"], yv, apv[eval_rows], f"{args.tag} direct"))
        a = art[eval_rows]
        po = a * dsch[eval_rows] + (1 - a) * (mu + ps * 86400 * np.maximum(dd[eval_rows], 1))
        print(f"  oracle-art ceiling {rmse(po, yv):.2f} | no-shift-term {rmse(p_art * dsch[eval_rows] + (1 - p_art) * mu, yv):.2f}")
        imp = pd.Series(m_c.feature_importance("gain"), index=cols).sort_values(ascending=False)
        print("  clf top gain:", (imp / imp.sum()).head(12).round(3).to_dict())
        imp = pd.Series(m_r.feature_importance("gain"), index=cols).sort_values(ascending=False)
        print("  reg top gain:", (imp / imp.sum()).head(12).round(3).to_dict())
        if args.seed == 0:
            iters_path.write_text(json.dumps(iters))
    out = CACHE / f"{args.setting}_{args.tag}_s{args.seed}.parquet"
    pd.DataFrame(res).to_parquet(out, index=False)
    print(f"  saved {out.name} total {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
