# SPDX-License-Identifier: GPL-3.0-only
"""XGBoost leg: an XGBoost regression of mu on the same rows and features as src/model3.py, mixed half and half with the base's mu.

G rows keep the base's values. Row split, early-stopping days, iteration rule, --drop and --anchor as in src/cat_mu.py; CAT_FEATS enter
as categories. The parameters were fixed before any score (depth 8, eta 0.05, min_child_weight 50, subsample and colsample 0.8,
lambda 10, max_bin 256, early stopping after 200 rounds). It has weight 0 in the final blend, whose definition still reads its file.

    python src/xgb_mu.py --setting val --seed 0 --tag <tag> --base <model3 tag> --feats <feature table>
Output: cache/<setting>_<tag>_s<seed>.parquet (MVT_ID_mvt, pred, mu_xgb, mu, p_art, g, dsch and, except for full, y) and the
early-stopping rows.

Original note (Korean):
셋째 학습기 후보: XGBoost 회귀(mu)를 model3, cat_mu 와 같은 행, 같은 피처로 GPU 에서 학습해 바탕 모형(--base)의 mu 와 반씩 섞는다.

  pred = clip(p_art * d_sch + (1 - p_art) * mu_b, 0)   mu_b = (1 - W) * mu_base + W * mu_xgb   (W = 0.5, 점수 전에 고정)
  G 행(LIRF NM 미매칭)은 바탕 모형의 값을 그대로 둔다. 셋을 고르게 섞을 때는 이 출력의 mu_xgb 를 blend_mu --col mu_xgb --w 0.333333 으로 쓴다.
행 나누기, 조기 종료 날, 반복 수 규칙(val 과 fwd 는 제 학습 범위 안 내부 조기 종료, full 은 val 이 찾은 반복 수 x1.1), --drop, --anchor 는 src/cat_mu.py 와 같다.
CAT_FEATS 칸은 XGBoost 의 범주형(enable_categorical)으로 넣는다. 파라미터는 점수 전에 고정했다(depth 8, eta 0.05, min_child_weight 50,
subsample 0.8, colsample_bytree 0.8, lambda 10, max_bin 256, 조기 종료 200, 상한 --iters).
출력: cache/<setting>_<tag>_s<seed>.parquet (MVT_ID_mvt, pred, mu_xgb, mu, p_art, g, dsch, [y]), preds/es_valid_ids_<tag>_<setting>_s<seed>.parquet
    .venv/Scripts/python.exe -u src/xgb_mu.py --setting val --seed 0 --tag m6r25xg --base m6r25 --feats feats_v25
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, ".")
from src.cat_mu import frame  # noqa: E402
from src.common import breakdown, feature_cols, rmse  # noqa: E402
from src.feats import CAT_FEATS, build  # noqa: E402
from src.load import CACHE  # noqa: E402
from src.model3 import REFIT, es_day_mask  # noqa: E402
from src.te import add_te  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
W = 0.5
T0 = time.time()


def log(msg: str):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def xframe(F: pd.DataFrame, cols: list[str], rows: np.ndarray, ncat: dict[str, int]) -> pd.DataFrame:
    """The integer codes of cat_mu.frame (missing = -1) as pandas categories over one category set for each column in ncat,
    so that the training, early-stopping and evaluated rows share one code table.

    Original note (Korean):
    cat_mu.frame 의 정수 코드(결측 -1)를 같은 범주 집합의 pandas 범주형으로 바꾼다(학습, 조기 종료, 평가 행이 같은 코드표를 쓴다).
    """
    X = frame(F, cols, rows)
    for c, k in ncat.items():
        X[c] = pd.Categorical.from_codes(X[c].to_numpy(), categories=np.arange(k))
    return X


def main():
    """Command line entry: fit the XGBoost regression for one setting and seed and write the outputs of the module docstring."""
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", default="val", choices=["val", "fwd", "full"])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--base", required=True, help="섞을 출력의 태그(cache/<setting>_<base>_s<seed>.parquet, mu 와 p_art 칸이 있어야 한다)")
    pa.add_argument("--feats", default="feats_v25")
    pa.add_argument("--iters", type=int, default=20000)
    pa.add_argument("--drop", default="", help="쉼표로 뺄 피처 칸(model3 --drop 과 같다)")
    pa.add_argument("--anchor", default="", choices=["", "aobt", "win"], help="model3 --anchor 와 같은 회귀 목표")
    args = pa.parse_args()
    F = build(name=args.feats)
    if args.drop:
        F = F.drop(columns=[c for c in args.drop.split(",") if c in F.columns])
    src = F["src"].to_numpy()
    te = src == 0
    if args.setting == "val":
        fit_rows, eval_rows = ~np.isin(src, [1, 7]) & ~te, np.isin(src, [1, 7])
    elif args.setting == "fwd":
        fit_rows, eval_rows = np.isin(src, [1, 2, 3, 4, 5, 6]), src == 7
    else:
        fit_rows, eval_rows = ~te, te
    es_rows = fit_rows & es_day_mask(F["MVT_ID_mvt"].to_numpy()) if args.setting != "full" else np.zeros(len(F), bool)
    if args.setting != "full":
        pd.DataFrame({"MVT_ID_mvt": F["MVT_ID_mvt"].to_numpy()[es_rows]}).to_parquet(
            CACHE.parent / "preds" / f"es_valid_ids_{args.tag}_{args.setting}_s{args.seed}.parquet", index=False)
    T = add_te(F, fit_rows)
    F = pd.concat([F, T], axis=1)
    del T
    cols = feature_cols(F, None)
    ncat = {c: len(F[c].cat.categories) for c in cols if c in CAT_FEATS and isinstance(F[c].dtype, pd.CategoricalDtype)}
    y_all = F["y"].to_numpy().astype(float)
    dsch_all = F["d_sch"].to_numpy().astype(float)
    art_all = np.abs(y_all - dsch_all) <= 60
    shift_all = (y_all > 60000) & ~art_all
    reg_all = fit_rows & ~art_all & ~shift_all  # model3 의 회귀 행(조기 종료 날 포함한 학습 범위)
    if args.anchor == "aobt":  # model3 --anchor aobt 와 같은 기준값
        da = F["d_aobt"].to_numpy().astype(float)
        apv = F["ap"].astype(str).to_numpy()
        nfit = fit_rows & ~art_all & ~shift_all & (y_all < 3600)
        med = pd.Series(y_all[nfit]).groupby(apv[nfit]).median()
        fb = pd.Series(apv).map(med).fillna(float(np.median(y_all[nfit]))).to_numpy()
        anchor_all = np.where(np.isnan(da), fb, np.clip(da, 0, 7200))
    elif args.anchor == "win":  # LOBT 창 기준(vault research/2026-10-06-r5-paradigms.md): 공항별 정상 학습 행 중앙값을 창 [d_lobt - 3606, d_lobt + 3606] 안으로
        dl = F["d_lobt"].to_numpy().astype(float)
        apv = F["ap"].astype(str).to_numpy()
        nfit = fit_rows & ~art_all & ~shift_all & (y_all < 3600)
        med = pd.Series(y_all[nfit]).groupby(apv[nfit]).median()
        fb = pd.Series(apv).map(med).fillna(float(np.median(y_all[nfit]))).to_numpy()
        anchor_all = np.where(np.isnan(dl), fb, np.clip(fb, dl - 3606.0, dl + 3606.0))
    else:
        anchor_all = np.zeros(len(F))
    tgt = y_all - anchor_all
    ids_e = F["MVT_ID_mvt"].to_numpy()[eval_rows]
    ap_e = F["ap"].astype(str).to_numpy()[eval_rows]
    tr = reg_all & ~es_rows
    va = reg_all & es_rows
    Xe = xframe(F, cols, eval_rows, ncat)
    params = dict(objective="reg:squarederror", tree_method="hist", device="cuda", max_depth=8, learning_rate=0.05, min_child_weight=50,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, max_bin=256, seed=args.seed)
    log(f"[{args.tag} {args.setting} seed{args.seed}] features {len(cols)} (cat {len(ncat)}) reg rows {int(reg_all.sum())} es {int(va.sum())} eval {len(Xe)}")
    iters_path = CACHE / f"iters_{args.tag}.json"
    if args.setting == "full":  # fwd 는 제 학습 범위(1~6월) 안의 내부 조기 종료로 찾는다(10-04 닫힘 감사 2회차, cat_mu 와 같은 고침)
        n_it = max(1, round(json.loads(iters_path.read_text())["xgb"] * REFIT))
    else:
        dtr = xgb.QuantileDMatrix(xframe(F, cols, tr, ncat), tgt[tr], enable_categorical=True, max_bin=256)
        dva = xgb.QuantileDMatrix(xframe(F, cols, va, ncat), tgt[va], enable_categorical=True, ref=dtr)
        m = xgb.train(params, dtr, num_boost_round=args.iters, evals=[(dva, "es")], early_stopping_rounds=200, verbose_eval=500)
        best = int(m.best_iteration) + 1
        log(f"xgb best iters {best}")
        if args.setting == "val" and args.seed == 0:
            iters_path.write_text(json.dumps({"xgb": best, "refit_factor": REFIT, "cap": args.iters, "params": {k: v for k, v in params.items() if k != "seed"}}))
        n_it = max(1, round(best * REFIT))
        del m, dtr, dva
    dall = xgb.QuantileDMatrix(xframe(F, cols, reg_all, ncat), tgt[reg_all], enable_categorical=True, max_bin=256)
    del F
    m = xgb.train(params, dall, num_boost_round=n_it)
    del dall
    mu_xgb = anchor_all[eval_rows] + m.inplace_predict(Xe)
    log(f"refit on {int(reg_all.sum())} regression rows with {n_it} iters")
    b = pd.read_parquet(CACHE / f"{args.setting}_{args.base}_s{args.seed}.parquet")
    assert np.array_equal(b["MVT_ID_mvt"].to_numpy(), ids_e), "바탕 모형 출력과 평가 행 순서가 다르다"
    mu_b = (1 - W) * b["mu"].to_numpy() + W * mu_xgb
    pred = np.clip(b["p_art"].to_numpy() * b["dsch"].to_numpy() + (1 - b["p_art"].to_numpy()) * mu_b, 0, None)
    g = b["g"].to_numpy().astype(bool)
    pred[g] = b["pred"].to_numpy()[g]
    res = {"MVT_ID_mvt": ids_e, "pred": pred, "mu_xgb": mu_xgb, "mu": b["mu"].to_numpy(), "p_art": b["p_art"].to_numpy(),
           "g": g.astype(np.int8), "dsch": b["dsch"].to_numpy()}
    if "y" in b.columns:
        ye = b["y"].to_numpy()
        res["y"] = ye
        print(breakdown(pred, ye, ap_e, f"{args.tag} {args.setting} mixture+G, mu = base/xgb 0.5"))
        print(f"  base {args.base} {rmse(b['pred'].to_numpy(), ye):.2f} | mu_xgb alone in mixture "
              f"{rmse(np.where(g, b['pred'].to_numpy(), np.clip(b['p_art'].to_numpy() * b['dsch'].to_numpy() + (1 - b['p_art'].to_numpy()) * mu_xgb, 0, None)), ye):.2f}")
        imp = pd.Series(m.get_score(importance_type="total_gain")).sort_values(ascending=False)
        print("  xgb top gain:", (imp / imp.sum()).head(12).round(3).to_dict())
    out = CACHE / f"{args.setting}_{args.tag}_s{args.seed}.parquet"
    pd.DataFrame(res).to_parquet(out, index=False)
    log(f"saved {out.name}")


if __name__ == "__main__":
    main()
