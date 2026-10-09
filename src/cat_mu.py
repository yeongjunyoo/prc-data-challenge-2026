# SPDX-License-Identifier: GPL-3.0-only
"""다른 학습기 후보: CatBoost 회귀(mu)를 model3 와 같은 행, 같은 피처로 학습해 바탕 모형(--base)의 mu 와 반씩 섞는다.

  pred = clip(p_art * d_sch + (1 - p_art) * mu_b, 0)   mu_b = (1 - W) * mu_lgb + W * mu_cat   (W = 0.5, 점수 전에 고정)
  G 행(LIRF NM 미매칭)은 바탕 모형의 값을 그대로 둔다. 바탕 모형은 --eobt_comp 없는 model3 출력이어야 한다(mu 가 art, shift 를 뺀 행의 회귀).
--setting val|fwd|full, --es internal 의 행 나누기는 model3 와 같다(es_day_mask). val 은 내부 조기 종료로 반복 수를 찾은 뒤
학습 범위 전체에 x1.1 로 다시 맞춘다(model3 --refit). fwd 도 제 학습 범위(1~6월) 안의 내부 조기 종료로 반복 수를 찾고(10-04 닫힘 감사 2회차:
val 이 고른 반복 수에는 7월 뒤 달의 정보가 들어 있어 전방 자가 깨끗하지 않았다), full 은 val 이 cache/iters_<tag>.json 에 쓴 반복 수 x1.1 이다.
--drop, --anchor aobt 는 model3 의 같은 이름 옵션과 같다(뺄 칸, 회귀 목표 y - clip(MVT - AOBT_3, 0, 7200)).
출력: cache/<setting>_<tag>_s<seed>.parquet (MVT_ID_mvt, pred, mu_cat, mu, p_art, g, dsch, [y]), preds/es_valid_ids_<tag>_<setting>_s<seed>.parquet
메모리(10-06, 입력과 결과는 그대로): 목표 인코딩 칸을 float32 로 바꿔 붙이고(범주 아닌 피처는 모두 float32 로 읽으므로 값이 같다), 표는 배열을 복사하지 않고
만든다(pandas 3 의 DataFrame(dict) 는 같은 dtype 칸을 한 덩어리로 복사한다. Pool 이 어차피 제 사본을 만든다).
    .venv/Scripts/python.exe -u src/cat_mu.py --setting val --seed 0 --tag m6r25cb --base m6r25 --feats feats_v25 --threads 4
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, ".")
from src.common import breakdown, feature_cols, rmse  # noqa: E402
from src.feats import CAT_FEATS, build  # noqa: E402
from src.load import CACHE  # noqa: E402
from src.model3 import REFIT, es_day_mask  # noqa: E402
from src.te import add_te  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
W = 0.5
T0 = time.time()


def release() -> None:
    """Free the reference cycles that keep the feature table's blocks alive and return Arrow's unused memory to the system (10-06:
    1.1 GiB of working set on the fwd setting). Memory only; nothing the model sees changes."""
    gc.collect()
    pa.default_memory_pool().release_unused()


def log(msg: str):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def frame(F: pd.DataFrame, cols: list[str], rows: np.ndarray) -> pd.DataFrame:
    """CatBoost 입력: CAT_FEATS 범주 칸은 정수 코드(결측 -1), 다른 범주형 칸은 model3.to_matrix 처럼 코드(결측 NaN), 나머지는 float32."""
    out = {}
    for c in cols:
        s = F[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            v = s.cat.codes.to_numpy()[rows]
            out[c] = v.astype(np.int32) if c in CAT_FEATS else np.where(v < 0, np.nan, v).astype(np.float32)
        else:
            out[c] = s.to_numpy(dtype=np.float32, na_value=np.nan)[rows]
    return pd.DataFrame(out, copy=False)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", default="val", choices=["val", "fwd", "full"])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--base", required=True, help="섞을 model3 출력의 태그(cache/<setting>_<base>_s<seed>.parquet)")
    pa.add_argument("--feats", default="feats_v25")
    pa.add_argument("--iters", type=int, default=4000)
    pa.add_argument("--lr", type=float, default=0.1)
    pa.add_argument("--depth", type=int, default=8)
    pa.add_argument("--threads", type=int, default=4)
    pa.add_argument("--drop", default="", help="쉼표로 뺄 피처 칸(model3 --drop 과 같다)")
    pa.add_argument("--anchor", default="", choices=["", "aobt", "win"], help="model3 --anchor 와 같은 회귀 목표")
    pa.add_argument("--gpu", action="store_true", help="CatBoost 를 GPU(장치 0)로 학습한다. CPU 판과 결과가 같지 않다")
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
    T = add_te(F, fit_rows).astype(np.float32)
    F = pd.concat([F, T], axis=1)
    del T
    release()
    cols = feature_cols(F, None)
    cats = [c for c in cols if c in CAT_FEATS]
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
    Xe = frame(F, cols, eval_rows)
    params = dict(loss_function="RMSE", depth=args.depth, learning_rate=args.lr, l2_leaf_reg=10.0, border_count=254,
                  thread_count=args.threads, random_seed=args.seed, allow_writing_files=False, verbose=250)
    if args.gpu:
        params.update(task_type="GPU", devices="0")
    log(f"[{args.tag} {args.setting} seed{args.seed}] features {len(cols)} (cat {len(cats)}) reg rows {int(reg_all.sum())} es {int(va.sum())} eval {len(Xe)}")
    iters_path = CACHE / f"iters_{args.tag}.json"
    if args.setting == "full":
        n_it = max(1, round(json.loads(iters_path.read_text())["cat"] * REFIT))
    else:
        m = CatBoostRegressor(iterations=args.iters, od_type="Iter", od_wait=200, use_best_model=True, **params)
        m.fit(Pool(frame(F, cols, tr), tgt[tr], cat_features=cats), eval_set=Pool(frame(F, cols, va), tgt[va], cat_features=cats))
        best = int(m.get_best_iteration()) + 1
        log(f"cat best iters {best}")
        if args.setting == "val" and args.seed == 0:
            iters_path.write_text(json.dumps({"cat": best, "refit_factor": REFIT, "lr": args.lr, "depth": args.depth, "gpu": args.gpu, "cap": args.iters}))
        n_it = max(1, round(best * REFIT))
        del m
    m = CatBoostRegressor(iterations=n_it, **params)
    m.fit(Pool(frame(F, cols, reg_all), tgt[reg_all], cat_features=cats))
    del F
    mu_cat = anchor_all[eval_rows] + m.predict(Pool(Xe, cat_features=cats))
    log(f"refit on {int(reg_all.sum())} regression rows with {n_it} iters")
    b = pd.read_parquet(CACHE / f"{args.setting}_{args.base}_s{args.seed}.parquet")
    assert np.array_equal(b["MVT_ID_mvt"].to_numpy(), ids_e), "바탕 모형 출력과 평가 행 순서가 다르다"
    mu_b = (1 - W) * b["mu"].to_numpy() + W * mu_cat
    pred = np.clip(b["p_art"].to_numpy() * b["dsch"].to_numpy() + (1 - b["p_art"].to_numpy()) * mu_b, 0, None)
    g = b["g"].to_numpy().astype(bool)
    pred[g] = b["pred"].to_numpy()[g]
    res = {"MVT_ID_mvt": ids_e, "pred": pred, "mu_cat": mu_cat, "mu": b["mu"].to_numpy(), "p_art": b["p_art"].to_numpy(),
           "g": g.astype(np.int8), "dsch": b["dsch"].to_numpy()}
    if "y" in b.columns:
        ye = b["y"].to_numpy()
        res["y"] = ye
        print(breakdown(pred, ye, ap_e, f"{args.tag} {args.setting} mixture+G, mu = lgb/cat 0.5"))
        print(f"  base {args.base} {rmse(b['pred'].to_numpy(), ye):.2f} | mu_cat alone in mixture "
              f"{rmse(np.where(g, b['pred'].to_numpy(), np.clip(b['p_art'].to_numpy() * b['dsch'].to_numpy() + (1 - b['p_art'].to_numpy()) * mu_cat, 0, None)), ye):.2f}")
        imp = pd.Series(m.get_feature_importance(), index=cols).sort_values(ascending=False)
        print("  cat top importance:", (imp / imp.sum()).head(12).round(3).to_dict())
    out = CACHE / f"{args.setting}_{args.tag}_s{args.seed}.parquet"
    pd.DataFrame(res).to_parquet(out, index=False)
    log(f"saved {out.name}")


if __name__ == "__main__":
    main()
