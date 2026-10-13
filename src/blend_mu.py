# SPDX-License-Identifier: GPL-3.0-only
"""Mix the regressions of two outputs and rebuild the mixture prediction: p_art, dsch and the G values of a model3 output (--base)
with mu_cat of a src/cat_mu.py output (--cat).

    pred = clip(p_art * d_sch + (1 - p_art) * ((1 - W) * mu_base + W * mu_cat), 0); G rows keep the base's values (W = --w, default 0.5)
The early-stopping rows of both inputs must be the same; the base's are kept under the new tag.

    python src/blend_mu.py --setting val --seed 0 --tag <tag> --base <model3 tag> --cat <cat_mu tag>

Original note (Korean):
두 출력의 회귀(mu)를 반씩 섞어 혼합 예측을 다시 짓는다: model3 바탕 출력(--base)의 p_art, dsch, G 값과 cat_mu 출력(--cat)의 mu_cat.

  pred = clip(p_art * d_sch + (1 - p_art) * ((1 - W) * mu_base + W * mu_cat), 0),  G 행은 바탕 모형 값 그대로 (W = --w, 기본 0.5)
cat_mu 의 CatBoost 회귀는 바탕 모형과 무관하게 같은 행, 같은 피처로 학습하므로(src/cat_mu.py), 다른 바탕 모형(예: --mu_sep LIRF)에 그대로 다시 쓴다.
조기 종료 행은 두 입력의 것이 같아야 하고, 바탕 모형의 것을 이 태그 이름으로 남긴다.
    .venv/Scripts/python.exe -u src/blend_mu.py --setting val --seed 0 --tag m6r25scb --base m6r25s --cat m6r25cb
"""
from __future__ import annotations

import argparse
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.common import breakdown, rmse  # noqa: E402
from src.load import CACHE  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", default="val", choices=["val", "fwd", "full"])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--base", required=True)
    pa.add_argument("--cat", required=True)
    pa.add_argument("--w", type=float, default=0.5, help="CatBoost mu 의 몫(기본 0.5 는 옛 동작)")
    pa.add_argument("--col", default="mu_cat", help="--cat 출력에서 섞을 칸(xgb_mu 출력이면 mu_xgb)")
    args = pa.parse_args()
    W = args.w
    b = pd.read_parquet(CACHE / f"{args.setting}_{args.base}_s{args.seed}.parquet")
    c = pd.read_parquet(CACHE / f"{args.setting}_{args.cat}_s{args.seed}.parquet", columns=["MVT_ID_mvt", args.col])
    assert np.array_equal(b["MVT_ID_mvt"].to_numpy(), c["MVT_ID_mvt"].to_numpy()), "두 출력의 행 순서가 다르다"
    if args.setting != "full":
        P = CACHE.parent / "preds"
        eb = pd.read_parquet(P / f"es_valid_ids_{args.base}_{args.setting}_s{args.seed}.parquet")["MVT_ID_mvt"].to_numpy()
        ec = pd.read_parquet(P / f"es_valid_ids_{args.cat}_{args.setting}_s{args.seed}.parquet")["MVT_ID_mvt"].to_numpy()
        assert np.array_equal(np.sort(eb), np.sort(ec)), "두 입력의 조기 종료 행이 다르다"
        shutil.copyfile(P / f"es_valid_ids_{args.base}_{args.setting}_s{args.seed}.parquet",
                        P / f"es_valid_ids_{args.tag}_{args.setting}_s{args.seed}.parquet")
    mu = (1 - W) * b["mu"].to_numpy() + W * c[args.col].to_numpy()
    p = b["p_art"].to_numpy()
    pred = np.clip(p * b["dsch"].to_numpy() + (1 - p) * mu, 0, None)
    g = b["g"].to_numpy().astype(bool)
    pred[g] = b["pred"].to_numpy()[g]
    res = {"MVT_ID_mvt": b["MVT_ID_mvt"].to_numpy(), "pred": pred, "mu": mu, "p_art": p, "g": g.astype(np.int8), "dsch": b["dsch"].to_numpy()}
    if "y" in b.columns:
        res["y"] = b["y"].to_numpy()
        ap = pd.read_parquet(CACHE / "feats_v25.parquet", columns=["MVT_ID_mvt", "ap"]).set_index("MVT_ID_mvt")["ap"].astype(str)
        print(breakdown(pred, res["y"], ap.reindex(res["MVT_ID_mvt"]).to_numpy(), f"{args.tag} {args.setting} mixture+G, mu = {args.base}/{args.cat} w {W}"))
        print(f"  base {args.base} {rmse(b['pred'].to_numpy(), res['y']):.2f}")
    out = CACHE / f"{args.setting}_{args.tag}_s{args.seed}.parquet"
    pd.DataFrame(res).to_parquet(out, index=False)
    print(f"saved {out.name}")


if __name__ == "__main__":
    main()
