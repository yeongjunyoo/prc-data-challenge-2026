"""저장된 model3 출력(--base)의 G 행만 g_override(--g_band_avg 규칙)로 다시 짓는다. 학습을 다시 돌리지 않는다.

model3 와 같은 피처 판, 같은 행 나누기로 G 표를 만들고 g_probs 를 다시 계산한다. 먼저 규칙을 끈 채로 저장된 G 예측을 그대로
재현하는지 확인하고(어긋나면 멈춘다), 그다음 지연 1~4시간 G 행의 art 확률을 G 모형과 주 분류기(저장된 p_art)의 평균으로 바꾼다.
출력: cache/<setting>_<tag>_s<seed>.parquet (바탕 출력과 같은 칸, pred 의 G 행만 다름), 조기 종료 행은 바탕의 것을 이 태그로 복사.
    .venv/Scripts/python.exe -u src/g_band.py --setting val --seed 0 --tag m6r25sg --base m6r25s --feats feats_v25
"""
from __future__ import annotations

import argparse
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, ".")
from src.common import breakdown, rmse  # noqa: E402
from src.load import CACHE  # noqa: E402
from src.model3 import g_override, g_probs  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", default="val", choices=["val", "fwd", "fj", "full"])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--base", required=True)
    pa.add_argument("--feats", default="feats_v25")
    args = pa.parse_args()
    fp = CACHE / f"{args.feats}.parquet"
    names = set(pq.read_schema(fp).names)
    gkey = "prefix" if "prefix" in names else "ades"  # model3 와 같은 키 고르기
    want = ["MVT_ID_mvt", "src", "ap", "nm", "y", "d_sch", "dd", gkey, "hour_mvt", "hour_sch", "garb", "fl_len", "stand"]
    F = pd.read_parquet(fp, columns=[c for c in dict.fromkeys(want) if c in names])
    src = F["src"].to_numpy()
    te = src == 0
    if args.setting == "val":
        fit_rows, eval_rows = ~np.isin(src, [1, 7]) & ~te, np.isin(src, [1, 7])
    elif args.setting == "fwd":
        fit_rows, eval_rows = np.isin(src, [1, 2, 3, 4, 5, 6]), src == 7
    elif args.setting == "fj":  # F2 달 빼기 자: 2, 6월
        fit_rows, eval_rows = ~np.isin(src, [2, 6]) & ~te, np.isin(src, [2, 6])
    else:
        fit_rows, eval_rows = ~te, te
    g_all = ((F["ap"] == "LIRF") & (F["nm"] == 0)).to_numpy()
    Gcols = [c for c in ["y", "d_sch", "dd", gkey, "hour_mvt", "hour_sch", "garb", "fl_len", "stand"] if c in F.columns]
    G = F.loc[g_all & (fit_rows | eval_rows), Gcols].copy()
    G["pfx"] = G[gkey].astype(str).fillna("nan")
    G["stand_d"] = pd.to_numeric(G["stand"].astype(str).fillna("nan").str[5:6], errors="coerce")
    g_fit_mask = fit_rows[g_all & (fit_rows | eval_rows)]
    Gf = G[g_fit_mask & ~np.isnan(G["y"].to_numpy())]
    Ge = G[~g_fit_mask]
    P = g_probs(Gf, Ge)
    b = pd.read_parquet(CACHE / f"{args.setting}_{args.base}_s{args.seed}.parquet")
    assert np.array_equal(b["MVT_ID_mvt"].to_numpy(), F["MVT_ID_mvt"].to_numpy()[eval_rows]), "평가 행 순서가 바탕 출력과 다르다"
    g = b["g"].to_numpy().astype(bool)
    assert np.array_equal(g, g_all[eval_rows]) and g.sum() == len(Ge), "G 행이 바탕 출력과 다르다"
    mu_g = b["mu"].to_numpy()[g]
    d, dd = Ge["d_sch"].to_numpy(), Ge["dd"].to_numpy()
    g0, _ = g_override(P, d, dd, mu_g, None)
    err = float(np.max(np.abs(np.clip(g0, 0, None) - b["pred"].to_numpy()[g])))
    print(f"reproduce saved G predictions with the rule off: max abs diff {err:.3e} over {int(g.sum())} rows")
    assert err < 1e-3, "규칙을 끈 재계산이 저장된 G 예측과 다르다"
    g1, _ = g_override(P, d, dd, mu_g, b["p_art"].to_numpy()[g])
    pred = b["pred"].to_numpy().copy()
    pred[g] = np.clip(g1, 0, None)
    band = (d >= 3600) & (d < 14400)
    print(f"G rows in the 1-4h band: {int(band.sum())}; mean |change| there {np.abs(pred[g] - b['pred'].to_numpy()[g])[band].mean():.1f} s")
    out = b.copy()
    out["pred"] = pred
    if args.setting != "full":
        P_ = CACHE.parent / "preds"
        shutil.copyfile(P_ / f"es_valid_ids_{args.base}_{args.setting}_s{args.seed}.parquet", P_ / f"es_valid_ids_{args.tag}_{args.setting}_s{args.seed}.parquet")
    if "y" in out.columns:
        ap = pd.read_parquet(fp, columns=["MVT_ID_mvt", "ap"]).set_index("MVT_ID_mvt")["ap"].astype(str)
        print(breakdown(pred, out["y"].to_numpy(), ap.reindex(out["MVT_ID_mvt"]).to_numpy(), f"{args.tag} {args.setting} G band rule on {args.base}"))
        print(f"  base {args.base} {rmse(b['pred'].to_numpy(), out['y'].to_numpy()):.2f}")
    o = CACHE / f"{args.setting}_{args.tag}_s{args.seed}.parquet"
    out.to_parquet(o, index=False)
    print(f"saved {o.name}")


if __name__ == "__main__":
    main()
