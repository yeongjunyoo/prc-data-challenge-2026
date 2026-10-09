# SPDX-License-Identifier: GPL-3.0-only
"""넷째 학습기 후보: RealMLP(pytabkit 1.7.3, Apache-2.0) 회귀(mu)를 model3, cat_mu, xgb_mu 와 같은 회귀 행, 같은 피처로 GPU 에서 학습한다.

  pred = clip(p_art * d_sch + (1 - p_art) * mu_b, 0)   mu_b = (1 - W) * mu_base + W * mu_rmlp   (W = 0.5, 점수 전에 고정)
  G 행(LIRF NM 미매칭)은 바탕 모형의 값을 그대로 둔다. 다른 학습기와 고르게 섞을 때는 이 출력의 mu_rmlp 를 blend_mu --col mu_rmlp 로 쓴다.
행 나누기와 조기 종료 날은 src/cat_mu.py 와 같다. 다른 점: 신경망이라 다시 맞추기(refit)를 하지 않는다. 세 설정 모두 학습 범위에서 조기 종료 날(UTC 날짜 % 10 == 0)을
뺀 행으로 학습하고 그 날들로 가장 나은 에포크를 고른다(full 도 같다, 평가 행은 쓰지 않는다). 결측은 학습 행 중앙값으로 채우고 결측이 있던 칸마다 표시 칸을 더한다.
CAT_FEATS 는 pytabkit 의 범주형(높은 기수는 8차원 임베딩)으로 넣는다. 설정은 점수 전에 고정했다(RealMLP_TD 기본값, batch 2048, epochs 64, 출력 표준화는 기본).
출력: cache/<setting>_<tag>_s<seed>.parquet (MVT_ID_mvt, pred, mu_rmlp, mu, p_art, g, dsch, [y]), preds/es_valid_ids_<tag>_<setting>_s<seed>.parquet
    .venv/Scripts/python.exe -u src/rmlp_mu.py --setting val --seed 0 --tag m6r25rm --base m6r25 --feats feats_v25
--drop, --anchor aobt 는 cat_mu 와 같다(표류에 강한 계열 m6r25axsg 위에 얹을 때).
메모리(10-06, 입력과 결과는 그대로): 목표 인코딩 칸을 float32 로 바꿔 붙인다(모든 피처를 float32 로 읽으므로 값이 같다). 표는 배열을 복사하지 않고 만든다
(pandas 3 의 DataFrame(dict) 는 같은 dtype 칸을 한 덩어리로 복사한다). 학습 표만큼(경로 B 피처에서 약 1.3GB)의 순간 복사와 약 0.3GB 가 준다.
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

sys.path.insert(0, ".")
from src.common import breakdown, feature_cols, rmse  # noqa: E402
from src.feats import CAT_FEATS, build  # noqa: E402
from src.load import CACHE  # noqa: E402
from src.model3 import es_day_mask  # noqa: E402
from src.te import add_te  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
W = 0.5
BATCH, EPOCHS = 2048, 64
T0 = time.time()


def release() -> None:
    """Free the reference cycles that keep the feature table's blocks alive and return Arrow's unused memory to the system (10-06:
    1.1 GiB of working set on the fwd setting). Memory only; nothing the model sees changes."""
    gc.collect()
    pa.default_memory_pool().release_unused()


def log(msg: str):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def nframe(F: pd.DataFrame, cols: list[str], rows: np.ndarray, med: dict[str, float], na_cols: list[str]) -> pd.DataFrame:
    """RealMLP 입력: CAT_FEATS 는 범주형 그대로(결측은 "<NA>" 범주), 다른 범주형은 코드(float), 수치는 float32. 결측은 med 로 채우고 na_cols 마다 표시 칸."""
    out = {}
    for c in cols:
        s = F[c]
        if isinstance(s.dtype, pd.CategoricalDtype) and c in CAT_FEATS:
            v = s.cat.codes.to_numpy()[rows]
            # 범주 이름 대신 정수 범주(마지막이 결측). pandas 3 의 문자열 범주는 pyarrow 배열이라 pytabkit 의 OrdinalEncoder 가 object 로 바꾸다
            # memcap 7GB 에서 죽었다(10-04 06:18 ArrowException). 범주의 정체성은 같고 이름만 다르다
            out[c] = pd.Categorical.from_codes(np.where(v < 0, len(s.cat.categories), v), categories=np.arange(len(s.cat.categories) + 1))
            continue
        v = (np.where(s.cat.codes.to_numpy() < 0, np.nan, s.cat.codes.to_numpy()).astype(np.float32) if isinstance(s.dtype, pd.CategoricalDtype)
             else s.to_numpy(dtype=np.float32, na_value=np.nan))[rows]
        if c in na_cols:
            out[f"{c}__na"] = np.isnan(v).astype(np.float32)
        out[c] = np.where(np.isnan(v), med.get(c, 0.0), v).astype(np.float32)
    return pd.DataFrame(out, copy=False)


def light_checks() -> None:
    """pytabkit 의 fit, predict 첫머리 검사(sklearn check_X_y, check_array, dtype=None)는 범주와 수치가 섞인 표 전체를 object 배열로 바꿔
    학습 행 140만 x 155칸에 약 7GB 를 잠깐 쓴다(10-04 06:16 memcap 7GB 에서 MemoryError). 데이터를 바꾸지 않는 검사라 앞 2,000행으로만 한다."""
    import pytabkit.models.sklearn.sklearn_base as skb
    xy, arr = skb.check_X_y_wrapper, skb.check_array_wrapper
    head = lambda v: v.iloc[:2000] if hasattr(v, "iloc") else v[:2000]  # noqa: E731
    skb.check_X_y_wrapper = lambda X, y, **kw: xy(head(X), head(y), **kw)
    skb.check_array_wrapper = lambda X, **kw: arr(head(X), **kw)


def main():
    from pytabkit import RealMLP_TD_Regressor
    light_checks()

    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", default="val", choices=["val", "fwd", "full"])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--base", required=True, help="섞을 출력의 태그(cache/<setting>_<base>_s<seed>.parquet, mu 와 p_art 칸이 있어야 한다)")
    pa.add_argument("--feats", default="feats_v25")
    pa.add_argument("--drop", default="", help="쉼표로 뺄 피처 칸(model3, cat_mu 의 --drop 과 같다)")
    pa.add_argument("--anchor", default="", choices=["", "aobt", "win"], help="model3, cat_mu 의 --anchor 와 같은 회귀 목표")
    pa.add_argument("--device", default="cuda", help="학습 장치(기본 cuda, PC). 맥북의 경로 B 다리는 mps 나 cpu(10-06)")
    pa.add_argument("--threads", type=int, default=0, help="pytabkit 의 torch CPU 스레드(0 이면 그 기본값, 물리 코어 수). 맥북은 1: lightgbm, sklearn, torch 가 저마다 libomp 를 올려 torch 의 병렬 복사가 SIGSEGV 로 죽는다(research/ctx_corr/mac/omp_repro.py, 10-06)")
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
    es_rows = fit_rows & es_day_mask(F["MVT_ID_mvt"].to_numpy())
    if args.setting != "full":
        pd.DataFrame({"MVT_ID_mvt": F["MVT_ID_mvt"].to_numpy()[es_rows]}).to_parquet(
            CACHE.parent / "preds" / f"es_valid_ids_{args.tag}_{args.setting}_s{args.seed}.parquet", index=False)
    T = add_te(F, fit_rows).astype(np.float32)
    F = pd.concat([F, T], axis=1)
    del T
    release()
    cols = feature_cols(F, None)
    cats = [c for c in cols if c in CAT_FEATS and isinstance(F[c].dtype, pd.CategoricalDtype)]
    y_all = F["y"].to_numpy().astype(float)
    dsch_all = F["d_sch"].to_numpy().astype(float)
    art_all = np.abs(y_all - dsch_all) <= 60
    shift_all = (y_all > 60000) & ~art_all
    reg_all = fit_rows & ~art_all & ~shift_all  # model3 의 회귀 행(조기 종료 날 포함한 학습 범위)
    if args.anchor == "aobt":  # cat_mu --anchor aobt 와 같은 기준값
        da = F["d_aobt"].to_numpy().astype(float)
        apv = F["ap"].astype(str).to_numpy()
        nfit = fit_rows & ~art_all & ~shift_all & (y_all < 3600)
        amed = pd.Series(y_all[nfit]).groupby(apv[nfit]).median()
        fb = pd.Series(apv).map(amed).fillna(float(np.median(y_all[nfit]))).to_numpy()
        anchor_all = np.where(np.isnan(da), fb, np.clip(da, 0, 7200))
    elif args.anchor == "win":  # LOBT 창 기준(vault research/2026-10-06-r5-paradigms.md): 공항별 정상 학습 행 중앙값을 창 [d_lobt - 3606, d_lobt + 3606] 안으로
        dl = F["d_lobt"].to_numpy().astype(float)
        apv = F["ap"].astype(str).to_numpy()
        nfit = fit_rows & ~art_all & ~shift_all & (y_all < 3600)
        amed = pd.Series(y_all[nfit]).groupby(apv[nfit]).median()
        fb = pd.Series(apv).map(amed).fillna(float(np.median(y_all[nfit]))).to_numpy()
        anchor_all = np.where(np.isnan(dl), fb, np.clip(fb, dl - 3606.0, dl + 3606.0))
    else:
        anchor_all = np.zeros(len(F))
    tgt = y_all - anchor_all
    tr = reg_all & ~es_rows
    va = reg_all & es_rows
    med, na_cols = {}, []
    for c in cols:
        if c in cats:
            continue
        s = F[c]
        v = (np.where(s.cat.codes.to_numpy() < 0, np.nan, s.cat.codes.to_numpy()).astype(np.float32) if isinstance(s.dtype, pd.CategoricalDtype)
             else s.to_numpy(dtype=np.float32, na_value=np.nan))
        if np.isnan(v).any():
            na_cols.append(c)
        m_ = np.nanmedian(v[tr]) if np.isfinite(v[tr]).any() else 0.0
        med[c] = float(m_) if np.isfinite(m_) else 0.0
    ids_e = F["MVT_ID_mvt"].to_numpy()[eval_rows]
    ap_e = F["ap"].astype(str).to_numpy()[eval_rows]
    Xtr, Xva, Xe = nframe(F, cols, tr, med, na_cols), nframe(F, cols, va, med, na_cols), nframe(F, cols, eval_rows, med, na_cols)
    ytr, yva = tgt[tr].astype(np.float32), tgt[va].astype(np.float32)
    anchor_e = anchor_all[eval_rows]
    del F
    release()
    log(f"[{args.tag} {args.setting} seed{args.seed}] features {Xtr.shape[1]} (cat {len(cats)}, na flags {len(na_cols)}) train {len(Xtr)} es {len(Xva)} eval {len(Xe)}")
    m = RealMLP_TD_Regressor(device=args.device, n_cv=1, batch_size=BATCH, n_epochs=EPOCHS, random_state=args.seed, verbosity=0,
                             **({"n_threads": args.threads} if args.threads else {}))
    m.fit(Xtr, ytr, Xva, yva, cat_col_names=cats)
    del Xtr
    log("fitted")
    p_va = m.predict(Xva)
    log(f"es-day rmse {rmse(p_va, yva):.2f}")
    mu_rmlp = anchor_e + m.predict(Xe).astype(float)
    if args.setting == "val" and args.seed == 0:
        (CACHE / f"iters_{args.tag}.json").write_text(json.dumps({"rmlp": {"batch_size": BATCH, "n_epochs": EPOCHS, "refit": False, "es_rmse": round(rmse(p_va, yva), 2)}}))
    b = pd.read_parquet(CACHE / f"{args.setting}_{args.base}_s{args.seed}.parquet")
    assert np.array_equal(b["MVT_ID_mvt"].to_numpy(), ids_e), "바탕 모형 출력과 평가 행 순서가 다르다"
    mu_b = (1 - W) * b["mu"].to_numpy() + W * mu_rmlp
    pred = np.clip(b["p_art"].to_numpy() * b["dsch"].to_numpy() + (1 - b["p_art"].to_numpy()) * mu_b, 0, None)
    g = b["g"].to_numpy().astype(bool)
    pred[g] = b["pred"].to_numpy()[g]
    res = {"MVT_ID_mvt": ids_e, "pred": pred, "mu_rmlp": mu_rmlp, "mu": b["mu"].to_numpy(), "p_art": b["p_art"].to_numpy(),
           "g": g.astype(np.int8), "dsch": b["dsch"].to_numpy()}
    if "y" in b.columns:
        ye = b["y"].to_numpy()
        res["y"] = ye
        print(breakdown(pred, ye, ap_e, f"{args.tag} {args.setting} mixture+G, mu = base/rmlp 0.5"))
        print(f"  base {args.base} {rmse(b['pred'].to_numpy(), ye):.2f} | mu_rmlp alone in mixture "
              f"{rmse(np.where(g, b['pred'].to_numpy(), np.clip(b['p_art'].to_numpy() * b['dsch'].to_numpy() + (1 - b['p_art'].to_numpy()) * mu_rmlp, 0, None)), ye):.2f}")
    out = CACHE / f"{args.setting}_{args.tag}_s{args.seed}.parquet"
    pd.DataFrame(res).to_parquet(out, index=False)
    log(f"saved {out.name}")


if __name__ == "__main__":
    main()
