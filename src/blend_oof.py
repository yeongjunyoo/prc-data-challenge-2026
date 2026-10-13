# SPDX-License-Identifier: GPL-3.0-only
"""Blend the four learners' regression parts (mu) with weights fitted on out-of-fold predictions instead of fixed quarters.

Rule fixed before any score was seen (10-05 17:2x KST):
  learners  LightGBM mu (--base, m6r25sg: also gives p_art, dsch, G rows), CatBoost mu_cat (m6r25cb), XGBoost mu_xgb (m6r25xg),
            RealMLP mu_rmlp (m6r25rm). The current line (m6r25sgcbxr) uses a quarter each.
  mixture   pred = clip(p_art * dsch + (1 - p_art) * sum_i w_i * mu_i, 0) on non-G rows; G rows keep the base value (as src/blend_mu.py).
  weights   w_i >= 0, sum w_i = 1, least squares of the unclipped mixture against y over the non-G rows of the fitting days (exact: every
            support of the four learners is solved with its KKT system and the best feasible one is kept).
  val  : day folds over the ruler days (src/adsb_corr.day_folds with --seed, the ADS-B corrector's partition): the rows of fold k get
         weights fitted on the other folds' days, so no ruler day's labels enter its own weights.
  fwd  : weights fitted on the January ruler days (val predictions), applied to the fwd learners (July 2025 from January-June models).
  full : weights fitted on all ruler days, applied to the full learners (2026 test).
The predictions blended are each learner's genuine out-of-period predictions (val: models trained on the ten other 2025 months).
Output: cache/<setting>_<tag>_s<seed>.parquet (MVT_ID_mvt, pred, mu, p_art, g, dsch[, y]), notes/blend_oof_<tag>_<setting>_s<seed>.json.
    .venv/Scripts/python.exe -u src/blend_oof.py --setting val --seed 0 --tag m6r25sgoof4 --base m6r25sg --learners m6r25cb:mu_cat,m6r25xg:mu_xgb,m6r25rm:mu_rmlp --folds 10
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, ".")
from src.adsb_corr import day_folds  # noqa: E402
from src.common import breakdown, rmse  # noqa: E402
from src.load import CACHE, DATA  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def simplex_ls(A: np.ndarray, r: np.ndarray) -> np.ndarray:
    """argmin ||A w - r||^2 subject to w >= 0, sum w = 1 (exact over all supports)."""
    Q, c = A.T @ A, A.T @ r
    k = A.shape[1]
    best, bw = np.inf, None
    for size in range(1, k + 1):
        for S in itertools.combinations(range(k), size):
            S = list(S)
            K = np.zeros((size + 1, size + 1))
            K[:size, :size], K[:size, size], K[size, :size] = Q[np.ix_(S, S)], 1.0, 1.0
            try:
                sol = np.linalg.solve(K, np.r_[c[S], 1.0])
            except np.linalg.LinAlgError:
                continue
            w = np.zeros(k)
            w[S] = sol[:size]
            if (w < -1e-12).any():
                continue
            w = np.clip(w, 0, None)
            w /= w.sum()
            f = w @ Q @ w - 2 * c @ w
            if f < best:
                best, bw = f, w
    return bw


def ruler_days(ids: np.ndarray) -> np.ndarray:
    """UTC day (YYYY-MM-DD) of each ruler departure in ids, read from the January and July 2025 training files."""
    fr = []
    for m in ("01", "07"):
        f = DATA / f"training_2025-{m}-01_2025-{int(m) + 1:02d}-01.parquet"
        t = pq.read_table(f, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).to_pandas()
        fr.append(pd.DataFrame({"MVT_ID_mvt": t["MVT_ID_mvt"].astype("float64"), "day": t["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")}))
    D = pd.concat(fr).set_index("MVT_ID_mvt")["day"]
    out = D.reindex(ids)
    assert out.notna().all()
    return out.astype(str).to_numpy()


def load(setting: str, a) -> tuple[pd.DataFrame, np.ndarray]:
    """The base output of a setting and the regression column of each learner in --learners, as a matrix (rows in the base's order)."""
    b = pd.read_parquet(CACHE / f"{setting}_{a.base}_s{a.base_seed}.parquet")
    mus = [b["mu"].to_numpy(dtype=float)]
    for spec in a.learners.split(","):
        tag, col = spec.split(":")
        c = pd.read_parquet(CACHE / f"{setting}_{tag}_s{a.base_seed}.parquet", columns=["MVT_ID_mvt", col])
        assert np.array_equal(b["MVT_ID_mvt"].to_numpy(), c["MVT_ID_mvt"].to_numpy()), f"{tag}: row order differs"
        mus.append(c[col].to_numpy(dtype=float))
    return b, np.column_stack(mus)


def design(b: pd.DataFrame, MU: np.ndarray):
    """Terms of the mixture that the weights multiply: ((1 - p_art) * mu of each learner, p_art * dsch)."""
    p, d = b["p_art"].to_numpy(dtype=float), b["dsch"].to_numpy(dtype=float)
    return (1 - p)[:, None] * MU, p * d


def mix(b: pd.DataFrame, MU: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """W: weights per row (n x k). Returns (pred, mu)."""
    mu = (MU * W).sum(axis=1)
    p = b["p_art"].to_numpy(dtype=float)
    pred = np.clip(p * b["dsch"].to_numpy(dtype=float) + (1 - p) * mu, 0, None)
    g = b["g"].to_numpy().astype(bool)
    pred[g] = b["pred"].to_numpy(dtype=float)[g]
    return pred, mu


def main():
    """Command line entry: fit the blend weights for one setting (as the module docstring says) and write the blend."""
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", default="val", choices=["val", "fwd", "full"])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--base_seed", type=int, default=0)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--base", required=True)
    pa.add_argument("--learners", required=True)
    pa.add_argument("--folds", type=int, default=10)
    a = pa.parse_args()
    val, MUv = load("val", a)
    yv = val["y"].to_numpy(dtype=float)
    dv = ruler_days(val["MVT_ID_mvt"].to_numpy())
    fit_ok = ~val["g"].to_numpy().astype(bool) & np.isfinite(yv)
    Av, offv = design(val, MUv)
    k = MUv.shape[1]
    names = ["lgbm"] + [s.split(":")[1] for s in a.learners.split(",")]
    rep = {"tag": a.tag, "setting": a.setting, "seed": a.seed, "base": a.base, "learners": names}
    if a.setting == "val":
        b, MU = val, MUv
        fo = day_folds(np.unique(dv), a.folds, a.seed)
        f = pd.Series(dv).map(fo).to_numpy()
        W = np.zeros((len(b), k))
        rep["fold_weights"] = {}
        for j in range(a.folds):
            tr = fit_ok & (f != j)
            w = simplex_ls(Av[tr], (yv - offv)[tr])
            W[f == j] = w
            rep["fold_weights"][j] = [round(float(x), 4) for x in w]
            print(f"  fold {j}: weights {np.round(w, 4)}", flush=True)
        ws = np.array(list(rep["fold_weights"].values()))
        rep["weights_mean"], rep["weights_sd"] = [round(float(x), 4) for x in ws.mean(0)], [round(float(x), 4) for x in ws.std(0)]
    else:
        b, MU = load(a.setting, a)
        sel = fit_ok & (pd.Series(dv).str[5:7] == "01").to_numpy() if a.setting == "fwd" else fit_ok
        w = simplex_ls(Av[sel], (yv - offv)[sel])
        W = np.tile(w, (len(b), 1))
        rep["weights"] = [round(float(x), 4) for x in w]
        rep["fit_rows"] = int(sel.sum())
        print(f"  weights ({'January' if a.setting == 'fwd' else 'all ruler days'}): {np.round(w, 4)}", flush=True)
    pred, mu = mix(b, MU, W)
    eq, _ = mix(b, MU, np.full((len(b), k), 1.0 / k))
    res = {"MVT_ID_mvt": b["MVT_ID_mvt"].to_numpy(), "pred": pred, "mu": mu, "p_art": b["p_art"].to_numpy(dtype=float),
           "g": b["g"].to_numpy().astype(np.int8), "dsch": b["dsch"].to_numpy(dtype=float)}
    if "y" in b.columns:
        y = b["y"].to_numpy(dtype=float)
        res["y"] = y
        ap = pd.read_parquet(CACHE / "feats_v25.parquet", columns=["MVT_ID_mvt", "ap"]).set_index("MVT_ID_mvt")["ap"].astype(str)
        print(breakdown(pred, y, ap.reindex(res["MVT_ID_mvt"]).to_numpy(), f"{a.tag} {a.setting} OOF weights"))
        rep.update(rmse=round(rmse(pred, y), 3), rmse_equal_quarters=round(rmse(eq, y), 3))
        print(f"  equal quarters {rep['rmse_equal_quarters']:.3f} -> OOF weights {rep['rmse']:.3f}")
    if a.setting != "full":
        P = CACHE.parent / "preds"
        es = [pd.read_parquet(P / f"es_valid_ids_{t}_{a.setting}_s{a.base_seed}.parquet")["MVT_ID_mvt"].to_numpy()
              for t in [a.base] + [s.split(":")[0] for s in a.learners.split(",")]]
        for e in es[1:]:
            assert np.array_equal(np.sort(e), np.sort(es[0])), "learners' early-stopping rows differ"
        shutil.copyfile(P / f"es_valid_ids_{a.base}_{a.setting}_s{a.base_seed}.parquet", P / f"es_valid_ids_{a.tag}_{a.setting}_s{a.seed}.parquet")
    out = CACHE / f"{a.setting}_{a.tag}_s{a.seed}.parquet"
    pd.DataFrame(res).to_parquet(out, index=False)
    (CACHE.parent / "notes" / f"blend_oof_{a.tag}_{a.setting}_s{a.seed}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {out.name}")


if __name__ == "__main__":
    main()
