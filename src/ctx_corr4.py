# SPDX-License-Identifier: GPL-3.0-only
"""A3 with a CatBoost twin in stage 2: every stage-2 fit of src/ctx_corr3.py (imported, unchanged; it imports src/ctx_corr2.py and
src/ctx_corr.py unchanged) is paired with a CatBoost model on the same matrix, rows and clipped target, and the stage-2 correction is the
plain mean of the two (weight 0.5, fixed in advance, not tuned). Stage 1, folds, pairwise nesting, the LightGBM model, --scale2 (applied to
the mean) and the LOBT window last are A3's. Run it exactly like src/ctx_corr3.py.
  The idea of a second model family on the same inputs, averaged simply, is credited to the public posts of team GREKI (OpenSky Discord
  #prc-data-competition, 2026-09-27 UTC: “A second model family on the same inputs helped us more than new features”, “Another tree model,
  simple average”; README 「외부 아이디어」); the code is ours.
CatBoost settings, fixed before any result, mirror the LightGBM stage-2 settings where the two share a meaning: --rounds2 iterations,
  learning rate 0.05, depth 6 (64 leaves against LightGBM's 63), l2_leaf_reg 10, rsm 0.8 (feature_fraction), Bernoulli subsample 0.8
  (bagging_fraction), border_count 254 (LightGBM's 255 bins), RMSE, thread_count --threads, random_seed --seed; apt_code and stand_p_c are
  categorical (integer codes, missing -1; one-hot up to 10 values, so the ten airports are one-hot), no early stopping.
Research-only flags as in src/ctx_corr3.py (--drop2, --fit_cache), plus --budget <s>: stop with exit code 75 before a CatBoost fit that would
  end after <s> seconds from the start (with --fit_cache the fits already done are kept, so calling again resumes). With --fit_cache the
  CatBoost models are kept in the same store as c2_<hash>.cbm under a hash of their matrix, target, settings, the catboost version and this
  file; a loaded model predicts what the fit it stores predicts.
    .venv/Scripts/python.exe -u src/ctx_corr4.py --setting val --seed 0 --base m6r25sgoof4 --base_seed 0 --tag <tag> --feats feats_v25
        --adsb adsb_v1nd --adsb_sha <16> --geo geo_v1nd --geo_sha <16> --ctx ctx_v3 --ctx_sha <16> --folds 10 --rounds 300 --rounds2 400
        --max_pred 3600 --threads 4 --scale2 0.5
"""
from __future__ import annotations

import atexit
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import src.ctx_corr as cc  # noqa: E402
import src.ctx_corr2 as c2  # noqa: E402
import src.ctx_corr3 as c3  # noqa: E402

W_CAT = 0.5
CB = dict(loss_function="RMSE", learning_rate=0.05, depth=6, l2_leaf_reg=10.0, rsm=0.8, bootstrap_type="Bernoulli", subsample=0.8,
          border_count=254, one_hot_max_size=10, allow_writing_files=False, verbose=0)
T0 = time.time()


def cat_frame(M: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(np.asarray(M, dtype=float), columns=cc.X2)
    for c in cc.CAT2:
        v = df[c].to_numpy()
        df[c] = np.where(np.isfinite(v), v, -1).astype(np.int64)
    return df


class Twin:
    """A3's LightGBM stage-2 model and its CatBoost twin. predict is their mean; importances stay LightGBM's (report only)."""

    def __init__(self, lgbm, cat):
        self.lgbm, self.cat = lgbm, cat

    def predict(self, M):
        from catboost import Pool
        return (1 - W_CAT) * self.lgbm.predict(M) + W_CAT * self.cat.predict(Pool(cat_frame(M), cat_features=cc.CAT2))

    def feature_importance(self, kind):
        return self.lgbm.feature_importance(kind)


def install(cache: Path | None, budget: float | None) -> None:
    import catboost
    from catboost import CatBoostRegressor, Pool
    lgb_fit = cc.fit2
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    stats = {"hit": 0, "fit": 0, "longest": 0.0}

    def fit_cat(M, target, seed, rounds, threads):
        params = dict(CB, iterations=rounds, thread_count=threads, random_seed=seed)
        p = None
        if cache is not None:
            key = c2.digest(code, catboost.__version__, cc.X2, cc.CAT2, params, np.asarray(M, dtype=float), np.asarray(target, dtype=float))
            p = cache / f"c2_{key}.cbm"
            if p.exists():
                stats["hit"] += 1
                m = CatBoostRegressor()
                m.load_model(str(p))
                return m
        if budget is not None and time.time() - T0 + max(stats["longest"], 150.0) > budget:
            print(f"  budget: stop before a CatBoost fit at {time.time() - T0:.0f} s (call again)", flush=True)
            raise SystemExit(75)
        t = time.time()
        m = CatBoostRegressor(**params)
        m.fit(Pool(cat_frame(M), np.clip(target, -cc.CLIP, cc.CLIP), cat_features=cc.CAT2))
        stats["longest"] = max(stats["longest"], time.time() - t)
        stats["fit"] += 1
        print(f"  CatBoost twin fit {time.time() - t:.0f} s on {len(target):,} rows", flush=True)
        if p is not None:
            tmp = p.with_suffix(".tmp")
            m.save_model(str(tmp))
            tmp.replace(p)
        return m

    def fit2_twin(M, target, seed, rounds, threads):
        return Twin(lgb_fit(M, target, seed, rounds, threads), fit_cat(M, target, seed, rounds, threads))

    cc.fit2 = fit2_twin
    atexit.register(lambda: print(f"  CatBoost twins: {stats['hit']} loaded, {stats['fit']} fitted", flush=True))


def main():
    budget = c2.pop_flag("--budget")
    cache = sys.argv[sys.argv.index("--fit_cache") + 1] if "--fit_cache" in sys.argv else None
    run_a3 = cc.main

    def run():
        install(Path(cache) if cache else None, float(budget) if budget else None)
        run_a3()

    cc.main = run
    c3.main()


if __name__ == "__main__":
    main()
