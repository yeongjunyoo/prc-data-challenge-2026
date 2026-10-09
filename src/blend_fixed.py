# SPDX-License-Identifier: GPL-3.0-only
"""Four-learner blend whose weights are a reference blend's own (316's B weights), applied to another LightGBM leg (route B).

The weights are the ones src/blend_oof.py fits for the reference base (--ref_base m6r25sg, the blend 316 stands on, m6r25sgoof4), recomputed
here with the same functions on the same rows and folds: val gets the day-fold weights of its fold (fitted without that fold's days), fwd
the January weights, full the weights of all ruler days (0.1252 / 0.4951 / 0 / 0.3798 for LightGBM, CatBoost, XGBoost, RealMLP). They are
applied unchanged to the blend whose LightGBM leg is --base (m6r27skg, route B: its mu, p_art, dsch and G rows) with the same other three
learners. No weight is fitted on the new leg's predictions (the 353 / 374 refit put 0.60 on LightGBM).
--ref_learners (default: --learners) are the reference blend's other three learners, when --learners swaps some of them for route B legs
(the weights stay attached to their slot: LightGBM, CatBoost mu_cat, XGBoost mu_xgb, RealMLP mu_rmlp, in that order).
--check: the reference blend recomputed here is compared with cache/<setting>_<ref_tag>_s<seed>.parquet (largest difference printed).
Output as src/blend_oof.py: cache/<setting>_<tag>_s<seed>.parquet (MVT_ID_mvt, pred, mu, p_art, g, dsch[, y]),
notes/blend_fixed_<tag>_<setting>_s<seed>.json, preds/es_valid_ids_<tag>_<setting>_s<seed>.parquet (val, fwd).
    .venv/Scripts/python.exe -u src/blend_fixed.py --setting val --seed 0 --tag m6r27skgfw4 --base m6r27skg --ref_base m6r25sg
        --ref_tag m6r25sgoof4 --learners m6r25cb:mu_cat,m6r25xg:mu_xgb,m6r25rm:mu_rmlp --folds 10
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import src.blend_oof as bo  # noqa: E402
from src.adsb_corr import day_folds  # noqa: E402
from src.common import breakdown, rmse  # noqa: E402
from src.load import CACHE  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def ref_weights(a) -> tuple[np.ndarray, np.ndarray, dict]:
    """Weights per val row (n x k) for val, or one row of weights for fwd and full, exactly as src/blend_oof.py fits them on the reference."""
    r = argparse.Namespace(base=a.ref_base, base_seed=a.base_seed, learners=a.ref_learners)
    val, MUv = bo.load("val", r)
    yv = val["y"].to_numpy(dtype=float)
    dv = bo.ruler_days(val["MVT_ID_mvt"].to_numpy())
    fit_ok = ~val["g"].to_numpy().astype(bool) & np.isfinite(yv)
    Av, offv = bo.design(val, MUv)
    if a.setting == "val":
        fo = day_folds(np.unique(dv), a.folds, a.seed)
        f = pd.Series(dv).map(fo).to_numpy()
        W = np.zeros((len(val), MUv.shape[1]))
        per = {}
        for j in range(a.folds):
            w = bo.simplex_ls(Av[fit_ok & (f != j)], (yv - offv)[fit_ok & (f != j)])
            W[f == j] = w
            per[j] = [round(float(x), 4) for x in w]
            print(f"  fold {j}: reference weights {np.round(w, 4)}", flush=True)
        return W, val["MVT_ID_mvt"].to_numpy(), {"fold_weights": per}
    sel = fit_ok & (pd.Series(dv).str[5:7] == "01").to_numpy() if a.setting == "fwd" else fit_ok
    w = bo.simplex_ls(Av[sel], (yv - offv)[sel])
    print(f"  reference weights ({'January' if a.setting == 'fwd' else 'all ruler days'}): {np.round(w, 4)}", flush=True)
    return w[None, :], val["MVT_ID_mvt"].to_numpy(), {"weights": [round(float(x), 4) for x in w], "fit_rows": int(sel.sum())}


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", default="val", choices=["val", "fwd", "full"])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--base_seed", type=int, default=0)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--base", required=True)
    pa.add_argument("--ref_base", required=True)
    pa.add_argument("--ref_tag", required=True)
    pa.add_argument("--learners", required=True)
    pa.add_argument("--ref_learners", default=None, help="the reference blend's learners (default: --learners)")
    pa.add_argument("--folds", type=int, default=10)
    pa.add_argument("--check", action="store_true")
    a = pa.parse_args()
    a.ref_learners = a.ref_learners or a.learners
    slots = lambda spec: [x.split(":")[1] for x in spec.split(",")]  # noqa: E731
    assert slots(a.learners) == slots(a.ref_learners), "--learners and --ref_learners must fill the same slots in the same order"
    Wr, val_ids, wrep = ref_weights(a)
    b, MU = bo.load(a.setting, a)
    if a.setting == "val":
        assert np.array_equal(b["MVT_ID_mvt"].to_numpy(), val_ids), "the leg's val rows differ from the reference's"
    W = Wr if len(Wr) == len(b) and a.setting == "val" else np.tile(Wr[0], (len(b), 1))
    pred, mu = bo.mix(b, MU, W)
    rep = {"tag": a.tag, "setting": a.setting, "seed": a.seed, "base": a.base, "ref_base": a.ref_base, "ref_tag": a.ref_tag,
           "learners": ["lgbm"] + [s.split(":")[1] for s in a.learners.split(",")], "leg_tags": [a.base] + [s.split(":")[0] for s in a.learners.split(",")],
           "ref_leg_tags": [a.ref_base] + [s.split(":")[0] for s in a.ref_learners.split(",")], **wrep}
    if a.check:
        r = argparse.Namespace(base=a.ref_base, base_seed=a.base_seed, learners=a.ref_learners)
        rb, rMU = bo.load(a.setting, r)
        rp, rm = bo.mix(rb, rMU, Wr if a.setting == "val" else np.tile(Wr[0], (len(rb), 1)))
        st = pd.read_parquet(CACHE / f"{a.setting}_{a.ref_tag}_s{a.seed}.parquet")
        assert np.array_equal(st["MVT_ID_mvt"].to_numpy(), rb["MVT_ID_mvt"].to_numpy())
        rep["check_max_abs_diff"] = {"pred": float(np.max(np.abs(rp - st["pred"].to_numpy()))), "mu": float(np.max(np.abs(rm - st["mu"].to_numpy())))}
        print(f"  check: reference {a.ref_tag} recomputed here vs its file, largest difference pred {rep['check_max_abs_diff']['pred']:.3g} s, "
              f"mu {rep['check_max_abs_diff']['mu']:.3g} s", flush=True)
    res = {"MVT_ID_mvt": b["MVT_ID_mvt"].to_numpy(), "pred": pred, "mu": mu, "p_art": b["p_art"].to_numpy(dtype=float),
           "g": b["g"].to_numpy().astype(np.int8), "dsch": b["dsch"].to_numpy(dtype=float)}
    if "y" in b.columns:
        y = b["y"].to_numpy(dtype=float)
        res["y"] = y
        ap = pd.read_parquet(CACHE / "feats_v25.parquet", columns=["MVT_ID_mvt", "ap"]).set_index("MVT_ID_mvt")["ap"].astype(str)
        print(breakdown(pred, y, ap.reindex(res["MVT_ID_mvt"]).to_numpy(), f"{a.tag} {a.setting} reference weights"))
        rep["rmse"] = round(rmse(pred, y), 3)
        print(f"  {a.tag} {a.setting}: {rep['rmse']:.3f}")
    if a.setting != "full":
        P = CACHE.parent / "preds"
        es = [pd.read_parquet(P / f"es_valid_ids_{t}_{a.setting}_s{a.base_seed}.parquet")["MVT_ID_mvt"].to_numpy()
              for t in [a.base] + [s.split(":")[0] for s in a.learners.split(",")]]
        for e in es[1:]:
            assert np.array_equal(np.sort(e), np.sort(es[0])), "learners' early-stopping rows differ"
        shutil.copyfile(P / f"es_valid_ids_{a.base}_{a.setting}_s{a.base_seed}.parquet", P / f"es_valid_ids_{a.tag}_{a.setting}_s{a.seed}.parquet")
    out = CACHE / f"{a.setting}_{a.tag}_s{a.seed}.parquet"
    pd.DataFrame(res).to_parquet(out, index=False)
    (CACHE.parent / "notes" / f"blend_fixed_{a.tag}_{a.setting}_s{a.seed}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                                                                         encoding="utf-8")
    print(f"saved {out.name}")


if __name__ == "__main__":
    main()
