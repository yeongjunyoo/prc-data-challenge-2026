# SPDX-License-Identifier: GPL-3.0-only
"""R13-L: add one new leg to an existing base blend with one fitted weight, keeping the blend's own weights for its legs.

  mu  = (1 - w) * mu_blend + w * mu_leg          (mu_blend: the src blend's mu, e.g. blend_fixed m6r27wincbrmfw4sv114)
  pred = clip(p_art * dsch + (1 - p_art) * mu, 0) on non-G rows; G rows keep the src blend's pred (src/blend_oof.py mixture)
w in [0, 1] is the least squares of the unclipped mixture against y over the non-G rows of the fitting days (src/blend_oof.simplex_ls on the
two columns), with src/blend_oof's fitting protocol: val = day folds over the ruler days (src/adsb_corr.day_folds, --folds 10, --seed 0, the
rows of fold k get w fitted on the other folds' days), fwd = w fitted on the January ruler days (val predictions) applied to the fwd files,
full = w fitted on all ruler days applied to the full files. p_art, dsch, g and the G rows come from the src blend unchanged.
Output (blend_fixed layout): cache/<setting>_<tag>_s0.parquet (MVT_ID_mvt, pred, mu, p_art, g, dsch[, y]),
preds/es_valid_ids_<tag>_<setting>_s0.parquet (val, fwd: the src blend's, checked equal to the leg's), research/r13_l/runs/blend_add_<tag>_<setting>.json.
New names only (refuses if a target exists).
    .venv/Scripts/python.exe -X utf8 research/r13_l/blend_add.py --setting val --tag m6r27wincbrmfw4sv114tm --src m6r27wincbrmfw4sv114 --leg m6r27tmwin:mu_tabm
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.blend_oof as bo  # noqa: E402
from src.adsb_corr import day_folds  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
C, P = ROOT / "cache", ROOT / "preds"


def frames(setting: str, src: str, leg: str, col: str) -> tuple[pd.DataFrame, np.ndarray]:
    b = pd.read_parquet(C / f"{setting}_{src}_s0.parquet")
    lg = pd.read_parquet(C / f"{setting}_{leg}_s0.parquet", columns=["MVT_ID_mvt", col])
    assert np.array_equal(b["MVT_ID_mvt"].to_numpy(), lg["MVT_ID_mvt"].to_numpy()), f"{leg}: row order differs from {src}"
    MU = np.column_stack([b["mu"].to_numpy(dtype=float), lg[col].to_numpy(dtype=float)])
    assert np.isfinite(MU).all()
    return b, MU


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--tag", required=True)
    pa.add_argument("--src", required=True)
    pa.add_argument("--leg", required=True, help="<leg tag>:<column>")
    pa.add_argument("--folds", type=int, default=10)
    pa.add_argument("--seed", type=int, default=0)
    a = pa.parse_args()
    leg, col = a.leg.split(":")
    out = C / f"{a.setting}_{a.tag}_s0.parquet"
    if out.exists():
        raise SystemExit(f"exists, refusing to overwrite: {out}")
    val, MUv = frames("val", a.src, leg, col)
    yv = val["y"].to_numpy(dtype=float)
    dv = bo.ruler_days(val["MVT_ID_mvt"].to_numpy())
    fit_ok = ~val["g"].to_numpy().astype(bool) & np.isfinite(yv)
    Av, offv = bo.design(val, MUv)
    rep = {"tag": a.tag, "setting": a.setting, "src": a.src, "leg": a.leg, "folds": a.folds, "seed": a.seed}
    if a.setting == "val":
        b, MU = val, MUv
        fo = day_folds(np.unique(dv), a.folds, a.seed)
        f = pd.Series(dv).map(fo).to_numpy()
        W = np.zeros((len(b), 2))
        rep["fold_w"] = {}
        for j in range(a.folds):
            w = bo.simplex_ls(Av[fit_ok & (f != j)], (yv - offv)[fit_ok & (f != j)])
            W[f == j] = w
            rep["fold_w"][j] = round(float(w[1]), 4)
        print(f"  fold weights of the new leg: {rep['fold_w']}", flush=True)
    else:
        b, MU = frames(a.setting, a.src, leg, col)
        sel = fit_ok & (pd.Series(dv).str[5:7] == "01").to_numpy() if a.setting == "fwd" else fit_ok
        w = bo.simplex_ls(Av[sel], (yv - offv)[sel])
        W = np.tile(w, (len(b), 1))
        rep["w"] = round(float(w[1]), 4)
        rep["fit_rows"] = int(sel.sum())
        print(f"  new leg weight ({'January' if a.setting == 'fwd' else 'all ruler days'}): {w[1]:.4f}", flush=True)
    pred, mu = bo.mix(b, MU, W)
    res = {"MVT_ID_mvt": b["MVT_ID_mvt"].to_numpy(), "pred": pred, "mu": mu, "p_art": b["p_art"].to_numpy(dtype=float),
           "g": b["g"].to_numpy().astype(np.int8), "dsch": b["dsch"].to_numpy(dtype=float)}
    if "y" in b.columns:
        y = b["y"].to_numpy(dtype=float)
        res["y"] = y
        r0 = float(np.sqrt(np.mean((b["pred"].to_numpy(dtype=float) - y) ** 2)))
        r1 = float(np.sqrt(np.mean((pred - y) ** 2)))
        rep.update(rmse_src=round(r0, 3), rmse=round(r1, 3))
        print(f"  {a.setting}: {a.src} {r0:.3f} -> {a.tag} {r1:.3f}", flush=True)
    if a.setting != "full":
        e_src = P / f"es_valid_ids_{a.src}_{a.setting}_s0.parquet"
        e_leg = P / f"es_valid_ids_{leg}_{a.setting}_s0.parquet"
        e_dst = P / f"es_valid_ids_{a.tag}_{a.setting}_s0.parquet"
        x0, x1 = (pd.read_parquet(e)["MVT_ID_mvt"].to_numpy() for e in (e_src, e_leg))
        assert np.array_equal(np.sort(x0), np.sort(x1)), "the leg's early-stopping rows differ from the blend's"
        if e_dst.exists():
            raise SystemExit(f"exists, refusing to overwrite: {e_dst}")
        shutil.copyfile(e_src, e_dst)
    pd.DataFrame(res).to_parquet(out, index=False)
    rd = ROOT / "research" / "r13_l" / "runs"
    rd.mkdir(exist_ok=True)
    (rd / f"blend_add_{a.tag}_{a.setting}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
