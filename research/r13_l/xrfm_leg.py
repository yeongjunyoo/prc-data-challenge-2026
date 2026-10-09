# SPDX-License-Identifier: GPL-3.0-only
"""R13-L second new learner family: xRFM (Beaglehole, Holzmueller, Radhakrishnan, Belkin, arXiv 2508.10053; package xrfm 0.4.5) as a
regression leg (mu) with the leg contract of research/r13_l/tabm_leg.py (same rows, early-stopping days as the validation set, target
y - anchor, output layout with mu_xrfm), from the arrays of research/r13_l/prep_x.py.
Inputs (fixed before any score, pytabkit XRFM_D defaults where they apply): the 140 numeric features after a per-column normal quantile
transform fitted on the training rows, the 78 missing flags and the one-hot categoricals with cardinality <= 16, every column centred and
scaled to RMS 1 on the training rows (pytabkit 'mean_center', 'l2_normalize'); the larger categoricals enter only through their target
encodings (already numeric features). Kernel 'l2' Laplace, bandwidth 10, exponent 1, diagonal AGOP, reg 1e-3, 5 iterations with early stop,
leaf size from xRFM's GPU-memory rule; target standardised on the training rows.
Output: cache/<setting>_<tag>_s<seed>.parquet (MVT_ID_mvt, pred, mu_xrfm, mu, p_art, g, dsch, [y]), preds/es_valid_ids_<tag>_<setting>_s<seed>.parquet
(val, fwd), research/r13_l/runs/<tag>_<setting>_s<seed>.json
    .venv/Scripts/python.exe -X utf8 research/r13_l/xrfm_leg.py --setting val --seed 0 --tag m6r27xrwin --base m6r27skwin --anchor win
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if sys.platform == "win32":
    import ctypes
    ctypes.WinDLL("kernel32").SetPriorityClass(ctypes.WinDLL("kernel32").GetCurrentProcess(), 0x00004000)
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def main() -> int:
    import torch
    import xrfm
    from sklearn.preprocessing import QuantileTransformer

    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--base", required=True)
    pa.add_argument("--anchor", default="none", choices=["none", "win"])
    pa.add_argument("--bandwidth", type=float, default=10.0)
    pa.add_argument("--reg", type=float, default=1e-3)
    pa.add_argument("--iters", type=int, default=5)
    pa.add_argument("--max_leaf", type=int, default=60000, help="xRFM base leaf size (scaled by its GPU-memory rule)")
    a = pa.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    xd = ROOT / "research" / "r13_l" / "x" / a.setting
    meta = json.loads((xd / "meta.json").read_text(encoding="utf-8"))
    R = np.load(xd / "rows.npz")
    tr, va, ev = R["tr_idx"], R["va_idx"], R["ev_idx"]
    y = R["y"]
    anchor = R["anchor_win"] if a.anchor == "win" else np.zeros(len(y))
    tgt = y - anchor
    t_mean, t_sd = float(tgt[tr].mean()), float(tgt[tr].std())
    X = np.load(xd / "X.npy", mmap_mode="r")
    NA = np.load(xd / "NA.npy", mmap_mode="r")
    C = np.load(xd / "C.npy", mmap_mode="r")
    nums, cats, card = meta["nums"], meta["cats"], meta["card"]
    low = [j for j, c in enumerate(cats) if card[c] <= 16]
    d = len(nums) + NA.shape[0] + sum(card[cats[j]] for j in low)
    rng = np.random.default_rng(a.seed)
    sub = np.sort(rng.choice(tr, size=min(200_000, len(tr)), replace=False))
    parts = {"tr": tr, "va": va, "ev": ev}
    G = {nm: torch.empty((len(idx), d), dtype=torch.float32, device=dev) for nm, idx in parts.items()}
    k = 0

    def put(vals: dict[str, np.ndarray]) -> None:
        nonlocal k
        m = float(vals["tr"].mean())
        s = float(np.sqrt(((vals["tr"] - m) ** 2).mean()))
        s = s if s > 0 else 1.0
        for nm in parts:
            G[nm][:, k] = torch.from_numpy(((vals[nm] - m) / s).astype(np.float32)).to(dev)
        k += 1

    for j in range(len(nums)):
        col = np.asarray(X[j], dtype=np.float32)
        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=10**9, random_state=a.seed)
        qt.fit(col[sub].reshape(-1, 1))
        put({nm: qt.transform(col[idx].reshape(-1, 1)).ravel() for nm, idx in parts.items()})
    for j in range(NA.shape[0]):
        col = np.asarray(NA[j]).astype(np.float32)
        put({nm: col[idx] for nm, idx in parts.items()})
    for j in low:
        col = np.asarray(C[j])
        for c in range(card[cats[j]]):
            put({nm: (col[idx] == c).astype(np.float32) for nm, idx in parts.items()})
    assert k == d
    ytr = torch.from_numpy(((tgt[tr] - t_mean) / t_sd).astype(np.float32)).to(dev)
    yva = torch.from_numpy(((tgt[va] - t_mean) / t_sd).astype(np.float32)).to(dev)
    log(f"[{a.tag} {a.setting} seed{a.seed} anchor {a.anchor}] d {d}; train {len(tr)} es {len(va)} eval {len(ev)}; target mean {t_mean:.1f} "
        f"sd {t_sd:.1f}; GPU {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    rfm_params = {"model": {"kernel": "l2", "bandwidth": a.bandwidth, "exponent": 1.0, "norm_p": 2.0, "bandwidth_mode": "constant", "diag": True},
                  "fit": {"reg": a.reg, "iters": a.iters, "verbose": False, "early_stop_rfm": True, "early_stop_multiplier": 1.1,
                          "M_batch_size": 8000, "solver": "solve"}}
    m = xrfm.xRFM(rfm_params, device=dev, max_leaf_size=a.max_leaf, tuning_metric="mse", random_state=a.seed, verbose=True)
    log(f"xRFM leaf size after its GPU-memory rule: {m.max_leaf_size}")
    m.fit(G["tr"], ytr, G["va"], yva)
    log("fitted")

    def predict(nm: str) -> np.ndarray:
        out = []
        for s in range(0, G[nm].shape[0], 50_000):
            p = m.predict(G[nm][s:s + 50_000])
            out.append(np.asarray(p.detach().cpu().numpy() if torch.is_tensor(p) else p, dtype=np.float64).reshape(-1))
        return np.concatenate(out) * t_sd + t_mean

    pv = predict("va")
    es_rmse = float(np.sqrt(np.mean((pv - tgt[va]) ** 2)))
    log(f"es-day rmse {es_rmse:.3f}")
    mu = anchor[ev] + predict("ev")
    ids_e = R["ids"][ev]
    b0 = pd.read_parquet(ROOT / "cache" / f"{a.setting}_{a.base}_s0.parquet")
    assert np.array_equal(b0["MVT_ID_mvt"].to_numpy(), ids_e), "base rows differ from the eval rows"
    pa_, ds = b0["p_art"].to_numpy(), b0["dsch"].to_numpy()
    g_ = b0["g"].to_numpy().astype(bool)
    pred = np.clip(pa_ * ds + (1 - pa_) * (0.5 * b0["mu"].to_numpy() + 0.5 * mu), 0, None)
    pred[g_] = b0["pred"].to_numpy()[g_]
    res = {"MVT_ID_mvt": ids_e, "pred": pred, "mu_xrfm": mu, "mu": b0["mu"].to_numpy(), "p_art": pa_, "g": g_.astype(np.int8), "dsch": ds}
    rep = {"tag": a.tag, "setting": a.setting, "seed": a.seed, "anchor": a.anchor, "args": vars(a), "es_rmse": round(es_rmse, 3), "d": d,
           "leaf_size": m.max_leaf_size, "target_mean": t_mean, "target_sd": t_sd, "minutes": round((time.time() - T0) / 60, 1),
           "versions": {"xrfm": getattr(xrfm, "__version__", "0.4.5"), "torch": torch.__version__}}
    if "y" in b0.columns:
        ye = b0["y"].to_numpy()
        res["y"] = ye
        alone = np.where(g_, b0["pred"].to_numpy(), np.clip(pa_ * ds + (1 - pa_) * mu, 0, None))
        body = (ye <= 3600) & ~g_
        rep.update(rmse_alone=round(float(np.sqrt(np.mean((alone - ye) ** 2))), 3),
                   body_rmse_alone=round(float(np.sqrt(np.mean((alone[body] - ye[body]) ** 2))), 3),
                   body_rmse_base=round(float(np.sqrt(np.mean((b0["pred"].to_numpy()[body] - ye[body]) ** 2))), 3))
        log(f"mu_xrfm alone in mixture {rep['rmse_alone']} | body: alone {rep['body_rmse_alone']} base {rep['body_rmse_base']}")
    out = ROOT / "cache" / f"{a.setting}_{a.tag}_s{a.seed}.parquet"
    if out.exists():
        raise SystemExit(f"exists, refusing to overwrite: {out}")
    pd.DataFrame(res).to_parquet(out, index=False)
    pe = ROOT / "preds" / f"es_valid_ids_{a.tag}_{a.setting}_s{a.seed}.parquet"
    if a.setting != "full" and not pe.exists():
        pd.DataFrame({"MVT_ID_mvt": R["ids"][R["es"]]}).to_parquet(pe, index=False)
    rd = ROOT / "research" / "r13_l" / "runs"
    rd.mkdir(exist_ok=True)
    (rd / f"{a.tag}_{a.setting}_s{a.seed}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"saved {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
