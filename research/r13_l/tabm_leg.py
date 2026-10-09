# SPDX-License-Identifier: GPL-3.0-only
"""R13-L new learner family: TabM (Gorishniy et al., ICLR 2025, package tabm 0.0.3, Apache-2.0) as a regression leg (mu) on the same rows,
features and target as src/rmlp_mu.py, from the arrays of research/r13_l/prep_x.py (GPU-resident data, little host memory).

Leg contract (as rmlp_mu): regression rows = fit rows minus art (|y - d_sch| <= 60) and shift (y > 60000); trained on those rows minus the
early-stopping days (es_day_mask), the best epoch chosen on the early-stopping days (all three settings, no refit, as rmlp_mu); target
y - anchor (anchor "win" as rmlp_mu --anchor win, or none); prediction mu_tabm on every eval row. The output file has the rmlp_mu layout with
mu_tabm: pred = clip(p_art * dsch + (1 - p_art) * (0.5 mu_base + 0.5 mu_tabm), 0), G rows keep the base value (diagnostic only; blends use
mu_tabm).
Model (fixed before any score): TabM (arch "tabm", k 32, 2 blocks of 512, dropout 0.1) with piecewise-linear embeddings (48 quantile bins,
d 16, version B) of the 140 numeric features after a per-column normal quantile transform fitted on the training rows; the 78 missing
flags as 0/1 inputs; categorical features with cardinality <= 16 one-hot, rwy, actype, ades_reg as 8-d and stand, op, ades as 16-d
embeddings. AdamW lr 2e-3, weight decay 3e-4, batch 1024 shared by the 32 members, bf16 autocast, gradient clip 1.0, loss = mean of the members' MSE on
the standardised target, prediction = mean of the members. At most --epochs epochs, early stop after --patience epochs without a better
es-day RMSE.
Output: cache/<setting>_<tag>_s<seed>.parquet (MVT_ID_mvt, pred, mu_tabm, mu, p_art, g, dsch, [y]),
preds/es_valid_ids_<tag>_<setting>_s<seed>.parquet (val, fwd, full: the es-day ids of the fit rows, as rmlp_mu), research/r13_l/runs/<tag>_<setting>_s<seed>.json
    .venv/Scripts/python.exe -X utf8 research/r13_l/tabm_leg.py --setting val --seed 0 --tag m6r27tmwin --base m6r27skwin --anchor win
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
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
HIGH = ("stand", "op", "ades")  # 16-d embeddings; other categoricals with cardinality > 16 get 8-d embeddings, the rest one-hot


def log(m: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def main() -> int:
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    import rtdl_num_embeddings as rne
    import tabm
    from sklearn.preprocessing import QuantileTransformer

    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--base", required=True, help="cache/<setting>_<base>_s0.parquet: the model3 leg whose mu, p_art, g, dsch the output carries")
    pa.add_argument("--anchor", default="none", choices=["none", "win"])
    pa.add_argument("--k", type=int, default=32)
    pa.add_argument("--n_blocks", type=int, default=2)
    pa.add_argument("--d_block", type=int, default=512)
    pa.add_argument("--dropout", type=float, default=0.1)
    pa.add_argument("--d_emb", type=int, default=16)
    pa.add_argument("--n_bins", type=int, default=48)
    pa.add_argument("--lr", type=float, default=2e-3)
    pa.add_argument("--wd", type=float, default=3e-4)
    pa.add_argument("--bs", type=int, default=1024)
    pa.add_argument("--epochs", type=int, default=30)
    pa.add_argument("--patience", type=int, default=5)
    pa.add_argument("--max_minutes", type=float, default=150.0)
    pa.add_argument("--sched", default="const", choices=["const", "cos"], help="cos: learning rate decays as a cosine to 0 over --epochs (per step)")
    pa.add_argument("--device", default="cuda", help="cuda (PC), mps (Mac) or cpu")
    pa.add_argument("--threads", type=int, default=0, help="torch CPU threads (0: torch default)")
    pa.add_argument("--amp", default="bf16", choices=["bf16", "none"], help="bf16 autocast (default) or full float32")
    a = pa.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device)
    if a.threads:
        torch.set_num_threads(a.threads)
    ac = (lambda: torch.autocast(dev.type, dtype=torch.bfloat16)) if a.amp == "bf16" else contextlib.nullcontext  # noqa: E731
    gib = (lambda: torch.cuda.memory_allocated() / 2**30) if dev.type == "cuda" else (lambda: 0.0)  # noqa: E731
    gpk = (lambda: torch.cuda.max_memory_allocated() / 2**30) if dev.type == "cuda" else (lambda: 0.0)  # noqa: E731
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
    high = [j for j, c in enumerate(cats) if card[c] > 16]
    rng = np.random.default_rng(a.seed)
    sub = np.sort(rng.choice(tr, size=min(200_000, len(tr)), replace=False))
    parts = {"tr": tr, "va": va, "ev": ev}
    G = {}
    for nm, idx in parts.items():
        G[nm] = {"xn": torch.empty((len(idx), len(nums)), dtype=torch.float32, device=dev),
                 "xf": torch.empty((len(idx), NA.shape[0]), dtype=torch.bfloat16, device=dev),
                 "xl": torch.empty((len(idx), len(low)), dtype=torch.int64, device=dev),
                 "xh": torch.empty((len(idx), len(high)), dtype=torch.int64, device=dev)}
    for j in range(len(nums)):
        col = np.asarray(X[j], dtype=np.float32)
        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=10**9, random_state=a.seed)
        qt.fit(col[sub].reshape(-1, 1))
        for nm, idx in parts.items():
            G[nm]["xn"][:, j] = torch.from_numpy(qt.transform(col[idx].reshape(-1, 1)).astype(np.float32).ravel()).to(dev)
    for j in range(NA.shape[0]):
        col = np.asarray(NA[j])
        for nm, idx in parts.items():
            G[nm]["xf"][:, j] = torch.from_numpy(col[idx].astype(np.float32)).to(dev).to(torch.bfloat16)
    for dst, js in (("xl", low), ("xh", high)):
        for i, j in enumerate(js):
            col = np.asarray(C[j])
            for nm, idx in parts.items():
                G[nm][dst][:, i] = torch.from_numpy(col[idx].astype(np.int64)).to(dev)
    yt = torch.from_numpy(((tgt[tr] - t_mean) / t_sd).astype(np.float32)).to(dev)
    log(f"[{a.tag} {a.setting} seed{a.seed} anchor {a.anchor}] numeric {len(nums)}, flags {NA.shape[0]}, one-hot {[cats[j] for j in low]}, "
        f"embedded {[cats[j] for j in high]}; train {len(tr)} es {len(va)} eval {len(ev)}; target mean {t_mean:.1f} sd {t_sd:.1f}; "
        f"GPU {gib():.2f} GiB, device {dev}, amp {a.amp}")
    xs = G["tr"]["xn"][torch.from_numpy(rng.choice(len(tr), size=min(200_000, len(tr)), replace=False)).to(dev)]
    if dev.type != "cuda":  # torch.quantile on the CPU off CUDA
        xs = xs.cpu()
    lv = torch.linspace(0.0, 1.0, a.n_bins + 1, device=xs.device)
    bins, n_flat = [], 0
    for j in range(len(nums)):  # rne.compute_bins per column (torch.quantile size limit); a column constant on the subsample gets [min, max] of all training rows
        u = torch.quantile(xs[:, j], lv).unique()
        if len(u) < 2:
            lo, hi = G["tr"]["xn"][:, j].min().to(xs.device), G["tr"]["xn"][:, j].max().to(xs.device)
            u = torch.stack([lo, hi]) if hi > lo else torch.stack([lo - 1, lo + 1])
            n_flat += 1
        bins.append(u.float())
    del xs
    log(f"bins: {sum(len(b) - 1 for b in bins)} in total, {n_flat} columns constant on the subsample")
    low_card = [card[cats[j]] for j in low]
    high_card = [card[cats[j]] for j in high]
    high_dim = [16 if cats[j] in HIGH else 8 for j in high]

    class Leg(nn.Module):
        def __init__(self):
            super().__init__()
            self.num = rne.PiecewiseLinearEmbeddings(bins, d_embedding=a.d_emb, activation=False, version="B")
            self.emb = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(high_card, high_dim)])
            d_features = [a.d_emb] * len(nums) + [1] * NA.shape[0] + low_card + high_dim
            self.ev = tabm.EnsembleView(k=a.k)
            self.backbone = tabm.make_tabm_backbone(d_in=sum(d_features), n_blocks=a.n_blocks, d_block=a.d_block, dropout=a.dropout,
                                                    activation="ReLU", k=a.k, arch_type="tabm", start_scaling_init="normal",
                                                    start_scaling_init_chunks=d_features)
            self.out = tabm.LinearEnsemble(a.d_block, 1, k=a.k)

        def forward(self, xn, xf, xl, xh):
            z = [self.num(xn).flatten(1), xf.to(xn.dtype)]
            z += [Fn.one_hot(xl[:, i], c).to(xn.dtype) for i, c in enumerate(low_card)]
            z += [e(xh[:, i]) for i, e in enumerate(self.emb)]
            x = torch.cat(z, 1)
            return self.out(self.backbone(self.ev(x))).squeeze(-1)  # (B, k)

    model = Leg().to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    total_steps = a.epochs * math.ceil(len(tr) / a.bs)
    step_i = 0

    def predict(nm: str) -> np.ndarray:
        model.eval()
        g = G[nm]
        n = g["xn"].shape[0]
        out = torch.empty(n, dtype=torch.float32, device=dev)
        with torch.no_grad(), ac():
            for s in range(0, n, 2048):
                out[s:s + 2048] = model(g["xn"][s:s + 2048], g["xf"][s:s + 2048], g["xl"][s:s + 2048], g["xh"][s:s + 2048]).float().mean(1)
        return out.cpu().numpy().astype(np.float64) * t_sd + t_mean

    yva = tgt[va]
    best, best_ep, best_state, hist, bad = np.inf, -1, None, [], 0
    n = len(tr)
    steps = math.ceil(n / a.bs)
    for ep in range(a.epochs):
        model.train()
        t1 = time.time()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for s in range(steps):
            b = perm[s * a.bs:(s + 1) * a.bs]
            g = G["tr"]
            with ac():
                p = model(g["xn"][b], g["xf"][b], g["xl"][b], g["xh"][b]).float()
            loss = ((p - yt[b][:, None]) ** 2).mean()
            if a.sched == "cos":
                for pg in opt.param_groups:
                    pg["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * min(step_i, total_steps) / total_steps))
            step_i += 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) if s % 50 == 0 else 0.0
        pv = predict("va")
        r = float(np.sqrt(np.mean((pv - yva) ** 2)))
        hist.append({"epoch": ep, "es_rmse": round(r, 3), "sec": round(time.time() - t1, 1)})
        log(f"epoch {ep}: es-day rmse {r:.3f} ({time.time() - t1:.0f}s, GPU peak {gpk():.2f} GiB)")
        if r < best - 1e-6:
            best, best_ep, bad = r, ep, 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= a.patience or (time.time() - T0) / 60 > a.max_minutes:
            break
    model.load_state_dict(best_state)
    log(f"best epoch {best_ep}, es-day rmse {best:.3f}, params {n_par}")
    mu_tabm = anchor[ev] + predict("ev")
    ids_e = R["ids"][ev]
    b0 = pd.read_parquet(ROOT / "cache" / f"{a.setting}_{a.base}_s0.parquet")
    assert np.array_equal(b0["MVT_ID_mvt"].to_numpy(), ids_e), "base rows differ from the eval rows"
    pa_, ds = b0["p_art"].to_numpy(), b0["dsch"].to_numpy()
    g_ = b0["g"].to_numpy().astype(bool)
    pred = np.clip(pa_ * ds + (1 - pa_) * (0.5 * b0["mu"].to_numpy() + 0.5 * mu_tabm), 0, None)
    pred[g_] = b0["pred"].to_numpy()[g_]
    res = {"MVT_ID_mvt": ids_e, "pred": pred, "mu_tabm": mu_tabm, "mu": b0["mu"].to_numpy(), "p_art": pa_, "g": g_.astype(np.int8), "dsch": ds}
    rep = {"tag": a.tag, "setting": a.setting, "seed": a.seed, "anchor": a.anchor, "args": vars(a), "best_epoch": best_ep, "es_rmse": round(best, 3),
           "hist": hist, "params": n_par, "target_mean": t_mean, "target_sd": t_sd, "minutes": round((time.time() - T0) / 60, 1),
           "versions": {"tabm": tabm.__version__, "torch": torch.__version__}}
    if "y" in b0.columns:
        ye = b0["y"].to_numpy()
        res["y"] = ye
        alone = np.where(g_, b0["pred"].to_numpy(), np.clip(pa_ * ds + (1 - pa_) * mu_tabm, 0, None))
        body = (ye <= 3600) & ~g_
        rep.update(rmse_mix05=round(float(np.sqrt(np.mean((pred - ye) ** 2))), 3), rmse_alone=round(float(np.sqrt(np.mean((alone - ye) ** 2))), 3),
                   rmse_base=round(float(np.sqrt(np.mean((b0["pred"].to_numpy() - ye) ** 2))), 3),
                   body_rmse_alone=round(float(np.sqrt(np.mean((alone[body] - ye[body]) ** 2))), 3),
                   body_rmse_base=round(float(np.sqrt(np.mean((b0["pred"].to_numpy()[body] - ye[body]) ** 2))), 3))
        log(f"base {a.base} {rep['rmse_base']} | mu_tabm alone in mixture {rep['rmse_alone']} | 0.5 mix {rep['rmse_mix05']} | body: alone "
            f"{rep['body_rmse_alone']} base {rep['body_rmse_base']}")
    out = ROOT / "cache" / f"{a.setting}_{a.tag}_s{a.seed}.parquet"
    if out.exists():
        raise SystemExit(f"exists, refusing to overwrite: {out}")
    pd.DataFrame(res).to_parquet(out, index=False)
    es_ids = R["ids"][R["es"]]
    pe = ROOT / "preds" / f"es_valid_ids_{a.tag}_{a.setting}_s{a.seed}.parquet"
    if a.setting != "full" and not pe.exists():
        pd.DataFrame({"MVT_ID_mvt": es_ids}).to_parquet(pe, index=False)
    rd = ROOT / "research" / "r13_l" / "runs"
    rd.mkdir(exist_ok=True)
    (rd / f"{a.tag}_{a.setting}_s{a.seed}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"saved {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
