# SPDX-License-Identifier: GPL-3.0-only
"""R18 copy of research/r13_l/tabm_leg.py (the TabM leg of chain A, tag m6r27tmcwin: anchor win, k 32, 2 x 512, cosine LR, 20 epochs) for
measuring a candidate change against the same leg on the same rows. Differences from the original, nothing else:
  * reads the arrays of --xdir (default research/r13_l/x/<setting>; the fj setting lives in research/r18/x/fj, built by prep_x_r18.py);
  * writes only under research/r18 (cache/<setting>_<tag>_s<seed>.parquet with MVT_ID_mvt, mu_tabm, y; runs/<tag>_<setting>_s<seed>.json);
    no base file is needed (the mixture is evaluated later by screen_r18.py);
  * --loss chooses the training loss of the members:
      mse      the original (mean of the members' MSE on the standardised target);
      hlgauss  regression as classification (Imani and White 2018, HL-Gauss): the target is mapped to its rank u in (0, 1) by the empirical
               CDF of the training targets, u is covered by --n_cls equal-mass bins, the label of a row is the N(u, (sigma/n_cls)^2) mass of
               each bin, the loss is the cross-entropy, and the prediction is sum_k p_k c_k with c_k the mean training target in bin k (the
               conditional mean under RMSE).
  * --tail_mse (hlgauss only): weight of an extra MSE term on the members' expected value (0 = pure classification).
  * tabm_r18c.py (copy of tabm_r18.py sha16 41343b8c8f9aec8a) adds --opt muon: the matrix weights of the backbone's linear layers (the shared
    BatchEnsemble weights, shapes (512, d_in) and (512, 512)) are trained with Muon (momentum SGD whose update is orthogonalised by a quintic
    Newton-Schulz iteration; Jordan 2024; benchmarked for tabular MLPs and TabM by Gorishniy, Rubachev, Feoktistov, Babenko, arXiv 2604.15297,
    whose code assigns Muon to the backbone matrices only), written here from the published algorithm: Nesterov momentum 0.95, 5 Newton-Schulz
    steps with coefficients (3.4445, -4.7750, 2.0315) in bfloat16, update scaled by max(1, rows / cols) ** 0.5, decoupled weight decay --wd,
    learning rate --muon_lr with the same cosine factor as the AdamW groups. Every other parameter (embeddings, BatchEnsemble adapters r and s,
    biases, output layer) stays on the original AdamW (--lr, --wd).
    .venv/Scripts/python.exe -X utf8 research/r18/tabm_r18.py --setting val --seed 0 --tag r18tmc --anchor win --sched cos --epochs 20 --patience 99
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
R18 = ROOT / "research" / "r18"
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if sys.platform == "win32":
    import ctypes
    ctypes.WinDLL("kernel32").SetPriorityClass(ctypes.WinDLL("kernel32").GetCurrentProcess(), 0x00004000)
T0 = time.time()
HIGH = ("stand", "op", "ades")


def log(m: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def _peak_wset() -> float:  # peak working set of this process (GiB), to size the launch memory floor
    try:
        import psutil
        return float(getattr(psutil.Process().memory_info(), "peak_wset", 0)) / 2**30
    except Exception:
        return -1.0


def main() -> int:
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    import rtdl_num_embeddings as rne
    import tabm
    from sklearn.preprocessing import QuantileTransformer

    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "fj", "full"])
    pa.add_argument("--xdir", default="", help="array folder (default research/r13_l/x/<setting>, research/r18/x/fj for fj)")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--anchor", default="win", choices=["none", "win"])
    pa.add_argument("--k", type=int, default=32)
    pa.add_argument("--n_blocks", type=int, default=2)
    pa.add_argument("--d_block", type=int, default=512)
    pa.add_argument("--dropout", type=float, default=0.1)
    pa.add_argument("--d_emb", type=int, default=16)
    pa.add_argument("--n_bins", type=int, default=48)
    pa.add_argument("--lr", type=float, default=2e-3)
    pa.add_argument("--wd", type=float, default=3e-4)
    pa.add_argument("--bs", type=int, default=1024)
    pa.add_argument("--epochs", type=int, default=20)
    pa.add_argument("--patience", type=int, default=99)
    pa.add_argument("--max_minutes", type=float, default=150.0)
    pa.add_argument("--sched", default="cos", choices=["const", "cos"])
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--threads", type=int, default=0)
    pa.add_argument("--amp", default="bf16", choices=["bf16", "none"])
    pa.add_argument("--loss", default="mse", choices=["mse", "hlgauss"])
    pa.add_argument("--n_cls", type=int, default=256)
    pa.add_argument("--sigma", type=float, default=0.75, help="HL-Gauss width in bins")
    pa.add_argument("--tail_mse", type=float, default=0.0)
    pa.add_argument("--opt", default="adamw", choices=["adamw", "muon"])
    pa.add_argument("--muon_lr", type=float, default=0.012)
    pa.add_argument("--pred_chunk", type=int, default=2048, help="rows per prediction chunk (2048 as the original; smaller for CPU tests)")
    pa.add_argument("--smoke", type=int, default=0, help="code test: keep this many rows of each part, write under smoke_ names")
    a = pa.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device)
    if a.threads:
        torch.set_num_threads(a.threads)
    ac = (lambda: torch.autocast(dev.type, dtype=torch.bfloat16)) if a.amp == "bf16" else contextlib.nullcontext  # noqa: E731
    gib = (lambda: torch.cuda.memory_allocated() / 2**30) if dev.type == "cuda" else (lambda: 0.0)  # noqa: E731
    gpk = (lambda: torch.cuda.max_memory_allocated() / 2**30) if dev.type == "cuda" else (lambda: 0.0)  # noqa: E731
    xd = Path(a.xdir) if a.xdir else (R18 / "x" / "fj" if a.setting == "fj" else ROOT / "research" / "r13_l" / "x" / a.setting)
    out = R18 / "cache" / f"{'smoke_' if a.smoke else ''}{a.setting}_{a.tag}_s{a.seed}.parquet"
    if out.exists():
        raise SystemExit(f"exists, refusing to overwrite: {out}")
    meta = json.loads((xd / "meta.json").read_text(encoding="utf-8"))
    R = np.load(xd / "rows.npz")
    tr, va, ev = R["tr_idx"], R["va_idx"], R["ev_idx"]
    if a.smoke:
        tr, va, ev = tr[:a.smoke], va[:a.smoke], ev[:a.smoke]
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
    zt = (tgt[tr] - t_mean) / t_sd
    yt = torch.from_numpy(zt.astype(np.float32)).to(dev)
    n_out = 1
    cls = {}
    if a.loss == "hlgauss":
        K = a.n_cls
        # randomized probability integral transform: u = F(z-) + V (F(z) - F(z-)), V ~ U(0, 1) fixed per row, so a tied target value (whole-minute
        # airports give large ties) is spread over the bins its CDF jump covers and every one of the K equal-mass bins holds about n / K rows
        zs = np.sort(zt)
        lo_r = np.searchsorted(zs, zt, side="left")
        hi_r = np.searchsorted(zs, zt, side="right")
        u_tr = (lo_r + np.random.default_rng(a.seed + 7919).random(len(zt)) * (hi_r - lo_r)) / len(zt)
        b_tr = np.minimum((u_tr * K).astype(np.int64), K - 1)
        cnt = np.bincount(b_tr, minlength=K)
        assert (cnt > 0).all(), "empty class bin"
        centers = np.bincount(b_tr, weights=zt, minlength=K) / cnt
        del zs, lo_r, hi_r
        edges = torch.linspace(0.0, 1.0, K + 1, device=dev)
        cls = {"u": torch.from_numpy(u_tr.astype(np.float32)).to(dev), "edges": edges, "sig": a.sigma / K,
               "centers": torch.from_numpy(centers.astype(np.float32)).to(dev)}
        n_out = K
        log(f"hlgauss: {K} equal-mass bins, sigma {a.sigma} bins; centers (sd units) min {centers.min():.2f} p50 {np.median(centers):.2f} "
            f"max {centers.max():.2f}")
    log(f"[{a.tag} {a.setting} seed{a.seed} anchor {a.anchor} loss {a.loss}] numeric {len(nums)}, flags {NA.shape[0]}, one-hot "
        f"{[cats[j] for j in low]}, embedded {[cats[j] for j in high]}; train {len(tr)} es {len(va)} eval {len(ev)}; target mean "
        f"{t_mean:.1f} sd {t_sd:.1f}; GPU {gib():.2f} GiB, device {dev}, amp {a.amp}, xdir {xd}")
    xs = G["tr"]["xn"][torch.from_numpy(rng.choice(len(tr), size=min(200_000, len(tr)), replace=False)).to(dev)]
    if dev.type != "cuda":
        xs = xs.cpu()
    lv = torch.linspace(0.0, 1.0, a.n_bins + 1, device=xs.device)
    bins, n_flat = [], 0
    for j in range(len(nums)):
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
            self.out = tabm.LinearEnsemble(a.d_block, n_out, k=a.k)

        def forward(self, xn, xf, xl, xh):
            z = [self.num(xn).flatten(1), xf.to(xn.dtype)]
            z += [Fn.one_hot(xl[:, i], c).to(xn.dtype) for i, c in enumerate(low_card)]
            z += [e(xh[:, i]) for i, e in enumerate(self.emb)]
            x = torch.cat(z, 1)
            o = self.out(self.backbone(self.ev(x)))  # (B, k, n_out)
            return o.squeeze(-1) if n_out == 1 else o

    def expect(o: torch.Tensor) -> torch.Tensor:  # (B, k[, K]) -> (B, k) in standardised target units
        if a.loss == "mse":
            return o
        with torch.autocast(dev.type, enabled=False):  # the expectation in float32 (a bf16 matmul would round the bin centers)
            return torch.softmax(o.float(), -1) @ cls["centers"]

    model = Leg().to(dev)
    n_par = sum(p.numel() for p in model.parameters())

    def newton_schulz(g: torch.Tensor, steps: int) -> torch.Tensor:
        x = g.bfloat16()
        tr = g.size(-2) > g.size(-1)
        if tr:
            x = x.mT
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(steps):
            m_ = x @ x.mT
            x = 3.4445 * x + (-4.7750 * m_ + 2.0315 * m_ @ m_) @ x
        return x.mT if tr else x

    class Muon(torch.optim.Optimizer):
        def __init__(self, params, lr, momentum=0.95, weight_decay=0.0, ns_steps=5):
            super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps))

        @torch.no_grad()
        def step(self):
            for gr in self.param_groups:
                for p in gr["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if not st:
                        st["buf"] = torch.zeros_like(p)
                    st["buf"].lerp_(p.grad, 1 - gr["momentum"])
                    u = newton_schulz(p.grad.lerp(st["buf"], gr["momentum"]), gr["ns_steps"])
                    u = u.to(p.dtype) * max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                    p.mul_(1 - gr["lr"] * gr["weight_decay"])
                    p.add_(u, alpha=-gr["lr"])

    if a.opt == "muon":
        mu_p = [m_.weight for m_ in model.backbone.modules()
                if isinstance(m_, (nn.Linear, tabm.LinearBatchEnsemble, tabm.LinearEnsemble)) and 1 not in m_.weight.shape[-2:]]
        ids_mu = {id(p) for p in mu_p}
        rest = [p for p in model.parameters() if id(p) not in ids_mu]
        opts = [torch.optim.AdamW(rest, lr=a.lr, weight_decay=a.wd), Muon(mu_p, lr=a.muon_lr, weight_decay=a.wd)]
        log(f"muon on {[tuple(p.shape) for p in mu_p]} ({sum(p.numel() for p in mu_p)} parameters), lr {a.muon_lr}; AdamW on the other "
            f"{sum(p.numel() for p in rest)}")
    else:
        opts = [torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)]
    base_lr = [[pg["lr"] for pg in o.param_groups] for o in opts]
    total_steps = a.epochs * math.ceil(len(tr) / a.bs)
    step_i = 0

    def predict(nm: str) -> np.ndarray:
        model.eval()
        g = G[nm]
        n = g["xn"].shape[0]
        res = torch.empty(n, dtype=torch.float32, device=dev)
        with torch.no_grad(), ac():
            pc = a.pred_chunk
            for s in range(0, n, pc):
                res[s:s + pc] = expect(model(g["xn"][s:s + pc], g["xf"][s:s + pc], g["xl"][s:s + pc], g["xh"][s:s + pc])).float().mean(1)
        return res.cpu().numpy().astype(np.float64) * t_sd + t_mean

    def loss_fn(o: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.loss == "mse":
            return ((o.float() - yt[b][:, None]) ** 2).mean()
        u = cls["u"][b][:, None]  # (B, 1)
        cdf = torch.special.ndtr((cls["edges"][None, :] - u) / cls["sig"])  # (B, K + 1)
        lab = cdf[:, 1:] - cdf[:, :-1]
        lab = lab / lab.sum(1, keepdim=True).clamp_min(1e-12)
        lp = torch.log_softmax(o.float(), -1)  # (B, k, K)
        ce = -(lab[:, None, :] * lp).sum(-1).mean()
        if a.tail_mse > 0:
            ce = ce + a.tail_mse * ((lp.exp() @ cls["centers"] - yt[b][:, None]) ** 2).mean()
        return ce

    yva = tgt[va]
    best, best_ep, best_state, hist, bad = np.inf, -1, None, [], 0
    n = len(tr)
    steps = math.ceil(n / a.bs)
    for ep in range(a.epochs):
        model.train()
        t1 = time.time()
        perm = torch.randperm(n, device=dev)
        for s in range(steps):
            b = perm[s * a.bs:(s + 1) * a.bs]
            g = G["tr"]
            with ac():
                o = model(g["xn"][b], g["xf"][b], g["xl"][b], g["xh"][b])
            loss = loss_fn(o, b)
            if a.sched == "cos":
                fac = 0.5 * (1 + math.cos(math.pi * min(step_i, total_steps) / total_steps))
                for o, bl in zip(opts, base_lr):
                    for pg, b0 in zip(o.param_groups, bl):
                        pg["lr"] = b0 * fac
            step_i += 1
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for o in opts:
                o.step()
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
    log(f"best epoch {best_ep}, es-day rmse {best:.3f}, params {n_par}, host peak working set {_peak_wset():.2f} GiB")
    mu_tabm = anchor[ev] + predict("ev")
    res = {"MVT_ID_mvt": R["ids"][ev], "mu_tabm": mu_tabm, "y": y[ev]}
    rep = {"tag": a.tag, "setting": a.setting, "seed": a.seed, "anchor": a.anchor, "args": vars(a), "xdir": str(xd), "best_epoch": best_ep,
           "es_rmse": round(best, 3), "hist": hist, "params": n_par, "target_mean": t_mean, "target_sd": t_sd,
           "minutes": round((time.time() - T0) / 60, 1), "gpu_peak_gib": round(gpk(), 2), "host_peak_wset_gib": round(_peak_wset(), 2),
           "versions": {"tabm": tabm.__version__, "torch": torch.__version__}}
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(res).to_parquet(out, index=False)
    rd = R18 / "runs"
    rd.mkdir(exist_ok=True)
    (rd / f"{'smoke_' if a.smoke else ''}{a.tag}_{a.setting}_s{a.seed}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"saved {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
