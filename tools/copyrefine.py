# SPDX-License-Identifier: GPL-3.0-only
"""R10-B C1: copy-probability refinement for non-G LIRF departures with a large copy stake, as the last step of 491, 492 or 493.
Pre-registration: private notes research/2026-10-07-r10-b-aggregates.md 5.2 (2026-10-07 05:30 KST, before any dMSE was measured).

Rows touched: LIRF, not G, not antiwin, q < 0.98, z = d - mn_f >= 900 s, where q is the chain's non-G copy probability (the corrector's p_art
after artmm and arthist, as src/aiu_shift.frame reads it) and mn_f = (f - q d)/(1 - q) the normal component the final prediction f implies.
  logit(q') = logit(q) + b0 + b1 x1 + ... + b5 x5
  x1 = clip((LOBT - SCHED)/60, -30, 90)/30, x2 = clip((EOBT_1 - SCHED)/60, -30, 90)/30,
  x3, x4, x5 = logit of the 2025 non-G LIRF copy share of (carrier x stand group), (carrier x local SCHED 3-hour band), (stand), each shrunk
  to its base share with k = 50, minus the logit of the base share; shares count 2025 months outside Jan and Jul (val, test) or Feb-Jun (fwd).
  Fitted by logistic loss (Newton, L2 1e-3, offset logit q) on labelled rows meeting the same conditions: val Jan from val Jul and the reverse,
  fwd from val Jan, test from val Jan + Jul.  f' = f + (q' - q)(d - mn_f), then the LOBT window [d_lobt - 3606, d_lobt + 3606] and 0.
G rows, antiwin rows and every other row stay byte for byte, so the step composes with a G-only step in either order.
493 = 0.5 x 491 + 0.5 x 492 (tools/chainavg.py): the step is applied to each half with its own chain's q and averaged.
Usage (repo root, inside memcap):
    .venv/Scripts/python.exe -X utf8 tools/copyrefine.py eval <491|492|493>
    .venv/Scripts/python.exe -X utf8 tools/copyrefine.py build <491|492|493>      -> research/r10_b/c1_<chain>/ files + manifest
    .venv/Scripts/python.exe -X utf8 tools/copyrefine.py --setting {val,fwd,full} --chain <491|492|493> --tag <tag>   (one file again)
"""
from __future__ import annotations

import ctypes
import datetime as dt
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
OUT = ROOT / "research" / "r10_b"
SNAP = "preds/code/cddee405393b5d43_aiu_shift.py"
PRE = {"491": "m6r27wincbrmfw4adsbgeondctxv3x50wamfhgkcxncr", "492": "m6r27skcbrmfw4adsbgeondctxv3x50wamfhgkcxncr"}
FINAL = {"491": PRE["491"] + "dbawgp", "492": PRE["492"] + "dbawgp", "493": PRE["491"] + "dbawgpavsk"}
KEY = {"val": "holdout", "fwd": "forward", "full": "test"}
ENC_MONTHS = {"val": [2, 3, 4, 5, 6, 8, 9, 10, 11, 12], "fwd": [2, 3, 4, 5, 6], "full": [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]}
ZMIN, QMAX, K, L2, HALF = 900.0, 0.98, 50.0, 1e-3, 3606.0
N_TEST = 344841
TODAY_Q = ["423", "456", "446", "491", "493", "492"]
XCOLS = ["x1", "x2", "x3", "x4", "x5"]


def below_normal() -> None:
    if sys.platform == "win32":
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)


def sha16(rel) -> str:
    h = hashlib.sha256()
    with open(ROOT / rel, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def now_kst() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")


def load_mod(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = load_mod(SNAP, "aiu_snap_c1")
A.ROOT = ROOT


def man(spec: str) -> dict:
    return json.loads((ROOT / "preds" / f"manifest_{spec}.json").read_text(encoding="utf-8"))


def pfile(spec: str, st: str) -> pd.Series:
    return pd.read_parquet(ROOT / man(spec)[KEY[st]]["path"]).set_index("MVT_ID_mvt")["pred"].astype(float)


def lirf_fields() -> pd.DataFrame:
    """LIRF NM-matched departures (train and test): fields of the score and the copy label."""
    cols = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "t_mvt", "t_sch", "t_lobt", "t_eobt1", "y", "src", "STAND_mvt"]
    t = pq.read_table(ROOT / "cache" / "all.parquet", columns=cols, filters=[("PHASE_mvt", "==", "DEP"), ("ADEP_mvt", "==", "LIRF")],
                      read_dictionary=["FLIGHT_mvt", "STAND_mvt"])
    X = t.to_pandas()
    X = X[X["FLIGHT_ID_mvt"].notna()].drop(columns="FLIGHT_ID_mvt")
    X["FLIGHT_mvt"] = X["FLIGHT_mvt"].astype(str)
    X["STAND_mvt"] = X["STAND_mvt"].astype(str)
    X["car"] = X["FLIGHT_mvt"].str[:3]
    X["stand_g"] = X["STAND_mvt"].str[:1]
    lh = pd.to_datetime(X["t_sch"], unit="s", utc=True).dt.tz_convert("Europe/Rome").dt.hour
    X["car_stand"] = X["car"] + "|" + X["stand_g"]
    X["car_h"] = X["car"] + "|" + (lh // 3 * 3).astype(str)
    X["art"] = (np.abs(X["y"] - (X["t_mvt"] - X["t_sch"])) <= 60).astype(float)
    X["x1"] = np.clip((X["t_lobt"] - X["t_sch"]) / 60.0, -30, 90) / 30.0
    X["x2"] = np.clip((X["t_eobt1"] - X["t_sch"]) / 60.0, -30, 90) / 30.0
    X["x1"] = X["x1"].fillna(0.0)
    X["x2"] = X["x2"].fillna(X["x1"])
    return X.set_index("MVT_ID_mvt")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def encode(X: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    T = X[X["src"].isin(months)]
    base = float(T["art"].mean())
    out = pd.DataFrame(index=X.index)
    for nm, f in (("x3", "car_stand"), ("x4", "car_h"), ("x5", "STAND_mvt")):
        g = T.groupby(f)["art"].agg(["sum", "size"])
        rate = (g["sum"] + K * base) / (g["size"] + K)
        out[nm] = logit(X[f].map(rate).fillna(base).to_numpy()) - logit(base)
    out["x1"], out["x2"] = X["x1"], X["x2"]
    return out[XCOLS]


def chain_frame(chain: str, st: str) -> pd.DataFrame:
    """rows of the setting: q, d, G, antiwin mask, d_lobt from src/aiu_shift.frame on the chain's pre-db source; f = the chain's final file."""
    cfg = man(PRE[chain] + "db")["config"]
    cx, fx, _ = A.inputs(cfg["--ctx"], cfg["--ctx_sha"], {"--aiu25_sha": cfg["--aiu25_sha"], "--aiu26_sha": cfg["--aiu26_sha"]})
    ch = A.chain(f"preds/manifest_{PRE[chain]}.json")
    D = A.frame(st, ch, cx, fx)
    D = D[["id", "ap", "day", "q", "d", "g", "aw", "dl"] + (["y"] if "y" in D.columns else [])].copy()
    D["f"] = pfile(FINAL[chain], st).reindex(D["id"].to_numpy()).to_numpy()
    if D["f"].isna().any():
        raise SystemExit(f"{chain} {st}: final file rows do not line up")
    return D


def select(D: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    q, d, f = D["q"].to_numpy(), D["d"].to_numpy(), D["f"].to_numpy()
    with np.errstate(all="ignore"):
        mnf = (f - q * d) / (1 - q)
    sel = (D["ap"].to_numpy() == "LIRF") & ~D["g"].to_numpy() & ~D["aw"].to_numpy() & (q < QMAX) & np.isfinite(mnf) & ((d - mnf) >= ZMIN)
    return sel, mnf


def fit(Z: pd.DataFrame) -> np.ndarray:
    """Newton on logistic loss with offset logit q, L2 on all coefficients."""
    Xm = np.column_stack([np.ones(len(Z))] + [Z[c].to_numpy(dtype=float) for c in XCOLS])
    off = logit(Z["q"].to_numpy())
    yv = Z["art"].to_numpy(dtype=float)
    b = np.zeros(Xm.shape[1])
    for _ in range(50):
        eta = off + Xm @ b
        p = 1 / (1 + np.exp(-eta))
        gr = Xm.T @ (p - yv) / len(yv) + L2 * b
        H = (Xm * (p * (1 - p))[:, None]).T @ Xm / len(yv) + L2 * np.eye(len(b))
        step = np.linalg.solve(H, gr)
        b = b - step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b


def apply(D: pd.DataFrame, E: pd.DataFrame, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sel, mnf = select(D)
    f = D["f"].to_numpy().copy()
    ids = D["id"].to_numpy()
    Xs = E.reindex(ids[sel])
    if Xs.isna().any().any():
        raise SystemExit("rows to touch without score fields")
    q = D["q"].to_numpy()[sel]
    Xm = np.column_stack([np.ones(sel.sum())] + [Xs[c].to_numpy(dtype=float) for c in XCOLS])
    q2 = 1 / (1 + np.exp(-(logit(q) + Xm @ b)))
    new = f[sel] + (q2 - q) * (D["d"].to_numpy()[sel] - mnf[sel])
    dl = D["dl"].to_numpy()[sel]
    new = np.where(np.isfinite(dl), np.clip(new, dl - HALF, dl + HALF), new)
    f[sel] = np.maximum(new, 0.0)
    return f, sel


def fit_rows(V: pd.DataFrame, X: pd.DataFrame, E: pd.DataFrame) -> pd.DataFrame:
    sel, _ = select(V)
    Z = V[sel][["id", "q"]].copy()
    Z["art"] = X["art"].reindex(Z["id"].to_numpy()).to_numpy()
    Z = Z.join(E, on="id")
    return Z.dropna()


def run_chain(chain: str, st: str, X: pd.DataFrame, frames: dict) -> tuple[pd.Series, dict]:
    D = frames[(chain, st)]
    V = frames[(chain, "val")]
    E = encode(X, ENC_MONTHS[st])
    jan = (V["day"].str[5:7] == "01").to_numpy()
    rep = {"fits": []}
    if st == "val":
        f = D["f"].to_numpy().copy()
        for msk, fitm in ((jan, ~jan), (~jan, jan)):
            b = fit(fit_rows(V[fitm], X, E))
            fm, sel = apply(D[msk].reset_index(drop=True), E, b)
            f[msk] = fm
            rep["fits"].append({"b": [round(float(x), 4) for x in b], "rows_touched": int(sel.sum())})
    else:
        Vf = V[jan] if st == "fwd" else V
        b = fit(fit_rows(Vf, X, E))
        f, sel = apply(D, E, b)
        rep["fits"].append({"b": [round(float(x), 4) for x in b], "rows_touched": int(sel.sum())})
    return pd.Series(f, index=D["id"].to_numpy()), rep


def run(chain: str, st: str, X: pd.DataFrame, frames: dict) -> tuple[pd.Series, dict]:
    if chain != "493":
        return run_chain(chain, st, X, frames)
    s1, r1 = run_chain("491", st, X, frames)
    s2, r2 = run_chain("492", st, X, frames)
    s = 0.5 * s1 + 0.5 * s2.reindex(s1.index)
    return s, {"491": r1, "492": r2}


def day_boot(dd: np.ndarray, day: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    g = pd.DataFrame({"d": dd, "day": day}).groupby("day")["d"].agg(["sum", "size"])
    s, c = g["sum"].to_numpy(), g["size"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    dr = rng.integers(0, len(s), size=(n, len(s)))
    bt = s[dr].sum(axis=1) / c[dr].sum(axis=1)
    return float(s.sum() / c.sum()), float(bt.std(ddof=1)), float((bt < 0).mean())


def guard(pred: pd.Series) -> dict:
    import src.shot as S
    tpl = S.template_ids()
    vals = S.submission_values(pred, tpl)
    gd = S.guard(vals, tpl)
    qd = {}
    for num in TODAY_Q:
        p = sorted((ROOT / "queue").glob(f"{num}_*.parquet")) + sorted((ROOT / "queue" / "done").glob(f"{num}_*.parquet"))
        if p:
            q = pd.read_parquet(p[0]).set_index("MVT_ID_mvt")["pred"]
            qd[p[0].stem] = round(float(np.mean(S.submission_values(q, tpl) != vals)), 5)
    return {"ledger_pass": bool(gd["pass"]), "ledger_nearest": gd["nearest"], "ledger_min_diff": round(float(gd["min_diff_frac"]), 5) if gd["min_diff_frac"] is not None else None,
            "queue_diff": qd, "queue_min_diff": min(qd.values()) if qd else None}


def load_all(chain: str) -> tuple[pd.DataFrame, dict]:
    X = lirf_fields()
    frames = {}
    for c in (["491", "492"] if chain == "493" else [chain]):
        for st in ("val", "fwd", "full"):
            frames[(c, st)] = chain_frame(c, st)
    if chain == "493":  # 493 must be 0.5 x 491 + 0.5 x 492
        for st in ("val", "fwd", "full"):
            f3 = pfile(FINAL["493"], st)
            a = frames[("491", st)].set_index("id")["f"]
            b2 = frames[("492", st)].set_index("id")["f"]
            err = float(np.max(np.abs((0.5 * a + 0.5 * b2.reindex(a.index)) - f3.reindex(a.index))))
            print(f"  493 {st}: |0.5 x 491 + 0.5 x 492 - 493| max {err:.2e}", flush=True)
            if err > 1e-6:
                raise SystemExit("493 is not the half-half blend of 491 and 492")
    return X, frames


def evaluate(chain: str) -> dict:
    X, frames = load_all(chain)
    base = "491" if chain == "493" else chain
    out = {"chain": chain, "kst": now_kst()}
    for st in ("val", "fwd"):
        D = frames[(base, st)]
        s, rep = run(chain, st, X, frames)
        f0 = pfile(FINAL[chain], st).reindex(s.index).to_numpy()
        y = D.set_index("id")["y"].reindex(s.index).to_numpy()
        day = D.set_index("id")["day"].reindex(s.index).to_numpy()
        dd = (s.to_numpy() - y) ** 2 - (f0 - y) ** 2
        pt, sd, p = day_boot(dd, day)
        mon = {m: round(float(dd[pd.Series(day).str[5:7].to_numpy() == m].sum() / len(dd)), 1) for m in sorted(set(pd.Series(day).str[5:7]))}
        moved = np.abs(s.to_numpy() - f0) > 0
        out[st] = {"rmse_base": round(float(np.sqrt(((f0 - y) ** 2).mean())), 3), "rmse": round(float(np.sqrt(((s.to_numpy() - y) ** 2).mean())), 3),
                   "dmse": round(pt, 2), "boot_sd": round(sd, 2), "p_below0": round(p, 4), "months": mon, "rows_moved": int(moved.sum()),
                   "improved": int(((s.to_numpy() - y) ** 2 < (f0 - y) ** 2)[moved].sum()), "fits": rep}
        print(f"[C1 {chain}] {st}: rmse {out[st]['rmse_base']} -> {out[st]['rmse']}, dMSE {pt:+.2f} +- {sd:.2f} (P<0 {p:.4f}), months {mon}, "
              f"rows moved {int(moved.sum())} (better {out[st]['improved']}), fits {json.dumps(rep)}", flush=True)
    s, rep = run(chain, "full", X, frames)
    f0 = pfile(FINAL[chain], "full").reindex(s.index).to_numpy()
    mv = s.to_numpy() - f0
    out["full"] = {"move_mse": round(float(np.sum(mv ** 2) / N_TEST), 2), "rows_moved": int((np.abs(mv) > 0).sum()), "rows_moved_0.5": int((np.abs(mv) > 0.5).sum()),
                   "mean_move": round(float(mv.mean()), 3), "fits": rep, "guard": guard(s)}
    print(f"[C1 {chain}] full: move MSE {out['full']['move_mse']}, rows moved {out['full']['rows_moved']} (> 0.5 s {out['full']['rows_moved_0.5']}), "
          f"mean move {out['full']['mean_move']}, guard {out['full']['guard']}", flush=True)
    g = {"1_val": out["val"]["dmse"] < 0 and out["val"]["p_below0"] >= 0.9, "2_fwd": out["fwd"]["dmse"] < 0 and out["fwd"]["p_below0"] >= 0.9,
         "3_months": all(v <= 0 for v in out["val"]["months"].values()),
         "6_guard": out["full"]["guard"]["ledger_pass"] and (out["full"]["guard"]["queue_min_diff"] or 0) >= 0.01}
    out["gates"] = g
    print(f"[C1 {chain}] gates {g}", flush=True)
    (OUT / f"c1_eval_{chain}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def main() -> int:
    below_normal()
    a = sys.argv[1:]
    if a and a[0] == "eval":
        evaluate(a[1])
        return 0
    raise SystemExit("usage: eval <491|492|493> | build <chain> (after eval)")


if __name__ == "__main__":
    sys.exit(main())
