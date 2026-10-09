# SPDX-License-Identifier: GPL-3.0-only
"""R10-B: AIU day-total reconciliation candidates in the 467 (db) slot of the 491 chain. Pre-registration and results: private notes
research/2026-10-07-r10-b-aggregates.md 3 (registered 2026-10-07 05:16 KST, before any candidate was measured).

The candidates replace the db step only: 464 -> candidate -> tools/antiwin.py rules -> 491's G log-pool delta (491 - 468, G rows only).
No candidate moves G rows or antiwin rows, so the G log-pool step is unchanged. Frames (q, mn, d, G, antiwin mask, d_lobt per row) come from
preds/code/cddee405393b5d43_aiu_shift.py, the code 467 ran; `frames` first checks that the rebuilt db (lam 0.5) reproduces 467, 468 and 491.
  grouping v1 (db today): estimated off-block UTC date (MVT - mn), all rows, airport offset = median over fitting days.
  grouping v2: VFR rows (FLIGHT_RULE_mvt = V) leave the day sums and do not move, except at EDDF (AIU counts them there); a day whose row
    count (estimated off-block date) differs from FLT_DEP_IFR_2 by more than max(3, 0.005 FLT) is not reconciled; fitting days pass the same
    check on the true off-block date; offset = median over passing fitting days.
  B3: v1, db's uniform shift (+-60, S = pred < 3600, not G, not antiwin), lam fitted.   A2: v2, uniform, lam 0.5.   A2L: v2, uniform, lam fitted.
  B1b: v2, variance-weighted: per row z = d - mn, e = g(z), g' = (g(z+30) - g(z-30))/60; type cov q(1-q)(d-mn)e, var q(1-q)e^2; within-component
    cov (1-q) v g', var (1-q) v g'^2; common loading n = (1-q) g', N = day sum; V = sum(vars) + tau2 N^2 + sigma2; r = T - off - sum (1-q) e;
    movable rows (not G, not antiwin, not dropped VFR; pred >= 3600 included) get -lam (cov_type + cov_within + (1-q) tau2 N) r / V, cut at
    +-600 s, then the LOBT window and 0 (all rows, as db).  B1a: B1b with tau2 = 0.
  Fitting: val Jan rows from val Jul and the reverse, fwd from val Jan, test and nd from val Jan+Jul. lam = sum(u r)/sum(u^2) on the fitting
    rows, u = f(lam=1) - f(lam=0) with the parameters fitted on those same rows, r = y - f(lam=0); one lam for all airports, cut to [0, 1.5].
Usage (repo root, inside memcap):
    .venv/Scripts/python.exe -X utf8 tools/aiurecon.py frames            -> research/r10_b/frames/{val,fwd,full,nd}.parquet (+ frames.json)
    .venv/Scripts/python.exe -X utf8 tools/aiurecon.py eval B3,A2,A2L,B1a,B1b   -> research/r10_b/eval_<variant>.json (no prediction files)
    .venv/Scripts/python.exe -X utf8 tools/aiurecon.py build <variant>   -> research/r10_b/<variant>/{holdout,fwd,test}_<tag>.parquet, manifest_<tag>.json
    .venv/Scripts/python.exe -X utf8 tools/aiurecon.py --setting {val,fwd,full} --variant <v> --frames_sha <16> --tag <tag>   (one file again)
"""
from __future__ import annotations

import ctypes
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
OUT = ROOT / "research" / "r10_b"
FR = OUT / "frames"
SNAP = "preds/code/cddee405393b5d43_aiu_shift.py"
C464 = "m6r27wincbrmfw4adsbgeondctxv3x50wamfhgkcxncr"
CH = {"464": C464, "467": C464 + "db", "468": C464 + "dbaw", "491": C464 + "dbawgp"}
KEY = {"val": "holdout", "fwd": "forward", "full": "test"}
APS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
VFR_KEEP = {"EDDF"}
CMAX, HALF, CUT_B1 = 60.0, 3606.0, 600.0
LAM_LO, LAM_HI = 0.0, 1.5
N_TEST = 344841
VARIANTS = {"B3": ("v1", "uniform"), "A2": ("v2", "uniform"), "A2L": ("v2", "uniform"), "B1a": ("v2", "blup0"), "B1b": ("v2", "blup")}
FIXED_LAM = {"A2": 0.5}
TODAY_Q = ["423", "456", "446", "491", "493", "492"]


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


A = load_mod(SNAP, "aiu_snap")
A.ROOT = ROOT
AW = load_mod("tools/antiwin.py", "antiwin_r10b")


def man(spec: str) -> dict:
    return json.loads((ROOT / "preds" / f"manifest_{spec}.json").read_text(encoding="utf-8"))


def pfile(spec: str, st: str) -> pd.Series:
    return pd.read_parquet(ROOT / man(spec)[KEY[st]]["path"]).set_index("MVT_ID_mvt")["pred"].astype(float)


def aiu_counts() -> pd.Series:
    out = []
    for rel in A.AIU.values():
        a = pd.read_csv(ROOT / rel)
        a = a[a["APT_ICAO"].isin(APS)]
        out.append(pd.Series(a["FLT_DEP_IFR_2"].to_numpy(dtype=float), index=(a["APT_ICAO"] + "|" + a["FLT_DATE"]).to_numpy()))
    return pd.concat(out)


def all_cols(src_in: list[int]) -> pd.DataFrame:
    t = pq.read_table(ROOT / "cache" / "all.parquet", columns=["MVT_ID_mvt", "t_mvt", "FLIGHT_RULE_mvt", "src"],
                      filters=[("PHASE_mvt", "==", "DEP"), ("src", "in", src_in)], read_dictionary=["FLIGHT_RULE_mvt"])
    return pd.DataFrame({"t": t.column("t_mvt").to_numpy().astype(float), "vfr": (t.column("FLIGHT_RULE_mvt").to_pandas().astype(str) == "V").to_numpy()},
                        index=t.column("MVT_ID_mvt").to_numpy())


# ---------------------------------------------------------------- frames
def frames() -> int:
    m467 = man(CH["467"])
    cfg = m467["config"]
    assert cfg.get("--group") == "block" and cfg["--lam"] == "0.5" and cfg["--cmax"] == "60", cfg
    assert sha16(SNAP) == m467["code_sha"]["src/aiu_shift.py"], "snapshot differs from the code 467 ran"
    cx, fx, tot = A.inputs(cfg["--ctx"], cfg["--ctx_sha"], {"--aiu25_sha": cfg["--aiu25_sha"], "--aiu26_sha": cfg["--aiu26_sha"]})
    ch = A.chain(f"preds/manifest_{C464}.json")
    X = all_cols([0, 1, 7, 11, 12])
    FR.mkdir(parents=True, exist_ok=True)
    info = {"snapshot": SNAP, "snapshot_sha16": sha16(SNAP), "chain": CH, "written_kst": now_kst(), "files": {}}
    V = None
    t_ctx = A.ctx_t(cfg["--ctx"])
    for st in ("val", "fwd", "full"):
        D = A.frame(st, ch, cx, fx)
        if st == "val":
            V = D
        tt = t_ctx.reindex(D["id"].to_numpy())
        if tt.isna().any():
            raise SystemExit(f"{st}: rows without t_mvt")
        tser = pd.Series(tt.to_numpy(), index=D["id"].to_numpy())
        D["bday"] = A.block_days(D, tser, "mn")
        if "y" in D.columns:
            D["ydate"] = A.block_days(D, tser, "y")
        D["vfr"] = X["vfr"].reindex(D["id"].to_numpy()).to_numpy().astype(bool)
        D["t"] = tt.to_numpy()
        # db rebuilt at lam 0.5, same fitting as src/aiu_shift.shifted(group block)
        f = np.full(len(D), np.nan)
        if st == "val":
            jan = (D["day"].str[5:7] == "01").to_numpy()
            parts = [(jan, ~jan), (~jan, jan)]
            for msk, fit in parts:
                Dm = D[msk].reset_index(drop=True)
                F = D[fit]
                fm, _ = A.shift_days(Dm.assign(day=Dm["bday"]), A.fit_g(F), A.offsets(F.assign(day=F["ydate"]), tot), tot, 0.5, CMAX)
                f[msk] = fm
        else:
            jan = (V["day"].str[5:7] == "01").to_numpy()
            F = V[jan] if st == "fwd" else V
            f, _ = A.shift_days(D.assign(day=D["bday"]), A.fit_g(F), A.offsets(F.assign(day=F["ydate"]), tot), tot, 0.5, CMAX)
        D["f467"] = f
        ids = D["id"].to_numpy()
        for k in ("467", "468", "491"):
            D[f"file{k}"] = pfile(CH[k], st).reindex(ids).to_numpy()
        err467 = float(np.nanmax(np.abs(D["f467"] - D["file467"])))
        aw468, _ = AW.apply_rules(pd.Series(D["f467"].to_numpy(), index=ids), AW.build_link(), st)
        err468 = float(np.nanmax(np.abs(aw468.reindex(ids).to_numpy() - D["file468"].to_numpy())))
        dg = D["file491"] - D["file468"]
        nonG = int(((dg.abs() > 0) & ~D["g"]).sum())
        print(f"[frames] {st}: rows {len(D)}, db rebuilt vs 467 max |diff| {err467:.2e}, + antiwin vs 468 {err468:.2e}, 491 - 468 non-zero on non-G rows {nonG}, "
              f"VFR rows {int(D['vfr'].sum())}", flush=True)
        if not (err467 < 1e-6 and err468 < 1e-6 and nonG == 0):
            raise SystemExit(f"{st}: the rebuilt db does not reproduce the chain")
        keep = ["id", "ap", "day", "bday", "t", "vfr", "f", "q", "mn", "d", "g", "aw", "dl", "f467", "file467", "file468", "file491"]
        keep += [c for c in ("y", "ydate") if c in D.columns]
        rel = f"research/r10_b/frames/{st}.parquet"
        D[keep].to_parquet(ROOT / rel, index=False)
        info["files"][st] = {"path": rel, "sha16": sha16(rel), "rows": int(len(D)), "err467": err467, "err468": err468}
    # nd: R7-U e7's old-leg emulation (cache/nd_m6r25_s0.parquet), no G log pool
    b = pd.read_parquet(ROOT / "cache" / "nd_m6r25_s0.parquet")
    t = pq.read_table(ROOT / "cache" / "feats_v25.parquet", columns=["MVT_ID_mvt", "src", "ap", "d_lobt", "dd"], filters=[("src", "in", [11, 12])]).to_pandas().set_index("MVT_ID_mvt")
    ids = b["MVT_ID_mvt"].to_numpy()
    N = pd.DataFrame({"id": ids, "fc": b["pred"].to_numpy(), "f": b["pred"].to_numpy(), "p": b["p_art"].to_numpy(), "mu": b["mu"].to_numpy(),
                      "g": b["g"].to_numpy().astype(bool), "d": b["dsch"].to_numpy(), "base": b["pred"].to_numpy(), "y": b["y"].to_numpy()})
    N["ap"] = t["ap"].astype(str).reindex(ids).to_numpy()
    tt = X["t"].reindex(ids)
    N["t"] = tt.to_numpy()
    N["day"] = pd.to_datetime(tt.to_numpy(), unit="s").strftime("%Y-%m-%d")
    N["dl"] = t["d_lobt"].reindex(ids).to_numpy(dtype=float)
    N["aw"] = A.antiwin_mask("nd", pd.Series(N["f"].to_numpy(), index=ids)).reindex(ids).to_numpy()
    N["m"] = N["mu"]
    N = A.add_classes(N, t["dd"].astype(float))
    tser = pd.Series(N["t"].to_numpy(), index=ids)
    N["bday"] = A.block_days(N, tser, "mn")
    N["ydate"] = A.block_days(N, tser, "y")
    N["vfr"] = X["vfr"].reindex(ids).to_numpy().astype(bool)
    keep = ["id", "ap", "day", "bday", "t", "vfr", "f", "q", "mn", "d", "g", "aw", "dl", "y", "ydate"]
    rel = "research/r10_b/frames/nd.parquet"
    N[keep].to_parquet(ROOT / rel, index=False)
    info["files"]["nd"] = {"path": rel, "sha16": sha16(rel), "rows": int(len(N))}
    print(f"[frames] nd: rows {len(N)}, antiwin rows {int(N['aw'].sum())}, VFR rows {int(N['vfr'].sum())}", flush=True)
    (FR / "frames.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- engine
class Ctx:
    def __init__(self):
        _, _, self.tot = A.inputs("ctx_v3", man(CH["467"])["config"]["--ctx_sha"],
                                  {"--aiu25_sha": man(CH["467"])["config"]["--aiu25_sha"], "--aiu26_sha": man(CH["467"])["config"]["--aiu26_sha"]})
        self.cnt = aiu_counts()
        self.link = AW.build_link()


def kept(D: pd.DataFrame, grouping: str) -> np.ndarray:
    if grouping == "v1":
        return np.ones(len(D), bool)
    return ~(D["vfr"].to_numpy() & ~D["ap"].isin(VFR_KEEP).to_numpy())


def count_ok(keys: pd.Index, n: np.ndarray, cnt: pd.Series) -> np.ndarray:
    c = cnt.reindex(keys).to_numpy(dtype=float)
    return np.isfinite(c) & (np.abs(n - c) <= np.maximum(3.0, 0.005 * np.nan_to_num(c)))


def fit_nuis(F: pd.DataFrame, cx: Ctx, grouping: str) -> dict:
    """nuisance parameters from fitting rows F (with y): g, offsets; for v2 also sigma2, v, tau2 per airport."""
    g = A.fit_g(F)
    if grouping == "v1":
        return {"g": g, "off": A.offsets(F.assign(day=F["ydate"]), cx.tot), "grouping": "v1"}
    k = kept(F, "v2")
    Fk = F[k]
    key = (Fk["ap"] + "|" + Fk["ydate"]).to_numpy()
    ours = pd.Series(np.maximum(0.0, Fk["d"].to_numpy() - Fk["y"].to_numpy())).groupby(key).sum()
    n = pd.Series(1.0, index=range(len(Fk))).groupby(key).sum().reindex(ours.index)
    ok = count_ok(ours.index, n.to_numpy(), cx.cnt) & ~ours.index.str.endswith("|out")
    diff = (cx.tot.reindex(ours.index) - ours)[ok]
    off = {a: float(x.dropna().median()) for a, x in diff.groupby(diff.index.str[:4])}
    sig2 = {a: float(np.nanvar((x - off[a]).dropna().to_numpy())) for a, x in diff.groupby(diff.index.str[:4])}
    y, d, mn = Fk["y"].to_numpy(), Fk["d"].to_numpy(), Fk["mn"].to_numpy()
    nm = (np.abs(y - d) > 60) & ~Fk["g"].to_numpy() & (y < 60000) & ~Fk["aw"].to_numpy() & (Fk["ydate"].to_numpy() != "out")
    R = pd.DataFrame({"ap": Fk["ap"].to_numpy()[nm], "k": key[nm], "e": (y - mn)[nm]})
    dm = R.groupby("k")["e"].transform("mean")
    R["w"] = (R["e"] - dm) ** 2
    v = R.groupby("ap")["w"].mean().to_dict()
    S = R.groupby("k").agg(ap=("ap", "first"), m=("e", "mean"), n=("e", "size"))
    tau2 = {a: float(max(0.0, s["m"].var() - np.mean(v[a] / s["n"]))) for a, s in S.groupby("ap")}
    return {"g": g, "off": off, "sig2": sig2, "v": v, "tau2": tau2, "grouping": "v2"}


def gprime(g, z: np.ndarray) -> np.ndarray:
    return (g(z + 30.0) - g(z - 30.0)) / 60.0


def apply(D: pd.DataFrame, nu: dict, kind: str, lam: float, cx: Ctx, rep: dict | None = None) -> np.ndarray:
    """new predictions (before antiwin) for the rows of D, grouped by D['bday']. kind: uniform, blup, blup0."""
    g, off, tot = nu["g"], nu["off"], cx.tot
    f_in = D["f"].to_numpy(dtype=float)
    f = f_in.copy()
    q, d, mn = D["q"].to_numpy(), D["d"].to_numpy(), D["mn"].to_numpy()
    G, aw = D["g"].to_numpy(), D["aw"].to_numpy()
    k = kept(D, nu["grouping"])
    key = (D["ap"] + "|" + D["bday"]).to_numpy()
    z = d - mn
    e = g(z)
    E = (1 - q) * e
    if kind == "uniform":
        shiftable = (f_in < 3600) & ~G & ~aw & k
    else:
        gp = gprime(g, z)
        move = ~G & ~aw & k
    base = np.flatnonzero(k) if nu["grouping"] == "v2" else np.arange(len(D))
    idx = pd.Series(base).groupby(key[base]).indices
    days_done, days_skip = 0, 0
    for kk, jj in idx.items():
        ii = base[jj]
        ap = kk[:4]
        if ap not in APS or kk.endswith("|out"):
            continue
        T = tot.get(kk, np.nan)
        if not np.isfinite(T):
            continue
        if nu["grouping"] == "v2":
            c_ = cx.cnt.get(kk, np.nan)
            if not (np.isfinite(c_) and abs(len(ii) - c_) <= max(3.0, 0.005 * c_)):
                days_skip += 1
                continue
        T = T - off.get(ap, 0.0)
        days_done += 1
        if kind == "uniform":
            sh = ii[shiftable[ii]]
            rest = E[ii[~shiftable[ii]]].sum()

            def tot_c(c):
                return np.sum((1 - q[sh]) * g(d[sh] - mn[sh] - c)) + rest
            lo, hi = -CMAX, CMAX
            if T >= tot_c(lo):
                c = lo
            elif T <= tot_c(hi):
                c = hi
            else:
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if tot_c(mid) > T:
                        lo = mid
                    else:
                        hi = mid
                c = 0.5 * (lo + hi)
            f[sh] = f[sh] + lam * (1 - q[sh]) * c
        else:
            v, tau2, sig2 = nu["v"].get(ap, 0.0), (0.0 if kind == "blup0" else nu["tau2"].get(ap, 0.0)), nu["sig2"].get(ap, 0.0)
            qi, ei, gpi, zi = q[ii], e[ii], gp[ii], z[ii]
            ct = qi * (1 - qi) * zi * ei
            st_ = qi * (1 - qi) * ei ** 2
            ci = (1 - qi) * v * gpi
            si = (1 - qi) * v * gpi ** 2
            nl = (1 - qi) * gpi
            Nk = nl.sum()
            Vk = st_.sum() + si.sum() + tau2 * Nk ** 2 + sig2
            r = T - E[ii].sum()
            if Vk <= 0:
                continue
            dlt = -lam * (ct + ci + (1 - qi) * tau2 * Nk) * r / Vk
            mv = move[ii]
            f[ii[mv]] = f[ii[mv]] + np.clip(dlt[mv], -CUT_B1, CUT_B1)
    dl = D["dl"].to_numpy()
    f = np.where(np.isfinite(dl), np.clip(f, dl - HALF, dl + HALF), f)
    f = np.maximum(f, 0.0)
    if rep is not None:
        rep.update(days_done=days_done, days_skipped_count=days_skip)
    return f


def fit_lam(F: pd.DataFrame, nu: dict, kind: str, cx: Ctx) -> tuple[float, float]:
    f0 = apply(F, nu, kind, 0.0, cx)
    f1 = apply(F, nu, kind, 1.0, cx)
    u, r = f1 - f0, F["y"].to_numpy() - f0
    raw = float(np.sum(u * r) / np.sum(u * u)) if np.sum(u * u) > 0 else 0.0
    return float(np.clip(raw, LAM_LO, LAM_HI)), raw


def run_setting(st: str, var: str, cx: Ctx, Fr: dict) -> tuple[pd.Series, dict]:
    """final predictions of the setting (pre-antiwin -> antiwin -> + 491's G delta; nd: no G delta) and a report."""
    grouping, kind = VARIANTS[var]
    V = Fr["val"]
    jan = (V["day"].str[5:7] == "01").to_numpy()
    D = Fr[st]
    rep = {"variant": var, "setting": st, "fits": []}
    if st == "val":
        pieces = [(jan, V[~jan]), (~jan, V[jan])]
    else:
        pieces = [(np.ones(len(D), bool), V[jan] if st == "fwd" else V)]
    f = np.full(len(D), np.nan)
    for msk, F in pieces:
        nu = fit_nuis(F, cx, grouping)
        if var in FIXED_LAM:
            lam, raw = FIXED_LAM[var], None
        else:
            lam, raw = fit_lam(F, nu, kind, cx)
        r1 = {}
        Dm = D[msk].reset_index(drop=True)
        f[msk] = apply(Dm, nu, kind, lam, cx, r1)
        fit = {"eval_rows": int(msk.sum()), "fit_rows": int(len(F)), "lam": lam, "lam_raw": raw, **r1}
        if grouping == "v2":
            fit.update(v={a: round(x, 0) for a, x in nu["v"].items()}, tau2={a: round(x, 1) for a, x in nu["tau2"].items()},
                       sig2={a: round(x, 0) for a, x in nu["sig2"].items()}, off={a: round(x, 1) for a, x in nu["off"].items()})
        rep["fits"].append(fit)
    ids = D["id"].to_numpy()
    s = pd.Series(f, index=ids)
    s, awi = AW.apply_rules(s, cx.link, st)
    s = s.reindex(ids)
    if st != "nd":  # 491's G log pool: on its rows the candidate must hold 468's value; there 491's value is what tools/gpool.py writes
        pre, f468, f491 = s.to_numpy(), D["file468"].to_numpy(), D["file491"].to_numpy()
        gmove = f491 != f468
        if np.any(gmove & (pre != f468)):
            raise SystemExit(f"{st}: the candidate moved rows of 491's G log pool")
        s = pd.Series(np.where(gmove, f491, pre), index=ids)
    rep["antiwin"] = awi
    return s, rep


def nd_base(cx: Ctx, Fr: dict) -> pd.Series:
    V, N = Fr["val"], Fr["nd"]
    nu = fit_nuis(V, cx, "v1")
    f = apply(N, nu, "uniform", 0.5, cx)
    s, _ = AW.apply_rules(pd.Series(f, index=N["id"].to_numpy()), cx.link, "nd")
    return s.reindex(N["id"].to_numpy())


def day_boot(dd: np.ndarray, day: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    g = pd.DataFrame({"d": dd, "day": day}).groupby("day")["d"].agg(["sum", "size"])
    s, c = g["sum"].to_numpy(), g["size"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    dr = rng.integers(0, len(s), size=(n, len(s)))
    bt = s[dr].sum(axis=1) / c[dr].sum(axis=1)
    return float(s.sum() / c.sum()), float(bt.std(ddof=1)), float((bt < 0).mean())


def load_frames() -> dict:
    info = json.loads((FR / "frames.json").read_text(encoding="utf-8"))
    Fr = {}
    for st, x in info["files"].items():
        if sha16(x["path"]) != x["sha16"]:
            raise SystemExit(f"{x['path']} sha16 differs from frames.json")
        Fr[st] = pd.read_parquet(ROOT / x["path"])
    return Fr


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


def evaluate(var: str, cx: Ctx, Fr: dict, ndb: pd.Series) -> dict:
    out = {"variant": var, "kst": now_kst()}
    for st in ("val", "fwd"):
        D = Fr[st]
        s, rep = run_setting(st, var, cx, Fr)
        y = D["y"].to_numpy()
        e1, e0 = (s.to_numpy() - y) ** 2, (D["file491"].to_numpy() - y) ** 2
        dd = e1 - e0
        pt, sd, p = day_boot(dd, D["day"].to_numpy())
        mon = {m: round(float(dd[(D["day"].str[5:7] == m).to_numpy()].sum() / len(D)), 1) for m in sorted(D["day"].str[5:7].unique())}
        apd = {a: round(float(dd[(D["ap"] == a).to_numpy()].sum() / len(D)), 1) for a in APS}
        g_same = bool(np.array_equal(s.to_numpy()[D["g"].to_numpy()], D["file491"].to_numpy()[D["g"].to_numpy()]))
        aw_same = bool(np.array_equal(s.to_numpy()[D["aw"].to_numpy()], D["file491"].to_numpy()[D["aw"].to_numpy()]))
        out[st] = {"rmse_491": round(float(np.sqrt(e0.mean())), 3), "rmse": round(float(np.sqrt(e1.mean())), 3), "dmse": round(pt, 2), "boot_sd": round(sd, 2),
                   "p_below0": round(p, 4), "months": mon, "airports": apd, "g_rows_same": g_same, "aw_rows_same": aw_same,
                   "moved_vs_491": int((np.abs(s.to_numpy() - D["file491"].to_numpy()) > 0.5).sum()), "fits": rep["fits"]}
        print(f"[{var}] {st}: rmse {out[st]['rmse_491']} -> {out[st]['rmse']}, dMSE {pt:+.2f} +- {sd:.2f} (P<0 {p:.4f}), months {mon}, G same {g_same}, aw same {aw_same}", flush=True)
        print(f"[{var}] {st}: airports {apd}", flush=True)
        print(f"[{var}] {st}: fits {json.dumps([{k: v for k, v in x.items() if k in ('lam', 'lam_raw', 'days_done', 'days_skipped_count', 'eval_rows')} for x in rep['fits']])}", flush=True)
    N = Fr["nd"]
    s, rep = run_setting("nd", var, cx, Fr)
    y = N["y"].to_numpy()
    dd = (s.to_numpy() - y) ** 2 - (ndb.to_numpy() - y) ** 2
    pt, sd, p = day_boot(dd, N["day"].to_numpy())
    mon = {m: round(float(dd[(N["day"].str[5:7] == m).to_numpy()].sum() / len(N)), 1) for m in ("11", "12")}
    out["nd"] = {"dmse": round(pt, 2), "boot_sd": round(sd, 2), "p_below0": round(p, 4), "months": mon, "fits": rep["fits"]}
    print(f"[{var}] nd: dMSE {pt:+.2f} +- {sd:.2f} (P<0 {p:.4f}), months {mon}, lam {[x['lam'] for x in rep['fits']]}", flush=True)
    Dt = Fr["full"]
    s, rep = run_setting("full", var, cx, Fr)
    mv = s.to_numpy() - Dt["file491"].to_numpy()
    out["full"] = {"move_mse": round(float(np.sum(mv ** 2) / N_TEST), 2), "rows_moved": int((np.abs(mv) > 0.5).sum()), "mean_move": round(float(mv.mean()), 3),
                   "fits": rep["fits"], "guard": guard(s)}
    print(f"[{var}] full: move MSE {out['full']['move_mse']}, rows moved {out['full']['rows_moved']}, mean move {out['full']['mean_move']}, "
          f"guard {out['full']['guard']}, lam {[x['lam'] for x in rep['fits']]}, days {[(x['days_done'], x['days_skipped_count']) for x in rep['fits']]}", flush=True)
    mo = [out["val"]["months"].get("01", 0), out["val"]["months"].get("07", 0), out["nd"]["months"]["11"], out["nd"]["months"]["12"]]
    gates = {"1_val": out["val"]["dmse"] < 0 and out["val"]["p_below0"] >= 0.9, "2_fwd": out["fwd"]["dmse"] < 0 and out["fwd"]["p_below0"] >= 0.9,
             "3_nd": out["nd"]["dmse"] < 0, "4_months": sum(x <= 0 for x in mo) >= 3,
             "5_same": out["val"]["g_rows_same"] and out["val"]["aw_rows_same"] and out["fwd"]["g_rows_same"] and out["fwd"]["aw_rows_same"],
             "6_guard": out["full"]["guard"]["ledger_pass"] and (out["full"]["guard"]["queue_min_diff"] or 0) >= 0.01}
    out["gates"] = gates
    out["pass"] = all(gates.values())
    print(f"[{var}] gates {gates} -> {'PASS' if out['pass'] else 'FAIL'}", flush=True)
    (OUT / f"eval_{var}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def main() -> int:
    below_normal()
    a = sys.argv[1:]
    if a and a[0] == "frames":
        return frames()
    if a and a[0] == "eval":
        cx = Ctx()
        Fr = load_frames()
        ndb = nd_base(cx, Fr)
        for var in a[1].split(","):
            evaluate(var, cx, Fr, ndb)
        return 0
    raise SystemExit("usage: frames | eval <variants> | build <variant> (build comes after eval)")


if __name__ == "__main__":
    sys.exit(main())
