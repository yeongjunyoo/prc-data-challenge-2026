# SPDX-License-Identifier: GPL-3.0-only
"""AIU day shift on top of a candidate (R5-A's rake3, ported into src/ on 10-06; method and measurements in the vault note
research/2026-10-06-r5-drift.md 4, reference implementation research/r5_drift/rake3_test.py).

Per airport-day the normal component of every ordinary row is moved by one shift c so that the predictions imply the published day total of
pre-departure delay (EUROCONTROL AIU all_pre_departure_delays, organizer-approved, README 외부 데이터; 60 x DLY_ALL_PRE_2 equals the sum of
max(0, BLOCK - SCHED) over the day's departures on 2025):
  1. per row: q = the corrector output's p_art (non-G rows), normal mean mn = mu + (corrector pred - base pred), d = dsch. G rows: the three
     components of model3's G override recovered from the stored pred, p_art and mu (q = copy + day push, mn = mu).
  2. g(z) = E[max(0, d - y) | z = d - mn] on non-copy, non-G ruler rows (y < 60,000): 100 z-quantile bins, running maximum, linear
     interpolation, slope 1 beyond the top. Airport offset = median over the fitting days of (AIU total - our sum of max(0, d - y)).
     Fitted on the other ruler month for val (January rows from July, July rows from January), on January for fwd, on both months for test.
  3. per airport-day: T = AIU total - offset; shift rows S = {prediction < 3,600, not G, not changed by antiwin}; the other rows add
     R = sum (1 - q) g(d - mn); c in [-cmax, cmax] solves sum_S (1 - q) g(d - mn - c) + R = T (bisection, the end value when it does not
     cross).
  4. rows in S get + lam (1 - q) c; then the LOBT window [d_lobt - 3,606, d_lobt + 3,606] again (feats_v25 d_lobt) and 0 below.
Sources: an antiwin entry (S excludes the rows antiwin changed; the shift is then the last step), or a pre-antiwin candidate (S
excludes the rows tools/antiwin.py will change, computed with its rules; antiwin is built on the result afterwards: base legs, V5
geometry, A3, gkey, this shift, antiwin last). When a tools/gkey.py step is in the source chain, the G rows' q and mn come from its G
model (q = 1 - the normal probability of model3.g_override, mn = the G leg's mu), recomputed with gkey's own functions and checked
against the gkey file first.
Inputs, read only: the source candidate's final predictions (its manifest), the corrector output it was built on
(cache/<setting>_<corrector run_tag>_s0.parquet: pred, p_art, mu, dsch, g, y) and its base blend (cache/<setting>_<--base>_s<seed>.parquet: pred),
feats_v25 (ap, d_lobt, dd), ctx_v3 (day), the two AIU files (sha16 checked against the config).
    .venv/Scripts/python.exe -X utf8 src/aiu_shift.py build <source queue name> <new queue number>
        computes val, fwd and test, writes preds/{holdout,fwd,test}_<source spec>ds files and preds/manifest_<source spec>ds.json (no enqueue)
    .venv/Scripts/python.exe -X utf8 src/aiu_shift.py --setting {val,fwd,full} --src <manifest> --src_sha <16> --lam 0.5 --cmax 60
        --ctx ctx_v3 --ctx_sha <16> --aiu25_sha <16> --aiu26_sha <16> --tag <spec>
        the reproduction command the manifest records; rewrites that setting's file
--group block (option, 10-07; without it everything above is unchanged): the airport-day groups use the UTC date of the estimated off-block time
    (MVT - mn) instead of the UTC date of MVT, and the airport offsets are fitted on the fitting rows grouped by their off-block date (MVT - y).
    AIU counts a departure's pre-departure delay on its off-block date (vault research/2026-10-07-r7-u.md 6.1, research/r7_u/e3_dayattr.py and
    e4_blockdate.py). Rows whose estimated off-block date is not one of the frame's MVT dates (the day before the month) are not shifted.
    build <source queue name> <new queue number> --group block writes preds/{holdout,fwd,test}_<source spec>db and the manifest records --group block.
--lam <x> on build (10-07): the shift multiplier instead of 0.5; the tag gets "l<1000 x>" after ds or db (e.g. dbl652), so a build at another lam
    never takes the 0.5 build's name. Without --lam (or with 0.5) the tag and every output are as before.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
APS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
HALF = 3606.0
AIU = {"--aiu25_sha": "data/ext/aiu/all_pre_departure_delays_2025.csv", "--aiu26_sha": "data/ext/aiu/all_pre_departure_delays_2026.csv"}
SETTING_KEY = {"val": "holdout", "fwd": "forward", "full": "test"}


def sha16(rel) -> str:
    h = hashlib.sha256()
    with open(ROOT / rel, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def now_kst() -> str:
    return dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")


def chain(src_mp: str) -> dict:
    """Walk the source chain of a candidate: its final files; the corrector output and base blend it was built on (the first manifest in the
    chain with cache/val_<run_tag>_s0.parquet); for an antiwin entry its direct source's files (to see which rows antiwin changed); and the
    tools/gkey.py manifest when a gkey step is in the chain (its G model gives the G rows' class probabilities)."""
    def load(rel):
        return json.loads((ROOT / rel).read_text(encoding="utf-8"))
    links = [load(src_mp)]
    while links[-1].get("source", {}).get("manifest"):
        links.append(load(links[-1]["source"]["manifest"]))
    corr = next(x for x in links if (ROOT / "cache" / f"val_{x.get('run_tag', x['spec'])}_s0.parquet").exists())
    m = links[0]
    is_aw = m["spec"].endswith("aw")
    return {"manifest": m, "corr": corr.get("run_tag", corr["spec"]), "base": corr["config"]["--base"],
            "base_seed": corr["config"].get("--base_seed", "0"), "final": {k: m[v]["path"] for k, v in SETTING_KEY.items()}, "is_aw": is_aw,
            "pre_final": {k: links[1][v]["path"] for k, v in SETTING_KEY.items()} if is_aw else None,
            "gkey": next((x for x in links if "--leg" in x.get("config", {}) and "--variant" in x.get("config", {})), None)}


def load_tool(name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def antiwin_mask(setting: str, f: pd.Series) -> pd.Series:
    """Rows tools/antiwin.py will change on these predictions (a pre-antiwin source: antiwin comes last, after this step)."""
    aw = load_tool("antiwin")
    new, _ = aw.apply_rules(f.astype(float), aw.build_link(), setting)
    return pd.Series(new.reindex(f.index).to_numpy() != f.to_numpy(), index=f.index)


def gkey_classes(setting: str, gm: dict, f_src: pd.Series, f_gk: pd.Series) -> pd.DataFrame:
    """The G rows' normal-class probability and normal mean under the tools/gkey.py G model of manifest gm, recomputed with its own
    functions. Checked first: the recomputed G values reproduce the gkey file's G rows (f_gk) from its source's (f_src) within 1e-3 s."""
    if gm["config"].get("--variant") == "avg_x":  # tools/gkeyx.py step (gkey avg + record columns): its own functions and reproduction check
        return load_tool("gkeyx").classes(setting, gm, f_src, f_gk)
    if gm["config"].get("--variant") in ("avg_xn", "avg_xnc", "avg_xncr"):  # tools/gkeyxn.py step (gkeyx + R3 columns [+ arrival copy shares]): its own check
        return load_tool("gkeyxn").classes(setting, gm, f_src, f_gk)
    gk = load_tool("gkey")
    from src.model3 import g_override, g_probs
    cfg = gm["config"]
    if sha16(f"cache/{cfg['--table']}.parquet") != cfg["--table_sha"]:
        raise SystemExit(f"cache/{cfg['--table']}.parquet sha16 differs from the gkey manifest's {cfg['--table_sha']}")
    T = pd.read_parquet(ROOT / "cache" / f"{cfg['--table']}.parquet")
    G = gk.g_frame(cfg["--leg_feats"])
    s = G["src"].to_numpy()
    Gf = G[gk.FIT[setting](s) & ~np.isnan(G["y"].to_numpy(dtype=float))]
    Ge = G[gk.EVAL[setting](s)]
    b = pd.read_parquet(ROOT / "cache" / f"{setting}_{cfg['--leg']}_s0.parquet", columns=["MVT_ID_mvt", "mu", "p_art"]).set_index("MVT_ID_mvt")
    bb = b.reindex(Ge["MVT_ID_mvt"].to_numpy())
    mu_g, pa_g = bb["mu"].to_numpy(dtype=float), bb["p_art"].to_numpy(dtype=float)
    d, dd = Ge["d_sch"].to_numpy(dtype=float), Ge["dd"].to_numpy()
    g_old = np.clip(g_override(g_probs(Gf, Ge), d, dd, mu_g, pa_g)[0], 0, None)
    Hf, He = gk.add_new(Gf, T), gk.add_new(Ge, T)
    extra = gk.NEW if cfg["--variant"] == "avg_new" else []
    Pa = gk.probs(Hf, He, extra)
    Hf["pfx"], He["pfx"] = Hf["k_car"], He["k_car"]
    Pc = gk.probs(Hf, He, extra)
    g_new, pn_new = g_override(0.5 * (Pa + Pc), d, dd, mu_g, pa_g)
    ids = Ge["MVT_ID_mvt"].to_numpy()
    want = np.clip(f_src.reindex(ids).to_numpy(dtype=float) + np.clip(g_new, 0, None) - g_old, 0, None)
    err = float(np.max(np.abs(want - f_gk.reindex(ids).to_numpy(dtype=float)))) if len(ids) else 0.0
    print(f"  {setting}: gkey G model recomputed for {len(ids)} G rows; its G values against the gkey file max diff {err:.2e} s", flush=True)
    if not err < 1e-3:
        raise SystemExit(f"{setting}: the recomputed gkey G model does not reproduce the gkey file (max diff {err})")
    return pd.DataFrame({"pn": pn_new, "mu_g": mu_g}, index=ids)


def inputs(ctx: str, ctx_sha: str, aiu_sha: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if sha16(f"cache/{ctx}.parquet") != ctx_sha:
        raise SystemExit(f"cache/{ctx}.parquet sha16 differs from {ctx_sha}")
    for k, rel in AIU.items():
        if sha16(rel) != aiu_sha[k]:
            raise SystemExit(f"{rel} sha16 differs from {aiu_sha[k]}")
    cx = pq.read_table(ROOT / "cache" / f"{ctx}.parquet", columns=["MVT_ID_mvt", "day"]).to_pandas().set_index("MVT_ID_mvt")
    cx["day"] = cx["day"].astype(str)
    t = pq.read_table(ROOT / "cache" / "feats_v25.parquet", columns=["MVT_ID_mvt", "src", "ap", "d_lobt", "dd"])
    keep = np.isin(t.column("src").to_numpy(), [0, 1, 7])
    ids = t.column("MVT_ID_mvt").to_numpy()[keep]
    fx = pd.DataFrame({"ap": t.column("ap").to_pandas().astype(str).to_numpy()[keep],
                       "dl": np.asarray(t.column("d_lobt").to_numpy(zero_copy_only=False), dtype=float)[keep],
                       "dd": np.asarray(t.column("dd").to_numpy(zero_copy_only=False), dtype=float)[keep]}, index=ids)
    del t
    tot = []
    for rel in AIU.values():
        a = pd.read_csv(ROOT / rel)
        a = a[a["APT_ICAO"].isin(APS)]
        tot.append(pd.Series(60.0 * a["DLY_ALL_PRE_2"].to_numpy(), index=(a["APT_ICAO"] + "|" + a["FLT_DATE"]).to_numpy()))
    return cx, fx, pd.concat(tot)


def ctx_t(ctx: str) -> pd.Series:
    """--group block: MVT time (UTC seconds) per row from the context file (its sha16 is checked by inputs())."""
    t = pq.read_table(ROOT / "cache" / f"{ctx}.parquet", columns=["MVT_ID_mvt", "t"])
    return pd.Series(np.asarray(t.column("t").to_numpy(zero_copy_only=False), dtype=float), index=t.column("MVT_ID_mvt").to_numpy())


def block_days(D: pd.DataFrame, t: pd.Series, by: str) -> np.ndarray:
    """--group block: UTC date of MVT - mn (by="mn", rows to shift) or of MVT - y (by="y", fitting rows for the offsets); dates that are not
    among the frame's MVT dates become "out" (no AIU total is matched to them)."""
    tt = t.reindex(D["id"].to_numpy()).to_numpy(dtype=float)
    lag = D["y"].to_numpy(dtype=float) if by == "y" else np.nan_to_num(D["mn"].to_numpy(dtype=float), nan=0.0)
    est = pd.to_datetime(tt - lag, unit="s").strftime("%Y-%m-%d").to_numpy()
    return np.where(pd.Series(est).isin(set(D["day"])).to_numpy(), est, "out")


def frame(setting: str, ch: dict, cx: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    c = pd.read_parquet(ROOT / "cache" / f"{setting}_{ch['corr']}_s0.parquet")
    b = pd.read_parquet(ROOT / "cache" / f"{setting}_{ch['base']}_s{ch['base_seed']}.parquet", columns=["MVT_ID_mvt", "pred"]).set_index("MVT_ID_mvt")["pred"]
    a = pd.read_parquet(ROOT / ch["final"][setting]).set_index("MVT_ID_mvt")["pred"].astype(float)
    ids = c["MVT_ID_mvt"].to_numpy()
    D = pd.DataFrame({"id": ids, "fc": c["pred"].to_numpy(), "f": a.reindex(ids).to_numpy(), "p": c["p_art"].to_numpy(), "mu": c["mu"].to_numpy(),
                      "g": c["g"].to_numpy().astype(bool), "d": c["dsch"].to_numpy(), "base": b.reindex(ids).to_numpy()})
    if "y" in c.columns:
        D["y"] = c["y"].to_numpy()
    D["ap"], D["day"], D["dl"] = fx["ap"].reindex(ids).to_numpy(), cx["day"].reindex(ids).to_numpy(), fx["dl"].reindex(ids).to_numpy()
    if ch["is_aw"]:  # the rows antiwin changed against its direct source
        pre = pd.read_parquet(ROOT / ch["pre_final"][setting]).set_index("MVT_ID_mvt")["pred"].astype(float)
        D["aw"] = np.abs(D["f"].to_numpy() - pre.reindex(ids).to_numpy()) > 1e-6
    else:  # the rows antiwin will change when it comes last
        D["aw"] = antiwin_mask(setting, a).reindex(ids).to_numpy()
    D["m"] = D["mu"] + (D["fc"] - D["base"])
    D = add_classes(D, fx["dd"])
    if ch["gkey"] is not None:
        gm = ch["gkey"]
        src_m = json.loads((ROOT / gm["source"]["manifest"]).read_text(encoding="utf-8"))
        src_f = pd.read_parquet(ROOT / src_m[SETTING_KEY[setting]]["path"]).set_index("MVT_ID_mvt")["pred"]
        gk_f = pd.read_parquet(ROOT / gm[SETTING_KEY[setting]]["path"]).set_index("MVT_ID_mvt")["pred"]
        K = gkey_classes(setting, gm, src_f, gk_f)
        gi = np.flatnonzero(D["g"].to_numpy())
        if set(D["id"].to_numpy()[gi]) != set(K.index):
            raise SystemExit(f"{setting}: the corrector's G rows and the gkey G rows differ")
        q, mn = D["q"].to_numpy().copy(), D["mn"].to_numpy().copy()
        q[gi] = 1 - K["pn"].reindex(D["id"].to_numpy()[gi]).to_numpy()
        mn[gi] = K["mu_g"].reindex(D["id"].to_numpy()[gi]).to_numpy()
        D["q"], D["mn"] = q, mn
    if D[["f", "base", "ap", "day"]].isna().any().any():
        raise SystemExit(f"{setting}: rows of the source files, base or inputs do not line up")
    return D


def add_classes(D: pd.DataFrame, dd: pd.Series) -> pd.DataFrame:
    """q = probability of the zero-contribution class (copy for non-G rows; copy or day push for G rows), mn = normal mean. G rows: model3's
    three G components recovered from the stored pred, p_art and mu (= mu_g)."""
    g = D["g"].to_numpy()
    p, d, mu, base = D["p"].to_numpy(), D["d"].to_numpy(), D["mu"].to_numpy(), D["base"].to_numpy()
    q, mn = p.copy(), D["m"].to_numpy().copy()
    S = 86400 * np.maximum(dd.reindex(D["id"].to_numpy()).to_numpy(), 1) + np.clip(mu, 300, 3000)
    with np.errstate(all="ignore"):
        pn = np.clip((p * d + (1 - p) * S - base) / (S - mu), 0, 1 - p)
    qg = p + np.clip(1 - p - pn, 0, 1)
    q[g], mn[g] = qg[g], mu[g]
    D["q"], D["mn"] = q, mn
    return D


def fit_g(D: pd.DataFrame):
    s = D[(np.abs(D["y"] - D["d"]) > 60) & (~D["g"]) & (D["y"] < 60000)]
    z = (s["d"] - s["mn"]).to_numpy()
    t = np.maximum(0.0, (s["d"] - s["y"]).to_numpy())
    qs = np.unique(np.quantile(z, np.linspace(0, 1, 101)))
    idx = np.clip(np.searchsorted(qs, z, side="right") - 1, 0, len(qs) - 2)
    cen, mean = [], []
    for i in range(len(qs) - 1):
        k = idx == i
        if k.sum() < 20:
            continue
        cen.append(z[k].mean())
        mean.append(t[k].mean())
    cen, mean = np.array(cen), np.maximum.accumulate(np.array(mean))

    def g(zz):
        zz = np.asarray(zz, float)
        r = np.interp(zz, cen, mean)
        hi = zz > cen[-1]
        r[hi] = mean[-1] + (zz[hi] - cen[-1])
        return np.maximum(r, 0.0)
    return g


def offsets(D: pd.DataFrame, tot: pd.Series) -> dict:
    """per airport: median over the fitting days of (AIU day total - our sum of max(0, d - y))."""
    our = pd.Series(np.maximum(0.0, D["d"].to_numpy() - D["y"].to_numpy())).groupby((D["ap"] + "|" + D["day"]).to_numpy()).sum()
    diff = tot.reindex(our.index) - our
    return {a: float(x.dropna().median()) for a, x in diff.groupby(diff.index.str[:4])}


def shift_days(D: pd.DataFrame, g, off: dict, tot: pd.Series, lam: float, cmax: float) -> tuple[np.ndarray, dict]:
    f_in, q = D["f"].to_numpy(), D["q"].to_numpy()
    f = f_in.copy()
    d, mn = D["d"].to_numpy(), D["mn"].to_numpy()
    shiftable = (f_in < 3600) & ~D["g"].to_numpy() & ~D["aw"].to_numpy()
    E = (1 - q) * g(d - mn)
    cs = {}
    for k, ii in pd.Series(np.arange(len(D))).groupby((D["ap"] + "|" + D["day"]).to_numpy()):
        if k[:4] not in APS:
            continue
        T = tot.get(k, np.nan)
        if not np.isfinite(T):
            continue
        T = T - off.get(k[:4], 0.0)
        ii = ii.to_numpy()
        sh = ii[shiftable[ii]]
        rest = E[ii[~shiftable[ii]]].sum()

        def tot_c(c):
            return np.sum((1 - q[sh]) * g(d[sh] - mn[sh] - c)) + rest
        lo, hi = -cmax, cmax
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
        cs[k] = c
        f[sh] = f[sh] + lam * (1 - q[sh]) * c
    dl = D["dl"].to_numpy()
    f = np.where(np.isfinite(dl), np.clip(f, dl - HALF, dl + HALF), f)
    return np.maximum(f, 0.0), cs


def shifted(setting: str, ch: dict, cx, fx, tot, lam: float, cmax: float, group: str = "mvt", t: pd.Series | None = None) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """(frame, new predictions aligned to it, day shifts). Fits never use the rows they move. group="block": see --group block."""
    def g_app(D):
        return D if group == "mvt" else D.assign(day=block_days(D, t, "mn"))

    def g_fit(D):
        return D if group == "mvt" else D.assign(day=block_days(D, t, "y"))
    V = frame("val", ch, cx, fx)
    jan = (V["day"].str[5:7] == "01").to_numpy()
    if setting == "val":
        f = V["f"].to_numpy().copy()
        cs = {}
        for msk, fit in ((jan, ~jan), (~jan, jan)):
            Dm = V[msk].reset_index(drop=True)
            fm, c = shift_days(g_app(Dm), fit_g(V[fit]), offsets(g_fit(V[fit]), tot), tot, lam, cmax)
            f[msk] = fm
            cs.update(c)
        return V, f, cs
    D = frame(setting, ch, cx, fx)
    fit = V[jan] if setting == "fwd" else V
    f, cs = shift_days(g_app(D), fit_g(fit), offsets(g_fit(fit), tot), tot, lam, cmax)
    return D, f, cs


def write(setting: str, ch: dict, D: pd.DataFrame, f: np.ndarray, tag: str) -> tuple[str, pd.Series, pd.Series]:
    old = pd.read_parquet(ROOT / ch["final"][setting])
    newp = pd.Series(f, index=D["id"].to_numpy()).reindex(old["MVT_ID_mvt"].to_numpy())
    if newp.isna().any():
        raise SystemExit(f"{setting}: {int(newp.isna().sum())} rows of the source file have no shifted value")
    new = old.copy()
    new["pred"] = newp.to_numpy()
    if setting == "val":
        from src.batches import hold
        y = hold()["TAXITIME_SEC_mvt"].astype(float)
        rm = float(np.sqrt(((new.set_index("MVT_ID_mvt")["pred"].reindex(y.index) - y) ** 2).mean()))
        rel = f"preds/holdout_{tag}_rmse{rm:.2f}.parquet"
    else:
        rel = f"preds/{'fwd' if setting == 'fwd' else 'test'}_{tag}.parquet"
    new.to_parquet(ROOT / rel, index=False)
    return rel, old.set_index("MVT_ID_mvt")["pred"].astype(float), new.set_index("MVT_ID_mvt")["pred"].astype(float)


def summary(old: pd.Series, new: pd.Series, D: pd.DataFrame, cs: dict) -> dict:
    mv = (new - old).to_numpy()
    c = pd.Series(cs, dtype=float)
    out = {"rows": int(len(mv)), "rows_moved": int((np.abs(mv) > 0.5).sum()), "move_mse": round(float(np.mean(mv ** 2)), 1),
           "c_by_airport": {a: [round(float(x.mean()), 1), round(float(x.std()), 1), round(float((x.abs() >= 59.9).mean()), 2)]
                            for a, x in c.groupby(c.index.str[:4])}}
    if "y" in D.columns:
        y = D.set_index("id")["y"].reindex(old.index).astype(float)
        ok = np.isfinite(y.to_numpy())
        se0, se1 = (old.to_numpy() - y.to_numpy())[ok] ** 2, (new.to_numpy() - y.to_numpy())[ok] ** 2
        out.update(rmse_old=round(float(np.sqrt(se0.mean())), 3), rmse_new=round(float(np.sqrt(se1.mean())), 3), dmse=round(float(se1.mean() - se0.mean()), 1))
    return out


def build(src_stem: str, num: str, lam: float = 0.5, cmax: float = 60.0, group: str = "mvt") -> int:
    from src.batches import manifest_of
    from src.ruler import ruler_hash
    meta = json.loads((ROOT / "queue" / f"{src_stem}.json").read_text(encoding="utf-8")) if (ROOT / "queue" / f"{src_stem}.json").exists() else \
        json.loads((ROOT / "queue" / "done" / f"{src_stem}.json").read_text(encoding="utf-8"))
    sm, why = manifest_of(meta)
    if why:
        print("거부: 원 후보 manifest 검사", why)
        return 1
    src_mp = meta["manifest"]
    tag = sm["spec"] + ("ds" if group == "mvt" else "db") + ("" if lam == 0.5 else f"l{round(lam * 1000)}")
    name = f"{num}_{tag}"
    if (ROOT / "queue" / f"{name}.json").exists() or (ROOT / f"preds/manifest_{tag}.json").exists():
        print("거부: 같은 이름이 이미 있다", name)
        return 1
    ch = chain(src_mp)
    cfg = {"--src": src_mp, "--src_sha": sha16(src_mp), "--lam": f"{lam:g}", "--cmax": f"{cmax:g}", "--ctx": "ctx_v3", "--ctx_sha": sha16("cache/ctx_v3.parquet"),
           "--aiu25_sha": sha16(AIU["--aiu25_sha"]), "--aiu26_sha": sha16(AIU["--aiu26_sha"]), "--tag": tag}
    if group != "mvt":
        cfg = {**{k: v for k, v in cfg.items() if k != "--tag"}, "--group": group, "--tag": tag}
    cx, fx, tot = inputs("ctx_v3", cfg["--ctx_sha"], cfg)
    t_mvt = ctx_t("ctx_v3") if group != "mvt" else None
    files, rep = {}, {}
    for setting in ("full", "val", "fwd"):
        D, f, cs = shifted(setting, ch, cx, fx, tot, lam, cmax, group, t_mvt) if group != "mvt" else shifted(setting, ch, cx, fx, tot, lam, cmax)
        rel, old, new = write(setting, ch, D, f, tag)
        files[setting], rep[setting] = rel, summary(old, new, D, cs)
        print(f"[{name}] {setting}: " + json.dumps({k: v for k, v in rep[setting].items() if k != "c_by_airport"}), flush=True)
        print(f"[{name}] {setting} day c by airport (mean, sd, share at the clip): {rep[setting]['c_by_airport']}", flush=True)
    es_rel = f"preds/es_valid_ids_{tag}.parquet"
    shutil.copy2(ROOT / sm["es_valid_ids"]["path"], ROOT / es_rel)
    me = "src/aiu_shift.py"
    h = sha16(me)
    snap = f"preds/code/{h}_aiu_shift.py"
    if not (ROOT / snap).exists():
        shutil.copy2(ROOT / me, ROOT / snap)
    flags = " ".join(f"{k} {v}" for k, v in cfg.items())
    cmd = lambda s: f".venv/Scripts/python.exe -X utf8 src/aiu_shift.py --setting {s} {flags}"  # noqa: E731
    hold_ids = pd.read_parquet(ROOT / files["val"])["MVT_ID_mvt"].to_numpy()
    man = {
        "spec": tag, "run_tag": tag, "column": "pred", "seeds": sm["seeds"], "feats": sm["feats"], "feats_sha": sm["feats_sha"], "config": cfg,
        "commands": {"holdout": [cmd("val")], "test": [cmd("full")], "forward": [cmd("fwd")]}, "command": cmd("val"),
        "code_sha": {**sm["code_sha"], me: h}, "code_snapshots": {**sm.get("code_snapshots", {}), me: snap},
        "holdout": {"path": files["val"], "sha": sha16(files["val"]), "rmse": round(rep["val"]["rmse_new"], 2), "ruler_hash": ruler_hash(hold_ids)},
        "test": {"path": files["full"], "sha": sha16(files["full"])},
        "forward": {"path": files["fwd"], "sha": sha16(files["fwd"]), "rmse_2025_07": round(rep["fwd"]["rmse_new"], 2)},
        "es_valid_ids": {**sm["es_valid_ids"], "path": es_rel, "sha": sha16(es_rel)},
        "iteration_rule": sm.get("iteration_rule", "") + " | aiu_shift: no model fitting; g and the airport offsets come from ruler months outside the moved rows",
        "source": {"queue": src_stem, "manifest": src_mp, "manifest_sha": cfg["--src_sha"], "spec": sm["spec"], "holdout_sha": sm["holdout"]["sha"],
                   "test_sha": sm["test"]["sha"], "forward_sha": sm["forward"]["sha"]},
        "aiu_shift": {"corrector": ch["corr"], "base": ch["base"],
                      "inputs_sha16": {**{f"cache/{s}_{ch['corr']}_s0.parquet": sha16(f"cache/{s}_{ch['corr']}_s0.parquet") for s in SETTING_KEY},
                                       **{f"cache/{s}_{ch['base']}_s{ch['base_seed']}.parquet": sha16(f"cache/{s}_{ch['base']}_s{ch['base_seed']}.parquet") for s in SETTING_KEY},
                                       "cache/feats_v25.parquet": sha16("cache/feats_v25.parquet")},
                      "report": rep},
        "written_kst": now_kst(),
    }
    if group != "mvt":
        man["aiu_shift"]["group"] = {"group": group, "rule": "airport-day groups by the UTC date of MVT - mn (rows to shift) and of MVT - y (fitting rows "
                                     "for the airport offsets); dates outside the frame's MVT dates are not matched",
                                     "evidence": "vault research/2026-10-07-r7-u.md 6.1 (pre-registered 00:04 KST 10-07): AIU totals match label totals grouped by "
                                                 "off-block UTC date (research/r7_u/e3_dayattr.py); against the MVT-date shift on the 456 stack after antiwin "
                                                 "val -45.9, fwd -82.5, nd -21.9 (day bootstrap 1.00 each), 12/12 months of the label-only match proxy",
                                     "caveats": ["the off-block date finding came from looking at 2025 AIU totals after the pre-registered tail variant "
                                                 "had failed; the rule for this option was written before its rulers were run"]}
    mp = f"preds/manifest_{tag}.json"
    (ROOT / mp).write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{name}] manifest {mp}", flush=True)
    where, aw_rows = ("맨 끝에 얹은 판", "반대 창이 바꾸지 않은 행") if ch["is_aw"] else ("얹은 판(반대 창은 이 뒤에 둔다)", "반대 창이 바꿀 행이 아닌 행")
    gq = " G 행의 q 는 사슬의 tools/gkey.py G 모형으로 다시 계산해(그 파일의 G 값과 맞는지 먼저 확인) 그날 합의 나머지 항에 쓴다." if ch["gkey"] is not None else ""
    method = (f"{src_stem} 위에 AIU 날 이동을 {where}(src/aiu_shift.py, R5-A 의 rake3 를 옮김). 공항-날마다 보통 행(예측 3,600초 미만, G 아님, {aw_rows})"
              "의 정상 몫을 한 값 c 만큼 옮겨, 예측이 뜻하는 출발 전 지연 합이 EUROCONTROL AIU 의 그날 합(60 x DLY_ALL_PRE_2)에서 공항 치우침을 뺀 값과 "
              f"같아지게 한다. c 는 ±{cmax:g}초로 자르고 {lam:g} x (1 − q) x c 를 더한 뒤 LOBT 창을 다시 건다. g(z) 와 공항 치우침은 옮기는 행 밖의 자 달에서 맞춘다"
              f"(자는 다른 달, 7월 전방은 1월, 시험은 두 달).{gq} "
              f"자 {rep['val']['rmse_old']} → {rep['val']['rmse_new']}(ΔMSE {rep['val']['dmse']}), 7월 전방 {rep['fwd']['rmse_old']} → {rep['fwd']['rmse_new']}"
              f"(ΔMSE {rep['fwd']['dmse']}), 시험 바뀐 행 {rep['full']['rows_moved']}, 이동 MSE {rep['full']['move_mse']}. 근거 vault research/2026-10-06-r5-drift.md 4. "
              "Source: EUROCONTROL(주최 허용, README 외부 데이터)")
    if group != "mvt":
        method += (" 공항-날 묶음은 MVT 의 UTC 날짜가 아니라 오프블록 어림(MVT − 정상 몫 mn)의 UTC 날짜로 한다(--group block). AIU 날 합은 출발을 오프블록 "
                   "UTC 날짜로 묶는다(2025 정답 합과의 일치, vault research/2026-10-07-r7-u.md 6.1). 공항 치우침은 맞춤 달 행을 정답 오프블록 날짜로 묶어 잰다. "
                   "약점: 이 날짜 정의는 사전 등록한 꼬리 판이 진 뒤 2025 AIU 합을 보다가 찾았고, 채택 규칙은 이 판의 자를 돌리기 전에 적었다.")
    (ROOT / "notes" / f"aiu_shift_{name}.json").write_text(json.dumps({"name": name, "manifest": mp, "source": src_stem, "method": method, "report": rep,
                                                                       "written_kst": now_kst()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    a = sys.argv[1:]
    if a and a[0] == "build":
        kw = {}
        if "--group" in a:
            kw["group"] = a[a.index("--group") + 1]
        if "--lam" in a:
            kw["lam"] = float(a[a.index("--lam") + 1])
        return build(a[1], a[2], **kw)
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--src", required=True)
    pa.add_argument("--src_sha", required=True)
    pa.add_argument("--lam", type=float, required=True)
    pa.add_argument("--cmax", type=float, required=True)
    pa.add_argument("--ctx", required=True)
    pa.add_argument("--ctx_sha", required=True)
    pa.add_argument("--aiu25_sha", required=True)
    pa.add_argument("--aiu26_sha", required=True)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--group", default="mvt", choices=["mvt", "block"])
    args = pa.parse_args()
    if sha16(args.src) != args.src_sha:
        raise SystemExit(f"{args.src} sha16 differs from {args.src_sha}")
    ch = chain(args.src)
    cx, fx, tot = inputs(args.ctx, args.ctx_sha, {"--aiu25_sha": args.aiu25_sha, "--aiu26_sha": args.aiu26_sha})
    if args.group != "mvt":
        D, f, cs = shifted(args.setting, ch, cx, fx, tot, args.lam, args.cmax, args.group, ctx_t(args.ctx))
    else:
        D, f, cs = shifted(args.setting, ch, cx, fx, tot, args.lam, args.cmax)
    rel, old, new = write(args.setting, ch, D, f, args.tag)
    print(rel, json.dumps({k: v for k, v in summary(old, new, D, cs).items() if k != "c_by_airport"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
