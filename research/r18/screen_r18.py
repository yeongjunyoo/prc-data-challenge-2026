# SPDX-License-Identifier: GPL-3.0-only
"""R18 screen of a candidate TabM-type leg against the chain-A TabM leg of 514/531 (measurement only, writes its report under research/r18).

Rulers
  val, fwd  the chain-A base blend of 514 without TabM (cache/<st>_m6r27wincbrmfw4sv434_s0.parquet, blend_fixed of the seed-averaged
            LightGBM/CatBoost/XGBoost/RealMLP legs) plus ONE TabM-type leg with research/r13_l/blend_add.py's rule (one weight; val = 10 day
            folds over the ruler days, seed 0; fwd = weight fitted on the January ruler days of the val files). REF = the deployed leg
            (cache/<st>_m6r27tmcwin_s0.parquet, or a re-run given with --ref); each CAND replaces it. dMSE = mean over all rows of
            (pred_cand - y)^2 - (pred_ref - y)^2; body_nonG = y <= 3600 and not G (LIRF without NM flight match), as research/r13_l/screen.py.
  fj        F2's M3 ruler (February and June 2025 unseen, legs fitted on the other ten months). No mixture exists on the PC for these rows,
            so the leg is scored on the regression rows it is meant for: not art (|y - d_sch| <= 60), not shift (y > 60000), not G; "body" adds
            y <= 3600. dMSE of mu_cand against mu_ref on those rows (the same leg-level numbers are printed for val and fwd too).
Noise: UTC-day block bootstrap (take-off day), 2,000 draws, seed 0; P = share of draws with dMSE < 0.
    .venv/Scripts/python.exe -X utf8 research/r18/screen_r18.py --name <report> --st val,fwd,fj --ref <file pattern> --cand <file pattern> [...]
A file pattern is "<folder>/<setting>_<tag>_s<seed>.parquet" with {st} for the setting, e.g. research/r18/cache/{st}_r18tmc_s0.parquet;
"deployed" means cache/{st}_m6r27tmcwin_s0.parquet. Several --cand files given as a+b are averaged (seed average).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.blend_oof as bo  # noqa: E402
from src.adsb_corr import day_folds  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if sys.platform == "win32":
    import ctypes
    ctypes.WinDLL("kernel32").SetPriorityClass(ctypes.WinDLL("kernel32").GetCurrentProcess(), 0x00004000)
C = ROOT / "cache"
SRC = "m6r27wincbrmfw4sv434"
DEPLOYED = "cache/{st}_m6r27tmcwin_s0.parquet"


def boot(dd: np.ndarray, day: np.ndarray, mask: np.ndarray | None = None) -> dict:
    x = dd if mask is None else np.where(mask, dd, 0.0)
    g = pd.DataFrame({"d": x, "day": day}).groupby("day")["d"].sum()
    n = pd.Series(1.0, index=np.arange(len(day))).groupby(day).sum().reindex(g.index).to_numpy()
    rng = np.random.default_rng(0)
    dr = rng.integers(0, len(g), size=(2000, len(g)))
    bt = g.to_numpy()[dr].sum(axis=1) / n[dr].sum(axis=1)
    return {"dmse": round(float(x.sum() / len(x)), 2), "sd": round(float(bt.std(ddof=1)), 2), "p_below0": round(float((bt < 0).mean()), 4)}


def leg(pattern: str, st: str, ids: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = [(ROOT / (DEPLOYED if p == "deployed" else p).format(st=st)) for p in pattern.split("+")]
    mus, y, fid = [], None, None
    for f in files:
        d = pd.read_parquet(f, columns=None)
        col = "mu_tabm" if "mu_tabm" in d.columns else "mu"
        if fid is None:
            fid = d["MVT_ID_mvt"].to_numpy()
            y = d["y"].to_numpy(dtype=float) if "y" in d.columns else None
        assert np.array_equal(d["MVT_ID_mvt"].to_numpy(), fid), f"row order differs: {f}"
        mus.append(d[col].to_numpy(dtype=float))
    if ids is not None:
        assert np.array_equal(fid, ids), f"rows differ from the base: {pattern}"
    return fid, np.mean(mus, axis=0), y


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("--name", required=True)
    pa.add_argument("--st", default="val,fwd")
    pa.add_argument("--ref", default="deployed")
    pa.add_argument("--cand", action="append", required=True)
    pa.add_argument("--both", action="store_true", help="exploratory: also the blend with both the ref and the cand leg")
    a = pa.parse_args()
    t = pq.read_table(C / "all.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "t_mvt"], filters=[("PHASE_mvt", "==", "DEP")],
                      read_dictionary=["ADEP_mvt"])
    A = pd.DataFrame({"ap": t.column("ADEP_mvt").to_pandas().astype(str).to_numpy(), "nm": t.column("FLIGHT_ID_mvt").is_valid().to_numpy(zero_copy_only=False),
                      "day": (t.column("t_mvt").to_numpy() // 86400).astype(np.int64)}, index=t.column("MVT_ID_mvt").to_numpy())
    del t
    rep = {"name": a.name, "ref": a.ref, "cand": a.cand, "src_blend": SRC}
    W = {}
    for st in a.st.split(","):
        r = {}
        if st in ("val", "fwd"):
            b = pd.read_parquet(C / f"{st}_{SRC}_s0.parquet")
            ids = b["MVT_ID_mvt"].to_numpy()
            y = b["y"].to_numpy(dtype=float)
            g = b["g"].to_numpy().astype(bool)
            dsch = b["dsch"].to_numpy(dtype=float)
        else:
            ids, y, g, dsch, b = None, None, None, None, None
        legs = {"ref": a.ref}
        legs.update({f"cand{i}": c for i, c in enumerate(a.cand)})
        mus = {}
        for k, p in legs.items():
            fid, mu, yy = leg(p, st, ids)
            if ids is None:
                ids, y = fid, yy
            mus[k] = mu
        M = A.reindex(ids)
        G = (M["ap"].to_numpy() == "LIRF") & ~M["nm"].to_numpy()
        day = M["day"].to_numpy()
        if dsch is None:
            F = pd.read_parquet(C / "feats_v27.parquet", columns=["MVT_ID_mvt", "d_sch"]).set_index("MVT_ID_mvt")["d_sch"]
            dsch = F.reindex(ids).to_numpy(dtype=float)
        art = np.abs(y - dsch) <= 60
        shift = (y > 60000) & ~art
        reg = ~art & ~shift & ~G
        regb = reg & (y <= 3600)
        r["n"], r["n_reg"], r["n_reg_body"] = int(len(y)), int(reg.sum()), int(regb.sum())
        r["leg_rmse_reg"] = {k: round(float(np.sqrt(np.mean((mu[reg] - y[reg]) ** 2))), 3) for k, mu in mus.items()}
        r["leg_rmse_reg_body"] = {k: round(float(np.sqrt(np.mean((mu[regb] - y[regb]) ** 2))), 3) for k, mu in mus.items()}
        e_ref = (mus["ref"] - y) ** 2
        for k in mus:
            if k == "ref":
                continue
            dd = np.where(reg, (mus[k] - y) ** 2 - e_ref, 0.0)
            apv = M["ap"].to_numpy()
            mon_l = pd.to_datetime(day * 86400, unit="s").month
            r[f"leg_{k}"] = {"reg": boot(dd, day, reg), "reg_body": boot(dd, day, regb),
                             "corr_err_reg_body_with_ref": round(float(np.corrcoef((mus[k] - y)[regb], (mus["ref"] - y)[regb])[0, 1]), 4),
                             "reg_by_airport": {ap_: round(float(dd[apv == ap_].sum() / len(y)), 2) for ap_ in sorted(set(apv))},
                             "reg_by_month": {int(m_): round(float(dd[mon_l == m_].sum() / len(y)), 2) for m_ in np.unique(mon_l)},
                             "reg_tail_y_gt_3600": round(float(dd[reg & (y > 3600)].sum() / len(y)), 2),
                             "mean_mu_minus_ref_reg": round(float((mus[k] - mus["ref"])[reg].mean()), 3)}
        if b is not None:
            body = (y <= 3600) & ~G
            dv = bo.ruler_days(ids) if st == "val" else None
            preds = {}
            for k, mu in mus.items():
                MU = np.column_stack([b["mu"].to_numpy(dtype=float), mu])
                Ad, off = bo.design(b, MU)
                if st == "val":
                    fo = day_folds(np.unique(dv), 10, 0)
                    fdx = pd.Series(dv).map(fo).to_numpy()
                    fit_ok = ~g & np.isfinite(y)
                    Wm = np.zeros((len(y), 2))
                    fw = []
                    for j in range(10):
                        w = bo.simplex_ls(Ad[fit_ok & (fdx != j)], (y - off)[fit_ok & (fdx != j)])
                        Wm[fdx == j] = w
                        fw.append(round(float(w[1]), 4))
                    jan = fit_ok & (pd.Series(dv).str[5:7] == "01").to_numpy()
                    W[k] = bo.simplex_ls(Ad[jan], (y - off)[jan])
                    r[f"w_{k}"] = {"val_folds": fw, "january": round(float(W[k][1]), 4)}
                else:
                    Wm = np.tile(W[k], (len(y), 1))
                pr, _ = bo.mix(b, MU, Wm)
                preds[k] = pr
                r[f"blend_{k}"] = {"rmse": round(float(np.sqrt(np.mean((pr - y) ** 2))), 3), "body_rmse": round(float(np.sqrt(np.mean((pr[body] - y[body]) ** 2))), 3)}
            r["src_rmse"] = round(float(np.sqrt(np.mean((b["pred"].to_numpy(dtype=float) - y) ** 2))), 3)
            mon = pd.to_datetime(day * 86400, unit="s").month
            for k in preds:
                if k == "ref":
                    continue
                dd = (preds[k] - y) ** 2 - (preds["ref"] - y) ** 2
                r[f"blend_d_{k}"] = {"all": boot(dd, day), "body_nonG": boot(dd, day, body)}
                r[f"blend_d_{k}"]["by_month"] = {int(m_): round(float(dd[mon == m_].sum() / len(dd)), 2) for m_ in np.unique(mon)}
                if a.both:  # exploratory: base blend + ref leg + cand leg, three simplex weights with the same protocol
                    MU3 = np.column_stack([b["mu"].to_numpy(dtype=float), mus["ref"], mus[k]])
                    Ad3, off3 = bo.design(b, MU3)
                    if st == "val":
                        Wm3 = np.zeros((len(y), 3))
                        fw3 = []
                        for j in range(10):
                            w = bo.simplex_ls(Ad3[fit_ok & (fdx != j)], (y - off3)[fit_ok & (fdx != j)])
                            Wm3[fdx == j] = w
                            fw3.append([round(float(x), 4) for x in w])
                        W[f"both_{k}"] = bo.simplex_ls(Ad3[jan], (y - off3)[jan])
                        r[f"w_both_{k}"] = {"val_folds": fw3, "january": [round(float(x), 4) for x in W[f"both_{k}"]]}
                    else:
                        Wm3 = np.tile(W[f"both_{k}"], (len(y), 1))
                    p3, _ = bo.mix(b, MU3, Wm3)
                    d3 = (p3 - y) ** 2 - (preds["ref"] - y) ** 2
                    r[f"both_d_{k}"] = {"all": boot(d3, day), "body_nonG": boot(d3, day, body),
                                        "by_month": {int(m_): round(float(d3[mon == m_].sum() / len(d3)), 2) for m_ in np.unique(mon)}}
        rep[st] = r
        print(f"[{st}] {json.dumps(r, ensure_ascii=False)}", flush=True)
    out = ROOT / "research" / "r18" / "screens" / f"{a.name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
