# SPDX-License-Identifier: GPL-3.0-only
"""R18-M1-B (registered 2026-10-09 15:03 KST in the private note 2026-10-09-r18-latest-hits.md section 8, before any fj blend number):
the second fj test of M1, at the blend stage. For each TabM-type leg (ref = MSE r18tmc, cand = HL-Gauss r18tmhl, seed 0):
  w  = research/r13_l/blend_add.py's full rule: src/blend_oof.simplex_ls of [mu_base, mu_leg] on every val ruler day (non-G rows with a label) of
       cache/val_m6r27wincbrmfw4sv434_s0.parquet and the leg's val file;
  fj = src/blend_oof.mix(F, [mu_F, mu_leg_fj], w) on F2's chain-A fj blend F (fj_f2wincbrmfw4_s0, fetched read-only from the Mac), G rows keep F.
dMSE = mean over all fj rows of (pred_cand - y)^2 - (pred_ref - y)^2; UTC take-off day block bootstrap, 2,000 draws, seed 0, P(dMSE < 0).
    .venv/Scripts/python.exe -X utf8 research/r18/m1b_fj.py --fj_base research/r18/mac_fj/fj_f2wincbrmfw4_s0.parquet [--name m1b]
    (--smoke: use the val base as the fj base, to test the code path only)
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

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if sys.platform == "win32":
    import ctypes
    ctypes.WinDLL("kernel32").SetPriorityClass(ctypes.WinDLL("kernel32").GetCurrentProcess(), 0x00004000)
C, R18 = ROOT / "cache", ROOT / "research" / "r18"
VBASE = "m6r27wincbrmfw4sv434"
LEGS = {"ref": "r18tmc", "cand": "r18tmhl"}


def boot(dd: np.ndarray, day: np.ndarray, mask: np.ndarray | None = None) -> dict:
    x = dd if mask is None else np.where(mask, dd, 0.0)
    g = pd.DataFrame({"d": x, "day": day}).groupby("day")["d"].sum()
    n = pd.Series(1.0, index=np.arange(len(day))).groupby(day).sum().reindex(g.index).to_numpy()
    rng = np.random.default_rng(0)
    dr = rng.integers(0, len(g), size=(2000, len(g)))
    bt = g.to_numpy()[dr].sum(axis=1) / n[dr].sum(axis=1)
    return {"dmse": round(float(x.sum() / len(x)), 2), "sd": round(float(bt.std(ddof=1)), 2), "p_below0": round(float((bt < 0).mean()), 4)}


def leg(setting: str, tag: str, ids: np.ndarray) -> np.ndarray:
    d = pd.read_parquet(R18 / "cache" / f"{setting}_{tag}_s0.parquet", columns=["MVT_ID_mvt", "mu_tabm"]).set_index("MVT_ID_mvt")["mu_tabm"]
    v = d.reindex(ids).to_numpy(dtype=float)
    assert np.isfinite(v).all(), f"{setting}_{tag}: rows missing for the base"
    return v


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("--fj_base", default=str(R18 / "mac_fj" / "fj_f2wincbrmfw4_s0.parquet"))
    pa.add_argument("--name", default="m1b")
    pa.add_argument("--smoke", action="store_true")
    a = pa.parse_args()
    V = pd.read_parquet(C / f"val_{VBASE}_s0.parquet")
    F = V.copy() if a.smoke else pd.read_parquet(a.fj_base)
    fset = "val" if a.smoke else "fj"
    yv = V["y"].to_numpy(dtype=float)
    fit_ok = ~V["g"].to_numpy().astype(bool) & np.isfinite(yv)
    idsV, idsF = V["MVT_ID_mvt"].to_numpy(), F["MVT_ID_mvt"].to_numpy()
    yf = F["y"].to_numpy(dtype=float)
    t = pq.read_table(C / "all.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "t_mvt"], filters=[("PHASE_mvt", "==", "DEP")],
                      read_dictionary=["ADEP_mvt"])
    A = pd.DataFrame({"ap": t.column("ADEP_mvt").to_pandas().astype(str).to_numpy(), "nm": t.column("FLIGHT_ID_mvt").is_valid().to_numpy(zero_copy_only=False),
                      "day": (t.column("t_mvt").to_numpy() // 86400).astype(np.int64)}, index=t.column("MVT_ID_mvt").to_numpy()).reindex(idsF)
    day, apv = A["day"].to_numpy(), A["ap"].to_numpy()
    G = (apv == "LIRF") & ~A["nm"].to_numpy()
    body = (yf <= 3600) & ~G
    mon = pd.to_datetime(day * 86400, unit="s").month
    rep = {"name": a.name, "fj_base": a.fj_base, "val_base": VBASE, "legs": LEGS, "n_fj": int(len(F)), "smoke": a.smoke}
    preds = {}
    for k, tag in LEGS.items():
        MUv = np.column_stack([V["mu"].to_numpy(dtype=float), leg("val", tag, idsV)])
        Av, offv = bo.design(V, MUv)
        w = bo.simplex_ls(Av[fit_ok], (yv - offv)[fit_ok])
        MUf = np.column_stack([F["mu"].to_numpy(dtype=float), leg(fset, tag, idsF)])
        pf, _ = bo.mix(F, MUf, np.tile(w, (len(F), 1)))
        preds[k] = pf
        rep[f"w_{k}"] = round(float(w[1]), 4)
        rep[f"rmse_{k}"] = round(float(np.sqrt(np.mean((pf - yf) ** 2))), 3)
        rep[f"body_rmse_{k}"] = round(float(np.sqrt(np.mean((pf[body] - yf[body]) ** 2))), 3)
    rep["rmse_base"] = round(float(np.sqrt(np.mean((F["pred"].to_numpy(dtype=float) - yf) ** 2))), 3)
    dd = (preds["cand"] - yf) ** 2 - (preds["ref"] - yf) ** 2
    rep["d_all"] = boot(dd, day)
    rep["d_body_nonG"] = boot(dd, day, body)
    rep["d_by_month"] = {int(m_): round(float(dd[mon == m_].sum() / len(dd)), 2) for m_ in np.unique(mon)}
    rep["d_by_airport"] = {ap_: round(float(dd[apv == ap_].sum() / len(dd)), 2) for ap_ in sorted(set(apv))}
    rep["d_tail_y_gt_3600_nonG"] = round(float(dd[(yf > 3600) & ~G].sum() / len(dd)), 2)
    rep["pass"] = bool(rep["d_all"]["dmse"] < 0 and rep["d_all"]["p_below0"] >= 0.9)
    print(json.dumps(rep, ensure_ascii=False))
    out = R18 / "screens" / f"{a.name}{'_smoke' if a.smoke else ''}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
