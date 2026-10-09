# SPDX-License-Identifier: GPL-3.0-only
"""Prototype: the route A corrector (src/adsb_corr.py, unchanged, imported) plus stand-geometry columns, measured on the ruler.

New columns (research/obs2/coverage.py -> obs_geo.parquet, ADS-B and PRC inputs only, no labels):
  d_stand_first  distance from the PRC stand's empirical position (median ADS-B stand stop of that STAND_mvt) to the first point
  st_known       that stand has an empirical position
  d_stand_stop   distance from the ADS-B stand stop to the PRC stand position (tow or remote stand when large)
  d_any_stand    distance from the first point to the nearest known stand of the airport (apron vs taxiway)
  seg_code       0 trace_start, 1 gap_cut, 2 landing (segment start reason)
  to_gap         ground -> air gap at take-off
Same folds, seed, rounds and parameters as adsb_corr.py val; the LOBT window last. Prints ruler RMSE for the original columns and
for original + new, and writes research/obs2/corr_plus_oof.parquet (per ruler row: both corrected predictions, y, flags).
Usage: .venv/Scripts/python.exe research/obs2/corr_plus.py --base m6r25sgc75
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import src.adsb_corr as ac  # noqa: E402
from src.lobt_window import feats_d_lobt, mu_of, project  # noqa: E402

NEW = ["d_stand_first", "st_known", "d_stand_stop", "d_any_stand", "seg_code", "to_gap"]


GEO_READ = ["MVT_ID_mvt", "day", "ADEP_mvt", "stand_id", "matched", "fm_ok", "ev_first_lat", "ev_first_lon", "ev_stand_lat",
            "ev_stand_lon", "ev_seg_reason", "ev_to_gap", "st_lat", "st_lon", "d_stand_first"]


def geo_cols(which: str = "adsb") -> pd.DataFrame:
    g = pd.read_parquet(HERE / "obs_geo.parquet", columns=GEO_READ)
    if which == "filled":  # empirical ADS-B stand where known, else OSM parking_position (osm_stands.py)
        sm = pd.read_parquet(HERE / "stand_map_filled.parquet")
        g = g.drop(columns=["st_lat", "st_lon"]).join(
            sm.set_index(["ADEP_mvt", "stand_id"])[["lat", "lon"]].rename(columns={"lat": "st_lat", "lon": "st_lon"}),
            on=["ADEP_mvt", "stand_id"])
        k0 = np.cos(np.radians(g.st_lat.fillna(0)))
        g["d_stand_first"] = np.hypot((g.ev_first_lon - g.st_lon) * 111200 * k0, (g.ev_first_lat - g.st_lat) * 111200)
    else:
        sm = pd.read_parquet(HERE / "stand_map.parquet")
        sm = sm[sm.ok]
    k = np.cos(np.radians(g.ev_first_lat.fillna(0)))
    g["d_any_stand"] = np.nan
    for ap, s in sm.groupby("ADEP_mvt"):
        la0 = s.lat.mean()
        c = np.cos(np.radians(la0))
        tree = cKDTree(np.c_[s.lon * 111200 * c, s.lat * 111200])
        sel = (g.ADEP_mvt == ap) & g.ev_first_lat.notna()
        d, _ = tree.query(np.c_[g.loc[sel, "ev_first_lon"] * 111200 * c, g.loc[sel, "ev_first_lat"] * 111200])
        g.loc[sel, "d_any_stand"] = d
    g["st_known"] = g.st_lat.notna().astype(float)
    kk = np.cos(np.radians(g.st_lat.fillna(0)))
    g["d_stand_stop"] = np.hypot((g.ev_stand_lon - g.st_lon) * 111200 * kk, (g.ev_stand_lat - g.st_lat) * 111200)
    g["seg_code"] = g.ev_seg_reason.map({"trace_start": 0.0, "gap_cut": 1.0, "landing": 2.0})
    g["to_gap"] = g.ev_to_gap
    del k
    return g.set_index("MVT_ID_mvt")[NEW + ["fm_ok", "matched", "ADEP_mvt", "day"]].rename(
        columns={"fm_ok": "g_fm_ok", "matched": "g_matched", "ADEP_mvt": "g_ap", "day": "g_day"})


def run(val, A, cols, seed, folds, rounds, threads):
    ac.ADSB_COLS = cols
    T = ac.design(val, A)
    T = T[np.isfinite(T["y"].to_numpy())]
    models, f = ac.fold_models(T, folds, seed, rounds, threads)
    oof = np.zeros(len(T))
    for j, mdl in enumerate(models):
        oof[f == j] = ac.predict(mdl, T[f == j])
    delta = np.zeros(len(val))
    delta[T["pos"].to_numpy()] = oof
    imp = pd.Series(np.mean([m.feature_importance("gain") for m in models], axis=0), index=ac.BASE_COLS + cols)
    return delta, imp / imp.sum()


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--base", default="m6r25sgc75")
    pa.add_argument("--feats", default="feats_v25")
    pa.add_argument("--adsb_sha", default="2bb0952d93192201")
    pa.add_argument("--threads", type=int, default=3)
    pa.add_argument("--map", default="adsb", choices=["adsb", "filled"])
    pa.add_argument("--only_plus", action="store_true")
    a = pa.parse_args()
    ac.below_normal()
    orig_cols = list(ac.ADSB_COLS)
    A = ac.load_adsb("adsb_v1", a.adsb_sha)
    G = geo_cols(a.map)
    A = A.join(G[NEW], how="left")
    val = pd.read_parquet(ROOT / "cache" / f"val_{a.base}_s0.parquet")
    y = val["y"].to_numpy(dtype=float)
    which, mu = mu_of(val)
    p_art, g, dsch = val["p_art"].to_numpy(dtype=float), val["g"].to_numpy().astype(bool), val["dsch"].to_numpy(dtype=float)
    d = feats_d_lobt(a.feats, val["MVT_ID_mvt"].to_numpy())
    base = val["pred"].to_numpy(dtype=float)
    out = pd.DataFrame({"MVT_ID_mvt": val["MVT_ID_mvt"].to_numpy(), "y": y})
    out["pred_w"], _, _ = project(base, mu, p_art, g, dsch, d)
    for name, cols in ((("plus", orig_cols + NEW),) if a.only_plus else (("orig", orig_cols), ("plus", orig_cols + NEW))):
        delta, imp = run(val, A, cols, 0, 10, 300, a.threads)
        pw, _, _ = project(np.clip(base + delta, 0, None), mu, p_art, g, dsch, d)
        out[f"pred_{name}"] = pw
        print(f"{name}: ruler RMSE {np.sqrt(np.mean((pw - y) ** 2)):.3f}")
        if name == "plus":
            print("importance of new columns:", imp[NEW].round(4).to_dict())
    out = out.join(G[["g_fm_ok", "g_matched", "g_ap", "g_day", "d_stand_first", "st_known"]], on="MVT_ID_mvt")
    out.to_parquet(HERE / f"corr_plus_oof_{a.base}{'' if a.map == 'adsb' else '_' + a.map}.parquet", index=False)
    print(f"base+window {np.sqrt(np.mean((out.pred_w - y) ** 2)):.3f}")


if __name__ == "__main__":
    main()
