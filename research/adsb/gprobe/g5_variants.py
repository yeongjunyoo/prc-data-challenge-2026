# SPDX-License-Identifier: GPL-3.0-only
"""Overnight probe (10-06): does the geometry corrector (stage 1 of 316/331/352) transfer across ADS-B coverage regimes, and does a
variant without the where-heard inputs transfer better? GREKI's check (validate January <-> July) on our 2025 ruler days, where the
receiver coverage also moves between the months (EGLL 0.85 -> 0.39, EDDM 0.98 -> 0.38, LEMD, LFPG, LIRF 0 -> 0.3 ~ 0.94).

Corrector replicas: src/adsb_corr.fit's parameters and target on the oof4 base (val_m6r25sgoof4, the base of 316/331/352), matched rows
of adsb_v1 with the six geometry columns. Variants (inputs only, no label used to make them):
  V0 all inputs (the deployed stage 1)
  V1 drop the location-of-first-heard inputs LOC (where the first point was, how dense and gapped the reception was)
  V2 V1 and also drop the first-heard timing (est_first, first_minus_pred, first_minus_sched)
  V3 LOC as quantile ranks within (airport, month), est_first kept raw
Measures: ruler OOF (10 day folds, the deployed partition) on matched rows, January -> July and July -> January (fit on one month,
apply to the other), and the 2026 corrections of a model on all 2025 rows (mean per airport and month, flags as g4_shift.py).

  .venv/Scripts/python.exe -X utf8 research/adsb/gprobe/g5_variants.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
os.chdir(ROOT)
import src.adsb_corr as ac  # noqa: E402

N = 344419
GEO = ["d_stand_first", "st_known", "d_stand_stop", "d_any_stand", "seg_code", "to_gap"]
LOC = ["d_stand_first", "d_any_stand", "ev_first_dist_m", "ev_first_gs", "ev_first_stationary", "seg_code", "to_gap", "ev_to_gap", "ev_n_pts",
       "ev_max_gap_after_fm", "ev_share_mlat"]
FIRST_T = ["est_first", "first_minus_pred", "first_minus_sched"]
DENS = ["ev_n_pts", "ev_to_gap", "to_gap", "ev_max_gap_after_fm", "ev_share_mlat"]
DIST = ["d_stand_first", "d_any_stand", "ev_first_dist_m"]
ALL = ac.BASE_COLS + ac.ADSB_COLS + GEO


def qrank(X, cols):
    X = X.copy()
    key = X["ADEP_mvt"].astype(str) + "|" + X["day"].str[:7]
    for c in cols:
        X[c] = X.groupby(key)[c].rank(pct=True)
    return X


def fit(X, cols, threads=4, seed=0, rounds=300):
    import lightgbm as lgb
    params = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.9, bagging_fraction=0.9,
                  bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=threads, seed=seed, deterministic=True, force_col_wise=True)
    tgt = np.clip(X["y"].to_numpy() - X["pred"].to_numpy(), -ac.CLIP, ac.CLIP)
    return lgb.train(params, lgb.Dataset(X[cols].astype(float).to_numpy(), tgt), num_boost_round=rounds)


def pr(m, X, cols):
    return m.predict(X[cols].astype(float).to_numpy())


def dmse(X, corr):
    y, p = X["y"].to_numpy(), X["pred"].to_numpy()
    return float((((np.clip(p + corr, 0, None) - y) ** 2).sum() - ((p - y) ** 2).sum()) / N)


def main():
    A = ac.load_adsb("adsb_v1", "2bb0952d93192201").join(pd.read_parquet(ROOT / "cache" / "adsb_geo_v1.parquet").set_index("MVT_ID_mvt")[GEO], how="left")
    val = pd.read_parquet(ROOT / "cache" / "val_m6r25sgoof4_s0.parquet")
    full = pd.read_parquet(ROOT / "cache" / "full_m6r25sgoof4_s0.parquet")
    Xr0, Xt0 = ac.design(val, A), ac.design(full, A)
    Xr0 = Xr0[np.isfinite(Xr0["y"].to_numpy())].reset_index(drop=True)
    variants = {"V0 all": (ALL, False), "V1 drop LOC": ([c for c in ALL if c not in LOC], False),
                "V2 drop LOC + first timing": ([c for c in ALL if c not in LOC + FIRST_T], False), "V3 LOC as ranks in airport-month": (ALL, True),
                "V5 drop density and gaps": ([c for c in ALL if c not in DENS], False), "V6 drop first-point distances": ([c for c in ALL if c not in DIST], False)}
    sel = sys.argv[1:]
    variants = {k: v for k, v in variants.items() if not sel or k.split()[0] in sel}
    tag = "_".join(sel) if sel else "all"
    fo = ac.day_folds(Xr0["day"].unique(), 10, 0)
    f = Xr0["day"].map(fo).to_numpy()
    jan = (Xr0["day"].str[5:7] == "01").to_numpy()
    rows, corr_rows = [], []
    for nm, (cols, rank) in variants.items():
        Xr = qrank(Xr0, LOC) if rank else Xr0
        Xt = qrank(Xt0, LOC) if rank else Xt0
        oof = np.zeros(len(Xr))
        for j in range(10):
            oof[f == j] = pr(fit(Xr[f != j], cols), Xr[f == j], cols)
        cj = pr(fit(Xr[~jan], cols), Xr[jan], cols)      # July -> January
        cjl = pr(fit(Xr[jan], cols), Xr[~jan], cols)     # January -> July
        d_oof = dmse(Xr, oof)
        d_oof_j, d_oof_l = dmse(Xr[jan], oof[jan]), dmse(Xr[~jan], oof[~jan])
        d_jl, d_lj = dmse(Xr[~jan], cjl), dmse(Xr[jan], cj)
        mall = fit(Xr, cols)
        ct = pr(mall, Xt, cols)
        T = pd.DataFrame({"ap": Xt["ADEP_mvt"].astype(str).to_numpy(), "mon": Xt["day"].str[:7].to_numpy(), "c": ct})
        R = pd.DataFrame({"ap": Xr["ADEP_mvt"].astype(str).to_numpy(), "mon": Xr["day"].str[:7].to_numpy(), "c": oof})
        g = pd.concat([R, T]).groupby(["ap", "mon"])["c"].agg(["mean", "std", "size"])
        g["variant"] = nm
        corr_rows.append(g.reset_index())
        flags = 0
        for ap in g.index.get_level_values(0).unique():
            gg = g.loc[ap]
            m25 = [gg.at[c, "mean"] for c in ("2025-01", "2025-07") if c in gg.index and gg.at[c, "size"] >= 200]
            for c in ("2026-01", "2026-07"):
                if c in gg.index and gg.at[c, "size"] >= 200 and m25 and (gg.at[c, "mean"] < min(m25) - 5 or gg.at[c, "mean"] > max(m25) + 5):
                    flags += 1
        egll = g.loc[("EGLL", "2026-07"), "mean"] if ("EGLL", "2026-07") in g.index else np.nan
        rows.append(dict(variant=nm, n_cols=len(cols), ruler_oof=round(d_oof, 1), oof_jan=round(d_oof_j, 1), oof_jul=round(d_oof_l, 1),
                         jan_to_jul=round(d_jl, 1), jul_to_jan=round(d_lj, 1), keep_jul=round(d_jl / d_oof_l, 3) if d_oof_l else np.nan,
                         keep_jan=round(d_lj / d_oof_j, 3) if d_oof_j else np.nan, flags_2026=flags, egll_2026_07=round(float(egll), 1),
                         sd_2026=round(float(T.c.std()), 1), sd_2025=round(float(R.c.std()), 1)))
        print(rows[-1])
    print("\nvariants (ΔMSE over the ruler N on matched rows, negative = better; keep = cross-month gain / in-month OOF gain on the same rows):")
    print(pd.DataFrame(rows).to_string(index=False))
    C = pd.concat(corr_rows, ignore_index=True)
    C.to_csv(OUT / f"g5_variant_corr_{tag}.csv", index=False)
    W = C.pivot_table(index=["ap", "mon"], columns="variant", values="mean").round(1)
    print("\nmean correction per airport and month (2025 = OOF, 2026 = one model on all 2025):")
    print(W.to_string())
    pd.DataFrame(rows).to_csv(OUT / f"g5_variants_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
