# SPDX-License-Identifier: GPL-3.0-only
"""Overnight probe (10-05/06): 2025 -> 2026 shift of the ADS-B corrector inputs and of the corrections they make.

1. Shift table: every input of the x1.0 corrector (src/adsb_corr.py, 291 line) and of the stand-geometry corrector (src/adsb_geo.py, 321
   line; the same stage 1 inside src/ctx_corr.py for 316/331/352), matched rows, per airport: the corrector's 2025 training rows (ruler
   days 2025-01 and 07, pooled) against test 2026-01 and 2026-07. Quantiles, PSI (2025 deciles + a missing bin), KS. Base columns use
   the oof4 base (val for 2025, full for 2026), the base of 316/331/352.
2. Attribution: one LightGBM with the corrector's parameters on all 2025 matched rows (the full setting averages ten such fold models),
   per-feature contributions (pred_contrib) on 2025 and 2026 rows, summed by kind (where heard, movement, base), per airport and month.
3. Correction shift from the deployed outputs: geometry stage 1 of 352 (s1 - base, notes/ctx_corr_*_delta.parquet), its stage 2 (A2,
   x0.5), 291 (x1.0 corrector on v16) and 321 (geometry on v16) as candidate minus base after the window, per airport and month.

  .venv/Scripts/python.exe -X utf8 research/adsb/gprobe/g4_shift.py
Outputs: g4_shift_table.csv, g4_contrib.csv, g4_corr_shift.csv, printed summary (g4_shift.txt).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import os  # noqa: E402

os.chdir(ROOT)
import src.adsb_corr as ac  # noqa: E402

GEO = ["d_stand_first", "st_known", "d_stand_stop", "d_any_stand", "seg_code", "to_gap"]
WHERE = {"ev_first_dist_m", "ev_first_gs", "ev_first_stationary", "ev_stand_seen", "ev_stand_dur", "est_first", "first_minus_pred",
         "first_minus_sched", "ev_n_pts", "ev_share_mlat", "ev_max_gap_after_fm", "fm_seen", "fm_ok", "ev_fm_gap", "ev_to_gap",
         "d_stand_first", "st_known", "d_stand_stop", "d_any_stand", "seg_code", "to_gap"}
MOVE = {"est_mvt", "est_to", "est_mvt_last", "dt_to", "fm_minus_pred", "last_minus_pred", "fm_minus_sched", "ev_n_mid_stops",
        "ev_mid_stop_s", "ev_mid_stop_max", "ev_n_long_stops", "ev_fm_last_gap", "ev_last_long_dur"}
BASE = ["pred", "mu", "p_art", "dsch", "g", "apt_code", "hour"]


def kind(c):
    return "where heard" if c in WHERE else ("movement" if c in MOVE else "base")


def psi(r, t):
    r, t = np.asarray(r, float), np.asarray(t, float)
    rn, tn = np.isnan(r), np.isnan(t)
    rv = r[~rn]
    if len(rv) < 50:
        return np.nan
    edges = np.unique(np.quantile(rv, np.linspace(0, 1, 11)[1:-1]))
    def dist(x, nan):
        h = np.bincount(np.searchsorted(edges, x[~nan], side='right'), minlength=len(edges) + 1).astype(float)
        return np.append(h, nan.sum()) / max(len(x), 1)
    p, q = dist(r, rn) + 1e-4, dist(t, tn) + 1e-4
    return float(np.sum((q - p) * np.log(q / p)))


def ks(r, t):
    r, t = np.sort(np.asarray(r, float)[~np.isnan(r)]), np.sort(np.asarray(t, float)[~np.isnan(t)])
    if len(r) < 50 or len(t) < 50:
        return np.nan
    x = np.concatenate([r, t])
    return float(np.max(np.abs(np.searchsorted(r, x, side='right') / len(r) - np.searchsorted(t, x, side='right') / len(t))))


def main():
    A = ac.load_adsb("adsb_v1", "2bb0952d93192201").join(pd.read_parquet(ROOT / "cache" / "adsb_geo_v1.parquet").set_index("MVT_ID_mvt")[GEO], how="left")
    val = pd.read_parquet(ROOT / "cache" / "val_m6r25sgoof4_s0.parquet")
    full = pd.read_parquet(ROOT / "cache" / "full_m6r25sgoof4_s0.parquet")
    Xr, Xt = ac.design(val, A), ac.design(full, A)
    for X in (Xr, Xt):
        X["mon"] = X["day"].str[:7]
    feats = [c for c in BASE if c != "apt_code"] + ac.ADSB_COLS + GEO
    print(f"matched rows: 2025 ruler {len(Xr):,} ({Xr.mon.value_counts().sort_index().to_dict()}), 2026 test {len(Xt):,} ({Xt.mon.value_counts().sort_index().to_dict()})")

    # 1. shift table
    rows = []
    for ap in ac.APTS:
        r = Xr[Xr.ADEP_mvt == ap]
        for mon in ("2026-01", "2026-07"):
            t = Xt[(Xt.ADEP_mvt == ap) & (Xt.mon == mon)]
            rs = r[r.mon == mon.replace("2026", "2025")]
            for c in feats:
                rv, tv = r[c].to_numpy(float), t[c].to_numpy(float)
                q = lambda v, p: float(np.nanquantile(v, p)) if np.isfinite(v).sum() else np.nan  # noqa: E731
                rows.append(dict(feature=c, kind=kind(c), ap=ap, test=mon, n_2025=len(r), n_test=len(t), q10_2025=q(rv, .1), q50_2025=q(rv, .5), q90_2025=q(rv, .9),
                                 q10_test=q(tv, .1), q50_test=q(tv, .5), q90_test=q(tv, .9), q50_2025_same=q(rs[c].to_numpy(float), .5),
                                 nan_2025=float(np.isnan(rv).mean()) if len(rv) else np.nan, nan_test=float(np.isnan(tv).mean()) if len(tv) else np.nan,
                                 psi=psi(rv, tv) if len(t) >= 200 and len(r) >= 200 else np.nan, ks=ks(rv, tv) if len(t) >= 200 and len(r) >= 200 else np.nan))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "g4_shift_table.csv", index=False)
    s = S.dropna(subset=["psi"])
    summ = s.groupby(["feature", "kind"]).agg(cells=("psi", "size"), psi_med=("psi", "median"), psi_max=("psi", "max"), n_large=("psi", lambda x: int((x >= 0.25).sum())),
                                              n_mod=("psi", lambda x: int(((x >= 0.10) & (x < 0.25)).sum())), ks_med=("ks", "median"), ks_max=("ks", "max"))
    worst = s.loc[s.groupby("feature")["psi"].idxmax(), ["feature", "ap", "test"]].set_index("feature")
    summ = summ.reset_index().set_index("feature").join(worst.rename(columns={"ap": "worst_ap", "test": "worst_test"}))
    print("\nshift by feature (PSI over airport x test month cells with >= 200 matched rows each side; large >= 0.25, moderate 0.10 - 0.25):")
    print(summ.sort_values(["kind", "psi_med"], ascending=[True, False]).round(3).to_string())
    print("\nby kind: share of cells with PSI >= 0.25:", s.assign(L=s.psi >= 0.25).groupby("kind").L.mean().round(3).to_dict(),
          "| median PSI:", s.groupby("kind").psi.median().round(3).to_dict())
    key = s[s.feature.isin(["d_stand_first", "d_any_stand", "ev_first_dist_m", "est_first", "ev_stand_seen", "est_mvt", "ev_share_mlat", "ev_n_pts"])]
    print("\nmedians, 2025 pooled (same season) -> test, for where-heard keys and est_mvt, cells with PSI >= 0.25:")
    kk = key[key.psi >= 0.25][["feature", "ap", "test", "n_2025", "n_test", "q50_2025", "q50_2025_same", "q50_test", "nan_2025", "nan_test", "psi", "ks"]]
    print(kk.sort_values(["feature", "psi"], ascending=[True, False]).round(3).to_string(index=False))

    # 2. attribution with one corrector-like model on all 2025 matched rows
    import lightgbm as lgb
    cols = ac.BASE_COLS + ac.ADSB_COLS + GEO
    params = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.9, bagging_fraction=0.9,
                  bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=3, seed=0, deterministic=True, force_col_wise=True)
    tgt = np.clip(Xr["y"].to_numpy() - Xr["pred"].to_numpy(), -ac.CLIP, ac.CLIP)
    m = lgb.train(params, lgb.Dataset(Xr[cols].astype(float).to_numpy(), tgt), num_boost_round=300)
    imp = pd.Series(m.feature_importance("gain"), index=cols)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    print("\ngain share by kind:", imp.groupby(imp.index.map(kind)).sum().round(3).to_dict(), "| top 12:", imp.head(12).round(3).to_dict())
    out = []
    for nm, X in (("2025", Xr), ("2026", Xt)):
        C = m.predict(X[cols].astype(float).to_numpy(), pred_contrib=True)
        C = pd.DataFrame(C[:, :-1], columns=cols)
        K = pd.DataFrame({k: C[[c for c in cols if kind(c) == k]].sum(axis=1) for k in ("where heard", "movement", "base")})
        K["total"] = K.sum(axis=1)
        K["ap"], K["mon"] = X["ADEP_mvt"].to_numpy(), X["mon"].to_numpy()
        g = K.groupby(["ap", "mon"]).agg(n=("total", "size"), where_mean=("where heard", "mean"), move_mean=("movement", "mean"), base_mean=("base", "mean"),
                                         total_mean=("total", "mean"), where_sd=("where heard", "std"), total_sd=("total", "std"))
        out.append(g)
        if nm == "2026":
            top = C.abs().mean().sort_values(ascending=False)
            print("mean |contribution| on 2026 rows, top 10:", top.head(10).round(1).to_dict())
    G2 = pd.concat(out).sort_index()
    G2.to_csv(OUT / "g4_contrib.csv")
    print("\ncorrector-like model (in-sample on 2025), mean contribution by kind per airport and month (s):")
    print(G2.round(1).unstack("mon").to_string())

    # 3. correction shift of the deployed candidates
    P = pd.read_parquet(ROOT / "research" / "adsb" / "prc_all.parquet", columns=["MVT_ID_mvt", "day", "ADEP_mvt"]).set_index("MVT_ID_mvt")
    P["mon"] = P["day"].str[:7]
    mt = set(A.index)
    res = []

    def add(name, s_val, s_full, rows_only_matched=True):
        for setting, sr in (("2025", s_val), ("2026", s_full)):
            d = sr.to_frame("c").join(P[["ADEP_mvt", "mon"]])
            if rows_only_matched:
                d = d[d.index.isin(mt)]
            g = d.groupby(["ADEP_mvt", "mon"])["c"].agg(n="size", mean="mean", sd="std", q05=lambda x: x.quantile(.05), q95=lambda x: x.quantile(.95))
            g["corr"] = name
            res.append(g.reset_index())

    dv = pd.read_parquet(ROOT / "notes" / "ctx_corr_m6r25sgoof4adsbgeoctxv2x50w_val_s0_delta.parquet").set_index("MVT_ID_mvt")
    df_ = pd.read_parquet(ROOT / "notes" / "ctx_corr_m6r25sgoof4adsbgeoctxv2x50w_full_s0_delta.parquet").set_index("MVT_ID_mvt")
    bv, bf = val.set_index("MVT_ID_mvt")["pred"], full.set_index("MVT_ID_mvt")["pred"]
    add("geometry stage 1 (331/352 line)", dv["s1"] - bv.reindex(dv.index), df_["s1"] - bf.reindex(df_.index))
    add("A2 stage 2 x0.5 (352), all rows", 0.5 * dv["d2"], 0.5 * df_["d2"], rows_only_matched=False)

    def diff(c, b, s):
        x = pd.read_parquet(ROOT / "cache" / f"{s}_{c}_s0.parquet", columns=["MVT_ID_mvt", "pred"]).set_index("MVT_ID_mvt")["pred"]
        y = pd.read_parquet(ROOT / "cache" / f"{s}_{b}_s0.parquet", columns=["MVT_ID_mvt", "pred"]).set_index("MVT_ID_mvt")["pred"]
        return x - y.reindex(x.index)
    add("x1.0 corrector 291 (v16), after window", diff("m6r25sgcbxradsbw", "m6r25sgcbxrw", "val"), diff("m6r25sgcbxradsbw", "m6r25sgcbxrw", "full"))
    add("geometry 321 (v16), after window", diff("m6r25sgcbxradsbgeow", "m6r25sgcbxrw", "val"), diff("m6r25sgcbxradsbgeow", "m6r25sgcbxrw", "full"))
    R = pd.concat(res, ignore_index=True)
    R.to_csv(OUT / "g4_corr_shift.csv", index=False)
    print("\ncorrection per airport and month (mean / sd, s); flag: test mean outside the two 2025 means by > 5 s, or sign opposite to the same-season 2025 mean (|mean| > 3 s):")
    for nm, g in R.groupby("corr", sort=False):
        W = g.pivot(index="ADEP_mvt", columns="mon", values="mean")
        SD = g.pivot(index="ADEP_mvt", columns="mon", values="sd")
        N_ = g.pivot(index="ADEP_mvt", columns="mon", values="n")
        lines = []
        for ap in W.index:
            m25 = [W.at[ap, c] for c in ("2025-01", "2025-07") if c in W and pd.notna(W.at[ap, c]) and N_.at[ap, c] >= 200]
            fl = []
            for c, same in (("2026-01", "2025-01"), ("2026-07", "2025-07")):
                if c not in W or pd.isna(W.at[ap, c]) or N_.at[ap, c] < 200 or not m25:
                    continue
                v = W.at[ap, c]
                if v < min(m25) - 5 or v > max(m25) + 5:
                    fl.append(f"{c} out of range")
                sv = W.at[ap, same] if same in W and pd.notna(W.at[ap, same]) and N_.at[ap, same] >= 200 else np.nan
                if pd.notna(sv) and abs(sv) > 3 and abs(v) > 3 and np.sign(v) != np.sign(sv):
                    fl.append(f"{c} sign flip")
            cells = " | ".join(f"{c[2:]} {W.at[ap, c]:+.1f}/{SD.at[ap, c]:.0f} (n {int(N_.at[ap, c])})" for c in W.columns if pd.notna(W.at[ap, c]))
            lines.append(f"  {ap}: {cells}{'  <-- ' + ', '.join(fl) if fl else ''}")
        print(f"{nm}:\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
