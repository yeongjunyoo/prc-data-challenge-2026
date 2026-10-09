# SPDX-License-Identifier: GPL-3.0-only
"""R10-B S1 (step 2): how closely do the AIU aggregates match our 2025 labels, and which rows explain the mismatch.
Daily: per airport and off-block UTC date (t_blk), T = 60 x DLY_ALL_PRE_2 against ours = sum max(0, t_blk - t_sch) over our departures.
  Per airport: median offset, SD and MAD of (T - ours - offset), share within 60 s, count ratio n / FLT_DEP_IFR_2.
  Mismatch explanations (per airport-day regressors): rows with |y - d| <= 60 (copy), y > 60,000 (shift), G (LIRF, NM unmatched),
  delay > 6 h, rows whose off-block date differs from the MVT date, n - FLT_DEP_IFR_2.
Monthly: per airport-month, AIU mean taxi-out m = 60 (REF + ADD) / NB seconds against our mean y over subsets of the month's departures
  (grouped by MVT month and by off-block month): all, non-copy, y < 3,600, non-copy and y < 3,600, non-copy and y < 7,200, plus TF / n.
  Per airport: offset (median over months) and the residual SD across months after the offset (what a monthly reconciliation could use).
Labels only (2025, src 1..12). Writes s1_days.parquet, s1_months.parquet.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "r10_b"
APS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetCurrentProcess.restype = wintypes.HANDLE
k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)

t = pq.read_table(ROOT / "cache" / "all.parquet", columns=["ADEP_mvt", "t_mvt", "t_blk", "t_sch", "y", "src", "FLIGHT_ID_mvt"],
                  filters=[("PHASE_mvt", "==", "DEP"), ("src", ">", 0)])
D = pd.DataFrame({"ap": t.column("ADEP_mvt").to_pandas().astype(str).to_numpy(), "tm": t.column("t_mvt").to_numpy().astype(float),
                  "tb": t.column("t_blk").to_numpy().astype(float), "ts": t.column("t_sch").to_numpy().astype(float),
                  "y": t.column("y").to_numpy().astype(float), "src": t.column("src").to_numpy(),
                  "nm": t.column("FLIGHT_ID_mvt").is_valid().to_numpy(zero_copy_only=False)})
del t
D = D[D["ap"].isin(APS)].reset_index(drop=True)
print("rows", len(D), "src values", sorted(pd.unique(D["src"]))[:20], "y NaN", int(D["y"].isna().sum()), flush=True)
D["d"] = D["tm"] - D["ts"]
D["dly"] = np.maximum(0.0, D["tb"] - D["ts"])
D["copy"] = np.abs(D["y"] - D["d"]) <= 60
D["shift"] = (D["y"] > 60000) & ~D["copy"]
D["G"] = (D["ap"] == "LIRF") & ~D["nm"]
D["big6h"] = D["dly"] > 6 * 3600
D["bday"] = pd.to_datetime(D["tb"], unit="s").dt.strftime("%Y-%m-%d")
D["mday"] = pd.to_datetime(D["tm"], unit="s").dt.strftime("%Y-%m-%d")
D["dchg"] = D["bday"] != D["mday"]
print("check y = t_mvt - t_blk:", float(np.nanmax(np.abs(D["y"] - (D["tm"] - D["tb"])))), flush=True)

# ---------------- daily ----------------
A = pd.read_csv(ROOT / "data" / "ext" / "aiu" / "all_pre_departure_delays_2025.csv")
A = A[A["APT_ICAO"].isin(APS)].rename(columns={"APT_ICAO": "ap", "FLT_DATE": "day"})
A["T"] = 60.0 * A["DLY_ALL_PRE_2"]
A = A.set_index(["ap", "day"])[["T", "FLT_DEP_IFR_2", "FLT_DEP_1"]]
g = D.groupby(["ap", "bday"])
S = pd.DataFrame({"ours": g["dly"].sum(), "n": g.size(), "n_copy": g["copy"].sum(), "n_shift": g["shift"].sum(), "n_G": g["G"].sum(),
                  "dly_G": g.apply(lambda s: s.loc[s["G"], "dly"].sum()), "dly_6h": g.apply(lambda s: s.loc[s["big6h"], "dly"].sum()),
                  "n_6h": g["big6h"].sum(), "n_dchg": g["dchg"].sum(), "dly_shift": g.apply(lambda s: s.loc[s["shift"], "dly"].sum())})
S.index.names = ["ap", "day"]
J = S.join(A, how="inner")
J["diff"] = J["T"] - J["ours"]
J["cr"] = J["n"] / J["FLT_DEP_IFR_2"]
J["dn"] = J["n"] - J["FLT_DEP_IFR_2"]
J = J.reset_index()
J["month"] = J["day"].str[5:7].astype(int)
rows = []
for ap, z in J.groupby("ap"):
    off = z["diff"].median()
    e = z["diff"] - off
    mad = 1.4826 * np.median(np.abs(e - np.median(e)))
    # the error of the AIU total as a statement about sum(d - y) over our delayed rows, per departure
    rows.append({"ap": ap, "days": len(z), "off_med": off, "sd": e.std(), "mad_sd": mad, "in60": (e.abs() <= 60).mean(), "p95": e.abs().quantile(0.95),
                 "per_dep_sd": (e / z["n"]).std(), "per_dep_mad": 1.4826 * np.median(np.abs(e / z["n"] - np.median(e / z["n"]))),
                 "cr_med": z["cr"].median(), "cr_off2": (np.abs(z["cr"] - 1) > 0.02).mean(), "corr_dn": np.corrcoef(e, z["dn"])[0, 1] if z["dn"].std() > 0 else np.nan,
                 "corr_ncopy": np.corrcoef(e, z["n_copy"])[0, 1], "corr_dly6h": np.corrcoef(e, z["dly_6h"])[0, 1] if z["dly_6h"].std() > 0 else np.nan,
                 "corr_dlyshift": np.corrcoef(e, z["dly_shift"])[0, 1] if z["dly_shift"].std() > 0 else np.nan,
                 "corr_n": np.corrcoef(e, z["n"])[0, 1], "corr_ours": np.corrcoef(e, z["ours"])[0, 1],
                 "off_month_sd": z.groupby("month")["diff"].median().std()})
R = pd.DataFrame(rows).set_index("ap")
print("\n== daily: T - ours (off-block UTC date), after the airport median offset")
print(R.round(3).to_string())
# slope check: is the mismatch multiplicative (T = a ours)?
print("\n== daily: least-squares T = a x ours + b per airport (a near 1 means additive offset only)")
for ap, z in J.groupby("ap"):
    a, b = np.polyfit(z["ours"], z["T"], 1)
    print(f"  {ap}: a {a:.4f}, b {b:,.0f}, corr {np.corrcoef(z['ours'], z['T'])[0, 1]:.5f}, sd(ours) {z['ours'].std():,.0f}")
J.to_parquet(OUT / "s1_days.parquet", index=False)

# ---------------- monthly ----------------
M = pd.read_csv(ROOT / "data" / "ext" / "aiu" / "taxi_out_additional_time_2025.csv")
M = M[M["APT_ICAO"].isin(APS)].rename(columns={"APT_ICAO": "ap"})
M["month"] = M["MONTH_NUM"].astype(int)
M["m_aiu"] = 60 * (M["TOTAL_REF_TIME_MIN"] + M["TOTAL_ADD_TIME_MIN"]) / M["TOTAL_REF_NB_FL"]
M["ref_aiu"] = 60 * M["TOTAL_REF_TIME_MIN"] / M["TOTAL_REF_NB_FL"]
M = M.set_index(["ap", "month"])[["TF", "VALID_FL", "NO_REF", "TOTAL_REF_NB_FL", "m_aiu", "ref_aiu"]]
D["mm"] = pd.to_datetime(D["tm"], unit="s").dt.month
D["bm"] = pd.to_datetime(D["tb"], unit="s").dt.month
D = D[pd.to_datetime(D["tm"], unit="s").dt.year == 2025]
subsets = {"all": np.ones(len(D), bool), "noncopy": ~D["copy"].to_numpy(), "y<3600": (D["y"] < 3600).to_numpy(),
           "nc&y<3600": (~D["copy"] & (D["y"] < 3600)).to_numpy(), "nc&y<7200": (~D["copy"] & (D["y"] < 7200)).to_numpy(),
           "nc&y<2700": (~D["copy"] & (D["y"] < 2700)).to_numpy(), "nc&60<y<3600": (~D["copy"] & (D["y"] < 3600) & (D["y"] > 60)).to_numpy()}
out = []
for key in ("mm", "bm"):
    for sname, msk in subsets.items():
        z = D[msk].groupby(["ap", key])["y"].agg(["mean", "size"])
        z.index.names = ["ap", "month"]
        J2 = z.join(M, how="inner")
        J2["gap"] = J2["m_aiu"] - J2["mean"]
        J2 = J2.reset_index()
        for ap, w in J2.groupby("ap"):
            out.append({"group": key, "subset": sname, "ap": ap, "gap_med": w["gap"].median(), "gap_sd": w["gap"].std(),
                        "gap_jan": float(w.loc[w["month"] == 1, "gap"].iloc[0]) if (w["month"] == 1).any() else np.nan,
                        "gap_jul": float(w.loc[w["month"] == 7, "gap"].iloc[0]) if (w["month"] == 7).any() else np.nan,
                        "n_over_nb": (w["size"] / w["TOTAL_REF_NB_FL"]).median(), "corr_mean": np.corrcoef(w["mean"], w["m_aiu"])[0, 1],
                        "sd_mean": w["mean"].std(), "sd_aiu": w["m_aiu"].std()})
O = pd.DataFrame(out)
O.to_parquet(OUT / "s1_months.parquet", index=False)
for key in ("mm", "bm"):
    print(f"\n== monthly ({'MVT month' if key == 'mm' else 'off-block month'}): gap = AIU mean taxi - our mean y; SD across the 12 months after the airport median")
    z = O[O["group"] == key]
    print("  gap SD across months (s):")
    print(z.pivot(index="ap", columns="subset", values="gap_sd").round(1).to_string())
    print("  gap median (airport offset, s):")
    print(z.pivot(index="ap", columns="subset", values="gap_med").round(1).to_string())
    print("  corr(our monthly mean, AIU monthly mean) across months:")
    print(z.pivot(index="ap", columns="subset", values="corr_mean").round(3).to_string())
    print("  SD of our monthly mean across months / SD of AIU monthly mean:")
    print(z.pivot(index="ap", columns="subset", values="sd_mean").round(1).to_string())
    print(z.groupby("ap")["sd_aiu"].first().round(1).to_string())
print("\n== monthly counts: our departures / TF and / TOTAL_REF_NB_FL (MVT month, all rows)")
z = D.groupby(["ap", "mm"]).size().rename("n")
z.index.names = ["ap", "month"]
J3 = z.to_frame().join(M, how="inner")
J3["n_tf"] = J3["n"] / J3["TF"]
J3["n_valid"] = J3["n"] / J3["VALID_FL"]
J3["n_nb"] = J3["n"] / J3["TOTAL_REF_NB_FL"]
print(J3.groupby(level=0)[["n_tf", "n_valid", "n_nb"]].median().round(4).to_string())
# the monthly additional time as seen from our labels: our mean y minus AIU reference, against AIU additional
print("\n== AIU 2025 and 2026 monthly mean taxi-out (s), Jan and Jul")
M26 = pd.read_csv(ROOT / "data" / "ext" / "aiu" / "taxi_out_additional_time_2026.csv")
M26 = M26[M26["APT_ICAO"].isin(APS)].rename(columns={"APT_ICAO": "ap"})
M26["m_aiu"] = 60 * (M26["TOTAL_REF_TIME_MIN"] + M26["TOTAL_ADD_TIME_MIN"]) / M26["TOTAL_REF_NB_FL"]
M26["month"] = M26["MONTH_NUM"].astype(int)
w = pd.concat([M.reset_index()[["ap", "month", "m_aiu"]].assign(yr=2025), M26[["ap", "month", "m_aiu"]].assign(yr=2026)])
print(w[w["month"].isin([1, 7])].pivot_table(index="ap", columns=["yr", "month"], values="m_aiu").round(1).to_string())
print("\n== AIU monthly mean taxi-out 2025 all months (s)")
print(M.reset_index().pivot(index="ap", columns="month", values="m_aiu").round(0).to_string())
