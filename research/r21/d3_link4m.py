# SPDX-License-Identifier: GPL-3.0-only
"""R21 d3: the antiwin link (tools/antiwin.build_link definition, re-implemented read-only) on final_ranking.parquet (four 2026 months),
written to research/r21/d3_link4m.parquet only. Then R2a rows over the four months and their movement on the staged final 598c."""
import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]; WIN = 3606.0
R = pq.read_table("data/final_phase/final_ranking.parquet", columns=["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
                                                                     "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt"]).to_pandas()
for c, s in (("MVT_TIME_UTC_mvt", "t_mvt"), ("SCHED_TIME_UTC_mvt", "t_sch"), ("LOBT_flt", "t_lobt"), ("AOBT_3_flt", "t_aobt3")):
    v = R[c]; R[s] = np.where(v.notna(), v.astype("int64") / 1e6, np.nan)
ok = R.FLIGHT_mvt.notna()
dep = R[(R.PHASE_mvt == "DEP") & R.ADES_mvt.isin(APTS) & ok].copy()
arr = R[(R.PHASE_mvt == "ARR") & R.ADEP_mvt.isin(APTS) & ok].copy()
for d in (dep, arr):
    d["k"] = d.ADEP_mvt.astype(str) + "|" + d.ADES_mvt.astype(str) + "|" + d.FLIGHT_mvt.astype(str)
arr = arr.sort_values(["k", "t_mvt"]).reset_index(drop=True)
a_k, a_t = arr.k.to_numpy(), arr.t_mvt.to_numpy()
uk, st = np.unique(a_k, return_index=True); en = np.append(st[1:], len(a_k)); amap = {k: (s, e) for k, s, e in zip(uk, st, en)}
dep = dep[dep.FLIGHT_ID_mvt.isna()].reset_index(drop=True)
link = np.full(len(dep), -1, np.int64)
for i, (k, t) in enumerate(zip(dep.k.to_numpy(), dep.t_mvt.to_numpy())):
    r = amap.get(k)
    if r is None: continue
    s, e = r; j = s + np.searchsorted(a_t[s:e], t + 300, side="left")
    if j < e and a_t[j] - t <= 8 * 3600: link[i] = j
okl = link >= 0
A = arr.iloc[link[okl]].reset_index(drop=True)
L = dep.loc[okl, ["MVT_ID_mvt", "ADEP_mvt", "t_mvt", "t_sch"]].reset_index(drop=True)
L["a_nm"] = A.FLIGHT_ID_mvt.notna().to_numpy(); L["a_t_lobt"] = A.t_lobt.to_numpy(); L["a_t_aobt3"] = A.t_aobt3.to_numpy()
L = L[L.a_nm].reset_index(drop=True)
L["d_lobt"] = L.t_mvt - L.a_t_lobt; L["d_aobt"] = L.t_mvt - L.a_t_aobt3; L["d_sch"] = L.t_mvt - L.t_sch; L["sch_m_lobt"] = L.t_sch - L.a_t_lobt
L["r1"] = (L.ADEP_mvt == "LIRF") & (L.d_sch >= 3600) & (np.abs(L.sch_m_lobt) <= WIN) & (L.d_lobt > WIN) & np.isfinite(L.d_aobt)
L["r2"] = L.ADEP_mvt.isin(["LIRF", "LTFM"]) & (L.d_lobt > 0) & (L.d_lobt <= WIN)
L["mon"] = pd.to_datetime(L.t_mvt, unit="s").dt.month
L.to_parquet("research/r21/d3_link4m.parquet", index=False)
old = pd.read_parquet("cache/antiwin_link_v1.parquet"); old = old[old.src == 0]
j = L[L.mon.isin([1, 7])]
print("link rows 4 months:", len(L), " by month:", L.mon.value_counts().sort_index().to_dict(), " Jan/Jul vs cache link test rows:", len(j), len(old),
      " same ids:", len(set(j.MVT_ID_mvt) & set(old.MVT_ID_mvt)))
print("R1 by month:", L[L.r1].mon.value_counts().sort_index().to_dict(), " R2 by month:", L[L.r2].mon.value_counts().sort_index().to_dict())
x = L[L.r2 & (L.ADEP_mvt == "LIRF") & (L.sch_m_lobt.abs() > 3606)].copy()
F = pd.read_parquet("research/f3/queue_ready_final/598c_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3rBbFamfhgkcxncrdbawgpglgjaavskapff9.parquet").set_index("MVT_ID_mvt").pred
x["f"] = F.reindex(x.MVT_ID_mvt).to_numpy()
pi = 0.896
x["new"] = pi * x.d_sch + (1 - pi) * x.f
print("R2a rows by month:", x.mon.value_counts().sort_index().to_dict())
print(f"four-month move MSE (N=670,790): {((x.new - x.f) ** 2).sum() / 670790:.2f}; Feb/Jun part {((x[x.mon.isin([2,6])].new - x[x.mon.isin([2,6])].f) ** 2).sum() / 670790:.2f}")
eg = (pi * ((x.new - x.d_sch) ** 2 - (x.f - x.d_sch) ** 2) + (1 - pi) * ((x.new - x.f) ** 2)).sum()
print(f"expected four-month dMSE if 2025 art share holds: {eg / 670790:+.1f}  (board-equivalent for Jan/Jul part {((pi*((x[x.mon.isin([1,7])].new-x[x.mon.isin([1,7])].d_sch)**2-(x[x.mon.isin([1,7])].f-x[x.mon.isin([1,7])].d_sch)**2)+(1-pi)*(x[x.mon.isin([1,7])].new-x[x.mon.isin([1,7])].f)**2).sum())/344841:+.1f})")
print(x[["MVT_ID_mvt", "mon", "d_sch", "d_lobt", "sch_m_lobt", "f", "new"]].round(1).to_string())
