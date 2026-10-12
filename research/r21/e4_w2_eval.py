# SPDX-License-Identifier: GPL-3.0-only
"""R21 e4: W2 as pre-registered in research/r21/W2_preregistration.txt (02:09 KST). val, fwd (final chain 1/7-form ruler files),
fj (2025 Feb/Jun, F2 gate19 Lm three-chain mean = the final's Feb/Jun form at the corrector stage), and the staged final's test movement."""
import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
cols = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "FLIGHT_ID_mvt", "t_mvt", "t_lobt", "t_aobt3", "y", "src"]
T = pq.read_table("cache/all.parquet", columns=cols, filters=[("PHASE_mvt", "==", "DEP")]).to_pandas()
T = T[T.t_lobt.notna()].copy()
T["dl"] = T.t_mvt - T.t_lobt; T["da"] = T.t_mvt - T.t_aobt3; T["ap"] = T.ADEP_mvt.astype(str)
T["day"] = pd.to_datetime(T.t_mvt, unit="s").dt.strftime("%Y-%m-%d")
T["row"] = (T.dl > 7200) & ((T.da - T.dl) < -4000)
T = T.set_index("MVT_ID_mvt")
def delta(months):
    x = T[T.src.isin(months) & T.row]
    return float((x.y - x.dl).mean()), len(x)
def run(name, f, evm, fitm, N, lam):
    dlt, nfit = delta(fitm)
    E = T[T.src.isin(evm) & T.row].copy()
    E["f"] = f.reindex(E.index)
    E = E[E.f.notna()]
    new = np.clip(E.f + lam * (E.dl + dlt - E.f), E.dl - 3606, E.dl + 3606).clip(lower=0)
    dse = (new - E.y) ** 2 - (E.f - E.y) ** 2
    dd = dse.groupby(E.day).sum()
    allday = T[T.src.isin(evm)].day.unique()
    dd = dd.reindex(allday).fillna(0).to_numpy()
    b = np.random.default_rng(0).integers(0, len(dd), (4000, len(dd)))
    bias0 = (E.f - E.y).mean(); bias1 = (new - E.y).mean()
    print(f"{name:4s} lam {lam}: delta {dlt:+.0f} (fit rows {nfit}), rows {len(E)}, dMSE {dse.sum()/N:+7.1f}, better {int((dse<0).sum())} worse {int((dse>0).sum())}, "
          f"day-boot P(<0) {np.mean(dd[b].sum(1) < 0):.3f}, bias f-y {bias0:+.0f} -> {bias1:+.0f}, move MSE {((new-E.f)**2).sum()/N:.1f}, airports {E.ap.value_counts().to_dict()}")
    return dse.sum() / N
FILES = {"val": ("preds/holdout_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap_rmse281.37.parquet", [1, 7], [2, 3, 4, 5, 6, 8, 9, 10, 11, 12], 344419),
         "fwd": ("preds/fwd_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.parquet", [7], [1, 2, 3, 4, 5, 6], 190713)}
fj = [pd.read_parquet(f"research/f2/out_mac/gate19_{c}_F.parquet", columns=["MVT_ID_mvt", "Lm"]).set_index("MVT_ID_mvt").Lm for c in "ABC"]
fjm = (fj[0] + fj[1] + fj[2]) / 3
for lam in (0.5, 1.0):
    for st, (fp, evm, fitm, N) in FILES.items():
        run(st, pd.read_parquet(fp).set_index("MVT_ID_mvt").pred.astype(float), evm, fitm, N, lam)
    run("fj", fjm, [2, 6], [1, 3, 4, 5, 7, 8, 9, 10, 11, 12], 326873, lam)
# test movement on the staged final (four months)
R = pq.read_table("data/final_phase/final_ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt"]).to_pandas()
D = R[(R.PHASE_mvt == "DEP") & R.LOBT_flt.notna()].copy()
D["dl"] = (D.MVT_TIME_UTC_mvt.astype("int64") - D.LOBT_flt.astype("int64")) / 1e6
D["da"] = np.where(D.AOBT_3_flt.notna(), (D.MVT_TIME_UTC_mvt.astype("int64") - D.AOBT_3_flt.astype("int64").where(D.AOBT_3_flt.notna(), 0)) / 1e6, np.nan)
D["mon"] = D.MVT_TIME_UTC_mvt.dt.month
F = pd.read_parquet("research/f3/queue_ready_final/598c_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3rBbFamfhgkcxncrdbawgpglgjaavskapff9.parquet").set_index("MVT_ID_mvt").pred
D["f"] = F.reindex(D.MVT_ID_mvt).to_numpy()
Z = D[(D.dl > 7200) & ((D.da - D.dl) < -4000)].copy()
dlt, nfit = delta(list(range(1, 13)))
for lam in (0.5, 1.0):
    new = np.clip(Z.f + lam * (Z.dl + dlt - Z.f), Z.dl - 3606, Z.dl + 3606).clip(lower=0)
    mv = (new - Z.f)
    print(f"test lam {lam}: delta(full) {dlt:+.0f} (fit {nfit}), rows {len(Z)} by month {Z.mon.value_counts().sort_index().to_dict()}, airports {Z.ADEP_mvt.value_counts().to_dict()}")
    print(f"   move MSE four-month {(mv**2).sum()/670790:.1f}; Jan/Jul board {(mv[Z.mon.isin([1,7])]**2).sum()/344841:.1f}; Feb/Jun {(mv[Z.mon.isin([2,6])]**2).sum()/670790:.1f} (4m N); mean move {mv.mean():+.0f} s")
    if lam == 0.5:
        Z.assign(new=new).to_parquet("research/r21/e4_w2_test_rows.parquet")
