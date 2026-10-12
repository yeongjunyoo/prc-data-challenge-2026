# SPDX-License-Identifier: GPL-3.0-only
"""R21 g5: R2a cross-fitted by month. Its only fitted constant is pi (art share among LIRF antiwin-R2 rows where art is possible,
|SCHED - LOBT_arr| > 3606). For each evaluated month pi uses 2025 rows of the other eleven months only. The 3606 is the matching window
(|BLOCK - LOBT| <= 3606 on all 2025 matched departures), not tuned. Rulers: val (Jan, Jul; final chain 1/7 form), fwd (Jul), fj (Feb, Jun;
gate19 Lm mean, with the antiwin R2 floor emulated as max(Lm, d_lobt + 3606 + m), m = out-of-month median as tools/antiwin)."""
import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
L = pd.read_parquet("cache/antiwin_link_v1.parquet")
L2 = L[L.r2 & (L.ADEP_mvt == "LIRF")].copy()
L2["art_ok"] = L2.sch_m_lobt.abs() > 3606; L2["art"] = np.abs(L2.y - L2.d_sch) <= 60
L2["day"] = pd.to_datetime(L2.t_mvt, unit="s").dt.strftime("%Y-%m-%d")
allr2 = L[L.r2 & (L.src > 0)]
def pi_ex(m):
    x = L2[(L2.src > 0) & (L2.src != m) & L2.art_ok]
    return (x.art.sum() + 0.5) / (len(x) + 1.0)
pv = pd.read_parquet("preds/holdout_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap_rmse281.37.parquet").set_index("MVT_ID_mvt").pred
pw = pd.read_parquet("preds/fwd_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.parquet").set_index("MVT_ID_mvt").pred
fj = [pd.read_parquet(f"research/f2/out_mac/gate19_{c}_F.parquet", columns=["MVT_ID_mvt", "Lm"]).set_index("MVT_ID_mvt").Lm for c in "ABC"]
pf = (fj[0] + fj[1] + fj[2]) / 3
alldays = pq.read_table("cache/all.parquet", columns=["t_mvt", "src", "PHASE_mvt"], filters=[("PHASE_mvt", "==", "DEP")]).to_pandas()
alldays["day"] = pd.to_datetime(alldays.t_mvt, unit="s").dt.strftime("%Y-%m-%d")
def run(name, P, months, N, emulate_floor=False):
    parts = []
    for m in months:
        E = L2[(L2.src == m) & L2.art_ok].set_index("MVT_ID_mvt").copy()
        E["f"] = P.reindex(E.index)
        if emulate_floor:
            mm = float(np.median((allr2[allr2.src != m].y - (allr2[allr2.src != m].d_lobt + 3606))))
            E["f"] = np.maximum(E.f, E.d_lobt + 3606 + mm)
        pi = pi_ex(m)
        E["new"] = pi * E.d_sch + (1 - pi) * E.f; E["pi"] = pi
        parts.append(E)
    Z = pd.concat(parts)
    dse = (Z.new - Z.y) ** 2 - (Z.f - Z.y) ** 2
    days = alldays[alldays.src.isin(months)].day.unique()
    dd = dse.groupby(Z.day).sum().reindex(days).fillna(0).to_numpy()
    b = np.random.default_rng(0).integers(0, len(dd), (4000, len(dd)))
    print(f"R2a {name}: pi by month {dict((m, round(pi_ex(m), 3)) for m in months)}, rows {len(Z)}, dMSE {dse.sum()/N:+.1f}, better {int((dse<0).sum())} worse {int((dse>0).sum())}, "
          f"day-boot P(<0) {np.mean(dd[b].sum(1) < 0):.3f}" + (f", without the one day-shift row {dse[Z.y < 60000].sum()/N:+.1f}" if (Z.y >= 60000).any() else ""))
run("(b) val", pv, [1, 7], 344419); run("(c) fwd", pw, [7], 190713); run("    fj ", pf, [2, 6], 326873, emulate_floor=True)
# label-level 12-month check (chain lomo P base as in d1) is in d1_r2art.py: 22 of 23 art-possible rows better, pi out of month.
x = L2[(L2.src > 0) & L2.art_ok]
print("art share among art-possible LIRF R2 rows by month:", x.groupby("src").art.agg(["sum", "size"]).T.to_dict())
