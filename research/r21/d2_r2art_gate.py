# SPDX-License-Identifier: GPL-3.0-only
"""R21 d2: candidate R2a on the final's own rulers (val = 2025-01/07 holdout file, fwd = 2025-07 forward file of the 1/7 form L3S)
and its test movement on the staged final 598c (Jan/Jul rows; Feb/Jun need a four-month link table, see d3).
R2a: LIRF rows of the antiwin link table with 0 < d_lobt <= 3606 (R2) and |SCHED - LOBT_arr| > 3606 (art possible):
     new = pi * d_sch + (1 - pi) * current, pi = art share of such rows in the setting's fit months (val: not 1/7, fwd: 1-6, full: 1-12)."""
import sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
L = pd.read_parquet("cache/antiwin_link_v1.parquet")
L = L[L.r2 & (L.ADEP_mvt == "LIRF")].copy()
L["art_ok"] = L.sch_m_lobt.abs() > 3606
L["art"] = np.abs(L.y - L.d_sch) <= 60
FIT = {"val": [m for m in range(1, 13) if m not in (1, 7)], "fwd": [1, 2, 3, 4, 5, 6], "full": list(range(1, 13))}
EV = {"val": [1, 7], "fwd": [7]}
FILES = {"val": "preds/holdout_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap_rmse281.37.parquet",
         "fwd": "preds/fwd_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.parquet"}
def pi_of(months):
    x = L[L.src.isin(months) & L.art_ok]
    return (x.art.sum() + 0.5) / (len(x) + 1.0), len(x)
for st, N in (("val", 344419), ("fwd", 190713)):
    P = pd.read_parquet(FILES[st]).set_index("MVT_ID_mvt").pred.astype(float)
    pi, n = pi_of(FIT[st])
    x = L[L.src.isin(EV[st]) & L.art_ok].copy()
    x["f"] = P.reindex(x.MVT_ID_mvt).to_numpy()
    x["new"] = pi * x.d_sch + (1 - pi) * x.f
    x["dse"] = (x.new - x.y) ** 2 - (x.f - x.y) ** 2
    print(f"{st}: pi {pi:.3f} (fit rows {n}), changed rows {len(x)}, better {int((x.dse<0).sum())}, worse {int((x.dse>0).sum())}, dMSE {x.dse.sum()/N:+.2f}, move MSE {((x.new-x.f)**2).sum()/N:.2f}")
    print(x[["MVT_ID_mvt", "src", "d_sch", "d_lobt", "sch_m_lobt", "y", "f", "new"]].round(1).to_string())
# test movement on the staged final, Jan/Jul rows
F = pd.read_parquet("research/f3/queue_ready_final/598c_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3rBbFamfhgkcxncrdbawgpglgjaavskapff9.parquet").set_index("MVT_ID_mvt").pred
pi, n = pi_of(FIT["full"])
t = L[(L.src == 0) & L.art_ok].copy()
t["f"] = F.reindex(t.MVT_ID_mvt).to_numpy(); t["new"] = pi * t.d_sch + (1 - pi) * t.f
print(f"test Jan/Jul: pi {pi:.3f}, rows {len(t)}, move MSE (board N) {((t.new-t.f)**2).sum()/344841:.2f}, mean move {(t.new-t.f).mean():+.0f} s")
print(t[["MVT_ID_mvt", "d_sch", "d_lobt", "sch_m_lobt", "f", "new"]].round(1).to_string())
# expected board gain if the 2025 art share holds: E[dSE] = pi*((new-d)^2-(f-d)^2) + (1-pi)*(... other outcome ~ f) approx
eg = (pi * ((t.new - t.d_sch) ** 2 - (t.f - t.d_sch) ** 2) + (1 - pi) * ((t.new - t.f) ** 2)).sum() / 344841
print(f"expected board dMSE if 2025 art share holds and non-art rows land at the current value: {eg:+.1f}")
