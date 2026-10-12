# SPDX-License-Identifier: GPL-3.0-only
"""R21 e9: W5 (addendum 3 of research/r21/W2_preregistration.txt)."""
import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
cols = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "t_mvt", "t_sch", "t_lobt", "t_eobt1", "t_aobt3", "y", "src"]
T = pq.read_table("cache/all.parquet", columns=cols, filters=[("PHASE_mvt", "==", "DEP"), ("ADEP_mvt", "==", "LIRF")]).to_pandas()
T = T[T.t_lobt.notna()].copy()
T["dl"] = T.t_mvt - T.t_lobt; T["da"] = T.t_mvt - T.t_aobt3; T["de"] = T.t_mvt - T.t_eobt1; T["d"] = T.t_mvt - T.t_sch
T["day"] = pd.to_datetime(T.t_mvt, unit="s").dt.strftime("%Y-%m-%d")
T["row"] = (T.da - T.dl) < -4000
T["Ldist"] = (T.t_sch - T.t_lobt).abs() > 60
T["Edist"] = ((T.t_eobt1 - T.t_lobt).abs() > 60) & ((T.t_eobt1 - T.t_sch).abs() > 60)
T["cL"] = (np.abs(T.y - T.dl) <= 6.5); T["cE"] = (np.abs(T.y - T.de) <= 6.5)
T = T.set_index("MVT_ID_mvt")
FILES = {"val": ("preds/holdout_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap_rmse281.37.parquet", [1, 7], [2, 3, 4, 5, 6, 8, 9, 10, 11, 12], 344419),
         "fwd": ("preds/fwd_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.parquet", [7], [1, 2, 3, 4, 5, 6], 190713)}
P3 = {k: pd.read_parquet(v[0]).set_index("MVT_ID_mvt").pred.astype(float) for k, v in FILES.items()}
fj = [pd.read_parquet(f"research/f2/out_mac/gate19_{c}_F.parquet", columns=["MVT_ID_mvt", "Lm"]).set_index("MVT_ID_mvt").Lm for c in "ABC"]
P3["fj"] = (fj[0] + fj[1] + fj[2]) / 3
spec = {**{k: (v[1], v[2], v[3]) for k, v in FILES.items()}, "fj": ([2, 6], [1, 3, 4, 5, 7, 8, 9, 10, 11, 12], 326873)}
allday = pq.read_table("cache/all.parquet", columns=["t_mvt", "src", "PHASE_mvt"], filters=[("PHASE_mvt", "==", "DEP")]).to_pandas()
allday["day"] = pd.to_datetime(allday.t_mvt, unit="s").dt.strftime("%Y-%m-%d")
for st, (evm, fitm, N) in spec.items():
    fit = T[T.src.isin(fitm) & T.row]
    pL = fit[fit.Ldist].cL.mean(); pE = fit[fit.Edist].cE.mean()
    E = T[T.src.isin(evm) & T.row].copy(); E["f"] = P3[st].reindex(E.index); E = E[E.f.notna()]
    new = E.f + pL * E.Ldist * (E.dl - E.f) + pE * E.Edist.fillna(False) * (E.de - E.f).fillna(0)
    new = np.clip(new, E.dl - 3606, E.dl + 3606).clip(lower=0)
    dse = (new - E.y) ** 2 - (E.f - E.y) ** 2
    days = allday[allday.src.isin(evm)].day.unique()
    dd = dse.groupby(E.day).sum().reindex(days).fillna(0).to_numpy()
    b = np.random.default_rng(0).integers(0, len(dd), (4000, len(dd)))
    print(f"W5 {st:3s}: pL {pL:.3f} pE {pE:.3f} (fit rows {len(fit)}), rows {len(E)}, dMSE {dse.sum()/N:+7.1f}, better {int((dse<0).sum())} worse {int((dse>0).sum())}, "
          f"P(<0) {np.mean(dd[b].sum(1) < 0):.3f}, move {((new-E.f)**2).sum()/N:.1f}")
