# SPDX-License-Identifier: GPL-3.0-only
"""R21 e5: W2b (d_lobt > 5400) and W2u (upper edge) as in the addendum of research/r21/W2_preregistration.txt."""
import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
cols = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "t_mvt", "t_lobt", "t_aobt3", "y", "src"]
T = pq.read_table("cache/all.parquet", columns=cols, filters=[("PHASE_mvt", "==", "DEP")]).to_pandas()
T = T[T.t_lobt.notna()].copy()
T["dl"] = T.t_mvt - T.t_lobt; T["da"] = T.t_mvt - T.t_aobt3; T["ap"] = T.ADEP_mvt.astype(str)
T["day"] = pd.to_datetime(T.t_mvt, unit="s").dt.strftime("%Y-%m-%d")
T = T.set_index("MVT_ID_mvt")
RULES = {"W2b": (T.dl > 5400) & ((T.da - T.dl) < -4000), "W2u": (T.dl + 3606 < 1500)}
FILES = {"val": ("preds/holdout_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap_rmse281.37.parquet", [1, 7], [2, 3, 4, 5, 6, 8, 9, 10, 11, 12], 344419),
         "fwd": ("preds/fwd_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.parquet", [7], [1, 2, 3, 4, 5, 6], 190713)}
fj = [pd.read_parquet(f"research/f2/out_mac/gate19_{c}_F.parquet", columns=["MVT_ID_mvt", "Lm"]).set_index("MVT_ID_mvt").Lm for c in "ABC"]
fjm = (fj[0] + fj[1] + fj[2]) / 3
preds = {k: pd.read_parquet(v[0]).set_index("MVT_ID_mvt").pred.astype(float) for k, v in FILES.items()}
preds["fj"] = fjm
spec = {**{k: (v[1], v[2], v[3]) for k, v in FILES.items()}, "fj": ([2, 6], [1, 3, 4, 5, 7, 8, 9, 10, 11, 12], 326873)}
for rn, rm in RULES.items():
    for st, (evm, fitm, N) in spec.items():
        fit = T[T.src.isin(fitm) & rm]
        dlt = float((fit.y - fit.dl).mean())
        E = T[T.src.isin(evm) & rm].copy(); E["f"] = preds[st].reindex(E.index); E = E[E.f.notna()]
        new = np.clip(E.f + 0.5 * (E.dl + dlt - E.f), E.dl - 3606, E.dl + 3606).clip(lower=0)
        dse = (new - E.y) ** 2 - (E.f - E.y) ** 2
        dd = dse.groupby(E.day).sum().reindex(T[T.src.isin(evm)].day.unique()).fillna(0).to_numpy()
        b = np.random.default_rng(0).integers(0, len(dd), (4000, len(dd)))
        print(f"{rn} {st:3s}: delta {dlt:+.0f} (fit {len(fit)}), rows {len(E)}, dMSE {dse.sum()/N:+7.1f}, better {int((dse<0).sum())} worse {int((dse>0).sum())}, "
              f"P(<0) {np.mean(dd[b].sum(1) < 0):.3f}, bias {(E.f-E.y).mean():+.0f} -> {(new-E.y).mean():+.0f}, move {((new-E.f)**2).sum()/N:.1f}")
