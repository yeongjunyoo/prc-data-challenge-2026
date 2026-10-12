# SPDX-License-Identifier: GPL-3.0-only
"""R21 e12: W8 (addendum 5 of research/r21/W2_preregistration.txt): cross-ruler residual correction, 3606 < d_lobt <= 7200, far AOBT_3."""
import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
cols = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "t_mvt", "t_lobt", "t_aobt3", "y", "src"]
T = pq.read_table("cache/all.parquet", columns=cols, filters=[("PHASE_mvt", "==", "DEP")]).to_pandas()
T = T[T.t_lobt.notna() & (T.src > 0)].copy()
T["dl"] = T.t_mvt - T.t_lobt; T["da"] = T.t_mvt - T.t_aobt3; T["ap"] = T.ADEP_mvt.astype(str)
T["day"] = pd.to_datetime(T.t_mvt, unit="s").dt.strftime("%Y-%m-%d")
T["row"] = (T.dl > 3606) & (T.dl <= 7200) & ((T.da - T.dl) < -4000)
T = T.set_index("MVT_ID_mvt")
FILES = {"val": "preds/holdout_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap_rmse281.37.parquet",
         "fwd": "preds/fwd_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.parquet"}
NN = {"val": 344419, "fwd": 190713, "fj": 326873}; EV = {"val": [1, 7], "fwd": [7], "fj": [2, 6]}
P3 = {k: pd.read_parquet(v).set_index("MVT_ID_mvt").pred.astype(float) for k, v in FILES.items()}
fj = [pd.read_parquet(f"research/f2/out_mac/gate19_{c}_F.parquet", columns=["MVT_ID_mvt", "Lm"]).set_index("MVT_ID_mvt").Lm for c in "ABC"]
P3["fj"] = (fj[0] + fj[1] + fj[2]) / 3
def rows(st):
    E = T[T.src.isin(EV[st]) & T.row].copy(); E["f"] = P3[st].reindex(E.index); return E[E.f.notna()]
def bias_tab(st):
    E = rows(st); g = (E.f - E.y).groupby(E.ap).agg(["sum", "size"])
    return {a: (r["sum"] / r["size"]) * r["size"] / (r["size"] + 20) for a, r in g.iterrows()}
src_of = {"val": "fj", "fwd": "fj", "fj": "val"}
for st in ("val", "fwd", "fj"):
    bt = bias_tab(src_of[st]); E = rows(st)
    b = E.ap.map(bt).fillna(0.0)
    new = np.clip(E.f - 0.5 * b, E.dl - 3606, E.dl + 3606).clip(lower=0)
    dse = (new - E.y) ** 2 - (E.f - E.y) ** 2
    days = T[T.src.isin(EV[st])].day.unique()
    dd = dse.groupby(E.day).sum().reindex(days).fillna(0).to_numpy()
    bb = np.random.default_rng(0).integers(0, len(dd), (4000, len(dd)))
    print(f"W8 {st:3s} (bias from {src_of[st]}: {dict((k, round(v)) for k, v in bt.items())}): rows {len(E)}, dMSE {dse.sum()/NN[st]:+6.1f}, "
          f"better {int((dse<0).sum())} worse {int((dse>0).sum())}, P(<0) {np.mean(dd[bb].sum(1) < 0):.3f}, move {((new-E.f)**2).sum()/NN[st]:.1f}")
