# SPDX-License-Identifier: GPL-3.0-only
"""R21 g3: W2 family cross-fitted by day folds (our alternative): calendar days of the ruler months (2025 Jan, Feb, Jun, Jul) go to
5 folds (seeded); for fold k the variant is selected on all ruler blocks' rows of the other folds (val A+B, fwd C, fj D+E), and delta is
re-estimated from 2025 labels on days outside fold k (all twelve months otherwise). Grids: 'actual' (what I looked at before registering W2)
and 'wide'. Window +-3606 and floor 0 fixed. Then the gate on the assembled out-of-fold rows."""
import sys, itertools
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
cols = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "t_mvt", "t_lobt", "t_aobt3", "y", "src"]
T = pq.read_table("cache/all.parquet", columns=cols, filters=[("PHASE_mvt", "==", "DEP")]).to_pandas()
T = T[T.t_lobt.notna() & T.t_aobt3.notna() & (T.src > 0)].copy()
T["dl"] = T.t_mvt - T.t_lobt; T["gap"] = (T.t_mvt - T.t_aobt3) - T.dl; T["ap"] = T.ADEP_mvt.astype(str)
T["day"] = pd.to_datetime(T.t_mvt, unit="s").dt.strftime("%Y-%m-%d"); T["off"] = T.y - T.dl
T = T[T.dl > 5400].set_index("MVT_ID_mvt")
pv = pd.read_parquet("preds/holdout_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap_rmse281.37.parquet").set_index("MVT_ID_mvt").pred
pw = pd.read_parquet("preds/fwd_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.parquet").set_index("MVT_ID_mvt").pred
fj = [pd.read_parquet(f"research/f2/out_mac/gate19_{c}_F.parquet", columns=["MVT_ID_mvt", "Lm"]).set_index("MVT_ID_mvt").Lm for c in "ABC"]
pf = (fj[0] + fj[1] + fj[2]) / 3
BL = {"A": (pv, 1), "B": (pv, 7), "C": (pw, 7), "D": (pf, 2), "E": (pf, 6)}
blk = {}
for k, (P, m) in BL.items():
    E = T[T.src == m].copy(); E["f"] = P.reindex(E.index); blk[k] = E[E.f.notna()]
rdays = sorted(set(pd.concat([b.day for b in blk.values()])))
alldays = pq.read_table("cache/all.parquet", columns=["t_mvt", "src", "PHASE_mvt"], filters=[("PHASE_mvt", "==", "DEP")]).to_pandas()
alldays["day"] = pd.to_datetime(alldays.t_mvt, unit="s").dt.strftime("%Y-%m-%d")
cal = sorted(alldays[alldays.src.isin([1, 2, 6, 7])].day.unique())
fold = dict(zip(cal, np.random.default_rng(7).integers(0, 5, len(cal))))
def mask(E, t, a):
    mm = E.dl > t
    if a is not None: mm &= E.gap < a
    return mm
def apply(E, v, F):
    t, a, lam, mode = v
    Z = E[mask(E, t, a)]
    G = F[mask(F, t, a)]
    pool = G.off.mean()
    if mode == "pool":
        dl_ = np.full(len(Z), pool)
    else:
        g = G.groupby("ap").off.agg(["sum", "size"]); apd = ((g["sum"] + 10 * pool) / (g["size"] + 10)).to_dict()
        dl_ = Z.ap.map(apd).fillna(pool).to_numpy()
    new = np.clip(Z.f + lam * (Z.dl + dl_ - Z.f), Z.dl - 3606, Z.dl + 3606).clip(lower=0)
    return Z.assign(new=new)
GRIDS = {"actual": list(itertools.product([5400, 7200, 9000], [None, -4000], [0.5, 1.0], ["pool", "airport"])),
         "wide": list(itertools.product([5400, 6300, 7200, 8100, 9000], [None, -2400, -3200, -4000, -4800, -6000], [0.25, 0.5, 0.75, 1.0], ["pool", "airport"]))}
for gname, GRID in GRIDS.items():
    outs = {k: [] for k in BL}
    for k in range(5):
        F = T[T.day.map(lambda d: fold.get(d, -1)) != k]      # labels outside fold k (other months always in)
        best, bv = None, None
        for v in GRID:
            s = 0.0
            for b in ("A", "B", "C", "D", "E"):
                E = blk[b][blk[b].day.map(fold) != k]
                Z = apply(E, v, F)
                s += float(((Z.new - Z.y) ** 2 - (Z.f - Z.y) ** 2).sum())
            if best is None or s < best: best, bv = s, v
        print(f"[{gname}] fold {k}: selected {bv}")
        for b in BL:
            E = blk[b][blk[b].day.map(fold) == k]
            outs[b].append(apply(E, bv, F))
    def gate(name, keys, N, months):
        Z = pd.concat([pd.concat(outs[x]) for x in keys])
        dse = (Z.new - Z.y) ** 2 - (Z.f - Z.y) ** 2
        days = alldays[alldays.src.isin(months)].day.unique()
        dd = dse.groupby(Z.day).sum().reindex(days).fillna(0).to_numpy()
        bb = np.random.default_rng(0).integers(0, len(dd), (4000, len(dd)))
        print(f"  [{gname}] {name}: rows {len(Z)}, dMSE {dse.sum()/N:+7.1f}, better {int((dse<0).sum())} worse {int((dse>0).sum())}, day-boot P(<0) {np.mean(dd[bb].sum(1) < 0):.3f}")
    gate("(b) val A+B", "AB", 344419, [1, 7]); gate("(c) fwd C  ", "C", 190713, [7]); gate("    fj  D+E", "DE", 326873, [2, 6])
