# SPDX-License-Identifier: GPL-3.0-only
"""R21 g4: every W2-family variant on the three rulers with delta re-estimated without the ruler's months (val: not 1/7, fwd: not 7, fj: not 2/6),
for the selection record we asked for. The 24 'actual grid' variants (T 5400/7200/9000 x A none/-4000 x lambda 0.5/1.0 x pool/airport)."""
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
alldays = pq.read_table("cache/all.parquet", columns=["t_mvt", "src", "PHASE_mvt"], filters=[("PHASE_mvt", "==", "DEP")]).to_pandas()
alldays["day"] = pd.to_datetime(alldays.t_mvt, unit="s").dt.strftime("%Y-%m-%d")
R = {"val": (pv, [1, 7], 344419), "fwd": (pw, [7], 190713), "fj": (pf, [2, 6], 326873)}
rows = []
for t, a, lam, mode in itertools.product([5400, 7200, 9000], [None, -4000], [0.5, 1.0], ["pool", "airport"]):
    rec = {"T": t, "A": a, "lam": lam, "delta": mode}
    for st, (P, ev, N) in R.items():
        F = T[~T.src.isin(ev)]
        G = F[(F.dl > t) & ((F.gap < a) if a is not None else True)]
        pool = G.off.mean()
        E = T[T.src.isin(ev)].copy(); E["f"] = P.reindex(E.index); E = E[E.f.notna()]
        Z = E[(E.dl > t) & ((E.gap < a) if a is not None else True)]
        if mode == "pool":
            d_ = np.full(len(Z), pool)
        else:
            g = G.groupby("ap").off.agg(["sum", "size"]); apd = ((g["sum"] + 10 * pool) / (g["size"] + 10)).to_dict(); d_ = Z.ap.map(apd).fillna(pool).to_numpy()
        new = np.clip(Z.f + lam * (Z.dl + d_ - Z.f), Z.dl - 3606, Z.dl + 3606).clip(lower=0)
        dse = (new - Z.y) ** 2 - (Z.f - Z.y) ** 2
        days = alldays[alldays.src.isin(ev)].day.unique()
        dd = dse.groupby(Z.day).sum().reindex(days).fillna(0).to_numpy()
        b = np.random.default_rng(0).integers(0, len(dd), (2000, len(dd)))
        rec[f"{st}"] = round(dse.sum() / N, 1); rec[f"{st}_P"] = round(float(np.mean(dd[b].sum(1) < 0)), 3)
    rows.append(rec)
S = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(S.to_string(index=False))
S.to_csv("research/r21/g4_selection_table.csv", index=False)
print("variants with val P>=0.9 and fwd P>=0.9 and fj<=0:", int(((S.val_P >= 0.9) & (S.fwd_P >= 0.9) & (S.fj <= 0)).sum()), "of", len(S))
