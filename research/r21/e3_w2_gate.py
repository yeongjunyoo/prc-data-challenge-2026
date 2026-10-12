# SPDX-License-Identifier: GPL-3.0-only
"""R21 e3: candidate W2 on the final chain's own rulers. Rows: NM-matched, non-G, d_lobt = MVT - LOBT > T.
new = f + lam * (d_lobt + delta - f), delta = mean(y - d_lobt) over the band rows of the setting's fit months (val: 2025 months except 1, 7;
fwd: 1-6), pooled over airports (and an airport-specific variant with shrinkage to the pool, k = 10 rows).
Report dMSE (all rows / N of the ruler), rows changed, better/worse, day-bootstrap P(dMSE < 0). Gate 7.8 (a) non-G body: these rows are not G;
they are 'body' or 'B' or 'copy' rows; reported separately as the change on non-G rows."""
import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
cols = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "FLIGHT_ID_mvt", "t_mvt", "t_sch", "t_lobt", "y", "src"]
T = pq.read_table("cache/all.parquet", columns=cols, filters=[("PHASE_mvt", "==", "DEP")]).to_pandas()
T = T[T.t_lobt.notna()].copy()
T["dl"] = T.t_mvt - T.t_lobt; T["ap"] = T.ADEP_mvt.astype(str); T["off"] = T.y - T.dl
T["day"] = pd.to_datetime(T.t_mvt, unit="s").dt.strftime("%Y-%m-%d")
T = T.set_index("MVT_ID_mvt")
FILES = {"val": ("preds/holdout_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap_rmse281.37.parquet", [1, 7], [2, 3, 4, 5, 6, 8, 9, 10, 11, 12], 344419),
         "fwd": ("preds/fwd_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.parquet", [7], [1, 2, 3, 4, 5, 6], 190713)}
def delta_tab(fit, thr):
    b = fit[fit.dl > thr]
    pool = b.off.mean()
    g = b.groupby("ap").off.agg(["sum", "size"])
    ap = ((g["sum"] + 10 * pool) / (g["size"] + 10)).to_dict()
    return pool, ap, len(b)
for st, (fp, evm, fitm, N) in FILES.items():
    P = pd.read_parquet(fp).set_index("MVT_ID_mvt").pred.astype(float)
    E = T.reindex(P.index).dropna(subset=["dl"])
    E["f"] = P.reindex(E.index)
    fit = T[T.src.isin(fitm)]
    for thr in (5400, 7200, 9000):
        pool, apd, nfit = delta_tab(fit, thr)
        z = E[E.dl > thr].copy()
        for var in ("pool", "airport"):
            dlt = np.full(len(z), pool) if var == "pool" else z.ap.map(apd).fillna(pool).to_numpy()
            for lam in (0.5, 1.0):
                new = z.f + lam * (z.dl + dlt - z.f)
                new = np.clip(new, z.dl - 3606, z.dl + 3606).clip(lower=0)
                dse = (new - z.y) ** 2 - (z.f - z.y) ** 2
                dd = dse.groupby(z.day).sum().to_numpy()
                b = np.random.default_rng(0).integers(0, len(dd), (4000, len(dd)))
                print(f"{st} thr {thr} {var:7s} lam {lam}: delta_pool {pool:+.0f} (fit rows {nfit}), rows {len(z)}, dMSE {dse.sum()/N:+7.1f}, better {int((dse<0).sum())} worse {int((dse>0).sum())}, "
                      f"P(<0) {np.mean(dd[b].sum(1) < 0):.3f}, move {((new-z.f)**2).sum()/N:6.1f}")
