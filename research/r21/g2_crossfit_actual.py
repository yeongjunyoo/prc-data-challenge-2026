# SPDX-License-Identifier: GPL-3.0-only
"""R21 g2: as g1 but with the grid I actually searched before registering W2 (T 5400, 7200, 9000; A none or -4000; lambda 0.5, 1.0; pool or airport).
R21 g1 text follows: fully cross-fitted W2 (requested 10-11 02:2x). Every constant that was chosen by looking at ruler results (T, A, lambda,
pooled vs airport delta) is re-selected inside nested folds; delta is re-estimated without the evaluated month (and without the selection month
when scoring a selection block). Fixed by data property, not tuned: the LOBT window +-3606 (|BLOCK - LOBT| <= 3606 on all 2,062,577 matched 2025
departures) and the floor 0.
Blocks with chain predictions: A = val form Jan, B = val form Jul, C = fwd form Jul, D = fj (gate19 Lm mean) Feb, E = fj Jun.
Selection set for an evaluated block = blocks of other months (July enters selection only through B).
Grid: T in {5400, 6300, 7200, 8100, 9000} x A in {none, -2400, -3200, -4000, -4800, -6000} x lambda in {0.25, 0.5, 0.75, 1.0} x delta {pool, airport k=10}.
Output: per block the selected variant and dSSE; assembled val (A+B), fwd (C), fj (D+E) dMSE with day bootstrap. fwd also with delta from months 1-6 only."""
import sys, itertools
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
cols = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "t_mvt", "t_lobt", "t_aobt3", "y", "src"]
T = pq.read_table("cache/all.parquet", columns=cols, filters=[("PHASE_mvt", "==", "DEP")]).to_pandas()
T = T[T.t_lobt.notna() & T.t_aobt3.notna() & (T.src > 0)].copy()
T["dl"] = T.t_mvt - T.t_lobt; T["gap"] = (T.t_mvt - T.t_aobt3) - T.dl; T["ap"] = T.ADEP_mvt.astype(str)
T["day"] = pd.to_datetime(T.t_mvt, unit="s").dt.strftime("%Y-%m-%d"); T["off"] = T.y - T.dl
T = T[T.dl > 5400].set_index("MVT_ID_mvt")          # every grid variant lives inside d_lobt > 5400
pv = pd.read_parquet("preds/holdout_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap_rmse281.37.parquet").set_index("MVT_ID_mvt").pred
pw = pd.read_parquet("preds/fwd_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.parquet").set_index("MVT_ID_mvt").pred
fj = [pd.read_parquet(f"research/f2/out_mac/gate19_{c}_F.parquet", columns=["MVT_ID_mvt", "Lm"]).set_index("MVT_ID_mvt").Lm for c in "ABC"]
pf = (fj[0] + fj[1] + fj[2]) / 3
BL = {"A": (pv, 1), "B": (pv, 7), "C": (pw, 7), "D": (pf, 2), "E": (pf, 6)}
SEL = {"A": "BDE", "B": "ADE", "C": "ADE", "D": "ABE", "E": "ABD"}
GRID = list(itertools.product([5400, 7200, 9000], [None, -4000], [0.5, 1.0], ["pool", "airport"]))
blk = {}
for k, (P, m) in BL.items():
    E = T[T.src == m].copy(); E["f"] = P.reindex(E.index); blk[k] = E[E.f.notna()]
def mask(E, t, a):
    mm = E.dl > t
    if a is not None: mm &= E.gap < a
    return mm
def delta(t, a, mode, excl_months, only_months=None):
    F = T[~T.src.isin(excl_months)] if only_months is None else T[T.src.isin(only_months)]
    F = F[mask(F, t, a)]
    pool = F.off.mean()
    if mode == "pool":
        return pool, None
    g = F.groupby("ap").off.agg(["sum", "size"])
    return pool, ((g["sum"] + 10 * pool) / (g["size"] + 10)).to_dict()
def dsse(E, t, a, lam, mode, excl, only=None):
    mm = mask(E, t, a); Z = E[mm]
    if len(Z) == 0: return 0.0, Z.assign(new=Z.f)
    pool, apd = delta(t, a, mode, excl, only)
    dl_ = np.full(len(Z), pool) if apd is None else Z.ap.map(apd).fillna(pool).to_numpy()
    new = np.clip(Z.f + lam * (Z.dl + dl_ - Z.f), Z.dl - 3606, Z.dl + 3606).clip(lower=0)
    return float(((new - Z.y) ** 2 - (Z.f - Z.y) ** 2).sum()), Z.assign(new=new)
res = {}
for X, (P, mX) in BL.items():
    best, bestv = None, None
    for v in GRID:
        t, a, lam, mode = v
        s = 0.0
        for Y in SEL[X]:
            mY = BL[Y][1]
            s += dsse(blk[Y], t, a, lam, mode, excl=[mX, mY])[0]
        if best is None or s < best:
            best, bestv = s, v
    t, a, lam, mode = bestv
    d_lomo, Zl = dsse(blk[X], t, a, lam, mode, excl=[mX])
    out = {"variant": bestv, "sel_dSSE": best, "dSSE": d_lomo, "rows": int(len(Zl)), "Z": Zl}
    if X == "C":
        d_fw, Zf = dsse(blk[X], t, a, lam, mode, excl=[], only=[1, 2, 3, 4, 5, 6])
        out["dSSE_fwdstrict"] = d_fw; out["Zfw"] = Zf
    res[X] = out
    print(f"block {X} (month {mX}): selected T {t}, A {a}, lambda {lam}, delta {mode}; selection dSSE {best/1e6:+.1f}e6; eval rows {len(Zl)}, dSSE {d_lomo/1e6:+.2f}e6"
          + (f"; forward-strict delta dSSE {out['dSSE_fwdstrict']/1e6:+.2f}e6" if X == "C" else ""))
days_all = pq.read_table("cache/all.parquet", columns=["t_mvt", "src", "PHASE_mvt"], filters=[("PHASE_mvt", "==", "DEP")]).to_pandas()
days_all["day"] = pd.to_datetime(days_all.t_mvt, unit="s").dt.strftime("%Y-%m-%d")
def gate(name, keys, N, months, zkey="Z"):
    Z = pd.concat([res[k][zkey] for k in keys])
    dse = (Z.new - Z.y) ** 2 - (Z.f - Z.y) ** 2
    days = days_all[days_all.src.isin(months)].day.unique()
    dd = dse.groupby(Z.day).sum().reindex(days).fillna(0).to_numpy()
    b = np.random.default_rng(0).integers(0, len(dd), (4000, len(dd)))
    print(f"  {name}: rows {len(Z)}, dMSE {dse.sum()/N:+7.1f}, better {int((dse<0).sum())} worse {int((dse>0).sum())}, day-boot P(<0) {np.mean(dd[b].sum(1) < 0):.3f}")
print("cross-fitted gate:")
gate("(b) val  A+B", "AB", 344419, [1, 7])
gate("(c) fwd  C  ", "C", 190713, [7])
gate("(c) fwd  C forward-strict delta", "C", 190713, [7], zkey="Zfw")
gate("    fj   D+E", "DE", 326873, [2, 6])
