# SPDX-License-Identifier: GPL-3.0-only
"""R21 d1: antiwin R2 rows at LIRF (NM-unmatched departure, linked arrival gives NM LOBT, 0 < MVT - LOBT <= 3606).
Normal BLOCK would have matched NM unless taxi > d_lobt + 3606, so the row is art (BLOCK = SCHED, only if |SCHED - LOBT| > 3606),
shift, or a long taxi. R2 today: pred = max(pred, d_lobt + 3606 + m). Candidate R2a: where art is possible, pred = pi d + (1 - pi) max(pred, lb),
pi = out-of-month art share among such rows (2025 labels). Chain values: lomo P (research/r21/c0_gbase.parquet) mixture with normal 1100,
then the R2 floor (m = out-of-month median, as tools/antiwin.constants)."""
import sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
L = pd.read_parquet("cache/antiwin_link_v1.parquet")
B = pd.read_parquet("research/r21/c0_gbase.parquet")[["MVT_ID_mvt", "pn1", "pa1", "ps1", "dd", "s"]]
X = L.merge(B, on="MVT_ID_mvt", how="left")
X = X[X.r2 & (X.ADEP_mvt == "LIRF")].copy()
X["art_ok"] = X.sch_m_lobt.abs() > 3606
X["art"] = (np.abs(X.y - X.d_sch) <= 60)
X["lb0"] = X.d_lobt + 3606
tr = X[X.src > 0].copy()
print("LIRF R2 rows 2025:", len(tr), " art possible:", int(tr.art_ok.sum()), " art among possible:", int((tr.art & tr.art_ok).sum()))
vs = 86400 * np.maximum(tr.dd, 1) + 1000
mix = tr.pn1 * 1100 + tr.pa1 * tr.d_sch + tr.ps1 * vs
res = []
for i, r in tr.iterrows():
    oth = tr[tr.src != r.src]
    m = float(np.median((oth.y - oth.lb0)))       # R2 floor offset, out of month (all R2 LIRF rows; antiwin uses LIRF+LTFM)
    po = oth[oth.art_ok]
    pi = (po.art.sum() + 0.5) / (len(po) + 1.0)
    base = max(mix.loc[i], r.lb0 + m)
    new = pi * r.d_sch + (1 - pi) * base if r.art_ok else base
    res.append((r.src, r.art_ok, r.y, r.d_sch, base, new, pi, r.pa1))
R = pd.DataFrame(res, columns=["src", "art_ok", "y", "d", "base", "new", "pi", "pa_chain"])
R["dse"] = (R.new - R.y) ** 2 - (R.base - R.y) ** 2
SC = 2 / 12 / 344419
print(R.round(2).to_string())
ms = R.groupby("src").dse.sum()
b = np.random.default_rng(0).integers(0, len(ms), (5000, len(ms)))
print(f"dSSE*2/12/N {R.dse.sum()*SC:+.1f}  (rows better {int((R.dse<0).sum())}, worse {int((R.dse>0).sum())}), month-boot P(<0) {np.mean(ms.to_numpy()[b].sum(1) < 0):.3f}")
print("without the 87,168 shift row:", round(R[R.y < 60000].dse.sum() * SC, 1))
