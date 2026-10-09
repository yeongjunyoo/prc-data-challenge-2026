# SPDX-License-Identifier: GPL-3.0-only
"""로컬 자 위 오라클 절제: 검증 예측 파일 하나에서 특정 행들을 정답으로 바꿨을 때의 RMSE.

사용: python src/ablate.py val_m3_s0.parquet pred
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
f, col = sys.argv[1], sys.argv[2]
v = pd.read_parquet(ROOT / "cache" / f)
F = pd.read_parquet(ROOT / "cache" / "feats.parquet", columns=["MVT_ID_mvt", "ap", "nm"])
v = v.merge(F, on="MVT_ID_mvt", how="left")
y = v["y"].to_numpy()
p = v[col].to_numpy()
se = (p - y) ** 2
n = len(y)
print(f"{f} {col}: RMSE {np.sqrt(se.mean()):.2f} MSE {se.mean():,.0f}")


def perfect(mask, label):
    s = se.copy()
    s[mask] = 0.0
    print(f"  if {label} were perfect ({int(mask.sum())} rows): RMSE {np.sqrt(s.mean()):.2f} | their MSE contrib {se[mask].sum() / n:,.0f}")


order = np.argsort(-se)
for k in [2, 5, 10, 26, 50, 100, 500]:
    m = np.zeros(n, bool)
    m[order[:k]] = True
    perfect(m, f"top-{k} squared-error rows")
perfect(y >= 80000, "y>=80000 rows")
perfect(y >= 20000, "y>=20000 rows")
perfect(y >= 3600, "y>=3600 rows")
noise = (y > 50000) & (v["dsch"].to_numpy() < 20000)
perfect(noise, "no-signal rows (y>50000, MVT-SCHED<20000)")
g = v["g"].to_numpy() == 1
perfect(g, "G rows (LIRF, NM unmatched)")
core = y < 3600
print("  core (y<3600) RMSE by airport:", {a: round(float(np.sqrt(se[core & (v['ap'].astype(str).to_numpy() == a)].mean())))
                                         for a in sorted(v["ap"].astype(str).unique())})
print("  all-row RMSE by airport:", {a: round(float(np.sqrt(se[v['ap'].astype(str).to_numpy() == a].mean())))
                                    for a in sorted(v["ap"].astype(str).unique())})
print("  MSE contrib by airport:", {a: round(float(se[v['ap'].astype(str).to_numpy() == a].sum() / n))
                                   for a in sorted(v["ap"].astype(str).unique())})
