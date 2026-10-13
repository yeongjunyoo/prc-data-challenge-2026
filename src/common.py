# SPDX-License-Identifier: GPL-3.0-only
"""Shared helpers of the models: the feature list, the row split of the ruler, the RMSE and a breakdown of the error.

Original note (Korean):
모델 공용: 피처 목록, 분할, RMSE 와 분해 표.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VAL_MONTHS = (1, 7)
DROP = {"MVT_ID_mvt", "src", "y", "sec_sch", "sec_aobt", "flight"}


def feature_cols(F: pd.DataFrame, extra_drop: set | None = None) -> list[str]:
    """Feature columns of F: every column except the identifiers and helper columns in DROP and in extra_drop."""
    drop = DROP | (extra_drop or set())
    return [c for c in F.columns if c not in drop]


def split(F: pd.DataFrame):
    """Masks (train, validation, test) by the column src: validation = the ruler months VAL_MONTHS (January and July), test = src 0."""
    va = F["src"].isin(VAL_MONTHS).to_numpy()
    te = (F["src"] == 0).to_numpy()
    tr = ~va & ~te
    return tr, va, te


def rmse(p, y) -> float:
    """Root mean square error of the predictions p against the labels y (seconds)."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def breakdown(p, y, ap, label: str = "") -> str:
    """Text summary of the error of p against y: RMSE, share of the squared error by label range, RMSE by airport (ap), RMSE below 3,600 s."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    se = (p - y) ** 2
    n = len(y)
    lines = [f"[{label}] RMSE {np.sqrt(se.mean()):.2f}  (n={n})"]
    bins = [-1e9, 1800, 3600, 7200, 80000, 1e12]
    b = pd.cut(y, bins)
    g = pd.DataFrame({"b": b, "se": se}).groupby("b", observed=True)["se"].agg(["size", "sum"])
    parts = []
    for k, r in g.iterrows():
        parts.append(f"{k.left:.0f}~{k.right:.0f}: n={int(r['size'])} mse_share={r['sum'] / se.sum():.3f}")
    lines.append("  by y: " + " | ".join(parts))
    ga = pd.DataFrame({"ap": np.asarray(ap), "se": se}).groupby("ap", observed=True)["se"].agg(["size", "sum", "mean"])
    lines.append("  by airport rmse: " + " ".join(f"{k}:{np.sqrt(r['mean']):.0f}" for k, r in ga.iterrows()))
    core = y < 3600
    lines.append(f"  RMSE on y<3600 rows: {np.sqrt(se[core].mean()):.2f} | MSE share of y>=3600: {se[~core].sum() / se.sum():.3f}")
    return "\n".join(lines)
