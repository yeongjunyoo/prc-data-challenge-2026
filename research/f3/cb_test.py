# SPDX-License-Identifier: GPL-3.0-only
"""F3 fitwrap CatBoost check on synthetic data (seconds per fit): fits CatBoostRegressor on a Pool built from a DataFrame with three
integer-coded categorical columns (as src/cat_mu.py does), predicts a test frame and saves the predictions. Run directly and through
research/f3/fitwrap.py (save, load, strict); "perturb" changes one training value; the thread count is the second argument.
    python -X utf8 research/f3/cb_test.py <out.npy> <threads> [perturb]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

out, th = sys.argv[1], int(sys.argv[2])
perturb = len(sys.argv) > 3 and sys.argv[3] == "perturb"
rng = np.random.default_rng(20261008)
n, nt, m = 300000, 50000, 20
X = pd.DataFrame(rng.normal(size=(n + nt, m)).astype(np.float32), columns=[f"x{j}" for j in range(m)])
X.iloc[rng.integers(0, n + nt, 20000), 3] = np.nan
X["c0"] = rng.integers(0, 50, n + nt)
X["c1"] = rng.integers(-1, 400, n + nt)
X["c2"] = rng.integers(0, 3000, n + nt)
y = (X["x0"] * 3 + np.sin(X["x1"]) * 2 + (X["c0"] % 7) * 0.5 + (X["c2"] % 11) * 0.2 + rng.normal(size=n + nt)).to_numpy()
tr, te = X.iloc[:n].copy(), X.iloc[n:].copy()
if perturb:
    tr.iloc[123, 2] = tr.iloc[123, 2] + np.float32(1)
cats = ["c0", "c1", "c2"]
mdl = CatBoostRegressor(iterations=400, depth=8, learning_rate=0.08, thread_count=th, random_seed=0, allow_writing_files=False, verbose=0)
mdl.fit(Pool(tr, y[:n], cat_features=cats))
p = mdl.predict(Pool(te, cat_features=cats))
np.save(out, p)
print(out, th, perturb, float(p.mean()), flush=True)
