"""F2 Package S: the one-minute RealMLP smoke fit that research/f2/s_learn.py runs before chain C's own RealMLP seeds (coordinator 10-11
01:17: OMP_NUM_THREADS=1, faulthandler, a 1-minute smoke fit first). It builds RealMLP_TD_Regressor exactly as src/rmlp_mu.py does (pytabkit,
n_cv=1, batch 2048, the device and thread count given, a categorical column) and fits two epochs on 40,000 synthetic rows with an
early-stopping split, then predicts. It touches no cache file. A crash of the MPS/OpenMP stack (the 10-07 night: seeds 4 and 5 died without
OMP_NUM_THREADS=1) shows here as a signal or a faulthandler trace within the minute instead of after a 15-minute fit.
    OMP_NUM_THREADS=1 PYTHONFAULTHANDLER=1 python -X utf8 -u research/f2/rm_smoke.py --device mps --threads 1 [--rows 40000 --epochs 2]
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import time

import numpy as np
import pandas as pd


def main() -> int:
    faulthandler.enable()
    pa = argparse.ArgumentParser()
    pa.add_argument("--device", default="mps")
    pa.add_argument("--threads", type=int, default=1)
    pa.add_argument("--rows", type=int, default=40000)
    pa.add_argument("--epochs", type=int, default=2)
    a = pa.parse_args()
    t0 = time.time()
    print(f"rm_smoke: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} PYTHONFAULTHANDLER={os.environ.get('PYTHONFAULTHANDLER')} "
          f"device {a.device} threads {a.threads}", flush=True)
    from pytabkit import RealMLP_TD_Regressor
    rng = np.random.default_rng(0)
    n = a.rows
    X = pd.DataFrame(rng.normal(size=(n, 40)).astype(np.float32), columns=[f"x{i}" for i in range(40)])
    X["ap"] = pd.Categorical(rng.integers(0, 10, n).astype(str))
    y = (X["x0"] * 3 + X["x1"] ** 2 + rng.normal(size=n)).astype(np.float32).to_numpy()
    tr = np.arange(n) % 10 != 0
    m = RealMLP_TD_Regressor(device=a.device, n_cv=1, batch_size=2048, n_epochs=a.epochs, random_state=0, verbosity=0,
                             **({"n_threads": a.threads} if a.threads else {}))
    m.fit(X[tr], y[tr], X[~tr], y[~tr], cat_col_names=["ap"])
    p = m.predict(X[~tr])
    r = float(np.sqrt(np.mean((p - y[~tr]) ** 2)))
    if not np.isfinite(r):
        print("rm_smoke: non-finite predictions", flush=True)
        return 2
    print(f"rm_smoke OK: rmse {r:.3f} on {int((~tr).sum())} rows in {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
