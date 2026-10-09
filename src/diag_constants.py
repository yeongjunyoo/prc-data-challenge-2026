# SPDX-License-Identifier: GPL-3.0-only
# One-off diagnostic of 2026-10-02 (docs/DESIGN.md section 9): the local side of the two constant submissions v2 (0) and v3 (1000).
# On the 2025 January and July departures it computes the mean, the variance and the RMSE of the two constants, to compare with the
# mean and variance that the two server scores give (RMSE^2 = Var(y) + (mean(y) - c)^2 for a constant c). No per-row label is obtained.
import glob
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
cols = ["PHASE_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
tr = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/training_*.parquet"))], ignore_index=True)
tr = tr[tr["PHASE_mvt"] == "DEP"]
y = tr.loc[tr["MVT_TIME_UTC_mvt"].dt.month.isin([1, 7]), "TAXITIME_SEC_mvt"].astype(float).to_numpy()
C1, C2 = 0.0, 1000.0  # 두 상수(0 은 E[y^2] 를 바로 준다)
out = {"holdout_rows": int(len(y)), "holdout_mean": float(y.mean()), "holdout_var": float(y.var()),
       "local_rmse_c0": float(np.sqrt(np.mean((y - C1) ** 2))), "local_rmse_c1000": float(np.sqrt(np.mean((y - C2) ** 2))),
       "holdout_max": float(y.max()), "share_over_3600": float((y > 3600).mean())}
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open("notes/diag_local_constants.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
