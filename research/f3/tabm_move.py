# SPDX-License-Identifier: GPL-3.0-only
"""F3: the move between the two full TabM legs of the final rebuild (coordinator 10-09 03:4x): CUDA (PC GPU, 03:13-03:26) against MPS (Mac,
514's device and command), on mu_tabm (the leg's own regression, what chain A's blend_add takes) and pred, by month of the final file and by
airport group. Report research/f3/out/tabm_move.json.
    python -X utf8 research/f3/tabm_move.py <cuda parquet> <mps parquet>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MINRES = ["EDDM", "EGLL", "LFPG", "LIRF", "LTFM"]


def main():
    a = pd.read_parquet(sys.argv[1]).set_index("MVT_ID_mvt")
    b = pd.read_parquet(sys.argv[2]).set_index("MVT_ID_mvt").reindex(a.index)
    A = pd.read_parquet("cache/all.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "t_mvt", "ADEP_mvt"])
    A = A[A["PHASE_mvt"] == "DEP"].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").reindex(a.index)
    month = pd.to_datetime(A["t_mvt"].to_numpy(), unit="s").month
    apv = A["ADEP_mvt"].astype(str).to_numpy()
    rep = {"rows": int(len(a))}
    for col in ("mu_tabm", "pred"):
        d = a[col].to_numpy(float) - b[col].to_numpy(float)
        r = {"all": {"mse": round(float(np.mean(d ** 2)), 3), "mean": round(float(np.mean(d)), 3), "max_abs": round(float(np.max(np.abs(d))), 1)}}
        for mo in sorted(set(month)):
            m = month == mo
            r[f"month_{mo:02d}"] = {"rows": int(m.sum()), "mse": round(float(np.mean(d[m] ** 2)), 3), "mean": round(float(np.mean(d[m])), 3)}
        for g, m in (("minute_airports", np.isin(apv, MINRES)), ("second_airports", ~np.isin(apv, MINRES))):
            r[g] = {"mse": round(float(np.mean(d[m] ** 2)), 3)}
        rep[col] = r
    Path("research/f3/out/tabm_move.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
