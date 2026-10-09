"""F2 gate 1.4 inputs: the leave-Feb/Jun-out (setting fj) blends of 514's three chains with blend_fixed's full reference weights
(research/f2/blend_fixed_fj.py = src/blend_fixed.py with the setting fj added). XGBoost's slot gets a placeholder column (the base's mu) because
its reference weight is exactly 0 (src/blend_oof.simplex_ls leaves weights outside the support at 0); checked after each blend.
A (no TabM): f2skwing + f2cbwin, f2xg, f2rmwin;  B: f2skg + f2cb, f2xg, f2rm;  C: f2skapg + f2cbap, f2xg, f2rm.
    python -u research/f2/fj_blends.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
C, P, N = Path("cache"), Path("preds"), Path("notes")
REF = ["--ref_base", "m6r25sg", "--ref_tag", "m6r25sgoof4", "--ref_learners", "m6r25cb:mu_cat,m6r25xg:mu_xgb,m6r25rm:mu_rmlp", "--folds", "10"]
CHAINS = [("f2wincbrmfw4", "f2skwing", "f2cbwin:mu_cat,f2xg:mu_xgb,f2rmwin:mu_rmlp"),
          ("f2skcbrmfw4", "f2skg", "f2cb:mu_cat,f2xg:mu_xgb,f2rm:mu_rmlp"),
          ("f2apcbrmfw4", "f2skapg", "f2cbap:mu_cat,f2xg:mu_xgb,f2rm:mu_rmlp")]


def main():
    b = pd.read_parquet(C / "fj_f2sk_s0.parquet", columns=["MVT_ID_mvt", "mu"])
    ids = b["MVT_ID_mvt"].to_numpy()
    for t in ("f2skwing", "f2skg", "f2skapg", "f2cbwin", "f2cb", "f2cbap", "f2rmwin", "f2rm"):
        o = pd.read_parquet(C / f"fj_{t}_s0.parquet", columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()
        assert np.array_equal(o, ids), f"{t}: row order differs from f2sk"
    pd.DataFrame({"MVT_ID_mvt": ids, "mu_xgb": b["mu"].to_numpy(dtype=float)}).to_parquet(C / "fj_f2xg_s0.parquet", index=False)
    shutil.copyfile(P / "es_valid_ids_f2sk_fj_s0.parquet", P / "es_valid_ids_f2xg_fj_s0.parquet")
    print("placeholder fj_f2xg_s0 written (mu_xgb = f2sk mu)", flush=True)
    for tag, base, learners in CHAINS:
        cmd = [sys.executable, "-u", "research/f2/blend_fixed_fj.py", "--setting", "fj", "--seed", "0", "--tag", tag, "--base", base, "--learners", learners] + REF
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        w = json.loads((N / f"blend_fixed_{tag}_fj_s0.json").read_text(encoding="utf-8"))["weights"]
        # neutrality check: the same blend with the placeholder shifted by 1,000 s must give identical predictions
        pd.DataFrame({"MVT_ID_mvt": ids, "mu_xgb": b["mu"].to_numpy(dtype=float) + 1000.0}).to_parquet(C / "fj_f2xgchk_s0.parquet", index=False)
        shutil.copyfile(P / "es_valid_ids_f2sk_fj_s0.parquet", P / "es_valid_ids_f2xgchk_fj_s0.parquet")
        chk = tag + "chk"
        subprocess.run([sys.executable, "-u", "research/f2/blend_fixed_fj.py", "--setting", "fj", "--seed", "0", "--tag", chk, "--base", base,
                        "--learners", learners.replace("f2xg:", "f2xgchk:")] + REF, check=True, stdout=subprocess.DEVNULL)
        a1 = pd.read_parquet(C / f"fj_{tag}_s0.parquet")
        a2 = pd.read_parquet(C / f"fj_{chk}_s0.parquet")
        d = float(np.max(np.abs(a1["pred"].to_numpy() - a2["pred"].to_numpy())))
        for f in (C / f"fj_{chk}_s0.parquet", N / f"blend_fixed_{chk}_fj_s0.json", P / f"es_valid_ids_{chk}_fj_s0.parquet",
                  C / "fj_f2xgchk_s0.parquet", P / "es_valid_ids_f2xgchk_fj_s0.parquet"):
            f.unlink(missing_ok=True)
        assert d == 0.0, f"{tag}: the XGBoost placeholder moves the blend by {d} s, its weight is not exactly 0"
        print(f"{tag}: weights {w}; placeholder shifted by 1,000 s moves the blend by {d} s (neutral)", flush=True)


if __name__ == "__main__":
    main()
