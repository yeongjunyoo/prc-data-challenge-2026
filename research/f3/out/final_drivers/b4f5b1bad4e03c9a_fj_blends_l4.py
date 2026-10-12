"""F3 L4 final (coordinator 10-10 10:2x, the consistent c form): the leave-Feb/Jun-out (setting fj) blends of L4's three chains, the extra rows
of the L4 four-month corrector. research/f3/fj_blends_l3.py with L4's CatBoost and RealMLP fj learners in place of F2's feats_v27 ones: the base
legs are L3's seed-0 fj legs with their G band (l3skwing, l3skg, l3skapg: the LightGBM part, feats_v28), the CatBoost and RealMLP fj learners
are F2's five refit on feats_v28 here (l4cbwin, l4cb, l4cbap, l4rmwin, l4rm: research/f3/l4_full_f3.sh fj, F2's cat_mu_fj.py, cat_muap_fj.py,
rmlp_mu_fj.py), the weights are blend_fixed's full reference weights (research/f2/blend_fixed_fj.py, unchanged), and the XGBoost slot is a
placeholder (the l3sk mu) checked neutral by shifting it 1,000 s.
A (no TabM, as F2's): l3skwing + l4cbwin, l4xg, l4rmwin;  B: l3skg + l4cb, l4xg, l4rm;  C: l3skapg + l4cbap, l4xg, l4rm.
Writes cache/fj_{l4wincbrmfw4,l4skcbrmfw4,l4apcbrmfw4}_s0.parquet and research/f3/out/l4_fj_sha16.txt (the pins research/f3/kit_l4c.sh checks).
    python -X utf8 -u research/f3/fj_blends_l4.py       (cwd = the final root)
"""
from __future__ import annotations

import hashlib
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
CHAINS = [("l4wincbrmfw4", "l3skwing", "l4cbwin:mu_cat,l4xg:mu_xgb,l4rmwin:mu_rmlp"),
          ("l4skcbrmfw4", "l3skg", "l4cb:mu_cat,l4xg:mu_xgb,l4rm:mu_rmlp"),
          ("l4apcbrmfw4", "l3skapg", "l4cbap:mu_cat,l4xg:mu_xgb,l4rm:mu_rmlp")]


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    for tag, _, _ in CHAINS:
        assert not (C / f"fj_{tag}_s0.parquet").exists(), f"exists: fj_{tag}_s0"
    b = pd.read_parquet(C / "fj_l3sk_s0.parquet", columns=["MVT_ID_mvt", "mu"])
    ids = b["MVT_ID_mvt"].to_numpy()
    for t in ("l3skwing", "l3skg", "l3skapg", "l4cbwin", "l4cb", "l4cbap", "l4rmwin", "l4rm"):
        o = pd.read_parquet(C / f"fj_{t}_s0.parquet", columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()
        assert np.array_equal(o, ids), f"{t}: row order differs from l3sk"
    pd.DataFrame({"MVT_ID_mvt": ids, "mu_xgb": b["mu"].to_numpy(dtype=float)}).to_parquet(C / "fj_l4xg_s0.parquet", index=False)
    shutil.copyfile(P / "es_valid_ids_l3sk_fj_s0.parquet", P / "es_valid_ids_l4xg_fj_s0.parquet")
    print("placeholder fj_l4xg_s0 written (mu_xgb = l3sk mu)", flush=True)
    pins, rep = [], {}
    for tag, base, learners in CHAINS:
        cmd = [sys.executable, "-u", "research/f2/blend_fixed_fj.py", "--setting", "fj", "--seed", "0", "--tag", tag, "--base", base, "--learners", learners] + REF
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        w = json.loads((N / f"blend_fixed_{tag}_fj_s0.json").read_text(encoding="utf-8"))["weights"]
        pd.DataFrame({"MVT_ID_mvt": ids, "mu_xgb": b["mu"].to_numpy(dtype=float) + 1000.0}).to_parquet(C / "fj_l4xgchk_s0.parquet", index=False)
        shutil.copyfile(P / "es_valid_ids_l3sk_fj_s0.parquet", P / "es_valid_ids_l4xgchk_fj_s0.parquet")
        chk = tag + "chk"
        subprocess.run([sys.executable, "-u", "research/f2/blend_fixed_fj.py", "--setting", "fj", "--seed", "0", "--tag", chk, "--base", base,
                        "--learners", learners.replace("l4xg:", "l4xgchk:")] + REF, check=True, stdout=subprocess.DEVNULL)
        a1 = pd.read_parquet(C / f"fj_{tag}_s0.parquet")
        a2 = pd.read_parquet(C / f"fj_{chk}_s0.parquet")
        d = float(np.max(np.abs(a1["pred"].to_numpy() - a2["pred"].to_numpy())))
        for f in (C / f"fj_{chk}_s0.parquet", N / f"blend_fixed_{chk}_fj_s0.json", P / f"es_valid_ids_{chk}_fj_s0.parquet",
                  C / "fj_l4xgchk_s0.parquet", P / "es_valid_ids_l4xgchk_fj_s0.parquet"):
            f.unlink(missing_ok=True)
        assert d == 0.0, f"{tag}: the XGBoost placeholder moves the blend by {d} s, its weight is not exactly 0"
        h = sha16(C / f"fj_{tag}_s0.parquet")
        pins.append(f"{h} {tag}")
        rep[tag] = {"weights": w, "placeholder_neutral": True, "sha16": h, "rows": int(len(a1))}
        print(f"{tag}: weights {w}; placeholder shifted by 1,000 s moves the blend by {d} s (neutral); sha16 {h}", flush=True)
    Path("research/f3/out/l4_fj_sha16.txt").write_text("\n".join(pins) + "\n", encoding="utf-8")
    Path("research/f3/out/fj_blends_l4.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
