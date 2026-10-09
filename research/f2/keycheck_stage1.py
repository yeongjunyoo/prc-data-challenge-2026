# SPDX-License-Identifier: GPL-3.0-only
"""F2: does the corrector's stage 1 see the same matrix with F3's final tables? (10-09 02:1x, independent of F3's geo_final.py)
Builds the stage-1 design of a val base (src/ctx_corr.stage1_inputs + src/adsb_corr.design, the code the corrector runs) twice, once with this
root's adsb_v1nd/geo_v1nd and once with the final root's (module CACHE pointed there for the load only, read-only), and compares the bytes of
X[BASE_COLS + ADSB_COLS], y and pred, and the src/ctx_corr2 fit-cache digest of the whole matrix (every stage-1 key is a digest of a row subset).
    python -X utf8 research/f2/keycheck_stage1.py <final root> <adsb sha new> <geo sha new> [<val base tag>]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import src.adsb_corr as ac  # noqa: E402
import src.ctx_corr as cc  # noqa: E402
import src.ctx_corr2 as c2  # noqa: E402
from src.load import CACHE  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def design(cache: Path, adsb_sha: str, geo_sha: str, V: pd.DataFrame):
    cols0 = list(ac.ADSB_COLS)
    ac.CACHE, cc.CACHE = cache, cache
    try:
        A = cc.stage1_inputs(SimpleNamespace(adsb="adsb_v1nd", adsb_sha=adsb_sha, geo="geo_v1nd", geo_sha=geo_sha))
        cols = ac.BASE_COLS + ac.ADSB_COLS
        T = ac.design(V, A)
    finally:
        ac.CACHE, cc.CACHE = CACHE, CACHE
        ac.ADSB_COLS = cols0
    T = T[np.isfinite(T["y"].to_numpy())]
    return T, cols


def main() -> int:
    root, sha_a, sha_g = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    base = sys.argv[4] if len(sys.argv) > 4 else "m6r27wincbrmfw4sv434tmc"
    V = pd.read_parquet(CACHE / f"val_{base}_s0.parquet")
    T0, cols0 = design(CACHE, "0d03ff06267e0dee", "43310b9bbeefe81b", V)
    T1, cols1 = design(root / "cache", sha_a, sha_g, V)
    code = [hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16] for f in ("src/adsb_corr.py", "src/ctx_corr.py")]
    out = {"base": base, "rows": [int(len(T0)), int(len(T1))], "cols_same": cols0 == cols1, "n_cols": len(cols0)}
    if out["cols_same"] and len(T0) == len(T1):
        x0, x1 = T0[cols0].astype(float).to_numpy(), T1[cols1].astype(float).to_numpy()
        out["pos_same"] = bool(np.array_equal(T0["pos"].to_numpy(), T1["pos"].to_numpy()))
        out["X_bytes_same"] = x0.tobytes() == x1.tobytes()
        out["y_pred_bytes_same"] = (T0["y"].to_numpy(float).tobytes() == T1["y"].to_numpy(float).tobytes()
                                    and T0["pred"].to_numpy(float).tobytes() == T1["pred"].to_numpy(float).tobytes())
        k = [c2.digest(code, cols, 0, 300, 2, x, T["y"].to_numpy(float), T["pred"].to_numpy(float)) for T, cols, x in ((T0, cols0, x0), (T1, cols1, x1))]
        out["digest_today"], out["digest_final"] = k
        out["digest_same"] = k[0] == k[1]
        if not out["X_bytes_same"]:
            out["cols_differing"] = [c for i, c in enumerate(cols0) if x0[:, i].tobytes() != x1[:, i].tobytes()]
    out["verdict"] = "SAME" if out.get("digest_same") and out.get("X_bytes_same") and out.get("pos_same") else "DIFFERS"
    print(json.dumps(out, ensure_ascii=False))
    (Path("research/f2/out") / f"keycheck_stage1_{root.name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if out["verdict"] == "SAME" else 1


if __name__ == "__main__":
    sys.exit(main())
