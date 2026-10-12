# SPDX-License-Identifier: GPL-3.0-only
"""F2 Package S: chain C's leave-Feb/Jun-out (setting fj) blend at the package seed counts, the extra rows of chain C's four-month corrector in
605c (coordinator 10-11 01:5x: chain C's fj learners go to the package seed counts and its fj blend is rebuilt from them). The recipe of
research/f3/fj_blends_l3.py (598c, sha16 672a269e14591734) and research/f3/fj_blends_l4.py (L4, b4f5b1bad4e03c9a) for chain C only, with the
S averaged fj legs in place of the seed-0 ones: base = the LightGBM ap G-band leg averaged over seeds 0-2 (l3skapgx3s), CatBoost ap averaged
over seeds 0-2, RealMLP = chain C's own seeds 4-7 (x4s, or x3s if a seed was dropped); research/f2/blend_fixed_fj.py (591cae9a9ac19ec8,
unchanged) with blend_fixed's full reference weights; the XGBoost slot is the family's existing placeholder (598c l3xg, L4 l4xg, written by
the scripts above), checked neutral by shifting it 1,000 s. Writes cache/fj_<tag>_s0.parquet and research/f2/out/s_fj_sha16_<family>.txt
(the pin the S corr4m step checks).
    python -X utf8 -u research/f2/fj_blend_s.py <598c|l4> [--rm_tag <tag>]       (cwd = the final root)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s_spec  # noqa: E402

C, P, N = Path("cache"), Path("preds"), Path("notes")
REF = s_spec.REF.split()


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main() -> int:
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    fam = a[0]
    F = s_spec.fam(fam)
    S = s_spec.chain_names(fam)["S"]["fj_blend"]
    tag, base, learners = S["tag"], S["base"], S["learners"]
    if "--rm_tag" in a:
        rt = a[a.index("--rm_tag") + 1]
        learners = ",".join((f"{rt}:mu_rmlp" if x.endswith(":mu_rmlp") else x) for x in learners.split(","))
    xg = F["fj_xg"]
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if (C / f"fj_{tag}_s0.parquet").exists():
        raise SystemExit(f"exists: fj_{tag}_s0")
    ids = pd.read_parquet(C / f"fj_{base}_s0.parquet", columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()
    for t in [x.split(":")[0] for x in learners.split(",")]:
        o = pd.read_parquet(C / f"fj_{t}_s0.parquet", columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()
        assert np.array_equal(o, ids), f"{t}: row order differs from {base}"
    cmd = [sys.executable, "-u", "research/f2/blend_fixed_fj.py", "--setting", "fj", "--seed", "0", "--tag", tag, "--base", base, "--learners", learners] + REF
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    w = json.loads((N / f"blend_fixed_{tag}_fj_s0.json").read_text(encoding="utf-8"))["weights"]
    chk_xg, chk = xg + "chks", tag + "chk"
    x = pd.read_parquet(C / f"fj_{xg}_s0.parquet")
    pd.DataFrame({"MVT_ID_mvt": x["MVT_ID_mvt"].to_numpy(), "mu_xgb": x["mu_xgb"].to_numpy(dtype=float) + 1000.0}).to_parquet(C / f"fj_{chk_xg}_s0.parquet", index=False)
    shutil.copyfile(P / f"es_valid_ids_{xg}_fj_s0.parquet", P / f"es_valid_ids_{chk_xg}_fj_s0.parquet")
    subprocess.run([sys.executable, "-u", "research/f2/blend_fixed_fj.py", "--setting", "fj", "--seed", "0", "--tag", chk, "--base", base,
                    "--learners", learners.replace(f"{xg}:", f"{chk_xg}:")] + REF, check=True, stdout=subprocess.DEVNULL)
    a1, a2 = pd.read_parquet(C / f"fj_{tag}_s0.parquet"), pd.read_parquet(C / f"fj_{chk}_s0.parquet")
    d = float(np.max(np.abs(a1["pred"].to_numpy() - a2["pred"].to_numpy())))
    for f in (C / f"fj_{chk}_s0.parquet", N / f"blend_fixed_{chk}_fj_s0.json", P / f"es_valid_ids_{chk}_fj_s0.parquet",
              C / f"fj_{chk_xg}_s0.parquet", P / f"es_valid_ids_{chk_xg}_fj_s0.parquet"):
        f.unlink(missing_ok=True)
    assert d == 0.0, f"{tag}: the XGBoost placeholder moves the blend by {d} s, its weight is not exactly 0"
    h = sha16(C / f"fj_{tag}_s0.parquet")
    out = Path("research/f2/out")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"s_fj_sha16_{fam}.txt").write_text(f"{h} {tag}\n", encoding="utf-8")
    (out / f"s_fj_blend_{fam}.json").write_text(json.dumps({"tag": tag, "base": base, "learners": learners, "weights": w, "placeholder_neutral": True,
                                                            "sha16": h, "rows": int(len(a1))}, indent=1), encoding="utf-8")
    print(f"{tag}: weights {w}; placeholder shifted by 1,000 s moves the blend by {d} s (neutral); sha16 {h}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
