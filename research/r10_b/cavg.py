# SPDX-License-Identifier: GPL-3.0-only
"""R10-B S3 (vault note 7.12): corrector seed average as one new corrector tag, so every later tool sees one consistent seed-0 cache file.
Reads cache/<setting>_<src>_s<k>.parquet for the given seeds (src/ctx_corr3.py outputs on the same base and arguments, --seed k), writes
cache/<setting>_<tag>_s0.parquet = the seed-0 file with pred replaced by the mean of pred over the seeds, floored at 0. The other columns (mu,
p_art, g, dsch, y) must be equal across the seed files (checked; mu, g, dsch, y are the base's, p_art is the corrector's and does not depend on
its seed). The seed-0 early-stopping ids are copied to the new tag
(ctx_corr copies the base's ids to every corrector seed, so they are equal). New names only: refuses if a target exists.
--base/--base_seed name the blend the corrector ran on (the later tools read them from the manifest config: artmm and arthist require --base,
aiu_shift loads that base); checked: mu, g, dsch (and y) of every seed file equal cache/<setting>_<base>_s<base_seed>.parquet.
    .venv/Scripts/python.exe -u research/r10_b/cavg.py --setting <val|fwd|full> --seed 0 --tag <new tag> --src <corrector tag> --seeds 0,1,2
        --base <blend tag> --base_seed 0
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--seed", type=int, default=0, choices=[0])
    pa.add_argument("--tag", required=True)
    pa.add_argument("--src", required=True)
    pa.add_argument("--seeds", required=True)
    pa.add_argument("--base", required=True)
    pa.add_argument("--base_seed", type=int, default=0)
    a = pa.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    assert seeds[0] == 0 and len(seeds) > 1 and a.tag != a.src
    out = ROOT / "cache" / f"{a.setting}_{a.tag}_s0.parquet"
    es_src = ROOT / "preds" / f"es_valid_ids_{a.src}_{a.setting}_s0.parquet"
    es_dst = ROOT / "preds" / f"es_valid_ids_{a.tag}_{a.setting}_s0.parquet"
    for p in (out, es_dst) if a.setting != "full" else (out,):
        if p.exists():
            raise SystemExit(f"exists, refusing to overwrite: {p}")
    fr = [pd.read_parquet(ROOT / "cache" / f"{a.setting}_{a.src}_s{k}.parquet") for k in seeds]
    b0 = fr[0]
    for k, b in zip(seeds[1:], fr[1:]):
        assert np.array_equal(b["MVT_ID_mvt"].to_numpy(), b0["MVT_ID_mvt"].to_numpy()), f"seed {k}: row order differs"
        for c in b0.columns:
            if c in ("MVT_ID_mvt", "pred"):
                continue
            x, z = b0[c].to_numpy(), b[c].to_numpy()
            same = np.array_equal(x, z) or (x.dtype.kind == "f" and np.array_equal(np.isnan(x), np.isnan(z)) and np.array_equal(x[~np.isnan(x)], z[~np.isnan(z)]))
            assert same, f"seed {k}: column {c} differs from seed 0"
    bb = pd.read_parquet(ROOT / "cache" / f"{a.setting}_{a.base}_s{a.base_seed}.parquet")
    assert np.array_equal(bb["MVT_ID_mvt"].to_numpy(), b0["MVT_ID_mvt"].to_numpy()), "base row order differs"
    for c in ("mu", "g", "dsch", "y"):
        if c in b0.columns:
            assert np.array_equal(b0[c].to_numpy(), bb[c].to_numpy(), equal_nan=b0[c].dtype.kind == "f"), f"column {c} differs from the base {a.base}"
    P = np.stack([b["pred"].to_numpy(dtype=float) for b in fr])
    res = b0.copy()
    res["pred"] = np.maximum(P.mean(axis=0), 0.0)
    sd = np.sqrt(np.mean((P - P.mean(axis=0)) ** 2))
    print(f"[{a.setting}] {a.src} seeds {seeds} -> {a.tag}: rows {len(res)}, pred spread across seeds RMS {sd:.2f}, "
          f"change vs seed 0 RMS {np.sqrt(np.mean((res['pred'].to_numpy() - P[0]) ** 2)):.2f}", flush=True)
    res.to_parquet(out, index=False)
    if a.setting != "full":
        for k in seeds[1:]:
            ek = pd.read_parquet(ROOT / "preds" / f"es_valid_ids_{a.src}_{a.setting}_s{k}.parquet")["MVT_ID_mvt"].to_numpy()
            e0 = pd.read_parquet(es_src)["MVT_ID_mvt"].to_numpy()
            assert np.array_equal(np.sort(ek), np.sort(e0)), f"seed {k}: early-stopping ids differ"
        shutil.copyfile(es_src, es_dst)
    print(f"saved {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
