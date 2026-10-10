# SPDX-License-Identifier: GPL-3.0-only
"""F3 L3 final (10-10 03:5x): clone (APFS, read-only use) from r17a_sbx, where R17-A built L3 and L3S, everything the full-setting
rebuild of L3S (596) and the L3 four-month corrector read, and from f2_run the CatBoost and RealMLP leave-Feb/Jun-out outputs F2 made for 594's L.
Same rules as research/f3/collect_ro.py and collect_s75.py: full_ caches are never copied (every full output is rebuilt on the final data, so a
missed rebuild fails loudly); refuses to overwrite; sha16 of every file recorded.
  r17a_sbx cache: val_ and fwd_ of the LightGBM legs (m6r28sk s0-3, m6r28skwin s0-3, m6r28skap s0), their G bands (m6r28skg, m6r28skwing,
      m6r28skapg), the x4 averages (m6r28skgx4, m6r28skwingx4), the blends (m6r28wincbrmfw4sv434, m6r28wincbrmfw4sv434tmc, m6r28skcbrmfw4sv434,
      m6r28apcbrmfw4sv114), the scale2-0.75 correctors (seeds 0-2) and their seed average (c3); iters_m6r28{sk,skwin,skap}.json (full reads them);
  r17a_sbx preds: the c3 es files (val, fwd; the manifest step reads them);
  r17a_sbx research/r13_l/fitcache -> research/f3/fitcache_l3 (L3's corrector models: ctx_corr3 at --threads 4, stage 1 and 2);
  r17a_sbx research/r17_a/{avg_r17a,adsb_day_r17a}.py -> research/r17_a/ (named in L3's manifests' code list);
  f2_run cache fj_{f2cbwin,f2cb,f2cbap,f2rmwin,f2rm}_s0.parquet and preds es_valid_ids_<same>_fj_s0.parquet (F2's fj learners for the L3 fj blends).
Writes research/f3/out/collect_l3.json.
    python -X utf8 research/f3/collect_l3.py       (cwd = the final root)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

P = Path(__import__("os").environ.get("PRC_WORK", "."))  # Mac work folder (the build sandboxes and final_run)
S, F2 = P / "r17a_sbx", P / "f2_run"
D = P / "final_run"
LEGS = [f"m6r28sk_s{k}" for k in range(4)] + [f"m6r28skwin_s{k}" for k in range(4)] + ["m6r28skap_s0", "m6r28skg_s0", "m6r28skwing_s0",
                                                                                        "m6r28skapg_s0", "m6r28skgx4_s0", "m6r28skwingx4_s0"]
BLENDS = ["m6r28wincbrmfw4sv434_s0", "m6r28wincbrmfw4sv434tmc_s0", "m6r28skcbrmfw4sv434_s0", "m6r28apcbrmfw4sv114_s0"]
CB = ["m6r28wincbrmfw4sv434tmc", "m6r28skcbrmfw4sv434", "m6r28apcbrmfw4sv114"]
MID = "adsbgeondctxv3x50ws75"
FJ = ["f2cbwin", "f2cb", "f2cbap", "f2rmwin", "f2rm"]


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    todo = []  # (source path, destination path)
    for st in ("val", "fwd"):
        for t in LEGS + BLENDS:
            todo.append((S / "cache" / f"{st}_{t}.parquet", D / "cache" / f"{st}_{t}.parquet"))
        for c in CB:
            for k in (0, 1, 2):
                todo.append((S / "cache" / f"{st}_{c}{MID}_s{k}.parquet", D / "cache" / f"{st}_{c}{MID}_s{k}.parquet"))
            todo.append((S / "cache" / f"{st}_{c}{MID}c3_s0.parquet", D / "cache" / f"{st}_{c}{MID}c3_s0.parquet"))
            todo.append((S / "preds" / f"es_valid_ids_{c}{MID}c3_{st}_s0.parquet", D / "preds" / f"es_valid_ids_{c}{MID}c3_{st}_s0.parquet"))
    for t in ("m6r28sk", "m6r28skwin", "m6r28skap"):
        todo.append((S / "cache" / f"iters_{t}.json", D / "cache" / f"iters_{t}.json"))
    for f in ("avg_r17a.py", "adsb_day_r17a.py"):
        todo.append((S / "research" / "r17_a" / f, D / "research" / "r17_a" / f))
    for t in FJ:
        todo.append((F2 / "cache" / f"fj_{t}_s0.parquet", D / "cache" / f"fj_{t}_s0.parquet"))
        todo.append((F2 / "preds" / f"es_valid_ids_{t}_fj_s0.parquet", D / "preds" / f"es_valid_ids_{t}_fj_s0.parquet"))
    for s, d in todo:
        assert s.exists(), f"missing: {s}"
        assert not d.exists(), f"exists: {d}"
    rep = {"copied": {}, "fitcache_l3": {}}
    (D / "research" / "r17_a").mkdir(exist_ok=True)
    for s, d in todo:
        h = sha16(s)
        subprocess.run(["cp", "-cp", str(s), str(d)], check=True)
        assert sha16(d) == h
        rep["copied"][str(d.relative_to(D))] = {"from": str(s), "sha16": h}
    fc_s, fc_d = S / "research" / "r13_l" / "fitcache", D / "research" / "f3" / "fitcache_l3"
    assert not fc_d.exists(), f"exists: {fc_d}"
    fc_d.mkdir(parents=True)
    n = 0
    for f in sorted(fc_s.glob("*.txt")):
        subprocess.run(["cp", "-cp", str(f), str(fc_d / f.name)], check=True)
        n += 1
    rep["fitcache_l3"] = {"from": str(fc_s), "files": n}
    (D / "research" / "f3" / "out" / "collect_l3.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(len(rep["copied"]), "files cloned;", n, "corrector models in research/f3/fitcache_l3")


if __name__ == "__main__":
    main()
