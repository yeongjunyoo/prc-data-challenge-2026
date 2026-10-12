"""F3 L4 final (coordinator 10-10 10:07, split (a)): clone (APFS, read-only use) from R17-A's public root ~/nova-adsb/prcm/r17a_sbx, where
R17-A fits L4's val and fwd under this root's names (R17-A 10-10 10:3x), everything the final root's L4S chains and the L4 c form read besides
the full learners F3 fits here:
  cache: val_ and fwd_ of the 15 L4 learners and their seed averages (research/r17_a/avg_l4.py), the L4 blends (m6r28xwincbrmfw4sv434[tmc],
         m6r28xskcbrmfw4sv434, m6r28xapcbrmfw4sv114), the scale2-0.75 correctors <T>adsbgeondctxv3x50ws75 seeds 0-2 and their seed average c3,
         and the CatBoost iteration files the full CatBoost learners read (iters_m6r28cb{win,,ap}.json, the val seed-0 runs; RealMLP's for record);
  preds: the c3 es files (val, fwd; the manifest step reads them);
  research/r17_a: avg_l4.py and l4_learn_r17a.sh (named in the L4 manifests' code list);
  research/r13_l/fitcache (R17-A's corrector fit cache, ctx_corr3 at --threads 4, L3's and L4's models; the full-setting correctors train on
         the val rows, so their fits load here) -> research/f3/fitcache_l4, and the same models -> research/f2/fitcache_4m_l4 (the four-month
         corrector's stage 1 loads from them).
Same rules as collect_l3.py: full caches are never copied, refuses to overwrite, sha16 of every file recorded. Writes research/f3/out/collect_l4.json.
    python -X utf8 research/f3/collect_l4.py        (cwd = the final root)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

P = Path.home() / "nova-adsb" / "prcm"
S, D = P / "r17a_sbx", P / "final_run"
FIT_CACHE = S / "research" / "r13_l" / "fitcache"
MID = "adsbgeondctxv3x50ws75"
LEGS = ([f"m6r28cbwin_s{k}" for k in range(3)] + [f"m6r28cb_s{k}" for k in range(3)] + ["m6r28cbap_s0"]
        + [f"m6r28rmwin_s{k}" for k in range(4)] + [f"m6r28rm_s{k}" for k in range(4)]
        + ["m6r28cbwinx3_s0", "m6r28cbx3_s0", "m6r28rmwinx4_s0", "m6r28rmx4_s0"])
BLENDS = ["m6r28xwincbrmfw4sv434_s0", "m6r28xwincbrmfw4sv434tmc_s0", "m6r28xskcbrmfw4sv434_s0", "m6r28xapcbrmfw4sv114_s0"]
CB = ["m6r28xwincbrmfw4sv434tmc", "m6r28xskcbrmfw4sv434", "m6r28xapcbrmfw4sv114"]


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
    for t in ("m6r28cbwin", "m6r28cb", "m6r28cbap", "m6r28rmwin", "m6r28rm"):
        todo.append((S / "cache" / f"iters_{t}.json", D / "cache" / f"iters_{t}.json"))
    for f in ("avg_l4.py", "l4_learn_r17a.sh"):  # named in the L4 manifests' code list (R17-A's val/fwd averager and learner runner)
        todo.append((S / "research" / "r17_a" / f, D / "research" / "r17_a" / f))
    for s, d in todo:
        assert s.exists() and not s.is_symlink(), f"missing or a link: {s}"
        assert not d.exists(), f"exists: {d}"
    rep = {"from": str(S), "copied": {}, "fitcache_l4": {}}
    for s, d in todo:
        h = sha16(s)
        subprocess.run(["cp", "-cp", str(s), str(d)], check=True)
        assert sha16(d) == h
        rep["copied"][str(d.relative_to(D))] = {"from": str(s), "sha16": h}
    models = sorted(FIT_CACHE.glob("*.txt"))
    assert models, f"no models in {FIT_CACHE}"
    for dst in (D / "research" / "f3" / "fitcache_l4", D / "research" / "f2" / "fitcache_4m_l4"):
        assert not dst.exists(), f"exists: {dst}"
        dst.mkdir(parents=True)
        for f in models:
            subprocess.run(["cp", "-cp", str(f), str(dst / f.name)], check=True)
        rep["fitcache_l4"][str(dst.relative_to(D))] = {"from": str(FIT_CACHE), "files": len(models)}
    (D / "research" / "f3" / "out" / "collect_l4.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(len(rep["copied"]), "files cloned;", rep["fitcache_l4"])


if __name__ == "__main__":
    main()
