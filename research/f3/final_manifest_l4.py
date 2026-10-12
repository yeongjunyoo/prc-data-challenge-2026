# SPDX-License-Identifier: GPL-3.0-only
"""F3 L4 final (10-10 10:07 and 10:5x): the build record of 602c, research/f3/final_manifest.py imported unchanged (its sha16 is
asserted) with one more profile, 602c = 598c's form on L4's chains (research/f3/final_chain_l4.py --cfg l4s75 as the default 600, then
research/f3/kit_l4c.sh family l4c). The record is written by final_manifest.py's own code as research/f3/out/final_build_manifest_602c.json;
its "what" line names final_manifest.py, and this wrapper's sha16 is in its code list (drivers).
    python -X utf8 research/f3/final_manifest_l4.py <602c spec> --name 602c [--pc <json>]        (cwd = the final root)
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN = "5524ea3f81b54858"  # research/f3/final_manifest.py as it wrote the 594c and 598c records (PC and Mac final_run, 10-10 10:50)
if hashlib.sha256((HERE / "final_manifest.py").read_bytes()).hexdigest()[:16] != PIN:
    raise SystemExit(f"research/f3/final_manifest.py is not {PIN}: stop")
sys.path.insert(0, str(HERE))
import final_manifest as fm  # noqa: E402

L4FJ = os.environ.get("KIT_L4FJ", "l4")
_p = dict(fm.PROFILES["598c"])
_p.update(
    ff="ff11", default="600", default_ff="ff10", logs="l4c_", kit="research/f3/kit_l4c.sh", kit_pins=f"research/f3/out/{L4FJ}_fj_sha16.txt",
    bases={"A": "m6r28xwincbrmfw4sv434tmc", "B": "m6r28xskcbrmfw4sv434", "C": "m6r28xapcbrmfw4sv114"},
    chains=["final_chain_l4_A.json", "final_chain_l4_B.json", "final_chain_l4_C.json"], c4m_cache="research/f2/fitcache_4m_l4",
    drivers=["research/f3/phase7.sh", "research/f3/l4_full_f3.sh", "research/f3/collect_l4.py", "research/f3/final_chain_l4.py",
             "research/f3/daily_candidate_l4.py", "research/f3/final_manifest_l4.py", "research/r17_a/avg_l4.py"]
            + (["research/f3/fj_blends_l4.py"] if L4FJ == "l4" else []),
    rebuild_of=("L4S on final_ranking as 600 (research/f3/final_chain_l4.py --cfg l4s75, which imports research/f3/final_chain.py 636d00637ab543bd "
                "unchanged: L3's LightGBM legs as in 596, L4's 15 CatBoost and RealMLP learners refit on feats_v28 in the full setting here with "
                "R17-A's commands (research/f3/l4_full_f3.sh), their seed averages with R17-A's research/r17_a/avg_l4.py, val and fwd of every stage "
                "R17-A's public-root files (research/f3/collect_l4.py), correctors at scale2 0.75 loaded from R17-A's fit cache, 570's tails), then "
                "598c's form with research/f3/kit_l4c.sh family l4c: Jan/Jul rows T 0.75 (600's corrector), Feb/Jun rows the four-month stage 2 L "
                "on L4's base with F2's fjF tables and "
                + ("L4's leave-Feb/Jun-out extras (F2's five fj learners refit on feats_v28, research/f3/fj_blends_l4.py)" if L4FJ == "l4" else
                   "598c's L3 leave-Feb/Jun-out extras (the cheap fallback, on our word)")
                + ", unmatched 0.75, matched 0.5, merged after cavg, then 570's tails and chainavg3 on the merged tag; the record is written by "
                  "research/f3/final_manifest_l4.py, which imports research/f3/final_manifest.py 5524ea3f81b54858 unchanged"),
    checks={"split_check (check finals: BbF against 600)": "research/f3/out/split_check_l4c.json",
            "corrector-stage check": "research/f3/out/split_check_l4c_corr.json",
            "twins A": "research/f3/out/twins_l4c_A.json", "twins B": "research/f3/out/twins_l4c_B.json", "twins C": "research/f3/out/twins_l4c_C.json",
            "iii_boundary (BbF against 600)": "research/f3/out/iii_boundary_l4c.json"},
    extra={**fm.PROFILES["598c"]["extra"], "l4_clones_from_r17a_sbx": "research/f3/out/collect_l4.json",
           "l4_learners_full_done": "research/f3/out/l4_learn_full.DONE",
           **({"l4_fj_blends": "research/f3/out/fj_blends_l4.json", "l4_learners_fj_done": "research/f3/out/l4_learn_fj.DONE"} if L4FJ == "l4" else {})})
fm.PROFILES["602c"] = _p

if __name__ == "__main__":
    if "--name" not in sys.argv or sys.argv[sys.argv.index("--name") + 1] != "602c":
        raise SystemExit(__doc__)
    fm.variant_main(sys.argv[1:])
