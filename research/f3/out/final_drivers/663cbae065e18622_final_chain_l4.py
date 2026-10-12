"""F3 L4 final (coordinator 10-10 10:07, split (a); 10:5x: L4's versions in new files, the drivers that ran for 594c and 598c stay byte-identical):
research/f3/final_chain.py, imported unchanged (its sha16 is asserted), with one more configuration, l4s75: L3's chains (596's, --cfg l3s75) with
L4's CatBoost and RealMLP learners on feats_v28 (R17-A's names: m6r28cbwinx3, m6r28cbx3, m6r28cbap, m6r28rmwinx4, m6r28rmx4) in place of 514's;
the LightGBM base legs, TabM and the XGBoost placeholder as L3's. Blend, add, the correctors (scale2 0.75, loaded from R17-A's fit cache cloned
to research/f3/fitcache_l4; a fitted corrector stops its chain), cavg, the manifest, the tails and chainavg3 run as in final_chain.py: only
full is computed here, val and fwd of every stage are R17-A's public-root files (research/f3/collect_l4.py). Queue numbers +17000, step and
state names prefixed l4_ (research/f3/out/final_chain_l4_<X>.json).
    python -u research/f3/final_chain_l4.py chain <A|B|C> --cfg l4s75 [--from <step>] [--to <step>]
    python -u research/f3/final_chain_l4.py avg3 --cfg l4s75
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN = "636d00637ab543bd"  # research/f3/final_chain.py as it ran for 596, 598b and 598c (PC and Mac final_run, 10-10 10:50)
if hashlib.sha256((HERE / "final_chain.py").read_bytes()).hexdigest()[:16] != PIN:
    raise SystemExit(f"research/f3/final_chain.py is not {PIN}: stop")
sys.path.insert(0, str(HERE))
import final_chain as fc  # noqa: E402

EXTRA_L4 = (fc.EXTRA_L3 + ",research/r17_a/avg_l4.py,research/r17_a/l4_learn_r17a.sh,research/f3/l4_full_f3.sh,research/f3/collect_l4.py")
CHAINS_L4 = {
    "A": dict(T="m6r28xwincbrmfw4sv434", base_leg="m6r28skwingx4", learners="m6r28cbwinx3:mu_cat,m6r25xg:mu_xgb,m6r28rmwinx4:mu_rmlp",
              TA="m6r28xwincbrmfw4sv434tmc", add_leg="m6r27tmcwin:mu_tabm", gleg="m6r28skwing", num0=2000, extra=EXTRA_L4),
    "B": dict(T="m6r28xskcbrmfw4sv434", base_leg="m6r28skgx4", learners="m6r28cbx3:mu_cat,m6r25xg:mu_xgb,m6r28rmx4:mu_rmlp",
              TA=None, add_leg=None, gleg="m6r28skg", num0=2100, extra=EXTRA_L4),
    "C": dict(T="m6r28xapcbrmfw4sv114", base_leg="m6r28skapg", learners="m6r28cbap:mu_cat,m6r25xg:mu_xgb,m6r28rmx4:mu_rmlp",
              TA=None, add_leg=None, gleg="m6r28skapg", num0=2200, extra=EXTRA_L4 + ",research/r13_f3/model3ap.py,research/r13_f3/cat_muap.py"),
}
fc.CFGS["l4s75"] = dict(mid="adsbgeondctxv3x50ws75", scale2="0.75", dnum=17000, pre="l4_",
                        src_note="L4's val and fwd from R17-A's public root r17a_sbx", chains=CHAINS_L4, feats="feats_v28",
                        fitcache="research/f3/fitcache_l4", shared=False)

if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("chain", "avg3") or "--cfg" not in a or a[a.index("--cfg") + 1] != "l4s75":
        raise SystemExit(__doc__)
    sys.exit(fc.main())
