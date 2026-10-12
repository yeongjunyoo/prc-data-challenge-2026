# SPDX-License-Identifier: GPL-3.0-only
"""F3 Package S, final root (10-11 01:17 and 01:2x; F2's job list 02:10, research/f2/s_spec.py): chain C at the package seed counts in
the T form, and 603. research/f3/final_chain.py (636d00637ab543bd) and, for the L4 family, research/f3/final_chain_l4.py (663cbae065e18622) are
imported unchanged (sha16 asserted); this file adds one configuration per family with chain C only:
  T_S, base and learners from F2's research/f2/s_spec.py chain_names(family, rm_seeds) (the s_spec.py and rm_seeds the sandbox exported,
  research/f3/out/collect_s_<family>.json; s_spec.py's sha16 is asserted against that record); gkey leg m6r28skapg, feats_v28; the manifest's
  extra code = the family chain C's list plus research/f2/avg_s.py (F2's rule 1 (d)); correctors at scale2 0.75 from research/f3/fitcache_s<family>
  (a fitted model stops the chain, final_chain.py's own check); val and fwd of every stage are the sandbox's (research/f3/collect_s.py).
  Queue numbers +30000 (598c) or +32000 (L4), step and state names prefixed s598c_ or sl4_ (research/f3/out/final_chain_s<family>_C.json).
603 = tools/chainavg3.py build <A default gja> <B default gja> <C S gja> --suffix skaps: A and B are the family default's own entries, untouched
(research/f3/out/final_chain_l3_<A|B>.json for 598c = 596's, final_chain_l4_<A|B>.json for L4 = 600's).
    python -u research/f3/final_chain_s.py <598c|l4> chain [--from <step>] [--to <step>]
    python -u research/f3/final_chain_s.py <598c|l4> avg3
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINS = {"final_chain.py": "636d00637ab543bd", "final_chain_l4.py": "663cbae065e18622"}
for f, h in PINS.items():
    if hashlib.sha256((HERE / f).read_bytes()).hexdigest()[:16] != h:
        raise SystemExit(f"research/f3/{f} is not {h}: stop")
sys.path.insert(0, str(HERE))
import final_chain as fc  # noqa: E402
import final_chain_l4 as fcl4  # noqa: E402

FAMS = {"598c": dict(cfg="s598c", dnum=30000, fam_pre="l3_", extra=fc.CHAINS_L3["C"]["extra"]),
        "l4": dict(cfg="sl4", dnum=32000, fam_pre="l4_", extra=fcl4.CHAINS_L4["C"]["extra"])}


def spec_module(fam: str):
    rec = json.loads((ROOT / "research" / "f3" / "out" / f"collect_s_{fam}.json").read_text(encoding="utf-8"))
    want = rec["tool_sha16"]["research/f2/s_spec.py"]
    have = hashlib.sha256((ROOT / "research" / "f2" / "s_spec.py").read_bytes()).hexdigest()[:16]
    if have != want:
        raise SystemExit(f"research/f2/s_spec.py is {have}, not the exported {want}: stop")
    sys.path.insert(0, str(ROOT / "research" / "f2"))
    import s_spec
    return s_spec, rec


def setup(fam: str) -> dict:
    if fam not in FAMS:
        raise SystemExit(__doc__)
    s_spec, rec = spec_module(fam)
    n = s_spec.chain_names(fam, rec["rm_seeds"])
    F, S, A = s_spec.fam(fam), n["S"], FAMS[fam]
    cfg = dict(mid="adsbgeondctxv3x50ws75", scale2="0.75", dnum=A["dnum"], pre=f"{A['cfg']}_", src_note="the S sandbox's val and fwd (research/f3/collect_s.py)",
               chains={"C": dict(T=S["T"], base_leg=S["base"], learners=S["learners"], TA=None, add_leg=None, gleg=F["GLEG"], num0=2200,
                                 extra=A["extra"] + ",research/f2/avg_s.py")},
               feats=F["feats"], fitcache=f"research/f3/fitcache_s{fam}", shared=False)
    fc.CFGS[A["cfg"]] = cfg
    fc.CFG = cfg
    return {"n": n, "F": F, "A": A, "s_spec": s_spec}


def avg3(fam: str, x: dict) -> None:
    n, F, A = x["n"], x["F"], x["A"]
    ends = {}
    for k, want in (("A", F["final_A"]["default"]), ("B", F["final_B"]["default"])):
        q = json.loads((fc.OUT / f"final_chain_{A['fam_pre']}{k}.json").read_text(encoding="utf-8"))["names"]["gja"]
        if not q.endswith("_" + want):
            raise SystemExit(f"chain {k}'s default gja entry {q} is not the family's {want}")
        ends[k] = q
    ends["C"] = json.loads((fc.OUT / f"final_chain_{fc.CFG['pre']}C.json").read_text(encoding="utf-8"))["names"]["gja"]
    spec = n["names"]["603"]
    fc.must_be_new(f"preds/manifest_{spec}.json")
    fc.run(f"{fc.CFG['pre']}avg3", "avg3", ["tools/chainavg3.py", "build", ends["A"], ends["B"], ends["C"], str(2400 + fc.CFG["dnum"]),
                                             "--suffix", x["s_spec"].SUFFIX, "--no-enqueue"])
    if not (fc.ROOT / "preds" / f"manifest_{spec}.json").exists():
        raise SystemExit(f"no manifest for 603 ({spec})")
    print(f"[{fc.ts()}] 603 = {spec} (A {ends['A']}, B {ends['B']}, C {ends['C']})", flush=True)


def main() -> int:
    a = sys.argv[1:]
    if len(a) < 2 or a[1] not in ("chain", "avg3"):
        raise SystemExit(__doc__)
    os.chdir(fc.ROOT)
    x = setup(a[0])
    if a[1] == "chain":
        fc.chain("C", a[a.index("--from") + 1] if "--from" in a else "blend", a[a.index("--to") + 1] if "--to" in a else "gja")
    else:
        avg3(a[0], x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
