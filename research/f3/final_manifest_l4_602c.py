# SPDX-License-Identifier: GPL-3.0-only
"""F3 L4 (10-11 16:4x): 602c's build record, research/f3/out/final_build_manifest_602c.json, with research/f3/final_manifest_l4.py
0c60a7709413f711 and research/f3/final_manifest.py 5524ea3f81b54858 imported unchanged. final_manifest_l4.py lists phase7's DONE markers
(research/f3/out/l4_learn_full.DONE and l4_learn_fj.DONE, a date line each) among the records final_manifest.py reads as json, so its own run stops
on them (Mac final root 10-11 16:50: JSONDecodeError, nothing written). Here every such record that exists and is not json moves to the checks,
which final_manifest.py records by path and sha16 without reading them; a record named .json that does not parse stops the run. Nothing else
changes; this file joins the drivers, so its sha16 is in the record's code list.
    python -X utf8 research/f3/final_manifest_l4_602c.py <602c spec> --name 602c [--pc <json>]        (cwd = the final root)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN = "0c60a7709413f711"  # research/f3/final_manifest_l4.py as deploy7 placed it (PC and Mac final_run)
if hashlib.sha256((HERE / "final_manifest_l4.py").read_bytes()).hexdigest()[:16] != PIN:
    raise SystemExit(f"research/f3/final_manifest_l4.py is not {PIN}: stop")
sys.path.insert(0, str(HERE))
import final_manifest_l4 as w  # noqa: E402  (asserts final_manifest.py 5524ea3f81b54858 and adds the 602c profile)

P = w.fm.PROFILES["602c"]
moved = {}
for k, v in list(P["extra"].items()):
    p = Path(v)
    if not p.exists():
        continue
    try:
        json.loads(p.read_text(encoding="utf-8"))
        continue
    except ValueError as e:
        if v.endswith(".json"):
            raise SystemExit(f"the record {v} ({k}) is named .json and does not parse: {e}")
    moved[k] = v
for k, v in moved.items():
    del P["extra"][k]
    P["checks"][f"{k} (not json: path and sha16 only)"] = v
P["drivers"] = P["drivers"] + ["research/f3/final_manifest_l4_602c.py"]

if __name__ == "__main__":
    if "--name" not in sys.argv or sys.argv[sys.argv.index("--name") + 1] != "602c":
        raise SystemExit(__doc__)
    print(f"records moved to the checks (not json): {moved}", flush=True)
    w.fm.variant_main(sys.argv[1:])
