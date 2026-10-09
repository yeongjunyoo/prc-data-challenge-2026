# SPDX-License-Identifier: GPL-3.0-only
"""R18-M1 decision (rules registered 2026-10-09 13:58 KST in the private note 2026-10-09-r18-latest-hits.md section 1), read from the screen report
research/r18/screens/<name>.json made with --ref r18tmc s0, cand0 = the HL-Gauss leg s0, cand1 = the MSE leg s1 (null).
    python research/r18/decide_m1.py [name]
"""
import json
import sys
from pathlib import Path

name = sys.argv[1] if len(sys.argv) > 1 else "m1"
r = json.loads((Path(__file__).parent / "screens" / f"{name}.json").read_text(encoding="utf-8"))
v, f, j = r["val"]["blend_d_cand0"]["all"], r["fwd"]["blend_d_cand0"]["all"], r["fj"]["leg_cand0"]["reg"]
null = r["val"]["blend_d_cand1"]["all"]
c = {
    "1 val blend all rows dMSE < 0 and P >= 0.9": (v["dmse"] < 0 and v["p_below0"] >= 0.9, v),
    "2 fwd blend all rows dMSE < 0": (f["dmse"] < 0, f),
    "3 fj leg regression rows dMSE < 0 and P >= 0.9": (j["dmse"] < 0 and j["p_below0"] >= 0.9, j),
    "4 val dMSE < -|null val dMSE| (MSE seed 1 vs seed 0)": (v["dmse"] < -abs(null["dmse"]), {"cand": v["dmse"], "null": null["dmse"]}),
}
for k, (ok, x) in c.items():
    print(f"{'PASS' if ok else 'FAIL'}  {k}: {x}")
print("R18-M1:", "PASS" if all(ok for ok, _ in c.values()) else "FAIL")
