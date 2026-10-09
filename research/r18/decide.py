# SPDX-License-Identifier: GPL-3.0-only
"""R18 decision for M2 and M3 (the four rules registered for M1 in the private note 2026-10-09-r18-latest-hits.md section 1, reused word for word by
M2 section 3 and M3 section 5). Reads two screen reports made by screen_r18.py with --ref r18tmc s0:
  <name>_all.json  : --st val,fwd,fj --cand <candidate>                      (rules 1, 2, 3)
  <name>_null.json : --st val --cand <candidate> --cand r18tmc s1 (null)      (rule 4: cand1 is the null)
    python research/r18/decide.py m2
"""
import json
import sys
from pathlib import Path

name = sys.argv[1]
S = Path(__file__).parent / "screens"
r = json.loads((S / f"{name}_all.json").read_text(encoding="utf-8"))
n = json.loads((S / f"{name}_null.json").read_text(encoding="utf-8"))
v, f, j = r["val"]["blend_d_cand0"]["all"], r["fwd"]["blend_d_cand0"]["all"], r["fj"]["leg_cand0"]["reg"]
null = n["val"]["blend_d_cand1"]["all"]
assert abs(n["val"]["blend_d_cand0"]["all"]["dmse"] - v["dmse"]) < 1e-9, "the two reports disagree on the candidate"
c = {
    "1 val blend all rows dMSE < 0 and P >= 0.9": (v["dmse"] < 0 and v["p_below0"] >= 0.9, v),
    "2 fwd blend all rows dMSE < 0": (f["dmse"] < 0, f),
    "3 fj leg regression rows dMSE < 0 and P >= 0.9": (j["dmse"] < 0 and j["p_below0"] >= 0.9, j),
    "4 val dMSE < -|null val dMSE| (MSE seed 1 vs seed 0)": (v["dmse"] < -abs(null["dmse"]), {"cand": v["dmse"], "null": null["dmse"]}),
}
for k, (ok, x) in c.items():
    print(f"{'PASS' if ok else 'FAIL'}  {k}: {x}")
print(f"R18-{name.upper()}:", "PASS" if all(ok for ok, _ in c.values()) else "FAIL")
