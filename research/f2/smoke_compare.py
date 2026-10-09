# SPDX-License-Identifier: GPL-3.0-only
"""F2: compare the smoke run of gate14.py / gate15.py on the 4.3 LightGBM world (research/f2/out_smoke, chain S) with the numbers the
independent scripts wrote (research/f2/out/m3_corr_f2m3.json, m3_corr_lomo_f2m3.json). Prints each pair and exits 1 if any differs by more
than 0.01 MSE.
    python -X utf8 research/f2/smoke_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("research/f2/out")
SM = Path("research/f2/out_smoke")


def main() -> int:
    m = json.loads((OUT / "m3_corr_f2m3.json").read_text(encoding="utf-8"))
    lo = json.loads((OUT / "m3_corr_lomo_f2m3.json").read_text(encoding="utf-8"))
    g14 = json.loads((SM / "gate14.json").read_text(encoding="utf-8"))["chains"]["S"]
    g15 = json.loads((SM / "gate15.json").read_text(encoding="utf-8"))["chains"]["S"]
    pairs = [("gate14 I_V_vs_stage1", g14["I_V_vs_stage1"], "m3 I_V", m["I_V"]),
             ("gate14 U_V_vs_stage1", g14["U_V_vs_stage1"], "m3 U_V", m["U_V"]),
             ("gate14 T_vs_stage1", g14["T_vs_stage1"], "m3 T", m["T"]),
             ("gate14 T1_vs_stage1", g14["T1_vs_stage1"], "m3 T1", m["T1"]),
             ("gate14 L_minus_T1_02", g14["L_minus_T1_02"], "lomo F2_L_minus_T1", lo["F2_L_minus_T1"]),
             ("gate14 L_minus_T1_06", g14["L_minus_T1_06"], "lomo F6_L_minus_T1", lo["F6_L_minus_T1"]),
             ("gate15 LV4_minus_LV2_01", g15["LV4_minus_LV2_01"], "lomo V1_plusF_minus_other", lo["V1_plusF_minus_other"]),
             ("gate15 LV4_minus_LV2_07", g15["LV4_minus_LV2_07"], "lomo V7_plusF_minus_other", lo["V7_plusF_minus_other"])]
    bad = 0
    for a, x, b, y in pairs:
        same = abs(x["dmse"] - y["dmse"]) <= 0.01
        bad += not same
        print(f"{'SAME' if same else 'DIFF'}  {a} {x['dmse']} (sd {x.get('sd')}, P {x.get('p_lt0')})  vs  {b} {y['dmse']} (sd {y.get('sd')}, P {y.get('p_lt0')})")
    print("smoke gate15 contrasts (information):", {k: g15[k]["dmse"] for k in ("T_nm_minus_T", "L_nm_minus_L", "U_V_nm_minus_I_V", "I_V_nm_minus_I_V",
                                                                              "LV2_nm_minus_LV2", "LV4_nm_minus_LV4")})
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
