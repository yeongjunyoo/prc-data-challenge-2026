#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 fitwrap test, follow-up: a second direct (unwrapped) m6r25 full run T1b, to measure LightGBM's own run-to-run difference
# (T1 vs T2 differed by at most 1.0e-12 s on 0.7% of rows while T2 vs T3 load was bit-equal). Compares T1b with T1 and T2.
set -u
P=${PRC_WORK:-.}
W=$P/f3_sbx/wt
PY=${PY_MAC:-python3}
A="--setting full --seed 0 --tag m6r25 --feats feats_v25 --es internal --refit --wpow 1 --leaves 255 --mdl 100 --no_direct --clf_drop_prefix m_ --full_mult 1.0 --threads 4"
cd $W/r1 && bash $P/final_run/research/f3/mac_step.sh wt_t1b 7 4 $W/r1/logs/T1b.log $PY -X utf8 -u src/model3.py $A
[ -e cache/full_m6r25_s0.parquet ] && mv cache/full_m6r25_s0.parquet $W/out/T1b.parquet
cd $W/out && $PY -X utf8 - <<'PYX'
import json
import numpy as np
import pandas as pd
T = {k: pd.read_parquet(f"{k}.parquet") for k in ("T1", "T1b", "T2")}
def cmp(a, b):
    r = {}
    for c in ("pred", "p_art", "mu"):
        x, y = a[c].to_numpy(), b[c].to_numpy()
        d = np.abs(x - y)
        r[c] = {"bits_differ": int((x.view("u8") != y.view("u8")).sum()), "max_abs": float(np.nanmax(d))}
    r["pred_int32_rounding_differ"] = int((np.rint(a["pred"].to_numpy()) != np.rint(b["pred"].to_numpy())).sum())
    return r
rep = {"T1_vs_T1b_direct_direct": cmp(T["T1"], T["T1b"]), "T1b_vs_T2_direct_save": cmp(T["T1b"], T["T2"]), "T1_vs_T2_direct_save": cmp(T["T1"], T["T2"])}
json.dump(rep, open("wrapper_test_t1b.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
PYX
