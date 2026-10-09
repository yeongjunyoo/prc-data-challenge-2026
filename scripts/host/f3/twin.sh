#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 end-to-end twin check of the fitwrap path (2026-10-08 23:5x KST): leg sk s0 refit directly, without the wrapper, on the
# real feats_v27 in a scratch root (${PRC_WORK:-.}/f3_sbx/twin); its output must equal the final root's cache/full_m6r27sk_s0.parquet
# (made through fitwrap load from the fits stored on the interim table). Same command as research/f3/final_orch.py job sk_s0.
# Criterion (fixed 2026-10-09 00:2x KST, before the twin runs; was bit-equal): every column within 1e-9 (seconds for pred and mu) and the
# same int32 rounding of pred on every row. Two direct LightGBM runs of the same leg already differ by up to 1.0e-12 s (wrapper test T1 vs
# T2 and T1b: LightGBM's run-to-run floating-point order), so bit equality cannot be asked of a refit.
# Report: final_run/research/f3/out/twin_sk_s0.json.
set -u
P=${PRC_WORK:-.}
FR=$P/final_run
TW=$P/f3_sbx/twin
PY=${PY_MAC:-python3}
ts() { date '+%m-%d %H:%M:%S'; }
until [ -e $FR/cache/full_m6r27sk_s0.parquet ]; do sleep 30; done
mkdir -p $TW/cache $TW/preds $TW/logs && cd $TW || exit 1
[ -d src ] || cp -c -R $FR/src src
for f in feats_v27.parquet all.parquet iters_m6r27sk.json; do rm -f cache/$f; cp -c $FR/cache/$f cache/$f; done
rm -f cache/full_m6r27sk_s0.parquet
LGB="--feats feats_v27 --es internal --refit --wpow 1 --leaves 255 --mdl 100 --no_direct --clf_drop_prefix m_,ab_ --full_mult 1.0"
echo "$(ts) twin sk s0 start"
bash $FR/research/f3/mac_step.sh twin_sk_s0 6.5 4 $TW/logs/twin_sk_s0.log $PY -X utf8 -u src/model3.py --setting full --seed 0 --tag m6r27sk $LGB --threads 4 --mu_sep LIRF
echo "$(ts) twin sk s0 rc $?"
$PY -X utf8 - <<'EOF'
import json
import numpy as np
import pandas as pd
a = pd.read_parquet("cache/full_m6r27sk_s0.parquet")
b = pd.read_parquet("<mac-work>/final_run/cache/full_m6r27sk_s0.parquet")
r = {"columns_equal": list(a.columns) == list(b.columns), "rows": [len(a), len(b)]}
if r["columns_equal"] and len(a) == len(b):
    d = {}
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        e = (x == y) | (pd.isna(x) & pd.isna(y))
        if (~e).any():
            d[c] = {"cells": int((~e).sum()), "max_abs": float(np.nanmax(np.abs(x.astype(float) - y.astype(float))))}
    r["cells_differing"] = d
    r["bit_equal"] = not d
    r["int32_pred_rows_differing"] = int((np.rint(a["pred"].to_numpy()) != np.rint(b["pred"].to_numpy())).sum())
    r["PASS"] = bool(all(v["max_abs"] <= 1e-9 for v in d.values()) and r["int32_pred_rows_differing"] == 0)
else:
    r["bit_equal"] = False
    r["PASS"] = False
json.dump(r, open("<mac-work>/final_run/research/f3/out/twin_sk_s0.json", "w"), indent=1)
print(json.dumps(r, indent=1))
EOF
echo "$(ts) twin done"
