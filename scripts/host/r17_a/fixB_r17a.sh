#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# R17-A chain B fix and L3 resume on the Mac (10-10 02:0x; 01:5x: a mechanical cause may be fixed in the smallest way, then chain B
# resumes from its state file at the failed step; anything that changes what chain B computes stops for a report).
# Cause (sb_chain_B.log, 00:47:53): chain B's bfix stopped with FileNotFoundError cache/val_m6r27cbx3_s0.parquet. 514's B chain used the
# CatBoost learner m6r27cbx3 (3-seed average), which this sandbox never had: the B chain of 514 was not rebuilt on the Mac, so the clone
# lacks its learner caches (A's and C's learners were present, and their blends reproduced at 22:11 on 10-08).
# Fix: the PC's cache/{val,fwd,full}_m6r27cbx3_s0.parquet, with 514's B base m6r27skgx4 and 514's B base blend m6r27skcbrmfw4sv434, arrive
# unpacked in research/r17_a/runs/fixB_in/ (02:06), and the PC's preds/es_valid_ids_{m6r27cbx3,m6r27skgx4}_{val,fwd}_s0.parquet (02:16:
# src/blend_fixed.py reads the early-stopping row ids of the base and of every learner for val and fwd and asserts they are equal; the
# 02:06 reproduction stopped on the missing es_valid_ids_m6r27skgx4_val_s0, and chain B's own bfix would have stopped next on the missing
# es_valid_ids_m6r27cbx3). This script
#   1. adds each file to cache/ only if it is missing there; a file that exists with another sha16 stops everything (exit 2)
#   2. proves they are the learner caches 514's B chain used: src/blend_fixed.py with 514's B arguments (base m6r27skgx4, learners
#      m6r27cbx3:mu_cat,m6r25xg:mu_xgb,m6r27rmx4:mu_rmlp, reference weights) under the tag r17aBrepro must give 514's B blend
#      m6r27skcbrmfw4sv434 (max |diff| < 1e-4 on every numeric column, val, fwd, full), else exit 3 and nothing is resumed;
#      the repro outputs (cache/, notes/, preds/ r17aBrepro files) are removed afterwards
#   3. moves research/r17_a/l3.FAILED to l3.FAILED.0111 and starts research/r17_a/l3r_r17a.sh (chain B from its state, then final) and
#      research/r17_a/l3s_r17a.sh again (it had stopped on l3.FAILED), both nohup and appending to their old logs
#   ssh ${MAC_HOST:-mac} "cd ${PRC_WORK:-.}/r17a_sbx && mkdir -p research/r17_a/runs/fixB_in && tar -xf - -C research/r17_a/runs/fixB_in && \
#     perl -e 'alarm shift; exec @ARGV' 500 bash research/r17_a/runs/fixB_in/research/r17_a/fixB_r17a.sh" < fixB.tar
cd ${PRC_WORK:-.}/r17a_sbx || exit 1
PY=${PY_MAC:-python3}
IN=research/r17_a/runs/fixB_in
REF="--ref_base m6r25sg --ref_tag m6r25sgoof4 --ref_learners m6r25cb:mu_cat,m6r25xg:mu_xgb,m6r25rm:mu_rmlp --folds 10 --check"
ts() { date '+%m-%d %H:%M:%S'; }
echo "== $(ts) fixB start"
for f in research/r17_a/l3r_r17a.sh research/r17_a/fixB_r17a.sh; do cp -p "$IN/$f" "$f"; echo "script $(shasum -a 256 "$f" | cut -c1-16) $f"; done
for f in $(cd "$IN" && find cache preds -type f 2> /dev/null | sort); do
  b=$(shasum -a 256 "$IN/$f" | cut -c1-16)
  if [ -e "$f" ]; then
    a=$(shasum -a 256 "$f" | cut -c1-16)
    if [ "$a" = "$b" ]; then echo "same already: $a $f"; else echo "DIFFERS, stop: $f sandbox $a, PC $b"; exit 2; fi
  else
    cp -p "$IN/$f" "$f"; echo "added: $(shasum -a 256 "$f" | cut -c1-16) $f"
  fi
done
for s in val fwd full; do
  for x in "cache/${s}_r17aBrepro_s0.parquet" "notes/blend_fixed_r17aBrepro_${s}_s0.json" "preds/es_valid_ids_r17aBrepro_${s}_s0.parquet"; do
    [ -e "$x" ] && { echo "repro target exists, stop: $x"; exit 3; }
  done
done
ok=1
for s in val fwd full; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 "$PY" -X utf8 -u src/blend_fixed.py --setting $s --seed 0 --tag r17aBrepro --base m6r27skgx4 \
    --learners m6r27cbx3:mu_cat,m6r25xg:mu_xgb,m6r27rmx4:mu_rmlp $REF > "logs/r17aBrepro_$s.log" 2>&1 || { echo "repro run failed ($s):"; tail -3 "logs/r17aBrepro_$s.log"; ok=0; break; }
  "$PY" -X utf8 - "$s" <<'PYEOF' || ok=0
import sys
import numpy as np
import pandas as pd
s = sys.argv[1]
a = pd.read_parquet(f"cache/{s}_r17aBrepro_s0.parquet").set_index("MVT_ID_mvt")
b = pd.read_parquet(f"cache/{s}_m6r27skcbrmfw4sv434_s0.parquet").set_index("MVT_ID_mvt")
same_ids = len(a) == len(b) and bool(a.index.equals(b.index))
b = b.reindex(a.index)
cols = [c for c in a.columns if a[c].dtype.kind in "fiub" and c in b.columns]
d = {c: float(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))) for c in cols}
nan_same = all(bool(np.array_equal(np.isnan(a[c].to_numpy(float)), np.isnan(b[c].to_numpy(float)))) for c in cols)
ok = same_ids and nan_same and set(a.columns) == set(b.columns) and max(d.values()) < 1e-4
print(f"repro {s}: rows {len(a)}, ids equal {same_ids}, columns equal {set(a.columns) == set(b.columns)}, NaN places equal {nan_same}, "
      f"max |diff| {d} -> {'SAME' if ok else 'DIFFERENT'}", flush=True)
sys.exit(0 if ok else 1)
PYEOF
  [ $ok = 1 ] || break
done
for s in val fwd full; do rm -f "cache/${s}_r17aBrepro_s0.parquet" "notes/blend_fixed_r17aBrepro_${s}_s0.json" "preds/es_valid_ids_r17aBrepro_${s}_s0.parquet" "logs/r17aBrepro_$s.log"; done
if [ $ok != 1 ]; then echo "$(ts) reproduction FAILED: nothing resumed (the added cache files stay)"; exit 3; fi
echo "$(ts) reproduction passed: these are the learner caches of 514's B chain"
[ -e research/r17_a/l3.FAILED ] && mv research/r17_a/l3.FAILED research/r17_a/l3.FAILED.0111 && echo "l3.FAILED moved to l3.FAILED.0111"
pgrep -f 'research/r17_a/l3r_r17a.sh' > /dev/null && { echo "l3r already running"; exit 4; }
nohup /bin/bash research/r17_a/l3r_r17a.sh >> research/r17_a/logs/l3.log 2>&1 < /dev/null &
echo "$(ts) l3r started pid $!"
if pgrep -f 'research/r17_a/l3s_r17a.sh' > /dev/null; then echo "l3s already running"
else nohup /bin/bash research/r17_a/l3s_r17a.sh >> research/r17_a/logs/l3s.log 2>&1 < /dev/null & echo "$(ts) l3s started pid $!"; fi
sleep 3
tail -2 research/r17_a/logs/l3.log | cut -c1-200
for d in ../*.lock; do [ -d "$d" ] && echo "lock $(basename "$d") $(grep -m1 -i '^threads' "$d/info")"; done
memory_pressure | tail -1
