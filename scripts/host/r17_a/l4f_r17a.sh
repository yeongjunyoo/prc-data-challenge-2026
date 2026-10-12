#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# R17-A L4 fwd phase on the Mac (private notes research/2026-10-08-r17-a-adsb-fill.md section 8), started by research/r17_a/l4v_r17a.sh only when the
# registered val parts of gate 7.8 (L4S vs 570) passed. Holds no lock while it waits; every step takes its own.
#   1. waits for research/r17_a/l4_learn_fwd.DONE (the 15 fwd learners)
#   2. public root: research/r17_a/l4_stage_r17a.py fwd
#   3. in the clone ${PRC_WORK:-.}/r17a_l4g_sbx: the phase-v fwd placeholder links of the five L4 learner inputs are removed and the public
#      root's real fwd files copied in (cp -c), then l4g_build_r17a.py prep --phase f (full stays a link to 514's), chains A, B, C at 0.75 and
#      0.5, avg3, gate --phase f (gate 7.8 (a) to (c) on val and fwd; (d) and the test move are F3's on the final root)
#   4. checks that full reproduced L3's chain (test sha16 of the 0.75 end = 570's, of the 0.5 end = 571's)
#   nohup /bin/bash research/r17_a/l4f_r17a.sh >> research/r17_a/logs/l4f.log 2>&1 < /dev/null &       (from the public root)
cd ${PRC_WORK:-.}/r17a_sbx || exit 1
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
LANE=research/r17_a
C=$P/r17a_l4g_sbx
PUB=${PRC_WORK:-.}/r17a_sbx
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "$(ts) FAILED: $1"; echo "f: $1" > ${PRC_WORK:-.}/r17a_sbx/$LANE/l4.FAILED; exit 1; }
echo "$(ts) L4 fwd-phase orchestrator pid $$: waiting for $LANE/l4_learn_fwd.DONE"
until [ -e "$LANE/l4_learn_fwd.DONE" ]; do
  [ -e "$LANE/l4.FAILED" ] && { echo "$(ts) l4.FAILED: $(cat "$LANE/l4.FAILED"); stopping"; exit 1; }
  sleep 60
done
echo "$(ts) fwd learners done; public-root stages"
"$PY" -X utf8 -u $LANE/l4_stage_r17a.py fwd >> $LANE/logs/l4_stage_fwd.log 2>&1 || fail "public-root stages fwd"
tail -2 $LANE/logs/l4_stage_fwd.log
cd "$C" || fail "cd clone"
for t in m6r28cbwinx3 m6r28cbx3 m6r28cbap m6r28rmwinx4 m6r28rmx4; do
  for f in cache/fwd_${t}_s0.parquet preds/es_valid_ids_${t}_fwd_s0.parquet; do
    [ -L "$f" ] && rm "$f"
    [ -e "$f" ] && fail "$f exists in the clone and is not a placeholder link"
    cp -c "$PUB/$f" "$f" || fail "copy $f"
    echo "$(ts) real fwd file in the clone: $(shasum -a 256 "$f" | cut -c1-16) $f (public root $(shasum -a 256 "$PUB/$f" | cut -c1-16))"
  done
done
"$PY" -X utf8 -u $LANE/l4g_build_r17a.py prep --phase f > $LANE/logs/l4g_prep_f.log 2>&1 || fail "prep f"
pids=""
for sc in 0.75 0.5; do for ch in A B C; do
  "$PY" -X utf8 -u $LANE/l4g_build_r17a.py chain --phase f --chain $ch --scale $sc > $LANE/logs/l4g_f_$ch$sc.log 2>&1 &
  pids="$pids $!"; sleep 3
done; done
bad=""; for p in $pids; do wait $p || bad="$bad $p"; done
[ -z "$bad" ] && echo "$(ts) l4g chains f done" || fail "l4g chain f failed (pids$bad; logs $LANE/logs/l4g_f_*.log in the clone)"
for sc in 0.75 0.5; do "$PY" -X utf8 -u $LANE/l4g_build_r17a.py avg3 --phase f --scale $sc > $LANE/logs/l4g_avg3_f$sc.log 2>&1 || fail "avg3 f $sc"; done
"$PY" -X utf8 -u $LANE/l4g_build_r17a.py gate --phase f > $LANE/logs/l4g_gate_f.log 2>&1 || fail "gate f"
"$PY" -X utf8 - << 'PYEOF2' > $LANE/logs/l4g_repro_f.log 2>&1
import json
from pathlib import Path
L = Path("research/r17_a")
out = {}
for stag, ref in (("s75", "preds/manifest_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.json"),
                  ("h", "preds/manifest_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgjaavskap.json")):
    spec = (L / f"l4g_spec_f{stag}.txt").read_text().strip()
    mc, mr = json.loads(Path(f"preds/manifest_{spec}.json").read_text()), json.loads(Path(ref).read_text())
    out[spec] = {k: {"candidate": mc[k]["sha"], "reference": mr[k]["sha"], "equal": mc[k]["sha"] == mr[k]["sha"]} for k in ("forward", "test", "holdout")}
(L / "l4g_repro_f.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
PYEOF2
cat $LANE/logs/l4g_repro_f.log
grep -h "gate7.8 parts" logs/gate_l4f_registered.log logs/gate_l4f_information.log 2> /dev/null
date '+%Y-%m-%d %H:%M:%S' > $P/r17a_L4f.DONE
echo "$(ts) L4 fwd phase done"
