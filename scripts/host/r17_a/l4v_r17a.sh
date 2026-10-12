#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# R17-A L4 val phase on the Mac (private notes research/2026-10-08-r17-a-adsb-fill.md section 8; 10-10 10:2x). Holds no lock while it
# waits; every step takes its own.
#   1. waits for research/r17_a/l4_learn_val.DONE (the 15 val learners, research/r17_a/l4_learn_r17a.sh val)
#   2. public root: research/r17_a/l4_stage_r17a.py val (averages, blends, correctors 0.75 and 0.5 with the absolute fitcache, cavg)
#   3. the APFS clone ${PRC_WORK:-.}/r17a_l4g_sbx (cp -c -R of the public root, made once), then in it: l4g_build_r17a.py prep --phase v
#      (fwd and full of the L4 learner inputs link to 514's), chains A, B, C at scale2 0.75 and 0.5 side by side, avg3 per scale, gate --phase v
#   4. checks that the placeholder settings reproduced L3's chain (forward and test sha16 of the 0.75 end = 570's, of the 0.5 end = 571's)
#   5. registered val parts of gate 7.8 (L4S vs 570): pass -> starts the fwd learners and research/r17_a/l4f_r17a.sh (nohup, public root);
#      fail -> stops (research/r17_a/l4.STOPPED_VAL) and fits nothing more
#   nohup /bin/bash research/r17_a/l4v_r17a.sh >> research/r17_a/logs/l4v.log 2>&1 < /dev/null &       (from the public root)
cd ${PRC_WORK:-.}/r17a_sbx || exit 1
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
LANE=research/r17_a
C=$P/r17a_l4g_sbx
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "$(ts) FAILED: $1"; echo "v: $1" > ${PRC_WORK:-.}/r17a_sbx/$LANE/l4.FAILED; exit 1; }
echo "$(ts) L4 val-phase orchestrator pid $$: waiting for $LANE/l4_learn_val.DONE"
until [ -e "$LANE/l4_learn_val.DONE" ]; do
  [ -e "$LANE/l4.FAILED" ] && { echo "$(ts) l4.FAILED: $(cat "$LANE/l4.FAILED"); stopping"; exit 1; }
  sleep 60
done
echo "$(ts) val learners done; public-root stages"
"$PY" -X utf8 -u $LANE/l4_stage_r17a.py val >> $LANE/logs/l4_stage_val.log 2>&1 || fail "public-root stages val"
tail -2 $LANE/logs/l4_stage_val.log
if [ ! -d "$C" ]; then
  echo "$(ts) cloning the public root to $C"
  cp -c -R ${PRC_WORK:-.}/r17a_sbx "$C" || fail "clone"
  echo "$(ts) clone made; absolute links in it that point into the public root: $(find "$C" -type l -lname "${PRC_WORK:-.}/r17a_sbx/*" 2> /dev/null | wc -l | tr -d ' ')"
fi
cd "$C" || fail "cd clone"
"$PY" -X utf8 -u $LANE/l4g_build_r17a.py prep --phase v > $LANE/logs/l4g_prep_v.log 2>&1 || fail "prep v"
pids=""
for sc in 0.75 0.5; do for ch in A B C; do
  "$PY" -X utf8 -u $LANE/l4g_build_r17a.py chain --phase v --chain $ch --scale $sc > $LANE/logs/l4g_v_$ch$sc.log 2>&1 &
  pids="$pids $!"; sleep 3
done; done
bad=""; for p in $pids; do wait $p || bad="$bad $p"; done
[ -z "$bad" ] && echo "$(ts) l4g chains v done" || fail "l4g chain v failed (pids$bad; logs $LANE/logs/l4g_v_*.log in the clone)"
for sc in 0.75 0.5; do "$PY" -X utf8 -u $LANE/l4g_build_r17a.py avg3 --phase v --scale $sc > $LANE/logs/l4g_avg3_v$sc.log 2>&1 || fail "avg3 v $sc"; done
"$PY" -X utf8 -u $LANE/l4g_build_r17a.py gate --phase v > $LANE/logs/l4g_gate_v.log 2>&1 || fail "gate v"
"$PY" -X utf8 - << 'PYEOF' > $LANE/logs/l4g_repro_v.log 2>&1
import hashlib, json
from pathlib import Path
L = Path("research/r17_a")
out = {}
for stag, ref in (("s75", "preds/manifest_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.json"),
                  ("h", "preds/manifest_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgjaavskap.json")):
    spec = (L / f"l4g_spec_v{stag}.txt").read_text().strip()
    mc, mr = json.loads(Path(f"preds/manifest_{spec}.json").read_text()), json.loads(Path(ref).read_text())
    out[spec] = {k: {"candidate": mc[k]["sha"], "reference": mr[k]["sha"], "equal": mc[k]["sha"] == mr[k]["sha"]} for k in ("forward", "test", "holdout")}
(L / "l4g_repro_v.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
PYEOF
cat $LANE/logs/l4g_repro_v.log
pass=$("$PY" -c "import json, glob; g = json.load(open(glob.glob('research/r17_a/gate_l4v_registered_*.json')[0])); print(1 if g['parts_pass'] else 0)")
grep -h "gate7.8 parts" logs/gate_l4v_registered.log logs/gate_l4v_information.log 2> /dev/null
date '+%Y-%m-%d %H:%M:%S' > $P/r17a_L4v.DONE
cd ${PRC_WORK:-.}/r17a_sbx || exit 1
if [ "$pass" = 1 ]; then
  echo "$(ts) val parts of gate 7.8 (L4S vs 570) PASS: starting the fwd learners and the fwd phase"
  nohup /bin/bash $LANE/l4_learn_r17a.sh fwd >> $LANE/logs/l4_learn_fwd.log 2>&1 < /dev/null &
  echo "$(ts) fwd learners pid $!"
  nohup /bin/bash $LANE/l4f_r17a.sh >> $LANE/logs/l4f.log 2>&1 < /dev/null &
  echo "$(ts) fwd-phase orchestrator pid $!"
else
  echo "$(ts) val parts of gate 7.8 (L4S vs 570) FAIL: stopping, nothing more is fitted"
  date '+%Y-%m-%d %H:%M:%S' > $LANE/l4.STOPPED_VAL
fi
