#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# R17-A L3S follow-up on the Mac (private notes research/2026-10-08-r17-a-adsb-fill.md section 4.7; 10-09 09:5x), from the sandbox
# ${PRC_WORK:-.}/r17a_sbx. Holds no lock while it waits; every step takes its own lock (sb_build_l3s.py, stage_r17a.sh).
#   1. waits for ${PRC_WORK:-.}/r17a_L3.DONE (L3's gate ran); stops without building if research/r17_a/l3.FAILED appears first
#   2. if L3 did not pass 7.8 against 514 (research/r17_a/gate_<L3>.json pass78), stops: nothing more is built
#   3. L3S = L3 with --scale2 0.75 in every corrector: research/r17_a/sb_build_l3s.py chain C, A, B side by side, then avg3
#   4. method texts (research/r17_a/method_r17a.py), then stage_r17a.sh: 570 = L3S on L3's gate, then 571 = L3 (kept ready in the sandbox
#      queue only; the PC copies only 570 into research/r17_a/queue_ready/)
#   5. writes ${PRC_WORK:-.}/r17a_L3S.DONE; a failed step writes research/r17_a/l3s.FAILED with its name
#   nohup /bin/bash research/r17_a/l3s_r17a.sh >> research/r17_a/logs/l3s.log 2>&1 < /dev/null &
cd ${PRC_WORK:-.}/r17a_sbx || exit 1
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
L3=m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgjaavskap
L3S=m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "$(ts) FAILED: $1"; echo "$1" > research/r17_a/l3s.FAILED; exit 1; }
echo "$(ts) L3S follow-up pid $$: waiting for $P/r17a_L3.DONE"
until [ -e "$P/r17a_L3.DONE" ]; do
  if [ -e research/r17_a/l3.FAILED ]; then echo "$(ts) l3.FAILED ($(cat research/r17_a/l3.FAILED)); L3S not built"; exit 0; fi
  sleep 120
done
if ! "$PY" -c "import json, sys; g = json.load(open('research/r17_a/gate_$L3.json', encoding='utf-8')); sys.exit(0 if g.get('pass78') else 1)"; then
  echo "$(ts) L3 did not pass 7.8 against 514 (or no gate file); L3S not built"
  exit 0
fi
echo "$(ts) L3 passed 7.8 against 514; building L3S"
for ch in C A B; do
  "$PY" -X utf8 -u research/r17_a/sb_build_l3s.py chain $ch > research/r17_a/logs/l3s_chain_$ch.log 2>&1 &
  eval "pc$ch=\$!"
  sleep 5
done
bad=""
for ch in C A B; do eval "p=\$pc$ch"; wait $p || bad="$bad $ch"; done   # a failed chain does not stop the others mid-step (no orphan locks)
[ -z "$bad" ] || fail "chain$bad"
"$PY" -X utf8 -u research/r17_a/sb_build_l3s.py avg3 > research/r17_a/logs/l3s_avg3.log 2>&1 || fail avg3
[ "$(cat research/r17_a/l3s_spec.txt)" = "$L3S" ] || fail "spec $(cat research/r17_a/l3s_spec.txt)"
"$PY" -X utf8 research/r17_a/method_r17a.py "$L3" > research/r17_a/logs/l3s_method.log 2>&1 || fail method
bash research/r17_a/stage_r17a.sh "$L3S" "570_$L3S" "$(cat research/r17_a/stage_method_570.txt)" "$L3" > research/r17_a/logs/stage_570.log 2>&1 || fail "stage 570"
bash research/r17_a/stage_r17a.sh "$L3" "571_$L3" "$(cat research/r17_a/stage_method_571.txt)" > research/r17_a/logs/stage_571.log 2>&1 || fail "stage 571"
date '+%Y-%m-%d %H:%M:%S' > "$P/r17a_L3S.DONE"
echo "$(ts) L3S done: sandbox queue 570_$L3S (on L3's gate) and 571_$L3"
tail -4 research/r17_a/logs/stage_570.log
