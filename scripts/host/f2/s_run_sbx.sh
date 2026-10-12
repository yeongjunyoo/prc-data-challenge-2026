#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F2 Package S sandbox run in the clone, in order: s_chain prep, s_learn val,fwd (the new learner seeds), s_chain avg, final, gatechain, gate,
# export (research/f2/s_chain.py and s_learn.py hold the rules: PROTOCOL checks and a lock per job, one heavy job at a time, RealMLP alone
# with its smoke fit, OMP_NUM_THREADS=1 and faulthandler). This script holds no lock and fits nothing itself. It stops at the first failure
# (research/f2/out/s_run_<family>.FAILED); ${PRC_WORK:-.}/f2s_STOP ends it cleanly between jobs, ${PRC_WORK:-.}/f2s_HOLD pauses it.
# On success: research/f2/out/s_verdict_sbx_<family>.json (rules 1, 2, 3a), s_export_<family>.json (F3's clone list), s_run_<family>.DONE.
#   nohup /bin/bash research/f2/mac/s_run_sbx.sh <598c|l4> >> logs/f2s_run.log 2>&1 < /dev/null &      (cwd = the clone)
set -u
FAM=${1:?family 598c or l4}
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
case $FAM in 598c) D=$P/f2s_sbx598 ;; l4) D=$P/f2s_sbxl4 ;; *) echo "family 598c or l4"; exit 2 ;; esac
cd "$D" || exit 1
O=research/f2/out
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "$(ts) FAILED: $1"; echo "$(date '+%Y-%m-%d %H:%M:%S') $1" > "$O/s_run_$FAM.FAILED"; exit 1; }
[ -e "$O/s_run_$FAM.FAILED" ] && { echo "$(ts) $O/s_run_$FAM.FAILED exists: $(cat "$O/s_run_$FAM.FAILED")"; exit 1; }
rm -f "$O/s_run_$FAM.DONE"   # a rerun never leaves an earlier run's DONE standing
echo "$(ts) S sandbox run $FAM, pid $$, in $D"
step() {
  local s=$1 rc
  "$PY" -X utf8 -u research/f2/s_chain.py "$FAM" "$s" >> "logs/f2s_${FAM}_chain_$s.log" 2>&1; rc=$?
  [ $rc = 0 ] || fail "s_chain $s rc $rc (logs/f2s_${FAM}_chain_$s.log: $(tail -1 "logs/f2s_${FAM}_chain_$s.log" | cut -c1-200))"
  echo "$(ts) $s: $(tail -1 "logs/f2s_${FAM}_chain_$s.log" | cut -c1-220)"
}
step prep
"$PY" -X utf8 -u research/f2/s_learn.py "$FAM" val,fwd >> "logs/f2s_${FAM}_learn.log" 2>&1; rc=$?
[ $rc = 3 ] && { echo "$(ts) learners stopped by f2s_STOP"; exit 0; }
[ $rc = 0 ] || fail "learners rc $rc (logs/f2s_${FAM}_learn.log: $(tail -1 "logs/f2s_${FAM}_learn.log" | cut -c1-200))"
echo "$(ts) learners: $(grep -E 'RealMLP seeds finished|runner done' "logs/f2s_${FAM}_learn.log" | tail -2 | tr '\n' ' ' | cut -c1-240)"
for s in avg final gatechain gate export; do step $s; done
date '+%Y-%m-%d %H:%M:%S' > "$O/s_run_$FAM.DONE"
echo "$(ts) S sandbox run $FAM done: $(grep -h 'SANDBOX VERDICT' "logs/f2s_${FAM}_chain_gate.log" | tail -1)"
