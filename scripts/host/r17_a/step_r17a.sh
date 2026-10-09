#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# R17-A short Mac step under PROTOCOL (${PRC_WORK:-.}/PROTOCOL.txt): two checks 30 s apart of no arc_quiet.*, declared threads of the running
# locks + ours <= 12 (<= 13 for a 1-thread step) and memory_pressure free >= 30 %; lock prcm/r17a_<name>.lock only while the command runs; nice 10.
#   bash research/r17_a/step_r17a.sh <name> <threads> <peak GB> <minutes> <log> <command ...>     (from the sandbox root)
set -u
NAME=$1; THR=$2; PEAK=$3; MINS=$4; LOG=$5; shift 5
P=${PRC_WORK:-.}
ts() { date '+%m-%d %H:%M:%S'; }
declared() {
  local s=0 t f
  for f in "$P"/*.lock/info; do
    [ -e "$f" ] || continue
    t=$(grep -i -m1 '^threads' "$f" | grep -o '[0-9][0-9.]*' | head -1)
    s=$(echo "$s + ${t:-6}" | bc)
  done
  echo "$s"
}
ok_now() {
  { [ -e "$P/arc_quiet.request" ] || [ -e "$P/arc_quiet.lock" ]; } && return 1
  local cap=12; [ "$THR" = 1 ] && cap=13
  [ "$(echo "$(declared) + $THR <= $cap" | bc)" = 1 ] || return 1
  local free; free=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %)
  [ "${free:-0}" -ge 30 ]
}
n=0; while [ $n -lt 2 ]; do if ok_now; then n=$((n+1)); else n=0; fi; [ $n -lt 2 ] && sleep 30; done
L="$P/r17a_$NAME.lock"
mkdir "$L" || { echo "$(ts) lock $L exists"; exit 9; }
trap 'rm -rf "$L"' EXIT
printf 'owner: PRC R17-A (%s), coordinator main\npid: %s\npeak_gb: %s\nthreads: %s\nstart: %s\nexpected_end: %s KST\njob: %s\n' \
  "$NAME" "$$" "$PEAK" "$THR" "$(date '+%Y-%m-%d %H:%M')" "$(date -v+${MINS}M '+%Y-%m-%d %H:%M')" "$(echo "$*" | cut -c1-200)" > "$L/info"
echo "$(ts) start $NAME: $*"
OMP_NUM_THREADS=$THR OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8 nice -n 10 /usr/bin/time -l "$@" > "$LOG" 2>&1
rc=$?
echo "$(ts) end $NAME rc $rc ($(grep -E 'peak memory footprint' "$LOG" | awk '{printf "%.2f GB", $1/2^30}'))"
exit $rc
