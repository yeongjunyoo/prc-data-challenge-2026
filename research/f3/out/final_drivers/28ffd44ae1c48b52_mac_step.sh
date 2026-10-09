#!/bin/bash
# F3 Mac step runner (PROTOCOL server mode): waits WITHOUT a lock until two checks 30 s apart pass (no arc_quiet.request / arc_quiet.lock,
# declared threads of the running locks + ours <= 12 (<= 13 for a 1-thread step), memory_pressure free >= 30 %), then takes
# ~/nova-adsb/prcm/f3_<name>.lock with the declared peak and threads, runs the command under nice 10 and /usr/bin/time -l in the cwd,
# F3's own locks together (f3_*.lock + ours) stay within F3CAP threads (default 9, coordinator 2026-10-08 23:5x KST),
# writes the log, and releases the lock when the command ends (trap). Exit code = the command's.
#   bash mac_step.sh <name> <peak_gb> <threads> <log> <cmd ...>
set -u
P=~/nova-adsb/prcm
name=$1; gb=$2; th=$3; log=$4; shift 4
L=$P/f3_$name.lock
cap=${TOTCAP:-12}; [ "$th" -le 1 ] && [ -z "${TOTCAP:-}" ] && cap=13  # TOTCAP: coordinator 10-09 13:5x, total may reach 15 beside F2 fj
ok_once() {
  [ -e "$P/arc_quiet.request" ] && return 1
  [ -e "$P/arc_quiet.lock" ] && return 1
  local st=0 t d
  for d in "$P"/*.lock; do
    [ -d "$d" ] || continue
    [ "$d" = "$L" ] && continue
    t=$(grep -i -E '^threads' "$d/info" 2>/dev/null | head -1 | grep -o -E '[0-9]+(\.[0-9]+)?' | head -1)
    st=$(echo "$st + ${t:-6}" | bc)
  done
  local capfail=0
  [ "$(echo "$st + $th <= $cap" | bc)" = 1 ] || capfail=1
  local mine=0
  for d in "$P"/f3_*.lock; do
    [ -d "$d" ] || continue
    [ "$d" = "$L" ] && continue
    t=$(grep -i -E '^threads' "$d/info" 2>/dev/null | head -1 | grep -o -E '[0-9]+(\.[0-9]+)?' | head -1)
    mine=$(echo "$mine + ${t:-6}" | bc)
  done
  [ "$(echo "$mine + $th <= ${F3CAP:-9}" | bc)" = 1 ] || return 1
  local free=$(memory_pressure | tail -1 | grep -o -E '[0-9]+' | head -1)
  [ "${free:-0}" -ge 30 ] || return 1
  if [ $capfail = 1 ]; then
    # coordinator 10-09 14:1x: a step that has waited longer than research/f3/CAP_OVERRIDE_S seconds on the total cap alone runs anyway
    # (F3's own cap and the memory floor still hold); each override is logged in final_run/logs/cap_override.log
    local ov=$(cat "$P/final_run/research/f3/CAP_OVERRIDE_S" 2>/dev/null)
    if [ -n "$ov" ] && [ $(( $(date +%s) - T0 )) -ge "$ov" ]; then
      echo "$(date '+%m-%d %H:%M:%S') f3 $name total-cap override after $(( $(date +%s) - T0 )) s: declared others $st + ours $th > cap $cap, memory free ${free}%" >> "$P/final_run/logs/cap_override.log"
      return 0
    fi
    return 1
  fi
}
T0=$(date +%s)
ok=0; n=0
while [ $ok -lt 2 ]; do
  if ok_once; then ok=$((ok + 1)); else ok=0; fi
  if [ $ok -lt 2 ]; then
    [ $((n % 20)) = 0 ] && echo "$(date '+%m-%d %H:%M:%S') f3 $name waiting (no lock held)"
    n=$((n + 1)); sleep 30
  fi
done
mkdir "$L" || { echo "lock exists: $L"; exit 97; }
trap 'rm -rf "$L"' EXIT
printf 'owner: PRC F3 final rebuild (%s), coordinator main\npid: %s\npeak_gb: %s\nthreads: %s\nstart: %s\nexpected_end: see log %s\njob: %s\n' \
  "$name" "$$" "$gb" "$th" "$(date '+%Y-%m-%d %H:%M')" "$log" "$*" > "$L/info"
echo "$(date '+%m-%d %H:%M:%S') f3 $name start: $*"
OMP_NUM_THREADS=$th OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=$th PYTHONIOENCODING=utf-8 nice -n 10 /usr/bin/time -l "$@" > "$log" 2>&1
rc=$?
echo "$(date '+%m-%d %H:%M:%S') f3 $name rc $rc; $(grep -E 'peak memory footprint' "$log" | tr -s ' ')"
exit $rc
