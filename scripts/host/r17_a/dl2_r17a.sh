#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# R17-A download and extraction runner on the Mac (dl2: dl_r17a.sh plus a first-worker-number argument, so a second runner uses other lock names; dl_r17a.sh stays as it ran) (private notes research/2026-10-08-r17-a-adsb-fill.md). Runs from ${ADSB_WORK:-.}.
#   nohup /bin/bash dl_r17a.sh <workers> <curl rate per worker, e.g. 9500K> <days file> [<first worker number, default 1>] >> logs/runner.log 2>&1 < /dev/null &
# Each worker claims the next day of the days file (mkdir claim/<day>, atomic), skips days whose out/<day>/meta.json exists, and runs
# adsb_day_r17a.py --stream (curl at the given rate, nice 10, one Python thread). PROTOCOL (${PRC_WORK:-.}/PROTOCOL.txt): before each day
# two checks 30 s apart of no arc_quiet.request / arc_quiet.lock, declared threads of the running locks + 1 <= 13 (1-thread step rule) and
# memory_pressure free >= 30 %; the worker's lock prcm/r17a_dl<w>.lock is held only while a day runs (info: 1 thread, 0.9 GB).
# A failed day (exit code != 0, e.g. 3 = short stream) is moved to failed/<day>.<try> and retried up to 3 times.
# Stop: touch ${ADSB_WORK:-.}/STOP (workers finish their current day and exit).
set -u
cd ${ADSB_WORK:-.} || exit 1
W=${1:?workers}; RATE=${2:?rate}; DAYS=${3:?days file}; W0=${4:-1}
P=${PRC_WORK:-.}
PY=${PY_ADSB:-python3}
mkdir -p out logs claim failed
ts() { date '+%m-%d %H:%M:%S'; }
declared() {  # sum of declared threads of other lanes' locks plus mine
  local s=0 t f
  for f in "$P"/*.lock/info; do
    [ -e "$f" ] || continue
    t=$(grep -i -m1 '^threads' "$f" | grep -o '[0-9][0-9.]*' | head -1)
    s=$(echo "$s + ${t:-6}" | bc)
  done
  echo "$s"
}
ok_now() {
  [ -e "$P/arc_quiet.request" ] || [ -e "$P/arc_quiet.lock" ] && return 1
  local d free
  d=$(declared)
  [ "$(echo "$d + 1 <= 13" | bc)" = 1 ] || return 1
  free=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %)
  [ "${free:-0}" -ge 30 ] || return 1
  return 0
}
wait_ok() { local n=0; while [ $n -lt 2 ]; do if ok_now; then n=$((n+1)); else n=0; fi; [ $n -lt 2 ] && sleep 30; done; }
worker() {
  local w=$1 day try rc lock t0 t1
  lock="$P/r17a_dl$w.lock"
  while read -r day <&3; do
    [ -n "$day" ] || continue
    [ -e STOP ] && { echo "$(ts) w$w STOP seen"; return 0; }
    [ -s "out/$day/meta.json" ] && continue
    mkdir "claim/$day" 2>/dev/null || continue
    for try in 1 2 3; do
      wait_ok
      mkdir "$lock" || { echo "$(ts) w$w lock exists?"; sleep 30; continue; }
      printf 'owner: PRC R17-A ADS-B day download+extract (worker %s), coordinator main\npid: %s\npeak_gb: 0.9\nthreads: 1\nstart: %s\nexpected_end: %s KST\njob: adsb_day_r17a.py --day %s --stream --limit-rate %s\n' \
        "$w" "$$" "$(date '+%Y-%m-%d %H:%M')" "$(date -v+12M '+%Y-%m-%d %H:%M')" "$day" "$RATE" > "$lock/info"
      t0=$(date +%s)
      nice -n 10 /usr/bin/time -l "$PY" -I adsb_day_r17a.py --day "$day" --manifest manifest_preferred.tsv --out out --stream \
        --limit-rate "$RATE" > "logs/$day.log" 2>&1
      rc=$?
      t1=$(date +%s)
      rm -rf "$lock"
      if [ $rc -eq 0 ] && [ -s "out/$day/meta.json" ]; then
        echo "$(ts) w$w $day ok try $try wall $((t1 - t0)) s | $("$PY" -I -c "import json,sys; m=json.load(open(sys.argv[1])); print('MB', round(m['bytes']/1e6), 'MB/s', round(m['bytes']/1e6/max(m['wall_s'],1),1), 'parse_cpu', m['parse_cpu_s'], 'gz/bytes', round(m['gz_bytes']/m['bytes'],3), 'events', m['n_events'], 'bad', m.get('bad_members',0), 'rss_mb', m.get('max_rss_mb'))" "out/$day/meta.json")"
        break
      fi
      echo "$(ts) w$w $day FAILED try $try rc $rc wall $((t1 - t0)) s: $(tail -2 "logs/$day.log" | tr '\n' ' ' | cut -c1-300)"
      mv "out/$day" "failed/$day.$try" 2>/dev/null
      mv "logs/$day.log" "failed/$day.$try.log" 2>/dev/null
      sleep 60
    done
  done 3< "$DAYS"
  echo "$(ts) w$w done"
}
echo "$(ts) runner start pid $$: $W workers at $RATE each, days $DAYS ($(grep -c . "$DAYS") lines)"
for w in $(seq "$W0" $((W0 + W - 1))); do worker "$w" & sleep 5; done
wait
echo "$(ts) runner end"
touch ${ADSB_WORK:-.}/runner_$(basename "$DAYS" .txt).DONE
