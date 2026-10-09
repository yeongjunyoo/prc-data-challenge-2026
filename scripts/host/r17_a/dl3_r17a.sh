#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# R17-A download and extraction runner on the Mac, third form (private notes research/2026-10-08-r17-a-adsb-fill.md section 4.1), for the PROTOCOL line of
# 10-08 23:20: a network-bound curl stream counts 0.25 thread, and the coordinator keeps 4 R17-A streams in the budget. So each day's lock declares
# 0.25 thread, a day starts after two checks 30 s apart of no arc_quiet.* and memory_pressure free >= 30 % (no thread-sum check), and at most 4
# r17a_dl*.lock exist at once (counted under a mutex, older runners' locks included): the curl caps stay <= 4 x 4800K = 19.7 MB/s.
# Otherwise as dl_r17a.sh: claim a day with mkdir claim/<day> (shared with the other runners), adsb_day_r17a.py --stream at nice 10, meta.json only
# for a complete stream, a failed day goes to failed/<day>.<try> and is retried up to 3 times. Stop: touch ${ADSB_WORK:-.}/STOP3.
#   nohup /bin/bash dl3_r17a.sh <workers> <rate> <days file> <first worker number> >> logs/runner3.log 2>&1 < /dev/null &
set -u
cd ${ADSB_WORK:-.} || exit 1
W=${1:?workers}; RATE=${2:?rate}; DAYS=${3:?days file}; W0=${4:?first worker number}
MAXS=4
P=${PRC_WORK:-.}
PY=${PY_ADSB:-python3}
mkdir -p out logs claim failed
ts() { date '+%m-%d %H:%M:%S'; }
ok_now() {
  { [ -e "$P/arc_quiet.request" ] || [ -e "$P/arc_quiet.lock" ]; } && return 1
  local free; free=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %)
  [ "${free:-0}" -ge 30 ]
}
take_slot() {  # 0 when this worker got a stream slot (its lock dir made), 1 when all 4 are in use
  local lock=$1 n
  until mkdir dl_mutex 2>/dev/null; do sleep 1; done
  n=$(ls -d "$P"/r17a_dl*.lock 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -lt "$MAXS" ] && mkdir "$lock" 2>/dev/null; then rmdir dl_mutex; return 0; fi
  rmdir dl_mutex
  return 1
}
worker() {
  local w=$1 day try rc lock t0 t1 n
  lock="$P/r17a_dl$w.lock"
  while read -r day <&3; do
    [ -n "$day" ] || continue
    [ -e STOP3 ] && { echo "$(ts) w$w STOP3 seen"; return 0; }
    [ -s "out/$day/meta.json" ] && continue
    mkdir "claim/$day" 2>/dev/null || continue
    for try in 1 2 3; do
      n=0; while [ $n -lt 2 ]; do if ok_now; then n=$((n+1)); else n=0; fi; [ $n -lt 2 ] && sleep 30; done
      until take_slot "$lock"; do sleep 20; done
      printf 'owner: PRC R17-A ADS-B day download+extract (worker %s, network-bound curl stream), coordinator main\npid: %s\npeak_gb: 0.9\nthreads: 0.25\nstart: %s\nexpected_end: %s KST\njob: adsb_day_r17a.py --day %s --stream --limit-rate %s\n' \
        "$w" "$$" "$(date '+%Y-%m-%d %H:%M')" "$(date -v+25M '+%Y-%m-%d %H:%M')" "$day" "$RATE" > "$lock/info"
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
echo "$(ts) runner3 start pid $$: $W workers at $RATE each (max $MAXS R17-A streams in all), days $DAYS ($(grep -c . "$DAYS") lines)"
for w in $(seq "$W0" $((W0 + W - 1))); do worker "$w" & sleep 5; done
wait
echo "$(ts) runner3 end"
touch ${ADSB_WORK:-.}/runner3_$(basename "$DAYS" .txt).DONE
