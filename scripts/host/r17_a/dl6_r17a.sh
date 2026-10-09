#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# R17-A download and extraction runner on the Mac, sixth form: dl5_r17a.sh whose per-day lock info also keeps rate_bps (in dl4 and dl5 the full
# info overwrote the slot-time rate_bps, so their streams counted as 4800K each: safe, but 4 streams instead of 6). dl5 text: fifth form (private notes research/2026-10-08-r17-a-adsb-fill.md section 4.1). dl4_r17a.sh plus the
# coordinator's 10-09 04:1x conditions: no new day while ${PRC_WORK:-.}/final_run/REALMLP_ALONE exists (F3's RealMLP needs the Mac alone), and
# the memory (>= 30 % free) and REALMLP_ALONE checks are repeated right after a slot is taken, just before curl starts (the slot is given back
# when they fail). dl4's text follows: dl3_r17a.sh with the
# stream limit counted in bytes per second instead of streams. 10-09 04:0x: GitHub paced each connection at about 2.2 MB/s and an extra
# connection added its own rate (probe: 8.3 -> 11.3 MB/s), so more, slower streams fill the same 20 MB/s cap. Each lock records its curl cap
# (rate_bps; a lock without it, from dl3, counts 4800K) and a day starts only when the caps of all r17a_dl*.lock plus its own stay
# <= 19,700,000 B/s (checked under a mutex). Each stream still declares 0.25 thread (PROTOCOL 10-08 23:20).
# Otherwise as dl_r17a.sh: claim a day with mkdir claim/<day> (shared with the other runners), adsb_day_r17a.py --stream at nice 10, meta.json only
# for a complete stream, a failed day goes to failed/<day>.<try> and is retried up to 3 times. Stop: touch ${ADSB_WORK:-.}/STOP6.
#   nohup /bin/bash dl6_r17a.sh <workers> <rate, e.g. 3200K> <days file> <first worker number> >> logs/runner6.log 2>&1 < /dev/null &
set -u
cd ${ADSB_WORK:-.} || exit 1
W=${1:?workers}; RATE=${2:?rate}; DAYS=${3:?days file}; W0=${4:?first worker number}
CAP=19700000
RBPS=$(( ${RATE%K} * 1024 ))
P=${PRC_WORK:-.}
PY=${PY_ADSB:-python3}
mkdir -p out logs claim failed
ts() { date '+%m-%d %H:%M:%S'; }
ok_now() {
  { [ -e "$P/arc_quiet.request" ] || [ -e "$P/arc_quiet.lock" ] || [ -e "$P/final_run/REALMLP_ALONE" ]; } && return 1
  local free; free=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %)
  [ "${free:-0}" -ge 30 ]
}
take_slot() {  # 0 when this worker got a slot (its lock dir made): the curl caps of all R17-A streams + this one <= CAP
  local lock=$1 sum=0 f r
  until mkdir dl_mutex 2>/dev/null; do sleep 1; done
  for f in "$P"/r17a_dl*.lock; do
    [ -d "$f" ] || continue
    r=$(grep -m1 '^rate_bps' "$f/info" 2>/dev/null | grep -o '[0-9]*' | head -1)
    sum=$((sum + ${r:-4915200}))
  done
  if [ $((sum + RBPS)) -le $CAP ] && mkdir "$lock" 2>/dev/null; then
    printf 'threads: 0.25
rate_bps: %s
' "$RBPS" > "$lock/info"; rmdir dl_mutex; return 0
  fi
  rmdir dl_mutex
  return 1
}
worker() {
  local w=$1 day try rc lock t0 t1 n
  lock="$P/r17a_dl$w.lock"
  while read -r day <&3; do
    [ -n "$day" ] || continue
    [ -e STOP6 ] && { echo "$(ts) w$w STOP6 seen"; return 0; }
    [ -s "out/$day/meta.json" ] && continue
    mkdir "claim/$day" 2>/dev/null || continue
    for try in 1 2 3; do
      while true; do  # two checks 30 s apart, a slot under the byte cap, then the same checks again right before the start
        n=0; while [ $n -lt 2 ]; do if ok_now; then n=$((n+1)); else n=0; fi; [ $n -lt 2 ] && sleep 30; done
        until take_slot "$lock"; do sleep 20; done
        ok_now && break
        rm -rf "$lock"; sleep 30
      done
      printf 'owner: PRC R17-A ADS-B day download+extract (worker %s, network-bound curl stream), coordinator main\npid: %s\npeak_gb: 0.9\nthreads: 0.25\nrate_bps: %s\nstart: %s\nexpected_end: %s KST\njob: adsb_day_r17a.py --day %s --stream --limit-rate %s\n' \
        "$w" "$$" "$RBPS" "$(date '+%Y-%m-%d %H:%M')" "$(date -v+25M '+%Y-%m-%d %H:%M')" "$day" "$RATE" > "$lock/info"
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
echo "$(ts) runner6 start pid $$: $W workers at $RATE each ($RBPS B/s; all R17-A curl caps <= $CAP B/s), days $DAYS ($(grep -c . "$DAYS") lines)"
for w in $(seq "$W0" $((W0 + W - 1))); do worker "$w" & sleep 5; done
wait
echo "$(ts) runner6 end"
touch ${ADSB_WORK:-.}/runner6_$(basename "$DAYS" .txt).DONE
