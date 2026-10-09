#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# R17-A pilot (private notes research/2026-10-08-r17-a-adsb-fill.md section 3): two days at once, each curl at 9500K (19.5 MB/s together).
#   day 1 on disk (download timed per part, then the parse from disk, raw deleted after), day 2 streamed (curl stdout parsed directly).
# PROTOCOL: memory_pressure free >= 30 % and declared threads + 2 <= 13 at launch, locks r17a_pilot<n>.lock held while each day runs, nice 10.
#   nohup /bin/bash pilot_r17a.sh 2026-02-01 2026-02-02 >> logs/pilot.log 2>&1 < /dev/null &
set -u
cd ${ADSB_WORK:-.} || exit 1
D1=${1:?disk day}; D2=${2:?stream day}
P=${PRC_WORK:-.}
PY=${PY_ADSB:-python3}
ts() { date '+%m-%d %H:%M:%S'; }
mkdir -p claim; mkdir "claim/$D1" "claim/$D2" || { echo "$(ts) pilot days already claimed"; exit 2; }
free=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %)
echo "$(ts) pilot start pid $$, memory free ${free}%"
[ "${free:-0}" -ge 30 ] || { echo "$(ts) memory under 30%, stop"; exit 3; }
run() {  # n day mode-args
  local n=$1 day=$2; shift 2
  local lock="$P/r17a_pilot$n.lock"
  mkdir "$lock" || return 9
  printf 'owner: PRC R17-A ADS-B pilot day %s, coordinator main\npid: %s\npeak_gb: 0.9\nthreads: 1\nstart: %s\nexpected_end: %s KST\njob: adsb_day_r17a.py --day %s %s\n' \
    "$day" "$$" "$(date '+%Y-%m-%d %H:%M')" "$(date -v+12M '+%Y-%m-%d %H:%M')" "$day" "$*" > "$lock/info"
  nice -n 10 /usr/bin/time -l "$PY" -I adsb_day_r17a.py --day "$day" --manifest manifest_preferred.tsv --out out --raw raw \
    --limit-rate 9500K "$@" > "logs/$day.log" 2>&1
  local rc=$?
  rm -rf "$lock"
  echo "$(ts) pilot $day ($*) rc $rc"
  return $rc
}
run 1 "$D1" &
sleep 2
run 2 "$D2" --stream &
wait
echo "$(ts) pilot end"
