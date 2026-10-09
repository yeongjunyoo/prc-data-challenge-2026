#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# R17-A PC download and extraction runner (private notes research/2026-10-08-r17-a-adsb-fill.md section 4). Same days file and same claim namespace as the
# Mac runner (${ADSB_WORK:-.}/dl_r17a.sh): a day is claimed with `ssh ${MAC_HOST:-mac} mkdir ${ADSB_WORK:-.}/claim/<day>` (atomic; the Mac workers mkdir in
# the same folder), extracted here by research/r17_a/adsb_day_r17a.py --stream --limit-rate <rate> --nice (BelowNormal) under
# scripts/memcap.py 1.5 GB into <pc adsb work>/out/<day>, then copied to the Mac (${ADSB_WORK:-.}/out/<day>/) and checked there.
# A day starts only when the PC guard holds (commit < 85 % and >= 4 GB free) and the Mac holds at most 2 r17a_dl*.lock (the curl caps of all
# running streams stay <= 20 MB/s: Mac 2 x 4800K + PC 2 x 4800K). A failed day is moved to failed/<day>.<try> and retried up to 3 times.
# Stop: touch <pc adsb work>/STOP.
#   powershell -NoProfile -ExecutionPolicy Bypass -File research/r17_a/detach.ps1 "bash pc_dl_r17a.sh 2 4800K days_r17a_all.txt"
set -u
W=${1:?workers}; RATE=${2:?rate}; DAYS=${3:?days file}
HERE=${PC_REPO:-.}/research/r17_a
WD=${ADSB_PC_WORK:-.}
PY=${PY_ADSB_PC:-python}
MEMCAP=${MEMCAP:-memcap.py}
mkdir -p "$WD/out" "$WD/logs" "$WD/failed"
cd "$WD" || exit 1
cp "$HERE/adsb_day_r17a.py" "$HERE/manifest_preferred.tsv" "$WD/"
ts() { date '+%m-%d %H:%M:%S'; }
guard_ok() {
  local v
  v=$(powershell -NoProfile -Command '$o=Get-CimInstance Win32_OperatingSystem; $c=(Get-Counter "\Memory\% Committed Bytes In Use").CounterSamples[0].CookedValue; "{0:N0} {1:N2}" -f $c, ($o.FreePhysicalMemory/1MB)' 2>/dev/null | tr -d '\r')
  local c=${v%% *} f=${v##* }
  [ -n "$c" ] && [ "$(awk -v c="${c//,/}" -v f="${f//,/}" 'BEGIN{print ((c < 85 && f >= 4) ? 1 : 0)}')" = 1 ]
}
mac_streams() { ssh -n -o BatchMode=yes -o ConnectTimeout=20 ${MAC_HOST:-mac} 'ls -d ${PRC_WORK:-.}/r17a_dl*.lock 2>/dev/null | wc -l' 2>/dev/null | tr -d ' \r'; }
worker() {
  local w=$1 day try rc t0 t1 n
  while read -r day <&3; do
    day=$(echo "$day" | tr -d '\r')
    [ -n "$day" ] || continue
    [ -e STOP ] && { echo "$(ts) pc w$w STOP seen"; return 0; }
    [ -s "out/$day/meta.json" ] && continue
    ssh -n -o BatchMode=yes -o ConnectTimeout=20 ${MAC_HOST:-mac} "test -s ${ADSB_WORK:-.}/out/$day/meta.json" && continue
    ssh -n -o BatchMode=yes -o ConnectTimeout=20 ${MAC_HOST:-mac} "mkdir ${ADSB_WORK:-.}/claim/$day" 2>/dev/null || continue
    for try in 1 2 3; do
      n=0
      while [ $n -lt 2 ]; do
        if guard_ok && [ "$(mac_streams)" -le 2 ]; then n=$((n+1)); else n=0; fi
        [ $n -lt 2 ] && sleep 30
      done
      t0=$(date +%s)
      python -X utf8 "$MEMCAP" --gb 1.5 -- "$PY" -I adsb_day_r17a.py --day "$day" --manifest manifest_preferred.tsv --out out --stream \
        --limit-rate "$RATE" --nice > "logs/$day.log" 2>&1
      rc=$?
      t1=$(date +%s)
      if [ $rc -eq 0 ] && [ -s "out/$day/meta.json" ]; then
        if tar cf - -C out "$day" | ssh -o BatchMode=yes ${MAC_HOST:-mac} 'cd ${ADSB_WORK:-.}/out && tar xf -' && \
           ssh -n -o BatchMode=yes ${MAC_HOST:-mac} "test -s ${ADSB_WORK:-.}/out/$day/meta.json && test -s ${ADSB_WORK:-.}/out/$day/events.parquet && test -s ${ADSB_WORK:-.}/out/$day/points.parquet"; then
          echo "$(ts) pc w$w $day ok try $try wall $((t1 - t0)) s | $("$PY" -I -c "import json,sys; m=json.load(open(sys.argv[1])); print('MB', round(m['bytes']/1e6), 'MB/s', round(m['bytes']/1e6/max(m['wall_s'],1),1), 'parse_cpu', m['parse_cpu_s'], 'gz/bytes', round(m['gz_bytes']/m['bytes'],3), 'events', m['n_events'], 'bad', m.get('bad_members',0), 'rss_mb', m.get('max_rss_mb'))" "out/$day/meta.json") | copied to Mac"
          break
        fi
        echo "$(ts) pc w$w $day extracted but the copy to the Mac failed (try $try); retry the copy next round"
        rc=8
      fi
      echo "$(ts) pc w$w $day FAILED try $try rc $rc wall $((t1 - t0)) s: $(tail -2 "logs/$day.log" | tr '\n' ' ' | cut -c1-300)"
      [ $rc -eq 8 ] || { mv "out/$day" "failed/$day.$try" 2>/dev/null; mv "logs/$day.log" "failed/$day.$try.log" 2>/dev/null; }
      sleep 60
      if [ $rc -eq 8 ] && tar cf - -C out "$day" | ssh -o BatchMode=yes ${MAC_HOST:-mac} 'cd ${ADSB_WORK:-.}/out && tar xf -'; then
        echo "$(ts) pc w$w $day copied to Mac on retry"; break
      fi
    done
  done 3< "$HERE/$DAYS"
  echo "$(ts) pc w$w done"
}
echo "$(ts) PC runner start pid $$: $W workers at $RATE each, days $DAYS"
for w in $(seq 1 "$W"); do worker "$w" & sleep 20; done
wait
echo "$(ts) PC runner end"
