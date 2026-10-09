#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Route A: download the remaining Jan+Jul 2025 and Jan+Jul 2026 adsb.lol days to $WROOT/raw (kept) with
# PARALLEL connections, and extract each finished day one at a time (memcap 2 GB, single thread, below-normal).
# Days already extracted (out/<day>/events.parquet and meta.json) are skipped, partial downloads resume.
# Usage: bash routeA.sh <manifest.tsv> <days file>   (set ROOT and WROOT to the working folder; default ./prc-adsb)
# Progress: $ROOT/routeA.log (one line per finished day, plus error lines)
set -u
ROOT=${ROOT:-./prc-adsb}
WROOT=${WROOT:-$ROOT}
PY=${PY:-python}
MEMCAP_CMD=${MEMCAP_CMD:-}  # optional memory-cap prefix, for example: python memcap.py --gb 2 --
MAN=$1
DAYS=($(cat "$2"))
PARALLEL=${PARALLEL:-2}
LOG=$ROOT/routeA.log
mkdir -p $ROOT/raw $ROOT/out $ROOT/logs
echo "$(date '+%F %T') START days=${#DAYS[@]} parallel=$PARALLEL pid=$$" >> $LOG

done_day() { [ -s "$ROOT/out/$1/events.parquet" ] && [ -s "$ROOT/out/$1/meta.json" ]; }

: > $ROOT/logs/routeA_parts.tsv
TODO=()
for d in "${DAYS[@]}"; do
  if done_day "$d"; then continue; fi
  line=$(awk -F'\t' -v d="$d" '$1==d' "$MAN" | head -1)
  if [ -z "$line" ]; then echo "$(date '+%F %T') ERROR $d not in manifest" >> $LOG; continue; fi
  TODO+=("$d")
  tag=$(echo "$line" | cut -f2); repo=$(echo "$line" | cut -f3)
  names=$(echo "$line" | cut -f4); sizes=$(echo "$line" | cut -f5)
  IFS=',' read -ra N <<< "$names"; IFS=',' read -ra S <<< "$sizes"
  mkdir -p $ROOT/raw/$d
  for i in "${!N[@]}"; do
    printf '%s\t%s\t%s\t%s\n' "$d" "https://github.com/adsblol/$repo/releases/download/$tag/${N[$i]}" "${S[$i]}" "$ROOT/raw/$d/${N[$i]}" >> $ROOT/logs/routeA_parts.tsv
  done
done

fetch() {
  d=$1; url=$2; size=$3; path=$4
  t0=$(date +%s)
  for attempt in 1 2 3 4 5 6; do
    have=$(stat -c %s "$path" 2>/dev/null || echo 0)
    [ "$have" = "$size" ] && break
    curl -sSL --retry 5 --retry-delay 10 --connect-timeout 30 -C - -o "$path" "$url" || sleep 20
  done
  have=$(stat -c %s "$path" 2>/dev/null || echo 0)
  t1=$(date +%s)
  if [ "$have" != "$size" ]; then echo "$(date '+%F %T') ERROR download $d $(basename $path) have=$have want=$size" >> $ROOT/routeA.log; fi
  echo "$(date '+%F %T') $d $(basename $path) have=$have want=$size sec=$((t1-t0))" >> $ROOT/logs/routeA_dl.log
}
export -f fetch
export ROOT

( tr '\t' '\n' < $ROOT/logs/routeA_parts.tsv | xargs -d '\n' -n 4 -P $PARALLEL bash -c 'fetch "$@"' _ ; echo "$(date '+%F %T') ALL_DOWNLOADS_DONE" >> $ROOT/logs/routeA_dl.log ) &

for d in "${TODO[@]}"; do
  while true; do
    ok=1; bytes=0
    while IFS=$'\t' read -r dd url size path; do
      [ "$dd" != "$d" ] && continue
      have=$(stat -c %s "$path" 2>/dev/null || echo 0)
      [ "$have" != "$size" ] && ok=0
      bytes=$((bytes+size))
    done < $ROOT/logs/routeA_parts.tsv
    [ $ok = 1 ] && break
    if grep -q ALL_DOWNLOADS_DONE $ROOT/logs/routeA_dl.log 2>/dev/null; then echo "$(date '+%F %T') ERROR $d incomplete after downloads ended" >> $LOG; break; fi
    sleep 15
  done
  [ $ok = 1 ] || continue
  t0=$(date +%s)
  OMP_NUM_THREADS=1 $MEMCAP_CMD "$PY" "$WROOT/adsb_day.py" --day "$d" --manifest "$MAN" \
     --raw "$WROOT/raw" --out "$WROOT/out" --no-download --keep-raw --nice > "$ROOT/logs/x_$d.log" 2>&1
  rc=$?
  t1=$(date +%s)
  if [ $rc = 0 ] && done_day "$d"; then
    info=$("$PY" -c "import json;m=json.load(open('$WROOT/out/$d/meta.json'));print('GB=%.2f parse_cpu_s=%s events=%s kept_ground=%s' % (m['bytes']/1e9, m['parse_cpu_s'], m['n_events'], sum(m['kept_aircraft_ground'].values())))")
    peak=$(grep -o 'peak_job=[0-9.]*GB' "$ROOT/logs/x_$d.log" | tail -1)
    echo "$(date '+%F %T') DAY $d ok $info wall_s=$((t1-t0)) $peak" >> $LOG
  else
    echo "$(date '+%F %T') ERROR extract $d rc=$rc (see logs/x_$d.log)" >> $LOG
  fi
done
wait
echo "$(date '+%F %T') ROUTE_A_DONE" >> $LOG
