#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# 2026-01-29: the preferred staging-0 archive has a corrupt gzip member (traces/15/trace_full_ac3315.json) and the tar
# stream ends after it (33,409 of about 66,000 traces). Re-take that day from the prod-0 release (README: "Download both
# files from the prod or staging release"). Waits for route A's downloads (keeps 2 connections) and its last extraction
# (keeps one extractor), then downloads prod-0 with 2 connections into raw_alt and re-extracts into out/2026-01-29.
set -u
ROOT=${ROOT:-./prc-adsb}
WROOT=${WROOT:-$ROOT}
PY=${PY:-python}
MEMCAP_CMD=${MEMCAP_CMD:-}  # optional memory-cap prefix, for example: python memcap.py --gb 2 --
LOG=$ROOT/routeA.log
MAN=$WROOT/manifest_fix_0129.tsv
until grep -q ALL_DOWNLOADS_DONE $ROOT/logs/routeA_dl.log 2>/dev/null; do sleep 30; done
mkdir -p $ROOT/raw_alt/2026-01-29
B=https://github.com/adsblol/globe_history_2026/releases/download/v2026.01.29-planes-readsb-prod-0
fetch() {
  url=$1; size=$2; path=$3
  for attempt in 1 2 3 4 5 6; do
    have=$(stat -c %s "$path" 2>/dev/null || echo 0)
    [ "$have" = "$size" ] && return 0
    curl -sSL --retry 5 --retry-delay 10 --connect-timeout 30 -C - -o "$path" "$url" || sleep 20
  done
  have=$(stat -c %s "$path" 2>/dev/null || echo 0)
  [ "$have" = "$size" ]
}
fetch $B/v2026.01.29-planes-readsb-prod-0.tar.aa 2000000000 $ROOT/raw_alt/2026-01-29/v2026.01.29-planes-readsb-prod-0.tar.aa &
p1=$!
fetch $B/v2026.01.29-planes-readsb-prod-0.tar.ab 1285065728 $ROOT/raw_alt/2026-01-29/v2026.01.29-planes-readsb-prod-0.tar.ab &
p2=$!
wait $p1; r1=$?; wait $p2; r2=$?
if [ $r1 != 0 ] || [ $r2 != 0 ]; then echo "$(date '+%F %T') ERROR 2026-01-29 prod-0 download failed" >> $LOG; exit 1; fi
until grep -q ROUTE_A_DONE $LOG 2>/dev/null; do sleep 15; done
mkdir -p $ROOT/out/2026-01-29_staging_partial
mv $ROOT/out/2026-01-29/* $ROOT/out/2026-01-29_staging_partial/ 2>/dev/null
OMP_NUM_THREADS=1 $MEMCAP_CMD "$PY" "$WROOT/adsb_day.py" --day 2026-01-29 --manifest "$MAN" \
   --raw "$WROOT/raw_alt" --out "$WROOT/out" --no-download --keep-raw --nice > "$ROOT/logs/x_2026-01-29_prod.log" 2>&1
rc=$?
info=$("$PY" -c "import json;m=json.load(open('$WROOT/out/2026-01-29/meta.json'));print('traces=%s gz_GB=%.2f events=%s bad=%s' % (m['traces'], m['gz_bytes']/1e9, m['n_events'], m.get('bad_members', 0)))" 2>&1)
echo "$(date '+%F %T') DAY 2026-01-29 prod-0 rc=$rc $info (replaces the partial staging extract, kept in out/2026-01-29_staging_partial)" >> $LOG
echo "$(date '+%F %T') FIX_0129_DONE" >> $LOG
