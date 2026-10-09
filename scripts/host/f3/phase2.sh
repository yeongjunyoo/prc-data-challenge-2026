#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 final root, phase 2: waits for R17-A's package (${PRC_WORK:-.}/r17a_sbx/final/manifest.json), then
#   1. research/f3/adsb_final_tables.py -> cache/adsb_v2.parquet, cache/adsb_v1nd.parquet
#   2. python -m src.feats_adsb feats_v27 --adsb adsb_v2 --mode raw -> cache/feats_v27.parquet
#   3. research/f3/check_v27.py (criteria fixed before the table exists); on pass, touches research/f3/out/feats_v27.PASS and starts the
#      LightGBM phase (research/f3/final_orch.py lgb --budget 10 --max_jobs 2) in the background; on fail stops here
#   4. research/f3/geo_final.py -> cache/geo_v1nd.parquet (1 thread, beside the legs)
#   5. sha16 list of cache/ -> research/f3/out/cache_sha16.txt
#   nohup bash research/f3/phase2.sh >> logs/phase2.log 2>&1 < /dev/null &
set -u
cd ${PRC_WORK:-.}/final_run || exit 1
P=${PRC_WORK:-.}
FIN=$P/r17a_sbx/final
PY=${PY_MAC:-python3}
STEP="bash research/f3/mac_step.sh"
ts() { date '+%m-%d %H:%M:%S'; }
echo "$(ts) phase2 waiting for $FIN/manifest.json"
until [ -e "$FIN/manifest.json" ] && [ -e "$FIN/adsb_final.parquet" ] && [ -e "$FIN/adsb_v1final.parquet" ]; do sleep 60; done
sleep 30
echo "$(ts) package present: $(ls $FIN | tr '\n' ' ')"
[ -e cache/adsb_v2.parquet ] || $STEP adsbtab 5 1 logs/adsb_final_tables.log $PY -X utf8 research/f3/adsb_final_tables.py $P/r16b_sbx/cache $FIN || { echo "$(ts) adsb tables failed"; exit 2; }
grep -v "^ " logs/adsb_final_tables.log | head -4 | cut -c1-1500
[ -e cache/feats_v27.parquet ] || $STEP feats27 6 1 logs/feats_v27.log $PY -X utf8 -m src.feats_adsb feats_v27 --adsb adsb_v2 --mode raw || { echo "$(ts) feats_v27 failed"; exit 3; }
$STEP check27 7 1 logs/check_v27.log $PY -X utf8 research/f3/check_v27.py $P/r16b_sbx/cache/feats_v27.parquet
rc=$?
grep -v "^ " logs/check_v27.log | head -60
if [ $rc = 0 ]; then
  touch research/f3/out/feats_v27.PASS
  echo "$(ts) feats_v27 PASS: starting the LightGBM phase"
  nohup $PY -u research/f3/final_orch.py lgb --budget 10 --max_jobs 2 >> logs/orch_lgb.log 2>&1 < /dev/null &
else
  touch research/f3/out/feats_v27.FAIL
  echo "$(ts) feats_v27 check FAILED: legs not started"
  exit 4
fi
[ -e cache/geo_v1nd.parquet ] || $STEP geo 5 1 logs/geo_final.log $PY -X utf8 research/f3/geo_final.py $P/r16b_sbx/cache research/f3/in/obs_geo.parquet $FIN || echo "$(ts) geo failed"
tail -30 logs/geo_final.log | grep -v "^ \{8\}"
(for f in cache/*.parquet cache/*.json; do echo "$(shasum -a 256 "$f" | cut -c1-16) $f"; done) > research/f3/out/cache_sha16.txt
echo "$(ts) phase2 done"
