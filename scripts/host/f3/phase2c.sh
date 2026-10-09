#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 final root, phase 2c (R17-A pack 2, coordinator 2026-10-09 02:1x KST; replaces phase2b.sh, which stopped at its own adsb step on pack
# 1 at 02:06 and built nothing from it):
#   1. research/f3/chk_adsb_pack2.py: pack-2 merged tables (sha16 54ba81f368669d1d and 07df03ca69833082 as in the manifest's pack2 block,
#      today's schema, 2025 rows bit-equal per id to today's adsb_v2 and adsb_v1nd, 2026-06-11 absent) -> cache/adsb_v2, cache/adsb_v1nd
#   2. python -m src.feats_adsb feats_v27 --adsb adsb_v2 --mode raw -> cache/feats_v27.parquet
#   3. research/f3/check_v27.py (criteria fixed before the table existed); on fail stops here (research/f3/out/feats_v27.FAIL)
#   4. on pass, as phase2b: STOP_F3 in the interim root; research/f3/cmp_v27_interim.py; geo_final.py beside (pack-2 no-data day left
#      out); the LightGBM phase through fitwrap load (jobs still running in the interim root waited for, then loaded); the sk s0 twin
#      check beside; the CatBoost phase through fitwrap load; the RealMLP phase; the cache sha16 list.
#   nohup bash research/f3/phase2c.sh >> logs/phase2.log 2>&1 < /dev/null &
set -u
cd ${PRC_WORK:-.}/final_run || exit 1
P=${PRC_WORK:-.}
FIN=$P/r17a_sbx/final
IR=$P/f3_sbx/interim
FC=$P/final_run/research/f3/fc_legs
PY=${PY_MAC:-python3}
STEP="bash research/f3/mac_step.sh"
ORCH="$PY -u research/f3/final_orch.py"
ts() { date '+%m-%d %H:%M:%S'; }
[ -e $P/r17a_sbx/research/r17_a/final_pack2.DONE ] || { echo "$(ts) no final_pack2.DONE"; exit 1; }
echo "$(ts) phase2c: pack 2 present"
[ -e cache/adsb_v2.parquet ] || $STEP adsbpack2 6 1 logs/adsb_pack2.log $PY -X utf8 research/f3/chk_adsb_pack2.py $P/r16b_sbx/cache $FIN || { echo "$(ts) pack-2 check failed"; exit 2; }
grep -E '"pass"|"sha16"|rows_2025_identical|rows_2026_06_11' logs/adsb_pack2.log | head -20
[ -e cache/feats_v27.parquet ] || $STEP feats27 6 1 logs/feats_v27.log $PY -X utf8 -m src.feats_adsb feats_v27 --adsb adsb_v2 --mode raw || { echo "$(ts) feats_v27 failed"; exit 3; }
$STEP check27 7 1 logs/check_v27.log $PY -X utf8 research/f3/check_v27.py $P/r16b_sbx/cache/feats_v27.parquet
rc=$?
grep -v "^ " logs/check_v27.log | head -60
if [ $rc != 0 ]; then touch research/f3/out/feats_v27.FAIL; echo "$(ts) feats_v27 check FAILED: legs not started"; exit 4; fi
touch research/f3/out/feats_v27.PASS
touch $IR/STOP_F3
echo "$(ts) feats_v27 PASS; interim launches stopped"
WRAP="--wrap load:$FC --defer_to $IR"
grep -q '"PASS": true' $P/f3_sbx/wt/out/wrapper_test.json 2>/dev/null || { WRAP=""; echo "$(ts) wrapper test not passed: legs fit directly"; }
$STEP v27same 7 1 logs/v27_interim_vs_real.log $PY -X utf8 research/f3/cmp_v27_interim.py $IR/cache/feats_v27.parquet cache/feats_v27.parquet
echo "$(ts) interim vs real 2025 rows identical: rc $? (nonzero = stored fits will miss and refit)"
( [ -e cache/geo_v1nd.parquet ] || $STEP geo 5 1 logs/geo_final.log $PY -X utf8 research/f3/geo_final.py $P/r16b_sbx/cache research/f3/in/obs_geo.parquet $FIN || echo "$(ts) geo failed" ) &
( bash research/f3/twin.sh >> logs/twin.log 2>&1 ) &
echo "$(ts) LightGBM phase through fitwrap load"
$ORCH lgb $WRAP --budget 9 --cap 9 --max_jobs 3 >> logs/orch_lgb.log 2>&1
echo "$(ts) LightGBM phase ended: $(tail -1 logs/orch_lgb.log)"
$ORCH cb $WRAP --budget 9 --cap 9 --max_jobs 2 >> logs/orch_cb.log 2>&1
echo "$(ts) CatBoost phase ended: $(tail -1 logs/orch_cb.log)"
$ORCH rm --budget 9 --cap 9 --max_jobs 1 >> logs/orch_rm.log 2>&1
echo "$(ts) RealMLP phase ended: $(tail -1 logs/orch_rm.log)"
wait
(for f in cache/*.parquet cache/*.json; do echo "$(shasum -a 256 "$f" | cut -c1-16) $f"; done) > research/f3/out/cache_sha16.txt
echo "$(ts) phase2b done"
