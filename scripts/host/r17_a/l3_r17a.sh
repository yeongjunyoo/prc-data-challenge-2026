#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# R17-A L3 orchestrator on the Mac (private notes research/2026-10-08-r17-a-adsb-fill.md section 1, the pre-registered L3), from the sandbox
# ${PRC_WORK:-.}/r17a_sbx. Holds no lock while it waits; every step takes its own lock through research/r17_a/sb_build_r17a.py.
#   1. waits until all 187 new 2025 days are extracted (${ADSB_WORK:-.}/out/2025-*/meta.json; the full fill the registration asks for)
#   2. sb_build_r17a.py feats   (adsb_v3, feats_v28 and the equality check; 1 thread, about 1 min)
#   3. waits for our go: ${PRC_WORK:-.}/r17a_L3.GO (the final rebuild's legs hold the Mac until then)
#   4. sb_build_r17a.py legs --lanes 2, and chain C, A, B and final side by side (each waits for its inputs); final writes prcm/r17a_L3.DONE
# Stops at the first failed step (marker research/r17_a/l3.FAILED with the step name).
#   nohup /bin/bash research/r17_a/l3_r17a.sh >> research/r17_a/logs/l3.log 2>&1 < /dev/null &
cd ${PRC_WORK:-.}/r17a_sbx || exit 1
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "$(ts) FAILED: $1"; echo "$1" > research/r17_a/l3.FAILED; exit 1; }
n25() { local n=0 d; for d in ${ADSB_WORK:-.}/out/2025-??-??; do [ -s "$d/meta.json" ] && n=$((n+1)); done; echo $n; }
echo "$(ts) L3 orchestrator pid $$: waiting for the 187 new 2025 days (now $(n25))"
until [ "$(n25)" -ge 187 ]; do sleep 120; done
echo "$(ts) 187 new 2025 days present"
"$PY" -X utf8 -u research/r17_a/sb_build_r17a.py feats > research/r17_a/logs/sb_feats.log 2>&1 || fail feats
tail -3 research/r17_a/logs/sb_feats.log
echo "$(ts) feats done; waiting for $P/r17a_L3.GO"
until [ -e "$P/r17a_L3.GO" ]; do sleep 60; done
echo "$(ts) GO seen; starting legs and chains"
"$PY" -X utf8 -u research/r17_a/sb_build_r17a.py legs --lanes 2 > research/r17_a/logs/sb_legs.log 2>&1 &
pl=$!
for ch in C A B; do
  "$PY" -X utf8 -u research/r17_a/sb_build_r17a.py chain --chain $ch > research/r17_a/logs/sb_chain_$ch.log 2>&1 &
  eval "pc$ch=\$!"
  sleep 5
done
"$PY" -X utf8 -u research/r17_a/sb_build_r17a.py final > research/r17_a/logs/sb_final.log 2>&1 &
pf=$!
wait $pl || { kill $pcC $pcA $pcB $pf 2>/dev/null; fail legs; }   # the chains and final only wait for leg markers here
for ch in C A B; do eval "p=\$pc$ch"; wait $p || { kill $pf 2>/dev/null; fail "chain $ch"; }; done   # final only waits for chain markers
wait $pf || fail final
echo "$(ts) L3 done: $(grep -E 'gate .* vs 514' research/r17_a/logs/sb_final.log | tail -1 | cut -c1-300)"
