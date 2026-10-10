#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# R17-A L3 resume after the chain B fix (10-10 02:0x, research/r17_a/fixB_r17a.sh): chain B from its state file (done = avg, so it starts
# at bfix with 514's B learners now in cache/), then final (chainavg3 1780 --suffix skap, the guard sync from the PC, the 7.8 gate against
# 514, prcm/r17a_L3.DONE). Chains A and C and all legs had finished (research/r17_a/done/chainA.done, chainC.done, legs.done). Holds no lock;
# every step takes its own (sb_build_r17a.py). A failed step writes research/r17_a/l3.FAILED with its name, as l3_r17a.sh did.
#   nohup /bin/bash research/r17_a/l3r_r17a.sh >> research/r17_a/logs/l3.log 2>&1 < /dev/null &
cd ${PRC_WORK:-.}/r17a_sbx || exit 1
PY=${PY_MAC:-python3}
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "$(ts) FAILED: $1"; echo "$1" > research/r17_a/l3.FAILED; exit 1; }
echo "$(ts) L3 resume pid $$: chain B from its state, then final"
"$PY" -X utf8 -u research/r17_a/sb_build_r17a.py chain --chain B > research/r17_a/logs/sb_chain_B2.log 2>&1 || fail "chain B (resume)"
echo "$(ts) chain B done: $(tail -1 research/r17_a/logs/sb_chain_B2.log | cut -c1-160)"
"$PY" -X utf8 -u research/r17_a/sb_build_r17a.py final > research/r17_a/logs/sb_final.log 2>&1 || fail final
echo "$(ts) L3 done: $(grep -E 'gate .* vs 514' research/r17_a/logs/sb_final.log | tail -1 | cut -c1-300)"
