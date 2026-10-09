#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 final root, phase 3 (option B, 2026-10-09 03:4x KST), after the CatBoost and RealMLP phases of phase2c.sh:
#   1. waits until the RealMLP phase has finished (logs/orch_rm.log) with no failed job
#   2. moves the CUDA TabM output (sha16 1fd9f495c0bfb04a, PC GPU 03:13-03:26) to research/f3/out/tabm_cuda/ and runs tmcwin_s0 on MPS with
#      514's exact command (final_orch.py tabm --only tmcwin_s0, 3 threads) in the background
#   3. beside it, one step at a time at 4 threads: chain A blend; chain B; chain C (research/f3/final_chain.py)
#   4. when the MPS TabM has ended: removes final_run/FIRST_CLAIM, touches f2_run/GO_FJ (the last memory-heavy Mac step of F3), then chain
#      A from add (blend_add with the MPS TabM, corrector, tail)
#   5. avg3; research/f3/tabm_move.py (CUDA vs MPS full TabM, move by month); the final file and its checks (research/f3/final_file.py);
#      the daily-format candidate (research/f3/daily_candidate.py)
# Any failure stops here (exit code), nothing is retried.
#   nohup bash research/f3/phase3.sh >> logs/phase3.log 2>&1 < /dev/null &
set -u
cd ${PRC_WORK:-.}/final_run || exit 1
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
SPEC=m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgjaavskap
ts() { date '+%m-%d %H:%M:%S'; }
echo "$(ts) phase3 waiting for the RealMLP phase"
until grep -q "phase rm finished" logs/orch_rm.log 2>/dev/null; do sleep 60; done
grep "phase rm finished" logs/orch_rm.log | tail -1
grep -q "phase rm finished; failed: \[\]" logs/orch_rm.log || { echo "$(ts) RealMLP phase has failures: stop"; exit 2; }
grep -q "phase cb finished; failed: \[\]" logs/orch_cb.log || { echo "$(ts) CatBoost phase has failures: stop"; exit 2; }
# 2. CUDA TabM aside, MPS TabM
mkdir -p research/f3/out/tabm_cuda
if [ -e cache/full_m6r27tmcwin_s0.parquet ] && [ ! -e research/f3/out/tabm_cuda/full_m6r27tmcwin_s0.parquet ]; then
  [ "$(shasum -a 256 cache/full_m6r27tmcwin_s0.parquet | cut -c1-16)" = 1fd9f495c0bfb04a ] || { echo "$(ts) cache TabM is not the CUDA file: stop"; exit 3; }
  mv cache/full_m6r27tmcwin_s0.parquet research/f3/out/tabm_cuda/
  [ -e preds/es_valid_ids_m6r27tmcwin_full_s0.parquet ] && mv preds/es_valid_ids_m6r27tmcwin_full_s0.parquet research/f3/out/tabm_cuda/
fi
echo "$(ts) TabM on MPS (514's command)"
$PY -u research/f3/final_orch.py tabm --only tmcwin_s0 --budget 3 --max_jobs 1 --cap 9 >> logs/orch_tabm.log 2>&1 &
TP=$!
# 3. chain A blend, chains B and C
$PY -u research/f3/final_chain.py chain A --to blend >> logs/final_chain_A.log 2>&1 || { echo "$(ts) chain A blend failed"; exit 4; }
echo "$(ts) chain A blend done"
$PY -u research/f3/final_chain.py chain B >> logs/final_chain_B.log 2>&1 || { echo "$(ts) chain B failed"; exit 4; }
echo "$(ts) chain B done"
$PY -u research/f3/final_chain.py chain C >> logs/final_chain_C.log 2>&1 || { echo "$(ts) chain C failed"; exit 4; }
echo "$(ts) chain C done"
# 4. after the MPS TabM
wait $TP
grep -q "tmcwin_s0 ended rc 0, outputs present" logs/orch_tabm.log || { echo "$(ts) MPS TabM failed: stop (fallback: CUDA TabM, we decide)"; exit 5; }
rm -f FIRST_CLAIM && touch $P/f2_run/GO_FJ && echo "$(ts) MPS TabM done: FIRST_CLAIM removed, f2_run/GO_FJ touched"
$PY -u research/f3/final_chain.py chain A --from add >> logs/final_chain_A.log 2>&1 || { echo "$(ts) chain A failed"; exit 4; }
echo "$(ts) chain A done"
# 5. three-chain average, TabM move, file, daily candidate
$PY -u research/f3/final_chain.py avg3 >> logs/final_chain_avg3.log 2>&1 || { echo "$(ts) avg3 failed"; exit 6; }
[ -e preds/manifest_$SPEC.json ] || { echo "$(ts) no manifest for $SPEC: stop"; exit 6; }
echo "$(ts) avg3 done: $SPEC"
$PY -X utf8 research/f3/tabm_move.py research/f3/out/tabm_cuda/full_m6r27tmcwin_s0.parquet cache/full_m6r27tmcwin_s0.parquet > logs/tabm_move.log 2>&1
tail -20 logs/tabm_move.log
bash research/f3/mac_step.sh finalfile 4 1 logs/final_file.log $PY -X utf8 research/f3/final_file.py $SPEC research/f3/out/spirited-tiger_final.parquet \
  research/f3/in/spirited-tiger_v35.parquet research/f3/in/test_$SPEC.parquet $P/r16b_sbx/data/ranking.parquet || { echo "$(ts) final file failed"; exit 7; }
echo "$(ts) final file written: research/f3/out/spirited-tiger_final.parquet ($(shasum -a 256 research/f3/out/spirited-tiger_final.parquet | cut -c1-16))"
bash research/f3/mac_step.sh dailycand 6 1 logs/daily_candidate.log $PY -X utf8 research/f3/daily_candidate.py $SPEC f3f1_$SPEC $P/r16b_sbx/data/ranking.parquet \
  $P/r16b_sbx/data/submitting.parquet research/f3/in/test_$SPEC.parquet research/f3/in/spirited-tiger_v35.parquet research/f3/queue_ready || { echo "$(ts) daily candidate failed"; exit 8; }
echo "$(ts) daily candidate written: research/f3/queue_ready/f3f1_$SPEC.parquet"
echo "$(ts) phase3 done"
