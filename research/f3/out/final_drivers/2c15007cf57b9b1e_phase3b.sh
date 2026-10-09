#!/bin/bash
# F3 final root, phase 3b (coordinator 2026-10-09 13:5x KST): resumes phase3.sh after its stop at 10:48:37 (chain A blend: the XGBoost
# weight check's sandbox had no data/; fixed in final_chain.py by linking data/). Step 4 of phase3 (FIRST_CLAIM removed, f2_run/GO_FJ) was
# done by the coordinator at 13:50:08 and the MPS TabM ended 12:37:07, so this runs:
#   1. chain A --to blend again (the 10:48 blend output and its json move to research/f3/out/blendA_1048/ and are compared with the rerun),
#      then chain B, chain C, chain A --from add, avg3
#   2. research/f3/tabm_move.py (CUDA vs MPS full TabM), the final file and its checks, the daily-format candidate (Mac side)
# At most 3 threads beside F2's fj CatBoosts: every step declares min(its threads, F3_MAXTH = 3) (OMP_NUM_THREADS follows), F3's locks
# together stay within F3CAP = 3, and the declared total of all locks may reach TOTCAP = 15. Any failure stops here (exit code).
#   nohup bash research/f3/phase3b.sh >> logs/phase3.log 2>&1 < /dev/null &
set -u
cd ~/nova-adsb/prcm/final_run || exit 1
P=~/nova-adsb/prcm
PY=~/nova-adsb/venv311/bin/python
SPEC=m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgjaavskap
export F3_MAXTH=3 F3CAP=3 TOTCAP=15
ts() { date '+%m-%d %H:%M:%S'; }
echo "$(ts) phase3b start (final_chain.py $(shasum -a 256 research/f3/final_chain.py | cut -c1-16), mac_step.sh $(shasum -a 256 research/f3/mac_step.sh | cut -c1-16))"
[ "$(shasum -a 256 cache/full_m6r27tmcwin_s0.parquet | cut -c1-16)" = a4c0f7e9085c644c ] || { echo "$(ts) cache TabM is not the MPS run a4c0f7e9085c644c: stop"; exit 3; }
[ "$(shasum -a 256 research/f3/out/tabm_cuda/full_m6r27tmcwin_s0.parquet | cut -c1-16)" = 1fd9f495c0bfb04a ] || { echo "$(ts) CUDA TabM missing: stop"; exit 3; }
# 1. chain A blend again
B=research/f3/out/blendA_1048
if [ -e cache/full_m6r27wincbrmfw4sv434_s0.parquet ] && [ ! -e $B/full_m6r27wincbrmfw4sv434_s0.parquet ]; then
  mkdir -p $B && mv cache/full_m6r27wincbrmfw4sv434_s0.parquet notes/blend_fixed_m6r27wincbrmfw4sv434_full_s0.json $B/
fi
$PY -u research/f3/final_chain.py chain A --to blend >> logs/final_chain_A.log 2>&1 || { echo "$(ts) chain A blend failed"; exit 4; }
echo "$(ts) chain A blend done: $(grep -o '"zeros_vs_ones_identical": [a-z]*' research/f3/out/final_chain_A.json); rerun vs 10:48 output: $(shasum -a 256 cache/full_m6r27wincbrmfw4sv434_s0.parquet | cut -c1-16) / $(shasum -a 256 $B/full_m6r27wincbrmfw4sv434_s0.parquet | cut -c1-16)"
$PY -u research/f3/final_chain.py chain B >> logs/final_chain_B.log 2>&1 || { echo "$(ts) chain B failed"; exit 4; }
echo "$(ts) chain B done"
$PY -u research/f3/final_chain.py chain C >> logs/final_chain_C.log 2>&1 || { echo "$(ts) chain C failed"; exit 4; }
echo "$(ts) chain C done"
$PY -u research/f3/final_chain.py chain A --from add >> logs/final_chain_A.log 2>&1 || { echo "$(ts) chain A failed"; exit 4; }
echo "$(ts) chain A done"
$PY -u research/f3/final_chain.py avg3 >> logs/final_chain_avg3.log 2>&1 || { echo "$(ts) avg3 failed"; exit 6; }
[ -e preds/manifest_$SPEC.json ] || { echo "$(ts) no manifest for $SPEC: stop"; exit 6; }
echo "$(ts) avg3 done: $SPEC"
# 2. TabM move, file, daily candidate
$PY -X utf8 research/f3/tabm_move.py research/f3/out/tabm_cuda/full_m6r27tmcwin_s0.parquet cache/full_m6r27tmcwin_s0.parquet > logs/tabm_move.log 2>&1
echo "$(ts) tabm_move rc $?"
bash research/f3/mac_step.sh finalfile 4 1 logs/final_file.log $PY -X utf8 research/f3/final_file.py $SPEC research/f3/out/spirited-tiger_final.parquet \
  research/f3/in/spirited-tiger_v35.parquet research/f3/in/test_$SPEC.parquet $P/r16b_sbx/data/ranking.parquet || { echo "$(ts) final file failed"; exit 7; }
echo "$(ts) final file written: research/f3/out/spirited-tiger_final.parquet ($(shasum -a 256 research/f3/out/spirited-tiger_final.parquet | cut -c1-16))"
bash research/f3/mac_step.sh dailycand 6 1 logs/daily_candidate.log $PY -X utf8 research/f3/daily_candidate.py $SPEC f3f1_$SPEC $P/r16b_sbx/data/ranking.parquet \
  $P/r16b_sbx/data/submitting.parquet research/f3/in/test_$SPEC.parquet research/f3/in/spirited-tiger_v35.parquet research/f3/queue_ready || { echo "$(ts) daily candidate failed"; exit 8; }
echo "$(ts) daily candidate written: research/f3/queue_ready/f3f1_$SPEC.parquet"
echo "$(ts) phase3b done"
