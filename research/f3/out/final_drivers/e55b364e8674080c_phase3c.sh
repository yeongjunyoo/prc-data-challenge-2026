#!/bin/bash
# F3 final root, phase 3c (2026-10-09 14:1x KST): resumes phase3b.sh after its stop at 14:10:18 in chain B's manifest step.
# src/manifest.py asserted the test file has 344,841 rows (the public template); the final template has 670,790. Patched in this root
# only (src/manifest.py is the recording tool, not in any manifest's code_sha): the row count is now the template's own. The failed run's
# partial output preds/fwd_<chain B TCS>.parquet is removed (must_be_new), then: chain B --from manifest, chain C, chain A --from add,
# avg3, TabM move, final file, daily candidate. Threads as phase3b (F3_MAXTH = F3CAP = 3, TOTCAP = 15) and a logged total-cap override
# after 300 s of waiting on the total cap alone (research/f3/CAP_OVERRIDE_S, coordinator 10-09 14:1x).
#   nohup bash research/f3/phase3c.sh >> logs/phase3.log 2>&1 < /dev/null &
set -u
cd ~/nova-adsb/prcm/final_run || exit 1
P=~/nova-adsb/prcm
PY=~/nova-adsb/venv311/bin/python
SPEC=m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgjaavskap
export F3_MAXTH=3 F3CAP=3 TOTCAP=15
ts() { date '+%m-%d %H:%M:%S'; }
echo "$(ts) phase3c start (final_chain.py $(shasum -a 256 research/f3/final_chain.py | cut -c1-16), mac_step.sh $(shasum -a 256 research/f3/mac_step.sh | cut -c1-16), src/manifest.py $(shasum -a 256 src/manifest.py | cut -c1-16))"
grep -q "assert len(t) == len(tpl) and set(t.index)" src/manifest.py || { echo "$(ts) src/manifest.py not patched: stop"; exit 2; }
echo 300 > research/f3/CAP_OVERRIDE_S
$PY -u research/f3/final_chain.py chain B --from manifest >> logs/final_chain_B.log 2>&1 || { echo "$(ts) chain B failed"; exit 4; }
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
rm -f research/f3/CAP_OVERRIDE_S
echo "$(ts) phase3c done"
