#!/bin/bash
# F3 second rebuild, phase 4a (2026-10-09 23:5x KST, coordinator 23:2x): the Default final = 531 (scale2 0.75 in every corrector stage)
# on the four-month file, name 592 (..ff2). research/f3/collect_s75.py clones 531's val and fwd corrector caches and c3 es files from
# var1_sbx (es checked against the live root), then chains B, C, A with --cfg s75 from corr (correctors from the fit cache; any fit is a
# miss and stops the chain), cavg, manifest, tail, avg3 --cfg s75, and the final file in research/f3/out/ff2/ with v36 and 531's float
# test as the Jan/Jul references. Threads as phase3c beside R17-A's L3: F3_MAXTH = F3CAP = 3, TOTCAP = 15, the logged total-cap override
# after 300 s (research/f3/CAP_OVERRIDE_S, removed on exit), memory floor 30 % in mac_step.sh.
#   nohup bash research/f3/phase4a.sh >> logs/phase4a.log 2>&1 < /dev/null &
set -u
cd ~/nova-adsb/prcm/final_run || exit 1
P=~/nova-adsb/prcm
PY=~/nova-adsb/venv311/bin/python
SPEC=m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap
export F3_MAXTH=3 F3CAP=3 TOTCAP=15
ts() { date '+%m-%d %H:%M:%S'; }
h() { shasum -a 256 "$1" | cut -c1-16; }
echo "$(ts) phase4a start (final_chain.py $(h research/f3/final_chain.py), collect_s75.py $(h research/f3/collect_s75.py), final_file.py $(h research/f3/final_file.py), mac_step.sh $(h research/f3/mac_step.sh), src/manifest.py $(h src/manifest.py))"
[ -e STOP_F3 ] && { echo "$(ts) STOP_F3 present: stop"; exit 1; }
[ -e preds/manifest_$SPEC.json ] && { echo "$(ts) manifest for $SPEC exists: stop"; exit 1; }
$PY -X utf8 research/f3/collect_s75.py || { echo "$(ts) collect_s75 failed"; exit 2; }
echo 300 > research/f3/CAP_OVERRIDE_S
trap 'rm -f research/f3/CAP_OVERRIDE_S' EXIT
for L in B C A; do
  $PY -u research/f3/final_chain.py chain $L --from corr --cfg s75 >> logs/final_chain_s75_$L.log 2>&1 || { echo "$(ts) chain $L failed"; exit 4; }
  echo "$(ts) chain $L done"
done
$PY -u research/f3/final_chain.py avg3 --cfg s75 >> logs/final_chain_s75_avg3.log 2>&1 || { echo "$(ts) avg3 failed"; exit 6; }
[ -e preds/manifest_$SPEC.json ] || { echo "$(ts) no manifest for $SPEC: stop"; exit 6; }
echo "$(ts) avg3 done: $SPEC"
mkdir -p research/f3/out/ff2
bash research/f3/mac_step.sh s75_finalfile 4 1 logs/final_file_s75.log $PY -X utf8 research/f3/final_file.py $SPEC research/f3/out/ff2/spirited-tiger_final.parquet \
  research/f3/in/spirited-tiger_v36.parquet research/f3/in/test_$SPEC.parquet $P/r16b_sbx/data/ranking.parquet || { echo "$(ts) final file failed"; exit 7; }
echo "$(ts) final file written: research/f3/out/ff2/spirited-tiger_final.parquet ($(h research/f3/out/ff2/spirited-tiger_final.parquet))"
echo "$(ts) phase4a done"
