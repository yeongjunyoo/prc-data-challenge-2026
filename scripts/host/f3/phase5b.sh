#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 L3 final, phase 5b (2026-10-10 04:5x KST; 04:44: the final form is Bb, so 598b is the primary build and 598a the comparison,
# like 594b and 594a). The 598 part of research/f3/phase5.sh with Bb first. phase5.sh keeps 596, 597's Mac part, the fj blends and the
# four-month correctors; its own 598a part stops at its first merge (research/f3/kit_l3.sh refuses merges unless KIT_BY=phase5b, so phase5.sh
# ends there with "merge A failed"). This script waits for that, then:
#   merges Bb, Ba, D of the three chains; the corrector-stage check (twins, every merged tag);
#   598b: manifests and tails of Bb (chains in parallel), avg3 Bb, check finals (only (iii)'s "Jan/Jul rows differ" lines may fail),
#         F2's iii_boundary.py against 596 (must pass), the final file ff7;
#   598a: the same for Ba (check finals then covers Ba and Bb), iii_boundary for Ba, the final file ff6.
# Same caps as phase5.sh (F3CAP 15, F3_MAXTH 3, the logged 300 s total-cap override, memory floor 30 % in mac_step.sh). STOP_F3 stops it.
#   nohup bash research/f3/phase5b.sh >> logs/phase5b.log 2>&1 < /dev/null &
set -u
cd ${PRC_WORK:-.}/final_run || exit 1
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
S5=m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap
S0=m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3r
export F3CAP=15 F3_MAXTH=3 KIT_BY=phase5b
unset TOTCAP
ts() { date '+%m-%d %H:%M:%S'; }
h() { shasum -a 256 "$1" | cut -c1-16; }
die() { echo "$(ts) $*"; touch research/f3/out/phase5b.FAILED; exit 2; }
echo "$(ts) phase5b start (phase5b.sh $(h research/f3/phase5b.sh), kit_l3.sh $(h research/f3/kit_l3.sh), mac_step.sh $(h research/f3/mac_step.sh)): waiting for phase5.sh to stop at its first merge"
while pgrep -f "research/f3/phase5[.]sh" > /dev/null; do
  [ -e STOP_F3 ] && die "STOP_F3 present"
  sleep 60
done
[ -e research/f3/out/l3_596.DONE ] || die "phase5.sh ended without 596 (logs/phase5.log: $(tail -1 logs/phase5.log))"
if ! { tail -1 logs/phase5.log | grep -q "merge A failed" && tail -3 logs/l3k_merge.log | grep -q "handed to phase5b"; }; then
  die "phase5.sh did not end at the merge hand-over (logs/phase5.log: $(tail -1 logs/phase5.log))"
fi
mv research/f3/out/phase5.FAILED research/f3/out/phase5.handed_to_phase5b
echo 300 > research/f3/CAP_OVERRIDE_S
trap 'rm -f research/f3/CAP_OVERRIDE_S' EXIT
echo "$(ts) phase5.sh ended at the hand-over as planned (596, 597's Mac part, fj blends and four-month correctors done)"
for X in A B C; do KIT_VARIANTS="Bb Ba D" bash research/f3/kit_l3.sh merge $X >> logs/l3k_merge.log 2>&1 || die "merge $X failed (logs/l3k_merge.log)"; done
bash research/f3/kit_l3.sh check >> logs/l3k_check.log 2>&1 || die "corrector-stage check failed (logs/l3k_check.log)"
echo "$(ts) merges (Bb Ba D) and the corrector-stage check done: $(grep -o 'SPLITCHECK pass [A-Za-z]*' logs/l3k_check_corr.log | tail -1)"
variant() {  # V final-dir name
  local V=$1 FF=$2 NM=$3 SV=${S0}$1amfhgkcxncrdbawgpglgjaavskap X ra rb rc
  for X in A B C; do
    ( trap - EXIT; bash research/f3/kit_l3.sh manifest $X $V && bash research/f3/kit_l3.sh tails $X $V ) >> logs/l3k_tails_$X$V.log 2>&1 &
    eval "PT_$X=$!"
  done
  wait $PT_A; ra=$?; wait $PT_B; rb=$?; wait $PT_C; rc=$?
  [ $ra = 0 ] && [ $rb = 0 ] && [ $rc = 0 ] || die "$V manifest or tails failed (A $ra, B $rb, C $rc)"
  bash research/f3/kit_l3.sh avg3 $V >> logs/l3k_avg3.log 2>&1 || die "avg3 $V failed"
  [ -e preds/manifest_$SV.json ] || die "no manifest for $SV"
  echo "$(ts) avg3 $V done: $SV"
  bash research/f3/kit_l3.sh check finals >> logs/l3k_check_finals_run.log 2>&1
  echo "$(ts) check finals rc $? (only (iii)'s boundary lines may fail)"
  cp -p logs/l3k_check_finals.log logs/l3k_check_finals_after$V.log
  cp -p research/f3/out/split_check_l3.json research/f3/out/split_check_l3_after$V.json
  $PY -c "import json,sys; f=json.load(open('research/f3/out/split_check_l3.json'))['fails']; print('fails:', f); sys.exit(0 if all(x in ('final Ba: Jan/Jul rows differ from the default final', 'final Bb: Jan/Jul rows differ from the default final') for x in f) else 1)" \
    || die "check finals failed beyond (iii)'s boundary lines"
  bash research/f3/mac_step.sh l3_iii_$V 4 1 logs/l3_iii_boundary_$V.log $PY -X utf8 -u research/f2/iii_boundary.py --default preds/manifest_$S5.json \
    --finals $V=preds/manifest_$SV.json --out research/f3/out/iii_boundary_l3_$V.json || die "iii_boundary $V did not pass (logs/l3_iii_boundary_$V.log)"
  echo "$(ts) iii_boundary $V pass"
  mkdir -p research/f3/out/$FF
  bash research/f3/mac_step.sh l3_finalfile_$V 4 1 logs/final_file_l3_$V.log $PY -X utf8 research/f3/final_file.py $SV research/f3/out/$FF/spirited-tiger_final.parquet \
    research/f3/in/l3s570_int.parquet research/f3/in/test_$S5.parquet $P/r16b_sbx/data/ranking.parquet || die "final file $NM failed"
  echo "$(ts) final file $NM written: research/f3/out/$FF/spirited-tiger_final.parquet ($(h research/f3/out/$FF/spirited-tiger_final.parquet))"
  touch research/f3/out/l3_$NM.DONE
}
variant Bb ff7 598b
variant Ba ff6 598a
echo "$(ts) phase5b done"
