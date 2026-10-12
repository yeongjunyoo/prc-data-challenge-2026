#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 L4 final-root build (10-10 10:07, 10:2x, 10:5x and 11:2x, split (a)). Start ONLY on our go, after R17-A's gate 7.8
# (a) to (c) passes on the public root (L4S against 570). L4's versions are new files; the drivers that built 594c and 598c are imported or called
# unchanged (final_chain.py 636d0063, daily_candidate.py 6277203e, mac_step.sh 28ffd44a, final_file.py a60d331c; each sha16 is checked here).
# Mac rules: one heavy job at a time (CatBoost one leg at a time at 10 threads, RealMLP alone with the 25 % floor and the 20 % watchdog:
# research/f3/l4_full_f3.sh). Light steps go through research/f3/mac_step.sh with F3CAP 15 and round c's logged 300 s total-cap override, allowed
# only because nothing else runs on the Mac then (11:2x): if any other lock, arc_quiet file or other lane's process is present when
# phase7 starts, F3CAP is 12 and there is no override (PHASE7_CAP_OVERRIDE_S=0 also turns the override off).
# Order (11:2x, PHASE7_ORDER=chains_first, the default):
#   0. research/f3/collect_l4.py: R17-A's val and fwd files under this root's names, the CatBoost iteration files, R17-A's avg_l4.py, R17-A's
#      corrector fit cache -> research/f3/fitcache_l4 and research/f2/fitcache_4m_l4;
#   1. research/f3/l4_full_f3.sh full: the 15 L4 learners in the full setting and their seed averages (R17-A's avg_l4.py, settings full);
#   2. 600: research/f3/final_chain_l4.py chain C, B, A --cfg l4s75 (in parallel; a corrector that fits instead of loading stops its chain), avg3,
#      the final file ff10, the test move against 596 (research/f3/test_move.py), 601's Mac part (research/f3/daily_candidate_l4.py: the Jan/Jul
#      twin for the board veto), l4_600.DONE;
#   3. research/f3/l4_full_f3.sh fj and research/f3/fj_blends_l4.py: F2's five leave-Feb/Jun-out learners on feats_v28 and L4's fj blends
#      (KIT_L4FJ=l4, the consistent c form we chose; KIT_L4FJ=l3, the cheap fallback, skips this step: only on our word);
#   4. 602c: research/f3/kit_l4c.sh l4c: corr4m A, B, C (in parallel, 4 threads each; 10 fitted per 0.75 run, 0 per twin), mergeD and merge
#      A, B, C, the corrector-stage check, manifests and tails (chains in parallel), avg3, check finals (only "final BbF: Jan/Jul rows differ
#      from the default final" may fail), F2's iii_boundary.py against 600 (must pass), the final file ff11, the test move against 598c,
#      l4_602c.DONE.
# PHASE7_ORDER=fj_first runs 3 before 2 (the 10:2x order).
# Hold and stop files (11:2x: if 601 loses the veto, phase7 stops before the fj learners), checked between steps 2 and 3, before
# each fj learner, before the fj blends and before step 4 (and before each full learner); no lock is held while waiting:
#   research/f3/PHASE7_HOLD   phase7 waits while it exists; delete it to go on
#   research/f3/PHASE7_STOP   phase7 ends cleanly at the next check (exit 0, no FAILED marker, research/f3/out/phase7.STOPPED says where)
# A rerun of phase7.sh skips the finished parts (collect_l4.json, each learner's output, l4_600.DONE, the fj pins, l4_602c.DONE).
# STOP_F3 stops new work as everywhere in this lane (FAILED).
#   nohup bash research/f3/phase7.sh >> logs/phase7.log 2>&1 < /dev/null &
set -u
cd ${PRC_WORK:-.}/final_run || exit 1
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
export F3_MAXTH=3 KIT_L4FJ=${KIT_L4FJ:-l4}
unset TOTCAP
ORDER=${PHASE7_ORDER:-chains_first}
OV=${PHASE7_CAP_OVERRIDE_S:-300}
S600=m6r28xwincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap
S602=m6r28xwincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3rBbFamfhgkcxncrdbawgpglgjaavskap
S596=m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap
S598C=m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3rBbFamfhgkcxncrdbawgpglgjaavskap
HOLD=research/f3/PHASE7_HOLD
STOP=research/f3/PHASE7_STOP
ts() { date '+%m-%d %H:%M:%S'; }
h() { shasum -a 256 "$1" | cut -c1-16; }
die() { echo "$(ts) $*"; touch research/f3/out/phase7.FAILED; exit 2; }
stopped() { echo "$(ts) phase7 stopped cleanly $1 ($STOP)"; echo "$(date '+%Y-%m-%d %H:%M:%S') $1" > research/f3/out/phase7.STOPPED; exit 0; }
hold_point() {  # $1: where
  local n=0
  while [ -e "$HOLD" ] && [ ! -e "$STOP" ]; do
    [ $((n % 20)) = 0 ] && echo "$(ts) paused $1: $HOLD present (no lock held); delete it to go on, or touch $STOP to end here"
    n=$((n + 1)); sleep 60
  done
  [ -e "$STOP" ] && stopped "$1"
  return 0
}
[ -e "$STOP" ] && { echo "$(ts) $STOP present at start: not starting"; exit 0; }
[ -e STOP_F3 ] && die "STOP_F3 present"
[ -e research/f3/out/phase7.FAILED ] && die "research/f3/out/phase7.FAILED exists"
busy=$( { ls -d "$P"/*.lock 2> /dev/null | grep -v "/f3_"; ls "$P"/arc_quiet.* 2> /dev/null; pgrep -fl "venv311/bin/python|research/r17_a/|research/f2/|f2_run" 2> /dev/null; } | head -6 | tr '\n' ';')
if [ -n "$busy" ]; then
  export F3CAP=12; OV=0
  echo "$(ts) other work on the Mac at start ($busy): F3CAP 12, no total-cap override (11:2x)"
else
  export F3CAP=15
fi
echo "$(ts) phase7 start (order $ORDER, KIT_L4FJ $KIT_L4FJ, F3CAP $F3CAP, cap override ${OV}s; phase7.sh $(h research/f3/phase7.sh), l4_full_f3.sh $(h research/f3/l4_full_f3.sh), collect_l4.py $(h research/f3/collect_l4.py), final_chain_l4.py $(h research/f3/final_chain_l4.py), daily_candidate_l4.py $(h research/f3/daily_candidate_l4.py), kit_l4c.sh $(h research/f3/kit_l4c.sh), fj_blends_l4.py $(h research/f3/fj_blends_l4.py), test_move.py $(h research/f3/test_move.py))"
for x in "research/f3/final_chain.py 636d00637ab543bd" "research/f3/daily_candidate.py 6277203e1ad12d2f" "research/f3/mac_step.sh 28ffd44ae1c48b52" \
         "research/f3/final_file.py a60d331cbd6f0570" "research/f3/kit_c.sh dcfabd38f56d23e4" "research/f3/avg_legs_final.py ffc1ca6dd80b1736" \
         "research/f3/in/spirited-tiger_v43.parquet da9e49faf08a03c6"; do
  set -- $x
  [ "$(h $1)" = "$2" ] || die "$1 is not $2"
done
for f in research/f3/queue_ready/f3f3_$S596.parquet preds/test_$S596.parquet preds/test_$S598C.parquet research/f3/in/l3s570_int.parquet; do
  [ -e "$f" ] || die "missing $f"
done
if [ "$OV" != 0 ]; then echo "$OV" > research/f3/CAP_OVERRIDE_S; trap 'rm -f research/f3/CAP_OVERRIDE_S' EXIT; fi
learners() {  # learners <full|fj>: research/f3/l4_full_f3.sh; its exit 3 is a clean stop at a hold point
  bash research/f3/l4_full_f3.sh $1 >> logs/l4_$1.log 2>&1
  local rc=$?
  [ $rc = 3 ] && stopped "in the $1 learners (logs/l4_$1.log)"
  [ $rc = 0 ] || die "$1 learners failed (logs/l4_$1.log, research/f3/out/l4.FAILED)"
}
# 0. clones
if [ ! -e research/f3/out/collect_l4.json ]; then
  bash research/f3/mac_step.sh l4_collect 4 1 logs/l4_collect.log $PY -X utf8 research/f3/collect_l4.py || die "collect_l4 failed (logs/l4_collect.log)"
fi
echo "$(ts) clones: $(grep -c '"sha16"' research/f3/out/collect_l4.json) files; fitcache_l4 $(ls research/f3/fitcache_l4 | wc -l | tr -d ' '), fitcache_4m_l4 $(ls research/f2/fitcache_4m_l4 | wc -l | tr -d ' ') models; avg_l4.py $(h research/r17_a/avg_l4.py)"
# 1. the 15 full learners
learners full
echo "$(ts) full learners and averages done: $(for t in m6r28cbwinx3 m6r28cbx3 m6r28cbap m6r28rmwinx4 m6r28rmx4; do printf '%s %s ' $t $(h cache/full_${t}_s0.parquet); done)"
fj_step() {
  [ "$KIT_L4FJ" = l4 ] || { echo "$(ts) KIT_L4FJ=$KIT_L4FJ: fj learners skipped, the c form takes 598c's L3 fj blends"; return 0; }
  hold_point "before the fj learners"
  learners fj
  if [ ! -e research/f3/out/l4_fj_sha16.txt ]; then
    hold_point "before the fj blends"
    bash research/f3/mac_step.sh l4_fjblends 4 1 logs/l4_fj_blends.log $PY -X utf8 -u research/f3/fj_blends_l4.py || die "fj blends failed (logs/l4_fj_blends.log)"
  fi
  echo "$(ts) fj learners and blends done: $(tr '\n' ' ' < research/f3/out/l4_fj_sha16.txt)"
}
chains_step() {
  if [ -e research/f3/out/l4_600.DONE ]; then echo "$(ts) 600 and 601's Mac part done earlier"; return 0; fi
  local X PC_ PB_ PA_ rc_c rc_b rc_a
  for X in C B A; do
    ( trap - EXIT; $PY -u research/f3/final_chain_l4.py chain $X --cfg l4s75 >> logs/final_chain_l4_$X.log 2>&1 ) &
    eval "P${X}_=$!"
    sleep 45
  done
  wait $PC_; rc_c=$?; wait $PB_; rc_b=$?; wait $PA_; rc_a=$?
  [ $rc_c = 0 ] && [ $rc_b = 0 ] && [ $rc_a = 0 ] || die "an L4S chain did not finish (C $rc_c, B $rc_b, A $rc_a; logs/final_chain_l4_<X>.log)"
  $PY -u research/f3/final_chain_l4.py avg3 --cfg l4s75 >> logs/final_chain_l4_avg3.log 2>&1 || die "avg3 l4s75 failed"
  [ -e preds/manifest_$S600.json ] || die "no manifest for $S600"
  echo "$(ts) 600 avg3 done: $S600"
  mkdir -p research/f3/out/ff10
  bash research/f3/mac_step.sh l4_finalfile_600 4 1 logs/final_file_600.log $PY -X utf8 research/f3/final_file.py $S600 research/f3/out/ff10/spirited-tiger_final.parquet \
    research/f3/in/l3s570_int.parquet research/f3/in/test_$S596.parquet $P/r16b_sbx/data/ranking.parquet || die "final file 600 failed"
  echo "$(ts) final file 600 written: research/f3/out/ff10/spirited-tiger_final.parquet ($(h research/f3/out/ff10/spirited-tiger_final.parquet))"
  bash research/f3/mac_step.sh l4_move600 3 1 logs/test_move_600.log $PY -X utf8 research/f3/test_move.py preds/test_$S600.parquet preds/test_$S596.parquet \
    data/ranking.parquet research/f3/out/test_move_600_vs_596.json || die "test move 600 failed"
  echo "$(ts) test move 600 vs 596: $(grep -A2 '"all"' logs/test_move_600.log | tr -d '\n ' | cut -c1-160)"
  bash research/f3/mac_step.sh l4_dailycand 6 1 logs/daily_candidate_l4.log $PY -X utf8 research/f3/daily_candidate_l4.py $S600 f3f4_$S600 $P/r16b_sbx/data/ranking.parquet \
    $P/r16b_sbx/data/submitting.parquet research/f3/queue_ready/f3f3_$S596.parquet research/f3/in/spirited-tiger_v43.parquet research/f3/queue_ready || die "daily candidate 601 failed"
  echo "$(ts) daily candidate written: research/f3/queue_ready/f3f4_$S600.parquet ($(h research/f3/queue_ready/f3f4_$S600.parquet))"
  touch research/f3/out/l4_600.DONE
}
if [ "$ORDER" = fj_first ]; then fj_step; chains_step; else chains_step; fj_step; fi
# 4. 602c
if [ -e research/f3/out/l4_602c.DONE ]; then echo "$(ts) 602c done earlier"; exit 0; fi
hold_point "before the c form (602c)"
K=research/f3/kit_l4c.sh
for X in A B C; do
  ( trap - EXIT; bash $K l4c corr4m $X >> logs/l4c_corr4m_$X.log 2>&1 ) &
  eval "Q$X=$!"
  sleep 45
done
wait $QA; ra=$?; wait $QB; rb=$?; wait $QC; rc=$?
[ $ra = 0 ] && [ $rb = 0 ] && [ $rc = 0 ] || die "l4c corr4m failed (A $ra, B $rb, C $rc; logs/l4c_corr4m_<X>.log)"
echo "$(ts) l4c corr4m done: $(grep -h 'fit cache' logs/l4c_c4m_*_s*.log | sed 's/.*fit cache [^:]*: //' | tr '\n' ' ')"
for X in A B C; do
  bash $K l4c mergeD $X >> logs/l4c_merge.log 2>&1 || die "l4c mergeD $X failed"
  bash $K l4c merge $X >> logs/l4c_merge.log 2>&1 || die "l4c merge $X failed"
done
bash $K l4c check >> logs/l4c_check.log 2>&1 || die "l4c corrector-stage check failed (logs/l4c_check_corr.log)"
echo "$(ts) l4c merges and corrector-stage check done: $(grep -o 'SPLITCHECK pass [A-Za-z]*' logs/l4c_check_corr.log | tail -1)"
for X in A B C; do
  ( trap - EXIT; bash $K l4c manifest $X && bash $K l4c tails $X ) >> logs/l4c_tails_$X.log 2>&1 &
  eval "T$X=$!"
done
wait $TA; ra=$?; wait $TB; rb=$?; wait $TC; rc=$?
[ $ra = 0 ] && [ $rb = 0 ] && [ $rc = 0 ] || die "l4c manifest or tails failed (A $ra, B $rb, C $rc)"
bash $K l4c avg3 >> logs/l4c_avg3.log 2>&1 || die "l4c avg3 failed"
[ -e preds/manifest_$S602.json ] || die "no manifest for $S602"
echo "$(ts) 602c avg3 done: $S602"
bash $K l4c check finals >> logs/l4c_check_finals_run.log 2>&1
echo "$(ts) l4c check finals rc $? (only (iii)'s boundary line may fail)"
$PY -c "import json,sys; f=json.load(open('research/f3/out/split_check_l4c.json'))['fails']; print('l4c fails:', f); sys.exit(0 if f in ([], ['final BbF: Jan/Jul rows differ from the default final']) else 1)" \
  || die "l4c check finals failed beyond (iii)'s boundary line"
bash research/f3/mac_step.sh l4c_iii 4 1 logs/l4c_iii_boundary.log $PY -X utf8 -u research/f2/iii_boundary.py --default preds/manifest_$S600.json \
  --finals BbF=preds/manifest_$S602.json --out research/f3/out/iii_boundary_l4c.json || die "l4c iii_boundary did not pass (logs/l4c_iii_boundary.log)"
echo "$(ts) l4c iii_boundary pass"
mkdir -p research/f3/out/ff11
bash research/f3/mac_step.sh l4c_finalfile 4 1 logs/final_file_602c.log $PY -X utf8 research/f3/final_file.py $S602 research/f3/out/ff11/spirited-tiger_final.parquet \
  research/f3/in/l3s570_int.parquet research/f3/in/test_$S596.parquet $P/r16b_sbx/data/ranking.parquet || die "final file 602c failed"
echo "$(ts) final file 602c written: research/f3/out/ff11/spirited-tiger_final.parquet ($(h research/f3/out/ff11/spirited-tiger_final.parquet))"
bash research/f3/mac_step.sh l4c_move 3 1 logs/test_move_602c.log $PY -X utf8 research/f3/test_move.py preds/test_$S602.parquet preds/test_$S598C.parquet \
  data/ranking.parquet research/f3/out/test_move_602c_vs_598c.json || die "test move 602c failed"
echo "$(ts) test move 602c vs 598c: $(grep -A2 '"all"' logs/test_move_602c.log | tr -d '\n ' | cut -c1-160)"
touch research/f3/out/l4_602c.DONE
echo "$(ts) phase7 done"
