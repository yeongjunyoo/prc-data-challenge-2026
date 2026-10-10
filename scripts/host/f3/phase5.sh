#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 L3 final, phase 5 (2026-10-10 04:5x KST; 03:5x, plan approved 04:2x): 596 (L3S on the four months), 597's Mac part, 598a.
#   0. F2's fj leg and blend tools and iii_boundary.py into research/f2/ (APFS clones from f2_run, sha16 checked against the PC copies)
#   1. research/f3/feats_v28_final.py (cache/adsb_v3, cache/feats_v28 and its checks; any failure stops the phase)
#   2. research/f3/collect_l3.py (L3's val/fwd/es/iters, L3's corrector models -> research/f3/fitcache_l3, F2's CatBoost/RealMLP fj files);
#      research/f2/fitcache_4m_l3 seeded with the same corrector models (the four-month corrector's stage 1 at 4 threads loads from them)
#   3. research/f3/final_orch.py l3legs in the background (9 full legs and 3 fj legs at 6 threads, G bands, averages)
#   4. 596: chains C, B, A of final_chain.py --cfg l3s75, each as soon as its legs are done; avg3; the final file ff5; 597's daily candidate f3f3
#   5. 598a: after the fj legs, research/f3/fj_blends_l3.py; L3 corr4m C, B, A (research/f3/kit_l3.sh), each after that chain's 596 cavg;
#      merges (Ba, D); the corrector-stage check; manifests and tails of Ba (chains in parallel); avg3 Ba; check finals (fails only on the two
#      "Jan/Jul rows differ" lines of (iii), by design since 10-10 03:0x: anything else stops); F2's iii_boundary.py against 596 (must pass);
#      the final file ff6
# Threads: the legs take 12 (13 with a 1-thread step, PROTOCOL); chain and kit steps run beside them through the logged total-cap override after
# 300 s (F3CAP 15 = our 15 with the earlier override rule). Chain steps at 3 threads (F3_MAXTH, as 592's), corr4m at 4 (L3's
# corrector key). Memory floor 30 % in mac_step.sh. STOP_F3 stops new work.
#   nohup bash research/f3/phase5.sh >> logs/phase5.log 2>&1 < /dev/null &
set -u
cd ${PRC_WORK:-.}/final_run || exit 1
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
S5=m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap
SB=m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3rBaamfhgkcxncrdbawgpglgjaavskap
export F3CAP=15 F3_MAXTH=3
unset TOTCAP
ts() { date '+%m-%d %H:%M:%S'; }
h() { shasum -a 256 "$1" | cut -c1-16; }
die() { echo "$(ts) $*"; touch research/f3/out/phase5.FAILED; exit 2; }
cbt() { case $1 in A) echo m6r28wincbrmfw4sv434tmc;; B) echo m6r28skcbrmfw4sv434;; C) echo m6r28apcbrmfw4sv114;; esac; }
echo "$(ts) phase5 start (phase5.sh $(h research/f3/phase5.sh), final_chain.py $(h research/f3/final_chain.py), final_orch.py $(h research/f3/final_orch.py), kit_l3.sh $(h research/f3/kit_l3.sh), mac_step.sh $(h research/f3/mac_step.sh), fj_blends_l3.py $(h research/f3/fj_blends_l3.py))"
[ -e STOP_F3 ] && die "STOP_F3 present"
echo 300 > research/f3/CAP_OVERRIDE_S
trap 'rm -f research/f3/CAP_OVERRIDE_S' EXIT
# 0. F2's tools
for f in model3_fj.py:25f42a490ac58e2d model3ap_fj.py:22b2ec5095f73b21 g_band_fj.py:a07b202b286eca98 blend_fixed_fj.py:591cae9a9ac19ec8 iii_boundary.py:1c878d90f66ee377; do
  n=${f%%:*}; w=${f##*:}
  [ -e research/f2/$n ] || cp -cp $P/f2_run/research/f2/$n research/f2/$n || die "copy research/f2/$n"
  [ "$(h research/f2/$n)" = "$w" ] || die "research/f2/$n sha16 $(h research/f2/$n) != $w"
done
echo "$(ts) F2 tools in research/f2 checked"
# 1. feats_v28
if [ ! -e cache/feats_v28.parquet ]; then
  bash research/f3/mac_step.sh l3_feats 6 1 logs/l3_feats_v28_final.log $PY -X utf8 research/f3/feats_v28_final.py || die "feats_v28_final failed (logs/l3_feats_v28_final.log)"
fi
grep -q '"pass": true' research/f3/out/feats_v28_final.json || die "feats_v28 checks did not pass"
echo "$(ts) feats_v28 $(h cache/feats_v28.parquet): checks pass"
# 2. clones
if [ ! -e research/f3/out/collect_l3.json ]; then $PY -X utf8 research/f3/collect_l3.py || die "collect_l3 failed"; fi
if [ ! -d research/f2/fitcache_4m_l3 ]; then
  mkdir research/f2/fitcache_4m_l3 && for f in research/f3/fitcache_l3/*.txt; do cp -cp "$f" research/f2/fitcache_4m_l3/; done
fi
echo "$(ts) clones: $(grep -c '"sha16"' research/f3/out/collect_l3.json) files; fitcache_l3 $(ls research/f3/fitcache_l3 | wc -l | tr -d ' '), fitcache_4m_l3 $(ls research/f2/fitcache_4m_l3 | wc -l | tr -d ' ') models"
# 3. legs
nohup $PY -u research/f3/final_orch.py l3legs --budget 13 --max_jobs 3 --cap 15 >> logs/orch_l3legs.log 2>&1 < /dev/null &
echo "$(ts) l3legs orchestrator pid $!"
legs_ok() { $PY -c "import json,sys; s=json.load(open('research/f3/out/orch_l3legs.json')); sys.exit(0 if all(s.get(n,{}).get('rc')==0 for n in sys.argv[1:]) else 1)" "$@" 2>/dev/null; }
legs_bad() { $PY -c "import json,sys; s=json.load(open('research/f3/out/orch_l3legs.json')); sys.exit(0 if any(s.get(n,{}).get('rc') not in (None,0) for n in sys.argv[1:]) else 1)" "$@" 2>/dev/null; }
wait_legs() {  # what legs...
  local w=$1; shift
  until legs_ok "$@"; do
    legs_bad "$@" && { echo "$(ts) $w: a leg it needs failed"; return 3; }
    [ -e STOP_F3 ] && { echo "$(ts) $w: STOP_F3"; return 3; }
    sleep 30
  done
}
# 4. 596's chains, each after its legs
chain_after() {  # label legs...
  trap - EXIT
  local L=$1; shift
  wait_legs "chain $L" "$@" || return 3
  $PY -u research/f3/final_chain.py chain $L --cfg l3s75 >> logs/final_chain_l3_$L.log 2>&1 || { echo "$(ts) chain $L failed"; return 4; }
  echo "$(ts) chain $L done"
}
chain_after C l3skap_s0 gband_m6r28skapg & PC=$!
chain_after B avg_l3skgx4 & PB=$!
chain_after A avg_l3skwingx4 & PA=$!
# 5a. 598a's fj blends and four-month correctors, beside the chains
fj_part() {
  trap - EXIT
  wait_legs "fj blends" gbandfj_l3skwing gbandfj_l3skg gbandfj_l3skapg fj_l3sk || return 3
  if [ ! -e research/f3/out/l3_fj_sha16.txt ]; then
    bash research/f3/mac_step.sh l3_fjblends 4 1 logs/l3_fj_blends.log $PY -X utf8 -u research/f3/fj_blends_l3.py || { echo "$(ts) fj blends failed"; return 4; }
  fi
  echo "$(ts) fj blends: $(tr '\n' ' ' < research/f3/out/l3_fj_sha16.txt)"
  for X in C B A; do
    until [ -e "cache/full_$(cbt $X)adsbgeondctxv3x50ws75c3_s0.parquet" ]; do [ -e STOP_F3 ] && return 3; [ -e research/f3/out/phase5.FAILED ] && return 3; sleep 30; done
    KIT_TH=4 bash research/f3/kit_l3.sh corr4m $X >> logs/l3k_corr4m_$X.log 2>&1 || { echo "$(ts) corr4m $X failed (logs/l3k_corr4m_$X.log)"; return 5; }
    echo "$(ts) corr4m $X done: $(grep -h 'fit cache' logs/l3k_c4m_${X}_0.75_s*.log logs/l3k_c4m_${X}_0.5_s*.log | sed 's/.*fit cache [^:]*: //' | tr '\n' ' ')"
  done
}
fj_part & PF=$!
wait $PC; rc_c=$?; wait $PB; rc_b=$?; wait $PA; rc_a=$?
[ $rc_c = 0 ] && [ $rc_b = 0 ] && [ $rc_a = 0 ] || die "a 596 chain did not finish (C $rc_c, B $rc_b, A $rc_a)"
# 596: avg3, final file, 597's Mac part
$PY -u research/f3/final_chain.py avg3 --cfg l3s75 >> logs/final_chain_l3_avg3.log 2>&1 || die "avg3 l3s75 failed"
[ -e preds/manifest_$S5.json ] || die "no manifest for $S5"
echo "$(ts) avg3 done: $S5"
mkdir -p research/f3/out/ff5
bash research/f3/mac_step.sh l3_finalfile 4 1 logs/final_file_l3.log $PY -X utf8 research/f3/final_file.py $S5 research/f3/out/ff5/spirited-tiger_final.parquet \
  research/f3/in/l3s570_int.parquet research/f3/in/test_$S5.parquet $P/r16b_sbx/data/ranking.parquet || die "final file 596 failed"
echo "$(ts) final file 596 written: research/f3/out/ff5/spirited-tiger_final.parquet ($(h research/f3/out/ff5/spirited-tiger_final.parquet))"
bash research/f3/mac_step.sh l3_dailycand 6 1 logs/daily_candidate_l3.log $PY -X utf8 research/f3/daily_candidate.py $S5 f3f3_$S5 $P/r16b_sbx/data/ranking.parquet \
  $P/r16b_sbx/data/submitting.parquet research/f3/in/test_$S5.parquet research/f3/in/l3s570_int.parquet research/f3/queue_ready || die "daily candidate 597 failed"
echo "$(ts) daily candidate written: research/f3/queue_ready/f3f3_$S5.parquet ($(h research/f3/queue_ready/f3f3_$S5.parquet))"
touch research/f3/out/l3_596.DONE
# 598a
wait $PF; rc_f=$?
[ $rc_f = 0 ] || die "598a's fj blends or four-month correctors did not finish ($rc_f)"
for X in A B C; do KIT_VARIANTS="Ba D" bash research/f3/kit_l3.sh merge $X >> logs/l3k_merge.log 2>&1 || die "merge $X failed (logs/l3k_merge.log)"; done
bash research/f3/kit_l3.sh check >> logs/l3k_check.log 2>&1 || die "corrector-stage check failed (logs/l3k_check.log)"
echo "$(ts) merges and the corrector-stage check done: $(grep -o 'SPLITCHECK pass [A-Za-z]*' logs/l3k_check_corr.log | tail -1)"
for X in A B C; do
  ( trap - EXIT; bash research/f3/kit_l3.sh manifest $X Ba && bash research/f3/kit_l3.sh tails $X Ba ) >> logs/l3k_tails_$X.log 2>&1 &
  eval "PT_$X=$!"
done
wait $PT_A; ra=$?; wait $PT_B; rb=$?; wait $PT_C; rc=$?
[ $ra = 0 ] && [ $rb = 0 ] && [ $rc = 0 ] || die "Ba manifest or tails failed (A $ra, B $rb, C $rc)"
bash research/f3/kit_l3.sh avg3 Ba >> logs/l3k_avg3.log 2>&1 || die "avg3 Ba failed"
[ -e preds/manifest_$SB.json ] || die "no manifest for $SB"
echo "$(ts) avg3 Ba done: $SB"
bash research/f3/kit_l3.sh check finals >> logs/l3k_check_finals_run.log 2>&1
echo "$(ts) check finals rc $? (only (iii)'s boundary lines may fail)"
$PY -c "import json,sys; r=json.load(open('research/f3/out/split_check_l3.json')); f=r['fails']; print('fails:', f); sys.exit(0 if all(x == 'final Ba: Jan/Jul rows differ from the default final' for x in f) else 1)" \
  || die "check finals failed beyond (iii)'s boundary lines"
bash research/f3/mac_step.sh l3_iii 4 1 logs/l3_iii_boundary.log $PY -X utf8 -u research/f2/iii_boundary.py --default preds/manifest_$S5.json \
  --finals Ba=preds/manifest_$SB.json --out research/f3/out/iii_boundary_l3.json || die "iii_boundary did not pass (logs/l3_iii_boundary.log)"
echo "$(ts) iii_boundary pass"
mkdir -p research/f3/out/ff6
bash research/f3/mac_step.sh l3_finalfile_Ba 4 1 logs/final_file_l3_Ba.log $PY -X utf8 research/f3/final_file.py $SB research/f3/out/ff6/spirited-tiger_final.parquet \
  research/f3/in/l3s570_int.parquet research/f3/in/test_$S5.parquet $P/r16b_sbx/data/ranking.parquet || die "final file 598a failed"
echo "$(ts) final file 598a written: research/f3/out/ff6/spirited-tiger_final.parquet ($(h research/f3/out/ff6/spirited-tiger_final.parquet))"
echo "$(ts) phase5 done"
