#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 second rebuild, phase 4b (2026-10-10 00:1x KST, 10-09 23:5x and 10-10 00:0x, F2 00:1x): option B through F2's kit in the
# final root, after phase4a (592).
#   1. wait for phase4a done (592's chains through cavg are what the merges read); stop if phase4a failed; then the daily check 581's Mac
#      part (research/f3/daily_candidate.py, 592's Jan/Jul rows in the public template order, 10-10 00:2x);
#   2. clone F2's kit (${PRC_WORK:-.}/f2_run/send_final, pushed 10-10 00:13) into research/f2/send_final, its files checked against its
#      sha16.txt and sha16.txt against the one F3 reviewed (KIT_SHA16TXT); kit install, prep (prep seeds research/f2/fitcache_4m from
#      f2_sbx's stage-1 models when absent);
#   3. corr4m A, B, C at KIT_TH=2 (the kit's guard: 10 fitted per 0.75 run, 0 per twin). Fail-fast guard here as well: the new files in
#      fitcache_4m during one chain's corr4m may not exceed EXPECT_FITS (30 = 3 runs x the 10 S_k); past it the kit's process tree (this
#      script's children only) is stopped and the phase ends;
#   4. merges A, B, C for KIT_VARIANTS (Ba Bb D), the corrector-stage check, manifests and tails of Ba and Bb at KIT_TH 3 (592's tails ran at
#      3 threads), avg3 Ba and Bb, the final check, and the final files ff3 (Ba = 594a) and ff4 (Bb = 594b) with v36 and 531's test as references.
# Caps as phase4a: F3CAP 3, TOTCAP 15, the logged 300 s total-cap override, memory floor 30 % (research/f3/mac_step.sh, used by the kit).
#   KIT_SHA16TXT=<sha16 of the reviewed sha16.txt> nohup bash research/f3/phase4b.sh >> logs/phase4b.log 2>&1 < /dev/null &
set -u
cd ${PRC_WORK:-.}/final_run || exit 1
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
KD=research/f2/send_final
K=$KD/kit_final.sh
S=m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3
EXPECT_FITS=${EXPECT_FITS:-30}
export F3CAP=3 TOTCAP=15 KIT_VARIANTS="Ba Bb D"
ts() { date '+%m-%d %H:%M:%S'; }
h() { shasum -a 256 "$1" | cut -c1-16; }
tree() { local p=$1 c; for c in $(ps -ax -o pid= -o ppid= | awk -v p="$p" '$2 == p {print $1}'); do tree "$c"; done; echo "$p"; }
echo "$(ts) phase4b start (EXPECT_FITS $EXPECT_FITS, phase4b.sh $(h research/f3/phase4b.sh), reviewed kit sha16.txt ${KIT_SHA16TXT:-?})"
if [ ! -e $KD ]; then
  [ "$(h $P/f2_run/send_final/sha16.txt)" = "${KIT_SHA16TXT:-none}" ] || { echo "$(ts) f2_run/send_final/sha16.txt $(h $P/f2_run/send_final/sha16.txt) is not the reviewed one: stop"; exit 2; }
  mkdir -p research/f2 && cp -cRp $P/f2_run/send_final $KD || exit 2
fi
[ "$(h $KD/sha16.txt)" = "${KIT_SHA16TXT:-none}" ] || { echo "$(ts) $KD/sha16.txt differs from the reviewed one: stop"; exit 2; }
while read -r want f; do [ "$(h $KD/$f)" = "$want" ] || { echo "$(ts) $KD/$f differs from its sha16.txt: stop"; exit 2; }; done < $KD/sha16.txt
echo "$(ts) kit $KD checked: $(tr '\n' ' ' < $KD/sha16.txt)"
until grep -q "phase4a done" logs/phase4a.log; do
  if grep -qE "failed|: stop" logs/phase4a.log; then echo "$(ts) phase4a did not finish: stop"; exit 2; fi
  sleep 30
done
sleep 5; echo "$(ts) phase4a done seen"
echo 300 > research/f3/CAP_OVERRIDE_S
trap 'rm -f research/f3/CAP_OVERRIDE_S' EXIT
# 581 (10-10 00:2x): 592's Jan/Jul rows as the daily check, before corr4m
S5=${S}amfhgkcxncrdbawgpglgjaavskap
bash research/f3/mac_step.sh dailycand_s75 6 1 logs/daily_candidate_s75.log $PY -X utf8 research/f3/daily_candidate.py $S5 f3f2_$S5 \
  $P/r16b_sbx/data/ranking.parquet $P/r16b_sbx/data/submitting.parquet research/f3/in/test_$S5.parquet research/f3/in/spirited-tiger_v36.parquet \
  research/f3/queue_ready || { echo "$(ts) daily candidate 581 failed"; exit 2; }
echo "$(ts) daily candidate written: research/f3/queue_ready/f3f2_$S5.parquet ($(h research/f3/queue_ready/f3f2_$S5.parquet))"
bash $K install || { echo "$(ts) kit install failed"; exit 3; }
KIT_TH=2 bash $K prep || { echo "$(ts) kit prep failed"; exit 3; }
mkdir -p research/f2/fitcache_4m
nfit() { ls research/f2/fitcache_4m | grep -c '\.txt$'; }
echo "$(ts) research/f2/fitcache_4m holds $(nfit) models before corr4m"
for X in A B C; do
  n0=$(nfit)
  KIT_TH=2 bash $K corr4m $X >> logs/phase4b_corr4m_$X.log 2>&1 &
  kp=$!
  while kill -0 $kp 2> /dev/null; do
    new=$(( $(nfit) - n0 ))
    if [ "$new" -gt "$EXPECT_FITS" ]; then
      echo "$(ts) corr4m $X: $new new fit files > EXPECT_FITS $EXPECT_FITS: stopping the kit (pids $(tree $kp | tr '\n' ' '))"
      kill -TERM $(tree $kp) 2> /dev/null
      exit 5
    fi
    sleep 15
  done
  wait $kp || { echo "$(ts) corr4m $X failed (logs/phase4b_corr4m_$X.log)"; exit 4; }
  echo "$(ts) corr4m $X done: $(( $(nfit) - n0 )) new fit files; $(grep -h 'fit cache' logs/f2k_c4m_${X}_0.75_s*.log logs/f2k_c4m_${X}_0.5_s*.log | sed 's/.*fit cache [^:]*: //' | tr '\n' ' ')"
done
for X in A B C; do KIT_TH=3 bash $K merge $X || { echo "$(ts) merge $X failed"; exit 6; }; done
KIT_TH=3 bash $K check || { echo "$(ts) corrector-stage check failed"; exit 7; }
for V in Ba Bb; do
  for X in A B C; do
    KIT_TH=3 bash $K manifest $X $V || { echo "$(ts) manifest $X $V failed"; exit 8; }
    KIT_TH=3 bash $K tails $X $V || { echo "$(ts) tails $X $V failed"; exit 8; }
  done
  KIT_TH=3 bash $K avg3 $V || { echo "$(ts) avg3 $V failed"; exit 9; }
  [ -e preds/manifest_${S}r${V}amfhgkcxncrdbawgpglgjaavskap.json ] || { echo "$(ts) no final manifest for $V: stop"; exit 9; }
  echo "$(ts) avg3 $V done"
done
KIT_TH=3 bash $K check finals || { echo "$(ts) final check failed"; exit 10; }
for V in Ba Bb; do
  ff=ff3; [ $V = Bb ] && ff=ff4
  mkdir -p research/f3/out/$ff
  bash research/f3/mac_step.sh finalfile_$V 4 1 logs/final_file_$V.log $PY -X utf8 research/f3/final_file.py ${S}r${V}amfhgkcxncrdbawgpglgjaavskap \
    research/f3/out/$ff/spirited-tiger_final.parquet research/f3/in/spirited-tiger_v36.parquet \
    research/f3/in/test_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.parquet $P/r16b_sbx/data/ranking.parquet \
    || { echo "$(ts) final file $V failed"; exit 11; }
  echo "$(ts) final file $V written: research/f3/out/$ff/spirited-tiger_final.parquet ($(h research/f3/out/$ff/spirited-tiger_final.parquet))"
done
echo "$(ts) phase4b done"
