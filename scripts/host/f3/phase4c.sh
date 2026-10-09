#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 second rebuild, phase 4c (2026-10-10 03:xx KST): the final files of option B's variants Ba (594a, ff3) and Bb (594b, ff4), the last part of
# phase4b, run on its own because phase4b stops at F2's check finals: check (iii) (Jan/Jul rows bit-equal to 592) fails on 556 rows that
# src/aiu_shift.py --group block pools across the month boundary (off-block-date groups LFPG 01-31 and two 07-01 rows near midnight). Run only
# after our decision on that. The step and its arguments are phase4b's (research/f3/final_file.py with v36 and 531's test as the
# Jan/Jul references); refuses to overwrite.
#   nohup bash research/f3/phase4c.sh >> logs/phase4c.log 2>&1 < /dev/null &
set -u
cd ${PRC_WORK:-.}/final_run || exit 1
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
S=m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3
S5=${S}amfhgkcxncrdbawgpglgjaavskap
export F3CAP=3 TOTCAP=15
ts() { date '+%m-%d %H:%M:%S'; }
h() { shasum -a 256 "$1" | cut -c1-16; }
echo "$(ts) phase4c start (phase4c.sh $(h research/f3/phase4c.sh), final_file.py $(h research/f3/final_file.py)); 10-10 03:0x: decision (a)"
while pgrep -f "research/f3/phase4b.sh" > /dev/null; do sleep 30; done   # phase4b ends (stopped or done)
grep -q "phase4b done" logs/phase4b.log && { echo "$(ts) phase4b finished by itself (final files written there): nothing to do"; exit 0; }
grep -q "final check failed" logs/phase4b.log || { echo "$(ts) phase4b stopped before the final check: $(tail -1 logs/phase4b.log)"; exit 2; }
echo "$(ts) phase4b stopped at the final check as expected (check iii); writing the final files"
for V in Ba Bb; do
  ff=ff3; [ $V = Bb ] && ff=ff4
  [ -e preds/manifest_${S}r${V}amfhgkcxncrdbawgpglgjaavskap.json ] || { echo "$(ts) no final manifest for $V: stop"; exit 3; }
  [ -e research/f3/out/$ff/spirited-tiger_final.parquet ] && { echo "$(ts) research/f3/out/$ff/spirited-tiger_final.parquet exists: stop"; exit 3; }
  mkdir -p research/f3/out/$ff
  bash research/f3/mac_step.sh finalfile_$V 4 1 logs/final_file_$V.log $PY -X utf8 research/f3/final_file.py ${S}r${V}amfhgkcxncrdbawgpglgjaavskap \
    research/f3/out/$ff/spirited-tiger_final.parquet research/f3/in/spirited-tiger_v36.parquet research/f3/in/test_$S5.parquet \
    $P/r16b_sbx/data/ranking.parquet || { echo "$(ts) final file $V failed"; exit 4; }
  echo "$(ts) final file $V written: research/f3/out/$ff/spirited-tiger_final.parquet ($(h research/f3/out/$ff/spirited-tiger_final.parquet))"
done
echo "$(ts) phase4c done"
