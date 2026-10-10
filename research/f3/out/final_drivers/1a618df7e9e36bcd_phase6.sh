#!/bin/bash
# F3 final round c (2026-10-10 08:3x KST; coordinator 08:2x: F2's lever 2 adopted): 594c (594b's form on the 514-config legs) and 598c (598b's
# form on L3's legs), each with the four-month corrector on F2's tables adsb_v1ndfjF and geo_v1ndfjF; research/f3/kit_c.sh.
#   0. the two tables into cache/ (APFS clones from f2_run, sha16 checked) and research/f3/check_fjF.py (this root's tables, every row unchanged,
#      plus 326,873 Feb/Jun 2025 rows); any failure stops the phase;
#   1. per family, in parallel: corr4m A, B, C in parallel (the kit's guard: 10 fitted per 0.75 run, 0 per twin, else stop); merges A, B, C;
#      the corrector-stage check; manifests and tails (chains in parallel); avg3; check finals (only "final BbF: Jan/Jul rows differ from the
#      default final" may fail, by design since 10-10 03:0x); F2's iii_boundary.py against the family's default (592 or 596; must pass); the
#      final file (594c: ff8 with 594b's references; 598c: ff9 with 598b's).
# Caps: corrector runs start 45 s apart (594c first); F3CAP 15 with the logged 300 s total-cap override (the coordinator: the whole PRC allowance for this round), memory floor 30 % in
# mac_step.sh. STOP_F3 stops new work.
#   nohup bash research/f3/phase6.sh >> logs/phase6.log 2>&1 < /dev/null &
set -u
cd ~/nova-adsb/prcm/final_run || exit 1
P=~/nova-adsb/prcm
PY=~/nova-adsb/venv311/bin/python
export F3CAP=15 F3_MAXTH=3
unset TOTCAP
ts() { date '+%m-%d %H:%M:%S'; }
h() { shasum -a 256 "$1" | cut -c1-16; }
die() { echo "$(ts) $*"; touch research/f3/out/phase6.FAILED; exit 2; }
echo "$(ts) phase6 start (phase6.sh $(h research/f3/phase6.sh), kit_c.sh $(h research/f3/kit_c.sh), check_fjF.py $(h research/f3/check_fjF.py), mac_step.sh $(h research/f3/mac_step.sh), final_file.py $(h research/f3/final_file.py))"
[ -e STOP_F3 ] && die "STOP_F3 present"
echo 300 > research/f3/CAP_OVERRIDE_S
trap 'rm -f research/f3/CAP_OVERRIDE_S' EXIT
# 0. tables
for t in adsb_v1ndfjF:e13437a2c55a77c5 geo_v1ndfjF:4fd677dc9060ce22; do
  n=${t%%:*}; w=${t##*:}
  [ -e cache/$n.parquet ] || cp -cp $P/f2_run/cache/$n.parquet cache/$n.parquet || die "copy cache/$n.parquet"
  [ "$(h cache/$n.parquet)" = "$w" ] || die "cache/$n.parquet sha16 $(h cache/$n.parquet) != $w"
done
bash research/f3/mac_step.sh c_tables 8 1 logs/c_check_fjF.log $PY -X utf8 research/f3/check_fjF.py || die "check_fjF failed (logs/c_check_fjF.log)"
grep -q '"pass": true' research/f3/out/check_fjF.json || die "check_fjF did not pass"
echo "$(ts) tables adsb_v1ndfjF, geo_v1ndfjF in cache/: check_fjF passes"
family() {  # 594c | 598c
  trap - EXIT
  local F=$1 S0 SV FF REFI REFT X ra rb rc PA PB PC TA TB TC
  case $F in
    594c) S0=m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3; FF=ff8; REFI=research/f3/in/spirited-tiger_v36.parquet;;
    598c) S0=m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3; FF=ff9; REFI=research/f3/in/l3s570_int.parquet;;
  esac
  SV=${S0}rBbFamfhgkcxncrdbawgpglgjaavskap; REFT=research/f3/in/test_${S0}amfhgkcxncrdbawgpglgjaavskap.parquet
  [ -e "$REFT" ] && [ -e "$REFI" ] || { echo "$(ts) $F: reference files missing"; return 2; }
  for X in A B C; do  # staggered by 45 s so that each step runner sees the locks of the ones before it (two checks 30 s apart)
    bash research/f3/kit_c.sh $F corr4m $X >> logs/c_${F}_corr4m_$X.log 2>&1 &
    eval "P$X=$!"
    sleep 45
  done
  wait $PA; ra=$?; wait $PB; rb=$?; wait $PC; rc=$?
  [ $ra = 0 ] && [ $rb = 0 ] && [ $rc = 0 ] || { echo "$(ts) $F corr4m failed (A $ra, B $rb, C $rc; logs/c_${F}_corr4m_<X>.log)"; return 3; }
  echo "$(ts) $F corr4m done: $(grep -h 'fit cache' logs/${F}_c4m_*_s*.log | sed 's/.*fit cache [^:]*: //' | tr '\n' ' ')"
  for X in A B C; do bash research/f3/kit_c.sh $F merge $X >> logs/c_${F}_merge.log 2>&1 || { echo "$(ts) $F merge $X failed"; return 4; }; done
  bash research/f3/kit_c.sh $F check >> logs/c_${F}_check.log 2>&1 || { echo "$(ts) $F corrector-stage check failed (logs/${F}_check_corr.log)"; return 5; }
  echo "$(ts) $F merges and corrector-stage check done: $(grep -o 'SPLITCHECK pass [A-Za-z]*' logs/${F}_check_corr.log | tail -1)"
  for X in A B C; do
    ( bash research/f3/kit_c.sh $F manifest $X && bash research/f3/kit_c.sh $F tails $X ) >> logs/c_${F}_tails_$X.log 2>&1 &
    eval "T$X=$!"
  done
  wait $TA; ra=$?; wait $TB; rb=$?; wait $TC; rc=$?
  [ $ra = 0 ] && [ $rb = 0 ] && [ $rc = 0 ] || { echo "$(ts) $F manifest or tails failed (A $ra, B $rb, C $rc)"; return 6; }
  bash research/f3/kit_c.sh $F avg3 >> logs/c_${F}_avg3.log 2>&1 || { echo "$(ts) $F avg3 failed"; return 7; }
  [ -e preds/manifest_$SV.json ] || { echo "$(ts) $F: no manifest $SV"; return 7; }
  echo "$(ts) $F avg3 done: $SV"
  bash research/f3/kit_c.sh $F check finals >> logs/c_${F}_check_finals_run.log 2>&1
  echo "$(ts) $F check finals rc $? (only (iii)'s boundary line may fail)"
  $PY -c "import json,sys; f=json.load(open('research/f3/out/split_check_$F.json'))['fails']; print('$F fails:', f); sys.exit(0 if f == ['final BbF: Jan/Jul rows differ from the default final'] or f == [] else 1)" \
    || { echo "$(ts) $F check finals failed beyond (iii)'s boundary line"; return 8; }
  bash research/f3/mac_step.sh ${F}_iii 4 1 logs/${F}_iii_boundary.log $PY -X utf8 -u research/f2/iii_boundary.py --default preds/manifest_${S0}amfhgkcxncrdbawgpglgjaavskap.json \
    --finals BbF=preds/manifest_$SV.json --out research/f3/out/iii_boundary_$F.json || { echo "$(ts) $F iii_boundary did not pass (logs/${F}_iii_boundary.log)"; return 9; }
  echo "$(ts) $F iii_boundary pass"
  [ -e research/f3/out/$FF/spirited-tiger_final.parquet ] && { echo "$(ts) $F: research/f3/out/$FF/spirited-tiger_final.parquet exists"; return 10; }
  mkdir -p research/f3/out/$FF
  bash research/f3/mac_step.sh ${F}_finalfile 4 1 logs/final_file_$F.log $PY -X utf8 research/f3/final_file.py $SV research/f3/out/$FF/spirited-tiger_final.parquet \
    $REFI $REFT $P/r16b_sbx/data/ranking.parquet || { echo "$(ts) $F final file failed"; return 11; }
  echo "$(ts) final file $F written: research/f3/out/$FF/spirited-tiger_final.parquet ($(h research/f3/out/$FF/spirited-tiger_final.parquet))"
  touch research/f3/out/c_$F.DONE
}
family 594c & PF1=$!   # first: its corrector runs at 2 threads (the seeded cache's key) and is the longer one
sleep 140
family 598c & PF2=$!
wait $PF1; r1=$?; wait $PF2; r2=$?
[ $r1 = 0 ] && [ $r2 = 0 ] || die "a family did not finish (594c $r1, 598c $r2)"
echo "$(ts) phase6 done"
