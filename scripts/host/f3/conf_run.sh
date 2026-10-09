#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 confirmation run (10-08 22:4x, low priority: <= 3 threads, nice): re-draw sampler -> three feature builds (2025 Jan/Jul
# re-drawn, everything else today's) -> one table -> one m6r25 val fit predicting all copies -> ΔMSE. Work dir ${PRC_WORK:-.}/f3_sbx/conf.
# Every step goes through research/f3/mac_step.sh (no lock while waiting, lock while running).
#   nohup bash ${PRC_WORK:-.}/final_run/research/f3/conf_run.sh >> ${PRC_WORK:-.}/f3_sbx/conf_run.log 2>&1 < /dev/null &
set -u
F=${PRC_WORK:-.}/f3_sbx
R=${PRC_WORK:-.}/final_run
C=$F/conf
PY=${PY_MAC:-python3}
STEP="bash $R/research/f3/mac_step.sh"
mkdir -p $C || exit 1
cd $C || exit 1
echo "$(date '+%m-%d %H:%M:%S') conf start"
[ -d $C/redraw ] || $STEP conf_redraw 3 1 $C/redraw.log $PY -X utf8 $R/research/f3/redraw.py --old $F/data/ranking.parquet \
  --new $F/data/final_phase/final_ranking.parquet --data $F/data --out $C/redraw --draws 1,2,3 || exit 2
for k in 1 2 3; do
  X=$C/xr$k
  [ -e $X/cache/feats_v25.parquet ] && continue
  mkdir -p $X/data $X/cache $X/preds $X/research/f3 || exit 3
  [ -d $X/src ] || cp -cRp $R/src $X/src
  cp -p $R/research/f3/build_feats_final.py $X/research/f3/
  for f in $F/data/training_*.parquet; do [ -e $X/data/$(basename $f) ] || cp -cp $f $X/data/; done
  cp -p $C/redraw/draw$k/training_2025-01-01_2025-02-01.parquet $C/redraw/draw$k/training_2025-07-01_2025-08-01.parquet $X/data/
  [ -e $X/data/ranking.parquet ] || cp -cp $F/data/ranking.parquet $X/data/ranking.parquet
  [ -d $X/data/metar ] || cp -cRp $F/data/metar $X/data/metar
  (cd $X && $STEP conf_feats$k 6 1 $X/build.log $PY -X utf8 research/f3/build_feats_final.py) || exit 4
done
Z=$C/xconf
mkdir -p $Z/cache $Z/preds $Z/notes || exit 5
[ -d $Z/src ] || cp -cRp $R/src $Z/src
[ -e $Z/cache/all.parquet ] || cp -cp $F/cache/all.parquet $Z/cache/all.parquet
cp -p $R/cache/iters_m6r25.json $Z/cache/
[ -e $Z/cache/feats_v25.parquet ] || $STEP conf_concat 7 1 $C/concat.log $PY -X utf8 $R/research/f3/conf_concat.py $F/cache/feats_v25.parquet \
  "$C/xr{k}/cache/feats_v25.parquet" $Z/cache/feats_v25.parquet || exit 6
grep -v "^ " $C/concat.log | head -3
[ -e $Z/cache/val_m6r25_s0.parquet ] || (cd $Z && $STEP conf_m6r25val 9 3 $C/val.log $PY -X utf8 -u src/model3.py --setting val --seed 0 --tag m6r25 \
  --feats feats_v25 --es internal --refit --wpow 1 --leaves 255 --mdl 100 --no_direct --clf_drop_prefix m_ --full_mult 1.0 --threads 3) || exit 7
$PY -X utf8 $R/research/f3/conf_eval.py $Z/cache/val_m6r25_s0.parquet $C/redraw $C/conf_eval.json > $C/conf_eval.log 2>&1 || exit 8
echo "$(date '+%m-%d %H:%M:%S') conf done"
