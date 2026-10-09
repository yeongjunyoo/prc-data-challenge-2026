#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 fitwrap test (coordinator 2026-10-08 23:2x KST), m6r25 full on the final root's feats_v25, scratch dir ${PRC_WORK:-.}/f3_sbx/wt:
#   T1 direct (root r1) and T2 save (root r2) in parallel, 4 threads each; T3 load (r2, clone of T2's models); T4 strict on r4 (one
#   training value changed: must stop); T5 strict on r5 (final table with the hybrid's old-draw Jan/Jul rows: every fit must load; the
#   real re-draw move); CatBoost synthetic C1 direct, C2 save, C3 load, C4 strict perturbed (thread invariance:
#   R16-D measured it on the deployed legs, research/2026-10-08-r16-d-nd-legs.md).
#   Report: research/f3/wt_cmp.py -> f3_sbx/wt/out/wrapper_test.json.
#   nohup bash research/f3/wt_run.sh >> ${PRC_WORK:-.}/f3_sbx/wt/run.log 2>&1 < /dev/null &
set -u
P=${PRC_WORK:-.}
FR=$P/final_run
W=$P/f3_sbx/wt
PY=${PY_MAC:-python3}
FW=$FR/research/f3/fitwrap.py
STEP="bash $FR/research/f3/mac_step.sh"
A="--setting full --seed 0 --tag m6r25 --feats feats_v25 --es internal --refit --wpow 1 --leaves 255 --mdl 100 --no_direct --clf_drop_prefix m_ --full_mult 1.0 --threads 4"
ts() { date '+%m-%d %H:%M:%S'; }
mkdir -p $W/out $W/cb
for r in r1 r2 r4 r5; do
  mkdir -p $W/$r/cache $W/$r/preds $W/$r/logs
  [ -d $W/$r/src ] || cp -c -R $FR/src $W/$r/src
  [ -e $W/$r/cache/iters_m6r25.json ] || cp -c $FR/cache/iters_m6r25.json $W/$r/cache/
done
for r in r1 r2; do [ -e $W/$r/cache/feats_v25.parquet ] || cp -c $FR/cache/feats_v25.parquet $W/$r/cache/; done
echo "$(ts) wt start; fitwrap sha16 $(shasum -a 256 $FW | cut -c1-16)"
(cd $W/r1 && $STEP wt_t1 7 4 $W/r1/logs/T1.log $PY -X utf8 -u src/model3.py $A) > $W/out/step_T1.log 2>&1 &
P1=$!
(cd $W && $STEP wt_prep 6 1 $W/out/prep.log $PY -X utf8 $FR/research/f3/wt_prep.py $W $FR/cache/feats_v25.parquet \
   $P/f3_sbx/conf/xhyb/cache/feats_v25.parquet $P/r16b_sbx/cache/feats_v25.parquet) > $W/out/step_prep.log 2>&1 &
PP=$!
cd $W/r2 && $STEP wt_t2 7 4 $W/r2/logs/T2.log $PY -X utf8 -u $FW save $W/fc_save src/model3.py $A
echo "$(ts) T2 rc $?"
[ -e $W/r2/cache/full_m6r25_s0.parquet ] && mv $W/r2/cache/full_m6r25_s0.parquet $W/out/T2.parquet
mkdir -p $W/fc_load $W/fc_strict $W/fc_t5
for d in fc_load fc_strict fc_t5; do cp -c $W/fc_save/lgb_* $W/$d/; done
cd $W/r2 && $STEP wt_t3 7 4 $W/r2/logs/T3.log $PY -X utf8 -u $FW load $W/fc_load src/model3.py $A
echo "$(ts) T3 rc $?"
[ -e $W/r2/cache/full_m6r25_s0.parquet ] && mv $W/r2/cache/full_m6r25_s0.parquet $W/out/T3.parquet
wait $PP
echo "$(ts) prep rc $?"
cd $W/r4 && $STEP wt_t4 7 4 $W/r4/logs/T4.log $PY -X utf8 -u $FW strict $W/fc_strict src/model3.py $A
echo "$(ts) T4 rc $? (must be nonzero)"
[ -e $W/r4/cache/full_m6r25_s0.parquet ] && mv $W/r4/cache/full_m6r25_s0.parquet $W/out/T4.parquet
cd $W/r5 && $STEP wt_t5 7 4 $W/r5/logs/T5.log $PY -X utf8 -u $FW strict $W/fc_t5 src/model3.py $A
echo "$(ts) T5 rc $?"
[ -e $W/r5/cache/full_m6r25_s0.parquet ] && mv $W/r5/cache/full_m6r25_s0.parquet $W/out/T5.parquet
cd $W/cb && $STEP wt_cb 3 4 $W/cb/cb.log bash -c "
  $PY -X utf8 $FR/research/f3/cb_test.py c1.npy 4 &&
  $PY -X utf8 $FW save $W/cb/fc_cb $FR/research/f3/cb_test.py c2.npy 4 &&
  mkdir -p fc_cb_load && cp -c fc_cb/cb_* fc_cb_load/ &&
  $PY -X utf8 $FW load $W/cb/fc_cb_load $FR/research/f3/cb_test.py c3.npy 4 &&
  mkdir -p fc_cb_strict && cp -c fc_cb/cb_* fc_cb_strict/ &&
  { $PY -X utf8 $FW strict $W/cb/fc_cb_strict $FR/research/f3/cb_test.py c4.npy 4 perturb; echo C4 rc \$?; }"
echo "$(ts) cb rc $?"
wait $P1
echo "$(ts) T1 done: $(tail -2 $W/out/step_T1.log | tr '\n' ' ')"
[ -e $W/r1/cache/full_m6r25_s0.parquet ] && mv $W/r1/cache/full_m6r25_s0.parquet $W/out/T1.parquet
cd $W && $PY -X utf8 $FR/research/f3/wt_cmp.py $W > $W/out/wt_cmp.log 2>&1
echo "$(ts) wt done rc $?"
