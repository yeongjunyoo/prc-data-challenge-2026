#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 interim legs (fitwrap save, 2026-10-08 23:5x KST): LightGBM phase, then CatBoost phase, of research/f3/final_orch.py in
# the interim root ${PRC_WORK:-.}/f3_sbx/interim, every fit stored in final_run/research/f3/fc_legs. Lock names start with i_. Stops
# launching when the interim root's STOP_F3 exists (research/f3/phase2b.sh touches it when the real feats_v27 passes its check). The
# CatBoost phase starts only if R17-A's package is not there yet when the LightGBM phase ends.
#   nohup bash research/f3/interim_run.sh >> ${PRC_WORK:-.}/f3_sbx/interim/logs/interim_run.log 2>&1 < /dev/null &
set -u
P=${PRC_WORK:-.}
IR=$P/f3_sbx/interim
FC=$P/final_run/research/f3/fc_legs
PY=${PY_MAC:-python3}
ORCH="$PY -u $P/final_run/research/f3/final_orch.py"
ts() { date '+%m-%d %H:%M:%S'; }
cd $IR || exit 1
echo "$(ts) interim LightGBM phase"
$ORCH lgb --root $IR --wrap save:$FC --prefix i_ --budget 9 --cap 9 --max_jobs 3 >> logs/orch_lgb.log 2>&1
echo "$(ts) interim LightGBM phase ended: $(tail -1 logs/orch_lgb.log)"
[ -e STOP_F3 ] && { echo "$(ts) STOP_F3: no interim CatBoost"; exit 0; }
# the real feats_v27 comes about 15 min after R17-A's package; an interim CatBoost (about 55 min at 8 threads) would then hold the threads
# the final LightGBM loads need, for no gain, so the interim CatBoost phase runs only while the package is not there
[ -e $P/r17a_sbx/final/manifest.json ] && { echo "$(ts) R17-A package present: no interim CatBoost"; exit 0; }
echo "$(ts) interim CatBoost phase"
$ORCH cb --root $IR --wrap save:$FC --prefix i_ --budget 9 --cap 9 --max_jobs 2 >> logs/orch_cb.log 2>&1
echo "$(ts) interim CatBoost phase ended: $(tail -1 logs/orch_cb.log)"
