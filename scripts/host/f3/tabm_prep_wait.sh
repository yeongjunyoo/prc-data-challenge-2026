#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3: the TabM leg's full-setting input (research/r13_l/prep_x.py through final_orch.py tabm --only tabm_prep, 1 thread, Mac) as soon as
# the real feats_v27 has passed check_v27 (research/f3/out/feats_v27.PASS); the leg itself runs on the PC GPU (research/f3/tabm_pc.sh).
set -u
cd ${PRC_WORK:-.}/final_run || exit 1
until [ -e research/f3/out/feats_v27.PASS ]; do sleep 30; done
echo "$(date '+%m-%d %H:%M:%S') feats_v27.PASS seen: tabm_prep"
${PY_MAC:-python3} -u research/f3/final_orch.py tabm --only tabm_prep --budget 1 --max_jobs 1 --cap 9
echo "$(date '+%m-%d %H:%M:%S') tabm_prep driver ended"
