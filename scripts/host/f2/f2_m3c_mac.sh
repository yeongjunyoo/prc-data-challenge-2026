#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F2 ruler M3 tail stages only (artmm, arthist, aiu_shift on the corrected worlds), PROTOCOL-compliant, 1 thread, nice.
set -u
cd ${PRC_WORK:-.}/f2_run || exit 1
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
ts() { date '+%m-%d %H:%M:%S'; }
lock="$P/f2_m3c.lock"; mkdir "$lock" || exit 9
printf 'owner: PRC F2 ruler M3 tail stages, coordinator main\npid: %s\npeak_gb: 4\nthreads: 1\nstart: %s\nexpected_end: %s KST\njob: research/f2/m3_tail.py\n' \
  "$$" "$(date '+%Y-%m-%d %H:%M')" "$(date -v+15M '+%Y-%m-%d %H:%M')" > "$lock/info"
echo "$(ts) START m3c"
OMP_NUM_THREADS=1 nice -n 10 /usr/bin/time -l "$PY" -u research/f2/m3_tail.py --tag f2m3 --vbase m6r27skwing --fbase f2skwing --ctx f2ctx_v3 > logs/m3_tail.log 2>&1
rc=$?; rm -rf "$lock"; echo "$(ts) END m3c rc $rc"
