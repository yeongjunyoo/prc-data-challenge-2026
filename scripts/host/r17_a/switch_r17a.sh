#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# R17-A (22:2x): move the Mac runner from 4 streams to 2 so the PC can add 2 streams with the total curl cap <= 20 MB/s (Mac 2 x 4800K + PC 2 x 4800K).
# Touches STOP (each Mac worker ends after its current day), waits for the old runner to exit, removes STOP, starts dl_r17a.sh with 2 workers.
#   nohup /bin/bash switch_r17a.sh <old runner pid> >> logs/runner.log 2>&1 < /dev/null &
set -u
cd ${ADSB_WORK:-.} || exit 1
OLD=${1:?old runner pid}
ts() { date '+%m-%d %H:%M:%S'; }
touch STOP
echo "$(ts) switch: STOP touched, waiting for runner $OLD"
while kill -0 "$OLD" 2>/dev/null; do sleep 15; done
rm -f STOP
echo "$(ts) switch: old runner gone, starting 2 workers"
exec /bin/bash dl_r17a.sh 2 4800K days_r17a_all.txt
