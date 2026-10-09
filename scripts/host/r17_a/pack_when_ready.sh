#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# R17-A: build the F3 package as soon as all 58 Feb/Jun 2026 days are extracted (every out/<day>/meta.json present). No lock while waiting;
# the pack itself runs through research/r17_a/step_r17a.sh (PROTOCOL checks, lock r17a_pack.lock, 1 thread, nice 10). Marker final_pack.DONE or .FAILED.
#   nohup /bin/bash research/r17_a/pack_when_ready.sh >> research/r17_a/logs/pack_when_ready.log 2>&1 < /dev/null &     (from the sandbox root)
cd ${PRC_WORK:-.}/r17a_sbx || exit 1
ts() { date '+%m-%d %H:%M:%S'; }
need() { local n=0 d; for d in ${ADSB_WORK:-.}/out/2026-02-?? ${ADSB_WORK:-.}/out/2026-06-??; do [ -s "$d/meta.json" ] && n=$((n+1)); done; echo $n; }
echo "$(ts) waiting for 58 Feb/Jun days (now $(need))"
until [ "$(need)" -ge 58 ]; do sleep 60; done
echo "$(ts) 58 days present, packing"
if bash research/r17_a/step_r17a.sh pack 1 2.5 15 research/r17_a/logs/final_pack.log ${PY_MAC:-python3} -X utf8 research/r17_a/final_pack_r17a.py; then
  touch research/r17_a/final_pack.DONE; echo "$(ts) pack done"
else
  touch research/r17_a/final_pack.FAILED; echo "$(ts) pack FAILED"
fi
