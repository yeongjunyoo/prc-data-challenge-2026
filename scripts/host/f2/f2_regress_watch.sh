#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F2 role 4: when F3's final_run gets feats_v27, adsb_v1nd and geo_v1nd, rerun the read-only regression check on each (1 thread, nice).
#   nohup /bin/bash f2_regress_watch.sh >> logs/f2_regress_watch.log 2>&1 < /dev/null &
# A table is checked once its file has not changed for 3 minutes. Results: research/f2/out/regress_final_run_<table>.json, logs/regress_<table>.log
# A check starts when the declared threads of the locks + 1 <= 13 (PROTOCOL 10-08 07:4x single-thread rule), with no ARC request and
# memory_pressure free >= 30%, two checks 30 s apart. It does not wait for F2's own 2-thread jobs: the check guards F3's final legs, and
# research/f2/sb_build_f2nm.py does not start a step while f2_reg.lock exists.
set -u
cd ${PRC_WORK:-.}/f2_run || exit 1
P=${PRC_WORK:-.}; R=$P/final_run/cache
PY=${PY_MAC:-python3}
ts() { date '+%m-%d %H:%M:%S'; }
declared() { local s=0 t f; for f in "$P"/*.lock/info; do [ -e "$f" ] || continue
  t=$(grep -i -m1 '^threads' "$f" | grep -o '[0-9][0-9.]*' | head -1); s=$(echo "$s + ${t:-6}" | bc); done; echo "$s"; }
room() { [ -e "$P/arc_quiet.request" ] || [ -e "$P/arc_quiet.lock" ] && return 1
  [ "$(echo "$(declared) + 1 <= 13" | bc)" = 1 ] || return 1
  local free; free=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %); [ "${free:-0}" -ge 30 ] || return 1; return 0; }
left="feats_v27 adsb_v1nd geo_v1nd"
while [ -n "$left" ]; do
  next=""
  for t in $left; do
    f="$R/$t.parquet"
    if [ -e "$f" ] && [ $(( $(date +%s) - $(stat -f %m "$f") )) -ge 180 ] && room && sleep 30 && room; then
      mkdir "$P/f2_reg.lock" && printf 'owner: PRC F2 regression check of final_run %s (read-only)\npid: %s\npeak_gb: 2\nthreads: 1\nstart: %s\nexpected_end: %s KST\njob: research/f2/regress_check.py\n' \
        "$t" "$$" "$(date '+%Y-%m-%d %H:%M')" "$(date -v+10M '+%Y-%m-%d %H:%M')" > "$P/f2_reg.lock/info"
      echo "$(ts) checking $t"
      OMP_NUM_THREADS=1 nice -n 10 "$PY" -u research/f2/regress_check.py --root "$P/final_run" --tables "$t" --out "$t" > "logs/regress_$t.log" 2>&1
      echo "$(ts) $t rc $?: $(head -1 logs/regress_$t.log | cut -c1-400)"
      rm -rf "$P/f2_reg.lock"
    else
      next="$next $t"
    fi
  done
  left=$(echo $next)
  [ -n "$left" ] && sleep 120
done
echo "$(ts) all three checked"
