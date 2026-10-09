#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F2 ruler M3 corrector run on the Mac (private notes research/2026-10-08-f2-four-months.md section 1.3), PROTOCOL-compliant, 4 threads, nice.
#   nohup /bin/bash f2_m3_mac.sh >> logs/f2_m3.log 2>&1 < /dev/null &
set -u
cd ${PRC_WORK:-.}/f2_run || exit 1
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
TH=4
ts() { date '+%m-%d %H:%M:%S'; }
declared() { local s=0 t f; for f in "$P"/*.lock/info; do [ -e "$f" ] || continue; case "$f" in *f2_m3b.lock*) continue;; esac
  t=$(grep -i -m1 '^threads' "$f" | grep -o '[0-9][0-9.]*' | head -1); s=$(echo "$s + ${t:-6}" | bc); done; echo "$s"; }
ok_now() { [ -e "$P/arc_quiet.request" ] || [ -e "$P/arc_quiet.lock" ] && return 1
  local d free; d=$(declared); [ "$(echo "$d + $TH <= 12" | bc)" = 1 ] || return 1
  free=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %); [ "${free:-0}" -ge 30 ] || return 1; return 0; }
wait_ok() { local n=0; while [ $n -lt 2 ]; do if ok_now; then n=$((n+1)); else n=0; fi; [ $n -lt 2 ] && sleep 30; done; }
sha() { shasum -a 256 "$1" | cut -c1-16; }
wait_ok
lock="$P/f2_m3b.lock"; mkdir "$lock" || exit 9
printf 'owner: PRC F2 ruler M3b (whole-month check and tail stages), coordinator main\npid: %s\npeak_gb: 5\nthreads: %s\nstart: %s\nexpected_end: %s KST\njob: research/f2/m3_corr.py\n' \
  "$$" "$TH" "$(date '+%Y-%m-%d %H:%M')" "$(date -v+40M '+%Y-%m-%d %H:%M')" > "$lock/info"
echo "$(ts) START m3b"
nice -n 10 /usr/bin/time -l "$PY" -u research/f2/m3_corr_lomo.py --vbase m6r27skwing --fbase f2skwing --seed 0 --ctx f2ctx_v3 --ctx_sha "$(sha cache/f2ctx_v3.parquet)" --tag f2m3 --threads $TH > logs/m3_corr_lomo.log 2>&1
echo "$(ts) lomo rc $?"
nice -n 10 /usr/bin/time -l "$PY" -u research/f2/m3_tail.py --tag f2m3 --vbase m6r27skwing --fbase f2skwing --ctx f2ctx_v3 > logs/m3_tail.log 2>&1
rc=$?; rm -rf "$lock"; echo "$(ts) END m3b rc $rc"
