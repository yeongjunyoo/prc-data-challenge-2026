#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F2 AIU-2026 ablation (local measurement, no candidate): corrector stage 2 without its AIU columns, chain A' bases and the F world. 2 threads.
set -u
cd ${PRC_WORK:-.}/f2_run || exit 1
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
TH=2
ts() { date '+%m-%d %H:%M:%S'; }
declared() { local s=0 t f; for f in "$P"/*.lock/info; do [ -e "$f" ] || continue; case "$f" in *f2_aiu.lock*) continue;; esac
  t=$(grep -i -m1 '^threads' "$f" | grep -o '[0-9][0-9.]*' | head -1); s=$(echo "$s + ${t:-6}" | bc); done; echo "$s"; }
ok_now() { [ -e "$P/arc_quiet.request" ] || [ -e "$P/arc_quiet.lock" ] && return 1
  local d free; d=$(declared); [ "$(echo "$d + $TH <= 12" | bc)" = 1 ] || return 1
  free=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %); [ "${free:-0}" -ge 30 ] || return 1; return 0; }
wait_ok() { local n=0; while [ $n -lt 2 ]; do if ok_now; then n=$((n+1)); else n=0; fi; [ $n -lt 2 ] && sleep 30; done; }
sha() { shasum -a 256 "$1" | cut -c1-16; }
run() { local name=$1; shift; wait_ok; local lock="$P/f2_aiu.lock"; mkdir "$lock" || return 9
  printf 'owner: PRC F2 AIU-2026 ablation (local, no candidate)\npid: %s\npeak_gb: 4\nthreads: %s\nstart: %s\nexpected_end: %s KST\njob: %s\n' \
    "$$" "$TH" "$(date '+%Y-%m-%d %H:%M')" "$(date -v+30M '+%Y-%m-%d %H:%M')" "$name" > "$lock/info"
  echo "$(ts) START $name"; nice -n 10 /usr/bin/time -l "$PY" -u "$@" > "logs/$name.log" 2>&1; local rc=$?; rm -rf "$lock"; echo "$(ts) END $name rc $rc"; return $rc; }
run aiu_A research/f2/aiu_ablate.py --vbase m6r27wincbrmfw4sv434tmc --fwdbase m6r27wincbrmfw4sv434tmc --ctx ctx_v3 --ctx_sha "$(sha cache/ctx_v3.parquet)" --tag A514 --threads $TH
run aiu_F research/f2/aiu_ablate.py --vbase m6r27skwing --fbase f2skwing --ctx f2ctx_v3 --ctx_sha "$(sha cache/f2ctx_v3.parquet)" --tag lgbF --threads $TH
touch f2_aiu.DONE
