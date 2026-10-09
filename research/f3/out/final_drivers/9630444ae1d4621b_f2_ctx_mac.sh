#!/bin/bash
# F2 ruler inputs on the Mac: context tables with 2025 months 1, 2, 6, 7 under new names (f2ctx_v1..v3), PROTOCOL-compliant, 1 thread, nice.
#   nohup /bin/bash f2_ctx_mac.sh >> logs/f2_ctx.log 2>&1 < /dev/null &
set -u
cd ~/nova-adsb/prcm/f2_run || exit 1
P=~/nova-adsb/prcm
PY=~/nova-adsb/venv311/bin/python
ts() { date '+%m-%d %H:%M:%S'; }
declared() { local s=0 t f; for f in "$P"/*.lock/info; do [ -e "$f" ] || continue; case "$f" in *f2_ctx.lock*) continue;; esac
  t=$(grep -i -m1 '^threads' "$f" | grep -o '[0-9][0-9.]*' | head -1); s=$(echo "$s + ${t:-6}" | bc); done; echo "$s"; }
ok_now() { [ -e "$P/arc_quiet.request" ] || [ -e "$P/arc_quiet.lock" ] && return 1
  local d free; d=$(declared); [ "$(echo "$d + 1 <= 13" | bc)" = 1 ] || return 1
  free=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %); [ "${free:-0}" -ge 30 ] || return 1; return 0; }
wait_ok() { local n=0; while [ $n -lt 2 ]; do if ok_now; then n=$((n+1)); else n=0; fi; [ $n -lt 2 ] && sleep 30; done; }
sha() { shasum -a 256 "$1" | cut -c1-16; }
step() { local name=$1; shift; wait_ok; local lock="$P/f2_ctx.lock"; mkdir "$lock" || return 9
  printf 'owner: PRC F2 ruler inputs (context tables 2025-01,02,06,07), coordinator main\npid: %s\npeak_gb: 4\nthreads: 1\nstart: %s\nexpected_end: %s KST\njob: %s\n' \
    "$$" "$(date '+%Y-%m-%d %H:%M')" "$(date -v+20M '+%Y-%m-%d %H:%M')" "$*" > "$lock/info"
  echo "$(ts) START $name"; OMP_NUM_THREADS=1 nice -n 10 /usr/bin/time -l "$PY" -u "$@" > "logs/$name.log" 2>&1; local rc=$?; rm -rf "$lock"; echo "$(ts) END $name rc $rc"; return $rc; }
step f2ctx_v1 src/feats_ctx.py --name f2ctx_v1 --months 1,2,6,7 || exit 1
step f2ctx_v2 src/feats_ctx2.py --name f2ctx_v2 --base f2ctx_v1 --base_sha "$(sha cache/f2ctx_v1.parquet)" || exit 1
step f2ctx_v3 src/feats_aiu.py --name f2ctx_v3 --base f2ctx_v2 --base_sha "$(sha cache/f2ctx_v2.parquet)" || exit 1
touch f2_ctx.DONE; echo "$(ts) ctx done"
