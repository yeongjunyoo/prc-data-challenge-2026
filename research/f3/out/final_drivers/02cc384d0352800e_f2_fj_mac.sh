#!/bin/bash
# F2 lever 2 (vault research/2026-10-08-f2-four-months.md 1.9): research/f2/fj_tables.py (1 thread), then research/f2/gate19.py (2 threads), on f2_run.
#   nohup /bin/bash f2_fj_mac.sh >> logs/f2_fj.log 2>&1 < /dev/null &      (from ~/nova-adsb/prcm/f2_run)
# PROTOCOL before each stage: two checks 30 s apart of no ARC request, declared threads of every other lock + mine <= 12, memory_pressure
# free >= 30%. Lock f2_fj.lock only while a stage runs; nice 10; time limits 60 min (tables) and 120 min (gate). Markers F2_FJ.DONE / .FAILED.
set -u
cd ~/nova-adsb/prcm/f2_run || exit 1
P=~/nova-adsb/prcm
PY=~/nova-adsb/venv311/bin/python
L="$P/f2_fj.lock"
ts() { date '+%m-%d %H:%M:%S'; }
declared() { local s=0 t f; for f in "$P"/*.lock/info; do [ -e "$f" ] || continue; case "$f" in "$L"/*) continue;; esac
  t=$(grep -i -m1 '^threads' "$f" | grep -o '[0-9][0-9.]*' | head -1); s=$(echo "$s + ${t:-6}" | bc); done; echo "$s"; }
ok_now() { [ -e "$P/arc_quiet.request" ] || [ -e "$P/arc_quiet.lock" ] && return 1
  [ "$(echo "$(declared) + $1 <= 12" | bc)" = 1 ] || return 1
  local free; free=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %); [ "${free:-0}" -ge 30 ] || return 1; return 0; }
wait_ok() { local n=0 w=0; while [ $n -lt 2 ]; do if ok_now "$1"; then n=$((n+1)); else n=0; w=$((w+1)); [ $((w % 20)) = 1 ] && echo "$(ts) waiting: declared $(declared) + $1"; fi; [ $n -lt 2 ] && sleep 30; done; }
lock() { mkdir "$L" || { echo "$(ts) lock busy"; exit 9; }
  printf 'owner: PRC F2 lever 2 (matched Feb/Jun 2025 extras), coordinator main\npid: %s\npeak_gb: %s\nthreads: %s\nstart: %s\nexpected_end: %s KST\njob: %s\n' \
    "$$" "$2" "$1" "$(date '+%Y-%m-%d %H:%M')" "$(date -v+"$3"M '+%Y-%m-%d %H:%M')" "$4" > "$L/info"; }
unlock() { if [ -e "$L/info" ] && grep -q "^pid: $$\$" "$L/info"; then rm -rf "$L"; fi; }
trap unlock EXIT
if [ ! -e research/f2/out/fj_tables.json ]; then
  wait_ok 1; lock 1 4 40 research/f2/fj_tables.py
  echo "$(ts) START fj_tables (1 thread)"
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 perl -e 'alarm shift; exec @ARGV' 3600 $PY -X utf8 -u research/f2/fj_tables.py > logs/fj_tables.log 2>&1
  rc=$?; unlock; echo "$(ts) END fj_tables rc $rc"
  if [ $rc != 0 ]; then tail -4 logs/fj_tables.log | cut -c1-400; echo "$(ts) FAILED fj_tables rc $rc" > F2_FJ.FAILED; exit 2; fi
fi
ash=$($PY -c "import json;print(json.load(open('research/f2/out/fj_tables.json'))['outputs_sha16']['adsb_v1ndfj'])")
gsh=$($PY -c "import json;print(json.load(open('research/f2/out/fj_tables.json'))['outputs_sha16']['geo_v1ndfj'])")
echo "$(ts) tables adsb_v1ndfj $ash geo_v1ndfj $gsh"
[ -d research/f2/fitcache_g19 ] || { cp -c -R research/f2/fitcache_g14 research/f2/fitcache_g19 && echo "$(ts) fitcache_g19 cloned: $(ls research/f2/fitcache_g19 | wc -l) files"; }
wait_ok 2; lock 2 4 60 research/f2/gate19.py
echo "$(ts) START gate19 (2 threads)"
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 perl -e 'alarm shift; exec @ARGV' 7200 $PY -u research/f2/gate19.py --adsb adsb_v1ndfj --adsb_sha "$ash" \
  --geo geo_v1ndfj --geo_sha "$gsh" --ctx f2ctx_v3 --ctx_sha 531e8113e7f26fff --threads 2 --fit_cache research/f2/fitcache_g19 > logs/gate19.log 2>&1
rc=$?; unlock; echo "$(ts) END gate19 rc $rc"
grep -E 'GATE|check|Error|Traceback|fit cache' logs/gate19.log | tail -6 | cut -c1-500
if [ $rc = 0 ]; then echo "$(ts) DONE" > F2_FJ.DONE; else echo "$(ts) FAILED gate19 rc $rc" > F2_FJ.FAILED; fi
