#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F2: smoke run of research/f2/gate14.py and gate15.py on the LightGBM-only world of section 4.3 (one chain S: V = val_m6r27skwing_s0,
# F = fj_f2skwing_s0, f2ctx_v3), into research/f2/out_smoke with its own model store. Its I_V, U_V, T, T1 and per-month L - T1 must equal
# research/f2/out/m3_corr_f2m3.json and m3_corr_lomo_f2m3.json (the same recipe, written independently); gate15 checks its stage 1 against
# the smoke gate14. Information only, no decision reads these numbers.
# Starts after the daily-format variant and its staging (f2nm_stage.DONE, or the builder gone), with no other F2 Mac job running (F2 <= 2
# threads), after two checks 30 s apart of no ARC request, declared threads of the locks + 2 <= 12, memory_pressure free >= 30%.
#   nohup /bin/bash f2_smoke_mac.sh >> logs/f2_smoke.log 2>&1 < /dev/null &      (from ${PRC_WORK:-.}/f2_run)
set -u
cd ${PRC_WORK:-.}/f2_run || exit 1
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
ts() { date '+%m-%d %H:%M:%S'; }
sha() { shasum -a 256 "$1" | cut -c1-16; }
declared() { local s=0 t f; for f in "$P"/*.lock/info; do [ -e "$f" ] || continue
  t=$(grep -i -m1 '^threads' "$f" | grep -o '[0-9][0-9.]*' | head -1); s=$(echo "$s + ${t:-6}" | bc); done; echo "$s"; }
f2_busy() { ls -d "$P"/f2_*.lock "$P"/f2nm_*.lock > /dev/null 2>&1 && return 0
  pgrep -f "f2_fjlegs_mac.sh|sb_build_f2nm.py|f2_stage_mac.sh" > /dev/null && return 0; return 1; }
ok_now() { [ -e "$P/arc_quiet.request" ] || [ -e "$P/arc_quiet.lock" ] && return 1; f2_busy && return 1
  [ "$(echo "$(declared) + 2 <= 12" | bc)" = 1 ] || return 1
  local free; free=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %); [ "${free:-0}" -ge 30 ] || return 1; return 0; }
until [ -e "$P/f2nm_stage.DONE" ] || ! pgrep -f "sb_build_f2nm.py|f2_stage_mac.sh" > /dev/null; do sleep 60; done
[ -e GO_FJ ] && { echo "$(ts) GO_FJ exists, the heavy phase has the F2 share; smoke skipped"; exit 0; }
n=0; while [ $n -lt 2 ]; do if ok_now; then n=$((n+1)); else n=0; fi; [ $n -lt 2 ] && sleep 30; done
L="$P/f2_smoke.lock"
mkdir "$L" || { echo "$(ts) lock busy"; exit 9; }
trap 'rm -rf "$L"' EXIT
printf 'owner: PRC F2 smoke run of gate14.py and gate15.py (LightGBM world), coordinator main\npid: %s\npeak_gb: 5\nthreads: 2\nstart: %s\nexpected_end: %s KST\njob: research/f2/gate14.py, gate15.py --out research/f2/out_smoke\n' \
  "$$" "$(date '+%Y-%m-%d %H:%M')" "$(date -v+30M '+%Y-%m-%d %H:%M')" > "$L/info"
C=$(sha cache/f2ctx_v3.parquet)
echo "$(ts) START smoke gate14"
nice -n 10 /usr/bin/time -l "$PY" -u research/f2/gate14.py --ctx f2ctx_v3 --ctx_sha "$C" --threads 2 --chains S:m6r27skwing:f2skwing \
  --out research/f2/out_smoke --fit_cache research/f2/fitcache_smoke > logs/smoke_gate14.log 2>&1
echo "$(ts) smoke gate14 rc $?"
nice -n 10 /usr/bin/time -l "$PY" -u research/f2/gate15.py --ctx f2ctx_v3 --ctx_sha "$C" --threads 2 \
  --out research/f2/out_smoke --fit_cache research/f2/fitcache_smoke > logs/smoke_gate15.log 2>&1
echo "$(ts) smoke gate15 rc $?"
