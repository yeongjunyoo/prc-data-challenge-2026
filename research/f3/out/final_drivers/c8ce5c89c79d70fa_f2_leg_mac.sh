#!/bin/bash
# F2 (vault research/2026-10-08-f2-four-months.md): leave-Feb/Jun-out LightGBM leg (setting fj) on the Mac, PROTOCOL-compliant.
#   nohup /bin/bash f2_leg_mac.sh <threads> >> logs/f2_leg.log 2>&1 < /dev/null &
# Steps: model3_fj (copy of src/model3.py with setting fj = train 2025 minus Feb, Jun; eval Feb + Jun), then g_band_fj.
# Lock prcm/f2_leg.lock held only while a step runs. Before each step: no arc_quiet.request/lock, declared threads + mine <= 12,
# memory_pressure free >= 30%, two checks 30 s apart. nice 10.
set -u
cd ~/nova-adsb/prcm/f2_run || exit 1
TH=${1:-4}
P=~/nova-adsb/prcm
PY=~/nova-adsb/venv311/bin/python
ts() { date '+%m-%d %H:%M:%S'; }
declared() { local s=0 t f; for f in "$P"/*.lock/info; do [ -e "$f" ] || continue; case "$f" in *f2_leg.lock*) continue;; esac
  t=$(grep -i -m1 '^threads' "$f" | grep -o '[0-9][0-9.]*' | head -1); s=$(echo "$s + ${t:-6}" | bc); done; echo "$s"; }
ok_now() { [ -e "$P/arc_quiet.request" ] || [ -e "$P/arc_quiet.lock" ] && return 1
  local d free; d=$(declared); [ "$(echo "$d + $TH <= 12" | bc)" = 1 ] || return 1
  free=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %); [ "${free:-0}" -ge 30 ] || return 1; return 0; }
wait_ok() { local n=0; while [ $n -lt 2 ]; do if ok_now; then n=$((n+1)); else n=0; fi; [ $n -lt 2 ] && sleep 30; done; }
step() {  # name expected_min cmd...
  local name=$1 mins=$2; shift 2
  wait_ok
  local lock="$P/f2_leg.lock"
  mkdir "$lock" || { echo "$(ts) lock busy"; return 9; }
  printf 'owner: PRC F2 (four-month generalization, leave-Feb/Jun-out leg), coordinator main\npid: %s\npeak_gb: 6.5\nthreads: %s\nstart: %s\nexpected_end: %s KST\njob: %s\n' \
    "$$" "$TH" "$(date '+%Y-%m-%d %H:%M')" "$(date -v+${mins}M '+%Y-%m-%d %H:%M')" "$*" > "$lock/info"
  echo "$(ts) START $name: $*"
  nice -n 10 /usr/bin/time -l "$PY" -u "$@" > "logs/$name.log" 2>&1
  local rc=$?
  rm -rf "$lock"
  echo "$(ts) END $name rc $rc"
  return $rc
}
step f2skwin_fj_s0 45 research/f2/model3_fj.py --setting fj --seed 0 --tag f2skwin --feats feats_v27 --es internal --refit --wpow 1 --leaves 255 --mdl 100 --no_direct --clf_drop_prefix m_,ab_ --full_mult 1.0 --threads "$TH" --mu_sep LIRF --anchor win || exit 1
step f2skwing_fj_s0 5 research/f2/g_band_fj.py --setting fj --seed 0 --tag f2skwing --base f2skwin --feats feats_v27 || exit 1
touch f2_leg.DONE
echo "$(ts) all done"
