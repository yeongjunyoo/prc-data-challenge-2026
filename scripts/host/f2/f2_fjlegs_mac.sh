#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F2: leave-Feb/Jun-out (setting fj) seed-0 legs of every 514 family except TabM, for the four-month corrector gate (private note 1.4).
#   nohup /bin/bash f2_fjlegs_mac.sh <lane> <cat_threads> >> logs/f2_fjlegs_<lane>.log 2>&1 < /dev/null &
#   lane 1: LightGBM sk -> g_band skg -> CatBoost cb        lane 2: LightGBM skap -> g_band skapg -> CatBoost cbwin -> CatBoost cbap
#   lane 3: RealMLP rmwin -> RealMLP rm (MPS, memory >= 50% free); lanes 4, 5, 6: CatBoost cb, cbwin, cbap one each (f2_heavy_mac.sh runs 4-6 then 3)
# PROTOCOL: each step waits for no ARC request, declared threads of the other locks + its own <= 12 (1-thread steps <= 13) and
# memory_pressure free >= 30% (RealMLP >= 50%), two checks 30 s apart; the lock is held only while a step runs; nice 10.
# Steps whose output exists are skipped. After the final root's first claim these steps are capped at 2 threads: steps launched after the file
# ${PRC_WORK:-.}/final_run/FIRST_CLAIM (or the env var CAP2=1) use 2 threads. LIGHT=1: only the LightGBM legs and g_band (lanes 1, 2).
set -u
cd ${PRC_WORK:-.}/f2_run || exit 1
LANE=${1:?lane}; CT=${2:-3}
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
ts() { date '+%m-%d %H:%M:%S'; }
declared() { local s=0 t f; for f in "$P"/*.lock/info; do [ -e "$f" ] || continue; case "$f" in *f2_fj$LANE.lock*) continue;; esac
  t=$(grep -i -m1 '^threads' "$f" | grep -o '[0-9][0-9.]*' | head -1); s=$(echo "$s + ${t:-6}" | bc); done; echo "$s"; }
ok_now() { local th=$1 mem=$2 cap=12; [ "$th" = 1 ] && cap=13
  [ -e "$P/arc_quiet.request" ] || [ -e "$P/arc_quiet.lock" ] && return 1
  local d free; d=$(declared); [ "$(echo "$d + $th <= $cap" | bc)" = 1 ] || return 1
  free=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %); [ "${free:-0}" -ge "$mem" ] || return 1; return 0; }
wait_ok() { local n=0; while [ $n -lt 2 ]; do if ok_now "$1" "$2"; then n=$((n+1)); else n=0; fi; [ $n -lt 2 ] && sleep 30; done; }
capped() { local th=$1; if [ -e "$P/final_run/FIRST_CLAIM" ] || [ "${CAP2:-0}" = 1 ]; then [ "$th" -gt 2 ] && th=2; fi; echo "$th"; }
step() {  # name out_file threads mem_pct expected_min cmd...   ({TH} in cmd is replaced by the thread count)
  local name=$1 out=$2 th; th=$(capped "$3"); local mem=$4 mins=$5; shift 5
  [ -e "$out" ] && { echo "$(ts) skip $name (exists)"; return 0; }
  wait_ok "$th" "$mem"; th=$(capped "$th")
  local lock="$P/f2_fj$LANE.lock"; mkdir "$lock" || { echo "$(ts) lock busy"; return 9; }
  printf 'owner: PRC F2 leave-Feb/Jun-out legs (lane %s)\npid: %s\npeak_gb: %s\nthreads: %s\nstart: %s\nexpected_end: %s KST\njob: %s\n' \
    "$LANE" "$$" "$([ "$mem" -ge 50 ] && echo 33 || echo 10)" "$th" "$(date '+%Y-%m-%d %H:%M')" "$(date -v+${mins}M '+%Y-%m-%d %H:%M')" "$name" > "$lock/info"
  local cmd=(); for x in "$@"; do cmd+=("${x//\{TH\}/$th}"); done
  echo "$(ts) START $name ($th threads): ${cmd[*]}"
  nice -n 10 /usr/bin/time -l "${cmd[@]}" > "logs/$name.log" 2>&1
  local rc=$?; rm -rf "$lock"; echo "$(ts) END $name rc $rc"; return $rc
}
L3="--es internal --refit --wpow 1 --leaves 255 --mdl 100 --no_direct --clf_drop_prefix m_,ab_ --full_mult 1.0"
if [ "$LANE" = 1 ]; then
  step f2sk_fj cache/fj_f2sk_s0.parquet 4 30 30 "$PY" -u research/f2/model3_fj.py --setting fj --seed 0 --tag f2sk --feats feats_v27 $L3 --threads {TH} --mu_sep LIRF || exit 1
  step f2skg_fj cache/fj_f2skg_s0.parquet 1 30 5 "$PY" -u research/f2/g_band_fj.py --setting fj --seed 0 --tag f2skg --base f2sk --feats feats_v27 || exit 1
  [ "${LIGHT:-0}" = 1 ] && { echo "$(ts) lane 1 light part done"; exit 0; }
  step f2cb_fj cache/fj_f2cb_s0.parquet "$CT" 30 200 "$PY" -u research/f2/cat_mu_fj.py --setting fj --seed 0 --tag f2cb --base f2sk --feats feats_v27 --threads {TH} || exit 1
  touch f2_fjlane1.DONE
elif [ "$LANE" = 2 ]; then
  step f2skap_fj cache/fj_f2skap_s0.parquet 4 30 25 "$PY" -u research/f2/model3ap_fj.py --setting fj --seed 0 --tag f2skap --feats feats_v27 $L3 --threads {TH} --mu_sep all || exit 1
  step f2skapg_fj cache/fj_f2skapg_s0.parquet 1 30 5 "$PY" -u research/f2/g_band_fj.py --setting fj --seed 0 --tag f2skapg --base f2skap --feats feats_v27 || exit 1
  [ "${LIGHT:-0}" = 1 ] && { echo "$(ts) lane 2 light part done"; exit 0; }
  step f2cbwin_fj cache/fj_f2cbwin_s0.parquet "$CT" 30 260 "$PY" -u research/f2/cat_mu_fj.py --setting fj --seed 0 --tag f2cbwin --base f2skwin --feats feats_v27 --anchor win --threads {TH} || exit 1
  step f2cbap_fj cache/fj_f2cbap_s0.parquet "$CT" 30 120 "$PY" -u research/f2/cat_muap_fj.py --setting fj --seed 0 --tag f2cbap --base f2skap --feats feats_v27 --threads {TH} --per_ap || exit 1
  touch f2_fjlane2.DONE
elif [ "$LANE" = 4 ]; then
  step f2cb_fj cache/fj_f2cb_s0.parquet "$CT" 30 200 "$PY" -u research/f2/cat_mu_fj.py --setting fj --seed 0 --tag f2cb --base f2sk --feats feats_v27 --threads {TH} || exit 1
elif [ "$LANE" = 5 ]; then
  step f2cbwin_fj cache/fj_f2cbwin_s0.parquet "$CT" 30 260 "$PY" -u research/f2/cat_mu_fj.py --setting fj --seed 0 --tag f2cbwin --base f2skwin --feats feats_v27 --anchor win --threads {TH} || exit 1
elif [ "$LANE" = 6 ]; then
  step f2cbap_fj cache/fj_f2cbap_s0.parquet "$CT" 30 120 "$PY" -u research/f2/cat_muap_fj.py --setting fj --seed 0 --tag f2cbap --base f2skap --feats feats_v27 --threads {TH} --per_ap || exit 1
elif [ "$LANE" = 3 ]; then
  step f2rmwin_fj cache/fj_f2rmwin_s0.parquet 1 50 20 env OMP_NUM_THREADS=1 "$PY" -u research/f2/rmlp_mu_fj.py --setting fj --seed 0 --tag f2rmwin --base f2skwin --feats feats_v27 --device mps --anchor win --threads 1 || exit 1
  step f2rm_fj cache/fj_f2rm_s0.parquet 1 50 20 env OMP_NUM_THREADS=1 "$PY" -u research/f2/rmlp_mu_fj.py --setting fj --seed 0 --tag f2rm --base f2sk --feats feats_v27 --device mps --threads 1 || exit 1
  touch f2_fjlane3.DONE
fi
echo "$(ts) lane $LANE done"
