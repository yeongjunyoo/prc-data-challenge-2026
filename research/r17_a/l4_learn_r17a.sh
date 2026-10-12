#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# R17-A L4 learners on the Mac public root ~/nova-adsb/prcm/r17a_sbx (coordinator 10-10 10:1x; registration in the PRC master 10-10 10:07
# "L4 착수와 거부권 고침"): the 15 CatBoost and RealMLP learners of 514 refit on feats_v28 and L3's LightGBM bases, one setting per call
# (val first; fwd only after the val part of gate 7.8 passes; full is never fitted here, F3 fits it on the final root).
#   cbwin seeds 0-2  src/cat_mu.py --setting S --seed k --tag m6r28cbwin --base <B> --feats feats_v28 --anchor win --threads 10
#   cb    seeds 0-2  src/cat_mu.py --setting S --seed k --tag m6r28cb --base <B> --feats feats_v28 --threads 10
#   cbap  seed 0     research/r13_f3/cat_muap.py --setting S --seed 0 --tag m6r28cbap --base m6r28skap --feats feats_v28 --threads 10 --per_ap
#   rmwin seeds 0-3  src/rmlp_mu.py --setting S --seed k --tag m6r28rmwin --base <B> --feats feats_v28 --device mps --anchor win --threads 1
#   rm    seeds 0-3  src/rmlp_mu.py --setting S --seed k --tag m6r28rm --base <B> --feats feats_v28 --device mps --threads 1
# <B>: seed 0 reads the L3 leg itself (m6r28skwin, m6r28sk); seeds >= 1 read <leg>B0, links cache/<S>_<leg>B0_s<k>.parquet -> the leg's
# seed-0 file, as 514's seeds >= 1 did (research/r10_b/mac_scripts/r10b_night9_mac.sh: "the base links <tag>B0_s<k> point at seed 0").
# Code is 514's (sha16 checked before any run: cat_mu 05956dbe5a32bfa9, rmlp_mu 28fe9c7543664c2b, cat_muap 0ab44b8b9e4b276a).
# Mac rules for L4: one heavy job at a time. CatBoost one leg at a time at 10 threads (R16-D: same output at 8 and 10), started after two
# checks 30 s apart of no arc_quiet.*, declared threads of the other locks + 10 <= 12, memory_pressure free >= 30 %. RealMLP alone on MPS:
# other locks declare <= 2 threads, free >= 25 % (the L4 override, logged with its time when free is below 30 % at a start), and a watchdog
# kills the run and stops everything if free falls below 20 %. Lock prcm/r17a_l4_<job>.lock only while a job runs; nice 10; /usr/bin/time -l.
# A finished output (cache/<S>_<tag>_s<k>.parquet) is kept and skipped on a rerun. Stops at the first failed job (research/r17_a/l4.FAILED).
#   nohup /bin/bash research/r17_a/l4_learn_r17a.sh val >> research/r17_a/logs/l4_learn_val.log 2>&1 < /dev/null &
cd ~/nova-adsb/prcm/r17a_sbx || exit 1
S=${1:?setting val or fwd}
case "$S" in val|fwd) ;; *) echo "setting must be val or fwd (full is fitted on the final root only)"; exit 2 ;; esac
P=~/nova-adsb/prcm
PY=~/nova-adsb/venv311/bin/python
export PATH=$HOME/nova-adsb/venv311/bin:$PATH   # as 514's learner scripts (r10b_rmretry_mac.sh)
LANE=research/r17_a
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "$(ts) FAILED: $1"; echo "$S: $1" > "$LANE/l4.FAILED"; exit 1; }
[ -e "$LANE/l4.FAILED" ] && { echo "$(ts) $LANE/l4.FAILED exists: $(cat "$LANE/l4.FAILED"); not starting"; exit 1; }
for x in "src/cat_mu.py 05956dbe5a32bfa9" "src/rmlp_mu.py 28fe9c7543664c2b" "research/r13_f3/cat_muap.py 0ab44b8b9e4b276a"; do
  set -- $x
  [ "$(shasum -a 256 "$1" | cut -c1-16)" = "$2" ] || fail "code sha16 of $1 is not 514's $2"
done
for b in m6r28skwin m6r28sk; do
  [ -s "cache/${S}_${b}_s0.parquet" ] || fail "missing cache/${S}_${b}_s0.parquet"
  for k in 1 2 3; do
    l="cache/${S}_${b}B0_s$k.parquet"
    if [ -L "$l" ]; then [ "$(readlink "$l")" = "${S}_${b}_s0.parquet" ] || fail "$l points elsewhere"
    elif [ -e "$l" ]; then fail "$l exists and is not a link"
    else ln -s "${S}_${b}_s0.parquet" "$l"; echo "$(ts) link $l -> ${S}_${b}_s0.parquet"; fi
  done
done
[ -s "cache/${S}_m6r28skap_s0.parquet" ] || fail "missing cache/${S}_m6r28skap_s0.parquet"
declared() {  # declared threads of every lock except this lane's L4 locks
  local s=0 t f
  for f in "$P"/*.lock/info; do
    [ -e "$f" ] || continue
    case "$f" in "$P"/r17a_l4_*) continue ;; esac
    t=$(grep -i -m1 '^threads' "$f" | grep -o '[0-9][0-9.]*' | head -1)
    s=$(echo "$s + ${t:-6}" | bc)
  done
  echo "$s"
}
free_pct() { memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %; }
ok_cb() { { [ -e "$P/arc_quiet.request" ] || [ -e "$P/arc_quiet.lock" ]; } && return 1; [ "$(echo "$(declared) + 10 <= 12" | bc)" = 1 ] || return 1; [ "$(free_pct)" -ge 30 ]; }
ok_rm() { { [ -e "$P/arc_quiet.request" ] || [ -e "$P/arc_quiet.lock" ]; } && return 1; [ "$(echo "$(declared) <= 2" | bc)" = 1 ] || return 1; [ "$(free_pct)" -ge 25 ]; }
run_job() {  # run_job <kind cb|rm> <tag> <seed> <peak GB> <threads> <command ...>
  local kind=$1 tag=$2 k=$3 peak=$4 thr=$5; shift 5
  local out="cache/${S}_${tag}_s$k.parquet" log="logs/${tag}_${S}_s$k.log" name="${tag}_${S}_s$k" L n=0 rc t0 wd=""
  if [ -s "$out" ]; then echo "$(ts) done earlier: $name"; return 0; fi
  while [ $n -lt 2 ]; do if ok_$kind; then n=$((n + 1)); else n=0; fi; [ $n -lt 2 ] && sleep 30; done
  local f; f=$(free_pct)
  [ "$kind" = rm ] && [ "$f" -lt 30 ] && echo "$(ts) L4 memory override: starting $name at memory free $f % (floor 25 % for L4 RealMLP, coordinator 10-10)"
  L="$P/r17a_l4_$name.lock"
  mkdir "$L" || fail "lock exists: $L"
  printf 'owner: PRC R17-A L4 learner (%s), coordinator main\npid: %s\npeak_gb: %s\nthreads: %s\nstart: %s\nexpected_end: %s KST\njob: %s\n' \
    "$name" "$$" "$peak" "$thr" "$(date '+%Y-%m-%d %H:%M')" "$(date -v+60M '+%Y-%m-%d %H:%M')" "$(echo "$*" | cut -c1-200)" > "$L/info"
  echo "$(ts) start $name (free $f %, declared by others $(declared)): $*"
  t0=$(date +%s)
  OMP_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8 PYTHONFAULTHANDLER=1 nice -n 10 /usr/bin/time -l "$PY" -X utf8 -u "$@" > "$log" 2>&1 &
  local jp=$!
  if [ "$kind" = rm ]; then
    ( while kill -0 $jp 2> /dev/null; do
        f=$(free_pct)
        if [ "${f:-100}" -lt 20 ]; then echo "$(ts) WATCHDOG: memory free $f % < 20 % during $name; killing pid $jp"; pkill -P $jp; kill $jp; touch "$LANE/l4_mem_stop.flag"; break; fi
        sleep 10
      done ) &
    wd=$!
  fi
  wait $jp; rc=$?
  [ -n "$wd" ] && kill $wd 2> /dev/null
  rm -rf "$L"
  echo "$(ts) end $name rc $rc ($(( $(date +%s) - t0 )) s, $(grep -E 'peak memory footprint' "$log" | awk '{printf "%.2f GB peak", $1/2^30}'))"
  [ -e "$LANE/l4_mem_stop.flag" ] && fail "memory watchdog stopped $name (free below 20 %)"
  [ $rc = 0 ] && [ -s "$out" ] || fail "$name rc $rc (log $log)"
}
echo "$(ts) L4 learners, setting $S, pid $$"
for k in 0 1 2; do b=m6r28skwin; [ $k -gt 0 ] && b=m6r28skwinB0
  run_job cb m6r28cbwin $k 10 10 src/cat_mu.py --setting "$S" --seed $k --tag m6r28cbwin --base $b --feats feats_v28 --anchor win --threads 10; done
for k in 0 1 2; do b=m6r28sk; [ $k -gt 0 ] && b=m6r28skB0
  run_job cb m6r28cb $k 10 10 src/cat_mu.py --setting "$S" --seed $k --tag m6r28cb --base $b --feats feats_v28 --threads 10; done
run_job cb m6r28cbap 0 10 10 research/r13_f3/cat_muap.py --setting "$S" --seed 0 --tag m6r28cbap --base m6r28skap --feats feats_v28 --threads 10 --per_ap
for k in 0 1 2 3; do b=m6r28skwin; [ $k -gt 0 ] && b=m6r28skwinB0
  run_job rm m6r28rmwin $k 14 1 src/rmlp_mu.py --setting "$S" --seed $k --tag m6r28rmwin --base $b --feats feats_v28 --device mps --anchor win --threads 1; done
for k in 0 1 2 3; do b=m6r28sk; [ $k -gt 0 ] && b=m6r28skB0
  run_job rm m6r28rm $k 14 1 src/rmlp_mu.py --setting "$S" --seed $k --tag m6r28rm --base $b --feats feats_v28 --device mps --threads 1; done
date '+%Y-%m-%d %H:%M:%S' > "$LANE/l4_learn_$S.DONE"
echo "$(ts) L4 learners $S done"
