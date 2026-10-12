#!/bin/bash
# F3 L4 learners on the Mac final root ~/nova-adsb/prcm/final_run (coordinator 10-10 10:07 "L4 착수와 거부권 고침", split (a): R17-A fits val and
# fwd on the public root and runs gate 7.8; F3 fits full on the final root only after the coordinator's go). A copy of R17-A's runner
# r17a_sbx research/r17_a/l4_learn_r17a.sh (sha16 e822599569c2b963) with the same job lines, --base strings, env, code checks and Mac rules;
# what differs: the root, the settings (full; fj = the coordinator's 10:2x consistent c form: F2's five leave-Feb/Jun-out learners refit on
# feats_v28 with F2's cat_mu_fj.py, cat_muap_fj.py, rmlp_mu_fj.py, the commands of f2_run research/f2/mac/f2_fjlegs_mac.sh with feats_v28, L3's
# fj LightGBM legs as bases and 10 CatBoost threads), and this lane's lock and file names.
#   full: cbwin seeds 0-2  src/cat_mu.py --setting full --seed k --tag m6r28cbwin --base <B> --feats feats_v28 --anchor win --threads 10
#         cb    seeds 0-2  src/cat_mu.py --setting full --seed k --tag m6r28cb --base <B> --feats feats_v28 --threads 10
#         cbap  seed 0     research/r13_f3/cat_muap.py --setting full --seed 0 --tag m6r28cbap --base m6r28skap --feats feats_v28 --threads 10 --per_ap
#         rmwin seeds 0-3  src/rmlp_mu.py --setting full --seed k --tag m6r28rmwin --base <B> --feats feats_v28 --device mps --anchor win --threads 1
#         rm    seeds 0-3  src/rmlp_mu.py --setting full --seed k --tag m6r28rm --base <B> --feats feats_v28 --device mps --threads 1
#         then the seed averages m6r28cbwinx3, m6r28cbx3, m6r28rmwinx4, m6r28rmx4 with R17-A's own averager research/r17_a/avg_l4.py (the code of
#         its val and fwd averages, cloned by research/f3/collect_l4.py with a sha check), settings full
#         <B>: seed 0 reads the L3 leg itself (m6r28skwin, m6r28sk); seeds >= 1 read <leg>B0, links cache/full_<leg>B0_s<k>.parquet -> the leg's
#         seed-0 file (as 514's and R17-A's seeds >= 1). CatBoost full reads cache/iters_m6r28cb{win,,ap}.json from R17-A's val seed-0 runs.
#   fj:   l4cbwin  research/f2/cat_mu_fj.py --setting fj --seed 0 --tag l4cbwin --base l3skwin --feats feats_v28 --anchor win --threads 10
#         l4cb     research/f2/cat_mu_fj.py --setting fj --seed 0 --tag l4cb --base l3sk --feats feats_v28 --threads 10
#         l4cbap   research/f2/cat_muap_fj.py --setting fj --seed 0 --tag l4cbap --base l3skap --feats feats_v28 --threads 10 --per_ap
#         l4rmwin  research/f2/rmlp_mu_fj.py --setting fj --seed 0 --tag l4rmwin --base l3skwin --feats feats_v28 --device mps --anchor win --threads 1
#         l4rm     research/f2/rmlp_mu_fj.py --setting fj --seed 0 --tag l4rm --base l3sk --feats feats_v28 --device mps --threads 1
# Mac rules (L4): one heavy job at a time. CatBoost one leg at a time at 10 threads, started after two checks 30 s apart of no arc_quiet.*,
# declared threads of the other locks + 10 <= 12, memory_pressure free >= 30 %. RealMLP alone on MPS: other locks declare <= 2 threads, free >= 25 %
# (logged when below 30 % at a start), and a watchdog kills the run and stops everything if free falls below 20 %. Lock prcm/f3_l4_<job>.lock only
# while a job runs; nice 10; /usr/bin/time -l. A finished output is kept and skipped on a rerun. Stops at the first failed job (research/f3/out/l4.FAILED).
# Before every learner: waits while research/f3/PHASE7_HOLD exists (no lock held) and ends cleanly with exit 3 if research/f3/PHASE7_STOP exists.
#   nohup /bin/bash research/f3/l4_full_f3.sh full >> logs/l4_full.log 2>&1 < /dev/null &
cd ~/nova-adsb/prcm/final_run || exit 1
S=${1:?setting full or fj}
case "$S" in full|fj) ;; *) echo "setting must be full or fj (val and fwd are R17-A's, on the public root)"; exit 2 ;; esac
P=~/nova-adsb/prcm
PY=~/nova-adsb/venv311/bin/python
export PATH=$HOME/nova-adsb/venv311/bin:$PATH   # as 514's and R17-A's learner scripts
LANE=research/f3/out
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "$(ts) FAILED: $1"; echo "$S: $1" > "$LANE/l4.FAILED"; exit 1; }
[ -e "$LANE/l4.FAILED" ] && { echo "$(ts) $LANE/l4.FAILED exists: $(cat "$LANE/l4.FAILED"); not starting"; exit 1; }
for x in "src/cat_mu.py 05956dbe5a32bfa9" "src/rmlp_mu.py 28fe9c7543664c2b" "research/r13_f3/cat_muap.py 0ab44b8b9e4b276a"; do
  set -- $x
  [ "$(shasum -a 256 "$1" | cut -c1-16)" = "$2" ] || fail "code sha16 of $1 is not 514's $2"
done
if [ "$S" = full ]; then
  for b in m6r28skwin m6r28sk; do
    [ -s "cache/full_${b}_s0.parquet" ] || fail "missing cache/full_${b}_s0.parquet"
    for k in 1 2 3; do
      l="cache/full_${b}B0_s$k.parquet"
      if [ -L "$l" ]; then [ "$(readlink "$l")" = "full_${b}_s0.parquet" ] || fail "$l points elsewhere"
      elif [ -e "$l" ]; then fail "$l exists and is not a link"
      else ln -s "full_${b}_s0.parquet" "$l"; echo "$(ts) link $l -> full_${b}_s0.parquet"; fi
    done
  done
  [ -s cache/full_m6r28skap_s0.parquet ] || fail "missing cache/full_m6r28skap_s0.parquet"
  for t in m6r28cbwin m6r28cb m6r28cbap; do [ -s "cache/iters_$t.json" ] || fail "missing cache/iters_$t.json (R17-A's val seed-0 run, research/f3/collect_l4.py)"; done
  a=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['copied']['research/r17_a/avg_l4.py']['sha16'])" "$LANE/collect_l4.json" 2> /dev/null)
  [ -n "$a" ] && [ "$(shasum -a 256 research/r17_a/avg_l4.py | cut -c1-16)" = "$a" ] || fail "research/r17_a/avg_l4.py is not the copy research/f3/collect_l4.py cloned ($a)"
else
  for x in "cat_mu_fj.py b3ade78c854c99f8" "cat_muap_fj.py dc756eb3fa606a94" "rmlp_mu_fj.py 75b9b77b0127e8b8"; do
    set -- $x
    [ -e "research/f2/$1" ] || cp -cp "$P/f2_run/research/f2/$1" "research/f2/$1" || fail "copy research/f2/$1 from f2_run"
    [ "$(shasum -a 256 "research/f2/$1" | cut -c1-16)" = "$2" ] || fail "code sha16 of research/f2/$1 is not F2's $2"
  done
  for b in l3skwin l3sk l3skap; do [ -s "cache/fj_${b}_s0.parquet" ] || fail "missing cache/fj_${b}_s0.parquet"; done
fi
declared() {  # declared threads of every lock except this lane's L4 locks
  local s=0 t f
  for f in "$P"/*.lock/info; do
    [ -e "$f" ] || continue
    case "$f" in "$P"/f3_l4_*) continue ;; esac
    t=$(grep -i -m1 '^threads' "$f" | grep -o '[0-9][0-9.]*' | head -1)
    s=$(echo "$s + ${t:-6}" | bc)
  done
  echo "$s"
}
free_pct() { memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %; }
ok_cb() { { [ -e "$P/arc_quiet.request" ] || [ -e "$P/arc_quiet.lock" ]; } && return 1; [ "$(echo "$(declared) + 10 <= 12" | bc)" = 1 ] || return 1; [ "$(free_pct)" -ge 30 ]; }
ok_rm() { { [ -e "$P/arc_quiet.request" ] || [ -e "$P/arc_quiet.lock" ]; } && return 1; [ "$(echo "$(declared) <= 2" | bc)" = 1 ] || return 1; [ "$(free_pct)" -ge 25 ]; }
hold_or_stop() {  # research/f3/phase7.sh's hold and stop files (coordinator 11:2x), checked before every learner; no lock is held while waiting
  local n=0
  while [ -e research/f3/PHASE7_HOLD ] && [ ! -e research/f3/PHASE7_STOP ]; do
    [ $((n % 20)) = 0 ] && echo "$(ts) paused before $1: research/f3/PHASE7_HOLD present (no lock held)"
    n=$((n + 1)); sleep 60
  done
  if [ -e research/f3/PHASE7_STOP ]; then echo "$(ts) stopped cleanly before $1: research/f3/PHASE7_STOP"; exit 3; fi
  return 0
}
run_job() {  # run_job <kind cb|rm> <tag> <seed> <peak GB> <threads> <command ...>
  local kind=$1 tag=$2 k=$3 peak=$4 thr=$5; shift 5
  local out="cache/${S}_${tag}_s$k.parquet" log="logs/${tag}_${S}_s$k.log" name="${tag}_${S}_s$k" L n=0 rc t0 wd=""
  if [ -s "$out" ]; then echo "$(ts) done earlier: $name"; return 0; fi
  hold_or_stop "$name"
  [ -e STOP_F3 ] && fail "STOP_F3 present before $name"
  while [ $n -lt 2 ]; do if ok_$kind; then n=$((n + 1)); else n=0; fi; [ $n -lt 2 ] && sleep 30; done
  local f; f=$(free_pct)
  [ "$kind" = rm ] && [ "$f" -lt 30 ] && echo "$(ts) L4 memory override: starting $name at memory free $f % (floor 25 % for L4 RealMLP, coordinator 10-10)"
  L="$P/f3_l4_$name.lock"
  mkdir "$L" || fail "lock exists: $L"
  printf 'owner: PRC F3 L4 learner on the final root (%s), coordinator main\npid: %s\npeak_gb: %s\nthreads: %s\nstart: %s\nexpected_end: %s KST\njob: %s\n' \
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
echo "$(ts) L4 learners on the final root, setting $S, pid $$ (l4_full_f3.sh $(shasum -a 256 research/f3/l4_full_f3.sh | cut -c1-16))"
if [ "$S" = full ]; then
  for k in 0 1 2; do b=m6r28skwin; [ $k -gt 0 ] && b=m6r28skwinB0
    run_job cb m6r28cbwin $k 10 10 src/cat_mu.py --setting full --seed $k --tag m6r28cbwin --base $b --feats feats_v28 --anchor win --threads 10; done
  for k in 0 1 2; do b=m6r28sk; [ $k -gt 0 ] && b=m6r28skB0
    run_job cb m6r28cb $k 10 10 src/cat_mu.py --setting full --seed $k --tag m6r28cb --base $b --feats feats_v28 --threads 10; done
  run_job cb m6r28cbap 0 10 10 research/r13_f3/cat_muap.py --setting full --seed 0 --tag m6r28cbap --base m6r28skap --feats feats_v28 --threads 10 --per_ap
  for k in 0 1 2 3; do b=m6r28skwin; [ $k -gt 0 ] && b=m6r28skwinB0
    run_job rm m6r28rmwin $k 14 1 src/rmlp_mu.py --setting full --seed $k --tag m6r28rmwin --base $b --feats feats_v28 --device mps --anchor win --threads 1; done
  for k in 0 1 2 3; do b=m6r28sk; [ $k -gt 0 ] && b=m6r28skB0
    run_job rm m6r28rm $k 14 1 src/rmlp_mu.py --setting full --seed $k --tag m6r28rm --base $b --feats feats_v28 --device mps --threads 1; done
  for x in "m6r28cbwin m6r28cbwinx3 0,1,2" "m6r28cb m6r28cbx3 0,1,2" "m6r28rmwin m6r28rmwinx4 0,1,2,3" "m6r28rm m6r28rmx4 0,1,2,3"; do
    set -- $x
    if [ -s "cache/full_$2_s0.parquet" ]; then echo "$(ts) done earlier: average $2"; continue; fi
    nice -n 10 "$PY" -X utf8 research/r17_a/avg_l4.py $1 $2 $3 full > "logs/avg_$2.log" 2>&1 || fail "average $2 (logs/avg_$2.log)"
    echo "$(ts) average $2: $(tail -1 "logs/avg_$2.log" | cut -c1-160)"
  done
else
  run_job cb l4cbwin 0 10 10 research/f2/cat_mu_fj.py --setting fj --seed 0 --tag l4cbwin --base l3skwin --feats feats_v28 --anchor win --threads 10
  run_job cb l4cb 0 10 10 research/f2/cat_mu_fj.py --setting fj --seed 0 --tag l4cb --base l3sk --feats feats_v28 --threads 10
  run_job cb l4cbap 0 10 10 research/f2/cat_muap_fj.py --setting fj --seed 0 --tag l4cbap --base l3skap --feats feats_v28 --threads 10 --per_ap
  run_job rm l4rmwin 0 14 1 research/f2/rmlp_mu_fj.py --setting fj --seed 0 --tag l4rmwin --base l3skwin --feats feats_v28 --device mps --anchor win --threads 1
  run_job rm l4rm 0 14 1 research/f2/rmlp_mu_fj.py --setting fj --seed 0 --tag l4rm --base l3sk --feats feats_v28 --device mps --threads 1
fi
date '+%Y-%m-%d %H:%M:%S' > "$LANE/l4_learn_$S.DONE"
echo "$(ts) L4 learners $S done"
