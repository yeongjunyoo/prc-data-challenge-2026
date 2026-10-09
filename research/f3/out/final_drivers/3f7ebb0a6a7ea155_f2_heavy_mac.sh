#!/bin/bash
# F2 gate 1.4 inputs and gate on the Mac (coordinator 10-08 23:1x: run after F3's legs finish, about 15:00-19:00 KST on 10-09).
#   nohup /bin/bash f2_heavy_mac.sh >> logs/f2_heavy.log 2>&1 < /dev/null &
# Waits for the marker ~/nova-adsb/prcm/f2_run/GO_FJ (touched by the coordinator, F3 or F2 when F3's legs are done), then:
#   1. CatBoost fj cb, cbwin, cbap in parallel at 4 threads each (each launch keeps the PROTOCOL checks: other locks + 4 <= 12, memory >= 30%)
#   2. RealMLP fj rmwin then rm, alone on MPS (memory free >= 50%)
#   3. XGBoost placeholder (its blend weight is exactly 0), blend_fixed fj x 3, weight check
#   4. gate14 at 2 threads
set -u
cd ~/nova-adsb/prcm/f2_run || exit 1
P=~/nova-adsb/prcm
PY=~/nova-adsb/venv311/bin/python
ts() { date '+%m-%d %H:%M:%S'; }
sha() { shasum -a 256 "$1" | cut -c1-16; }
until [ -e GO_FJ ]; do sleep 60; done
echo "$(ts) GO_FJ seen"
for L in 1 2; do [ -e "cache/fj_f2sk_s0.parquet" ] && [ -e "cache/fj_f2skap_s0.parquet" ] && break; sleep 1; done
[ -e cache/fj_f2skg_s0.parquet ] && [ -e cache/fj_f2skapg_s0.parquet ] || { echo "$(ts) LightGBM fj legs missing, running them first (4 threads)"; LIGHT=1 /bin/bash f2_fjlegs_mac.sh 1 4; LIGHT=1 /bin/bash f2_fjlegs_mac.sh 2 4; }
(/bin/bash f2_fjlegs_mac.sh 4 4 >> logs/f2_fjlegs_4.log 2>&1) &
sleep 90
(/bin/bash f2_fjlegs_mac.sh 5 4 >> logs/f2_fjlegs_5.log 2>&1) &
sleep 90
(/bin/bash f2_fjlegs_mac.sh 6 4 >> logs/f2_fjlegs_6.log 2>&1) &
wait
echo "$(ts) CatBoost fj done: $(ls cache/fj_f2cb_s0.parquet cache/fj_f2cbwin_s0.parquet cache/fj_f2cbap_s0.parquet 2>&1 | tr '\n' ' ')"
/bin/bash f2_fjlegs_mac.sh 3 1 >> logs/f2_fjlegs_3.log 2>&1
echo "$(ts) RealMLP fj done: $(ls cache/fj_f2rmwin_s0.parquet cache/fj_f2rm_s0.parquet 2>&1 | tr '\n' ' ')"
"$PY" -u research/f2/fj_blends.py > logs/fj_blends.log 2>&1 || { echo "$(ts) blends failed"; exit 1; }
echo "$(ts) blends done"
mkdir "$P/f2_gate.lock" && printf 'owner: PRC F2 gate 1.4 (four-month corrector), coordinator main\npid: %s\npeak_gb: 5\nthreads: 2\nstart: %s\nexpected_end: %s KST\njob: research/f2/gate14.py\n' \
  "$$" "$(date '+%Y-%m-%d %H:%M')" "$(date -v+60M '+%Y-%m-%d %H:%M')" > "$P/f2_gate.lock/info"
nice -n 10 /usr/bin/time -l "$PY" -u research/f2/gate14.py --ctx f2ctx_v3 --ctx_sha "$(sha cache/f2ctx_v3.parquet)" --threads 2 > logs/gate14.log 2>&1
rc=$?; rm -rf "$P/f2_gate.lock"; echo "$(ts) gate14 rc $rc"; grep -E "GATE 1.4" logs/gate14.log
