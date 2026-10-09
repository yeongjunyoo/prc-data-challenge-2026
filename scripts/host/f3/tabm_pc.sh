#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 final TabM leg on the PC GPU (coordinator: PC CUDA when the PC guard allows, else Mac MPS; the device is recorded). Copies the
# full-setting TabM input built on the Mac (final_run/research/r13_l/x/full by prep_x.py) and the LightGBM base full_m6r27skwin_s0 into the
# PC copy of the final root ($PC_FINAL_ROOT), waits until commit < 85 % and available >= 4 GB on three checks 60 s apart, takes
# $PC_LOCKS/gpu.lock and pc_heavy.lock (never while waiting), runs research/r13_l/tabm_leg.py there with the 514 arguments and
# --device cuda under memcap 7 GB (R13-L's measured commit peak 6.70 GB) at BelowNormal priority, and copies the output back to the Mac.
#   bash research/f3/tabm_pc.sh   (Git Bash on the PC)
set -u
R=${PC_FINAL_ROOT:-.}
M=${MAC_HOST:-mac}:${MAC_FINAL_ROOT:-final_run}
PYW=${PY_PC:-python}
HL=${PC_LOCKS:-.}/pc_heavy.lock
GL=${PC_LOCKS:-.}/gpu.lock
LOG=$R/logs/final_tmcwin_s0_pc.log
ts() { date '+%m-%d %H:%M:%S'; }
mem() { powershell -NoProfile -Command "\$c = Get-Counter '\Memory\Available MBytes','\Memory\Committed Bytes','\Memory\Commit Limit'; \$v = \$c.CounterSamples | ForEach-Object { \$_.CookedValue }; \"\$([int]\$v[0]) \$([int](100 * \$v[1] / \$v[2]))\""; }
mkdir -p $R/research/r13_l/x/full $R/cache $R/preds $R/logs $R/research/r13_l/runs
[ -e $R/cache/full_m6r27tmcwin_s0.parquet ] && { echo "$(ts) output exists"; exit 0; }
echo "$(ts) waiting for the Mac's tabm_prep and final skwin_s0 (orchestrator states rc 0)" | tee -a $LOG
until ssh ${MAC_HOST:-mac} 'cd ${PRC_WORK:-.}/final_run && ${PY_MAC:-python3} -c "import json,sys; a=json.load(open(\"research/f3/out/orch_tabm.json\")); b=json.load(open(\"research/f3/out/orch_lgb.json\")); sys.exit(0 if a.get(\"tabm_prep\",{}).get(\"rc\")==0 and b.get(\"skwin_s0\",{}).get(\"rc\")==0 else 1)" 2>/dev/null'; do sleep 60; done
echo "$(ts) copy inputs from the Mac" | tee -a $LOG
scp -q "$M/research/r13_l/x/full/*" $R/research/r13_l/x/full/ && scp -q $M/cache/full_m6r27skwin_s0.parquet $R/cache/ || { echo "copy failed"; exit 2; }
FL="research/r13_l/x/full/C.npy research/r13_l/x/full/NA.npy research/r13_l/x/full/X.npy research/r13_l/x/full/meta.json research/r13_l/x/full/rows.npz cache/full_m6r27skwin_s0.parquet"
ssh ${MAC_HOST:-mac} "cd ${PRC_WORK:-.}/final_run && for f in $FL; do shasum -a 256 \$f | cut -c1-16; done" > /tmp/tabm_mac_sha.txt
(cd $R && for f in $FL; do sha256sum $f | cut -c1-16; done) > /tmp/tabm_pc_sha.txt
cmp -s /tmp/tabm_mac_sha.txt /tmp/tabm_pc_sha.txt || { echo "copied inputs differ from the Mac's"; exit 3; }
while true; do
  ok=0
  while [ $ok -lt 3 ]; do
    read av cm < <(mem)
    if [ "${av:-0}" -ge 4096 ] && [ "${cm:-100}" -lt 85 ]; then ok=$((ok + 1)); else ok=0; fi
    [ $ok -lt 3 ] && sleep 60
  done
  mkdir "$GL" 2>/dev/null || { sleep 30; continue; }
  mkdir "$HL" 2>/dev/null || { rm -rf "$GL"; sleep 30; continue; }
  break
done
trap 'rm -rf "$GL" "$HL"' EXIT
for L in "$GL" "$HL"; do
  printf 'owner: PRC F3 final TabM leg on the PC GPU (coordinator main)\npid: %s\npeak_gb: 7\nstart: %s\nexpected_end: about 15 min\njob: tabm_leg full s0 m6r27tmcwin\n' "$$" "$(date '+%Y-%m-%d %H:%M:%S')" > "$L/info"
done
echo "$(ts) locks taken (available ${av} MB, commit ${cm} %), device cuda" | tee -a $LOG
cd $R && $PYW -c "import ctypes, subprocess, sys; k = ctypes.windll.kernel32; k.SetPriorityClass(k.GetCurrentProcess(), 0x4000); sys.exit(subprocess.call(sys.argv[1:]))" \
  $PYW ${MEMCAP:-memcap.py} --gb 7 -- $PYW -X utf8 -u research/r13_l/tabm_leg.py --setting full --seed 0 \
  --tag m6r27tmcwin --base m6r27skwin --anchor win --sched cos --epochs 20 --patience 99 --device cuda --threads 3 --amp bf16 >> $LOG 2>&1
rc=$?
echo "$(ts) tabm rc $rc" | tee -a $LOG
rm -rf "$GL" "$HL"; trap - EXIT
[ $rc -ne 0 ] && exit $rc
scp -q $R/cache/full_m6r27tmcwin_s0.parquet $M/cache/ && scp -q $R/preds/es_valid_ids_m6r27tmcwin_full_s0.parquet $M/preds/ 2>/dev/null
ssh ${MAC_HOST:-mac} "cd ${PRC_WORK:-.}/final_run && shasum -a 256 cache/full_m6r27tmcwin_s0.parquet | cut -c1-16"
sha256sum $R/cache/full_m6r27tmcwin_s0.parquet | cut -c1-16
echo "$(ts) TabM done on the PC GPU (cuda), output copied to the Mac root" | tee -a $LOG
