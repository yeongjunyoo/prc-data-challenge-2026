#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 (coordinator 10-09 04:xx KST): final_run/REALMLP_ALONE exists from the end of the CatBoost phase (the RealMLP phase starts right after
# it in phase2c.sh) until the RealMLP phase has finished; R17-A's downloader starts no new day while it exists.
set -u
cd ${PRC_WORK:-.}/final_run || exit 1
ts() { date '+%m-%d %H:%M:%S'; }
until grep -q "phase cb finished" logs/orch_cb.log 2>/dev/null; do sleep 10; done
printf 'owner: PRC F3 final rebuild (coordinator main)\npurpose: RealMLP full fits run alone on the Mac (18 GB); R17-A starts no new download day while this exists\ncreated: %s\nremoved: when logs/orch_rm.log says the RealMLP phase finished\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" > REALMLP_ALONE
echo "$(ts) CatBoost phase finished: REALMLP_ALONE created"
until grep -q "phase rm finished" logs/orch_rm.log 2>/dev/null; do sleep 10; done
rm -f REALMLP_ALONE
echo "$(ts) RealMLP phase finished: REALMLP_ALONE removed"
