#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F2 Package S sandbox clone (10-11 01:17: a new APFS clone; never edit r17a_sbx or final_run in place; new files only).
#   598c: ${PRC_WORK:-.}/r17a_sbx     -> ${PRC_WORK:-.}/f2s_sbx598   (L3S chains A, B, C at 0.75 = 570 = 596's val and fwd)
#   l4:   ${PRC_WORK:-.}/r17a_l4g_sbx -> ${PRC_WORK:-.}/f2s_sbxl4    (the L4 gate-only chains, phase f, 0.75 = 600's val and fwd)
# Refuses if the clone exists, if any process has its working directory inside the source, if the volume has under 15 GB free, or if a
# link would reach a live root (03:1x fix 5): research/f2/s_linkscan.py reads every link of the source as it will resolve in
# the clone before cloning, and the clone's own links after; a link into ${PRC_WORK:-.} or another root there (final_run, r17a_sbx,
# r17a_l4g_sbx, f2_run, r10b_run, r13f3_run, any other) is refused unless the allow json names it with its link text and a reason why no
# write can reach through it. Only then copies the S tools from the staging dir (flat: the nine research/f2/*.py tools, s_run_sbx.sh and
# s_pins.txt) as new files into the clone's research/f2/ (s_run_sbx.sh into research/f2/mac/), each checked against the staged
# s_pins.txt, and records research/f2/out/s_clone_<family>.json and s_linkscan_<family>_{pre,post}.json. Reads the source only.
#   bash s_clone.sh <598c|l4> <staging dir> [<allow json>]
set -u
FAM=${1:?family 598c or l4}; STG=${2:?staging dir}; ALLOW=${3:-}
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
case $FAM in 598c) SRC=$P/r17a_sbx; DST=$P/f2s_sbx598 ;; l4) SRC=$P/r17a_l4g_sbx; DST=$P/f2s_sbxl4 ;; *) echo "family 598c or l4"; exit 2 ;; esac
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "$(ts) s_clone $FAM FAILED: $1"; exit 1; }
scan() { if [ -n "$ALLOW" ]; then "$PY" -X utf8 "$STG/s_linkscan.py" "$@" --allow "$ALLOW"; else "$PY" -X utf8 "$STG/s_linkscan.py" "$@"; fi; }
TOOLS="s_spec.py avg_s.py s_check.py s_gate.py s_learn.py s_chain.py rm_smoke.py fj_blend_s.py s_linkscan.py s_run_sbx.sh"
[ -d "$SRC" ] || fail "no source $SRC"
[ -e "$DST" ] && fail "$DST exists"
[ -z "$ALLOW" ] || [ -s "$ALLOW" ] || fail "no allow file $ALLOW"
n=$(lsof -a -d cwd -Fn 2> /dev/null | grep -cE "^n$SRC(/|\$)")
[ "$n" = 0 ] || fail "$n processes have their working directory in $SRC"
free=$(df -g "$P" | tail -1 | awk '{print $4}')
[ "${free:-0}" -ge 15 ] || fail "only ${free} GB free on the volume of $P"
[ -s "$STG/s_pins.txt" ] || fail "staging lacks s_pins.txt"
for f in $TOOLS; do
  h=$(awk -v f="$f" '$2 == f {print $1}' "$STG/s_pins.txt")
  [ -n "$h" ] || fail "s_pins.txt has no line for $f"
  [ "$(shasum -a 256 "$STG/$f" | cut -c1-16)" = "$h" ] || fail "staged $f is not $h"
done
scan "$SRC" "$STG/s_linkscan_${FAM}_pre.json" --as "$DST" || fail "links of $SRC would reach a live root from the clone ($STG/s_linkscan_${FAM}_pre.json)"
t0=$(date +%s)
echo "$(ts) cloning $SRC -> $DST (cp -c -R)"
cp -c -R "$SRC" "$DST" || fail "clone"
scan "$DST" "$STG/s_linkscan_${FAM}_post.json" || fail "the clone's links reach a live root ($STG/s_linkscan_${FAM}_post.json); nothing written into $DST, no tools in it"
mkdir -p "$DST/research/f2/out" "$DST/research/f2/mac" "$DST/logs"
for f in $TOOLS s_pins.txt; do
  d="$DST/research/f2/$f"; [ "$f" = s_run_sbx.sh ] && d="$DST/research/f2/mac/$f"
  [ -e "$d" ] && fail "the clone already has $d (not overwriting)"
  cp -p "$STG/$f" "$d" || fail "copy $f"
done
for k in pre post; do cp "$STG/s_linkscan_${FAM}_$k.json" "$DST/research/f2/out/s_linkscan_${FAM}_$k.json" || fail "copy link scan $k"; done
pins=$(awk '{printf "%s\"%s\": \"%s\"", (NR>1?", ":""), $2, $1}' "$STG/s_pins.txt")
lk=$("$PY" -c 'import json, sys; r = json.load(open(sys.argv[1])); print(json.dumps({"links": r["links"], "inside": r["inside"], "live_allowed": len(r["live"]), "outside": len(r["outside"]), "allow_sha16": r["allow_sha16"]}))' "$STG/s_linkscan_${FAM}_post.json")
printf '{"family": "%s", "source": "%s", "clone": "%s", "made": "%s", "seconds": %s, "free_gb_before": %s, "link_scan": %s, "tools": {%s}}\n' \
  "$FAM" "$SRC" "$DST" "$(date '+%Y-%m-%d %H:%M:%S')" "$(( $(date +%s) - t0 ))" "$free" "$lk" "$pins" > "$DST/research/f2/out/s_clone_$FAM.json"
echo "$(ts) clone made in $(( $(date +%s) - t0 )) s; links $lk; tools copied and checked; $DST/research/f2/out/s_clone_$FAM.json"
