#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F2 Package S: put the S tools into a Mac root as new files (never overwriting, each checked against the staged s_pins.txt). Used for
# f2_run (598c's fj CatBoost ap and RealMLP seeds, 10-11 02:2x) and, if F3 wants them there, final_run. The sandbox clones get
# their tools from s_clone.sh. Reads the staging dir, writes only new files under <root>/research/f2/ and research/f2/out/s_deploy.json.
#   bash s_deploy.sh <root dir> <staging dir> <tool> [<tool> ...]
#   e.g. bash s_deploy.sh ${PRC_WORK:-.}/f2_run <stage> s_spec.py avg_s.py s_check.py s_learn.py rm_smoke.py s_fjpre.py
set -u
R=${1:?root}; STG=${2:?staging dir}; shift 2
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "$(ts) s_deploy FAILED: $1"; exit 1; }
[ -d "$R/research/f2" ] || fail "no $R/research/f2"
[ -s "$STG/s_pins.txt" ] || fail "staging lacks s_pins.txt"
for f in "$@"; do
  h=$(awk -v f="$f" '$2 == f {print $1}' "$STG/s_pins.txt")
  [ -n "$h" ] || fail "s_pins.txt has no line for $f"
  [ "$(shasum -a 256 "$STG/$f" | cut -c1-16)" = "$h" ] || fail "staged $f is not $h"
  if [ -e "$R/research/f2/$f" ]; then
    [ "$(shasum -a 256 "$R/research/f2/$f" | cut -c1-16)" = "$h" ] && { echo "$(ts) $f already there with $h"; continue; }
    fail "$R/research/f2/$f exists with another sha16 (not overwriting)"
  fi
done
mkdir -p "$R/research/f2/out"
for f in "$@"; do
  [ -e "$R/research/f2/$f" ] || cp -p "$STG/$f" "$R/research/f2/$f" || fail "copy $f"
done
printf '{"root": "%s", "deployed": "%s", "tools": "%s"}\n' "$R" "$(date '+%Y-%m-%d %H:%M:%S')" "$*" > "$R/research/f2/out/s_deploy.json"
echo "$(ts) deployed to $R/research/f2: $*"
