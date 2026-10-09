#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 (10-08 22:4x): make the final-rebuild root $PRC_WORK/final_run on the Mac. Code from the daily build root $PRC_WORK/r16b_sbx (its 47 files of
# 514's code_sha match, checked 10-08 22:4x), data as APFS clones of the unchanged training files, METAR and AIU, data/ranking.parquet =
# final_ranking.parquet, data/submitting.parquet = final_submitting.parquet. cache/ preds/ notes/ queue/ subs/ are empty and private.
# Corrector fit caches = the union of the Mac sandboxes' copies (content-addressed names). Refuses if final_run exists.
#   bash research/f3/mk_final_root_mac.sh
set -u
P=${PRC_WORK:-.}
S=$P/r16b_sbx
D=$P/final_run
[ -e "$D" ] && { echo "$D exists"; exit 1; }
mkdir -p "$D"/{data/ext,cache,preds/code,notes,queue/done,queue/hold,logs,subs,research} || exit 1
cp -cRp "$S/src" "$D/src" && cp -cRp "$S/tools" "$D/tools" || exit 2
find "$D/src" "$D/tools" -name "__pycache__" -type d -prune -exec rm -rf {} +
for r in r13_l r13_f3 r10_b obs2 adsb ctx_corr r16_b; do
  mkdir -p "$D/research/$r"
  # scripts and small fixed inputs only; big lane outputs stay out (fit caches are added below)
  (cd "$S/research/$r" && find . -maxdepth 1 -type f \( -name "*.py" -o -name "*.sh" -o -name "*.txt" -o -name "*.tsv" -o -name "*.json" \) -exec cp -cp {} "$D/research/$r/" \;)
done
# OSM stand map inputs of the geo build (frozen) and the gprobe nd builder
for sub in obs2/osm adsb/gprobe; do [ -d "$S/research/$sub" ] && cp -cRp "$S/research/$sub" "$D/research/$sub"; done
for f in stand_map.parquet stand_map_filled.parquet; do [ -e "$S/research/obs2/$f" ] && cp -cp "$S/research/obs2/$f" "$D/research/obs2/"; done
# corrector fit caches (union)
for d in r13_l r10_b r13_f3 ctx_corr; do
  mkdir -p "$D/research/$d/fitcache"
  for s in r16b_sbx r13l_sbx r10b_sbx r13f3_sbx; do
    [ -d "$P/$s/research/$d/fitcache" ] || continue
    (cd "$P/$s/research/$d/fitcache" && for f in *.txt; do [ -e "$D/research/$d/fitcache/$f" ] || cp -cp "$f" "$D/research/$d/fitcache/$f"; done)
  done
  echo "fitcache $d: $(ls "$D/research/$d/fitcache" | wc -l) files"
done
# data
for f in "$S"/data/training_*.parquet; do cp -cp "$f" "$D/data/"; done
cp -cRp "$S/data/metar" "$D/data/metar"
cp -cRp "$S/data/ext/aiu" "$D/data/ext/aiu"
cp -cp "$P/f3_sbx/data/final_phase/final_ranking.parquet" "$D/data/ranking.parquet"
cp -cp "$P/f3_sbx/data/final_phase/final_submitting.parquet" "$D/data/submitting.parquet"
cp -p "$S/GATE.json" "$D/GATE.json"
echo "root made $(date '+%m-%d %H:%M:%S'): $(du -sh "$D" | cut -f1)"
shasum -a 256 "$D/data/ranking.parquet" "$D/data/submitting.parquet" | cut -c1-16
ls "$D" "$D/data" "$D/research"
