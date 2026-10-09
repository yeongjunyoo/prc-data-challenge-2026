#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 interim root ${PRC_WORK:-.}/f3_sbx/interim (fitwrap path, 2026-10-08 23:5x KST): APFS clones of the final root's code
# and tables, plus an interim feats_v27 whose 2025 rows are the final ones (the final feats_v25 + today's adsb_v2 rows of 2025 days) and
# whose 2026 rows have no ADS-B day data yet. The leg scripts run here in fitwrap save mode while R17-A's Feb/Jun package is pending, so
# their fits are stored; their outputs here are never used. Checks: check_v27.py criteria 1-3 (schema and rows, 2025 rows equal today's
# except the seven all-row columns, non-ab_ columns equal feats_v25); 4-5 fail by construction (no 2026 day data).
#   bash research/f3/mk_interim.sh   (from ${PRC_WORK:-.}/final_run)
set -u
P=${PRC_WORK:-.}
FR=$P/final_run
IR=$P/f3_sbx/interim
PY=${PY_MAC:-python3}
mkdir -p $IR && cd $IR || exit 1
for d in src research data preds; do [ -e $d ] || cp -c -R $FR/$d $d; done
rm -f research/f3/out/orch_*.json
mkdir -p cache logs notes queue subs
for f in $FR/cache/*; do b=$(basename $f); [ -e cache/$b ] || cp -c $f cache/$b; done
[ -e cache/feats_v27.parquet ] && { echo "interim feats_v27 exists"; exit 0; }
$PY -X utf8 - <<'EOF'
import hashlib
import pyarrow.compute as pc
import pyarrow.parquet as pq
src = "<mac-work>/r16b_sbx/cache/adsb_v2.parquet"
h = hashlib.sha256(open(src, "rb").read()).hexdigest()[:16]
assert h == "8ee80c46165c4ef9", h
t = pq.read_table(src)
k = t.filter(pc.equal(pc.utf8_slice_codeunits(t.column("day"), 0, 4), "2025"))
pq.write_table(k, "cache/adsb_v2i.parquet")
print("adsb_v2i rows", k.num_rows, "of", t.num_rows)
EOF
$PY -X utf8 -m src.feats_adsb feats_v27 --adsb adsb_v2i --mode raw > logs/feats_v27_interim.log 2>&1 || { echo "interim feats_v27 failed"; exit 2; }
grep -E '"rows|sha16|seconds' logs/feats_v27_interim.log
$PY -X utf8 research/f3/check_v27.py $P/r16b_sbx/cache/feats_v27.parquet > logs/check_v27_interim.log 2>&1
echo "check_v27 on the interim table rc $? (criteria 4-5 fail by construction)"
grep -v "^ " logs/check_v27_interim.log | head -40
