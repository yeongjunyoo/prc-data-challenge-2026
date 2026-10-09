#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 final root: ctx_v1 (frozen stand_p vocabulary) -> ctx_v2 -> ctx_v3 with the unchanged src/feats_ctx2.py and src/feats_aiu.py, then the
# comparison of every table with today's (2025 rows must be bit-identical). Each step under research/f3/mac_step.sh (lock, nice, time).
#   cd <final root> && PRC_DAILY_CACHE=<daily root>/cache nohup bash research/f3/run_ctx_final.sh >> logs/run_ctx_final.log 2>&1 < /dev/null &
set -u
cd "${PRC_FINAL_ROOT:-.}" || exit 1
PY=${PY:-python}
OLD=${PRC_DAILY_CACHE:-../daily/cache}  # cache/ of the daily build (its ctx tables fix the stand codes)
ST="bash research/f3/mac_step.sh"
s16() { shasum -a 256 "$1" | cut -c1-16; }
[ "$(s16 $OLD/ctx_v1.parquet)" = 0383b7a4b36c5821 ] && [ "$(s16 $OLD/ctx_v2.parquet)" = ac01294f15b8a618 ] && [ "$(s16 $OLD/ctx_v3.parquet)" = 0b139ddbd6e70573 ] || { echo "old ctx sha mismatch"; exit 2; }
$ST ctx1 6 1 logs/ctx_v1.log $PY -X utf8 research/f3/feats_ctx_final.py --name ctx_v1 --months 1,7 --vocab_from $OLD/ctx_v1.parquet --vocab_sha 0383b7a4b36c5821 || exit 3
S1=$(s16 cache/ctx_v1.parquet)
$ST ctx2 6 1 logs/ctx_v2.log $PY -X utf8 src/feats_ctx2.py --name ctx_v2 --base ctx_v1 --base_sha $S1 || exit 4
S2=$(s16 cache/ctx_v2.parquet)
$ST ctx3 4 1 logs/ctx_v3.log $PY -X utf8 src/feats_aiu.py --name ctx_v3 --base ctx_v2 --base_sha $S2 || exit 5
for v in 1 2 3; do
  $ST cmpctx$v 4 1 logs/cmp_ctx_v$v.log $PY -X utf8 research/f3/cmp_tab.py cache/ctx_v$v.parquet $OLD/ctx_v$v.parquet research/f3/out/cmp_ctx_v$v.json --year_col part
done
echo "$(date '+%m-%d %H:%M:%S') ctx done: v1 $S1 v2 $S2 v3 $(s16 cache/ctx_v3.parquet)"
