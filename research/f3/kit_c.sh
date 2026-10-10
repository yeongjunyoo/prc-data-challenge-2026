#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 final round c (coordinator 10-10 08:2x: F2's lever 2, the matched Feb/Jun 2025 extras, adopted): 594c = 594b's form (Bb on the 514-config
# legs) and 598c = 598b's form (Bb on L3's legs), with the four-month corrector run on F2's tables adsb_v1ndfjF and geo_v1ndfjF (this root's
# adsb_v1nd and geo_v1nd, every row unchanged, plus the 326,873 Feb/Jun 2025 rows; research/f3/check_fjF.py). Everything else is the earlier
# kit's, with the same tools and arguments: research/f2/send_final/kit_final.sh (F2, 594b) for 594c, research/f3/kit_l3.sh (598b) for 598c.
# Differences from those kits, all of them names or the two tables:
#   the four-month corrector's tags end in F (<T>adsbgeondctxv3x50ws75m4F at scale2 0.75, <T>adsbgeondctxv3x50wm4F at 0.5) and it reads
#   --adsb adsb_v1ndfjF --geo geo_v1ndfjF; the merged variant is BbF (jj T 0.75 | fu L 0.75 | fm L 0.5; the merge groups are adsb_v1nd's
#   matched, unchanged); step and log names <family>_*, the twins, check and tails-state files under research/f3/out/ (the earlier kits' files are
#   left as they are); sandbox queue numbers num0 + 6000 (594c) or 13000 (598c) + step index, chainavg3 2400 + the same.
# Threads: the corrector as each family's earlier build (594c 2, 598c 4: the stage-1 keys of the seeded fit caches hold the thread count), tails 3.
#   bash research/f3/kit_c.sh <594c|598c> corr4m <X> | merge <X> | manifest <X> | tails <X> | avg3 | check [finals]      (X: A B C)
set -eu
FAM=${1:-}
shift || true
P=~/nova-adsb/prcm
R=$P/final_run
PY=$HOME/nova-adsb/venv311/bin/python
V=BbF
ADSB_SHA=07df03ca69833082; GEO_SHA=9dfa71b64e1099de; CTX_SHA=01ac984c7afd6e05   # final_run tables (as the earlier kits)
ADSBF=adsb_v1ndfjF; ADSBF_SHA=e13437a2c55a77c5; GEOF=geo_v1ndfjF; GEOF_SHA=4fd677dc9060ce22   # F2's lever-2 tables (f2_run, 10-10 06:53)
CTX4=f2ctx4f_v3
TAILTH=3
case $FAM in
594c)
  cb() { case $1 in A) echo m6r27wincbrmfw4sv434tmc;; B) echo m6r27skcbrmfw4sv434;; C) echo m6r27apcbrmfw4sv114;; *) echo "chain $1?" >&2; exit 9;; esac; }
  ex() { case $1 in A) echo f2wincbrmfw4;; B) echo f2skcbrmfw4;; C) echo f2apcbrmfw4;; esac; }
  exsha() { case $1 in A) echo bbdfbccf5821da14;; B) echo cd2cf1ea95631aad;; C) echo c0901a4fd4121cc7;; esac; }
  gleg() { case $1 in A) echo m6r27skwing;; B) echo m6r27skg;; C) echo m6r27skapg;; esac; }
  TH=2; FC=research/f2/fitcache_4m; FEATS=feats_v27; DNUM=6000
  E0="src/blend_fixed.py,src/ctx_corr2.py,src/ctx_corr3.py,src/feats_ctx2.py,src/feats_aiu.py,research/ctx_corr/freeze_geo.py,research/obs2/corr_plus.py,research/obs2/obs_rows.py,research/obs2/coverage.py,research/obs2/osm_stands.py,research/adsb/gprobe/build_nd.py,research/r10_b/cavg.py,research/f3/fitwrap.py"
  extra() { case $1 in A) echo "$E0,research/r13_l/prep_x.py,research/r13_l/tabm_leg.py,research/r13_l/xrfm_leg.py,research/r13_l/blend_add.py";; B) echo "$E0";;
                       C) echo "$E0,research/r13_f3/model3ap.py,research/r13_f3/cat_muap.py";; esac; }
  KITCODE="research/f2/merge_corr.py,research/f2/ctx_corr4.py,research/f2/ctx4_table.py,research/f3/kit_c.sh" ;;
598c)
  cb() { case $1 in A) echo m6r28wincbrmfw4sv434tmc;; B) echo m6r28skcbrmfw4sv434;; C) echo m6r28apcbrmfw4sv114;; *) echo "chain $1?" >&2; exit 9;; esac; }
  ex() { case $1 in A) echo l3wincbrmfw4;; B) echo l3skcbrmfw4;; C) echo l3apcbrmfw4;; esac; }
  exsha() { grep " $(ex $1)\$" "$R/research/f3/out/l3_fj_sha16.txt" | cut -c1-16; }
  gleg() { case $1 in A) echo m6r28skwing;; B) echo m6r28skg;; C) echo m6r28skapg;; esac; }
  TH=4; FC=research/f2/fitcache_4m_l3; FEATS=feats_v28; DNUM=13000
  E0="src/blend_fixed.py,src/ctx_corr2.py,src/ctx_corr3.py,src/feats_ctx2.py,src/feats_aiu.py,research/ctx_corr/freeze_geo.py,research/obs2/corr_plus.py,research/obs2/obs_rows.py,research/obs2/coverage.py,research/obs2/osm_stands.py,research/adsb/gprobe/build_nd.py,research/r10_b/cavg.py,research/f3/fitwrap.py,research/r13_l/prep_x.py,research/r13_l/tabm_leg.py,research/r13_l/xrfm_leg.py,research/r13_l/blend_add.py,research/r17_a/avg_r17a.py,research/r17_a/adsb_day_r17a.py,research/adsb/build_feats2.py,research/adsb/adsb_match.py,research/f3/avg_legs_final.py,research/f3/feats_v28_final.py"
  extra() { case $1 in C) echo "$E0,research/r13_f3/model3ap.py,research/r13_f3/cat_muap.py";; *) echo "$E0";; esac; }
  KITCODE="research/f2/merge_corr.py,research/f2/ctx_corr4.py,research/f2/ctx4_table.py,research/f2/model3_fj.py,research/f2/model3ap_fj.py,research/f2/g_band_fj.py,research/f2/blend_fixed_fj.py,research/f3/fj_blends_l3.py,research/f3/kit_l3.sh,research/f3/kit_c.sh" ;;
*) sed -n '2,15p' "$0"; exit 1 ;;
esac
num0() { case $1 in A) echo 2000;; B) echo 2100;; C) echo 2200;; esac; }
TAILS="am fh gkc gkcx xncr db aw gp gl gja"
suf() { case $1 in am) echo am;; fh) echo amfh;; gkc) echo amfhgkc;; gkcx) echo amfhgkcx;; xncr) echo amfhgkcxncr;; db) echo amfhgkcxncrdb;;
                   aw) echo amfhgkcxncrdbaw;; gp) echo amfhgkcxncrdbawgp;; gl) echo amfhgkcxncrdbawgpgl;; gja) echo amfhgkcxncrdbawgpglgja;; esac; }
idx() { case $1 in am) echo 5;; fh) echo 6;; gkc) echo 7;; gkcx) echo 8;; xncr) echo 9;; db) echo 10;; aw) echo 11;; gp) echo 12;; gl) echo 13;; gja) echo 14;; esac; }
L75() { echo "$(cb $1)adsbgeondctxv3x50ws75m4F"; }
L50() { echo "$(cb $1)adsbgeondctxv3x50wm4F"; }
ts() { date '+%m-%d %H:%M:%S'; }
new() { for f in "$@"; do if [ -e "$R/$f" ]; then echo "exists, refusing: $f"; exit 3; fi; done; }
step() {  # name peak_gb threads python-args...
  local name=$1 gb=$2 th=$3; shift 3
  if ! (cd "$R" && bash research/f3/mac_step.sh "${FAM}_$name" "$gb" "$th" "logs/${FAM}_$name.log" "$PY" -X utf8 -u "$@"); then
    echo "$(ts) step $name failed (final_run/logs/${FAM}_$name.log)"; exit 4
  fi
  echo "$(ts) $name: $(grep -vE '^[[:space:]]+[0-9]' "$R/logs/${FAM}_$name.log" | grep -v '^$' | tail -1 | cut -c1-220)"
}
mergeargs() {  # chain setting
  local T D; T=$(cb $1); D=${T}adsbgeondctxv3x50ws75c3
  echo "research/f2/merge_corr.py --setting $2 --seed 0 --tag ${D}r$V --base $T --base_seed 0 --default $D --jj $D --fu $(L75 $1)c3 --fm $(L50 $1)c3 --ctx ctx_v3 --ctx_sha $CTX_SHA --adsb adsb_v1nd --adsb_sha $ADSB_SHA --jj_ids research/f2/janjul_ids.parquet"
}
qentry() {  # number spec: a sandbox queue entry as research/f3/final_chain.py's qentry (the next tail step reads it)
  local q="$R/queue/${1}_$2" tp
  if [ -e "$q.json" ]; then echo "queue entry exists: $q.json" >&2; exit 3; fi
  printf '{\n "manifest": "preds/manifest_%s.json",\n "kind": "CANDIDATE",\n "sandbox": true\n}' "$2" > "$q.json"
  tp=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['test']['path'])" "$R/preds/manifest_$2.json")
  cp -p "$R/$tp" "$q.parquet"
  echo "${1}_$2"
}

case "${1:-}" in
corr4m)  # the four-month corrector on F2's tables, full setting, seeds 0-2: at 0.75 (10 fits: the S_k with the extra rows; stage 1 loads), then the 0.5 twin (0 fits)
  X=$2; T=$(cb $X); C4=$(cat "$R/research/f2/out/$CTX4.sha16"); XS=$(exsha $X)
  [ -n "$XS" ] || { echo "no sha16 for $(ex $X)"; exit 2; }
  ARGS="--feats feats_v25 --adsb $ADSBF --adsb_sha $ADSBF_SHA --folds 10 --rounds 300 --rounds2 400 --max_pred 3600 --threads $TH --geo $GEOF --geo_sha $GEOF_SHA --ctx $CTX4 --ctx_sha $C4 --extra_base $(ex $X) --extra_sha $XS --fit_cache $FC"
  for sc in 0.75 0.5; do
    tg=$(L75 $X); if [ $sc = 0.5 ]; then tg=$(L50 $X); fi
    for k in 0 1 2; do
      if [ -e "$R/cache/full_${tg}_s$k.parquet" ]; then echo "$(ts) skip full_${tg}_s$k (exists)"; continue; fi
      step "c4m_${X}_${sc}_s$k" 6 $TH research/f2/ctx_corr4.py --setting full --seed $k --base $T --base_seed 0 --tag $tg $ARGS --scale2 $sc
      want=10; if [ $sc = 0.5 ]; then want=0; fi
      got=$(grep -o '[0-9]* fitted' "$R/logs/${FAM}_c4m_${X}_${sc}_s$k.log" | tail -1 | grep -o '^[0-9]*')
      if [ "${got:-x}" != $want ]; then echo "$(ts) c4m_${X}_${sc}_s$k fitted ${got:-?} models, expected $want: stop and tell the coordinator"; exit 5; fi
    done
    if [ ! -e "$R/cache/full_${tg}c3_s0.parquet" ]; then
      step "c4m_${X}_${sc}_cavg" 3 1 research/r10_b/cavg.py --setting full --seed 0 --tag ${tg}c3 --src $tg --seeds 0,1,2 --base $T --base_seed 0
    fi
  done
  step "twins_$X" 4 1 research/f2/check_split.py --twins "$(L75 $X):$(L50 $X)" --out "research/f3/out/twins_${FAM}_$X.json" ;;
merge)  # the corrector-stage merge of chain X (val and fwd: byte copies of the default; full: pred by group) and its jobs file
  X=$2; T=$(cb $X); D=${T}adsbgeondctxv3x50ws75c3
  for s in val fwd full; do
    if [ ! -e "$R/cache/${s}_${D}_s0.parquet" ]; then echo "missing cache/${s}_${D}_s0.parquet"; exit 2; fi
  done
  if [ ! -e "$R/cache/full_$(L75 $X)c3_s0.parquet" ] || [ ! -e "$R/cache/full_$(L50 $X)c3_s0.parquet" ]; then echo "$(ts) corr4m $X not done"; exit 2; fi
  if [ -e "$R/notes/jobs_${D}r$V.txt" ]; then echo "$(ts) skip merge $X $V (jobs file exists)"; exit 0; fi
  lines="# F3 round-c kit (research/f3/kit_c.sh, $FAM), variant $V (corrector-stage merge after cavg; val, fwd = byte copies of $D; L on F2's fjF tables)"
  for s in val fwd full; do
    step "merge_${X}${V}_$s" 4 1 $(mergeargs $X $s)
    lines="$lines"$'\n'"logs/${FAM}_merge_${X}${V}_$s.log"$'\t'".venv/Scripts/python.exe -u $(mergeargs $X $s)"
  done
  printf '%s\n' "$lines" > "$R/notes/jobs_${D}r$V.txt" ;;
manifest)  # the merged tag's manifest (src.manifest reads the three merge commands from its jobs file; the tails read --base from them)
  X=$2; tg=$(cb $X)adsbgeondctxv3x50ws75c3r$V
  new "preds/manifest_$tg.json"
  export MANIFEST_EXTRA_CODE="$(extra $X),$KITCODE"
  step "man_${X}$V" 3 1 -m src.manifest $tg $tg pred 0 $FEATS notes/jobs_$tg.txt ;;
tails)  # artmm .. gjoint a on the merged tag with research/f3/final_chain.py's arguments
  X=$2; TCS=$(cb $X)adsbgeondctxv3x50ws75c3r$V; st="$R/research/f3/out/kit_c_tails_${FAM}_$X.txt"; touch "$st"
  prevq=""; prev=$TCS
  for t in $TAILS; do
    spec=$TCS$(suf $t); n=$(( $(num0 $X) + DNUM + $(idx $t) ))
    if grep -q "^$t " "$st"; then prevq=$(grep "^$t " "$st" | cut -d' ' -f2); prev=$spec; echo "$(ts) skip $t (done: $prevq)"; continue; fi
    new "preds/manifest_$spec.json"
    srcm=preds/manifest_$prev.json
    case $t in
      am) set -- tools/artmm.py build $srcm $n --no-enqueue ;;
      fh) set -- tools/arthist.py build $srcm $n --no-enqueue ;;
      gkc) set -- tools/gkey.py build $prevq $n --no-enqueue --variant avg --leg $(gleg $X) --leg_feats $FEATS ;;
      gkcx) set -- tools/gkeyx.py build $srcm $n --no-enqueue ;;
      xncr) set -- tools/gkeyxn.py build $srcm $n --no-enqueue --variant avg_xncr ;;
      db) set -- src/aiu_shift.py build $prevq $n --group block ;;
      aw) set -- tools/antiwin.py build $prevq $n --no-enqueue ;;
      gp) set -- tools/gpool.py build $prevq $n --no-enqueue ;;
      gl) set -- tools/glong.py build $srcm $n --no-enqueue ;;
      gja) set -- tools/gjoint.py build $srcm $n --mode a --no-enqueue ;;
    esac
    step "${t}_${X}$V" 4 $TAILTH "$@"
    if [ ! -e "$R/preds/manifest_$spec.json" ]; then echo "$t: no manifest $spec"; exit 4; fi
    prevq=$(qentry $n $spec); echo "$t $prevq" >> "$st"; prev=$spec
  done ;;
avg3)  # chainavg3 of the three chains' gjoint a entries
  qa=$(grep '^gja ' "$R/research/f3/out/kit_c_tails_${FAM}_A.txt" | cut -d' ' -f2); qb=$(grep '^gja ' "$R/research/f3/out/kit_c_tails_${FAM}_B.txt" | cut -d' ' -f2)
  qc=$(grep '^gja ' "$R/research/f3/out/kit_c_tails_${FAM}_C.txt" | cut -d' ' -f2)
  step "avg3_$V" 3 1 tools/chainavg3.py build "$qa" "$qb" "$qc" $(( 2400 + DNUM )) --suffix skap --no-enqueue ;;
check)  # twins and corrector stage (BbF and the null merge D); with "finals" also the BbF final against the family's default final (592 or 596)
  m=""; tw=""
  for X in A B C; do
    for W in $V D; do tg=$(cb $X)adsbgeondctxv3x50ws75c3r$W; if [ -e "$R/notes/merge_${tg}_full.json" ]; then m="$m,$tg"; fi; done
    if [ -e "$R/cache/full_$(L75 $X)c3_s0.parquet" ] && [ -e "$R/cache/full_$(L50 $X)c3_s0.parquet" ]; then tw="$tw,$(L75 $X):$(L50 $X)"; fi
  done
  if [ "${2:-}" = finals ]; then
    S=$(cb A)adsbgeondctxv3x50ws75c3
    step check_finals 6 1 research/f2/check_split.py --merged "${m#,}" --twins "${tw#,}" --ctx ctx_v3 --ctx_sha $CTX_SHA \
      --final_default "preds/manifest_${S}amfhgkcxncrdbawgpglgjaavskap.json" --finals "$V=preds/manifest_${S}r${V}amfhgkcxncrdbawgpglgjaavskap.json" \
      --out research/f3/out/split_check_$FAM.json
  else
    step check_corr 6 1 research/f2/check_split.py --merged "${m#,}" --twins "${tw#,}" --ctx ctx_v3 --ctx_sha $CTX_SHA --out research/f3/out/split_check_${FAM}_corr.json
  fi ;;
*) sed -n '2,15p' "$0"; exit 1 ;;
esac
