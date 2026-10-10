#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 L3 final kit (coordinator 10-10 03:5x and 04:2x; 598a = 594a's form on L3's legs). F2's research/f2/send_final/kit_final.sh (sha16
# 8069a08c39342cb4) with L3's names; the tools it calls are unchanged (research/f2/ctx_corr4.py, merge_corr.py, check_split.py, research/r10_b/cavg.py,
# src.manifest, the tail tools, tools/chainavg3.py). Differences, all of them names or inputs:
#   chain bases m6r28wincbrmfw4sv434tmc, m6r28skcbrmfw4sv434, m6r28apcbrmfw4sv114 (R17-A's L3, built here by research/f3/final_chain.py --cfg l3s75);
#   fj extras l3wincbrmfw4, l3skcbrmfw4, l3apcbrmfw4 (research/f3/fj_blends_l3.py; pins in research/f3/out/l3_fj_sha16.txt);
#   the four-month corrector at KIT_TH 4 with its own fit cache research/f2/fitcache_4m_l3, seeded from L3's corrector models
#   (research/f3/fitcache_l3: ctx_corr3 at --threads 4, so stage 1 loads; the S_k with the extra rows are new: 10 fitted per 0.75 run, 0 per twin);
#   feats_v28 for the manifests and the gkey leg; G legs m6r28skwing, m6r28skg, m6r28skapg; code list = final_chain.py's EXTRA_L3 + the kit's;
#   step and log names l3k_*, state and check files under research/f3/out/ (594's under research/f2/out/ are left as they are);
#   sandbox queue numbers num0 + 8000 (D) / 9000 (Ba) / 10000 (Bb) + step index; variants Bb (598b, the final form since the coordinator's
#   decision 10-10 04:44: T 0.75 | L 0.75 | L 0.5), Ba (598a, the comparison: T 0.75 | L 0.75 | L 0.75) and D (null merge, check iv).
# Merges run only from research/f3/phase5b.sh (KIT_BY=phase5b): phase5.sh's own 598a part (Ba first) stops at its first merge, so 598b comes first.
# Row groups as F2's: jj = research/f2/janjul_ids.parquet, fu / fm = Feb/Jun without / with an ADS-B match (adsb_v1nd matched).
#   bash research/f3/kit_l3.sh corr4m <X> | merge <X> | manifest <X> <V> | tails <X> <V> | avg3 <V> | check [finals]   (X: A B C; V: Bb Ba D)
set -eu
P=~/nova-adsb/prcm
R=$P/final_run
PY=$HOME/nova-adsb/venv311/bin/python
TH=${KIT_TH:-4}
ADSB_SHA=07df03ca69833082; GEO_SHA=9dfa71b64e1099de; CTX_SHA=01ac984c7afd6e05   # final_run tables (as F2's kit)
CTX4=f2ctx4f_v3
cb() { case $1 in A) echo m6r28wincbrmfw4sv434tmc;; B) echo m6r28skcbrmfw4sv434;; C) echo m6r28apcbrmfw4sv114;; *) echo "chain $1?" >&2; exit 9;; esac; }
ex() { case $1 in A) echo l3wincbrmfw4;; B) echo l3skcbrmfw4;; C) echo l3apcbrmfw4;; esac; }
exsha() { grep " $(ex $1)\$" "$R/research/f3/out/l3_fj_sha16.txt" | cut -c1-16; }
num0() { case $1 in A) echo 2000;; B) echo 2100;; C) echo 2200;; esac; }
dnum() { case $1 in D) echo 8000;; Ba) echo 9000;; Bb) echo 10000;; *) echo "variant $1?" >&2; exit 9;; esac; }
gleg() { case $1 in A) echo m6r28skwing;; B) echo m6r28skg;; C) echo m6r28skapg;; esac; }
E0="src/blend_fixed.py,src/ctx_corr2.py,src/ctx_corr3.py,src/feats_ctx2.py,src/feats_aiu.py,research/ctx_corr/freeze_geo.py,research/obs2/corr_plus.py,research/obs2/obs_rows.py,research/obs2/coverage.py,research/obs2/osm_stands.py,research/adsb/gprobe/build_nd.py,research/r10_b/cavg.py,research/f3/fitwrap.py,research/r13_l/prep_x.py,research/r13_l/tabm_leg.py,research/r13_l/xrfm_leg.py,research/r13_l/blend_add.py,research/r17_a/avg_r17a.py,research/r17_a/adsb_day_r17a.py,research/adsb/build_feats2.py,research/adsb/adsb_match.py,research/f3/avg_legs_final.py,research/f3/feats_v28_final.py"
extra() { case $1 in C) echo "$E0,research/r13_f3/model3ap.py,research/r13_f3/cat_muap.py";; *) echo "$E0";; esac; }
KITCODE="research/f2/merge_corr.py,research/f2/ctx_corr4.py,research/f2/ctx4_table.py,research/f2/model3_fj.py,research/f2/model3ap_fj.py,research/f2/g_band_fj.py,research/f2/blend_fixed_fj.py,research/f3/fj_blends_l3.py,research/f3/kit_l3.sh"
TAILS="am fh gkc gkcx xncr db aw gp gl gja"
suf() { case $1 in am) echo am;; fh) echo amfh;; gkc) echo amfhgkc;; gkcx) echo amfhgkcx;; xncr) echo amfhgkcxncr;; db) echo amfhgkcxncrdb;;
                   aw) echo amfhgkcxncrdbaw;; gp) echo amfhgkcxncrdbawgp;; gl) echo amfhgkcxncrdbawgpgl;; gja) echo amfhgkcxncrdbawgpglgja;; esac; }
idx() { case $1 in am) echo 5;; fh) echo 6;; gkc) echo 7;; gkcx) echo 8;; xncr) echo 9;; db) echo 10;; aw) echo 11;; gp) echo 12;; gl) echo 13;; gja) echo 14;; esac; }
L75() { echo "$(cb $1)adsbgeondctxv3x50ws75m4"; }
L50() { echo "$(cb $1)adsbgeondctxv3x50wm4"; }
sha() { shasum -a 256 "$1" | cut -c1-16; }
ts() { date '+%m-%d %H:%M:%S'; }
new() { for f in "$@"; do if [ -e "$R/$f" ]; then echo "exists, refusing: $f"; exit 3; fi; done; }
step() {  # name peak_gb threads python-args...
  local name=$1 gb=$2 th=$3; shift 3
  if ! (cd "$R" && bash research/f3/mac_step.sh "l3k_$name" "$gb" "$th" "logs/l3k_$name.log" "$PY" -X utf8 -u "$@"); then
    echo "$(ts) step $name failed (final_run/logs/l3k_$name.log)"; exit 4
  fi
  echo "$(ts) $name: $(grep -vE '^[[:space:]]+[0-9]' "$R/logs/l3k_$name.log" | grep -v '^$' | tail -1 | cut -c1-220)"
}
mergeargs() {  # chain variant setting
  local T D jj fu fm; T=$(cb $1); D=${T}adsbgeondctxv3x50ws75c3; jj=$D; fu=$D; fm=$D
  case $2 in Ba) fu=$(L75 $1)c3; fm=$(L75 $1)c3;; Bb) fu=$(L75 $1)c3; fm=$(L50 $1)c3;; esac
  echo "research/f2/merge_corr.py --setting $3 --seed 0 --tag ${D}r$2 --base $T --base_seed 0 --default $D --jj $jj --fu $fu --fm $fm --ctx ctx_v3 --ctx_sha $CTX_SHA --adsb adsb_v1nd --adsb_sha $ADSB_SHA --jj_ids research/f2/janjul_ids.parquet"
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
corr4m)  # four-month corrector on L3's base, full setting, seeds 0-2: at 0.75 (10 fits: the S_k with the extra rows; stage 1 loads), then the 0.5 twin (0 fits)
  X=$2; T=$(cb $X); C4=$(cat "$R/research/f2/out/$CTX4.sha16"); XS=$(exsha $X)
  [ -n "$XS" ] || { echo "no pin for $(ex $X) in research/f3/out/l3_fj_sha16.txt"; exit 2; }
  ARGS="--feats feats_v25 --adsb adsb_v1nd --adsb_sha $ADSB_SHA --folds 10 --rounds 300 --rounds2 400 --max_pred 3600 --threads $TH --geo geo_v1nd --geo_sha $GEO_SHA --ctx $CTX4 --ctx_sha $C4 --extra_base $(ex $X) --extra_sha $XS --fit_cache research/f2/fitcache_4m_l3"
  for sc in 0.75 0.5; do
    tg=$(L75 $X); if [ $sc = 0.5 ]; then tg=$(L50 $X); fi
    for k in 0 1 2; do
      if [ -e "$R/cache/full_${tg}_s$k.parquet" ]; then echo "$(ts) skip full_${tg}_s$k (exists)"; continue; fi
      step "c4m_${X}_${sc}_s$k" 6 $TH research/f2/ctx_corr4.py --setting full --seed $k --base $T --base_seed 0 --tag $tg $ARGS --scale2 $sc
      want=10; if [ $sc = 0.5 ]; then want=0; fi
      got=$(grep -o '[0-9]* fitted' "$R/logs/l3k_c4m_${X}_${sc}_s$k.log" | tail -1 | grep -o '^[0-9]*')
      if [ "${got:-x}" != $want ]; then echo "$(ts) c4m_${X}_${sc}_s$k fitted ${got:-?} models, expected $want: stop and tell the coordinator"; exit 5; fi
    done
    if [ ! -e "$R/cache/full_${tg}c3_s0.parquet" ]; then
      step "c4m_${X}_${sc}_cavg" 3 1 research/r10_b/cavg.py --setting full --seed 0 --tag ${tg}c3 --src $tg --seeds 0,1,2 --base $T --base_seed 0
    fi
  done
  step "twins_$X" 4 1 research/f2/check_split.py --twins "$(L75 $X):$(L50 $X)" --out "research/f3/out/twins_l3_$X.json" ;;
merge)  # corrector-stage merges of chain X (val and fwd: byte copies of the default; full: pred by group) and their jobs files
  X=$2; T=$(cb $X); D=${T}adsbgeondctxv3x50ws75c3
  if [ "${KIT_BY:-}" != phase5b ]; then
    echo "$(ts) merge $X refused: merges handed to research/f3/phase5b.sh (coordinator 10-10 04:44: the final form is Bb, so 598b goes first, then 598a); handed to phase5b"
    exit 7
  fi
  for s in val fwd full; do
    if [ ! -e "$R/cache/${s}_${D}_s0.parquet" ]; then echo "missing cache/${s}_${D}_s0.parquet: run final_chain.py chain $X --cfg l3s75 through cavg first"; exit 2; fi
  done
  for V in ${KIT_VARIANTS:-Ba D}; do
    case $V in Ba|Bb) if [ ! -e "$R/cache/full_$(L75 $X)c3_s0.parquet" ] || [ ! -e "$R/cache/full_$(L50 $X)c3_s0.parquet" ]; then
                        echo "$(ts) $V: corr4m $X not done, skipped"; continue; fi;; esac
    if [ -e "$R/notes/jobs_${D}r$V.txt" ]; then echo "$(ts) skip merge $X $V (jobs file exists)"; continue; fi
    lines="# F3 L3 kit (research/f3/kit_l3.sh, F2's kit with L3 names), variant $V (corrector-stage merge after cavg; val, fwd = byte copies of $D)"
    for s in val fwd full; do
      step "merge_${X}${V}_$s" 4 1 $(mergeargs $X $V $s)
      lines="$lines"$'\n'"logs/l3k_merge_${X}${V}_$s.log"$'\t'".venv/Scripts/python.exe -u $(mergeargs $X $V $s)"
    done
    printf '%s\n' "$lines" > "$R/notes/jobs_${D}r$V.txt"
  done ;;
manifest)  # the merged tag's manifest (src.manifest reads the three merge commands from its jobs file; the tails read --base from them)
  X=$2; V=$3; dnum $V > /dev/null; tg=$(cb $X)adsbgeondctxv3x50ws75c3r$V
  new "preds/manifest_$tg.json"
  export MANIFEST_EXTRA_CODE="$(extra $X),$KITCODE"
  step "man_${X}$V" 3 1 -m src.manifest $tg $tg pred 0 feats_v28 notes/jobs_$tg.txt ;;
tails)  # artmm .. gjoint a on the merged tag with research/f3/final_chain.py's arguments (sandbox queue numbers num0 + dnum + step index)
  X=$2; V=$3; TCS=$(cb $X)adsbgeondctxv3x50ws75c3r$V; st="$R/research/f3/out/kit_l3_tails_${X}$V.txt"; touch "$st"
  prevq=""; prev=$TCS
  for t in $TAILS; do
    spec=$TCS$(suf $t); n=$(( $(num0 $X) + $(dnum $V) + $(idx $t) ))
    if grep -q "^$t " "$st"; then prevq=$(grep "^$t " "$st" | cut -d' ' -f2); prev=$spec; echo "$(ts) skip $t (done: $prevq)"; continue; fi
    new "preds/manifest_$spec.json"
    srcm=preds/manifest_$prev.json
    case $t in
      am) set -- tools/artmm.py build $srcm $n --no-enqueue ;;
      fh) set -- tools/arthist.py build $srcm $n --no-enqueue ;;
      gkc) set -- tools/gkey.py build $prevq $n --no-enqueue --variant avg --leg $(gleg $X) --leg_feats feats_v28 ;;
      gkcx) set -- tools/gkeyx.py build $srcm $n --no-enqueue ;;
      xncr) set -- tools/gkeyxn.py build $srcm $n --no-enqueue --variant avg_xncr ;;
      db) set -- src/aiu_shift.py build $prevq $n --group block ;;
      aw) set -- tools/antiwin.py build $prevq $n --no-enqueue ;;
      gp) set -- tools/gpool.py build $prevq $n --no-enqueue ;;
      gl) set -- tools/glong.py build $srcm $n --no-enqueue ;;
      gja) set -- tools/gjoint.py build $srcm $n --mode a --no-enqueue ;;
    esac
    step "${t}_${X}$V" 4 3 "$@"
    if [ ! -e "$R/preds/manifest_$spec.json" ]; then echo "$t: no manifest $spec"; exit 4; fi
    prevq=$(qentry $n $spec); echo "$t $prevq" >> "$st"; prev=$spec
  done ;;
avg3)  # chainavg3 of the three chains' gjoint a entries of variant V
  V=$2
  qa=$(grep '^gja ' "$R/research/f3/out/kit_l3_tails_A$V.txt" | cut -d' ' -f2); qb=$(grep '^gja ' "$R/research/f3/out/kit_l3_tails_B$V.txt" | cut -d' ' -f2)
  qc=$(grep '^gja ' "$R/research/f3/out/kit_l3_tails_C$V.txt" | cut -d' ' -f2)
  step "avg3_$V" 3 1 tools/chainavg3.py build "$qa" "$qb" "$qc" $(( 2400 + $(dnum $V) )) --suffix skap --no-enqueue ;;
check)  # twins and corrector stage (every merged tag present); with "finals" also Bb, Ba (and D when built) against 596 (L3S on four months)
  m=""; tw=""
  for X in A B C; do
    for V in Ba Bb D; do tg=$(cb $X)adsbgeondctxv3x50ws75c3r$V; if [ -e "$R/notes/merge_${tg}_full.json" ]; then m="$m,$tg"; fi; done
    if [ -e "$R/cache/full_$(L75 $X)c3_s0.parquet" ] && [ -e "$R/cache/full_$(L50 $X)c3_s0.parquet" ]; then tw="$tw,$(L75 $X):$(L50 $X)"; fi
  done
  if [ "${2:-}" = finals ]; then
    S=$(cb A)adsbgeondctxv3x50ws75c3; fl=""
    for V in Ba Bb D; do f="preds/manifest_${S}r${V}amfhgkcxncrdbawgpglgjaavskap.json"; if [ -e "$R/$f" ]; then fl="$fl,$V=$f"; fi; done
    step check_finals 6 1 research/f2/check_split.py --merged "${m#,}" --twins "${tw#,}" --ctx ctx_v3 --ctx_sha $CTX_SHA \
      --final_default "preds/manifest_${S}amfhgkcxncrdbawgpglgjaavskap.json" --finals "${fl#,}" --out research/f3/out/split_check_l3.json
  else
    step check_corr 6 1 research/f2/check_split.py --merged "${m#,}" --twins "${tw#,}" --ctx ctx_v3 --ctx_sha $CTX_SHA --out research/f3/out/split_check_l3_corr.json
  fi ;;
*) sed -n '2,16p' "$0"; exit 1 ;;
esac
