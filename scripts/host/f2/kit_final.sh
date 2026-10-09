#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F2 final split kit for F3 (coordinator 10-09 23:4x: the final is option B with the monthly AIU columns kept; private notes
# research/2026-10-08-f2-four-months.md 5.8; post hoc, raised after chain A's gate14 numbers, decided by the coordinator). Every step works in the
# final root ${PRC_WORK:-.}/final_run, runs its python through F3's PROTOCOL step runner research/f3/mac_step.sh (F3's locks and caps; at most
# KIT_TH threads, default 2) and refuses to overwrite. bash 3.2 compatible. Row groups: jj = research/f2/janjul_ids.parquet (the test ids of the
# original Jan/Jul ranking.parquet, sha16 9db75e3a56fa09b8; final_run/data/ranking.parquet is the four-month file, so it is not used),
# fu = other test rows without an ADS-B match (final_run adsb_v1nd matched, the corrector's s1_matched), fm = with one. Variants (jj | fu | fm):
#   Ba (594a) : T 0.75 | L 0.75 | L 0.75      Bb (594b) : T 0.75 | L 0.75 | L 0.5     (L = research/f2/ctx_corr4.py, one stage-2 output, two scales)
#   D (null)  : T 0.75 | T 0.75 | T 0.75      A         : T 0.75 | T 0.75 | T 0.5     (comparison builds; merges are cheap, their tails optional)
# T 0.75 = F3's --cfg s75 corrector (x50ws75c3, = 531); T 0.5 = the 514-config corrector in final_run (x50wc3).
# Order: install, prep, corr4m A, B, C (then the twins check runs), [F3: final_chain.py chain X --cfg s75 through cavg], merge A, B, C,
#        check, manifest X V, tails X V (Ba and Bb; D and A optional), avg3 V, check finals.
#   bash kit_final.sh install | prep | corr4m <X> | merge <X> | manifest <X> <V> | tails <X> <V> | avg3 <V> | check [finals]    (X: A B C; V: Ba Bb D A)
set -eu
KITDIR=$(cd "$(dirname "$0")" && pwd)
P=${PRC_WORK:-.}
R=${KIT_ROOT:-$P/final_run}
PY=${KIT_PY:-${PY_MAC:-python3}}
TH=${KIT_TH:-2}   # 2: the stage-1 keys of the seeded fit cache (f2_sbx, --threads 2) include the thread count (F3 10-10 00:0x)
ADSB_SHA=${KIT_ADSB_SHA:-07df03ca69833082}; GEO_SHA=9dfa71b64e1099de; CTX_SHA=${KIT_CTX_SHA:-01ac984c7afd6e05}; F2CTX_SHA=531e8113e7f26fff   # final_run tables, read 10-09 23:33 (KIT_*: PC test only)
CTX4=f2ctx4f_v3
cb() { case $1 in A) echo m6r27wincbrmfw4sv434tmc;; B) echo m6r27skcbrmfw4sv434;; C) echo m6r27apcbrmfw4sv114;; *) echo "chain $1?" >&2; exit 9;; esac; }
ex() { case $1 in A) echo f2wincbrmfw4;; B) echo f2skcbrmfw4;; C) echo f2apcbrmfw4;; esac; }
exsha() { case $1 in A) echo bbdfbccf5821da14;; B) echo cd2cf1ea95631aad;; C) echo c0901a4fd4121cc7;; esac; }
num0() { case $1 in A) echo 2000;; B) echo 2100;; C) echo 2200;; esac; }
dnum() { case $1 in D) echo 2000;; A) echo 3000;; Ba) echo 4000;; Bb) echo 5000;; *) echo "variant $1?" >&2; exit 9;; esac; }
gleg() { case $1 in A) echo m6r27skwing;; B) echo m6r27skg;; C) echo m6r27skapg;; esac; }
# MANIFEST_EXTRA_CODE of research/f3/final_chain.py (EXTRA, EXTRA_A, EXTRA_C as of 10-09 23:32) plus the kit's code
E0="src/blend_fixed.py,src/ctx_corr2.py,src/ctx_corr3.py,src/feats_ctx2.py,src/feats_aiu.py,research/ctx_corr/freeze_geo.py,research/obs2/corr_plus.py,research/obs2/obs_rows.py,research/obs2/coverage.py,research/obs2/osm_stands.py,research/adsb/gprobe/build_nd.py,research/r10_b/cavg.py,research/f3/fitwrap.py"
extra() { case $1 in A) echo "$E0,research/r13_l/prep_x.py,research/r13_l/tabm_leg.py,research/r13_l/xrfm_leg.py,research/r13_l/blend_add.py";; B) echo "$E0";;
                     C) echo "$E0,research/r13_f3/model3ap.py,research/r13_f3/cat_muap.py";; esac; }
TAILS="am fh gkc gkcx xncr db aw gp gl gja"
suf() { case $1 in am) echo am;; fh) echo amfh;; gkc) echo amfhgkc;; gkcx) echo amfhgkcx;; xncr) echo amfhgkcxncr;; db) echo amfhgkcxncrdb;;
                   aw) echo amfhgkcxncrdbaw;; gp) echo amfhgkcxncrdbawgp;; gl) echo amfhgkcxncrdbawgpgl;; gja) echo amfhgkcxncrdbawgpglgja;; esac; }
idx() { case $1 in am) echo 5;; fh) echo 6;; gkc) echo 7;; gkcx) echo 8;; xncr) echo 9;; db) echo 10;; aw) echo 11;; gp) echo 12;; gl) echo 13;; gja) echo 14;; esac; }
L75() { echo "$(cb $1)adsbgeondctxv3x50ws75m4"; }
L50() { echo "$(cb $1)adsbgeondctxv3x50wm4"; }
sha() { if command -v shasum > /dev/null; then shasum -a 256 "$1"; else sha256sum "$1"; fi | cut -c1-16; }
ts() { date '+%m-%d %H:%M:%S'; }
new() { for f in "$@"; do if [ -e "$R/$f" ]; then echo "exists, refusing: $f"; exit 3; fi; done; }
step() {  # name peak_gb threads python-args...
  local name=$1 gb=$2 th=$3; shift 3
  if ! (cd "$R" && bash research/f3/mac_step.sh "f2k_$name" "$gb" "$th" "logs/f2k_$name.log" "$PY" -X utf8 -u "$@"); then
    echo "$(ts) step $name failed (final_run/logs/f2k_$name.log)"; exit 4
  fi
  echo "$(ts) $name: $(grep -vE '^[[:space:]]+[0-9]' "$R/logs/f2k_$name.log" | grep -v '^$' | tail -1 | cut -c1-220)"
}
mergeargs() {  # chain variant setting
  local T D jj fu fm; T=$(cb $1); D=${T}adsbgeondctxv3x50ws75c3; jj=$D; fu=$D; fm=$D
  case $2 in A) fm=${T}adsbgeondctxv3x50wc3;; Ba) fu=$(L75 $1)c3; fm=$(L75 $1)c3;; Bb) fu=$(L75 $1)c3; fm=$(L50 $1)c3;; esac
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
install)  # the kit's code into final_run/research/f2 (sha16 against sha16.txt; refuses a different file of the same name)
  mkdir -p "$R/research/f2/out"
  for f in ctx_corr4.py ctx4_table.py merge_corr.py check_split.py janjul_ids.parquet; do
    want=$(grep " $f\$" "$KITDIR/sha16.txt" | cut -c1-16)
    if [ "$(sha "$KITDIR/$f")" != "$want" ]; then echo "kit file $f differs from sha16.txt"; exit 2; fi
    if [ -e "$R/research/f2/$f" ]; then
      if [ "$(sha "$R/research/f2/$f")" != "$want" ]; then echo "a different research/f2/$f is in final_run"; exit 2; fi
    else cp -p "$KITDIR/$f" "$R/research/f2/$f"; fi
    echo "$(ts) research/f2/$f $want"
  done ;;
prep)  # Feb/Jun 2025 context rows and the gate's extra-row bases (APFS clones from f2_run, sha16-checked), then the four-month context table
  for f in f2ctx_v3.parquet f2ctx_v3.json fj_f2wincbrmfw4_s0.parquet fj_f2skcbrmfw4_s0.parquet fj_f2apcbrmfw4_s0.parquet; do
    if [ ! -e "$R/cache/$f" ]; then cp -c "$P/f2_run/cache/$f" "$R/cache/$f"; fi
  done
  if [ "$(sha "$R/cache/f2ctx_v3.parquet")" != $F2CTX_SHA ]; then echo "f2ctx_v3 sha16 differs"; exit 2; fi
  for X in A B C; do if [ "$(sha "$R/cache/fj_$(ex $X)_s0.parquet")" != "$(exsha $X)" ]; then echo "fj_$(ex $X) sha16 differs"; exit 2; fi; done
  if [ "$(sha "$R/cache/ctx_v3.parquet")" != $CTX_SHA ]; then echo "final_run ctx_v3 sha16 differs from $CTX_SHA"; exit 2; fi
  if [ ! -d "$R/research/f2/fitcache_4m" ]; then  # stage-1 models at --threads 2 (560 build and D2 in f2_sbx); no four-month stage-2 model is in it
    cp -c -R "$P/f2_sbx/research/f2/fitcache" "$R/research/f2/fitcache_4m"; echo "$(ts) seeded research/f2/fitcache_4m: $(ls "$R/research/f2/fitcache_4m" | wc -l) files"
  fi
  if [ ! -e "$R/cache/$CTX4.parquet" ]; then
    step ctx4 6 1 research/f2/ctx4_table.py --base ctx_v3 --base_sha $CTX_SHA --extra f2ctx_v3 --extra_sha $F2CTX_SHA --parts 2025-02,2025-06 --name $CTX4
  fi
  sha "$R/cache/$CTX4.parquet" > "$R/research/f2/out/$CTX4.sha16"
  echo "$(ts) $CTX4 $(cat "$R/research/f2/out/$CTX4.sha16")" ;;
corr4m)  # four-month corrector, full setting, seeds 0-2: at 0.75 (10 fits: the S_k with the extra rows; stage 1 loads from the seeded cache), then the 0.5 twin (0 fits)
  X=$2; T=$(cb $X); C4=$(cat "$R/research/f2/out/$CTX4.sha16")
  ARGS="--feats feats_v25 --adsb adsb_v1nd --adsb_sha $ADSB_SHA --folds 10 --rounds 300 --rounds2 400 --max_pred 3600 --threads $TH --geo geo_v1nd --geo_sha $GEO_SHA --ctx $CTX4 --ctx_sha $C4 --extra_base $(ex $X) --extra_sha $(exsha $X) --fit_cache research/f2/fitcache_4m"
  for sc in 0.75 0.5; do
    tg=$(L75 $X); if [ $sc = 0.5 ]; then tg=$(L50 $X); fi
    for k in 0 1 2; do
      if [ -e "$R/cache/full_${tg}_s$k.parquet" ]; then echo "$(ts) skip full_${tg}_s$k (exists)"; continue; fi
      step "c4m_${X}_${sc}_s$k" 6 $TH research/f2/ctx_corr4.py --setting full --seed $k --base $T --base_seed 0 --tag $tg $ARGS --scale2 $sc
      want=10; if [ $sc = 0.5 ]; then want=0; fi  # 0.75: the ten S_k with the extra rows; stage 1 (10 + 45) loads; the 0.5 twin loads everything
      got=$(grep -o '[0-9]* fitted' "$R/logs/f2k_c4m_${X}_${sc}_s$k.log" | tail -1 | grep -o '^[0-9]*')
      if [ "${got:-x}" != $want ]; then echo "$(ts) c4m_${X}_${sc}_s$k fitted ${got:-?} models, expected $want: stop and tell F2 and the coordinator"; exit 5; fi
    done
    if [ ! -e "$R/cache/full_${tg}c3_s0.parquet" ]; then
      step "c4m_${X}_${sc}_cavg" 3 1 research/r10_b/cavg.py --setting full --seed 0 --tag ${tg}c3 --src $tg --seeds 0,1,2 --base $T --base_seed 0
    fi
  done
  step "twins_$X" 4 1 research/f2/check_split.py --twins "$(L75 $X):$(L50 $X)" --out "research/f2/out/twins_$X.json" ;;
merge)  # corrector-stage merges of chain X (val and fwd: byte copies of the default; full: pred by group) and their jobs files
  X=$2; T=$(cb $X); D=${T}adsbgeondctxv3x50ws75c3
  for s in val fwd full; do
    if [ ! -e "$R/cache/${s}_${D}_s0.parquet" ]; then echo "missing cache/${s}_${D}_s0.parquet: run F3's chain $X --cfg s75 through cavg first"; exit 2; fi
  done
  for V in ${KIT_VARIANTS:-Ba Bb D A}; do
    case $V in Ba|Bb) if [ ! -e "$R/cache/full_$(L75 $X)c3_s0.parquet" ] || [ ! -e "$R/cache/full_$(L50 $X)c3_s0.parquet" ]; then
      echo "$(ts) $V: corr4m $X not done, skipped"; continue; fi;; esac
    if [ -e "$R/notes/jobs_${D}r$V.txt" ]; then echo "$(ts) skip merge $X $V (jobs file exists)"; continue; fi
    lines="# F2 final split kit, variant $V (corrector-stage merge after cavg; val, fwd = byte copies of $D)"
    for s in val fwd full; do
      step "merge_${X}${V}_$s" 4 1 $(mergeargs $X $V $s)
      lines="$lines"$'\n'"logs/f2k_merge_${X}${V}_$s.log"$'\t'".venv/Scripts/python.exe -u $(mergeargs $X $V $s)"
    done
    printf '%s\n' "$lines" > "$R/notes/jobs_${D}r$V.txt"
  done ;;
manifest)  # the merged tag's manifest (src.manifest reads the three merge commands from its jobs file; the tails read --base from them)
  X=$2; V=$3; dnum $V > /dev/null; tg=$(cb $X)adsbgeondctxv3x50ws75c3r$V
  new "preds/manifest_$tg.json"
  ex2="$(extra $X),research/f2/merge_corr.py"; case $V in Ba|Bb) ex2="$ex2,research/f2/ctx_corr4.py,research/f2/ctx4_table.py";; esac
  export MANIFEST_EXTRA_CODE="$ex2"
  step "man_${X}$V" 3 1 -m src.manifest $tg $tg pred 0 feats_v27 notes/jobs_$tg.txt ;;
tails)  # artmm .. gjoint a on the merged tag with research/f3/final_chain.py's arguments (sandbox queue numbers num0 + dnum + step index)
  X=$2; V=$3; TCS=$(cb $X)adsbgeondctxv3x50ws75c3r$V; st="$R/research/f2/out/kit_tails_${X}$V.txt"; touch "$st"
  prevq=""; prev=$TCS
  for t in $TAILS; do
    spec=$TCS$(suf $t); n=$(( $(num0 $X) + $(dnum $V) + $(idx $t) ))
    if grep -q "^$t " "$st"; then prevq=$(grep "^$t " "$st" | cut -d' ' -f2); prev=$spec; echo "$(ts) skip $t (done: $prevq)"; continue; fi
    new "preds/manifest_$spec.json"
    srcm=preds/manifest_$prev.json
    case $t in
      am) set -- tools/artmm.py build $srcm $n --no-enqueue ;;
      fh) set -- tools/arthist.py build $srcm $n --no-enqueue ;;
      gkc) set -- tools/gkey.py build $prevq $n --no-enqueue --variant avg --leg $(gleg $X) --leg_feats feats_v27 ;;
      gkcx) set -- tools/gkeyx.py build $srcm $n --no-enqueue ;;
      xncr) set -- tools/gkeyxn.py build $srcm $n --no-enqueue --variant avg_xncr ;;
      db) set -- src/aiu_shift.py build $prevq $n --group block ;;
      aw) set -- tools/antiwin.py build $prevq $n --no-enqueue ;;
      gp) set -- tools/gpool.py build $prevq $n --no-enqueue ;;
      gl) set -- tools/glong.py build $srcm $n --no-enqueue ;;
      gja) set -- tools/gjoint.py build $srcm $n --mode a --no-enqueue ;;
    esac
    step "${t}_${X}$V" 4 $TH "$@"
    if [ ! -e "$R/preds/manifest_$spec.json" ]; then echo "$t: no manifest $spec"; exit 4; fi
    prevq=$(qentry $n $spec); echo "$t $prevq" >> "$st"; prev=$spec
  done ;;
avg3)  # chainavg3 of the three chains' gjoint a entries of variant V
  V=$2
  qa=$(grep '^gja ' "$R/research/f2/out/kit_tails_A$V.txt" | cut -d' ' -f2); qb=$(grep '^gja ' "$R/research/f2/out/kit_tails_B$V.txt" | cut -d' ' -f2)
  qc=$(grep '^gja ' "$R/research/f2/out/kit_tails_C$V.txt" | cut -d' ' -f2)
  step "avg3_$V" 3 1 tools/chainavg3.py build "$qa" "$qb" "$qc" $(( 2400 + $(dnum $V) )) --suffix skap --no-enqueue ;;
check)  # twins and corrector stage (every merged tag present); with "finals" also Ba, Bb, D, A finals against the 531-config final (F3's --cfg s75)
  m=""; tw=""
  for X in A B C; do
    for V in Ba Bb D A; do tg=$(cb $X)adsbgeondctxv3x50ws75c3r$V; if [ -e "$R/notes/merge_${tg}_full.json" ]; then m="$m,$tg"; fi; done
    if [ -e "$R/cache/full_$(L75 $X)c3_s0.parquet" ] && [ -e "$R/cache/full_$(L50 $X)c3_s0.parquet" ]; then tw="$tw,$(L75 $X):$(L50 $X)"; fi
  done
  if [ "${2:-}" = finals ]; then
    S=$(cb A)adsbgeondctxv3x50ws75c3; fl=""
    for V in Ba Bb D A; do f="preds/manifest_${S}r${V}amfhgkcxncrdbawgpglgjaavskap.json"; if [ -e "$R/$f" ]; then fl="$fl,$V=$f"; fi; done
    step check_finals 6 1 research/f2/check_split.py --merged "${m#,}" --twins "${tw#,}" --ctx ctx_v3 --ctx_sha $CTX_SHA \
      --final_default "preds/manifest_${S}amfhgkcxncrdbawgpglgjaavskap.json" --finals "${fl#,}"
  else
    step check_corr 6 1 research/f2/check_split.py --merged "${m#,}" --twins "${tw#,}" --ctx ctx_v3 --ctx_sha $CTX_SHA
  fi
  head -c 3000 "$R/research/f2/out/split_check.json"; echo ;;
*) sed -n '2,15p' "$0"; exit 1 ;;
esac
