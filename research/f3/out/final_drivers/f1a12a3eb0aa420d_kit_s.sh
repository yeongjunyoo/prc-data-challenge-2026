#!/bin/bash
# F3 Package S, c form of chain C on the final root (coordinator 10-11 01:17 and 01:2x; F2's job list 02:10, research/f2/s_spec.py): 605c = the
# family's c final with chain C at the package seed counts. research/f3/kit_c.sh (598c, dcfabd38f56d23e4) and research/f3/kit_l4c.sh (l4c,
# dfdcdb9e35588597) stay as they ran; this file is their copy for chain C only, with F2's S names:
#   chain C = T_S (598c m6r28sapcbrmfw4sv114, L4 m6r28xsapcbrmfw4sv114); the four-month corrector on F2's fjF tables at 4 threads from
#   research/f2/fitcache_4m_s<family> (the S sandbox's corrector fits, research/f3/collect_s.py), extra rows = the S fj blend (598c
#   l3sapcbrmfw4, L4 l4sapcbrmfw4: research/f2/fj_blend_s.py, pin research/f2/out/s_fj_sha16_<family>.txt); tags <T_S>adsbgeondctxv3x50ws75m4F
#   (0.75: 10 fitted per seed, the S_k with the extra rows) and <T_S>adsbgeondctxv3x50wm4F (0.5: 0 fitted); merged variant BbF (jj T 0.75 | fu L
#   0.75 | fm L 0.5, merge groups adsb_v1nd's matched) and the null merge D; manifest, tails; 605c = chainavg3 of the family's c-form A and B gja
#   entries (research/f3/out/kit_c_tails_598c_<A|B>.txt or kit_c_tails_l4c_<A|B>.txt, untouched) and this chain C's, --suffix skaps; check finals
#   against 603 (the S default). Step and log names s598c_* or sl4_*; sandbox queue numbers 2200 + 31000 (598c) or 33000 (L4) + step index,
#   chainavg3 2400 + the same. The S names are checked against research/f2/s_spec.py before any step.
#   bash research/f3/kit_s.sh <598c|l4> corr4m | merge | mergeD | manifest | tails | avg3 | check [finals]
set -eu
FAM=${1:-}
shift || true
P=~/nova-adsb/prcm
R=$P/final_run
PY=$HOME/nova-adsb/venv311/bin/python
V=BbF
X=C
ADSB_SHA=07df03ca69833082; GEO_SHA=9dfa71b64e1099de; CTX_SHA=01ac984c7afd6e05   # final_run tables (as the earlier kits)
ADSBF=adsb_v1ndfjF; ADSBF_SHA=e13437a2c55a77c5; GEOF=geo_v1ndfjF; GEOF_SHA=4fd677dc9060ce22   # F2's lever-2 tables (as kit_c.sh and kit_l4c.sh)
CTX4=f2ctx4f_v3
TAILTH=3
TT=4
FEATS=feats_v28
E3="src/blend_fixed.py,src/ctx_corr2.py,src/ctx_corr3.py,src/feats_ctx2.py,src/feats_aiu.py,research/ctx_corr/freeze_geo.py,research/obs2/corr_plus.py,research/obs2/obs_rows.py,research/obs2/coverage.py,research/obs2/osm_stands.py,research/adsb/gprobe/build_nd.py,research/r10_b/cavg.py,research/f3/fitwrap.py,research/r13_l/prep_x.py,research/r13_l/tabm_leg.py,research/r13_l/xrfm_leg.py,research/r13_l/blend_add.py,research/r17_a/avg_r17a.py,research/r17_a/adsb_day_r17a.py,research/adsb/build_feats2.py,research/adsb/adsb_match.py,research/f3/avg_legs_final.py,research/f3/feats_v28_final.py"
case $FAM in
598c)
  T=m6r28sapcbrmfw4sv114; EX=l3sapcbrmfw4; FKIT=598c; DNUM=31000; XP=""
  E0="$E3"
  KITCODE="research/f2/merge_corr.py,research/f2/ctx_corr4.py,research/f2/ctx4_table.py,research/f2/model3_fj.py,research/f2/model3ap_fj.py,research/f2/g_band_fj.py,research/f2/blend_fixed_fj.py,research/f3/fj_blends_l3.py,research/f3/kit_l3.sh,research/f3/kit_s.sh,research/f2/fj_blend_s.py,research/f2/s_spec.py,research/f2/avg_s.py,research/f2/s_learn.py" ;;
l4)
  T=m6r28xsapcbrmfw4sv114; EX=l4sapcbrmfw4; FKIT=l4c; DNUM=33000; XP=x
  E0="$E3,research/r17_a/avg_l4.py,research/r17_a/l4_learn_r17a.sh,research/f3/l4_full_f3.sh,research/f3/collect_l4.py"
  KITCODE="research/f2/merge_corr.py,research/f2/ctx4_table.py,research/f2/ctx_corr4.py,research/f2/model3_fj.py,research/f2/model3ap_fj.py,research/f2/g_band_fj.py,research/f2/blend_fixed_fj.py,research/f3/fj_blends_l3.py,research/f3/kit_l3.sh,research/f2/cat_mu_fj.py,research/f2/cat_muap_fj.py,research/f2/rmlp_mu_fj.py,research/f3/fj_blends_l4.py,research/f3/kit_s.sh,research/f2/fj_blend_s.py,research/f2/s_spec.py,research/f2/avg_s.py,research/f2/s_learn.py" ;;
*) sed -n '2,16p' "$0"; exit 1 ;;
esac
FS=s$FAM
EXTRA="$E0,research/r13_f3/model3ap.py,research/r13_f3/cat_muap.py,research/f2/avg_s.py"
FC=research/f2/fitcache_4m_s$FAM
PIN=research/f2/out/s_fj_sha16_$FAM.txt
TAIL_A=m6r28${XP}wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3rBbFamfhgkcxncrdbawgpglgja
TAIL_B=m6r28${XP}skcbrmfw4sv434adsbgeondctxv3x50ws75c3rBbFamfhgkcxncrdbawgpglgja
S603=m6r28${XP}wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskaps
S605=m6r28${XP}wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3rBbFamfhgkcxncrdbawgpglgjaavskaps
(cd "$R" && "$PY" -c "
import sys; sys.path.insert(0, 'research/f2'); import s_spec
n = s_spec.chain_names('$FAM'); S = n['S']
assert (S['T'], S['fj_blend']['tag'], n['names']['603'], n['names']['605c'], S['L75'], S['L50']) == ('$T', '$EX', '$S603', '$S605', '${T}adsbgeondctxv3x50ws75m4F', '${T}adsbgeondctxv3x50wm4F'), n
") || { echo "kit_s.sh names differ from research/f2/s_spec.py"; exit 8; }
TAILS="am fh gkc gkcx xncr db aw gp gl gja"
suf() { case $1 in am) echo am;; fh) echo amfh;; gkc) echo amfhgkc;; gkcx) echo amfhgkcx;; xncr) echo amfhgkcxncr;; db) echo amfhgkcxncrdb;;
                   aw) echo amfhgkcxncrdbaw;; gp) echo amfhgkcxncrdbawgp;; gl) echo amfhgkcxncrdbawgpgl;; gja) echo amfhgkcxncrdbawgpglgja;; esac; }
idx() { case $1 in am) echo 5;; fh) echo 6;; gkc) echo 7;; gkcx) echo 8;; xncr) echo 9;; db) echo 10;; aw) echo 11;; gp) echo 12;; gl) echo 13;; gja) echo 14;; esac; }
L75="${T}adsbgeondctxv3x50ws75m4F"
L50="${T}adsbgeondctxv3x50wm4F"
D=${T}adsbgeondctxv3x50ws75c3
ts() { date '+%m-%d %H:%M:%S'; }
new() { for f in "$@"; do if [ -e "$R/$f" ]; then echo "exists, refusing: $f"; exit 3; fi; done; }
step() {  # name peak_gb threads python-args...
  local name=$1 gb=$2 th=$3; shift 3
  if ! (cd "$R" && bash research/f3/mac_step.sh "${FS}_$name" "$gb" "$th" "logs/${FS}_$name.log" "$PY" -X utf8 -u "$@"); then
    echo "$(ts) step $name failed (final_run/logs/${FS}_$name.log)"; exit 4
  fi
  echo "$(ts) $name: $(grep -vE '^[[:space:]]+[0-9]' "$R/logs/${FS}_$name.log" | grep -v '^$' | tail -1 | cut -c1-220)"
}
mergeargs() {  # setting
  echo "research/f2/merge_corr.py --setting $1 --seed 0 --tag ${D}r$V --base $T --base_seed 0 --default $D --jj $D --fu ${L75}c3 --fm ${L50}c3 --ctx ctx_v3 --ctx_sha $CTX_SHA --adsb adsb_v1nd --adsb_sha $ADSB_SHA --jj_ids research/f2/janjul_ids.parquet"
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
corr4m)  # the four-month corrector of chain C on F2's tables, full setting, seeds 0-2: 0.75 (10 fits: the S_k with the extra rows), then the 0.5 twin (0 fits)
  C4=$(cat "$R/research/f2/out/$CTX4.sha16"); XS=$(grep " $EX\$" "$R/$PIN" | cut -c1-16)
  [ -n "$XS" ] || { echo "no sha16 for $EX in $PIN"; exit 2; }
  [ "$(shasum -a 256 "$R/cache/fj_${EX}_s0.parquet" | cut -c1-16)" = "$XS" ] || { echo "cache/fj_${EX}_s0.parquet is not the pinned $XS"; exit 2; }
  ARGS="--feats feats_v25 --adsb $ADSBF --adsb_sha $ADSBF_SHA --folds 10 --rounds 300 --rounds2 400 --max_pred 3600 --threads $TT --geo $GEOF --geo_sha $GEOF_SHA --ctx $CTX4 --ctx_sha $C4 --extra_base $EX --extra_sha $XS --fit_cache $FC"
  for sc in 0.75 0.5; do
    tg=$L75; if [ $sc = 0.5 ]; then tg=$L50; fi
    for k in 0 1 2; do
      if [ -e "$R/cache/full_${tg}_s$k.parquet" ]; then echo "$(ts) skip full_${tg}_s$k (exists)"; continue; fi
      step "c4m_${X}_${sc}_s$k" 6 $TT research/f2/ctx_corr4.py --setting full --seed $k --base $T --base_seed 0 --tag $tg $ARGS --scale2 $sc
      want=10; if [ $sc = 0.5 ]; then want=0; fi
      got=$(grep -o '[0-9]* fitted' "$R/logs/${FS}_c4m_${X}_${sc}_s$k.log" | tail -1 | grep -o '^[0-9]*')
      if [ "${got:-x}" != $want ]; then echo "$(ts) c4m_${X}_${sc}_s$k fitted ${got:-?} models, expected $want: stop and tell the coordinator"; exit 5; fi
    done
    if [ ! -e "$R/cache/full_${tg}c3_s0.parquet" ]; then
      step "c4m_${X}_${sc}_cavg" 3 1 research/r10_b/cavg.py --setting full --seed 0 --tag ${tg}c3 --src $tg --seeds 0,1,2 --base $T --base_seed 0
    fi
  done
  step "twins_$X" 4 1 research/f2/check_split.py --twins "$L75:$L50" --out "research/f3/out/twins_${FS}_$X.json" ;;
merge)  # the corrector-stage merge of chain C (val and fwd: byte copies of the default; full: pred by group) and its jobs file
  for s in val fwd full; do
    if [ ! -e "$R/cache/${s}_${D}_s0.parquet" ]; then echo "missing cache/${s}_${D}_s0.parquet"; exit 2; fi
  done
  if [ ! -e "$R/cache/full_${L75}c3_s0.parquet" ] || [ ! -e "$R/cache/full_${L50}c3_s0.parquet" ]; then echo "$(ts) corr4m not done"; exit 2; fi
  if [ -e "$R/notes/jobs_${D}r$V.txt" ]; then echo "$(ts) skip merge $V (jobs file exists)"; exit 0; fi
  lines="# F3 kit (research/f3/kit_s.sh, $FAM), variant $V (corrector-stage merge after cavg; val, fwd = byte copies of $D; L on F2's fjF tables)"
  for s in val fwd full; do
    step "merge_${X}${V}_$s" 4 1 $(mergeargs $s)
    lines="$lines"$'\n'"logs/${FS}_merge_${X}${V}_$s.log"$'\t'".venv/Scripts/python.exe -u $(mergeargs $s)"
  done
  printf '%s\n' "$lines" > "$R/notes/jobs_${D}r$V.txt" ;;
mergeD)  # the null merge D of chain C (every group from the default: the merge machinery's check iv)
  if [ -e "$R/notes/jobs_${D}rD.txt" ]; then echo "$(ts) skip merge D (jobs file exists)"; exit 0; fi
  lines="# F3 kit (research/f3/kit_s.sh, $FAM), variant D (null merge; val, fwd = byte copies of $D)"
  for s in val fwd full; do
    a="research/f2/merge_corr.py --setting $s --seed 0 --tag ${D}rD --base $T --base_seed 0 --default $D --jj $D --fu $D --fm $D --ctx ctx_v3 --ctx_sha $CTX_SHA --adsb adsb_v1nd --adsb_sha $ADSB_SHA --jj_ids research/f2/janjul_ids.parquet"
    step "merge_${X}D_$s" 4 1 $a
    lines="$lines"$'\n'"logs/${FS}_merge_${X}D_$s.log"$'\t'".venv/Scripts/python.exe -u $a"
  done
  printf '%s\n' "$lines" > "$R/notes/jobs_${D}rD.txt" ;;
manifest)  # the merged tag's manifest (src.manifest reads the three merge commands from its jobs file)
  tg=${D}r$V
  new "preds/manifest_$tg.json"
  export MANIFEST_EXTRA_CODE="$EXTRA,$KITCODE"
  step "man_${X}$V" 3 1 -m src.manifest $tg $tg pred 0 $FEATS notes/jobs_$tg.txt ;;
tails)  # artmm .. gjoint a on the merged tag with research/f3/final_chain.py's arguments
  TCS=${D}r$V; st="$R/research/f3/out/kit_c_tails_${FS}_$X.txt"; touch "$st"
  prevq=""; prev=$TCS
  for t in $TAILS; do
    spec=$TCS$(suf $t); n=$(( 2200 + DNUM + $(idx $t) ))
    if grep -q "^$t " "$st"; then prevq=$(grep "^$t " "$st" | cut -d' ' -f2); prev=$spec; echo "$(ts) skip $t (done: $prevq)"; continue; fi
    new "preds/manifest_$spec.json"
    srcm=preds/manifest_$prev.json
    case $t in
      am) set -- tools/artmm.py build $srcm $n --no-enqueue ;;
      fh) set -- tools/arthist.py build $srcm $n --no-enqueue ;;
      gkc) set -- tools/gkey.py build $prevq $n --no-enqueue --variant avg --leg m6r28skapg --leg_feats $FEATS ;;
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
avg3)  # 605c: chainavg3 of the family's c-form A and B gja entries and this chain C's
  qa=$(grep '^gja ' "$R/research/f3/out/kit_c_tails_${FKIT}_A.txt" | cut -d' ' -f2); qb=$(grep '^gja ' "$R/research/f3/out/kit_c_tails_${FKIT}_B.txt" | cut -d' ' -f2)
  qc=$(grep '^gja ' "$R/research/f3/out/kit_c_tails_${FS}_C.txt" | cut -d' ' -f2)
  case "$qa" in *_$TAIL_A) ;; *) echo "chain A's c-form gja entry '$qa' is not the family's $TAIL_A"; exit 2;; esac
  case "$qb" in *_$TAIL_B) ;; *) echo "chain B's c-form gja entry '$qb' is not the family's $TAIL_B"; exit 2;; esac
  [ -n "$qc" ] || { echo "chain C's S gja entry missing"; exit 2; }
  new "preds/manifest_$S605.json"
  step "avg3_$V" 3 1 tools/chainavg3.py build "$qa" "$qb" "$qc" $(( 2400 + DNUM )) --suffix skaps --no-enqueue
  [ -e "$R/preds/manifest_$S605.json" ] || { echo "no manifest for 605c ($S605)"; exit 4; } ;;
check)  # twins and corrector stage (BbF and the null merge D of chain C); with "finals" also 605c against 603
  m=""; tw=""
  for W in $V D; do tg=${D}r$W; if [ -e "$R/notes/merge_${tg}_full.json" ]; then m="$m,$tg"; fi; done
  if [ -e "$R/cache/full_${L75}c3_s0.parquet" ] && [ -e "$R/cache/full_${L50}c3_s0.parquet" ]; then tw="$L75:$L50"; fi
  if [ "${2:-}" = finals ]; then
    step check_finals 6 1 research/f2/check_split.py --merged "${m#,}" --twins "$tw" --ctx ctx_v3 --ctx_sha $CTX_SHA \
      --final_default "preds/manifest_$S603.json" --finals "$V=preds/manifest_$S605.json" --out research/f3/out/split_check_$FS.json
  else
    step check_corr 6 1 research/f2/check_split.py --merged "${m#,}" --twins "$tw" --ctx ctx_v3 --ctx_sha $CTX_SHA --out research/f3/out/split_check_${FS}_corr.json
  fi ;;
*) sed -n '2,16p' "$0"; exit 1 ;;
esac
