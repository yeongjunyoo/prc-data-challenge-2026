#!/bin/bash
# F3 L4 c form (coordinator 10-10 10:07 and 10:2x; 10:5x: L4's versions in new files): research/f3/kit_c.sh (dcfabd38f56d23e4, the kit that built
# 594c and 598c, unchanged) copied with its families replaced by one, l4c, and one more subcommand, mergeD. Family l4c = 598c's form on L4's chains
# (research/f3/final_chain_l4.py --cfg l4s75: L4's CatBoost and RealMLP learners on feats_v28): the four-month corrector on F2's tables adsb_v1ndfjF
# and geo_v1ndfjF at 4 threads from research/f2/fitcache_4m_l4 (R17-A's corrector fits, research/f3/collect_l4.py); extras = L4's fj blends
# (KIT_L4FJ=l4, research/f3/fj_blends_l4.py: the consistent c form the coordinator chose) or, only on the coordinator's word, 598c's L3 fj blends
# (KIT_L4FJ=l3, the cheap fallback); corrector tags end in F (<T>adsbgeondctxv3x50ws75m4F at scale2 0.75, <T>adsbgeondctxv3x50wm4F at 0.5); the
# merged variant is BbF (jj T 0.75 | fu L 0.75 | fm L 0.5; merge groups adsb_v1nd's matched); mergeD adds the null merge D (check iv); step and log
# names l4c_*, the twins, check and tails-state files under research/f3/out/; sandbox queue numbers num0 + 19000 + step index, chainavg3 2400 + 19000.
# Everything else is kit_c.sh's, with the same tools and arguments.
#   bash research/f3/kit_l4c.sh l4c corr4m <X> | merge <X> | mergeD <X> | manifest <X> | tails <X> | avg3 | check [finals]   (X: A B C)
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
l4c)
  cb() { case $1 in A) echo m6r28xwincbrmfw4sv434tmc;; B) echo m6r28xskcbrmfw4sv434;; C) echo m6r28xapcbrmfw4sv114;; *) echo "chain $1?" >&2; exit 9;; esac; }
  L4FJ=${KIT_L4FJ:-l4}
  case $L4FJ in l4|l3) ;; *) echo "KIT_L4FJ must be l4 or l3"; exit 1;; esac
  ex() { case $1 in A) echo ${L4FJ}wincbrmfw4;; B) echo ${L4FJ}skcbrmfw4;; C) echo ${L4FJ}apcbrmfw4;; esac; }
  exsha() { grep " $(ex $1)\$" "$R/research/f3/out/${L4FJ}_fj_sha16.txt" | cut -c1-16; }
  gleg() { case $1 in A) echo m6r28skwing;; B) echo m6r28skg;; C) echo m6r28skapg;; esac; }
  TH=4; FC=research/f2/fitcache_4m_l4; FEATS=feats_v28; DNUM=19000
  E0="src/blend_fixed.py,src/ctx_corr2.py,src/ctx_corr3.py,src/feats_ctx2.py,src/feats_aiu.py,research/ctx_corr/freeze_geo.py,research/obs2/corr_plus.py,research/obs2/obs_rows.py,research/obs2/coverage.py,research/obs2/osm_stands.py,research/adsb/gprobe/build_nd.py,research/r10_b/cavg.py,research/f3/fitwrap.py,research/r13_l/prep_x.py,research/r13_l/tabm_leg.py,research/r13_l/xrfm_leg.py,research/r13_l/blend_add.py,research/r17_a/avg_r17a.py,research/r17_a/adsb_day_r17a.py,research/adsb/build_feats2.py,research/adsb/adsb_match.py,research/f3/avg_legs_final.py,research/f3/feats_v28_final.py,research/r17_a/avg_l4.py,research/r17_a/l4_learn_r17a.sh,research/f3/l4_full_f3.sh,research/f3/collect_l4.py"
  extra() { case $1 in C) echo "$E0,research/r13_f3/model3ap.py,research/r13_f3/cat_muap.py";; *) echo "$E0";; esac; }
  KITCODE="research/f2/merge_corr.py,research/f2/ctx4_table.py,research/f2/ctx_corr4.py,research/f2/model3_fj.py,research/f2/model3ap_fj.py,research/f2/g_band_fj.py,research/f2/blend_fixed_fj.py,research/f3/fj_blends_l3.py,research/f3/kit_l3.sh,research/f3/kit_l4c.sh"
  if [ $L4FJ = l4 ]; then KITCODE="$KITCODE,research/f2/cat_mu_fj.py,research/f2/cat_muap_fj.py,research/f2/rmlp_mu_fj.py,research/f3/fj_blends_l4.py"; fi ;;
*) sed -n '2,20p' "$0"; exit 1 ;;
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
  lines="# F3 kit (research/f3/kit_l4c.sh, $FAM), variant $V (corrector-stage merge after cavg; val, fwd = byte copies of $D; L on F2's fjF tables)"
  for s in val fwd full; do
    step "merge_${X}${V}_$s" 4 1 $(mergeargs $X $s)
    lines="$lines"$'\n'"logs/${FAM}_merge_${X}${V}_$s.log"$'\t'".venv/Scripts/python.exe -u $(mergeargs $X $s)"
  done
  printf '%s\n' "$lines" > "$R/notes/jobs_${D}r$V.txt" ;;
mergeD)  # the null merge D of chain X (every group from the default: the merge machinery's check iv), as the earlier kits' D
  X=$2; T=$(cb $X); D=${T}adsbgeondctxv3x50ws75c3
  if [ -e "$R/notes/jobs_${D}rD.txt" ]; then echo "$(ts) skip merge $X D (jobs file exists)"; exit 0; fi
  lines="# F3 kit (research/f3/kit_l4c.sh, $FAM), variant D (null merge; val, fwd = byte copies of $D)"
  for s in val fwd full; do
    a="research/f2/merge_corr.py --setting $s --seed 0 --tag ${D}rD --base $T --base_seed 0 --default $D --jj $D --fu $D --fm $D --ctx ctx_v3 --ctx_sha $CTX_SHA --adsb adsb_v1nd --adsb_sha $ADSB_SHA --jj_ids research/f2/janjul_ids.parquet"
    step "merge_${X}D_$s" 4 1 $a
    lines="$lines"$'\n'"logs/${FAM}_merge_${X}D_$s.log"$'\t'".venv/Scripts/python.exe -u $a"
  done
  printf '%s\n' "$lines" > "$R/notes/jobs_${D}rD.txt" ;;
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
