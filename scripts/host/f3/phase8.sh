#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# F3 Package S on the final root (10-11 01:17 and 01:2x; F2's job list 02:10, research/f2/s_spec.py). Start ONLY on our
# word, once the Mac is free (phase7 finished 602c, or we stopped it) and F2's S sandbox has exported (research/f2/out/
# s_export_<family>.json after its gate). Family 598c (if 601 lost the veto) or l4 (if L4 is the final). Names: 603 = S default, 604 = its
# Jan/Jul twin (597's form, Mac prefix f3f5), 605c = S final; the specs carry the family. Order (01:2x):
#   0. research/f3/collect_s.py: the sandbox's export (val and fwd, the S tools) and its fit cache twice;
#   1. full learners: F2's research/f2/s_learn.py <family> full (s_spec.py's exact commands, one heavy job at a time, RealMLP alone with its smoke
#      fit, watchdog and crash rule; the RealMLP seeds are the ones the sandbox finished), then research/f2/s_check.py cmds and rows on full;
#   2. full averages: research/f2/avg_s.py as s_spec.avg_cmds lists them, then s_check.py avg on full;
#   3. 603 and 604: research/f3/final_chain_s.py <family> chain (chain C, T form: blend, T corrector 0.75 with 0 fits, cavg, manifest, tails) and
#      avg3 (603), the final file ff12, the test move against the family default (596 or 600), s_check.py blend and manifest against the family
#      chain C, 604's Mac part (research/f3/daily_candidate_s.py: the Jan/Jul rows of 603 against the base twin's float file and fired int
#      file), research/f3/out/s_<family>_603.DONE; F3 stages 603 and 604 from the PC (research/f3/stage_s.sh) while the Mac goes on;
#   4. fj: s_learn.py <family> fj on final_run (598c: LightGBM ap only; its CatBoost and RealMLP fj seeds run in f2_run where their seed 0 ran,
#      F2's runner, and are cloned here once f2_run's research/f2/out/s_learn_598c_fj.DONE exists, each sha16 checked against f2_run's
#      research/f2/out/s_learn_598c_fj.json); the fj RealMLP seeds and rm tag are that fj state's rm_seeds_done and rm_tag (F2 10-11 02:4x;
#      l4: final_run's own s_learn_l4_fj.json), at least three and among the full seeds; s_check.py cmds and rows on fj (598c: also rows of
#      the cloned legs); fj averages; s_check.py avg --fj; research/f2/fj_blend_s.py <family> --rm_tag <that tag> (the S fj extra and its pin);
#   5. 605c: research/f3/kit_s.sh <family> corr4m (10 fits per 0.75 seed, 0 per twin), mergeD, merge, check, manifest, tails, avg3, check finals
#      (only "final BbF: Jan/Jul rows differ from the default final" may fail), F2's iii_boundary.py against 603 (must pass), the final file ff13,
#      the test move against the family final (598c or 602c), research/f2/s_gate.py move (rule 3b: over 150 stops here, no DONE, nothing to
#      stage), research/f3/out/s_<family>_605c.DONE. Every s_check record is also kept as research/f3/out/s<family>_s_check_<step>.json
#      (s_check.py writes one name per kind and setting, so two calls of the same kind would overwrite each other).
# Caps as phase7: light steps through research/f3/mac_step.sh with F3CAP 15 and the logged 300 s total-cap override only if nothing else runs on the
# Mac at start (else F3CAP 12, no override; PHASE8_CAP_OVERRIDE_S=0 turns the override off).
# Hold and stop: research/f3/PHASE8_HOLD (wait, no lock held) and research/f3/PHASE8_STOP (end cleanly, exit 0, research/f3/out/phase8.STOPPED),
# checked before each step group; inside the learner steps F2's runner reads ${PRC_WORK:-.}/f2s_HOLD and f2s_STOP before each job. A rerun skips
# finished parts. STOP_F3 stops new work (FAILED).
#   PHASE8_BASE_INT=<the base twin's fired int file in research/f3/in/> nohup bash research/f3/phase8.sh <598c|l4> >> logs/phase8.log 2>&1 < /dev/null &
#   (598c: the base twin is 597 = v43, research/f3/in/spirited-tiger_v43.parquet, the default; l4: 601's fired file, required)
set -u
FAM=${1:-}
case "$FAM" in 598c|l4) ;; *) echo "family 598c or l4"; exit 2 ;; esac
cd ${PRC_WORK:-.}/final_run || exit 1
P=${PRC_WORK:-.}
PY=${PY_MAC:-python3}
export F3_MAXTH=3
unset TOTCAP
OV=${PHASE8_CAP_OVERRIDE_S:-300}
HOLD=research/f3/PHASE8_HOLD
STOP=research/f3/PHASE8_STOP
FS=s$FAM
O=research/f3/out
if [ $FAM = 598c ]; then
  XP=""; FPRE=l3_; FKIT=598c; BASE_FLOAT=research/f3/queue_ready/f3f3_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.parquet
  BASE_INT=${PHASE8_BASE_INT:-research/f3/in/spirited-tiger_v43.parquet}
else
  XP=x; FPRE=l4_; FKIT=l4c; BASE_FLOAT=research/f3/queue_ready/f3f4_m6r28xwincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.parquet
  BASE_INT=${PHASE8_BASE_INT:-}
fi
SDEF=m6r28${XP}wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap        # the family default (596 or 600)
SFIN=m6r28${XP}wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3rBbFamfhgkcxncrdbawgpglgjaavskap    # the family final (598c or 602c)
S603=${SDEF}s
S605=${SFIN}s
S596=m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap
ts() { date '+%m-%d %H:%M:%S'; }
h() { shasum -a 256 "$1" | cut -c1-16; }
die() { echo "$(ts) $*"; touch $O/phase8_$FAM.FAILED; exit 2; }
stopped() { echo "$(ts) phase8 stopped cleanly $1 ($STOP)"; echo "$(date '+%Y-%m-%d %H:%M:%S') $FAM $1" > $O/phase8.STOPPED; exit 0; }
hold_point() {
  local n=0
  while [ -e "$HOLD" ] && [ ! -e "$STOP" ]; do
    [ $((n % 20)) = 0 ] && echo "$(ts) paused $1: $HOLD present (no lock held); delete it to go on, or touch $STOP to end here"
    n=$((n + 1)); sleep 60
  done
  [ -e "$STOP" ] && stopped "$1"
  [ -e STOP_F3 ] && die "STOP_F3 present $1"
  return 0
}
light() {  # light <name> <gb> <log> <python args ...>: one mac_step, 1 thread
  local name=$1 gb=$2 log=$3; shift 3
  bash research/f3/mac_step.sh "${FS}_$name" "$gb" 1 "$log" $PY -X utf8 "$@" || die "$name failed ($log)"
}
keep() {  # keep <s_check record name> <step>: copy research/f2/out/s_check_<name>.json to a name of this step
  cp -p "research/f2/out/s_check_$1.json" "$O/${FS}_s_check_$2.json" || die "the s_check record research/f2/out/s_check_$1.json is missing"
}
spec_py() { $PY -X utf8 -c "import sys, json; sys.path.insert(0, 'research/f2'); import s_spec; r = json.load(open('$O/collect_s_$FAM.json')); $1"; }
[ -e "$STOP" ] && { echo "$(ts) $STOP present at start: not starting"; exit 0; }
[ -e STOP_F3 ] && die "STOP_F3 present"
[ -e $O/phase8_$FAM.FAILED ] && die "$O/phase8_$FAM.FAILED exists"
pgrep -f "research/f3/phase7.sh" > /dev/null && die "phase7 still runs: the Mac is not free"
ps -axo pid=,ppid=,command= > $O/.phase8_ps.$$ 2> /dev/null   # a snapshot, then a filter: this script's own subshells cannot appear in it
n8=$(awk -v me=$$ '$1 != me && $2 != me && index($0, "research/f3/phase8.sh") > 0' $O/.phase8_ps.$$ | wc -l | tr -d ' ')
rm -f $O/.phase8_ps.$$
[ "$n8" = 0 ] || die "another phase8 runs"
busy=$( { ls -d "$P"/*.lock 2> /dev/null | grep -v "/f3_"; ls "$P"/arc_quiet.* 2> /dev/null; pgrep -fl "venv311/bin/python|research/r17_a/|research/f2/|f2_run|phase7" 2> /dev/null; } | head -6 | tr '\n' ';')
if [ -n "$busy" ]; then export F3CAP=12; OV=0; echo "$(ts) other work on the Mac at start ($busy): F3CAP 12, no total-cap override"; else export F3CAP=15; fi
echo "$(ts) phase8 $FAM start (F3CAP $F3CAP, cap override ${OV}s; phase8.sh $(h research/f3/phase8.sh), collect_s.py $(h research/f3/collect_s.py), final_chain_s.py $(h research/f3/final_chain_s.py), kit_s.sh $(h research/f3/kit_s.sh), daily_candidate_s.py $(h research/f3/daily_candidate_s.py))"
for x in "research/f3/final_chain.py 636d00637ab543bd" "research/f3/final_chain_l4.py 663cbae065e18622" "research/f3/daily_candidate.py 6277203e1ad12d2f" \
         "research/f3/mac_step.sh 28ffd44ae1c48b52" "research/f3/final_file.py a60d331cbd6f0570" "research/f3/kit_c.sh dcfabd38f56d23e4" \
         "research/f3/kit_l4c.sh dfdcdb9e35588597" "research/f3/avg_legs_final.py ffc1ca6dd80b1736" "research/f3/test_move.py 00f8cf1787e91b49"; do
  set -- $x; [ "$(h $1)" = "$2" ] || die "$1 is not $2"
done
[ -n "$BASE_INT" ] && [ -e "$BASE_INT" ] || die "the base twin's fired int file is missing (PHASE8_BASE_INT=$BASE_INT)"
for f in preds/manifest_$SDEF.json preds/manifest_$SFIN.json preds/test_$SDEF.parquet preds/test_$SFIN.parquet $BASE_FLOAT research/f3/in/l3s570_int.parquet \
         research/f3/in/test_$S596.parquet $O/final_chain_${FPRE}A.json $O/final_chain_${FPRE}B.json $O/kit_c_tails_${FKIT}_A.txt $O/kit_c_tails_${FKIT}_B.txt; do
  [ -e "$f" ] || die "missing $f"
done
if [ "$OV" != 0 ]; then echo "$OV" > research/f3/CAP_OVERRIDE_S; trap 'rm -f research/f3/CAP_OVERRIDE_S' EXIT; fi
# 0. clones
hold_point "before the clones"
if [ ! -e $O/collect_s_$FAM.json ]; then light collect 4 logs/${FS}_collect.log research/f3/collect_s.py $FAM; fi
for f in avg_s.py s_spec.py s_check.py s_gate.py fj_blend_s.py s_learn.py rm_smoke.py; do
  [ "$(h research/f2/$f)" = "$(spec_py "print(r['tool_sha16']['research/f2/$f'])")" ] || die "research/f2/$f is not the copy collect_s.py made"
done
RMS=$(spec_py "print(','.join(map(str, r['rm_seeds'] or [])))")
echo "$(ts) clones: $(grep -c '"' $O/collect_s_$FAM.json) lines in collect_s_$FAM.json; RealMLP seeds from the sandbox: ${RMS:-none (chain C keeps the family RealMLP)}"
RMARG="--rm_seeds ${RMS:-none}"
if [ -n "$RMS" ]; then LEARN_RM="--rm_seeds $RMS"; FULL_LEGS=""; else LEARN_RM=""; FULL_LEGS="--legs lgbap,cbap"; fi   # fewer than three own seeds: no RealMLP job
# 1. full learners
if [ ! -e research/f2/out/s_learn_${FAM}_full.DONE ]; then
  hold_point "before the full learners"
  $PY -X utf8 -u research/f2/s_learn.py $FAM full $FULL_LEGS $LEARN_RM >> logs/${FS}_learn_full.log 2>&1; rc=$?
  [ $rc = 3 ] && stopped "in the full learners (${PRC_WORK:-.}/f2s_STOP)"
  [ $rc = 0 ] || die "full learners failed (logs/${FS}_learn_full.log, research/f2/out/s_learn_${FAM}.FAILED)"
fi
light chk_cmds_full 3 logs/${FS}_chk_cmds_full.log research/f2/s_check.py cmds $FAM full research/f2/out/s_learn_${FAM}_full.json $RMARG $FULL_LEGS
keep cmds_${FAM}_full cmds_full
light chk_rows_full 3 logs/${FS}_chk_rows_full.log research/f2/s_check.py rows $FAM full $RMARG $FULL_LEGS
keep rows_${FAM}_full rows_full
echo "$(ts) full learners done; s_check cmds and rows on full pass"
# 2. full averages
spec_py "rs = r['rm_seeds'] or []; print('\n'.join(c for c in s_spec.avg_cmds('$FAM', 'full', rm_seeds=rs if len(rs) >= 3 else None) if len(rs) >= 3 or ' rm ' not in c))" > $O/${FS}_avg_full_cmds.txt || die "avg commands"
i=0
while read -r cmd; do
  i=$((i + 1)); set -- $cmd; out=cache/full_$5_s0.parquet
  if [ -e "$out" ]; then echo "$(ts) done earlier: $out"; continue; fi
  light avg_full_$i 4 logs/${FS}_avg_full_$i.log $cmd < /dev/null
done < $O/${FS}_avg_full_cmds.txt
light chk_avg_full 4 logs/${FS}_chk_avg_full.log research/f2/s_check.py avg $FAM full $RMARG
keep avg_${FAM}_full avg_full
echo "$(ts) full averages done: $(for t in $(cut -d' ' -f5 $O/${FS}_avg_full_cmds.txt); do [ -e cache/full_${t}_s0.parquet ] && printf '%s %s ' $t $(h cache/full_${t}_s0.parquet); done)"
# 3. 603 and 604
if [ ! -e $O/s_${FAM}_603.DONE ]; then
  hold_point "before chain C's T form"
  $PY -u research/f3/final_chain_s.py $FAM chain >> logs/${FS}_chain_C.log 2>&1 || die "chain C T form failed (logs/${FS}_chain_C.log)"
  [ -e preds/manifest_$S603.json ] || { $PY -u research/f3/final_chain_s.py $FAM avg3 >> logs/${FS}_avg3.log 2>&1 || die "603 avg3 failed (logs/${FS}_avg3.log)"; }
  echo "$(ts) 603 done: $S603"
  mkdir -p $O/ff12
  [ -e $O/ff12/spirited-tiger_final.parquet ] || light finalfile_603 4 logs/final_file_603.log research/f3/final_file.py $S603 $O/ff12/spirited-tiger_final.parquet \
    research/f3/in/l3s570_int.parquet research/f3/in/test_$S596.parquet $P/r16b_sbx/data/ranking.parquet
  echo "$(ts) final file 603: $O/ff12/spirited-tiger_final.parquet ($(h $O/ff12/spirited-tiger_final.parquet))"
  light move603 3 logs/test_move_603.log research/f3/test_move.py preds/test_$S603.parquet preds/test_$SDEF.parquet data/ranking.parquet $O/test_move_603_vs_default.json
  echo "$(ts) test move 603 vs the family default: $(grep -A2 '"all"' logs/test_move_603.log | tr -d '\n ' | cut -c1-160)"
  TS_=$(spec_py "print(s_spec.chain_names('$FAM', r['rm_seeds'])['S']['T'])")
  TF_=$(spec_py "print(s_spec.fam('$FAM')['fam_C']['T'])")
  light chk_blend 3 logs/${FS}_chk_blend.log research/f2/s_check.py blend $FAM $TS_ $TF_ full $RMARG   # the RealMLP slot: the sandbox's seeds
  keep blend_$TS_ blend
  light chk_manifest 3 logs/${FS}_chk_manifest.log research/f2/s_check.py manifest $FAM preds/manifest_${TS_}adsbgeondctxv3x50ws75c3.json \
    preds/manifest_${TF_}adsbgeondctxv3x50ws75c3.json "$TS_=$TF_"
  MN=${TS_}adsbgeondctxv3x50ws75c3; keep manifest_${MN:0:60} manifest
  light dailycand 6 logs/daily_candidate_s.log research/f3/daily_candidate_s.py $S603 f3f5_$S603 $P/r16b_sbx/data/ranking.parquet \
    $P/r16b_sbx/data/submitting.parquet $BASE_FLOAT $BASE_INT research/f3/queue_ready
  echo "$(ts) 604's Mac part: research/f3/queue_ready/f3f5_$S603.parquet ($(h research/f3/queue_ready/f3f5_$S603.parquet))"
  date '+%Y-%m-%d %H:%M:%S' > $O/s_${FAM}_603.DONE
fi
# 4. fj
if [ ! -e research/f2/out/s_fj_sha16_$FAM.txt ]; then
  hold_point "before the fj learners"
  FJ_LEGS=$FULL_LEGS; [ $FAM = 598c ] && FJ_LEGS="--legs lgbap"   # 598c: CatBoost and RealMLP fj seeds run in f2_run
  if [ ! -e research/f2/out/s_learn_${FAM}_fj.DONE ]; then
    $PY -X utf8 -u research/f2/s_learn.py $FAM fj $FJ_LEGS $LEARN_RM >> logs/${FS}_learn_fj.log 2>&1; rc=$?
    [ $rc = 3 ] && stopped "in the fj learners (${PRC_WORK:-.}/f2s_STOP)"
    [ $rc = 0 ] || die "fj learners failed (logs/${FS}_learn_fj.log)"
  fi
  if [ $FAM = 598c ]; then  # CatBoost and RealMLP fj seeds from f2_run (F2's runner there), cloned with their sha16 from F2's state file
    F2R=$P/f2_run
    n=0; seen=0; gone=0; W0=$(date +%s); WH=${PHASE8_F2RUN_WAIT_H:-8}
    while :; do  # FAILED first: a DONE beside a FAILED never ends the wait (review 10-11)
      [ -e $F2R/research/f2/out/s_learn_598c.FAILED ] && die "f2_run's fj runner failed: $(cat $F2R/research/f2/out/s_learn_598c.FAILED)"
      [ -e $F2R/research/f2/out/s_learn_598c_fj.DONE ] && break
      # liveness (final review 10-11): a runner killed by SIGTERM or SIGKILL writes neither DONE nor FAILED. Alive = an s_learn.py 598c fj
      # process (this root's own fj run has ended by now) or a job lock of it whose recorded pid is alive; gone for three checks in a row
      # after it was seen: stop
      alive=0
      pgrep -f "research/f2/s_learn.py 598c fj" > /dev/null 2>&1 && alive=1
      for L in $P/f2s_598c_*_fj_s*.lock; do
        [ -e "$L/info" ] || continue
        lp=$(sed -n 's/^pid: //p' "$L/info"); [ -n "$lp" ] && kill -0 "$lp" 2> /dev/null && alive=1
      done
      if [ $alive = 1 ]; then seen=1; gone=0
      elif [ $seen = 1 ]; then
        gone=$((gone + 1))
        [ $gone -ge 3 ] && die "f2_run's fj runner is gone (no s_learn.py 598c fj process and no live f2s_598c_*_fj_s*.lock for 3 checks) and left neither DONE nor FAILED: killed? f2_run's logs/f2s_598c_*.log and we decide"
      fi
      [ $(( $(date +%s) - W0 )) -gt $((WH * 3600)) ] && die "f2_run's fj runner: no DONE or FAILED within $WH h of waiting (runner seen running: $seen)"
      [ $((n % 20)) = 0 ] && echo "$(ts) waiting for f2_run's research/f2/out/s_learn_598c_fj.DONE (no lock held; runner seen running: $seen)"
      n=$((n + 1)); sleep 60; hold_point "waiting for f2_run's fj seeds"
    done
    [ -e $O/collect_s_598c_fj.json ] || light clone_fj 3 logs/${FS}_clone_fj.log -c "
import hashlib, json, subprocess, sys
from pathlib import Path
S, D = Path('$F2R'), Path('.')
st = json.loads((S / 'research/f2/out/s_learn_598c_fj.json').read_text())
dn, sp0 = S / 'research/f2/out/s_learn_598c_fj.DONE', S / 'research/f2/out/s_learn_598c_fj.json'
assert not (S / 'research/f2/out/s_learn_598c.FAILED').exists(), 'f2_run has a FAILED marker'
assert dn.stat().st_mtime >= sp0.stat().st_mtime - 2, 'the DONE file is older than the runner state: a stale DONE'
assert st.get('self_check') == {'cmds': True, 'rows': True}, ('the runner state does not record a passing self-check', st.get('self_check'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
rec = {}
for name, j in sorted(st['jobs'].items()):
    if j.get('status') != 'done':
        continue
    seed = name.rsplit('_s', 1)[1]
    tag = j['cmd'].split('--tag ')[1].split()[0]
    for rel in (f'cache/fj_{tag}_s{seed}.parquet', f'preds/es_valid_ids_{tag}_fj_s{seed}.parquet'):
        s, d = S / rel, D / rel
        assert s.exists() and not s.is_symlink(), rel
        if rel.startswith('cache/'):
            assert sha(s) == j['sha16'], (rel, sha(s), j['sha16'])
        if d.exists():
            assert not d.is_symlink() and sha(d) == sha(s), f'exists with other content: {rel}'
        else:
            subprocess.run(['cp', '-cp', str(s), str(d)], check=True)
        assert sha(d) == sha(s)
        rec[rel] = sha(d)
sp = S / 'research/f2/out/s_learn_598c_fj.json'
subprocess.run(['cp', '-p', str(sp), '$O/s_learn_598c_fj_f2run.json'], check=True)
Path('$O/collect_s_598c_fj.json').write_text(json.dumps({'from': str(S), 'state': str(sp), 'state_sha16': sha(sp), 'rm_seeds_done': st.get('rm_seeds_done'),
                                                          'rm_tag': st.get('rm_tag'), 'copied': rec}, indent=1))
print(len(rec), 'files cloned from f2_run; rm_seeds_done', st.get('rm_seeds_done'), 'rm_tag', st.get('rm_tag'))
"
    FJST=$O/collect_s_598c_fj.json
  else
    FJST=research/f2/out/s_learn_${FAM}_fj.json
  fi
  if [ -n "$RMS" ]; then   # chain C has its own RealMLP: the fj average takes the fj seeds that finished (at least three, among the full seeds)
    FJRMS=$(spec_py "j = json.load(open('$FJST')); rs = j.get('rm_seeds_done') or []; full = r['rm_seeds'] or []
assert len(rs) >= 3, f'fj RealMLP seeds finished {rs} (fewer than three) while full has {full}: we decide'
assert set(rs) <= set(full), f'fj RealMLP seeds {rs} are not among the full seeds {full}'
t = j.get('rm_tag'); assert t == s_spec.rm_tag('$FAM', rs, fj=True), f'rm_tag {t} is not the tag of {rs}'
print(','.join(map(str, rs)))") || die "fj RealMLP seeds from $FJST (the error is above)"
  else
    FJRMS=""
  fi
  FJRMARG="--rm_seeds ${FJRMS:-none}"
  echo "$(ts) fj RealMLP seeds ${FJRMS:-none, the family RealMLP fj as in full} (full ${RMS:-none}; from $FJST)"
  light chk_cmds_fj 3 logs/${FS}_chk_cmds_fj.log research/f2/s_check.py cmds $FAM fj research/f2/out/s_learn_${FAM}_fj.json $FJRMARG $FJ_LEGS
  keep cmds_${FAM}_fj cmds_fj
  light chk_rows_fj 3 logs/${FS}_chk_rows_fj.log research/f2/s_check.py rows $FAM fj $FJRMARG $FJ_LEGS
  keep rows_${FAM}_fj rows_fj
  if [ $FAM = 598c ]; then
    light chk_rows_fj_f2run 3 logs/${FS}_chk_rows_fj_f2run.log research/f2/s_check.py rows 598c fj $FJRMARG --legs cbap,rm
    keep rows_598c_fj rows_fj_f2run
  fi
  spec_py "rs = [int(x) for x in '$FJRMS'.split(',') if x]; print('\n'.join(c for c in s_spec.avg_cmds('$FAM', 'fj', fj=True, rm_seeds=rs if len(rs) >= 3 else None) if len(rs) >= 3 or ' rm ' not in c))" > $O/${FS}_avg_fj_cmds.txt || die "fj avg commands"
  i=0
  while read -r cmd; do
    i=$((i + 1)); set -- $cmd; out=cache/fj_$5_s0.parquet
    if [ -e "$out" ]; then echo "$(ts) done earlier: $out"; continue; fi
    light avg_fj_$i 4 logs/${FS}_avg_fj_$i.log $cmd < /dev/null
  done < $O/${FS}_avg_fj_cmds.txt
  light chk_avg_fj 4 logs/${FS}_chk_avg_fj.log research/f2/s_check.py avg $FAM fj --fj $FJRMARG
  keep avg_${FAM}_fj avg_fj
  RMT=$(spec_py "rs = [int(x) for x in '$FJRMS'.split(',') if x]; L = s_spec.chain_names('$FAM', rs)['S']['fj_blend']['learners']; print([x for x in L.split(',') if x.endswith(':mu_rmlp')][0].split(':')[0])")
  echo "$(ts) fj blend RealMLP learner: $RMT"
  hold_point "before the fj blend"
  light fjblend 4 logs/${FS}_fj_blend.log -u research/f2/fj_blend_s.py $FAM --rm_tag $RMT
fi
echo "$(ts) fj done: $(cat research/f2/out/s_fj_sha16_$FAM.txt)"
# 5. 605c
if [ -e $O/s_${FAM}_605c.DONE ]; then echo "$(ts) 605c done earlier"; exit 0; fi
hold_point "before the c form"
K=research/f3/kit_s.sh
bash $K $FAM corr4m >> logs/${FS}_corr4m.log 2>&1 || die "corr4m C failed (logs/${FS}_corr4m.log)"
echo "$(ts) corr4m C done: $(grep -h 'fit cache' logs/${FS}_c4m_C_*_s*.log | sed 's/.*fit cache [^:]*: //' | tr '\n' ' ')"
bash $K $FAM mergeD >> logs/${FS}_merge.log 2>&1 || die "mergeD C failed"
bash $K $FAM merge >> logs/${FS}_merge.log 2>&1 || die "merge C failed"
bash $K $FAM check >> logs/${FS}_check.log 2>&1 || die "corrector-stage check failed (logs/${FS}_check_corr.log)"
echo "$(ts) merges and corrector-stage check done: $(grep -o 'SPLITCHECK pass [A-Za-z]*' logs/${FS}_check_corr.log | tail -1)"
[ -e preds/manifest_m6r28${XP}sapcbrmfw4sv114adsbgeondctxv3x50ws75c3rBbF.json ] || { bash $K $FAM manifest >> logs/${FS}_tails.log 2>&1 || die "manifest C failed"; }
bash $K $FAM tails >> logs/${FS}_tails.log 2>&1 || die "tails C failed (logs/${FS}_tails.log)"
[ -e preds/manifest_$S605.json ] || { bash $K $FAM avg3 >> logs/${FS}_avg3c.log 2>&1 || die "605c avg3 failed"; }
echo "$(ts) 605c done: $S605"
bash $K $FAM check finals >> logs/${FS}_check_finals_run.log 2>&1
echo "$(ts) check finals rc $? (only (iii)'s boundary line may fail)"
$PY -c "import json,sys; f=json.load(open('$O/split_check_$FS.json'))['fails']; print('fails:', f); sys.exit(0 if f in ([], ['final BbF: Jan/Jul rows differ from the default final']) else 1)" \
  || die "check finals failed beyond (iii)'s boundary line"
light iii 4 logs/${FS}_iii_boundary.log -u research/f2/iii_boundary.py --default preds/manifest_$S603.json --finals BbF=preds/manifest_$S605.json --out $O/iii_boundary_$FS.json
echo "$(ts) iii_boundary 605c vs 603 pass"
mkdir -p $O/ff13
[ -e $O/ff13/spirited-tiger_final.parquet ] || light finalfile_605c 4 logs/final_file_605c.log research/f3/final_file.py $S605 $O/ff13/spirited-tiger_final.parquet \
  research/f3/in/l3s570_int.parquet research/f3/in/test_$S596.parquet $P/r16b_sbx/data/ranking.parquet
echo "$(ts) final file 605c: $O/ff13/spirited-tiger_final.parquet ($(h $O/ff13/spirited-tiger_final.parquet))"
light move605 3 logs/test_move_605c.log research/f3/test_move.py preds/test_$S605.parquet preds/test_$SFIN.parquet data/ranking.parquet $O/test_move_605c_vs_final.json
bash research/f3/mac_step.sh ${FS}_sgate_move 3 1 logs/${FS}_s_gate_move.log $PY -X utf8 research/f2/s_gate.py move preds/test_$S605.parquet preds/test_$SFIN.parquet $O/s_gate_move_$FAM.json
echo "$(ts) s_gate move rc $?"
echo "$(ts) 605c vs the family final: $(grep -A2 '"all"' logs/test_move_605c.log | tr -d '\n ' | cut -c1-140); s_gate move $(grep -o '"move_mse": [0-9.]*' $O/s_gate_move_$FAM.json) pass $(grep -o '"pass": [a-z]*' $O/s_gate_move_$FAM.json | tail -1)"
MV=$($PY -c "import json; d = json.load(open('$O/s_gate_move_$FAM.json')); print(d['raw']['move_mse'], bool(d['pass'] and d['rules']['3b_move_le_150']))") \
  || die "no s_gate move record ($O/s_gate_move_$FAM.json)"
set -- $MV
[ "$2" = True ] || die "rule 3b: the final move MSE of 605c against the family final $SFIN is $1, over 150 ($O/s_gate_move_$FAM.json); 605c is not marked done and is not to be staged"
echo "$(ts) rule 3b holds: final move MSE $1 <= 150"
date '+%Y-%m-%d %H:%M:%S' > $O/s_${FAM}_605c.DONE
echo "$(ts) phase8 $FAM done"
