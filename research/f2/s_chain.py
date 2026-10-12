# SPDX-License-Identifier: GPL-3.0-only
"""F2 Package S sandbox driver (Mac): chain C at the package seed counts in a new APFS clone (10-11 01:17: never edit r17a_sbx or
final_run in place; new files only; no pinned script edited). One clone per family, made by research/f2/mac/s_clone.sh:
  598c: <mac-workdir>/prcm/f2s_sbx598 = clone of r17a_sbx (L3S chains A, B, C at 0.75 there; 570's val and fwd = 596's)
  l4:   <mac-workdir>/prcm/f2s_sbxl4  = clone of r17a_l4g_sbx (the L4 gate-only chains, phase f, 0.75; their val and fwd = 600's)
Two name sets in the clone, as R17-A's L4 precedent kept them apart:
  final names (research/f2/s_spec.py T_S, e.g. m6r28sapcbrmfw4sv114): val and fwd only, everything real; F3 clones these (s_export) and
    computes full on the final root. No full file under a final name exists here, except the placeholder links of the averaged legs.
  gate names (T_G = T_S with "sg"/"xsg", never staged, never copied): the same chain with full from placeholder legs (the family's own
    seed-0 or shared files, research/f2/out/s_placeholders_<family>.json), so src.manifest, the tails and chainavg3 can run; the gate reads
    val and fwd only. Its val and fwd must equal the final-name files bit for bit (checked) and its correctors must fit nothing.
    The RealMLP placeholder is linked right before gatechain from the seeds that actually finished: full_<m6r2xrmx4s or x3s>_s0 -> the
    family RealMLP's full file; with fewer than three own seeds chain C keeps the family RealMLP, whose full file is real, and no link is made.
Steps (cwd = the clone, checked at every step; each tool run under <mac-workdir>/prcm/f2s_<family>_<step>.lock with the PROTOCOL checks):
  prep       inputs present (feature files, seed-0 legs, family chain C, A and B ends, reference), 598c: R13-F3's CatBoost ap seeds 1 and 2
             (val, fwd) cloned from r13f3_run with their sha16 pinned, feature files checked against the roots where seed 0 ran,
             placeholder links of the LightGBM ap and CatBoost ap averages, an empty corrector fit cache research/f2/fitcache_s
  [learners] research/f2/s_learn.py <family> val,fwd  (a separate process; this driver waits for its DONE file)
  avg        research/f2/avg_s.py for the three legs, val and fwd, then research/f2/s_check.py avg
  final      final names: blend_fixed val, fwd (--check), ctx_corr3 seeds 0-2 val, fwd at 0.75 with --fit_cache research/f2/fitcache_s
             (the fits are new: the val base changed), cavg val, fwd
  gatechain  the RealMLP placeholder; gate names: blend_fixed val, fwd, full; ctx_corr3 seeds 0-2 x val, fwd, full (0 fitted expected; val
             and fwd must equal the final names' bit for bit); cavg; src.manifest; tails artmm .. gjoint a as research/r17_a/sb_build_l3s.py;
             chainavg3 with the family's A and B ends, --suffix skaps
  gate       research/f2/s_check.py rows, cmds, avg, blend, manifest (rule 1) and research/f2/s_gate.py sbx against the family's three-chain end
             (rules 2, 3a; tested on the unrounded numbers) -> research/f2/out/s_gate_sbx_<family>.json and s_verdict_sbx_<family>.json.
             A FAIL (or a missing check) keeps those numbers, writes research/f2/out/s_chain_<family>.FAILED and exits non-zero.
  export     the files F3 clones to the final root, with sha16: research/f2/out/s_export_<family>.json (only after a PASS verdict)
Every step refuses to run while s_chain_<family>.FAILED stands (we clear it by hand) or before the steps ahead of it are done.
The RealMLP seeds come from the learner runner's record only once its s_learn_<family>_valfwd.DONE exists. A step that failed part way
resumes: every output this driver made is recorded with its sha16, the command that made it and the sha16 of its direct inputs (state
"made"); a rerun skips it only if all three still match and refuses anything else (a changed input, a changed command, a target this
driver did not make). A step before gate removes the export, verdict and gate records of an earlier pass; gate removes the export.
    <mac-workdir>/venv311/bin/python -X utf8 -u research/f2/s_chain.py <598c|l4> <prep|avg|final|gatechain|gate|export|all> [--dry]
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avg_s  # noqa: E402
import s_spec  # noqa: E402

ROOT = Path.cwd()
P = Path(os.environ.get("PRC_WORK", "."))  # Mac work folder that holds the build sandboxes
PY = sys.executable
OUT = ROOT / "research" / "f2" / "out"
FC = s_spec.FITCACHE_SBX
REFC = s_spec.REF + " --check"
STEPS = ["prep", "avg", "final", "gatechain", "gate", "export"]
DRY = False


def ts() -> str:
    return time.strftime("%m-%d %H:%M:%S")


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


_SHA: dict = {}


def fsha(p: Path) -> str:
    """sha16 of a file (through links), remembered for this process by path, size and mtime."""
    st = p.stat()
    k = (str(p), st.st_size, st.st_mtime_ns)
    if k not in _SHA:
        _SHA[k] = sha16(p)
    return _SHA[k]


def opts(flags: str) -> dict:
    t = flags.split()
    return {t[i]: t[i + 1] for i in range(len(t) - 1) if t[i].startswith("--") and not t[i + 1].startswith("--")}


REF_LEGS = [opts(s_spec.REF)["--ref_base"], opts(s_spec.REF)["--ref_tag"]] + [x.split(":")[0] for x in opts(s_spec.REF)["--ref_learners"].split(",")]
CORR_FEATS = opts(s_spec.CORR_SBX)["--feats"]


def budget_ok(thr: int) -> tuple[bool, str]:
    if (P / "arc_quiet.request").exists() or (P / "arc_quiet.lock").exists():
        return False, "ARC window"
    st, heavy = 0.0, []
    for d in P.glob("*.lock"):
        if d.is_dir() and d.name != "arc_quiet.lock":
            txt = (d / "info").read_text(errors="replace") if (d / "info").exists() else ""
            t = re.search(r"(?im)^threads[^0-9]*([0-9]+(?:\.[0-9]+)?)", txt)
            v = float(t.group(1)) if t else 6.0
            st += v
            if v >= 4:
                heavy.append(d.name)
    if thr >= 4 and heavy:
        return False, "another heavy job: " + ", ".join(heavy)
    if st + thr > (13 if thr == 1 else 12):
        return False, f"threads {st} + {thr}"
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.strip().splitlines()
    m = re.search(r"([0-9]+)%", out[-1]) if out else None
    f = int(m.group(1)) if m else 0
    return f >= 30, f"free {f} %"


def run(args: list[str], log: str, peak: float, thr: int, mins: int, lockname: str, fam: str, env: dict | None = None) -> str:
    if DRY:
        print(f"[dry] {' '.join(args)}" + (f"   env {env}" if env else ""), flush=True)
        return ""
    ok, k = 0, 0
    while ok < 2:
        if (P / "f2s_STOP").exists():
            raise SystemExit(f"f2s_STOP present before {lockname}")
        g, why = budget_ok(thr)
        ok = ok + 1 if g and not (P / "f2s_HOLD").exists() else 0
        if ok < 2:
            if k % 20 == 0:
                print(f"[{ts()}] waiting before {lockname}: {'f2s_HOLD' if (P / 'f2s_HOLD').exists() else why}", flush=True)
            k += 1
            time.sleep(30)
    lock = P / f"f2s_{fam}_{lockname}.lock"
    lock.mkdir()
    try:
        (lock / "info").write_text(f"owner: PRC F2 Package S sandbox chain C ({fam} {lockname})\npid: {os.getpid()}\npeak_gb: {peak}\n"
                                   f"threads: {thr}\nstart: {time.strftime('%Y-%m-%d %H:%M')}\n"
                                   f"expected_end: {time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time() + 60 * mins))} KST\n"
                                   f"job: {' '.join(args)[:200]}\n")
        print(f"[{ts()}] run {' '.join(args)[:240]}", flush=True)
        e = {**os.environ, "OMP_NUM_THREADS": str(thr), "OPENBLAS_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8", "PYTHONFAULTHANDLER": "1", **(env or {})}
        with open(ROOT / "logs" / log, "w", encoding="utf-8") as f:
            r = subprocess.run(["nice", "-n", "10", "/usr/bin/time", "-l", PY, "-X", "utf8", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=e)
    finally:
        shutil.rmtree(lock, ignore_errors=True)
    txt = (ROOT / "logs" / log).read_text(encoding="utf-8", errors="replace")
    rl = re.search(r"([0-9.]+)\s+real", txt)
    fc = re.search(r"fit cache \S+: (\d+) loaded, (\d+) fitted", txt)
    print(f"[{ts()}]   {log}: rc {r.returncode}" + (f" | real {float(rl.group(1)):.0f} s" if rl else "")
          + (f" | fit cache {fc.group(1)} loaded, {fc.group(2)} fitted" if fc else ""), flush=True)
    if r.returncode != 0:
        raise SystemExit(f"step failed: logs/{log}")
    return txt


def check_root(F: dict) -> None:
    """Every step runs only in its family's clone, <mac-workdir>/prcm/<f2s_sbx598|f2s_sbxl4> (03:1x: not only prep)."""
    if not DRY and (ROOT.name != F["sbx"] or ROOT.parent != P):
        raise SystemExit(f"run in <mac-workdir>/prcm/{F['sbx']}, not {ROOT}")


def jobs_write(tag: str, header: str, lines: list[str]) -> None:
    if not DRY:
        (ROOT / "notes" / f"jobs_{tag}.txt").write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


class Chain:
    def __init__(self, fam: str):
        self.fam = fam
        self.F = s_spec.fam(fam)
        self.sp = OUT / f"s_chain_{fam}.json"
        self.st = json.loads(self.sp.read_text(encoding="utf-8")) if self.sp.exists() else {"family": fam, "done": [], "names": {}, "fit": {}, "repro": {}}
        self.st.setdefault("made", {})
        self.learn = OUT / f"s_learn_{fam}_valfwd.json"
        self.failed = OUT / f"s_chain_{fam}.FAILED"

    def save(self) -> None:
        if not DRY:
            OUT.mkdir(parents=True, exist_ok=True)
            self.st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.sp.write_text(json.dumps(self.st, indent=1, ensure_ascii=False), encoding="utf-8")

    # Outputs this driver made (03:5x fix A): each records its own sha16, the command that made it and the sha16 of its direct
    # inputs. A rerun skips existing targets only if all three still match; any difference, a target this driver did not make, or only some
    # outputs of one command present, is refused (a partial rerun after a change cannot be resumed: start again in a new clone).
    def inputs(self, rels: list[str]) -> dict:
        if DRY:
            return {}
        return {r: (fsha(ROOT / r) if (ROOT / r).exists() else "missing") for r in sorted(set(rels))}

    def todo(self, rels: list[str], cmd: str, ins: dict) -> bool:
        have = [r for r in rels if (ROOT / r).exists() or (ROOT / r).is_symlink()]
        if not have:
            return True
        if DRY:
            raise SystemExit(f"targets already exist, refusing: {have}")
        bad = []
        for r in rels:
            p, m = ROOT / r, self.st["made"].get(r)
            if not p.exists() or p.is_symlink() or not isinstance(m, dict):
                bad.append(f"{r}: " + ("missing while other outputs of the same command exist" if not p.exists() else "not made by this driver"))
                continue
            d = (["output"] if m["sha16"] != fsha(p) else []) + (["command"] if m["cmd"] != cmd else [])
            d += [f"input {x}" for x in sorted(set(m["inputs"]) | set(ins)) if m["inputs"].get(x) != ins.get(x)]
            if d:
                bad.append(f"{r}: made from another {', '.join(d[:4])}")
        if bad:
            raise SystemExit("refusing to reuse or overwrite: " + "; ".join(bad[:4]) + " (start again in a new clone)")
        print(f"[{ts()}] made earlier by this driver from the same command and inputs, skipping: {', '.join(rels)}", flush=True)
        return False

    def mark(self, rels: list[str], cmd: str, ins: dict) -> None:
        if DRY:
            return
        for r in rels:
            if not (ROOT / r).exists():
                raise SystemExit(f"expected output missing: {r}")
            self.st["made"][r] = {"sha16": fsha(ROOT / r), "cmd": cmd, "inputs": ins}
        self.save()

    @staticmethod
    def es(tag: str, s: str, k: int = 0) -> list[str]:
        return [] if s == "full" else [f"preds/es_valid_ids_{tag}_{s}_s{k}.parquet"]

    def man_files(self, spec: str) -> list[str]:
        """A manifest and the prediction files it names."""
        rel = f"preds/manifest_{spec}.json"
        out = [rel]
        if not DRY and (ROOT / rel).exists():
            m = json.loads((ROOT / rel).read_text(encoding="utf-8"))
            out += [m[k]["path"] for k in ("holdout", "forward", "test") if isinstance(m.get(k), dict) and m[k].get("path")]
        return out

    def q_files(self, name: str) -> list[str]:
        """A queue entry, its prediction file, and its manifest with the files that one names."""
        return [f"queue/{name}.json", f"queue/{name}.parquet"] + (self.man_files(name.split("_", 1)[1]) if "_" in name else [])

    def rm_seeds(self) -> list[int]:
        """The RealMLP seeds chain C takes, from the learner runner's record; only a finished run counts (its DONE beside the record), so an
        unfinished record never reads as "none"."""
        if DRY and not self.learn.exists():
            return self.F["legs"]["rm"]["seeds"]
        if not DRY and not (OUT / f"s_learn_{self.fam}_valfwd.DONE").exists():
            raise SystemExit(f"learners not done: research/f2/out/s_learn_{self.fam}_valfwd.DONE missing")
        s = json.loads(self.learn.read_text(encoding="utf-8"))
        if "rm_seeds_done" not in s or "rm_use_own" not in s:
            raise SystemExit(f"{self.learn.name} has no RealMLP result")
        return s["rm_seeds_done"] if s["rm_use_own"] else []

    def names(self) -> dict:
        return s_spec.chain_names(self.fam, self.rm_seeds())

    def fam_sbx_T(self) -> str:
        return self.F["fam_C"]["sbx_TCS"][: -len(s_spec.MID + "c3")]

    def link_placeholder(self, new: str, old: str, rec: dict) -> None:
        p = ROOT / "cache" / f"full_{new}_s0.parquet"
        tgt = f"full_{old}_s0.parquet"
        if DRY:
            print(f"[dry] ln -s {tgt} cache/{p.name}", flush=True)
            return
        if not (ROOT / "cache" / tgt).exists():
            raise SystemExit(f"placeholder target cache/{tgt} missing")
        if p.is_symlink():
            if os.readlink(p) != tgt:
                raise SystemExit(f"{p.name} links elsewhere ({os.readlink(p)})")
        elif p.exists():
            raise SystemExit(f"{p.name} exists and is not a link")
        elif DRY:
            print(f"[dry] ln -s {tgt} cache/{p.name}", flush=True)
            return
        else:
            os.symlink(tgt, p)
        rec[f"cache/{p.name}"] = {"target": f"cache/{tgt}", "target_sha16": sha16(p)}

    def save_placeholders(self, links: dict) -> None:
        if DRY:
            return
        pf = OUT / f"s_placeholders_{self.fam}.json"
        cur = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else {"links": {}}
        cur["links"].update(links)
        cur["rule"] = ("the gate chain's full setting reads these links: the family's own files, so its full reproduces the family chain C's full; "
                       "the gate reads val and fwd only; the RealMLP link follows the seeds that finished (none when chain C keeps the family RealMLP)")
        OUT.mkdir(parents=True, exist_ok=True)
        pf.write_text(json.dumps(cur, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- prep
    def prep(self) -> None:
        F, fam = self.F, self.fam
        L = F["legs"]
        need = ["cache/feats_v28.parquet", "cache/all.parquet", "cache/ctx_v3.parquet"]
        if fam == "598c":
            need.append("cache/feats_v27.parquet")
        for st in ("val", "fwd"):
            need += [f"cache/{st}_m6r28skap_s0.parquet", f"cache/{st}_m6r28skapg_s0.parquet", f"preds/es_valid_ids_m6r28skapg_{st}_s0.parquet",
                     f"preds/es_valid_ids_m6r28skap_{st}_s0.parquet", f"cache/{st}_{L['cbap']['tag']}_s0.parquet", f"preds/es_valid_ids_{L['cbap']['tag']}_{st}_s0.parquet",
                     f"cache/{st}_{L['rm']['tag']}_s0.parquet", f"preds/es_valid_ids_{L['rm']['tag']}_{st}_s0.parquet",
                     f"cache/{st}_{L['rm']['base'][st][0]}_s0.parquet", f"cache/{st}_m6r25xg_s0.parquet",
                     f"cache/{st}_{s_spec.fam_rm(fam)}_s0.parquet", f"preds/es_valid_ids_{s_spec.fam_rm(fam)}_{st}_s0.parquet",
                     f"cache/{st}_{self.fam_sbx_T()}_s0.parquet", f"notes/blend_fixed_{self.fam_sbx_T()}_{st}_s0.json"]
        need += [f"preds/manifest_{F['fam_C']['sbx_TCS']}.json", "cache/full_m6r25xg_s0.parquet"]
        need += [f"cache/full_{tgt}_s0.parquet" for tgt in F["placeholders"].values()]
        need += list(F["ab_state"].values())
        if "ref_sbx_spec_file" in F:
            need.append(F["ref_sbx_spec_file"])
        miss = [x for x in need if not (ROOT / x).exists()]
        rec = {"root": str(ROOT), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "missing": miss}
        if miss:
            print(f"[{ts()}] prep: missing {miss}", flush=True)
            if not DRY:
                raise SystemExit("prep: inputs missing (see above)")
        # every S name must be new here (the RealMLP average under both of its possible tags)
        nm0 = s_spec.chain_names(fam)  # the chain names do not depend on the RealMLP seeds, which finish only after prep
        S, G = nm0["S"], nm0["G"]
        newnames = ([S["T"], S["TC"], S["TCS"], G["T"], G["TC"], G["TCS"]] + [L[x]["avg"]["new"] for x in s_spec.ORDER]
                    + [s_spec.rm_tag(fam, [0] * n) for n in (3, 4)])
        clash = [p.name for n in newnames for p in list((ROOT / "cache").glob(f"*_{n}_s*.parquet"))
                 if not p.is_symlink() and f"cache/{p.name}" not in self.st["made"]]
        if clash:
            raise SystemExit(f"prep: S names already used in this clone: {clash[:8]}")
        # the family's A and B ends and the reference three-chain end
        ends = {}
        for c, sp in F["ab_state"].items():
            s = json.loads((ROOT / sp).read_text(encoding="utf-8")) if (ROOT / sp).exists() else {"names": {"gja": "?"}, "done": "?"}
            if s.get("done") != "gja" and not DRY:
                raise SystemExit(f"prep: chain {c} of the family is not finished in this clone ({sp})")
            ends[c] = s["names"].get("gja")
        rec["ab_ends"] = ends
        ref = F.get("ref_sbx_manifest") or (f"preds/manifest_{(ROOT / F['ref_sbx_spec_file']).read_text().strip()}.json" if (ROOT / F["ref_sbx_spec_file"]).exists() else "?")
        rec["reference_manifest"] = ref
        if not DRY and not (ROOT / ref).exists():
            raise SystemExit(f"prep: reference manifest missing: {ref}")
        # 598c: R13-F3's CatBoost ap seeds 1 and 2 (val, fwd), cloned from r13f3_run with their PC-recorded sha16
        if fam == "598c":
            rec["cbap_clones"] = {}
            pins = L["cbap"]["pins"]
            for st in ("val", "fwd"):
                for k in (0, 1, 2):
                    for rel in (f"cache/{st}_m6r27cbap_s{k}.parquet", f"preds/es_valid_ids_m6r27cbap_{st}_s{k}.parquet"):
                        dst, src = ROOT / rel, P / "r13f3_run" / rel
                        if k == 0 and rel.startswith("preds/"):
                            continue
                        if not dst.exists():
                            if DRY:
                                print(f"[dry] cp -cp {src} {dst}", flush=True)
                                continue
                            if not src.exists():
                                raise SystemExit(f"prep: {src} missing")
                            subprocess.run(["cp", "-cp", str(src), str(dst)], check=True)
                        if DRY:
                            continue
                        h = sha16(dst)
                        key = Path(rel).stem
                        if key in pins and pins[key] != h:
                            raise SystemExit(f"prep: {rel} sha16 {h} is not the PC copy's {pins[key]} (R13-F3's file)")
                        rec["cbap_clones"][rel] = h
        # feature files against the roots where the family's seeds ran (a pure seed change needs the same features)
        rec["feats"] = {}
        chk = [("cache/feats_v28.parquet", P / "r17a_sbx" / "cache" / "feats_v28.parquet")]
        if fam == "598c":
            chk += [("cache/feats_v27.parquet", P / "r10b_run" / "cache" / "feats_v27.parquet"), ("cache/feats_v27.parquet", P / "r13f3_run" / "cache" / "feats_v27.parquet")]
        for rel, other in chk:
            if DRY:
                print(f"[dry] sha16 {rel} vs {other}", flush=True)
                continue
            a = sha16(ROOT / rel)
            b = sha16(other) if other.exists() else None
            rec["feats"][f"{rel} vs {other}"] = {"here": a, "there": b, "same": a == b}
            if b is not None and a != b:
                raise SystemExit(f"prep: {rel} differs from {other} ({a} vs {b}): the seeds would not be a pure seed change; stop and report")
        # placeholder links for the gate chain's full setting: the LightGBM ap and CatBoost ap averages now, RealMLP before gatechain
        rec["placeholders"] = {}
        rm_new = L["rm"]["avg"]["new"]
        for new, old in F["placeholders"].items():
            if new != rm_new:
                self.link_placeholder(new, old, rec["placeholders"])
        if not DRY:
            (ROOT / FC).mkdir(parents=True, exist_ok=True)
            rec["fitcache"] = {"dir": FC, "files_at_prep": len(list((ROOT / FC).glob("*")))}
            OUT.mkdir(parents=True, exist_ok=True)
            self.save_placeholders(rec["placeholders"])
            (OUT / f"s_prep_{fam}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        self.st["names"]["ab_ends"], self.st["names"]["reference"] = ends, ref
        print(f"[{ts()}] prep done: ends {ends}, reference {ref}", flush=True)

    # ---------------------------------------------------------------- avg
    def avg(self) -> None:
        if not DRY and not (OUT / f"s_learn_{self.fam}_valfwd.DONE").exists():
            raise SystemExit("learners not done (research/f2/out/s_learn_<family>_valfwd.DONE missing)")
        rs = self.rm_seeds()
        for c in s_spec.avg_cmds(self.fam, "val,fwd", rm_seeds=rs or None):
            if c.split()[1] == "rm" and not rs:
                print(f"[{ts()}] RealMLP: fewer than 3 own seeds finished, chain C keeps the family RealMLP; no average", flush=True)
                continue
            _, mode, donor, tag, new, seeds, sets = c.split()[:7]
            sets, ks = sets.split(","), [int(x) for x in seeds.split(",")]
            outs = [f"cache/{s}_{new}_s0.parquet" for s in sets] + [f"preds/es_valid_ids_{new}_{s}_s0.parquet" for s in sets]
            ins = self.inputs([f"cache/{s}_{donor}_s0.parquet" for s in sets] + [f"preds/es_valid_ids_{donor}_{s}_s0.parquet" for s in sets]
                              + [f"cache/{s}_{tag}_s{k}.parquet" for s in sets for k in ks] + [f"preds/es_valid_ids_{tag}_{s}_s{k}.parquet" for s in sets for k in ks])
            if not self.todo(outs, c, ins):
                continue
            run(c.split(), f"f2s_{self.fam}_avg_{new}.log", 3.0, 1, 5, "avg", self.fam)
            self.mark(outs, c, ins)
        run(["research/f2/s_check.py", "avg", self.fam, "val,fwd", "--rm_seeds", ",".join(map(str, rs)) if rs else "none"],
            f"f2s_{self.fam}_check_avg.log", 3.0, 1, 5, "check", self.fam)

    # ---------------------------------------------------------------- blend + corrector + cavg (one name set)
    def blend(self, T: str, settings: list[str], lockp: str) -> None:
        n = self.names()["S"]
        lines = []
        for s in settings:
            cmd = f"src/blend_fixed.py --setting {s} --seed 0 --tag {T} --base {n['base']} --learners {n['learners']} {REFC}"
            lines.append(f"logs/{T}_{s}_s0.log\t.venv/Scripts/python.exe -u {cmd}")
            out = f"cache/{s}_{T}_s0.parquet"
            legs = [n["base"]] + [x.split(":")[0] for x in n["learners"].split(",")] + REF_LEGS  # S legs, and the reference legs the weights come from
            ins = self.inputs([f"cache/{x}_{t}_s0.parquet" for x in sorted({s, "val"}) for t in legs] + self.es(n["base"], s))
            if not self.todo([out], cmd, ins):
                continue
            run(["-u", *cmd.split()], f"f2s_bfix_{T}_{s}_s0.log", 2.5, 1, 5, f"{lockp}bfix", self.fam)
            self.mark([out], cmd, ins)
        jobs_write(T, f"# F2 Package S chain C blend ({self.fam}), settings {settings}", lines)

    def corr(self, T: str, settings: list[str], lockp: str, expect0: bool) -> dict:
        TC = T + s_spec.MID
        lines, fits = [], self.st["fit"].setdefault("final" if lockp == "S" else "gate", {})
        for k in (0, 1, 2):
            for s in ("val", "fwd", "full"):
                cmd = f"src/ctx_corr3.py --setting {s} --seed {k} --base {T} --base_seed 0 --tag {TC} {s_spec.CORR_SBX}"
                lines.append(f"logs/{TC}_{s}_s{k}.log\t.venv/Scripts/python.exe -u {cmd}")
                if s not in settings:
                    continue
                out = f"cache/{s}_{TC}_s{k}.parquet"
                rc_cmd = f"{cmd} --fit_cache {FC}"
                ins = self.inputs([f"cache/{s}_{T}_s0.parquet", f"cache/val_{T}_s0.parquet", f"cache/{CORR_FEATS}.parquet"] + self.es(T, "val") + self.es(T, s))
                if not self.todo([out], rc_cmd, ins):
                    continue
                txt = run(["-u", *rc_cmd.split()], f"f2s_corr_{TC}_{s}_s{k}.log", 4.6, 4, 15, f"{lockp}corr", self.fam)
                m = re.search(r"fit cache \S+: (\d+) loaded, (\d+) fitted", txt)
                fits[f"{s}_s{k}"] = [int(m.group(1)), int(m.group(2))] if m else None
                if not DRY and (m is None or (expect0 and int(m.group(2)) != 0)):
                    self.save()
                    raise SystemExit(f"corrector {TC} {s} s{k}: fit cache {fits[f'{s}_s{k}']} (expected 0 fitted): stop")
                self.mark([out], rc_cmd, ins)
        jobs_write(TC, f"# F2 Package S chain C corrector (scale2 0.75), {self.fam}; settings run here {settings}", lines)
        return fits

    def cavg(self, T: str, settings: list[str], lockp: str) -> None:
        TC = T + s_spec.MID
        lines = []
        for s in ("val", "fwd", "full"):
            cmd = f"research/r10_b/cavg.py --setting {s} --seed 0 --tag {TC}c3 --src {TC} --seeds 0,1,2 --base {T} --base_seed 0"
            lines.append(f"logs/{TC}c3_{s}_s0.log\t.venv/Scripts/python.exe -u {cmd}")
            if s not in settings:
                continue
            out = f"cache/{s}_{TC}c3_s0.parquet"
            ins = self.inputs([f"cache/{s}_{TC}_s{k}.parquet" for k in (0, 1, 2)] + [f"cache/{s}_{T}_s0.parquet"] + self.es(TC, s) + self.es(T, s))
            if not self.todo([out], cmd, ins):
                continue
            run(["-u", *cmd.split()], f"f2s_cavg_{TC}c3_{s}_s0.log", 3.1, 1, 5, f"{lockp}cavg", self.fam)
            self.mark([out], cmd, ins)
        jobs_write(TC + "c3", f"# F2 Package S chain C corrector seed average; the seed runs are notes/jobs_{TC}.txt", lines)

    def final(self) -> None:
        T = self.names()["S"]["T"]
        self.blend(T, ["val", "fwd"], "S")
        self.st["fit"]["final"] = self.corr(T, ["val", "fwd"], "S", expect0=False)
        self.save()
        self.cavg(T, ["val", "fwd"], "S")

    # ---------------------------------------------------------------- gate chain
    def same_files(self, a: str, b: str) -> dict:
        if DRY:
            return {}
        x, y = pd.read_parquet(ROOT / a), pd.read_parquet(ROOT / b)
        bad = avg_s.frames_equal(x, y)
        return {"a": a, "b": b, "equal": not bad, "differences": bad}

    def rm_placeholder(self) -> None:
        """The RealMLP placeholder from the seeds that finished: full_<rm_tag(seeds)>_s0 -> the family RealMLP's full file; none when chain C
        keeps the family RealMLP (its own full file is real)."""
        rs = self.rm_seeds()
        links = {}
        if len(rs) >= 3:
            self.link_placeholder(s_spec.rm_tag(self.fam, rs), s_spec.fam_rm(self.fam), links)
        else:
            print(f"[{ts()}] RealMLP placeholder: none (chain C keeps the family RealMLP {s_spec.fam_rm(self.fam)}, its full file is real)", flush=True)
        self.save_placeholders(links)
        self.st["rm_placeholder"] = {"seeds": rs, "links": links}
        self.save()

    def gatechain(self) -> None:
        nm = self.names()
        S, G = nm["S"], nm["G"]
        F = self.F
        self.rm_placeholder()
        rep = self.st.setdefault("repro", {})

        def stop_if_differs(stage: str) -> None:
            self.save()
            bad = [k for k, v in rep.items() if v and not v["equal"]]
            if bad:
                raise SystemExit(f"gate-name val/fwd files differ from the final-name ones after {stage}: {bad}")
        self.blend(G["T"], ["val", "fwd", "full"], "G")
        for s in ("val", "fwd"):
            rep[f"blend_{s}"] = self.same_files(f"cache/{s}_{G['T']}_s0.parquet", f"cache/{s}_{S['T']}_s0.parquet")
        stop_if_differs("the blend")
        self.st["fit"]["gate"] = self.corr(G["T"], ["val", "fwd", "full"], "G", expect0=True)
        for s in ("val", "fwd"):
            for k in (0, 1, 2):
                rep[f"corr_{s}_s{k}"] = self.same_files(f"cache/{s}_{G['TC']}_s{k}.parquet", f"cache/{s}_{S['TC']}_s{k}.parquet")
                rep[f"delta_{s}_s{k}"] = self.same_files(f"notes/ctx_corr_{G['TC']}_{s}_s{k}_delta.parquet", f"notes/ctx_corr_{S['TC']}_{s}_s{k}_delta.parquet")
        stop_if_differs("the correctors")
        self.cavg(G["T"], ["val", "fwd", "full"], "G")
        for s in ("val", "fwd"):
            rep[f"c3_{s}"] = self.same_files(f"cache/{s}_{G['TCS']}_s0.parquet", f"cache/{s}_{S['TCS']}_s0.parquet")
        stop_if_differs("cavg")
        names = self.st["names"].setdefault("gate_chain", {})
        # manifest with the family chain C's code list plus the S averager
        man = f"preds/manifest_{G['TCS']}.json"
        fm = json.loads((ROOT / "preds" / f"manifest_{F['fam_C']['sbx_TCS']}.json").read_text(encoding="utf-8")) if not DRY else {"code_sha": {}}
        sys.path.insert(0, str(ROOT))
        os.environ.pop("MANIFEST_EXTRA_CODE", None)
        base_code = set()
        if not DRY:
            from src import manifest as M  # noqa: E402
            base_code = set(M.CODE)
        extra = ",".join([k for k in fm["code_sha"] if k not in base_code] + ["research/f2/avg_s.py"])
        margs = ["-m", "src.manifest", G["TCS"], G["TCS"], "pred", "0", F["feats"], f"notes/jobs_{G['TCS']}.txt"]
        mcmd = " ".join(margs) + f" (env MANIFEST_EXTRA_CODE={extra})"
        ins = self.inputs([f"cache/{s}_{G['TCS']}_s0.parquet" for s in ("val", "fwd", "full")] + self.es(G["TCS"], "val") + [f"notes/jobs_{G['TCS']}.txt"])
        if self.todo([man], mcmd, ins):
            run(margs, f"f2s_manifest_{G['TCS']}.log", 2.5, 1, 5, "Gmanifest", self.fam, {"MANIFEST_EXTRA_CODE": extra})
            self.mark([man], mcmd, ins)
        num0 = 3000 if self.fam == "598c" else 3100
        names["corr"] = self.qentry(str(num0), G["TCS"])
        prevq, prev = names["corr"], G["TCS"]
        for i, t in enumerate(s_spec.TAILS, start=1):
            spec = G["tails"][t]
            n = str(num0 + i)
            man = f"preds/manifest_{spec}.json"
            srcm = f"preds/manifest_{prev}.json"
            args = {"am": ["tools/artmm.py", "build", srcm, n, "--no-enqueue"],
                    "fh": ["tools/arthist.py", "build", srcm, n, "--no-enqueue"],
                    "gkc": ["tools/gkey.py", "build", prevq, n, "--no-enqueue", "--variant", "avg", "--leg", F["GLEG"], "--leg_feats", F["feats"]],
                    "gkcx": ["tools/gkeyx.py", "build", srcm, n, "--no-enqueue"],
                    "xncr": ["tools/gkeyxn.py", "build", srcm, n, "--no-enqueue", "--variant", "avg_xncr"],
                    "db": ["src/aiu_shift.py", "build", prevq, n, "--group", "block"],
                    "aw": ["tools/antiwin.py", "build", prevq, n, "--no-enqueue"],
                    "gp": ["tools/gpool.py", "build", prevq, n, "--no-enqueue"],
                    "gl": ["tools/glong.py", "build", srcm, n, "--no-enqueue"],
                    "gja": ["tools/gjoint.py", "build", srcm, n, "--mode", "a", "--no-enqueue"]}[t]
            tcmd = " ".join(args)
            ins = self.inputs(self.man_files(prev) + [f"queue/{prevq}.json", f"queue/{prevq}.parquet"])
            if self.todo([man], tcmd, ins):  # absent: build; made by this driver from the same command and inputs: reuse; else refused
                run(args, f"f2s_{t}_{n}_{spec[-40:]}.log", 3.0, 1, 10, f"G{t}", self.fam)
                if not DRY and not (ROOT / man).exists():
                    raise SystemExit(f"{t}: no manifest {spec}")
                self.mark([man], tcmd, ins)
            names[t] = self.qentry(n, spec)
            prevq, prev = names[t], spec
            self.save()
        ends = self.st["names"].get("ab_ends") or {"A": "?", "B": "?"}
        n = str(num0 + 60)
        a_spec = ends["A"].split("_", 1)[1] if "_" in ends["A"] else "?"
        spec = a_spec + "av" + s_spec.SUFFIX
        man = f"preds/manifest_{spec}.json"
        aargs = ["tools/chainavg3.py", "build", ends["A"], ends["B"], names["gja"], n, "--suffix", s_spec.SUFFIX, "--no-enqueue"]
        ins = self.inputs([x for e in (ends["A"], ends["B"], names["gja"]) for x in self.q_files(e)])
        if self.todo([man], " ".join(aargs), ins):
            run(aargs, f"f2s_avg3_{n}.log", 2.5, 1, 5, "Gavg3", self.fam)
            if not DRY and not (ROOT / man).exists():
                raise SystemExit(f"avg3: no manifest {spec}")
            self.mark([man], " ".join(aargs), ins)
        names["avg3"] = self.qentry(n, spec)
        names["avg3_spec"] = spec
        self.save()
        print(f"[{ts()}] gate chain end {spec}", flush=True)

    def qentry(self, num: str, spec: str) -> str:
        name = f"{num}_{spec}"
        if DRY:
            return name
        q = ROOT / "queue" / f"{name}.json"
        m = json.loads((ROOT / "preds" / f"manifest_{spec}.json").read_text(encoding="utf-8"))
        outs, cmd = [f"queue/{name}.json", f"queue/{name}.parquet"], f"queue entry {name} (GATE_ONLY_S) from preds/manifest_{spec}.json"
        ins = self.inputs([f"preds/manifest_{spec}.json", m["test"]["path"]])
        if self.todo(outs, cmd, ins):
            q.write_text(json.dumps({"manifest": f"preds/manifest_{spec}.json", "kind": "GATE_ONLY_S", "sandbox": True,
                                     "note": "F2 Package S gate-only chain, full from placeholder legs; never stage"}, indent=1), encoding="utf-8")
            shutil.copy2(ROOT / m["test"]["path"], ROOT / "queue" / f"{name}.parquet")
            self.mark(outs, cmd, ins)
        return name

    # ---------------------------------------------------------------- gate
    def gate(self) -> None:
        nm = self.names()
        rs = self.rm_seeds()
        rsa = ["--rm_seeds", ",".join(map(str, rs)) if rs else "none"]
        G, S = nm["G"], nm["S"]
        outs = []
        if not DRY:
            (OUT / f"s_export_{self.fam}.json").unlink(missing_ok=True)  # 03:5x: an export never outlives the gate run after it
        steps = [(["research/f2/s_check.py", "rows", self.fam, "val,fwd"] + rsa, f"s_check_rows_{self.fam}_valfwd.json"),
                 (["research/f2/s_check.py", "cmds", self.fam, "val,fwd", str(self.learn.relative_to(ROOT))] + rsa, f"s_check_cmds_{self.fam}_valfwd.json"),
                 (["research/f2/s_check.py", "avg", self.fam, "val,fwd"] + rsa, f"s_check_avg_{self.fam}_valfwd.json"),
                 (["research/f2/s_check.py", "blend", self.fam, S["T"], self.fam_sbx_T(), "val,fwd"] + rsa, f"s_check_blend_{S['T']}.json"),
                 (["research/f2/s_check.py", "manifest", self.fam, f"preds/manifest_{G['TCS']}.json", f"preds/manifest_{self.F['fam_C']['sbx_TCS']}.json",
                   f"{G['T']}={self.fam_sbx_T()}"], f"s_check_manifest_{G['TCS'][:60]}.json")]
        for args, js in steps:
            if not DRY:
                (ROOT / "research" / "f2" / "out" / js).unlink(missing_ok=True)  # a stale record of an earlier run never counts
            try:
                run(args, f"f2s_{self.fam}_{args[1]}.log", 3.0, 1, 5, "check", self.fam)
            except SystemExit as e:
                print(f"[{ts()}] {args[1]}: {e}", flush=True)
            outs.append(f"research/f2/out/{js}")
        spec = self.st["names"].get("gate_chain", {}).get("avg3_spec", "?")
        gj = f"research/f2/out/s_gate_sbx_{self.fam}.json"
        vj = f"research/f2/out/s_verdict_sbx_{self.fam}.json"
        if not DRY:
            (ROOT / gj).unlink(missing_ok=True)
            (ROOT / vj).unlink(missing_ok=True)
        try:
            run(["research/f2/s_gate.py", "sbx", f"preds/manifest_{spec}.json", self.st["names"].get("reference", "?"), gj], f"f2s_{self.fam}_gate.log",
                3.0, 1, 10, "gate", self.fam)
        except SystemExit as e:
            print(f"[{ts()}] s_gate: {e}", flush=True)
        outs.append(gj)
        missing = [o for o in outs if not DRY and not (ROOT / o).exists()]
        try:
            run(["research/f2/s_gate.py", "verdict", vj] + [o for o in outs if DRY or (ROOT / o).exists()], f"f2s_{self.fam}_verdict.log", 1.0, 1, 2,
                "verdict", self.fam)
        except SystemExit as e:
            print(f"[{ts()}] verdict: {e}", flush=True)
        if DRY:
            return
        v = json.loads((ROOT / vj).read_text(encoding="utf-8")) if (ROOT / vj).exists() else {"pass": False, "inputs": {}}
        ok = bool(v.get("pass")) and not missing
        self.st["verdict"] = {"pass": ok, "inputs": v.get("inputs"), "missing_records": missing}
        self.save()
        print(f"[{ts()}] SANDBOX VERDICT (rules 1, 2, 3a) {self.fam}: {'PASS' if ok else 'FAIL'} {v.get('inputs')}"
              + (f" missing records {missing}" if missing else ""), flush=True)
        if not ok:
            self.failed.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} gate FAIL: {v.get('inputs')} missing {missing} "
                                   f"(numbers kept in {gj} and {vj})\n", encoding="utf-8")
            raise SystemExit(f"sandbox verdict FAIL for {self.fam}: export skipped (research/f2/out/{self.failed.name})")
        self.failed.unlink(missing_ok=True)

    # ---------------------------------------------------------------- export list for F3
    def export(self) -> None:
        F, nm = self.F, self.names()
        S, L = nm["S"], F["legs"]
        vj = OUT / f"s_verdict_sbx_{self.fam}.json"
        if not DRY:
            v = json.loads(vj.read_text(encoding="utf-8")) if vj.exists() else {}
            if not v.get("pass") or not self.st.get("verdict", {}).get("pass") or self.failed.exists():
                raise SystemExit(f"export refused: the sandbox verdict is not PASS ({vj.name}, {self.failed.name})")
        rs = self.rm_seeds()
        files = []
        for st in ("val", "fwd"):
            for leg in s_spec.ORDER:
                tag = L[leg]["tag"]
                seeds = (rs if leg == "rm" else L[leg]["seeds"]) if (leg != "rm" or rs) else []
                for k in seeds:
                    files += [f"cache/{st}_{tag}_s{k}.parquet", f"preds/es_valid_ids_{tag}_{st}_s{k}.parquet"]
                new = (s_spec.rm_tag(self.fam, rs) if leg == "rm" else L[leg]["avg"]["new"]) if (leg != "rm" or rs) else None
                if new:
                    files += [f"cache/{st}_{new}_s0.parquet", f"preds/es_valid_ids_{new}_{st}_s0.parquet"]
            files += [f"cache/{st}_{S['T']}_s0.parquet", f"preds/es_valid_ids_{S['T']}_{st}_s0.parquet", f"notes/blend_fixed_{S['T']}_{st}_s0.json"]
            for k in (0, 1, 2):
                files += [f"cache/{st}_{S['TC']}_s{k}.parquet", f"preds/es_valid_ids_{S['TC']}_{st}_s{k}.parquet"]
            files += [f"cache/{st}_{S['TCS']}_s0.parquet", f"preds/es_valid_ids_{S['TCS']}_{st}_s0.parquet"]
        files += ["research/f2/avg_s.py", "research/f2/s_spec.py", "research/f2/s_check.py", "research/f2/s_gate.py", "research/f2/fj_blend_s.py",
                  "research/f2/s_learn.py", "research/f2/rm_smoke.py"]
        rec = {"family": self.fam, "from": str(ROOT), "rule": "clone each file with cp -cp, check sha16, refuse to overwrite; never a full cache; never a link",
               "files": {}, "fitcache": {}, "names": nm, "rm_seeds": rs, "verdict": str(vj.relative_to(ROOT)) if not DRY else "?"}
        miss = []
        for f in files:
            p = ROOT / f
            if not p.exists() or p.is_symlink():
                miss.append(f)
                continue
            rec["files"][f] = sha16(p) if not DRY else "?"
        fc = sorted((ROOT / FC).glob("*")) if (ROOT / FC).exists() else []
        rec["fitcache"] = {"dir": FC, "files": len(fc), "to": [f"research/f3/fitcache_s{self.fam}", f"research/f2/fitcache_4m_s{self.fam}"],
                           "expect_on_final_root": "T corrector full: 0 fitted per seed; four-month 0.75: 10 fitted per seed; 0.5: 0; twin: 0"}
        rec["missing"] = miss
        if not DRY:
            (OUT / f"s_export_{self.fam}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[{ts()}] export: {len(rec['files'])} files, {len(fc)} fit-cache files, missing {miss}", flush=True)
        if miss and not DRY:
            raise SystemExit("export: files missing")


def do_step(ch: Chain, step: str) -> None:
    """One step, with the checks every step makes: its family's clone, no standing FAIL (research/f2/out/s_chain_<family>.FAILED: nothing
    runs until we clear it by hand), and every earlier step done."""
    if step not in STEPS:
        raise SystemExit(__doc__)
    check_root(ch.F)
    if not DRY and ch.failed.exists():
        raise SystemExit(f"refused: {ch.failed.name} exists ({ch.failed.read_text(encoding='utf-8', errors='replace').strip()[:240]}); "
                         "a FAIL stands until we clear it")
    if step in ch.st["done"] and not DRY:
        print(f"[{ts()}] done earlier: {step}", flush=True)
        return
    before = [s for s in STEPS[: STEPS.index(step)] if s not in ch.st["done"]]
    if before and not DRY:
        raise SystemExit(f"refused: {step} before {before}")
    if not DRY and STEPS.index(step) < STEPS.index("gate"):  # the records of an earlier gate no longer describe the chain
        for f in (f"s_export_{ch.fam}.json", f"s_verdict_sbx_{ch.fam}.json", f"s_gate_sbx_{ch.fam}.json"):
            (OUT / f).unlink(missing_ok=True)
        ch.st.pop("verdict", None)
    t0 = time.time()
    print(f"[{ts()}] step {step} ({ch.fam})" + (" dry" if DRY else ""), flush=True)
    getattr(ch, step)()
    if not DRY:
        ch.st["done"].append(step)
        ch.st.setdefault("times", {})[step] = round(time.time() - t0)
        ch.save()


def main() -> int:
    global DRY
    a = sys.argv[1:]
    if len(a) < 2:
        raise SystemExit(__doc__)
    DRY = "--dry" in a
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ch = Chain(a[0])
    for step in (STEPS if a[1] == "all" else [a[1]]):
        do_step(ch, step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
