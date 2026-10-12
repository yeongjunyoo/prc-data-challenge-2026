# SPDX-License-Identifier: GPL-3.0-only
"""R17-A L4 gate-only chains in the APFS clone <mac-workdir>/prcm/r17a_l4g_sbx (private notes research/2026-10-08-r17-a-adsb-fill.md section 8;
10-10 10:2x: placeholders only in this clone, every chain tag carries "l4g", nothing from here is staged or copied to queue/,
final_run or the PC's preds/, the placeholder map goes into the note and the gate json, the gate reads only real-L4 settings).
src.manifest and every tail tool need val, fwd and full; L4 has only val (phase v) or val and fwd (phase f). The missing settings of the five
L4 learner inputs of the blends are links to 514's learner files (the inputs L3 used), so those settings reproduce L3's chain exactly:
  m6r28cbwinx3 -> m6r27cbwinx3, m6r28cbx3 -> m6r27cbx3, m6r28cbap -> m6r27cbap, m6r28rmwinx4 -> m6r27rmwinx4, m6r28rmx4 -> m6r27rmx4
  (cache/<S>_<L4>_s0.parquet -> <S>_<514>_s0.parquet, and preds/es_valid_ids_<L4>_fwd_s0.parquet -> es_valid_ids_<514>_fwd_s0.parquet)
The real settings are the public root's files, cloned. Chain names: m6r28l4g<phase>x... with phase v or f; dropping "l4g<phase>" gives F3's
names. Steps per chain and scale (0.75 tag s75, 0.5 tag ''), all three settings: bfix and blend (by the 0.75 builder; the 0.5 builder waits
for them), ctx_corr3 seeds 0, 1, 2 with --fit_cache on the PUBLIC root's fitcache (absolute path; real settings load the public root's fits,
placeholder settings load L3's), cavg, src.manifest (feats_v28), tails artmm .. gjoint a as L3 (gkey --leg_feats feats_v28), then
tools/chainavg3.py skap per scale and research/r17_a/s30_gate_l4.py on the real settings: the 0.75 end against 570 (registered) and the
0.5 end against 571 (information).
Subcommands (from the clone root):
  prep --phase v|f                        links and research/r17_a/l4g_placeholders_<phase>.json
  chain --phase v|f --chain A|B|C --scale 0.75|0.5
  avg3 --phase v|f --scale 0.75|0.5
  gate --phase v|f
PROTOCOL per step as research/r17_a/sb_build_r17a.py (lock prcm/r17a_l4g_<name>.lock, two checks 30 s apart, threads <= 12, free >= 30 %).
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

ROOT = Path(__file__).resolve().parents[2]
P = Path(os.environ.get("PRC_WORK", "."))  # Mac work folder that holds the build sandboxes
PY = sys.executable
LANE = ROOT / "research" / "r17_a"
FITCACHE = str(P / "r17a_sbx" / "research" / "r13_l" / "fitcache")
if ROOT.name != "r17a_l4g_sbx":
    raise SystemExit(f"this builder runs only in the l4g clone, not in {ROOT}")
CORR = ("--feats feats_v25 --adsb adsb_v1nd --adsb_sha 0d03ff06267e0dee --folds 10 --rounds 300 --rounds2 400 --max_pred 3600 --threads 4 "
        "--geo geo_v1nd --geo_sha 43310b9bbeefe81b --ctx ctx_v3 --ctx_sha 0b139ddbd6e70573 --scale2 {sc}")
REF = "--ref_base m6r25sg --ref_tag m6r25sgoof4 --ref_learners m6r25cb:mu_cat,m6r25xg:mu_xgb,m6r25rm:mu_rmlp --folds 10 --check"
MID = "adsbgeondctxv3x50w"
PLACE = {"m6r28cbwinx3": "m6r27cbwinx3", "m6r28cbx3": "m6r27cbx3", "m6r28cbap": "m6r27cbap", "m6r28rmwinx4": "m6r27rmwinx4", "m6r28rmx4": "m6r27rmx4"}
REAL = {"v": ("val",), "f": ("val", "fwd")}
STEPS = ["bfix", "blend", "corr", "cavg", "manifest", "am", "fh", "gkc", "gkcx", "xncr", "db", "aw", "gp", "gl", "gja"]
SUF = {"am": "am", "fh": "amfh", "gkc": "amfhgkc", "gkcx": "amfhgkcx", "xncr": "amfhgkcxncr", "db": "amfhgkcxncrdb", "aw": "amfhgkcxncrdbaw",
       "gp": "amfhgkcxncrdbawgp", "gl": "amfhgkcxncrdbawgpgl", "gja": "amfhgkcxncrdbawgpglgja"}
EXTRA = ("src/blend_fixed.py,src/ctx_corr2.py,src/ctx_corr3.py,src/feats_ctx2.py,src/feats_aiu.py,research/ctx_corr/freeze_geo.py,research/obs2/corr_plus.py,"
         "research/obs2/obs_rows.py,research/obs2/coverage.py,research/obs2/osm_stands.py,research/adsb/gprobe/build_nd.py,"
         "research/r13_l/prep_x.py,research/r13_l/tabm_leg.py,research/r13_l/xrfm_leg.py,research/r13_l/blend_add.py,research/r10_b/cavg.py,"
         "research/r17_a/avg_r17a.py,research/r17_a/avg_l4.py,research/r17_a/adsb_day_r17a.py,research/adsb/build_feats2.py,research/adsb/adsb_match.py,"
         "research/r13_f3/model3ap.py,research/r13_f3/cat_muap.py")
REF570 = "preds/manifest_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.json"
REF571 = "preds/manifest_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgjaavskap.json"
sys.stdout.reconfigure(line_buffering=True)


def ts() -> str:
    return time.strftime("%H:%M:%S")


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def chains(phase: str) -> dict:
    x = f"m6r28l4g{phase}x"
    return {"A": {"BF": x + "wincbrmfw4sv434", "BASE": "m6r28skwingx4", "LEARN": "m6r28cbwinx3:mu_cat,m6r25xg:mu_xgb,m6r28rmwinx4:mu_rmlp",
                  "T": x + "wincbrmfw4sv434tmc", "TABM": "m6r27tmcwin:mu_tabm", "GLEG": "m6r28skwing"},
            "B": {"BF": x + "skcbrmfw4sv434", "BASE": "m6r28skgx4", "LEARN": "m6r28cbx3:mu_cat,m6r25xg:mu_xgb,m6r28rmx4:mu_rmlp",
                  "T": x + "skcbrmfw4sv434", "TABM": None, "GLEG": "m6r28skg"},
            "C": {"BF": x + "apcbrmfw4sv114", "BASE": "m6r28skapg", "LEARN": "m6r28cbap:mu_cat,m6r25xg:mu_xgb,m6r28rmx4:mu_rmlp",
                  "T": x + "apcbrmfw4sv114", "TABM": None, "GLEG": "m6r28skapg"}}


def num0(phase: str, ch: str, stag: str) -> int:
    return {"v": 2000, "f": 2200}[phase] + (0 if stag else 100) + {"A": 0, "B": 20, "C": 40}[ch]


def budget_ok(thr: int) -> bool:
    if (P / "arc_quiet.request").exists() or (P / "arc_quiet.lock").exists():
        return False
    st = 0.0
    for d in P.glob("*.lock"):
        if d.is_dir():
            txt = (d / "info").read_text(errors="replace") if (d / "info").exists() else ""
            t = re.search(r"(?im)^threads[^0-9]*([0-9]+(?:\.[0-9]+)?)", txt)
            st += float(t.group(1)) if t else 6
    if st + thr > (13 if thr == 1 else 12):
        return False
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.strip().splitlines()
    m = re.search(r"([0-9]+)%", out[-1]) if out else None
    return bool(m) and int(m.group(1)) >= 30


def run(args: list[str], log: str, peak: float, thr: int, mins: int, lockname: str, env: dict | None = None) -> str:
    ok = 0
    while ok < 2:
        ok = ok + 1 if budget_ok(thr) else 0
        if ok < 2:
            time.sleep(30)
    lock = P / f"r17a_l4g_{lockname}.lock"
    lock.mkdir()
    try:
        (lock / "info").write_text(f"owner: PRC R17-A L4 gate-only chain in r17a_l4g_sbx ({lockname})\npid: {os.getpid()}\npeak_gb: {peak}\n"
                                   f"threads: {thr}\nstart: {time.strftime('%Y-%m-%d %H:%M')}\n"
                                   f"expected_end: {time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time() + 60 * mins))} KST\n"
                                   f"job: {' '.join(args)[:200]}\n")
        print(f"[{ts()}] run {' '.join(args)[:230]}", flush=True)
        e = {**os.environ, "OMP_NUM_THREADS": str(thr), "OPENBLAS_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8", **(env or {})}
        with open(ROOT / "logs" / log, "w", encoding="utf-8") as f:
            r = subprocess.run(["/usr/bin/time", "-l", "nice", "-n", "10", PY, "-X", "utf8", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=e)
    finally:
        shutil.rmtree(lock, ignore_errors=True)
    txt = (ROOT / "logs" / log).read_text(encoding="utf-8", errors="replace")
    rl = re.search(r"([0-9.]+)\s+real", txt)
    fc = re.search(r"fit cache \S+: (\d+) loaded, (\d+) fitted", txt)
    print(f"[{ts()}]   {log}: rc {r.returncode}" + (f" | real {float(rl.group(1)):.0f} s" if rl else "")
          + (f" | fit cache {fc.group(1)} loaded, {fc.group(2)} fitted" if fc else ""), flush=True)
    if r.returncode != 0:
        raise SystemExit(f"step failed: {log}")
    return txt


def must_be_new(*rels: str) -> None:
    for rel in rels:
        if (ROOT / rel).exists():
            raise SystemExit(f"target already exists, refusing: {rel}")


def qentry(num: str, spec: str) -> str:
    name = f"{num}_{spec}"
    if (ROOT / "queue" / f"{name}.json").exists():
        raise SystemExit(f"sandbox queue entry exists, refusing: {name}")
    m = json.loads((ROOT / "preds" / f"manifest_{spec}.json").read_text(encoding="utf-8"))
    (ROOT / "queue" / f"{name}.json").write_text(json.dumps({"manifest": f"preds/manifest_{spec}.json", "kind": "GATE_ONLY_L4G", "sandbox": True,
                                                             "note": "R17-A gate-only chain with placeholder settings; never stage"}, indent=1), encoding="utf-8")
    shutil.copy2(ROOT / m["test"]["path"], ROOT / "queue" / f"{name}.parquet")
    return name


def state_path(phase: str, ch: str, stag: str) -> Path:
    return LANE / f"l4g_state_{phase}_{ch}{stag or 'h'}.json"


def load_state(phase: str, ch: str, stag: str) -> dict:
    p = state_path(phase, ch, stag)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"names": {}, "done": None, "times": {}}


def save_state(phase: str, ch: str, stag: str, st: dict) -> None:
    state_path(phase, ch, stag).write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def jobs_write(tag: str, header: str, lines: list[str]) -> None:
    (ROOT / "notes" / f"jobs_{tag}.txt").write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def prep(phase: str) -> int:
    links = {}
    for st in ("val", "fwd", "full"):
        if st in REAL[phase]:
            for l4 in PLACE:
                if not (ROOT / "cache" / f"{st}_{l4}_s0.parquet").exists() or (ROOT / "cache" / f"{st}_{l4}_s0.parquet").is_symlink():
                    raise SystemExit(f"real setting {st}: cache/{st}_{l4}_s0.parquet missing or a link")
            continue
        for l4, old in PLACE.items():
            pairs = [(f"cache/{st}_{l4}_s0.parquet", f"{st}_{old}_s0.parquet")]
            if st == "fwd":
                pairs.append((f"preds/es_valid_ids_{l4}_fwd_s0.parquet", f"es_valid_ids_{old}_fwd_s0.parquet"))
            for rel, tgt in pairs:
                p = ROOT / rel
                if p.is_symlink():
                    if os.readlink(p) != tgt:
                        raise SystemExit(f"{rel} links elsewhere: {os.readlink(p)}")
                elif p.exists():
                    raise SystemExit(f"{rel} exists and is not a link (a real file); refusing")
                else:
                    os.symlink(tgt, p)
                links[rel] = {"target": str(p.parent.relative_to(ROOT) / tgt), "target_sha16": sha16(p)}
    m = {"phase": phase, "real_settings": list(REAL[phase]), "placeholder_links": links,
         "rule": "missing settings of the L4 learner inputs link to 514's learner files (the inputs L3 used), so those settings reproduce L3's chain",
         "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    (LANE / f"l4g_placeholders_{phase}.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(json.dumps(m, indent=1), flush=True)
    return 0


def wait_for(phase: str, ch: str, stag: str, step: str) -> None:
    while True:
        d = load_state(phase, ch, stag)["done"]
        if d and STEPS.index(d) >= STEPS.index(step):
            return
        time.sleep(20)


def chain(phase: str, ch: str, sc: str) -> int:
    stag = "s75" if sc == "0.75" else ""
    c = chains(phase)[ch]
    T, BF = c["T"], c["BF"]
    TC = T + MID + stag
    TCS = TC + "c3"
    st = load_state(phase, ch, stag)
    steps = [s for s in STEPS if not (s == "blend" and c["TABM"] is None)]
    if stag == "":
        steps = [s for s in steps if s not in ("bfix", "blend")]
        wait_for(phase, ch, "s75", "blend" if c["TABM"] else "bfix")
    start = steps[steps.index(st["done"]) + 1] if st["done"] in steps else steps[0]
    num = lambda k: str(num0(phase, ch, stag) + STEPS.index(k))  # noqa: E731
    for step in steps[steps.index(start):]:
        t0 = time.time()
        if step == "bfix":
            lines = []
            for s in ("val", "fwd", "full"):
                must_be_new(f"cache/{s}_{BF}_s0.parquet")
                cmd = f"src/blend_fixed.py --setting {s} --seed 0 --tag {BF} --base {c['BASE']} --learners {c['LEARN']} {REF}"
                run(["-u", *cmd.split()], f"bfix_{BF}_{s}_s0.log", 2.0, 1, 5, f"{phase}{ch}bfix")
                lines.append(f"logs/{BF}_{s}_s0.log\t.venv/Scripts/python.exe -u {cmd}")
            jobs_write(BF, f"# R17-A L4 gate-only base blend in r17a_l4g_sbx (phase {phase}; placeholder settings per research/r17_a/l4g_placeholders_{phase}.json)", lines)
        elif step == "blend":
            lines = []
            for s in ("val", "fwd", "full"):
                must_be_new(f"cache/{s}_{T}_s0.parquet")
                cmd = f"research/r13_l/blend_add.py --setting {s} --tag {T} --src {BF} --leg {c['TABM']} --folds 10 --seed 0"
                run(["-u", *cmd.split()], f"blend_{T}_{s}_s0.log", 3.0, 1, 5, f"{phase}{ch}blend")
                lines.append(f"logs/{T}_{s}_s0.log\t.venv/Scripts/python.exe -u {cmd}")
            jobs_write(T, f"# R17-A L4 gate-only blend ({BF} + TabM {c['TABM']}), phase {phase}", lines)
        elif step == "corr":
            lines, fc = [], {}
            for k in (0, 1, 2):
                for s in ("val", "fwd", "full"):
                    cmd = f"src/ctx_corr3.py --setting {s} --seed {k} --base {T} --base_seed 0 --tag {TC} {CORR.format(sc=sc)}"
                    lines.append(f"logs/{TC}_{s}_s{k}.log\t.venv/Scripts/python.exe -u {cmd}")
                    if (ROOT / f"cache/{s}_{TC}_s{k}.parquet").exists():
                        continue
                    txt = run(["-u", *cmd.split(), "--fit_cache", FITCACHE], f"corr_{TC}_{s}_s{k}.log", 4.6, 4, 15, f"{phase}{ch}{stag}corr")
                    m = re.search(r"fit cache \S+: (\d+) loaded, (\d+) fitted", txt)
                    fc[f"{s}_s{k}"] = [int(m.group(1)), int(m.group(2))] if m else None
            st["fit_cache"] = fc
            jobs_write(TC, f"# R17-A L4 gate-only corrector (scale2 {sc}), phase {phase}", lines)
        elif step == "cavg":
            lines = []
            for s in ("val", "fwd", "full"):
                must_be_new(f"cache/{s}_{TCS}_s0.parquet")
                cmd = f"research/r10_b/cavg.py --setting {s} --seed 0 --tag {TCS} --src {TC} --seeds 0,1,2 --base {T} --base_seed 0"
                run(["-u", *cmd.split()], f"cavg_{TCS}_{s}_s0.log", 2.0, 1, 5, f"{phase}{ch}{stag}cavg")
                lines.append(f"logs/{TCS}_{s}_s0.log\t.venv/Scripts/python.exe -u {cmd}")
            jobs_write(TCS, f"# R17-A L4 gate-only corrector seed average; the seed runs are notes/jobs_{TC}.txt", lines)
        elif step == "manifest":
            must_be_new(f"preds/manifest_{TCS}.json", f"preds/test_{TCS}.parquet", f"preds/fwd_{TCS}.parquet")
            run(["-m", "src.manifest", TCS, TCS, "pred", "0", "feats_v28", f"notes/jobs_{TCS}.txt"], f"manifest_{TCS}.log", 2.0, 1, 5,
                f"{phase}{ch}{stag}manifest", {"MANIFEST_EXTRA_CODE": EXTRA})
            st["names"]["corr"] = qentry(num("manifest"), TCS)
        else:
            prev = TCS if step == "am" else TCS + SUF[STEPS[STEPS.index(step) - 1]]
            spec = TCS + SUF[step]
            must_be_new(f"preds/manifest_{spec}.json", f"preds/test_{spec}.parquet", f"preds/fwd_{spec}.parquet")
            src_q = st["names"]["corr" if step == "am" else STEPS[STEPS.index(step) - 1]]
            srcm = f"preds/manifest_{prev}.json"
            n = num(step)
            args = {"am": ["tools/artmm.py", "build", srcm, n, "--no-enqueue"],
                    "fh": ["tools/arthist.py", "build", srcm, n, "--no-enqueue"],
                    "gkc": ["tools/gkey.py", "build", src_q, n, "--no-enqueue", "--variant", "avg", "--leg", c["GLEG"], "--leg_feats", "feats_v28"],
                    "gkcx": ["tools/gkeyx.py", "build", srcm, n, "--no-enqueue"],
                    "xncr": ["tools/gkeyxn.py", "build", srcm, n, "--no-enqueue", "--variant", "avg_xncr"],
                    "db": ["src/aiu_shift.py", "build", src_q, n, "--group", "block"],
                    "aw": ["tools/antiwin.py", "build", src_q, n, "--no-enqueue"],
                    "gp": ["tools/gpool.py", "build", src_q, n, "--no-enqueue"],
                    "gl": ["tools/glong.py", "build", srcm, n, "--no-enqueue"],
                    "gja": ["tools/gjoint.py", "build", srcm, n, "--mode", "a", "--no-enqueue"]}[step]
            run(args, f"{step}_{n}_{spec}.log", 3.0, 1, 10, f"{phase}{ch}{stag}tail")
            if not (ROOT / "preds" / f"manifest_{spec}.json").exists():
                raise SystemExit(f"{step}: no manifest {spec}")
            st["names"][step] = qentry(n, spec)
        st["done"], st["times"][step] = step, round(time.time() - t0)
        save_state(phase, ch, stag, st)
        print(f"[{ts()}] {phase} {ch} {stag or '0.5'} {step} done ({round(time.time() - t0)} s)", flush=True)
    return 0


def avg3(phase: str, sc: str) -> int:
    stag = "s75" if sc == "0.75" else ""
    ends = {}
    for ch in "ABC":
        st = load_state(phase, ch, stag)
        while st["done"] != "gja":
            time.sleep(30)
            st = load_state(phase, ch, stag)
        ends[ch] = st["names"]["gja"]
    spec = chains(phase)["A"]["T"] + MID + stag + "c3" + SUF["gja"] + "avskap"
    n = str({"v": 2060, "f": 2260}[phase] + (0 if stag else 100))
    must_be_new(f"preds/manifest_{spec}.json", f"queue/{n}_{spec}.json")
    run(["tools/chainavg3.py", "build", ends["A"], ends["B"], ends["C"], n, "--suffix", "skap", "--no-enqueue"], f"avg3_{n}_{spec}.log", 2.0, 1, 5, f"{phase}{stag}avg3")
    qentry(n, spec)
    (LANE / f"l4g_spec_{phase}{stag or 'h'}.txt").write_text(spec + "\n")
    print(f"[{ts()}] L4 gate-only end ({phase}, scale2 {sc}): {spec}", flush=True)
    return 0


def gate(phase: str) -> int:
    real = ",".join(REAL[phase])
    pm = f"research/r17_a/l4g_placeholders_{phase}.json"
    for stag, ref, kind in (("s75", REF570, "registered"), ("h", REF571, "information")):
        spec = (LANE / f"l4g_spec_{phase}{stag}.txt").read_text().strip()
        out = f"research/r17_a/gate_l4{phase}_{kind}_{spec}.json"
        run(["research/r17_a/s30_gate_l4.py", f"preds/manifest_{spec}.json", ref, real, pm, out], f"gate_l4{phase}_{kind}.log", 3.0, 1, 10, f"{phase}gate")
        print((ROOT / "logs" / f"gate_l4{phase}_{kind}.log").read_text(encoding="utf-8", errors="replace")[-1500:], flush=True)
    return 0


def main() -> int:
    a = sys.argv[1:]
    arg = lambda k: a[a.index(k) + 1]  # noqa: E731
    phase = arg("--phase")
    if phase not in REAL:
        raise SystemExit("--phase v or f")
    if a[0] == "prep":
        return prep(phase)
    if a[0] == "chain":
        return chain(phase, arg("--chain"), arg("--scale"))
    if a[0] == "avg3":
        return avg3(phase, arg("--scale"))
    if a[0] == "gate":
        return gate(phase)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
