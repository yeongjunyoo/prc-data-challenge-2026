# SPDX-License-Identifier: GPL-3.0-only
"""R17-A builder on the Mac (private notes research/2026-10-08-r17-a-adsb-fill.md, pre-registered 2026-10-08 21:55 KST), run from the sandbox
<mac-workdir>/prcm/r17a_sbx (an APFS clone of r16b_sbx, 10-08 21:5x). Candidate L3 = 514 with only the LightGBM legs retrained on feats_v28
(feats_v27 with the ab_* columns filled for every 2025 day), and every step below them rebuilt with 514's exact arguments.
The step helpers (PROTOCOL check, run, must_be_new, qentry, guard sync, gate) are research/r16_b/sb_build_r16b.py's (sha16 fd5b128aaa16b62d
family), with this lane's lock names, paths and owner text. The chain steps (corr .. gja) are its commands unchanged.
Subcommands (from the sandbox root, <mac-workdir>/venv311/bin/python -u research/r17_a/sb_build_r17a.py ...):
  feats             cache/adsb_v3 (research/adsb/build_feats2.py on <mac-workdir>/r17a/events_all: the 240 old days and the 187 new 2025 days),
                    cache/feats_v28 (python -m src.feats_adsb feats_v28 --adsb adsb_v3 --mode raw), then research/r17_a/check_feats_r17a.py
                    (feats_v28 equals feats_v27 on every non-ab_ column and on the ab_ columns of rows that had day data; adsb_v3 equals adsb_v2 there)
  legs [--lanes N]  the LightGBM leg runs (C m6r28skap seed 0, A m6r28skwin seeds 0-3, B m6r28sk seeds 0-3; val, fwd, full each; 6 threads),
                    and src/g_band.py of each tag's seed 0 as soon as its three settings exist; at most N model3 runs at once (default 1)
  chain --chain A|B|C   [avg (A, B)], bfix (src/blend_fixed.py with 514's learners and reference), [blend (A: research/r13_l/blend_add.py,
                    TabM m6r27tmcwin:mu_tabm)], corr (src/ctx_corr3.py seeds 0, 1, 2), cavg, manifest, am .. gja
  final             tools/chainavg3.py (A, B, C) --suffix skap, then the gate (research/r17_a/s30_gate_r17a.py) against 514
PROTOCOL per step: lock dir r17a_<name>.lock with the declared peak GB and threads, two checks 30 s apart of no arc_quiet.* , declared threads of
running locks + ours <= 12 (<= 13 for a 1-thread step), memory_pressure free >= 30 %; no lock is held while waiting; nice 10; targets checked new.
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
EVENTS = Path(os.environ.get("ADSB_WORK", ".")) / "events_all"  # ADS-B download folder
PY = sys.executable
LANE = ROOT / "research" / "r17_a"
CORR = ("--feats feats_v25 --adsb adsb_v1nd --adsb_sha 0d03ff06267e0dee --folds 10 --rounds 300 --rounds2 400 --max_pred 3600 --threads 4 "
        "--geo geo_v1nd --geo_sha 43310b9bbeefe81b --ctx ctx_v3 --ctx_sha 0b139ddbd6e70573 --scale2 0.5")
REF = "--ref_base m6r25sg --ref_tag m6r25sgoof4 --ref_learners m6r25cb:mu_cat,m6r25xg:mu_xgb,m6r25rm:mu_rmlp --folds 10 --check"
M3 = ("--feats feats_v28 --es internal --refit --wpow 1 --leaves 255 --mdl 100 --no_direct --clf_drop_prefix m_,ab_ --full_mult 1.0 "
      "--threads 6")
# MANIFEST_EXTRA_CODE: 514's A-chain list + this lane's average script and the code that made feats_v28's new days; chain C adds the ap scripts
EXTRA = ("src/blend_fixed.py,src/ctx_corr2.py,src/ctx_corr3.py,src/feats_ctx2.py,src/feats_aiu.py,research/ctx_corr/freeze_geo.py,research/obs2/corr_plus.py,"
         "research/obs2/obs_rows.py,research/obs2/coverage.py,research/obs2/osm_stands.py,research/adsb/gprobe/build_nd.py,"
         "research/r13_l/prep_x.py,research/r13_l/tabm_leg.py,research/r13_l/xrfm_leg.py,research/r13_l/blend_add.py,research/r10_b/cavg.py,"
         "research/r17_a/avg_r17a.py,research/r17_a/adsb_day_r17a.py,research/adsb/build_feats2.py,research/adsb/adsb_match.py")
CHAINS = {
    "A": {"LEG": "m6r28skwin", "SCRIPT": "src/model3.py", "ARGS": "--mu_sep LIRF --anchor win", "SEEDS": (0, 1, 2, 3), "GB": "m6r28skwing",
          "X4": "m6r28skwingx4", "BF": "m6r28wincbrmfw4sv434", "LEARN": "m6r27cbwinx3:mu_cat,m6r25xg:mu_xgb,m6r27rmwinx4:mu_rmlp",
          "T": "m6r28wincbrmfw4sv434tmc", "TABM": "m6r27tmcwin:mu_tabm", "NUM0": 1700, "EXTRA": EXTRA},
    "B": {"LEG": "m6r28sk", "SCRIPT": "src/model3.py", "ARGS": "--mu_sep LIRF", "SEEDS": (0, 1, 2, 3), "GB": "m6r28skg",
          "X4": "m6r28skgx4", "BF": "m6r28skcbrmfw4sv434", "LEARN": "m6r27cbx3:mu_cat,m6r25xg:mu_xgb,m6r27rmx4:mu_rmlp",
          "T": "m6r28skcbrmfw4sv434", "TABM": None, "NUM0": 1720, "EXTRA": EXTRA},
    "C": {"LEG": "m6r28skap", "SCRIPT": "research/r13_f3/model3ap.py", "ARGS": "--mu_sep all", "SEEDS": (0,), "GB": "m6r28skapg",
          "X4": None, "BF": "m6r28apcbrmfw4sv114", "LEARN": "m6r27cbap:mu_cat,m6r25xg:mu_xgb,m6r27rmx4:mu_rmlp",
          "T": "m6r28apcbrmfw4sv114", "TABM": None, "NUM0": 1740, "EXTRA": EXTRA + ",research/r13_f3/model3ap.py,research/r13_f3/cat_muap.py"},
}
STEPS = ["avg", "bfix", "blend", "corr", "cavg", "manifest", "am", "fh", "gkc", "gkcx", "xncr", "db", "aw", "gp", "gl", "gja"]
SUF = {"am": "am", "fh": "amfh", "gkc": "amfhgkc", "gkcx": "amfhgkcx", "xncr": "amfhgkcxncr", "db": "amfhgkcxncrdb", "aw": "amfhgkcxncrdbaw",
       "gp": "amfhgkcxncrdbawgp", "gl": "amfhgkcxncrdbawgpgl", "gja": "amfhgkcxncrdbawgpglgja"}
MID = "adsbgeondctxv3x50w"
REF514 = "m6r27wincbrmfw4sv434tmc" + MID + "c3" + SUF["gja"] + "avskap"
NUM_FINAL = 1780
sys.stdout.reconfigure(line_buffering=True)


def ts() -> str:
    return time.strftime("%H:%M:%S")


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def arc_blocks() -> bool:
    return (P / "arc_quiet.request").exists() or (P / "arc_quiet.lock").exists()


def declared() -> float:
    st = 0.0
    for d in P.glob("*.lock"):
        if d.is_dir():
            txt = (d / "info").read_text(errors="replace") if (d / "info").exists() else ""
            t = re.search(r"(?im)^threads[^0-9]*([0-9]+(?:\.[0-9]+)?)", txt)
            st += float(t.group(1)) if t else 6
    return st


def mem_free() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.strip().splitlines()
    m = re.search(r"([0-9]+)%", out[-1]) if out else None
    return int(m.group(1)) if m else 0


def budget_ok(thr: int) -> bool:
    if arc_blocks():
        return False
    if declared() + thr > (13 if thr == 1 else 12):
        return False
    return mem_free() >= 30


def wait_budget(thr: int) -> None:
    ok = 0
    while ok < 2:
        ok = ok + 1 if budget_ok(thr) else 0
        if ok < 2:
            time.sleep(30)


def lock_info(lock: Path, args: list[str], peak: float, thr: int, mins: int) -> None:
    (lock / "info").write_text(f"owner: PRC R17-A (LightGBM legs on feats_v28, sandbox r17a_sbx)\npid: {os.getpid()}\npeak_gb: {peak}\n"
                               f"threads: {thr}\nstart: {time.strftime('%Y-%m-%d %H:%M')}\n"
                               f"expected_end: {time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time() + 60 * mins))} KST\n"
                               f"job: {' '.join(args)[:200]}\n")


def env_for(thr: int, extra: dict | None = None) -> dict:
    return {**os.environ, "OMP_NUM_THREADS": str(thr), "OPENBLAS_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8", **(extra or {})}


def report(log: str, rc: int) -> None:
    txt = (ROOT / "logs" / log).read_text(encoding="utf-8", errors="replace")
    pk = re.search(r"([0-9]+)\s+peak memory footprint", txt)
    rl = re.search(r"([0-9.]+)\s+real", txt)
    last = [ln for ln in txt.splitlines() if ln.strip() and not re.match(r"^\s+[0-9]+\s", ln)][-2:]
    print(f"[{ts()}]   {log}: rc {rc}" + (f" | real {float(rl.group(1)):.0f} s" if rl else "") + (f" | peak footprint {int(pk.group(1)) / 2**30:.2f} GB" if pk else "")
          + " | " + " | ".join(x[:160] for x in last), flush=True)


def run(args: list[str], log: str, peak: float, thr: int, mins: int, lockname: str, env: dict | None = None) -> None:
    wait_budget(thr)
    lock = P / f"r17a_{lockname}.lock"
    lock.mkdir()
    try:
        lock_info(lock, args, peak, thr, mins)
        print(f"[{ts()}] run {' '.join(args)[:230]}", flush=True)
        with open(ROOT / "logs" / log, "w", encoding="utf-8") as f:
            r = subprocess.run(["/usr/bin/time", "-l", "nice", "-n", "10", PY, "-X", "utf8", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                               env=env_for(thr, env))
    finally:
        shutil.rmtree(lock, ignore_errors=True)
    report(log, r.returncode)
    if r.returncode != 0:
        raise SystemExit(f"step failed: {log}")


def must_be_new(*rels: str) -> None:
    for rel in rels:
        if (ROOT / rel).exists():
            raise SystemExit(f"target already exists, refusing: {rel}")


def qentry(num: str, spec: str, copy_test: bool = True) -> str:
    name = f"{num}_{spec}"
    if (ROOT / "queue" / f"{name}.json").exists():
        raise SystemExit(f"sandbox queue entry exists, refusing: {name}")
    m = json.loads((ROOT / "preds" / f"manifest_{spec}.json").read_text(encoding="utf-8"))
    (ROOT / "queue" / f"{name}.json").write_text(json.dumps({"manifest": f"preds/manifest_{spec}.json", "kind": "CANDIDATE", "sandbox": True},
                                                            ensure_ascii=False, indent=1), encoding="utf-8")
    if copy_test:
        shutil.copy2(ROOT / m["test"]["path"], ROOT / "queue" / f"{name}.parquet")
    return name


def marker(name: str) -> Path:
    return LANE / "done" / f"{name}.done"


def mark(name: str) -> None:
    marker(name).parent.mkdir(parents=True, exist_ok=True)
    marker(name).write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")


# ---------------------------------------------------------------- feats
def feats(a: list[str]) -> int:
    want = 427
    newout = Path(os.environ.get("ADSB_WORK", ".")) / "out"
    for d in sorted(newout.glob("2025-*")):  # the new 2025 days join the 240 old ones (symlinks, the extracts stay where they are)
        if (d / "meta.json").exists() and (d / "events.parquet").exists() and not (EVENTS / d.name).exists():
            (EVENTS / d.name).symlink_to(d)
    have = [d for d in EVENTS.iterdir() if (d / "meta.json").exists() and (d / "events.parquet").exists()] if EVENTS.exists() else []
    if len(have) < want and "--partial" not in a:
        raise SystemExit(f"events_all has {len(have)} complete days, want {want}")
    if not (ROOT / "cache" / "adsb_v3.parquet").exists():
        run(["-u", "research/adsb/build_feats2.py", "--name", "adsb_v3", "--out-dir", str(EVENTS)], "adsb_v3.log", 2.0, 1, 10, "adsb_v3")
    meta = json.loads((ROOT / "cache" / "adsb_v3.json").read_text(encoding="utf-8"))
    print(f"[{ts()}] adsb_v3: {meta['days']} days, missing {meta['days_missing_by_month']}, rows {meta['rows']}, sha16 {meta['sha16']}", flush=True)
    if not (ROOT / "cache" / "feats_v28.parquet").exists():
        run(["-u", "-m", "src.feats_adsb", "feats_v28", "--adsb", "adsb_v3", "--mode", "raw"], "feats_v28.log", 4.0, 1, 10, "feats_v28")
    run(["-u", "research/r17_a/check_feats_r17a.py"], "check_feats_r17a.log", 4.0, 1, 10, "checkfeats")
    mark("feats")
    return 0


# ---------------------------------------------------------------- legs
def leg_jobs() -> list[dict]:
    """C seed 0, A seed 0, B seed 0, then A and B seeds 1..3 alternating; val, fwd, full per seed (full reads cache/iters_<tag>.json of val seed 0)."""
    order = [("C", 0), ("A", 0), ("B", 0), ("A", 1), ("B", 1), ("A", 2), ("B", 2), ("A", 3), ("B", 3)]
    jobs = []
    for ch, k in order:
        c = CHAINS[ch]
        for s in ("val", "fwd", "full"):
            name = f"{c['LEG']}_{s}_s{k}"
            cmd = f"{c['SCRIPT']} --setting {s} --seed {k} --tag {c['LEG']} {M3} {c['ARGS']}"
            jobs.append({"name": name, "chain": ch, "seed": k, "setting": s, "cmd": cmd, "after": [] if (s == "val" and k == 0) else [f"{c['LEG']}_val_s0"],
                         "out": f"cache/{s}_{c['LEG']}_s{k}.parquet"})
    return jobs


def gband_jobs() -> list[dict]:
    jobs = []
    for ch in ("C", "A", "B"):
        c = CHAINS[ch]
        for s in ("val", "fwd", "full"):
            jobs.append({"name": f"{c['GB']}_{s}_s0", "chain": ch, "setting": s, "after": [f"{c['LEG']}_{x}_s0" for x in ("val", "fwd", "full")],
                         "cmd": f"src/g_band.py --setting {s} --seed 0 --tag {c['GB']} --base {c['LEG']} --feats feats_v28",
                         "out": f"cache/{s}_{c['GB']}_s0.parquet"})
    return jobs


def legs(a: list[str]) -> int:
    lanes = int(a[a.index("--lanes") + 1]) if "--lanes" in a else 1
    if not marker("feats").exists():
        raise SystemExit("feats not done (research/r17_a/done/feats.done)")
    for ch in ("A", "B", "C"):  # the jobs files (record of the commands, as the other lanes' notes/jobs_<tag>.txt)
        c = CHAINS[ch]
        lines = [f"logs/{j['name']}.log\t.venv/Scripts/python.exe -u {j['cmd']}" for j in leg_jobs() if j["chain"] == ch]
        (ROOT / "notes" / f"jobs_{c['LEG']}.txt").write_text(f"# R17-A LightGBM leg on feats_v28 (514's {c['LEG'].replace('m6r28', 'm6r27')} arguments with "
                                                          f"--feats feats_v28, 6 threads), Mac sandbox r17a_sbx\n" + "\n".join(lines) + "\n", encoding="utf-8")
        gl = [f"logs/{j['name']}.log\t.venv/Scripts/python.exe -u {j['cmd']}" for j in gband_jobs() if j["chain"] == ch]
        (ROOT / "notes" / f"jobs_{c['GB']}.txt").write_text(f"# R17-A G band of {c['LEG']} seed 0 (src/g_band.py, feats_v28)\n" + "\n".join(gl) + "\n",
                                                         encoding="utf-8")
    todo = leg_jobs() + gband_jobs()
    running: dict[str, tuple[subprocess.Popen, Path, float, str]] = {}
    for j in todo:
        if marker(j["name"]).exists():
            print(f"[{ts()}] {j['name']}: done before", flush=True)
    todo = [j for j in todo if not marker(j["name"]).exists()]
    while todo or running:
        for name, (pr, lock, t0, log) in list(running.items()):
            rc = pr.poll()
            if rc is None:
                continue
            shutil.rmtree(lock, ignore_errors=True)
            report(log, rc)
            del running[name]
            if rc != 0:
                for n2, (p2, l2, _, _) in running.items():
                    print(f"[{ts()}] waiting for {n2} to end before stopping", flush=True)
                    p2.wait()
                    shutil.rmtree(l2, ignore_errors=True)
                raise SystemExit(f"leg step failed: {name}")
            mark(name)
            print(f"[{ts()}] {name} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
        ready = [j for j in todo if all(marker(x).exists() for x in j["after"])]
        n_m3 = sum(1 for n in running if not n.startswith(tuple(CHAINS[c]["GB"] for c in CHAINS)))
        launched = False
        for j in ready:
            is_gb = j["cmd"].startswith("src/g_band.py")
            thr, peak, mins = (1, 3.0, 10) if is_gb else (6, 7.0, 40)
            if not is_gb and n_m3 >= lanes:
                continue
            if not budget_ok(thr):
                continue
            time.sleep(30)
            if not budget_ok(thr):
                continue
            must_be_new(j["out"])
            lock = P / f"r17a_{j['name']}.lock"
            lock.mkdir()
            args = ["-u", *j["cmd"].split()]
            lock_info(lock, args, peak, thr, mins)
            log = f"{j['name']}.log"
            print(f"[{ts()}] run {j['cmd'][:200]}", flush=True)
            f = open(ROOT / "logs" / log, "w", encoding="utf-8")
            pr = subprocess.Popen(["/usr/bin/time", "-l", "nice", "-n", "10", PY, "-X", "utf8", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                  env=env_for(thr))
            running[j["name"]] = (pr, lock, time.time(), log)
            todo.remove(j)
            launched = True
            break
        if not launched:
            time.sleep(30)
    mark("legs")
    print(f"[{ts()}] all legs and G bands done", flush=True)
    return 0


# ---------------------------------------------------------------- chains
def chain(a: list[str]) -> int:
    ch = a[a.index("--chain") + 1]
    c = CHAINS[ch]
    T, GLEG = c["T"], c["GB"]
    TC, TCS = T + MID, T + MID + "c3"
    state = LANE / f"sb_state_{TCS}.json"
    st = json.loads(state.read_text(encoding="utf-8")) if state.exists() else {"names": {}, "done": None}
    steps = [s for s in STEPS if not (s == "avg" and c["X4"] is None) and not (s == "blend" and c["TABM"] is None)]
    start = a[a.index("--from") + 1] if "--from" in a else (steps[steps.index(st["done"]) + 1] if st["done"] else steps[0])
    stop = a[a.index("--to") + 1] if "--to" in a else steps[-1]
    num = lambda k: str(c["NUM0"] + STEPS.index(k))  # noqa: E731
    need = [f"{c['LEG']}_{s}_s{k}" for k in c["SEEDS"] for s in ("val", "fwd", "full")] + [f"{c['GB']}_{s}_s0" for s in ("val", "fwd", "full")]
    while not all(marker(x).exists() for x in need):
        time.sleep(60)
    print(f"[{ts()}] chain {ch}: legs and G band ready", flush=True)
    for step in steps[steps.index(start):steps.index(stop) + 1]:
        if step == "avg":
            for s in ("val", "fwd", "full"):
                must_be_new(f"cache/{s}_{c['X4']}_s0.parquet")
            cmd = f"research/r17_a/avg_r17a.py {c['GB']} {c['X4']} {','.join(str(k) for k in c['SEEDS'])} {c['LEG']}"
            run(["-u", *cmd.split()], f"avg_{c['X4']}.log", 3.0, 1, 10, f"{ch}avg")
            (ROOT / "notes" / f"jobs_{c['X4']}.txt").write_text(f"# R17-A seed average of {c['LEG']} seeds {c['SEEDS']} on the G-band leg {c['GB']}\n"
                                                             f"logs/{c['X4']}.log\t.venv/Scripts/python.exe -u {cmd}\n", encoding="utf-8")
        elif step == "bfix":
            base = c["X4"] or c["GB"]
            lines = []
            for s in ("val", "fwd", "full"):
                must_be_new(f"cache/{s}_{c['BF']}_s0.parquet")
                cmd = f"src/blend_fixed.py --setting {s} --seed 0 --tag {c['BF']} --base {base} --learners {c['LEARN']} {REF}"
                run(["-u", *cmd.split()], f"bfix_{c['BF']}_{s}_s0.log", 1.5, 1, 10, f"{ch}bfix")
                lines.append(f"logs/{c['BF']}_{s}_s0.log\t.venv/Scripts/python.exe -u {cmd}")
            (ROOT / "notes" / f"jobs_{c['BF']}.txt").write_text(f"# R17-A base blend (514's learners and reference weights, LightGBM leg {base})\n"
                                                             + "\n".join(lines) + "\n", encoding="utf-8")
        elif step == "blend":
            lines = []
            for s in ("val", "fwd", "full"):
                must_be_new(f"cache/{s}_{T}_s0.parquet")
                cmd = f"research/r13_l/blend_add.py --setting {s} --tag {T} --src {c['BF']} --leg {c['TABM']} --folds 10 --seed 0"
                run(["-u", *cmd.split()], f"blend_{T}_{s}_s0.log", 3.0, 1, 5, f"{ch}blend")
                lines.append(f"logs/{T}_{s}_s0.log\t.venv/Scripts/python.exe -u {cmd}")
            (ROOT / "notes" / f"jobs_{T}.txt").write_text(f"# R17-A blend ({c['BF']} + TabM {c['TABM']}, research/r13_l/blend_add.py, as 514's A chain)\n"
                                                        + "\n".join(lines) + "\n", encoding="utf-8")
        elif step == "corr":
            lines = []
            for k in (0, 1, 2):
                for s in ("val", "fwd", "full"):
                    cmd = f"src/ctx_corr3.py --setting {s} --seed {k} --base {T} --base_seed 0 --tag {TC} {CORR}"
                    lines.append(f"logs/{TC}_{s}_s{k}.log\t.venv/Scripts/python.exe -u {cmd}")
                    if (ROOT / f"cache/{s}_{TC}_s{k}.parquet").exists():
                        print(f"[{ts()}] reuse this lane's cache/{s}_{TC}_s{k}.parquet", flush=True)
                        continue
                    run(["-u", *cmd.split(), "--fit_cache", "research/r13_l/fitcache"], f"corr_{TC}_{s}_s{k}.log", 4.6, 4, 25, f"{ch}corr")
            (ROOT / "notes" / f"jobs_{TC}.txt").write_text("# R17-A Mac corrector\n" + "\n".join(lines) + "\n", encoding="utf-8")
        elif step == "cavg":
            lines = []
            for s in ("val", "fwd", "full"):
                must_be_new(f"cache/{s}_{TCS}_s0.parquet")
                cmd = f"research/r10_b/cavg.py --setting {s} --seed 0 --tag {TCS} --src {TC} --seeds 0,1,2 --base {T} --base_seed 0"
                run(["-u", *cmd.split()], f"cavg_{TCS}_{s}_s0.log", 3.0, 1, 5, f"{ch}cavg")
                lines.append(f"logs/{TCS}_{s}_s0.log\t.venv/Scripts/python.exe -u {cmd}")
            (ROOT / "notes" / f"jobs_{TCS}.txt").write_text(f"# R17-A Mac corrector seed average; the seed runs are notes/jobs_{TC}.txt\n" + "\n".join(lines) + "\n",
                                                          encoding="utf-8")
        elif step == "manifest":
            must_be_new(f"preds/manifest_{TCS}.json", f"preds/test_{TCS}.parquet", f"preds/fwd_{TCS}.parquet")
            run(["-m", "src.manifest", TCS, TCS, "pred", "0", "feats_v28", f"notes/jobs_{TCS}.txt"], f"manifest_{TCS}.log", 3.0, 1, 10,
                f"{ch}manifest", {"MANIFEST_EXTRA_CODE": c["EXTRA"]})
            st["names"]["corr"] = qentry(num("manifest"), TCS)
        elif step in SUF:
            prev = TCS if step == "am" else TCS + SUF[STEPS[STEPS.index(step) - 1]]
            spec = TCS + SUF[step]
            must_be_new(f"preds/manifest_{spec}.json", f"preds/test_{spec}.parquet", f"preds/fwd_{spec}.parquet")
            src_q = st["names"]["corr" if step == "am" else STEPS[STEPS.index(step) - 1]]
            srcm = f"preds/manifest_{prev}.json"
            n = num(step)
            args = {"am": ["tools/artmm.py", "build", srcm, n, "--no-enqueue"],
                    "fh": ["tools/arthist.py", "build", srcm, n, "--no-enqueue"],
                    "gkc": ["tools/gkey.py", "build", src_q, n, "--no-enqueue", "--variant", "avg", "--leg", GLEG, "--leg_feats", "feats_v28"],
                    "gkcx": ["tools/gkeyx.py", "build", srcm, n, "--no-enqueue"],
                    "xncr": ["tools/gkeyxn.py", "build", srcm, n, "--no-enqueue", "--variant", "avg_xncr"],
                    "db": ["src/aiu_shift.py", "build", src_q, n, "--group", "block"],
                    "aw": ["tools/antiwin.py", "build", src_q, n, "--no-enqueue"],
                    "gp": ["tools/gpool.py", "build", src_q, n, "--no-enqueue"],
                    "gl": ["tools/glong.py", "build", srcm, n, "--no-enqueue"],
                    "gja": ["tools/gjoint.py", "build", srcm, n, "--mode", "a", "--no-enqueue"]}[step]
            run(args, f"{step}_{n}_{spec}.log", 3.0, 4, 20, f"{ch}tail")
            if not (ROOT / "preds" / f"manifest_{spec}.json").exists():
                raise SystemExit(f"{step}: no manifest {spec}")
            st["names"][step] = qentry(n, spec)
        st["done"], st["updated"], st["host"] = step, ts(), "mac"
        state.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{ts()}] {T}: {step} done (Mac)", flush=True)
    if st["done"] == "gja":
        mark(f"chain{ch}")
    return 0


# ---------------------------------------------------------------- final
def wait_guard_sync(spec: str) -> None:
    """The probing-guard inputs (realqueue/, every queue_ready, subs/) are refreshed from the PC right before the gate: write
    research/r17_a/guard_sync.request, then wait until the PC side (research/r17_a/sync_guard_r17a.py, then a touch of
    research/r17_a/guard_sync.stamp) has run after the request. No lock is held while waiting."""
    req, stamp = LANE / "guard_sync.request", LANE / "guard_sync.stamp"
    t_req = time.time()
    req.write_text(f"{spec}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
    print(f"[{ts()}] gate {spec[-40:]} waits for a fresh guard sync from the PC (research/r17_a/guard_sync.request)", flush=True)
    while not (stamp.exists() and stamp.stat().st_mtime > t_req):
        time.sleep(20)
    req.unlink(missing_ok=True)
    print(f"[{ts()}] guard sync stamp {time.strftime('%H:%M:%S', time.localtime(stamp.stat().st_mtime))} seen", flush=True)


def final(a: list[str]) -> int:
    ends = {}
    for ch in ("A", "B", "C"):
        while not marker(f"chain{ch}").exists():
            time.sleep(60)
        c = CHAINS[ch]
        ends[ch] = f"{c['NUM0'] + STEPS.index('gja')}_{c['T']}{MID}c3{SUF['gja']}"
        if not (ROOT / "queue" / f"{ends[ch]}.json").exists():
            raise SystemExit(f"missing sandbox queue entry {ends[ch]}")
    spec = ends["A"].split("_", 1)[1] + "avskap"
    if not (ROOT / "preds" / f"manifest_{spec}.json").exists():
        run(["tools/chainavg3.py", "build", ends["A"], ends["B"], ends["C"], str(NUM_FINAL), "--suffix", "skap", "--no-enqueue"],
            f"avg3_{NUM_FINAL}_{spec}.log", 3.0, 1, 10, "avg3")
        qentry(str(NUM_FINAL), spec)
    gp = LANE / f"gate_{spec}.json"
    if not gp.exists():
        wait_guard_sync(spec)
        run(["research/r17_a/s30_gate_r17a.py", f"preds/manifest_{spec}.json", f"preds/manifest_{REF514}.json"], f"gate_{spec}.log", 3.0, 1, 10, "gate")
    g = json.loads(gp.read_text(encoding="utf-8"))
    print(f"[{ts()}] gate {spec} vs 514: 7.8 {g['gate78']} -> {'PASS' if g['pass78'] else 'FAIL'}; val {g['val']}; fwd {g['fwd']}; "
          f"test {g['test_move_vs_ref']}; guard {g['guard']}", flush=True)
    (P / "r17a_L3.DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    mark("final")
    return 0


def main() -> int:
    a = sys.argv[1:]
    return {"feats": feats, "legs": legs, "chain": chain, "final": final}[a[0]](a)


if __name__ == "__main__":
    sys.exit(main())
