# SPDX-License-Identifier: GPL-3.0-only
"""R17-A L3S builder: research/var1/sb_build_var1.py (sha16 c4e923fc0bc5dae0) run on L3's three chains (private notes research/2026-10-08-r17-a-adsb-fill.md,
10-09 09:5x: if L3 passes gate 7.8 against 514, build L3S = L3 with --scale2 0.75 in every corrector, as 531 = 514 + S75; no local gate).
Changes, nothing else: the chains are L3's (bases m6r28wincbrmfw4sv434tmc, m6r28skcbrmfw4sv434, m6r28apcbrmfw4sv114, G-band legs m6r28skwing,
m6r28skg, m6r28skapg), the fit cache is the one L3's correctors filled in this sandbox (research/r13_l/fitcache), the reproduction check compares
with L3's own corrector deltas, src.manifest and gkey get feats_v28 (L3's legs), sandbox queue numbers 1800 / 1820 / 1840 / 1860, state, locks
and markers carry r17a_l3s. VAR-1 text follows.
VAR-1 S75 builder on the Mac (private note 2026-10-08-var1-scale2.md section 1, pre-registered 2026-10-08 13:13 KST).
S75 = candidate 514 (D3) with the stage-2 corrector strength --scale2 0.75 instead of 0.5 in every corrector of all three chains (A'_434, B_504, C_ap),
everything else as in 514. Per chain: ctx_corr3 seeds 0, 1, 2 x val, fwd, full (the 514 arguments, only --scale2 and --tag change; --fit_cache is the
research-only flag that loads the stored LightGBM fits, it never changes an output), research/r10_b/cavg.py --seeds 0,1,2, src.manifest, then the tail
artmm, arthist, gkey, gkeyx, gkeyxn, aiu_shift, antiwin, gpool, glong, gjoint a, with the same tool calls as research/r13_l/sb_build_mac4.py (D3). Then
tools/chainavg3.py (A', B, C) --suffix skap.
Cache reproduction proof (gate 1b): after every corrector run the unscaled stage-2 change it writes (notes/ctx_corr_<tag>_<setting>_s<k>_delta.parquet:
s1 and d2) is compared with the same file of 514's corrector; models loaded from the fit cache must give identical arrays, a refitted model at most 0.01 s
apart, otherwise the runner stops (marker file var1_REPRO_MISMATCH) and nothing else runs.
PROTOCOL (<mac-workdir>/prcm/PROTOCOL.txt): one lock dir var1_<chain>_<step>.lock per step with the real threads and peak GB, taken only while the step runs
(two checks 30 s apart before it: no arc_quiet.request / arc_quiet.lock, threads of running locks + ours <= 12 (13 for a single-thread step, 07:4x rule),
memory_pressure free >= 30 %), nice 10, every target checked to not exist before its step.
    <mac-workdir>/venv311/bin/python -u research/r17_a/sb_build_l3s.py chain <A|B|C> [--from <step>] [--to <step>] [--after <chain>:<step>]
    <mac-workdir>/venv311/bin/python -u research/r17_a/sb_build_l3s.py avg3
"""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[2]
P = Path(os.environ.get("PRC_WORK", "."))  # Mac work folder that holds the build sandboxes
PY = sys.executable
SC, TAG = "0.75", "s75"
CH = {"A": {"T": "m6r28wincbrmfw4sv434tmc", "GLEG": "m6r28skwing", "NUM0": 1800},
      "B": {"T": "m6r28skcbrmfw4sv434", "GLEG": "m6r28skg", "NUM0": 1820},
      "C": {"T": "m6r28apcbrmfw4sv114", "GLEG": "m6r28skapg", "NUM0": 1840}}
CORR = ("--feats feats_v25 --adsb adsb_v1nd --adsb_sha 0d03ff06267e0dee --folds 10 --rounds 300 --rounds2 400 --max_pred 3600 --threads 4 "
        f"--geo geo_v1nd --geo_sha 43310b9bbeefe81b --ctx ctx_v3 --ctx_sha 0b139ddbd6e70573 --scale2 {SC}")
FITCACHE = "research/r13_l/fitcache"
STEPS = ["corr", "cavg", "manifest", "am", "fh", "gkc", "gkcx", "xncr", "db", "aw", "gp", "gl", "gja"]
SUF = {"am": "am", "fh": "amfh", "gkc": "amfhgkc", "gkcx": "amfhgkcx", "xncr": "amfhgkcxncr", "db": "amfhgkcxncrdb", "aw": "amfhgkcxncrdbaw",
       "gp": "amfhgkcxncrdbawgp", "gl": "amfhgkcxncrdbawgpgl", "gja": "amfhgkcxncrdbawgpglgja"}
AVG_NUM = "1860"
sys.stdout.reconfigure(line_buffering=True)


def ts() -> str:
    return time.strftime("%H:%M:%S")


def names(c: str) -> dict:
    T = CH[c]["T"]
    tc0 = T + "adsbgeondctxv3x50w"
    return {"T": T, "TC0": tc0, "TCS0": tc0 + "c3", "TC": tc0 + TAG, "TCS": tc0 + TAG + "c3", "GLEG": CH[c]["GLEG"], "NUM0": CH[c]["NUM0"]}


def state_path(c: str) -> Path:
    return ROOT / "research" / "r17_a" / f"sb_state_l3s_{c}.json"


def load_state(c: str) -> dict:
    p = state_path(c)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"names": {}, "done": None, "corr": {}, "times": {}}


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


def run(args: list[str], log: str, peak: float, thr: int, mins: int, tag: str, env: dict | None = None) -> dict:
    ok = 0
    while ok < 2:
        ok = ok + 1 if budget_ok(thr) else 0
        if ok < 2:
            time.sleep(30)
    lock = P / f"r17a_l3s_{tag}.lock"
    lock.mkdir()
    try:
        (lock / "info").write_text(f"owner: PRC R17-A L3S build ({tag})\npid: {os.getpid()}\npeak_gb: {peak}\nthreads: {thr}\n"
                                   f"start: {time.strftime('%Y-%m-%d %H:%M')}\n"
                                   f"expected_end: {time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time() + 60 * mins))} KST\n"
                                   f"job: {' '.join(args)[:200]}\n")
        print(f"[{ts()}] run {' '.join(args)[:230]}", flush=True)
        e = {**os.environ, "OMP_NUM_THREADS": str(thr), "OPENBLAS_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8", **(env or {})}
        with open(ROOT / "logs" / log, "w", encoding="utf-8") as f:
            r = subprocess.run(["/usr/bin/time", "-l", "nice", "-n", "10", PY, "-X", "utf8", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=e)
    finally:
        shutil.rmtree(lock, ignore_errors=True)
    txt = (ROOT / "logs" / log).read_text(encoding="utf-8", errors="replace")
    pk = re.search(r"([0-9]+)\s+peak memory footprint", txt)
    rl = re.search(r"([0-9.]+)\s+real", txt)
    last = [ln for ln in txt.splitlines() if ln.strip() and not re.match(r"^\s+[0-9]+\s", ln) and "real" not in ln][-2:]
    print(f"[{ts()}]   rc {r.returncode}" + (f" | peak footprint {int(pk.group(1)) / 2**30:.2f} GB" if pk else "") + (f" | real {rl.group(1)} s" if rl else "")
          + " | " + " | ".join(x[:160] for x in last), flush=True)
    if r.returncode != 0:
        raise SystemExit(f"step failed: {log}")
    return {"text": txt, "peak_gb": round(int(pk.group(1)) / 2**30, 2) if pk else None, "real_s": float(rl.group(1)) if rl else None}


def must_be_new(*rels: str) -> None:
    for rel in rels:
        if (ROOT / rel).exists():
            raise SystemExit(f"target already exists, refusing: {rel}")


def qentry(num: str, spec: str, copy_test: bool = True) -> str:
    name = f"{num}_{spec}"
    m = json.loads((ROOT / "preds" / f"manifest_{spec}.json").read_text(encoding="utf-8"))
    (ROOT / "queue" / f"{name}.json").write_text(json.dumps({"manifest": f"preds/manifest_{spec}.json", "kind": "CANDIDATE", "sandbox": True},
                                                            ensure_ascii=False, indent=1), encoding="utf-8")
    if copy_test:
        shutil.copy2(ROOT / m["test"]["path"], ROOT / "queue" / f"{name}.parquet")
    return name


def extra_code(c: str) -> str:
    """MANIFEST_EXTRA_CODE of the chain: the files its 514-era corrector manifest lists beyond src.manifest's default list."""
    sys.path.insert(0, str(ROOT))
    os.environ.pop("MANIFEST_EXTRA_CODE", None)
    from src import manifest as M  # noqa: E402
    base = set(M.CODE)
    m = json.loads((ROOT / "preds" / f"manifest_{names(c)['TCS0']}.json").read_text(encoding="utf-8"))
    return ",".join(k for k in m["code_sha"] if k not in base)


def check_delta(c: str, s: str, k: int, loaded: int, fitted: int) -> dict:
    """Gate 1b: the unscaled stage-2 change of this run against the same run of 514's corrector."""
    n = names(c)
    new = pd.read_parquet(ROOT / "notes" / f"ctx_corr_{n['TC']}_{s}_s{k}_delta.parquet")
    old = pd.read_parquet(ROOT / "notes" / f"ctx_corr_{n['TC0']}_{s}_s{k}_delta.parquet")
    same_ids = bool((new["MVT_ID_mvt"].to_numpy() == old["MVT_ID_mvt"].to_numpy()).all()) and len(new) == len(old)
    res = {"rows": int(len(new)), "ids_equal": same_ids, "fit_cache_loaded": loaded, "fit_cache_fitted": fitted}
    for col in ("s1", "d2"):
        a, b = new[col].to_numpy(dtype=float), old[col].to_numpy(dtype=float)
        res[f"{col}_equal"] = bool(np.array_equal(a, b, equal_nan=True))
        res[f"{col}_maxabs"] = float(np.nanmax(np.abs(a - b))) if same_ids else None
    ok = same_ids and (res["s1_equal"] and res["d2_equal"] if fitted == 0 else (res["s1_maxabs"] <= 0.01 and res["d2_maxabs"] <= 0.01))
    res["pass"] = bool(ok)
    print(f"[{ts()}]   repro {c} {s} s{k}: {res}", flush=True)
    if not ok:
        (P / "r17a_l3s_REPRO_MISMATCH").write_text(f"{c} {s} s{k}: {res}\n")
        raise SystemExit(f"REPRO MISMATCH {c} {s} s{k}: {res}")
    return res


def wait_for(spec: str) -> None:
    c, step = spec.split(":")
    while True:
        d = load_state(c)["done"]
        if d and STEPS.index(d) >= STEPS.index(step):
            return
        if (P / "r17a_l3s_REPRO_MISMATCH").exists():
            raise SystemExit("REPRO MISMATCH marker present")
        time.sleep(20)


def chain(c: str, a: list[str]) -> int:
    n = names(c)
    T, TC, TCS = n["T"], n["TC"], n["TCS"]
    st = load_state(c)
    start = a[a.index("--from") + 1] if "--from" in a else (STEPS[STEPS.index(st["done"]) + 1] if st["done"] else "corr")
    stop = a[a.index("--to") + 1] if "--to" in a else STEPS[-1]
    if "--after" in a:
        wait_for(a[a.index("--after") + 1])
    num = lambda k: str(n["NUM0"] + STEPS.index(k))  # noqa: E731
    for step in STEPS[STEPS.index(start):STEPS.index(stop) + 1]:
        if (P / "r17a_l3s_REPRO_MISMATCH").exists():
            raise SystemExit("REPRO MISMATCH marker present")
        t0 = time.time()
        if step == "corr":
            lines = []
            for k in (0, 1, 2):
                for s in ("val", "fwd", "full"):
                    cmd = f"src/ctx_corr3.py --setting {s} --seed {k} --base {T} --base_seed 0 --tag {TC} {CORR}"
                    lines.append(f"logs/{TC}_{s}_s{k}.log\t.venv/Scripts/python.exe -u {cmd}")
            (ROOT / "notes" / f"jobs_{TC}.txt").write_text("# R17-A L3S Mac corrector (scale2 0.75 on L3)\n" + "\n".join(lines) + "\n", encoding="utf-8")
            for k in (0, 1, 2):
                for s in ("val", "fwd", "full"):
                    key = f"{s}_s{k}"
                    if key in st["corr"] and (ROOT / f"cache/{s}_{TC}_s{k}.parquet").exists():
                        print(f"[{ts()}] done earlier: {c} {key}", flush=True)
                        continue
                    must_be_new(f"cache/{s}_{TC}_s{k}.parquet")
                    cmd = f"src/ctx_corr3.py --setting {s} --seed {k} --base {T} --base_seed 0 --tag {TC} {CORR}"
                    r = run(["-u", *cmd.split(), "--fit_cache", FITCACHE], f"corr_{TC}_{s}_s{k}.log", 4.4, 4, 6, f"{c}_corr")
                    m = re.search(r"fit cache \S+: (\d+) loaded, (\d+) fitted", r["text"])
                    if not m:
                        raise SystemExit(f"no fit cache line in corr_{TC}_{s}_s{k}.log")
                    d = check_delta(c, s, k, int(m.group(1)), int(m.group(2)))
                    st["corr"][key] = {**d, "peak_gb": r["peak_gb"], "real_s": r["real_s"]}
                    state_path(c).write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        elif step == "cavg":
            lines = []
            for s in ("val", "fwd", "full"):
                must_be_new(f"cache/{s}_{TCS}_s0.parquet")
                cmd = f"research/r10_b/cavg.py --setting {s} --seed 0 --tag {TCS} --src {TC} --seeds 0,1,2 --base {T} --base_seed 0"
                run(["-u", *cmd.split()], f"cavg_{TCS}_{s}_s0.log", 1.0, 1, 2, f"{c}_cavg")
                lines.append(f"logs/{TCS}_{s}_s0.log\t.venv/Scripts/python.exe -u {cmd}")
            (ROOT / "notes" / f"jobs_{TCS}.txt").write_text(f"# R17-A L3S Mac corrector seed average; the seed runs are notes/jobs_{TC}.txt\n" + "\n".join(lines) + "\n",
                                                          encoding="utf-8")
        elif step == "manifest":
            must_be_new(f"preds/manifest_{TCS}.json", f"preds/test_{TCS}.parquet", f"preds/fwd_{TCS}.parquet")
            ex = extra_code(c)
            print(f"[{ts()}] MANIFEST_EXTRA_CODE ({c}): {ex}", flush=True)
            run(["-m", "src.manifest", TCS, TCS, "pred", "0", "feats_v28", f"notes/jobs_{TCS}.txt"], f"manifest_{TCS}.log", 1.0, 1, 2, f"{c}_manifest",
                {"MANIFEST_EXTRA_CODE": ex})
            st["names"]["corr"] = qentry(num("manifest"), TCS)
        else:
            prev = TCS if step == "am" else TCS + SUF[STEPS[STEPS.index(step) - 1]]
            spec = TCS + SUF[step]
            must_be_new(f"preds/manifest_{spec}.json", f"preds/test_{spec}.parquet", f"preds/fwd_{spec}.parquet")
            src_q = st["names"]["corr" if step == "am" else STEPS[STEPS.index(step) - 1]]
            srcm = f"preds/manifest_{prev}.json"
            nn = num(step)
            args = {"am": ["tools/artmm.py", "build", srcm, nn, "--no-enqueue"],
                    "fh": ["tools/arthist.py", "build", srcm, nn, "--no-enqueue"],
                    "gkc": ["tools/gkey.py", "build", src_q, nn, "--no-enqueue", "--variant", "avg", "--leg", n["GLEG"], "--leg_feats", "feats_v28"],
                    "gkcx": ["tools/gkeyx.py", "build", srcm, nn, "--no-enqueue"],
                    "xncr": ["tools/gkeyxn.py", "build", srcm, nn, "--no-enqueue", "--variant", "avg_xncr"],
                    "db": ["src/aiu_shift.py", "build", src_q, nn, "--group", "block"],
                    "aw": ["tools/antiwin.py", "build", src_q, nn, "--no-enqueue"],
                    "gp": ["tools/gpool.py", "build", src_q, nn, "--no-enqueue"],
                    "gl": ["tools/glong.py", "build", srcm, nn, "--no-enqueue"],
                    "gja": ["tools/gjoint.py", "build", srcm, nn, "--mode", "a", "--no-enqueue"]}[step]
            run(args, f"{step}_{nn}_{spec}.log", 1.5, 1, 5, f"{c}_{step}")
            if not (ROOT / "preds" / f"manifest_{spec}.json").exists():
                raise SystemExit(f"{step}: no manifest {spec}")
            st["names"][step] = qentry(nn, spec)
        st["done"], st["updated"], st["host"] = step, ts(), "mac"
        st["times"][step] = round(time.time() - t0)
        state_path(c).write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{ts()}] {c} {step} done ({round(time.time() - t0)} s)", flush=True)
    return 0


def avg3() -> int:
    ends = {}
    for c in "ABC":
        st = load_state(c)
        if st["done"] != "gja":
            raise SystemExit(f"chain {c} is not finished (done={st['done']})")
        ends[c] = st["names"]["gja"]
    spec = names("A")["TCS"] + SUF["gja"] + "avskap"
    must_be_new(f"preds/manifest_{spec}.json", f"preds/test_{spec}.parquet", f"queue/{AVG_NUM}_{spec}.json")
    run(["tools/chainavg3.py", "build", ends["A"], ends["B"], ends["C"], AVG_NUM, "--suffix", "skap", "--no-enqueue"], f"avg3_{AVG_NUM}_{spec}.log", 1.5, 1, 5, "avg3")
    qentry(AVG_NUM, spec)
    print(f"[{ts()}] L3S spec {spec}", flush=True)
    (ROOT / "research" / "r17_a" / "l3s_spec.txt").write_text(spec + "\n")
    return 0


def main() -> int:
    a = sys.argv[1:]
    if a and a[0] == "chain":
        return chain(a[1], a[2:])
    if a and a[0] == "avg3":
        return avg3()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
