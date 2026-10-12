# SPDX-License-Identifier: GPL-3.0-only
"""F2 Package S learner runner (Mac): the new seed fits of chain C's legs, one job at a time, exactly the commands of research/f2/s_spec.py.
  val, fwd   in the S sandbox clone (~/nova-adsb/prcm/f2s_sbx598 or f2s_sbxl4)
  full, fj   on the root s_spec.py names (final_run; 598c's CatBoost and RealMLP fj seeds in f2_run, where their seed 0 ran), if F3 uses it
Rules (coordinator 10-11 01:17 and ~/nova-adsb/prcm/PROTOCOL.txt): one heavy job at a time (no other lock declaring 4 or more threads) and
RealMLP alone (other locks declare at most 2 threads); before each job two checks 30 s apart of no arc_quiet.request or arc_quiet.lock,
declared threads of the other locks + its own <= 12 (13 for a one-thread job) and memory_pressure free >= 30 % (--rm_floor for RealMLP, 30
unless the coordinator grants less); lock ~/nova-adsb/prcm/f2s_<family>_<job>.lock with owner, pid, peak_gb, threads, start, expected_end
and job, held only while the job runs; nice 10, /usr/bin/time -l; env OMP_NUM_THREADS = the job's threads (1 for RealMLP),
OPENBLAS_NUM_THREADS=1, PYTHONFAULTHANDLER=1, PYTHONIOENCODING=utf-8. RealMLP: research/f2/rm_smoke.py (a one-minute fit) before the first
RealMLP job; a watchdog kills the job and stops the runner if memory free falls below 20 %; a crash (signal, or a faulthandler trace) is
retried once and a seed that crashes twice is dropped; chain C takes its own RealMLP only if at least three seeds finish every setting run
here (the state file says which). A LightGBM or CatBoost failure, or a non-crash RealMLP failure, stops the runner (research/f2/out/
s_learn_<family>.FAILED, first word = the settings run). ~/nova-adsb/prcm/f2s_HOLD pauses before the next job (no lock held); ~/nova-adsb/prcm/f2s_STOP ends cleanly.
B0 links are made before a job (a link that exists must point at the seed-0 file); a planned output that exists without this runner's
record stops the runner. State: research/f2/out/s_learn_<family>_<settings>.json. When every job is done the runner checks its own work
in process with research/f2/s_check.py cmds and rows (the commands against their references, the B0 links, the seed files' rows, order and
early-stopping ids against seed 0's), writes research/f2/out/s_check_{cmds,rows}_<family>_<settings>.json, and writes
s_learn_<family>_<settings>.DONE only if both pass (otherwise s_learn_<family>.FAILED). A run first removes its own old DONE and FAILED;
a FAILED left by another settings run in the root stops it; a failure removes this run's DONE. Any exception ends in FAILED too, after the
running job (if any) is ended and before its lock goes, and so do SIGTERM and SIGINT (trapped); only f2s_STOP (rc 3) and SIGKILL (which
cannot be trapped) end a run with neither marker.
    python -X utf8 -u research/f2/s_learn.py <598c|l4> <val,fwd|full|fj> [--legs lgbap,cbap,rm] [--rm_seeds 4,5,6,7|none | --rm_seeds_from <json>]
           [--rm_floor 30] [--dry] [--any_root]
(cwd = the root.) full and fj with RealMLP jobs need the sandbox's finished seeds, given as --rm_seeds or read by --rm_seeds_from from the
sandbox's research/f2/out/s_learn_<family>_valfwd.json (rm_seeds_done, or none if chain C kept the family RealMLP; only with that run's
DONE beside it, a passed self-check and every named seed done in val and fwd); without either the runner refuses. A seed crashing twice in full stops the runner instead of being dropped, since val and fwd already used it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s_spec  # noqa: E402

ROOT = Path.cwd()
P = Path.home() / "nova-adsb" / "prcm"
PY = str(Path.home() / "nova-adsb" / "venv311" / "bin" / "python")
OUT = ROOT / "research" / "f2" / "out"
CODE = {"research/r13_f3/model3ap.py": "21733839ba861621", "research/r13_f3/cat_muap.py": "0ab44b8b9e4b276a", "src/rmlp_mu.py": "28fe9c7543664c2b",
        "research/f2/model3ap_fj.py": "22b2ec5095f73b21", "research/f2/cat_muap_fj.py": "dc756eb3fa606a94", "research/f2/rmlp_mu_fj.py": "75b9b77b0127e8b8"}
CRASH = re.compile(r"Fatal Python error|Segmentation fault|Bus error|Abort trap|Current thread 0x")


def ts() -> str:
    return time.strftime("%m-%d %H:%M:%S")


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def free_pct() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.strip().splitlines()
    m = re.search(r"([0-9]+)%", out[-1]) if out else None
    return int(m.group(1)) if m else 0


def other_locks(mine: Path | None) -> list[tuple[str, float, float]]:
    res = []
    for d in P.glob("*.lock"):
        if not d.is_dir() or d == mine or d.name == "arc_quiet.lock":
            continue
        txt = (d / "info").read_text(errors="replace") if (d / "info").exists() else ""
        t = re.search(r"(?im)^threads[^0-9]*([0-9]+(?:\.[0-9]+)?)", txt)
        g = re.search(r"(?im)^ *(?:declared )?peak[^0-9]*([0-9]+(?:\.[0-9]+)?)", txt)
        res.append((d.name, float(t.group(1)) if t else 6.0, float(g.group(1)) if g else 10.0))
    return res


def ok_now(kind: str, thr: int, rm_floor: int) -> tuple[bool, str]:
    if (P / "arc_quiet.request").exists() or (P / "arc_quiet.lock").exists():
        return False, "ARC window"
    ol = other_locks(None)
    tsum = sum(t for _, t, _ in ol)
    if kind == "rm" and tsum > 2:
        return False, f"RealMLP runs alone: others declare {tsum} threads"
    if any(t >= 4 for _, t, _ in ol):
        return False, "another heavy job: " + ", ".join(n for n, t, _ in ol if t >= 4)
    if tsum + thr > (13 if thr == 1 else 12):
        return False, f"threads {tsum} + {thr}"
    f = free_pct()
    floor = rm_floor if kind == "rm" else 30
    if f < floor:
        return False, f"memory free {f} % < {floor} %"
    return True, f"others {tsum} threads, free {f} %"


class Terminated(Exception):
    """SIGTERM or SIGINT reached the runner (coordinator 10-11 07:4x item 2): it ends like any failure, with FAILED."""


def end_job(pr) -> None:
    """End a job and its session (it was started with start_new_session=True, so its pid is its process group)."""
    if pr is None or pr.poll() is not None:
        return
    print(f"[{ts()}] ending job pid {pr.pid} before its lock is removed", flush=True)
    try:
        if hasattr(os, "killpg"):
            os.killpg(pr.pid, signal.SIGTERM)
        else:
            pr.terminate()
        try:
            pr.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(pr.pid, signal.SIGKILL) if hasattr(os, "killpg") else pr.kill()
            pr.wait()
    except ProcessLookupError:
        pass


class Runner:
    def __init__(self, fam: str, settings: list[str], legs: list[str] | None, rm_floor: int, dry: bool, rm_seeds: list[int] | None = None):
        self.fam, self.settings, self.rm_floor, self.dry, self.legs, self.rm_seeds_arg = fam, settings, rm_floor, dry, legs, rm_seeds
        self.jobs = [j for j in s_spec.jobs(fam, settings, legs) if j["leg"] != "rm" or rm_seeds is None or j["seed"] in rm_seeds]
        self.state_path = OUT / f"s_learn_{fam}_{''.join(settings)}.json"
        self.failed = OUT / f"s_learn_{fam}.FAILED"
        self.done = OUT / f"s_learn_{fam}_{''.join(settings)}.DONE"
        self.st = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else \
            {"family": fam, "settings": settings, "root": str(ROOT), "jobs": {}, "rm_dropped": [], "smoke": []}
        self.smoked = False
        self.ending = False  # set once a failure or a stop signal is being handled: a second signal then waits for the cleanup

    def save(self) -> None:
        if self.dry:
            return
        OUT.mkdir(parents=True, exist_ok=True)
        self.st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")  # replaced whole: a stop signal mid-save leaves the last complete state
        tmp.write_text(json.dumps(self.st, indent=1), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def fail(self, why: str) -> None:
        """FAILED names the settings run that failed (first word); a DONE of this settings run never survives a failure. Another settings
        run's FAILED already there is kept (this line goes under it), so its first word and its reason stay."""
        self.ending = True
        print(f"[{ts()}] FAILED: {why}", flush=True)
        if not self.dry:
            OUT.mkdir(parents=True, exist_ok=True)
            self.done.unlink(missing_ok=True)
            tag = "".join(self.settings)
            prev = self.failed.read_text(errors="replace") if self.failed.exists() else ""
            keep = prev if prev.strip() and prev.split(" ", 1)[0] != tag else ""
            self.failed.write_text(keep + f"{tag} {time.strftime('%Y-%m-%d %H:%M:%S')} {why}\n")
        raise SystemExit(1)

    def clear_own_markers(self) -> None:
        """Coordinator 03:1x (stale DONE): a new run removes its own old DONE and its own old FAILED before anything else. A FAILED written by
        another settings run in this root (its first word) is not this run's to clear: the runner refuses instead of hiding it."""
        if self.done.exists():
            print(f"[{ts()}] removing this run's old {self.done.name} ({self.done.read_text().strip()})", flush=True)
            self.done.unlink()
        if self.failed.exists():
            txt = self.failed.read_text(errors="replace").strip()
            if txt.split(" ", 1)[0] == "".join(self.settings):
                print(f"[{ts()}] removing this run's old {self.failed.name}: {txt}", flush=True)
                self.failed.unlink()
            else:
                print(f"[{ts()}] {self.failed.name} belongs to another run here: {txt}; not starting", flush=True)
                raise SystemExit(1)

    def wait_gate(self, kind: str, thr: int, name: str) -> None:
        n, k = 0, 0
        while True:
            if (P / "f2s_STOP").exists():
                print(f"[{ts()}] f2s_STOP present before {name}: ending cleanly", flush=True)
                self.save()
                raise SystemExit(3)
            if (P / "f2s_HOLD").exists():
                if k % 20 == 0:
                    print(f"[{ts()}] f2s_HOLD: paused before {name} (no lock held)", flush=True)
                k += 1
                time.sleep(30)
                n = 0
                continue
            ok, why = ok_now(kind, thr, self.rm_floor)
            n = n + 1 if ok else 0
            if n >= 2:
                print(f"[{ts()}] gate open for {name}: {why}", flush=True)
                return
            if not ok and k % 20 == 0:
                print(f"[{ts()}] waiting before {name}: {why}", flush=True)
            k += 1
            time.sleep(30)

    def run_proc(self, name: str, argv: list[str], log: Path, thr: int, gb: float, mins: int, kind: str,
                 launch: dict | None = None) -> tuple[int, str, float]:
        """launch: seed 0's own way of starting the script (598c's fj seeds in f2_run: $PY -u, OMP_NUM_THREADS only as seed 0 set it)."""
        lock = P / f"f2s_{self.fam}_{name}.lock"
        lock.mkdir()
        try:
            (lock / "info").write_text(f"owner: PRC F2 Package S learner ({self.fam} {name}), coordinator main\npid: {os.getpid()}\npeak_gb: {gb}\n"
                                       f"threads: {thr}\nstart: {time.strftime('%Y-%m-%d %H:%M')}\n"
                                       f"expected_end: {time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time() + 60 * mins))} KST\n"
                                       f"job: {' '.join(argv)[:200]}\n")
            if launch is None:
                env = {**os.environ, "OMP_NUM_THREADS": str(thr), "OPENBLAS_NUM_THREADS": "1"}
                flags = ["-X", "utf8", "-u"]
            else:
                env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
                env.update(launch["env"])
                flags = list(launch["py_flags"])
            env.update({"PYTHONIOENCODING": "utf-8", "PYTHONFAULTHANDLER": "1", "PATH": str(Path(PY).parent) + os.pathsep + os.environ.get("PATH", "")})
            t0 = time.time()
            stop = {"mem": False}
            pr = None
            try:
                with open(log, "w", encoding="utf-8") as fh:
                    pr = subprocess.Popen(["nice", "-n", "10", "/usr/bin/time", "-l", PY, *flags, *argv], cwd=ROOT, stdout=fh,
                                          stderr=subprocess.STDOUT, env=env, start_new_session=True)

                    def watch():
                        while pr.poll() is None:
                            if free_pct() < 20:
                                stop["mem"] = True
                                print(f"[{ts()}] WATCHDOG: memory free below 20 % during {name}; killing pid {pr.pid}", flush=True)
                                try:
                                    os.killpg(pr.pid, signal.SIGTERM)
                                    time.sleep(10)
                                    os.killpg(pr.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                                return
                            time.sleep(10)
                    wt = threading.Thread(target=watch, daemon=True) if kind == "rm" else None
                    if wt:
                        wt.start()
                    rc = pr.wait()
            except BaseException:
                self.ending = True  # from here a stop signal waits: the job must end before its lock goes
                end_job(pr)  # never leave a job running once its lock is gone (fix C: the runner then writes FAILED)
                raise
            tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
            if stop["mem"]:
                (OUT / f"s_learn_{self.fam}_mem_stop.flag").write_text(f"{name}\n")
                self.fail(f"memory watchdog stopped {name} (free below 20 %)")
            return rc, tail, time.time() - t0
        finally:
            shutil.rmtree(lock, ignore_errors=True)

    def smoke(self) -> None:
        for attempt in (1, 2):
            log = ROOT / "logs" / f"f2s_{self.fam}_rm_smoke_{attempt}.log"
            if self.dry:
                print(f"[dry] smoke: OMP_NUM_THREADS=1 PYTHONFAULTHANDLER=1 {PY} -X utf8 -u research/f2/rm_smoke.py --device mps --threads 1", flush=True)
                self.smoked = True
                return
            self.wait_gate("rm", 1, "rm_smoke")
            rc, tail, sec = self.run_proc("rm_smoke", ["research/f2/rm_smoke.py", "--device", "mps", "--threads", "1"], log, 1, 8, 3, "rm")
            ok = rc == 0 and "rm_smoke OK" in tail
            self.st["smoke"].append({"attempt": attempt, "rc": rc, "sec": round(sec, 1), "ok": ok, "time": time.strftime("%H:%M:%S")})
            self.save()
            print(f"[{ts()}] RealMLP smoke attempt {attempt}: rc {rc} in {sec:.0f} s -> {'OK' if ok else 'FAILED'}", flush=True)
            if ok:
                self.smoked = True
                return
        self.st["rm_status"] = "smoke failed twice; no RealMLP seed was started"
        self.save()
        self.fail("RealMLP smoke fit failed twice")

    def links(self, j: dict) -> None:
        for link, target in j["links"]:
            p = ROOT / link
            if p.is_symlink():
                if os.readlink(p) != target:
                    self.fail(f"{link} links to {os.readlink(p)}, not {target}")
            elif p.exists():
                self.fail(f"{link} exists and is not a link")
            elif not (p.parent / target).exists():
                self.fail(f"link target {target} missing")
            elif self.dry:
                print(f"[dry] would link {link} -> {target}", flush=True)
            else:
                os.symlink(target, p)
                print(f"[{ts()}] link {link} -> {target}", flush=True)

    def on_signal(self, signum, frame) -> None:
        if self.ending:
            return  # already ending: let the job end and FAILED be written
        self.ending = True
        raise Terminated(f"signal {signal.Signals(signum).name}")

    def run(self) -> int:
        """Coordinator 03:5x fix C: any exception ends in FAILED (F3's phase8 waits for DONE or FAILED); 07:4x item 2: so do SIGTERM and SIGINT
        (trapped here; SIGKILL cannot be, F3's liveness check covers it). SystemExit passes as it is: fail() itself, f2s_STOP (rc 3, a
        deliberate stop: neither marker) and another run's FAILED (left untouched)."""
        old = {}
        if threading.current_thread() is threading.main_thread():
            for sg in (signal.SIGTERM, signal.SIGINT):
                old[sg] = signal.signal(sg, self.on_signal)
        try:
            return self._run()
        except (Exception, KeyboardInterrupt) as e:  # noqa: BLE001
            traceback.print_exc()
            self.fail(f"stopped by {e}" if isinstance(e, Terminated) else f"unhandled {type(e).__name__}: {str(e)[:300]}")
        finally:
            for sg, h in old.items():
                signal.signal(sg, h)
        return 1

    def _run(self) -> int:
        if not self.dry:  # first: no later check may overwrite another run's FAILED or leave this run's old DONE standing
            self.clear_own_markers()
        bad = [f"{f} {sha16(ROOT / f) if (ROOT / f).exists() else 'missing'} (want {h})" for f, h in CODE.items()
               if not (ROOT / f).exists() or sha16(ROOT / f) != h]
        used = {j["cmd"].split()[0] for j in self.jobs}
        bad = [b for b in bad if b.split()[0] in used]
        if bad:
            self.fail("code sha16: " + "; ".join(bad))
        if any(j["root"] == "f2_run" for j in self.jobs) and not self.dry:
            # coordinator 10-11 02:2x: the 598c fj seeds run in f2_run only after research/f2/s_fjpre.py recorded and asserted that code and
            # inputs are the ones seed 0 used; re-check those sha16s now
            pre = OUT / f"s_fjpre_{self.fam}.json"
            if not pre.exists():
                self.fail(f"{pre.name} missing: run research/f2/s_fjpre.py first")
            pr = json.loads(pre.read_text(encoding="utf-8"))
            if not pr.get("pass"):
                self.fail(f"{pre.name} did not pass: {pr.get('problems')}")
            moved = [f for grp in ("code", "inputs") for f, v in pr[grp].items() if not (ROOT / f).exists() or sha16(ROOT / f) != v["sha16"]]
            if moved:
                self.fail(f"changed since {pre.name}: {moved}")
            print(f"[{ts()}] f2_run pre-check {pre.name} ({pr['written']}): code {len(pr['code'])} and inputs {len(pr['inputs'])} unchanged", flush=True)
        print(f"[{ts()}] S learner runner {self.fam} {self.settings} in {ROOT}: {len(self.jobs)} jobs" + (" (dry)" if self.dry else ""), flush=True)
        dropped = set(self.st["rm_dropped"])
        earlier = set()  # outputs of the jobs before this one in this run (a CatBoost ap seed reads the LightGBM ap seed made just before)
        for j in self.jobs:
            earlier_now, earlier = set(earlier), earlier | {j["out"]}
            name, rec = j["name"], self.st["jobs"].get(j["name"], {})
            out = ROOT / j["out"]
            if j["leg"] == "rm" and j["seed"] in dropped:
                print(f"[{ts()}] skip {name}: seed {j['seed']} dropped after two crashes", flush=True)
                continue
            if rec.get("status") == "done" and out.exists():
                print(f"[{ts()}] done earlier: {name}", flush=True)
                continue
            if out.exists() or out.is_symlink():
                self.fail(f"{j['out']} exists without this runner's record; refusing")
            for need in j["needs"]:
                if not (ROOT / need).exists() and not (self.dry and need in earlier_now):
                    self.fail(f"{name} reads {need}, which does not exist (the LightGBM ap seed of the same number runs first)")
            self.links(j)
            argv = j["cmd"].split()
            if self.dry:
                print(f"[dry] {name} ({j['kind']}, {j['threads']} threads, {j['peak_gb']} GB, ~{j['mins']} min): {j['cmd']}", flush=True)
                continue
            if j["kind"] == "rm" and not self.smoked:
                self.smoke()
            attempts = rec.get("attempts", 0)
            while True:
                self.wait_gate(j["kind"], j["threads"], name)
                attempts += 1
                log = ROOT / "logs" / f"f2s_{self.fam}_{name}_a{attempts}.log"
                print(f"[{ts()}] start {name} attempt {attempts}: {j['cmd']}", flush=True)
                rc, tail, sec = self.run_proc(name, argv, log, j["threads"], j["peak_gb"], j["mins"], j["kind"], j.get("launch"))
                pk = re.search(r"([0-9]+)\s+peak memory footprint", tail)
                crash = rc < 0 or rc >= 128 or bool(CRASH.search(tail))
                rec = {"status": "running", "cmd": j["cmd"], "argv": argv, "attempts": attempts, "rc": rc, "real_s": round(sec),
                       "peak_gb": round(int(pk.group(1)) / 2**30, 2) if pk else None, "log": str(log.relative_to(ROOT)), "crash": crash,
                       "launch": j.get("launch") or {"py_flags": ["-X", "utf8", "-u"], "env": {"OMP_NUM_THREADS": str(j["threads"]), "OPENBLAS_NUM_THREADS": "1"}}}
                if rc == 0 and out.exists():
                    rec.update(status="done", sha16=sha16(out))
                    self.st["jobs"][name] = rec
                    self.save()
                    print(f"[{ts()}] end {name}: rc 0 in {sec:.0f} s, peak {rec['peak_gb']} GB, {j['out']} {rec['sha16']}", flush=True)
                    break
                self.st["jobs"][name] = {**rec, "status": "crashed" if crash else "failed"}
                self.save()
                print(f"[{ts()}] end {name}: rc {rc} ({'crash' if crash else 'failure'}) in {sec:.0f} s; log {log.name}", flush=True)
                if j["kind"] != "rm" or not crash:
                    self.fail(f"{name} rc {rc} (log {log.relative_to(ROOT)})")
                if out.exists():
                    out.unlink()  # a crashed run's partial output never stays
                if attempts >= 2:
                    if "full" in self.settings:  # full must use the seeds the sandbox's val and fwd used: no silent seed change
                        self.fail(f"{name} crashed twice in full; the val and fwd average used this seed: stop and tell the coordinator")
                    dropped.add(j["seed"])
                    self.st["rm_dropped"] = sorted(dropped)
                    self.smoked = False  # the next RealMLP seed starts after a fresh smoke fit
                    self.save()
                    print(f"[{ts()}] seed {j['seed']} dropped after two crashes", flush=True)
                    break
                self.smoked = False  # a fresh smoke fit before the retry
                self.smoke()
        rm_jobs = [j for j in self.jobs if j["leg"] == "rm"]
        if rm_jobs and not self.dry:
            done = sorted({j["seed"] for j in rm_jobs} - dropped)
            done = [k for k in done if all(self.st["jobs"].get(f"rm_{s}_s{k}", {}).get("status") == "done" for s in self.settings)]
            self.st["rm_seeds_done"] = done
            self.st["rm_use_own"] = len(done) >= 3
            self.st["rm_tag"] = s_spec.rm_tag(self.fam, done, fj="fj" in self.settings) if len(done) >= 3 else None
            self.save()
            print(f"[{ts()}] RealMLP seeds finished {done}; chain C {'takes its own RealMLP ' + self.st['rm_tag'] if len(done) >= 3 else 'keeps the family RealMLP (fewer than 3 own seeds)'}", flush=True)
        if not self.dry:
            self.self_check(rm_jobs)
            self.done.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        print(f"[{ts()}] runner done", flush=True)
        return 0

    def self_check(self, rm_jobs: list[dict]) -> None:
        """Rule 1 on this run's own jobs (research/f2/s_check.py cmds and rows, in process): the DONE file means ran and checked."""
        import s_check
        s_check.ROOT, s_check.OUT, s_check.LEGS = ROOT, OUT, self.legs
        if rm_jobs:  # the RealMLP seeds this run finished (three or more), else none of them is checked or used
            done = self.st.get("rm_seeds_done") or []
            rs = done if len(done) >= 3 else []
        else:  # no RealMLP job here: --legs left rm out (s_check's leg filter does it) or --rm_seeds none
            rs = [] if self.rm_seeds_arg is not None else None
        tag = "".join(self.settings)
        res = {}
        for kind, fn in (("cmds", lambda: s_check.check_cmds(self.fam, self.settings, str(self.state_path), rs)),
                         ("rows", lambda: s_check.check_rows(self.fam, self.settings, rs))):
            r = fn()
            r.update(root=str(ROOT), written=time.strftime("%Y-%m-%d %H:%M:%S"), legs=self.legs, rm_seeds=rs)
            (OUT / f"s_check_{kind}_{self.fam}_{tag}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
            res[kind] = r["pass"]
        self.st["self_check"] = res
        self.save()
        print(f"[{ts()}] construction check of this run (s_check cmds, rows): {res}", flush=True)
        if not all(res.values()):
            self.fail(f"construction check failed: {res} (research/f2/out/s_check_*_{self.fam}_{tag}.json)")


def main() -> int:
    a = sys.argv[1:]
    if len(a) < 2:
        raise SystemExit(__doc__)
    fam, settings = a[0], a[1].split(",")
    s_spec.fam(fam)
    if not set(settings) <= {"val", "fwd", "full", "fj"} or (("full" in settings or "fj" in settings) and len(settings) > 1):
        raise SystemExit("settings: val,fwd (sandbox) or full or fj alone")
    legs = a[a.index("--legs") + 1].split(",") if "--legs" in a else None
    rm_floor = int(a[a.index("--rm_floor") + 1]) if "--rm_floor" in a else 30
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    F = s_spec.fam(fam)
    if settings == ["fj"]:
        roots = {F["legs"][x]["fj"]["root"] for x in (legs or s_spec.ORDER)}
        if len(roots) != 1:
            raise SystemExit("fj runs per root: " + "; ".join(f"--legs {x} from {F['legs'][x]['fj']['root']}" for x in (legs or s_spec.ORDER)))
    if "--any_root" not in a:
        want = {F["sbx"]} if settings[0] in ("val", "fwd") else ({"final_run"} if settings == ["full"] else
                                                                 {F["legs"][x]["fj"]["root"] for x in (legs or s_spec.ORDER)})
        if ROOT.name not in want or ROOT.parent != P:
            raise SystemExit(f"run from ~/nova-adsb/prcm/{'|'.join(sorted(want))} (cwd {ROOT}); --any_root only for dry runs")
    rm_seeds = None
    if "--rm_seeds" in a:
        v = a[a.index("--rm_seeds") + 1]
        rm_seeds = [] if v == "none" else [int(x) for x in v.split(",")]
    if "--rm_seeds_from" in a:  # the sandbox's own record of the RealMLP seeds that finished val and fwd
        sbp = Path(a[a.index("--rm_seeds_from") + 1])
        if sbp.name != f"s_learn_{fam}_valfwd.json":
            raise SystemExit("--rm_seeds_from must be this family's sandbox s_learn_<family>_valfwd.json")
        sb = json.loads(sbp.read_text(encoding="utf-8"))
        # coordinator 03:1x: an unfinished record must not read as "none": the sandbox run's DONE (written only after its own construction
        # check) must sit beside it, the record must carry its RealMLP result, and every seed it names must be done in val and fwd
        if sb.get("family") != fam or not (sbp.parent / f"s_learn_{fam}_valfwd.DONE").exists():
            raise SystemExit(f"--rm_seeds_from: the sandbox valfwd run is not finished (no s_learn_{fam}_valfwd.DONE beside {sbp})")
        if "rm_seeds_done" not in sb or "rm_use_own" not in sb or not all((sb.get("self_check") or {"x": False}).values()):
            raise SystemExit("--rm_seeds_from: the record has no RealMLP result or no passed self-check")
        bad = [f"rm_{s}_s{k}" for k in sb["rm_seeds_done"] for s in ("val", "fwd") if sb["jobs"].get(f"rm_{s}_s{k}", {}).get("status") != "done"]
        if bad:
            raise SystemExit(f"--rm_seeds_from: jobs not done in the sandbox record: {bad}")
        from_sb = sb["rm_seeds_done"] if sb["rm_use_own"] else []
        if rm_seeds is not None and rm_seeds != from_sb:
            raise SystemExit(f"--rm_seeds {rm_seeds} is not the sandbox's {from_sb}")
        rm_seeds = from_sb
    wants_rm = (legs is None or "rm" in legs) and settings[0] in ("full", "fj")
    if wants_rm and rm_seeds is None:
        raise SystemExit("full and fj take the RealMLP seeds the sandbox finished: give --rm_seeds or --rm_seeds_from (or --legs without rm)")
    if wants_rm and rm_seeds and (len(rm_seeds) < 3 or not set(rm_seeds) <= set(F["legs"]["rm"]["seeds"])):
        raise SystemExit(f"--rm_seeds {rm_seeds}: three or more of the planned seeds {F['legs']['rm']['seeds']}, or none")
    return Runner(fam, settings, legs, rm_floor, "--dry" in a, rm_seeds).run()


if __name__ == "__main__":
    sys.exit(main())
