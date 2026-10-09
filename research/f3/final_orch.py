# SPDX-License-Identifier: GPL-3.0-only
"""F3 final rebuild orchestrator (Mac, the final root = the current working directory): runs the full-setting jobs of 514's three chains in dependency
order. Each job goes through research/f3/mac_step.sh (PROTOCOL: waits without a lock, then lock f3_<job>.lock with declared peak and
threads, nice 10, /usr/bin/time -l). This driver caps its own concurrency (--budget threads in flight, --max_jobs jobs) and keeps the
groups apart: an "mps_alone" job (RealMLP: 13 GB RSS, 33 GB footprint) starts only when none of this driver's other jobs runs, and no
other job starts beside it; "mps" jobs (TabM) never run beside an "mps_alone" job.
A job is done when its outputs exist and its step log ended with rc 0; outputs that already exist are never rebuilt. HOLD_F3 in the root
pauses new launches (running jobs go on), e.g. so that a waiting step outside this driver gets its lock first. State and a line per
event go to research/f3/out/orch_<phase>.json and logs/orch_<phase>.log. Stops launching (finishes running jobs) when STOP_F3 exists.
    nohup python -u research/f3/final_orch.py <phase> [--budget 10] [--max_jobs 2] >> logs/orch_<phase>.log 2>&1 &
Phases: lgb (LightGBM legs, g_band, LightGBM seed averages), cb (CatBoost legs and averages), rm (RealMLP legs and averages),
        tabm (prep + TabM leg).
Fit cache (research/f3/fitwrap.py, coordinator approval 2026-10-08 23:2x KST): --wrap save:<dir> or load:<dir> runs the LightGBM and
CatBoost leg scripts (model3, model3ap, cat_mu, cat_muap) through the wrapper, unchanged. --root runs the same jobs in another root (the
interim root), --prefix keeps its lock names apart, --defer_to <root> holds a job while the same job is still running in that root (its
stored fits are then loaded instead of refit). --cap bounds the threads of all F3 locks together (this driver's jobs, launched or waiting,
plus every other f3_*.lock in the lock folder PRC_LOCK_DIR). Every CatBoost job runs at 8 threads (coordinator 2026-10-08 23:5x KST: R16-D refit cbwin
at 8 threads where 514 ran it at 10 and reproduced the deployed mu_cat exactly, research/2026-10-08-r16-d-nd-legs.md).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

P = Path(os.environ.get("PRC_LOCK_DIR", ".")).resolve()  # lock folder of research/f3/mac_step.sh
ROOT = Path.cwd()
FITWRAP = str(ROOT / "research" / "f3" / "fitwrap.py")
WRAPPED = ("src/model3.py", "research/r13_f3/model3ap.py", "src/cat_mu.py", "research/r13_f3/cat_muap.py")
PY = sys.executable
LGB = "--feats feats_v27 --es internal --refit --wpow 1 --leaves 255 --mdl 100 --no_direct --clf_drop_prefix m_,ab_ --full_mult 1.0"


def jobs_for(phase: str) -> list[dict]:
    J = []
    if phase == "lgb":
        for k in range(4):
            J.append(dict(name=f"skwin_s{k}", th=6, gb=6.5, deps=[], out=[f"cache/full_m6r27skwin_s{k}.parquet"],
                          cmd=f"src/model3.py --setting full --seed {k} --tag m6r27skwin {LGB} --threads 6 --mu_sep LIRF --anchor win"))
            J.append(dict(name=f"sk_s{k}", th=4, gb=6.5, deps=[], out=[f"cache/full_m6r27sk_s{k}.parquet"],
                          cmd=f"src/model3.py --setting full --seed {k} --tag m6r27sk {LGB} --threads 4 --mu_sep LIRF"))
        J.append(dict(name="skap_s0", th=6, gb=6.5, deps=["skwin_s3"], out=["cache/full_m6r27skap_s0.parquet"],
                      cmd=f"research/r13_f3/model3ap.py --setting full --seed 0 --tag m6r27skap {LGB} --threads 6 --mu_sep all"))
        for g, b in (("m6r27skwing", "m6r27skwin"), ("m6r27skg", "m6r27sk"), ("m6r27skapg", "m6r27skap")):
            dep = {"m6r27skwin": "skwin_s0", "m6r27sk": "sk_s0", "m6r27skap": "skap_s0"}[b]
            J.append(dict(name=f"gband_{g}", th=1, gb=5.0, deps=[dep], out=[f"cache/full_{g}_s0.parquet"],
                          cmd=f"src/g_band.py --setting full --seed 0 --tag {g} --base {b} --feats feats_v27"))
        J.append(dict(name="avg_skwingx4", th=1, gb=2.0, deps=["gband_m6r27skwing", "skwin_s1", "skwin_s2", "skwin_s3"],
                      out=["cache/full_m6r27skwingx4_s0.parquet"], cmd="research/f3/avg_legs_final.py m6r27skwing m6r27skwingx4 0,1,2,3 m6r27skwin"))
        J.append(dict(name="avg_skgx4", th=1, gb=2.0, deps=["gband_m6r27skg", "sk_s1", "sk_s2", "sk_s3"],
                      out=["cache/full_m6r27skgx4_s0.parquet"], cmd="research/f3/avg_legs_final.py m6r27skg m6r27skgx4 0,1,2,3 m6r27sk"))
    elif phase == "cb":
        for k in range(3):
            J.append(dict(name=f"cbwin_s{k}", th=8, gb=10.5, deps=[], need=[f"cache/full_m6r27skwin_s{k}.parquet"],
                          out=[f"cache/full_m6r27cbwin_s{k}.parquet"],
                          cmd=f"src/cat_mu.py --setting full --seed {k} --tag m6r27cbwin --base m6r27skwin --feats feats_v27 --anchor win --threads 8"))
            J.append(dict(name=f"cb_s{k}", th=8, gb=10.5, deps=[], need=[f"cache/full_m6r27sk_s{k}.parquet"],
                          out=[f"cache/full_m6r27cb_s{k}.parquet"],
                          cmd=f"src/cat_mu.py --setting full --seed {k} --tag m6r27cb --base m6r27sk --feats feats_v27 --threads 8"))
        J.append(dict(name="cbap_s0", th=8, gb=10.5, deps=[], need=["cache/full_m6r27skap_s0.parquet"], out=["cache/full_m6r27cbap_s0.parquet"],
                      cmd="research/r13_f3/cat_muap.py --setting full --seed 0 --tag m6r27cbap --base m6r27skap --feats feats_v27 --threads 8 --per_ap"))
        J.append(dict(name="avg_cbwinx3", th=1, gb=2.0, deps=["cbwin_s0", "cbwin_s1", "cbwin_s2"], out=["cache/full_m6r27cbwinx3_s0.parquet"],
                      cmd="research/f3/avg_legs_final.py m6r27cbwin m6r27cbwinx3 0,1,2"))
        J.append(dict(name="avg_cbx3", th=1, gb=2.0, deps=["cb_s0", "cb_s1", "cb_s2"], out=["cache/full_m6r27cbx3_s0.parquet"],
                      cmd="research/f3/avg_legs_final.py m6r27cb m6r27cbx3 0,1,2"))
    elif phase == "rm":
        for tag, base, extra in (("m6r27rmwin", "m6r27skwin", " --anchor win"), ("m6r27rm", "m6r27sk", "")):
            for k in range(4):
                J.append(dict(name=f"{tag[5:]}_s{k}", th=1, gb=14.0, group="mps_alone", deps=[], need=[f"cache/full_{base}_s{k}.parquet"],
                              out=[f"cache/full_{tag}_s{k}.parquet"],
                              cmd=f"src/rmlp_mu.py --setting full --seed {k} --tag {tag} --base {base} --feats feats_v27 --device mps{extra} --threads 1"))
        J.append(dict(name="avg_rmwinx4", th=1, gb=2.0, deps=[f"rmwin_s{k}" for k in range(4)], out=["cache/full_m6r27rmwinx4_s0.parquet"],
                      cmd="research/f3/avg_legs_final.py m6r27rmwin m6r27rmwinx4 0,1,2,3"))
        J.append(dict(name="avg_rmx4", th=1, gb=2.0, deps=[f"rm_s{k}" for k in range(4)], out=["cache/full_m6r27rmx4_s0.parquet"],
                      cmd="research/f3/avg_legs_final.py m6r27rm m6r27rmx4 0,1,2,3"))
    elif phase == "tabm":
        J.append(dict(name="tabm_prep", th=1, gb=8.0, deps=[], out=["research/r13_l/x/full/meta.json"],
                      cmd="research/r13_l/prep_x.py --setting full"))
        J.append(dict(name="tmcwin_s0", th=3, gb=6.0, group="mps", deps=["tabm_prep"], need=["cache/full_m6r27skwin_s0.parquet"],
                      out=["cache/full_m6r27tmcwin_s0.parquet"],
                      cmd="research/r13_l/tabm_leg.py --setting full --seed 0 --tag m6r27tmcwin --base m6r27skwin --anchor win --sched cos --epochs 20 "
                          "--patience 99 --device mps --threads 3 --amp bf16"))
    else:
        raise SystemExit(f"unknown phase {phase}")
    return J


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--max_jobs", type=int, default=2)
    ap.add_argument("--only", default="")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--wrap", default="", help="save:<dir> or load:<dir> (absolute dir)")
    ap.add_argument("--prefix", default="", help="lock and log name prefix of this driver's jobs")
    ap.add_argument("--cap", type=float, default=9.0, help="threads of all F3 locks together")
    ap.add_argument("--defer_to", default="", help="another root whose running jobs of the same name this driver waits for")
    a = ap.parse_args()
    ROOT = Path(a.root)
    os.chdir(ROOT)
    wrap = a.wrap.split(":", 1) if a.wrap else None
    assert wrap is None or (wrap[0] in ("save", "load") and Path(wrap[1]).is_absolute()), a.wrap
    J = jobs_for(a.phase)
    if a.only:
        keep = set(a.only.split(","))
        J = [j for j in J if j["name"] in keep]
    st_p = ROOT / "research" / "f3" / "out" / f"orch_{a.phase}.json"
    state = json.loads(st_p.read_text()) if st_p.exists() else {}
    running: dict[str, subprocess.Popen] = {}

    def log(msg):
        print(f"{time.strftime('%m-%d %H:%M:%S')} {msg}", flush=True)

    def done(j):
        return state.get(j["name"], {}).get("rc") == 0 and all((ROOT / o).exists() for o in j["out"])

    def lock_threads(exclude: set) -> float:
        t = 0.0
        for d in P.glob("f3_*.lock"):
            if d.name in exclude:
                continue
            try:
                line = next(x for x in (d / "info").read_text().splitlines() if x.lower().startswith("threads"))
                t += float(line.split(":")[1])
            except Exception:
                t += 6.0
        return t

    def deferred(j) -> bool:
        if not a.defer_to:
            return False
        try:
            other = json.loads((Path(a.defer_to) / "research" / "f3" / "out" / f"orch_{a.phase}.json").read_text())
        except Exception:
            return False
        s_ = other.get(j["name"], {})
        return "start" in s_ and "end" not in s_ and s_.get("rc") is None

    for j in J:
        if all((ROOT / o).exists() for o in j["out"]) and j["name"] not in state:
            state[j["name"]] = {"rc": 0, "note": "outputs existed before this driver"}
    while True:
        if (ROOT / "STOP_F3").exists() and not running:
            log("STOP_F3 present and nothing running: stop")
            break
        for n, p in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            j = next(x for x in J if x["name"] == n)
            ok = rc == 0 and all((ROOT / o).exists() for o in j["out"])
            state[n] = {"rc": rc if ok else (rc or 99), "end": time.strftime("%m-%d %H:%M:%S"), **state.get(n, {})}
            state[n]["rc"] = rc if ok else (rc or 99)
            log(f"{n} ended rc {rc}, outputs {'present' if ok else 'MISSING'}")
            del running[n]
            st_p.write_text(json.dumps(state, indent=1), encoding="utf-8")
        pending = [j for j in J if not done(j) and j["name"] not in running and state.get(j["name"], {}).get("rc") in (None, 0)]
        failed = [n for n, s in state.items() if s.get("rc") not in (None, 0)]
        if not pending and not running:
            log(f"phase {a.phase} finished; failed: {failed}")
            break
        if not (ROOT / "STOP_F3").exists() and not (ROOT / "HOLD_F3").exists():  # HOLD_F3: no new launches for now (a step waits for room)
            own = sum(next(x for x in J if x["name"] == n)["th"] for n in running)
            used = own + lock_threads({f"f3_{a.prefix}{n}.lock" for n in running})
            groups = {next(x for x in J if x["name"] == n).get("group") for n in running}
            for j in pending:
                if len(running) >= a.max_jobs:
                    break
                if any(state.get(d, {}).get("rc") != 0 for d in j["deps"]):
                    continue
                if any(not (ROOT / q).exists() for q in j.get("need", [])):
                    continue
                if j.get("group") == "mps_alone" and running:
                    continue
                if "mps_alone" in groups or (j.get("group") == "mps" and "mps" in groups):
                    continue
                if own + j["th"] > a.budget or used + j["th"] > a.cap:
                    continue
                if deferred(j):
                    continue
                parts = j["cmd"].split()
                if wrap and parts[0] in WRAPPED:
                    parts = [FITWRAP, wrap[0], wrap[1]] + parts
                cmd = ["bash", str(ROOT / "research" / "f3" / "mac_step.sh"), f"{a.prefix}{j['name']}", str(j["gb"]), str(j["th"]),
                       f"logs/final_{j['name']}.log", PY, "-X", "utf8", "-u", *parts]
                with open(ROOT / "logs" / f"step_final_{j['name']}.log", "w") as f:
                    running[j["name"]] = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
                state[j["name"]] = {"start": time.strftime("%m-%d %H:%M:%S"), "cmd": " ".join(parts)}
                log(f"launched {a.prefix}{j['name']} ({j['th']} threads): {' '.join(parts)}")
                used += j["th"]
                own += j["th"]
                groups.add(j.get("group"))
                st_p.write_text(json.dumps(state, indent=1), encoding="utf-8")
        time.sleep(30)
    st_p.write_text(json.dumps(state, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
