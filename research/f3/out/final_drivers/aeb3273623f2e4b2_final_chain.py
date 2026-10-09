"""F3 final rebuild of 514 on the four-month file: the chain part after the legs, in the root ~/nova-adsb/prcm/final_run.
Same tools and arguments as 514's builders (research/r10_b/sb_build_mac3.py for chain B, research/r13_l/sb_build_mac4.py for chain A,
the R13-F3 ap builder for chain C, research/r16_b/sb_build_r16b.py for the three-chain average), with these differences, all recorded:
  - blend_fixed, blend_add, ctx_corr3 and cavg run only the full setting; their val and fwd outputs are today's read-only copies in
    cache/ (byte-identical to the build sandboxes). blend_fixed full runs without --check (the check reads the reference legs' full files of
    the old test rows; the flag does not change the output, src/blend_fixed.py:86-95).
  - XGBoost (m6r25xg) is not refit: its weight is exactly 0 in all three full blends (0.1252 / 0.4951 / 0 / 0.3798; checked); its full
    file is a placeholder with mu_xgb = 0 (blend step checks zeros and ones give the same output).
  - ctx_corr3 uses the union of the four corrector fit caches (research/f3/fitcache_union); the shas in its command are the final tables'.
  - The tail (artmm .. gjoint a) runs all three settings with the unchanged tools on tables rebuilt with --force in this root
    (step 'tables', order antiwin link, gkey, gkeyx, gkeyxn v2, artmm, arthist); its val and fwd outputs are compared with 514's.
Each tool run goes through research/f3/mac_step.sh. Every target is checked to not exist before its step.
--cfg s75 (second rebuild, coordinator 10-09 23:2x, the Default final = 531): every corrector stage at --scale2 0.75 with tag ..x50ws75,
then cavg, manifest, tail and chainavg3 as above; blend and add are 514's (shared). Its val and fwd inputs are 531's caches cloned from
var1_sbx (research/f3/collect_s75.py). Queue numbers +1000, step and state names prefixed s75_. Without --cfg everything is as before.
    python -u research/f3/final_chain.py tables
    python -u research/f3/final_chain.py chain <A|B|C> [--from <step>] [--to <step>] [--cfg s75]
    python -u research/f3/final_chain.py avg3 [--cfg s75]
State: research/f3/out/final_chain_[s75_]<A|B|C>.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.home() / "nova-adsb" / "prcm" / "final_run"
PY = str(Path.home() / "nova-adsb" / "venv311" / "bin" / "python")
OUT = ROOT / "research" / "f3" / "out"
REF = "--ref_base m6r25sg --ref_tag m6r25sgoof4 --ref_learners m6r25cb:mu_cat,m6r25xg:mu_xgb,m6r25rm:mu_rmlp --folds 10"
MID = "adsbgeondctxv3x50w"
STEPS = ["blend", "add", "corr", "cavg", "manifest", "am", "fh", "gkc", "gkcx", "xncr", "db", "aw", "gp", "gl", "gja"]
SUF = {"am": "am", "fh": "amfh", "gkc": "amfhgkc", "gkcx": "amfhgkcx", "xncr": "amfhgkcxncr", "db": "amfhgkcxncrdb", "aw": "amfhgkcxncrdbaw",
       "gp": "amfhgkcxncrdbawgp", "gl": "amfhgkcxncrdbawgpgl", "gja": "amfhgkcxncrdbawgpglgja"}
EXTRA = ("src/blend_fixed.py,src/ctx_corr2.py,src/ctx_corr3.py,src/feats_ctx2.py,src/feats_aiu.py,research/ctx_corr/freeze_geo.py,research/obs2/corr_plus.py,"
         "research/obs2/obs_rows.py,research/obs2/coverage.py,research/obs2/osm_stands.py,research/adsb/gprobe/build_nd.py,research/r10_b/cavg.py,"
         "research/f3/fitwrap.py")  # fitwrap: the fit cache the legs ran through (sha16 54114020d2aa3d3c, coordinator 10-09 00:2x)
EXTRA_A = EXTRA + ",research/r13_l/prep_x.py,research/r13_l/tabm_leg.py,research/r13_l/xrfm_leg.py,research/r13_l/blend_add.py"
EXTRA_C = EXTRA + ",research/r13_f3/model3ap.py,research/r13_f3/cat_muap.py"
CHAINS = {
    "A": dict(T="m6r27wincbrmfw4sv434", base_leg="m6r27skwingx4", learners="m6r27cbwinx3:mu_cat,m6r25xg:mu_xgb,m6r27rmwinx4:mu_rmlp",
              TA="m6r27wincbrmfw4sv434tmc", add_leg="m6r27tmcwin:mu_tabm", gleg="m6r27skwing", num0=2000, extra=EXTRA_A),
    "B": dict(T="m6r27skcbrmfw4sv434", base_leg="m6r27skgx4", learners="m6r27cbx3:mu_cat,m6r25xg:mu_xgb,m6r27rmx4:mu_rmlp",
              TA=None, add_leg=None, gleg="m6r27skg", num0=2100, extra=EXTRA),
    "C": dict(T="m6r27apcbrmfw4sv114", base_leg="m6r27skapg", learners="m6r27cbap:mu_cat,m6r25xg:mu_xgb,m6r27rmx4:mu_rmlp",
              TA=None, add_leg=None, gleg="m6r27skapg", num0=2200, extra=EXTRA_C),
}
CFGS = {"s50": dict(mid=MID, scale2="0.5", dnum=0, pre="", src_note="514's caches"),
        "s75": dict(mid="adsbgeondctxv3x50ws75", scale2="0.75", dnum=1000, pre="s75_", src_note="531's caches cloned from var1_sbx")}
CFG = CFGS["s50"]
DECL = {"blend": (2.5, 1), "add": (2.5, 1), "corr": (4.6, 4), "cavg": (3.1, 1), "manifest": (2.5, 1), "am": (3.0, 4), "fh": (2.6, 4), "gkc": (3.9, 4),
        "gkcx": (3.4, 4), "xncr": (3.4, 4), "db": (3.1, 4), "aw": (3.0, 4), "gp": (3.6, 4), "gl": (3.5, 4), "gja": (3.6, 4), "table": (4.0, 1), "avg3": (2.5, 1)}
sys.stdout.reconfigure(line_buffering=True)


def ts() -> str:
    return time.strftime("%m-%d %H:%M:%S")


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def run(name: str, step: str, args: list[str], env: dict | None = None) -> None:
    gb, th = DECL[step]
    th = min(th, int(os.environ.get("F3_MAXTH", th)))  # coordinator 10-09 13:5x: at most 3 threads beside F2's fj CatBoosts (OMP_NUM_THREADS = th)
    log = f"logs/chain_{name}.log"
    cmd = ["bash", "research/f3/mac_step.sh", name, str(gb), str(th), log, PY, "-X", "utf8", "-u", *args]
    print(f"[{ts()}] run {' '.join(args)[:230]}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **(env or {})})
    lines = (ROOT / log).read_text(encoding="utf-8", errors="replace").splitlines() if (ROOT / log).exists() else []
    body = [x for x in lines if not x.startswith(" ") and x.strip()][-3:]
    print(f"[{ts()}]   rc {r.returncode} | " + " | ".join(t[:220] for t in body), flush=True)
    if r.returncode != 0:
        raise SystemExit(f"step failed: {name} ({log})")


def must_be_new(*rels: str) -> None:
    for rel in rels:
        if (ROOT / rel).exists():
            raise SystemExit(f"target already exists, refusing to overwrite: {rel}")


def qentry(num: str, spec: str) -> str:
    name = f"{num}_{spec}"
    m = json.loads((ROOT / "preds" / f"manifest_{spec}.json").read_text(encoding="utf-8"))
    (ROOT / "queue" / f"{name}.json").write_text(json.dumps({"manifest": f"preds/manifest_{spec}.json", "kind": "CANDIDATE", "sandbox": True},
                                                            ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy2(ROOT / m["test"]["path"], ROOT / "queue" / f"{name}.parquet")
    return name


def corr_args() -> str:
    c = ROOT / "cache"
    return (f"--feats feats_v25 --adsb adsb_v1nd --adsb_sha {sha16(c / 'adsb_v1nd.parquet')} --folds 10 --rounds 300 --rounds2 400 --max_pred 3600 "
            f"--threads 4 --geo geo_v1nd --geo_sha {sha16(c / 'geo_v1nd.parquet')} --ctx ctx_v3 --ctx_sha {sha16(c / 'ctx_v3.parquet')} --scale2 {CFG['scale2']}")


def xgb_placeholder() -> None:
    """cache/full_m6r25xg_s0.parquet: the final test ids with mu_xgb = 0 (the blend weight of this slot is exactly 0 in the full setting)."""
    import pandas as pd
    p = ROOT / "cache" / "full_m6r25xg_s0.parquet"
    if p.exists():
        return
    f = pd.read_parquet(ROOT / "cache" / "feats_v25.parquet", columns=["MVT_ID_mvt", "src"])
    ids = f.loc[f["src"] == 0, "MVT_ID_mvt"].to_numpy()
    pd.DataFrame({"MVT_ID_mvt": ids, "mu_xgb": 0.0}).to_parquet(p, index=False)
    print(f"[{ts()}] wrote placeholder {p.name} ({len(ids)} rows, mu_xgb = 0)", flush=True)


def check_xgb_weight(T: str) -> dict:
    """The full blend with the placeholder at 0 and at 1 must be identical (weight exactly 0)."""
    import numpy as np
    import pandas as pd
    a = pd.read_parquet(ROOT / "cache" / f"full_{T}_s0.parquet")
    rep = json.loads((ROOT / "notes" / f"blend_fixed_{T}_full_s0.json").read_text(encoding="utf-8"))
    w = rep.get("weights")
    p1 = ROOT / "research" / "f3" / "out" / f"xgbcheck_{T}"
    if p1.exists():
        shutil.rmtree(p1)
    (p1 / "cache").mkdir(parents=True)
    (p1 / "notes").mkdir()
    (p1 / "preds").mkdir()
    for f in (ROOT / "cache").glob("*.parquet"):  # data files as links; src/ as a real copy (src/load.py takes ROOT from its own resolved path)
        if (f.name.startswith(("val_", "full_")) or f.name == "feats_v25.parquet") and f.name != "full_m6r25xg_s0.parquet":
            os.symlink(f, p1 / "cache" / f.name)
    x = pd.read_parquet(ROOT / "cache" / "full_m6r25xg_s0.parquet")
    x["mu_xgb"] = 1.0
    x.to_parquet(p1 / "cache" / "full_m6r25xg_s0.parquet", index=False)
    shutil.copytree(ROOT / "src", p1 / "src", ignore=shutil.ignore_patterns("__pycache__"))
    os.symlink(ROOT / "data", p1 / "data")  # blend_oof.ruler_days reads data/training_2025-01 and -07 (10-09 10:48 failure without it)
    c = CHAINS[[k for k, v in CHAINS.items() if v["T"] == T][0]]
    r = subprocess.run([PY, "-X", "utf8", "-u", "src/blend_fixed.py", "--setting", "full", "--seed", "0", "--tag", T + "_xg1", "--base", c["base_leg"],
                        "--learners", c["learners"], *REF.split()], cwd=p1, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    b = pd.read_parquet(p1 / "cache" / f"full_{T}_xg1_s0.parquet")
    same = all(np.array_equal(a[k].to_numpy(), b[k].to_numpy()) for k in a.columns)
    shutil.rmtree(p1)
    return {"weights": w, "zeros_vs_ones_identical": bool(same)}


def chain(label: str, start: str = "blend", stop: str = "gja") -> None:
    c = CHAINS[label]
    T, TA = c["T"], c["TA"]
    CB = TA or T
    TC = CB + CFG["mid"]
    TCS = TC + "c3"
    pre = CFG["pre"]
    sp = OUT / f"final_chain_{pre}{label}.json"
    st = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {"label": label, "cfg": CFG, "names": {}, "checks": {}}
    todo = STEPS[STEPS.index(start):STEPS.index(stop) + 1]
    num = lambda k: str(c["num0"] + CFG["dnum"] + STEPS.index(k))  # noqa: E731
    if pre and set(todo) & {"blend", "add"}:
        raise SystemExit("--cfg s75 starts at corr (blend and add are 514's, shared)")
    for step in todo:
        if step == "blend":
            xgb_placeholder()
            must_be_new(f"cache/full_{T}_s0.parquet")
            run(f"blend{label}", "blend", f"src/blend_fixed.py --setting full --seed 0 --tag {T} --base {c['base_leg']} --learners {c['learners']} {REF}".split())
            st["checks"]["xgb_weight"] = check_xgb_weight(T)
            print(f"[{ts()}] xgb weight check {st['checks']['xgb_weight']}", flush=True)
            if not st["checks"]["xgb_weight"]["zeros_vs_ones_identical"]:
                raise SystemExit("blend output depends on the XGBoost placeholder: stop")
        elif step == "add":
            if not TA:
                continue
            must_be_new(f"cache/full_{TA}_s0.parquet")
            run(f"add{label}", "add", f"research/r13_l/blend_add.py --setting full --tag {TA} --src {T} --leg {c['add_leg']} --folds 10 --seed 0".split())
        elif step == "corr":
            ca = corr_args()
            for k in (0, 1, 2):
                must_be_new(f"cache/full_{TC}_s{k}.parquet")
                run(f"{pre}corr{label}{k}", "corr", f"src/ctx_corr3.py --setting full --seed {k} --base {CB} --base_seed 0 --tag {TC} {ca} "
                                                    f"--fit_cache research/f3/fitcache_union".split())
                txt = (ROOT / f"logs/chain_{pre}corr{label}{k}.log").read_text(encoding="utf-8", errors="replace")
                st["checks"][f"fit_cache_s{k}"] = [x.strip() for x in txt.splitlines() if "fit cache" in x][-1:]
                sp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
                # coordinator 10-09 02:4x: a corrector cache miss in the final root stops the chain (F2 verified the keys of chain A stage 1)
                fl = st["checks"][f"fit_cache_s{k}"]
                fitted = int(fl[0].split(" loaded, ")[1].split(" fitted")[0]) if fl and " loaded, " in fl[0] else -1
                if fitted != 0:
                    raise SystemExit(f"corrector fit cache: {fl} (fitted must be 0): stop and tell the coordinator")
        elif step == "cavg":
            must_be_new(f"cache/full_{TCS}_s0.parquet")
            run(f"{pre}cavg{label}", "cavg", f"research/r10_b/cavg.py --setting full --seed 0 --tag {TCS} --src {TC} --seeds 0,1,2 --base {CB} --base_seed 0".split())
        elif step == "manifest":
            must_be_new(f"preds/manifest_{TCS}.json", f"preds/test_{TCS}.parquet", f"preds/fwd_{TCS}.parquet")
            lines = [f"logs/{TCS}_{s}_s0.log\t.venv/Scripts/python.exe -u research/r10_b/cavg.py --setting {s} --seed 0 --tag {TCS} --src {TC} "
                     f"--seeds 0,1,2 --base {CB} --base_seed 0" for s in ("val", "fwd", "full")]
            (ROOT / "notes" / f"jobs_{TCS}.txt").write_text(f"# F3 final rebuild (full rebuilt; val, fwd = {CFG['src_note']})\n" + "\n".join(lines) + "\n",
                                                            encoding="utf-8")
            run(f"{pre}manifest{label}", "manifest", ["-m", "src.manifest", TCS, TCS, "pred", "0", "feats_v27", f"notes/jobs_{TCS}.txt"],
                {"MANIFEST_EXTRA_CODE": c["extra"]})
            st["names"]["corr"] = qentry(num("manifest"), TCS)
        else:
            prev = TCS if step == "am" else TCS + SUF[STEPS[STEPS.index(step) - 1]]
            spec = TCS + SUF[step]
            must_be_new(f"preds/manifest_{spec}.json", f"preds/test_{spec}.parquet", f"preds/fwd_{spec}.parquet")
            src_q = st["names"].get("corr" if step == "am" else STEPS[STEPS.index(step) - 1])
            srcm = f"preds/manifest_{prev}.json"
            n = num(step)
            args = {"am": ["tools/artmm.py", "build", srcm, n, "--no-enqueue"],
                    "fh": ["tools/arthist.py", "build", srcm, n, "--no-enqueue"],
                    "gkc": ["tools/gkey.py", "build", src_q, n, "--no-enqueue", "--variant", "avg", "--leg", c["gleg"], "--leg_feats", "feats_v27"],
                    "gkcx": ["tools/gkeyx.py", "build", srcm, n, "--no-enqueue"],
                    "xncr": ["tools/gkeyxn.py", "build", srcm, n, "--no-enqueue", "--variant", "avg_xncr"],
                    "db": ["src/aiu_shift.py", "build", src_q, n, "--group", "block"],
                    "aw": ["tools/antiwin.py", "build", src_q, n, "--no-enqueue"],
                    "gp": ["tools/gpool.py", "build", src_q, n, "--no-enqueue"],
                    "gl": ["tools/glong.py", "build", srcm, n, "--no-enqueue"],
                    "gja": ["tools/gjoint.py", "build", srcm, n, "--mode", "a", "--no-enqueue"]}[step]
            run(f"{pre}{step}{label}", step, args)
            if not (ROOT / "preds" / f"manifest_{spec}.json").exists():
                raise SystemExit(f"{step}: no manifest {spec}")
            st["names"][step] = qentry(n, spec)
        st["done"] = step
        st["updated"] = ts()
        sp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{ts()}] chain {label}: {step} done", flush=True)


def tables() -> None:
    for name, args in (("tab_antiwin", ["tools/antiwin.py", "link", "--force"]), ("tab_gkey", ["tools/gkey.py", "table", "--force"]),
                       ("tab_gkeyx", ["tools/gkeyx.py", "table", "--force"]), ("tab_gkeyxn", ["tools/gkeyxn.py", "table", "--force", "--version", "gkeyxn_v2"]),
                       ("tab_artmm", ["tools/artmm.py", "table", "--force"]), ("tab_arthist", ["tools/arthist.py", "table", "--force"])):
        run(name, "table", args)


def avg3() -> None:
    pre = CFG["pre"]
    names = {k: json.loads((OUT / f"final_chain_{pre}{k}.json").read_text(encoding="utf-8"))["names"]["gja"] for k in ("A", "B", "C")}
    run(f"{pre}avg3", "avg3", ["tools/chainavg3.py", "build", names["A"], names["B"], names["C"], str(2400 + CFG["dnum"]), "--suffix", "skap", "--no-enqueue"])


def main() -> int:
    global CFG
    a = sys.argv[1:]
    if "--cfg" in a:
        CFG = CFGS[a[a.index("--cfg") + 1]]
    os.chdir(ROOT)
    if a[0] == "tables":
        tables()
    elif a[0] == "chain":
        chain(a[1], a[a.index("--from") + 1] if "--from" in a else "blend", a[a.index("--to") + 1] if "--to" in a else "gja")
    elif a[0] == "avg3":
        avg3()
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
