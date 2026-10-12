"""F2 Package S construction checks, rule 1 of the approval (coordinator 10-11 01:17): (a) a new seed's command is its leg's reference
command with --seed changed (and --base only by the B0 link convention, every link pointing at the seed-0 file of the same setting); (b) row
sets and order unchanged (every new seed file holds the seed-0 file's ids in the same order, and the same early-stopping ids); (c) every
averaged leg equals the per-seed mean bit for bit (research/f2/avg_s.py recomputes it in memory); (d) the S chain's blend notes and
manifests differ from the family chain C's only in leg tags and seed lists, with the same code sha16 (S adds only research/f2/avg_s.py).
Run from the root it checks. Exit 0 = PASS, 1 = FAIL; the json under research/f2/out/ holds every comparison.
    python -X utf8 research/f2/s_check.py cmds <598c|l4> <settings> <ran json> [--rm_seeds 4,5,6,7|none] [--legs lgbap,cbap,rm]
           ran json: research/f2/s_learn.py's state file, or a list of {"name","setting","seed","cmd"} written by another runner
    python -X utf8 research/f2/s_check.py rows <598c|l4> <settings> [--rm_seeds ...] [--legs ...]
    python -X utf8 research/f2/s_check.py avg <598c|l4> <settings> [--fj] [--rm_seeds ...]
    python -X utf8 research/f2/s_check.py blend <598c|l4> <S blend tag> <family blend tag> <settings> [--rm_seeds 4,5,6,7|4,5,7|none]
    python -X utf8 research/f2/s_check.py manifest <598c|l4> <S manifest> <family manifest> <S tag>=<family tag>[,...]
    python -X utf8 research/f2/s_check.py feats <root A> <root B> <feats name>   (fj note in s_spec.py: the 2025 rows of a feature file in two roots)
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avg_s  # noqa: E402
import s_spec  # noqa: E402

ROOT = Path.cwd()
OUT = ROOT / "research" / "f2" / "out"
ALLOWED_EXTRA_CODE = {"research/f2/avg_s.py"}


def parse(cmd: str) -> tuple[str, dict]:
    t = shlex.split(cmd)
    i = 0
    while i < len(t) and (t[i].endswith("python") or t[i].endswith("python.exe") or t[i] in ("-u", "-X", "utf8")):
        i += 1
    script, opts = t[i], {}
    i += 1
    while i < len(t):
        if not t[i].startswith("--"):
            raise SystemExit(f"positional argument {t[i]!r} in {cmd!r}")
        if i + 1 < len(t) and not t[i + 1].startswith("--"):
            if t[i] in opts:
                raise SystemExit(f"option {t[i]} twice in {cmd!r}")
            opts[t[i]] = t[i + 1]
            i += 2
        else:
            opts[t[i]] = True
            i += 1
    return script, opts


def cmd_diff(new: str, ref: str, setting: str, seed: int) -> list[str]:
    """Differences of a new seed's command from its reference beyond --seed, --setting and the B0 convention of --base."""
    sn, on = parse(new)
    sr, orf = parse(ref)
    bad = []
    if sn != sr:
        bad.append(f"script {sn} vs {sr}")
    if set(on) != set(orf):
        bad.append(f"options {sorted(set(on) ^ set(orf))} differ")
    for k in sorted(set(on) & set(orf)):
        if k == "--seed":
            if on[k] != str(seed):
                bad.append(f"--seed {on[k]} is not {seed}")
        elif k == "--setting":
            if on[k] != setting:
                bad.append(f"--setting {on[k]} is not {setting}")
        elif k == "--base":
            root = orf[k][:-2] if orf[k].endswith("B0") else orf[k]
            if on[k] not in (root, root + "B0"):
                bad.append(f"--base {on[k]} is neither {root} nor {root}B0")
        elif on[k] != orf[k]:
            bad.append(f"{k} {on[k]} vs {orf[k]}")
    return bad


def sha16(p: Path) -> str:
    return avg_s.sha16(p)


def rm_seeds_arg(a: list[str]) -> list[int] | None:
    """None = the planned own seeds 4-7; "none" = fewer than three finished, chain C keeps the family RealMLP (its jobs are not checked)."""
    if "--rm_seeds" not in a:
        return None
    v = a[a.index("--rm_seeds") + 1]
    return [] if v == "none" else [int(x) for x in v.split(",")]


LEGS: list[str] | None = None  # --legs: the legs run in this root (598c's fj jobs run in two roots)


def planned(f: str, settings: list[str], rm_seeds: list[int] | None) -> list[dict]:
    js = s_spec.jobs(f, settings, LEGS)
    if rm_seeds is not None:
        js = [j for j in js if j["leg"] != "rm" or j["seed"] in rm_seeds]
    return js


def check_cmds(f: str, settings: list[str], ran_path: str, rm_seeds: list[int] | None) -> dict:
    ran = json.loads(Path(ran_path).read_text(encoding="utf-8"))
    ran = ran["jobs"] if isinstance(ran, dict) else {r["name"]: r for r in ran}
    res, ok = {}, True
    for j in planned(f, settings, rm_seeds):
        r = ran.get(j["name"])
        if r is None or r.get("status", "done") != "done":
            res[j["name"]] = {"pass": False, "why": f"not run to completion ({None if r is None else r.get('status')})"}
            ok = False
            continue
        cmd = r["cmd"] if isinstance(r.get("cmd"), str) else " ".join(r["argv"])
        bad = cmd_diff(cmd, j["ref"], j["setting"], j["seed"])
        if cmd.split() != j["cmd"].split():
            bad.append("the command run is not the planned one")
        for link, target in j["links"]:
            p = ROOT / link
            if not p.is_symlink() or os.readlink(p) != target:
                bad.append(f"{link} is not a link to {target}")
        for need in j["needs"]:
            if not (ROOT / need).exists():
                bad.append(f"base file {need} missing")
        res[j["name"]] = {"pass": not bad, "diff_vs_ref": bad, "ref": j["ref"], "ref_src": j["ref_src"], "cmd": cmd}
        ok &= not bad
    return {"check": "cmds", "pass": ok, "jobs": res}


def check_rows(f: str, settings: list[str], rm_seeds: list[int] | None) -> dict:
    res, ok = {}, True
    for j in planned(f, settings, rm_seeds):
        s, tag = j["setting"], j["tag"]
        p, p0 = ROOT / j["out"], ROOT / "cache" / f"{s}_{tag}_s0.parquet"
        if not p.exists():
            res[j["name"]] = {"pass": False, "why": f"missing {j['out']}"}
            ok = False
            continue
        a, b = pd.read_parquet(p, columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy(), pd.read_parquet(p0, columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()
        r = {"rows": int(len(a)), "same_ids_same_order": bool(np.array_equal(a, b))}
        if s != "full":
            e = ROOT / "preds" / f"es_valid_ids_{tag}_{s}_s{j['seed']}.parquet"
            e0 = ROOT / "preds" / f"es_valid_ids_{tag}_{s}_s0.parquet"
            r["es_equal"] = bool(e.exists() and np.array_equal(np.sort(pd.read_parquet(e)["MVT_ID_mvt"].to_numpy()), np.sort(pd.read_parquet(e0)["MVT_ID_mvt"].to_numpy())))
        r["pass"] = all(v for k, v in r.items() if k != "rows")
        res[j["name"]] = r
        ok &= r["pass"]
    return {"check": "rows", "pass": ok, "files": res}


def check_avg(f: str, settings: list[str], fj: bool, rm_seeds: list[int] | None) -> dict:
    F = s_spec.fam(f)
    res, ok = {}, True
    for leg in s_spec.ORDER:
        L = F["legs"][leg]
        a = L["fj_avg"] if fj else L["avg"]
        sfile = L["fj"]["tag"] if fj else L["tag"]
        seeds, new = a["seeds"], a["new"]
        if leg == "rm" and rm_seeds is not None:
            if len(rm_seeds) < 3:
                continue  # chain C keeps the family RealMLP: no own average to check
            seeds, new = rm_seeds, s_spec.rm_tag(f, rm_seeds, fj)
        for st in settings:
            key = f"{new}_{st}"
            out = ROOT / "cache" / f"{st}_{new}_s0.parquet"
            try:
                frame, rec = avg_s.compute(ROOT, a["mode"], a["donor"], sfile, seeds, st)
            except SystemExit as e:
                res[key] = {"pass": False, "why": str(e)}
                ok = False
                continue
            bad = avg_s.frames_equal(frame, pd.read_parquet(out)) if out.exists() else [f"missing {out.name}"]
            if st != "full":
                en = ROOT / "preds" / f"es_valid_ids_{new}_{st}_s0.parquet"
                ed = ROOT / "preds" / f"es_valid_ids_{a['donor']}_{st}_s0.parquet"
                if not en.exists() or not np.array_equal(pd.read_parquet(en)["MVT_ID_mvt"].to_numpy(), pd.read_parquet(ed)["MVT_ID_mvt"].to_numpy()):
                    bad.append("es file is not the donor's")
            res[key] = {"pass": not bad, "differences": bad, "seeds": seeds, "mode": a["mode"], "donor": a["donor"],
                        "out_sha16": sha16(out) if out.exists() else None, "seed_files": rec["seed_files"]}
            ok &= not bad
    return {"check": "avg", "pass": ok, "legs": res}


def check_blend(f: str, s_tag: str, f_tag: str, settings: list[str], rm_seeds: list[int] | None = None) -> dict:
    """blend_fixed's notes: the same reference, weights and slots; leg tags differ only as the spec maps them. The RealMLP slot must hold
    exactly the learner the finished seeds give (coordinator 03:5x fix A): <tag>x<n>s for three or more own seeds, the family's RealMLP for
    fewer (--rm_seeds none); without --rm_seeds, the planned four (x4s)."""
    F = s_spec.fam(f)
    L = F["legs"]
    tmap = {F["fam_C"]["base"]: L["lgbap"]["avg"]["new"]}
    for x in F["fam_C"]["learners"].split(","):
        t = x.split(":")[0]
        if t.startswith(L["cbap"]["tag"]):
            tmap[t] = L["cbap"]["avg"]["new"]
        elif t.startswith(L["rm"]["tag"]):
            tmap[t] = L["rm"]["avg"]["new"]
    rs = L["rm"]["seeds"] if rm_seeds is None else rm_seeds
    want_rm = s_spec.rm_tag(f, rs) if len(rs) >= 3 else s_spec.fam_rm(f)
    res, ok = {}, True
    for st in settings:
        a = json.loads((ROOT / "notes" / f"blend_fixed_{s_tag}_{st}_s0.json").read_text(encoding="utf-8"))
        b = json.loads((ROOT / "notes" / f"blend_fixed_{f_tag}_{st}_s0.json").read_text(encoding="utf-8"))
        bad = []
        for k in sorted(set(a) | set(b)):
            if k in ("tag", "rmse", "check_max_abs_diff"):
                continue
            if k == "leg_tags":
                mapped = [want_rm if tmap.get(t, t) == L["rm"]["avg"]["new"] else tmap.get(t, t) for t in b[k]]
                if a[k] != mapped:
                    bad.append(f"leg_tags {a[k]} vs mapped {mapped}")
            elif k == "base":
                if a.get(k) != tmap.get(b.get(k), b.get(k)):
                    bad.append(f"base {a.get(k)} vs mapped {tmap.get(b.get(k), b.get(k))}")
            elif a.get(k) != b.get(k):
                bad.append(f"{k} differs")
        res[st] = {"pass": not bad, "differences": bad, "leg_tags": a.get("leg_tags"), "family_leg_tags": b.get("leg_tags"), "rm_learner": want_rm}
        ok &= not bad
    return {"check": "blend", "pass": ok, "settings": res}


def check_manifest(s_man: str, f_man: str, tagmap: dict) -> dict:
    a = json.loads((ROOT / s_man).read_text(encoding="utf-8"))
    b = json.loads((ROOT / f_man).read_text(encoding="utf-8"))
    bad = []

    def m(x):
        if isinstance(x, str):
            for s, t in tagmap.items():
                x = x.replace(s, t)
            return x
        if isinstance(x, list):
            return [m(y) for y in x]
        if isinstance(x, dict):
            return {k: m(v) for k, v in x.items()}
        return x
    ca, cb = a.get("code_sha", {}), b.get("code_sha", {})
    for k, v in cb.items():
        if ca.get(k) != v:
            bad.append(f"code {k}: {ca.get(k)} vs family {v}")
    extra = set(ca) - set(cb)
    if extra - ALLOWED_EXTRA_CODE:
        bad.append(f"code files beyond the family's and {sorted(ALLOWED_EXTRA_CODE)}: {sorted(extra - ALLOWED_EXTRA_CODE)}")
    for k in ("commands", "config"):
        if m(a.get(k)) != b.get(k):
            bad.append(f"{k} differ after the tag map")
    return {"check": "manifest", "pass": not bad, "differences": bad, "s": s_man, "family": f_man, "extra_code": sorted(extra),
            "tagmap": tagmap}


def check_feats(ra: str, rb: str, name: str) -> dict:
    """The labeled 2025 rows (src != 0) of cache/<name>.parquet in two roots: same ids, same order, every column bit-equal."""
    pa_, pb_ = Path(ra) / "cache" / f"{name}.parquet", Path(rb) / "cache" / f"{name}.parquet"
    a, b = pd.read_parquet(pa_), pd.read_parquet(pb_)
    a, b = a[a["src"] != 0].reset_index(drop=True), b[b["src"] != 0].reset_index(drop=True)
    bad = avg_s.frames_equal(a, b)
    return {"check": "feats", "pass": not bad, "differences": bad[:20], "rows": [int(len(a)), int(len(b))], "a": str(pa_), "b": str(pb_)}


def main() -> int:
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    rs = rm_seeds_arg(a)
    global LEGS
    LEGS = a[a.index("--legs") + 1].split(",") if "--legs" in a else None
    if a[0] == "cmds":
        r = check_cmds(a[1], a[2].split(","), a[3], rs)
        name = f"cmds_{a[1]}_{a[2].replace(',', '')}"
    elif a[0] == "rows":
        r = check_rows(a[1], a[2].split(","), rs)
        name = f"rows_{a[1]}_{a[2].replace(',', '')}"
    elif a[0] == "avg":
        r = check_avg(a[1], a[2].split(","), "--fj" in a, rs)
        name = f"avg_{a[1]}_{a[2].replace(',', '')}"
    elif a[0] == "blend":
        r = check_blend(a[1], a[2], a[3], a[4].split(","), rs)
        name = f"blend_{a[2]}"
    elif a[0] == "manifest":
        r = check_manifest(a[2], a[3], dict(x.split("=") for x in a[4].split(",")))
        name = f"manifest_{Path(a[2]).stem[9:][:60]}"
    elif a[0] == "feats":
        r = check_feats(a[1], a[2], a[3])
        name = f"feats_{a[3]}"
    else:
        raise SystemExit(__doc__)
    r["root"], r["written"] = str(ROOT), time.strftime("%Y-%m-%d %H:%M:%S")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"s_check_{name}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    fails = []
    for k in ("jobs", "files", "legs", "settings"):
        if k in r:
            fails = [n for n, v in r[k].items() if not v.get("pass")]
    print(f"s_check {a[0]}: {'PASS' if r['pass'] else 'FAIL'}" + (f" ({', '.join(fails)})" if fails else "")
          + (f" {r.get('differences')}" if r.get("differences") else "") + f" -> research/f2/out/s_check_{name}.json", flush=True)
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
