# SPDX-License-Identifier: GPL-3.0-only
"""F3: stage a final candidate with R2a applied once, as the last step (adopted 2026-10-11 02:4x KST). PC only.

    .venv/Scripts/python.exe -B -X utf8 research/f3/pack_r2a.py <base queue name or key, e.g. 598c> [--folder research/f3/queue_ready_r2a]
                                                                [--tool-out research/r21/f3_pack] [--expect-sha256 <hex>]
    (cwd = the live prc-2026 root; nothing goes into queue/, subs/, notes/ or preds/)

1 base: the staged final <key>_<spec> in research/f3/queue_ready_final, queue_ready_l4 or queue_ready_s (exactly one). Its parquet, manifest and
  queue json must match the base folder's SHA256SUMS.txt and the manifest's test sha16, and candidate() must accept it where it is.
2 R2a: research/r21/r2a_tool.py (sha256 pinned below) `apply <base parquet> <tool-out>/<name>.parquet`, the tool's own output folder under
  research/r21/. The output must keep the base's IDs and order, change exactly the rows the tool reports (all of them R2a rows) and leave every
  other row bit-identical.
3 stage "<key>r_<spec>r2a" into the folder (refuses to overwrite; never research/f3/queue_ready_final or another staging folder):
    <name>.parquet               the tool's output, byte copy
    manifest_<spec>r2a.json      the base's manifest with test = that parquet; holdout, forward, es and feats are the base's (R2a changes
                                 final rows only); base_final names the base's manifest, test and queue json with sha256; post_step records the
                                 R2a run with repo-relative paths and without test-row ids (their count, count by month and the sha256 of the
                                 sorted id list); code_sha_outside_preds = the base's plus research/r21/r2a_tool.py and the link table's builder
                                 research/r21/d3_link4m.py, snapshots in <folder>/code/ (code only: the link table's sha256 is in post_step).
                                 It ships unchanged as preds/manifest_<spec>r2a.json; the keys copied from the base keep the base's values.
    <name>.json                  queue entry (kind CANDIDATE, local_rmse re-measured on the live ruler, manifest)
    r2a_apply_<name>.json        the tool's apply record (the changed ids and old and new values; not shipped)
    check_<name>.json            the checks below; ships unchanged at its own path (repo-relative paths, no test-row ids); its top-level
                                 "packer" names this script and the sha256 of the copy that ran
    SHA256SUMS.txt               this name's files plus the base's three files (paths relative to the folder; other names' lines are kept)
4 checks: candidate(name, folder); fire-final --dry replayed in a child process with fire_queue.QUEUE pointed at the folder (src/fire_queue.py
  fire_final itself, dry=True; empty_on_final replaced by a read-only twin that reads the notes/ cache or scans without writing; every writer
  of the dry path's later steps replaced by a refusal); the int file built by fire_queue.build_final in two separate child processes (its
  ROOT pointed at a temporary folder outside the repo), checked with final_file_problems and submission_values, compared byte for byte, and
  compared with the base's int file built the same way (and with the base's Mac int file sha256 in the base SHA256SUMS.txt when listed).
5 shippable: no final test-row id in the manifest or the check json (any number or 9-digit token that is a final_submitting id), no absolute
  or home path in the fields written here, none of the release checks' private words (docs_release/_work/final/checks_final.py: its leftover
  words and prose labels, after its planted lines pass) in those fields, and docs_release/_work/secret_scan.py on a temporary tree that holds
  the manifest at preds/manifest_<spec>r2a.json and the check json at its own path: no hard hit, no path or notes hit outside preds/.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TOOL = ROOT / "research" / "r21" / "r2a_tool.py"
TOOL_SHA256 = "c5186b192ff55f01cb8853883cfb8785033dbca76883f5eadaa53dadb1a503e3"  # adopted 2026-10-11 02:4x KST
LINK = ROOT / "research" / "r21" / "d3_link4m.parquet"
LINK_SHA16 = "38c129a4bb1c18dc"
BUILDER = ROOT / "research" / "r21" / "d3_link4m.py"  # writes research/r21/d3_link4m.parquet (relative to its cwd)
BUILDER_SHA16 = "3bb4f4bf75774e5d"
BASE_DIRS = ["research/f3/queue_ready_final", "research/f3/queue_ready_l4", "research/f3/queue_ready_s"]
NO_WRITE = ["research/f3/queue_ready_final", "research/f3/queue_ready_l4", "research/f3/queue_ready_s", "research/f3/queue_ready",
            "queue", "subs", "notes", "preds", "cache", "src", "data"]
NOWIN = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW for the child processes
RELEASE_CHECKS = ROOT / "docs_release" / "_work" / "final" / "checks_final.py"
RELEASE_SCAN = ROOT / "docs_release" / "_work" / "secret_scan.py"
NEW_KEYS = ("spec", "run_tag", "test", "base_final", "post_step", "written_kst")  # the manifest keys written here (the rest are the base's)
ABS_RX = re.compile(r"\b[A-Za-z]:[\\/]|(?<![\w.])/(?:Users|home)/|~/")
ID_RX = re.compile(r"(?<![0-9A-Za-z])(\d{9})(?:\.0+)?(?![0-9A-Za-z])")
HARD_RULES = ("CRED", "KEY", "ASSIGN", "EMAIL", "PHONE", "FILE", "DISCORD")  # the scanner's other two rules (paths, notes words) are soft
IDS_RULE = "sha256 of the sorted MVT_ID_mvt values written as integers, one per line, joined by \\n without a trailing newline, UTF-8"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: Path) -> str:
    return Path(p).resolve().relative_to(ROOT).as_posix()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST")


def child(args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "-B", "-X", "utf8", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, env=env, creationflags=NOWIN)


def sums(folder: Path) -> dict[str, str]:
    p = folder / "SHA256SUMS.txt"
    out = {}
    for ln in (p.read_text(encoding="utf-8").splitlines() if p.exists() else []):
        if "  " in ln:
            h, f = ln.split("  ", 1)
            out[f] = h
    return out


def ids_digest(ids) -> str:
    return hashlib.sha256("\n".join(str(int(x)) for x in sorted(ids)).encode("utf-8")).hexdigest()


def by_month(ids, month: dict) -> dict[str, int]:
    return {str(m): n for m, n in sorted(Counter(month[float(x)] for x in ids).items())}


def strip_root(s: str) -> str:
    for pre in (str(ROOT) + "\\", str(ROOT) + "/", ROOT.as_posix() + "/"):
        s = s.replace(pre, "")
    return s


def strings(o, path: str = ""):
    """(dotted path, text) for every key and string value of a JSON object."""
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{path}.{k}" if path else str(k)
            yield p + " (key)", str(k)
            yield from strings(v, p)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from strings(v, f"{path}[{i}]")
    elif isinstance(o, str):
        yield path, o


def numbers(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from numbers(v)
    elif isinstance(o, list):
        for v in o:
            yield from numbers(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield o


def test_ids_in(o, ids: set[int]) -> int:
    n = sum(1 for x in numbers(o) if float(x).is_integer() and int(x) in ids)
    return n + sum(1 for _, s in strings(o) for t in ID_RX.findall(s) if int(t) in ids)


def hits(o, rxs) -> list[str]:
    return sorted({p for p, s in strings(o) if any(rx.search(s) for rx in rxs)})


def release_patterns() -> list:
    """The release checks' own patterns (leftover words, prose labels), loaded read-only after their planted lines pass."""
    import runpy
    argv = sys.argv
    try:
        sys.argv = [str(RELEASE_CHECKS), str(ROOT)]
        g = runpy.run_path(str(RELEASE_CHECKS), run_name="release_checks")
    finally:
        sys.argv = argv
    g["planted"]()
    return [g["LEFT_RX"], g["LABEL_RX"]]


def release_scan(files: dict[str, Path]) -> dict:
    """docs_release/_work/secret_scan.py on a temporary tree that holds only these files at their shipped paths; hard hits anywhere and
    path or notes hits outside preds/ fail (the release checks allow the latter in preds/ only)."""
    t = Path(tempfile.mkdtemp(prefix="pack_r2a_scan_"))
    try:
        for relp, src in files.items():
            d = t / "stage" / relp
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, d)
        r = child(["-I", str(RELEASE_SCAN), str(t / "stage"), str(t / "out.json")], timeout=900)
        out = json.loads((t / "out.json").read_text(encoding="utf-8")) if (t / "out.json").exists() else {}
    finally:
        shutil.rmtree(t, ignore_errors=True)
    raw = out.get("hits", [])
    hard = [x[:3] for x in raw if x[0] in HARD_RULES]
    soft = [["path or notes rule", *x[1:3]] for x in raw if x[0] not in HARD_RULES]  # recorded without the rule's own name
    out_soft = [x for x in soft if not str(x[1]).startswith("preds/")]
    return {"rc": r.returncode, "planted_ok": out.get("planted_ok"), "files_scanned": out.get("files_scanned"),
            "hits": {"hard": len(hard), "path_or_notes": len(soft)}, "hard": hard, "path_or_notes_outside_preds": out_soft,
            "path_or_notes_in_preds": [x for x in soft if str(x[1]).startswith("preds/")],
            "temp_tree_removed": not t.exists(),
            "clean": r.returncode == 0 and out.get("planted_ok") is True and out.get("files_scanned") == len(files) and not hard and not out_soft}


def find_base(arg: str) -> tuple[Path, str]:
    found = []
    for d in BASE_DIRS:
        q = ROOT / d
        if not q.is_dir():
            continue
        pat = f"{arg}.json" if "_" in arg else f"{arg}_*.json"
        found += [(q, p.stem) for p in q.glob(pat) if not p.name.startswith(("check_", "manifest_")) and (q / f"{p.stem}.parquet").exists()]
    if len(found) != 1:
        raise SystemExit(f"base {arg}: {len(found)} staged candidates found in {BASE_DIRS}: {[f'{rel(q)}/{n}' for q, n in found]}")
    return found[0]


def name_used(key_r: str) -> list[str]:
    used = []
    for d in [ROOT / "queue", ROOT / "queue" / "done"] + sorted(p for p in (ROOT / "research").glob("*/queue_ready*") if p.is_dir()):
        used += [rel(p) for p in d.glob(f"{key_r}_*")]
    for f in (ROOT / "subs" / "ledger.csv", ROOT / "subs" / "fire_events.jsonl"):
        if f.exists() and f"{key_r}_" in f.read_text(encoding="utf-8"):
            used.append(rel(f))
    return used


# ---- child modes (run with cwd = ROOT, -B) ----

def child_dryfire(folder: str, name: str) -> int:
    """src/fire_queue.py fire_final(name, dry=True) with QUEUE = the staged folder. Nothing is written: empty_on_final is replaced by a read-only
    twin and every writer after the dry return (event_append, push, build_final, save_result, record) by a refusal."""
    import src.fire_queue as fq
    from src import shot as S
    q = (ROOT / folder).resolve()
    assert ROOT.resolve() in q.parents and rel(q).split("/")[0] not in ("queue", "subs", "notes", "preds"), q
    fq.QUEUE, fq.DONE = q, q / "_no_such_done_dir"
    assert not fq.DONE.exists()

    def ro_empty_on_final(feats: str, feats_sha: str) -> dict:
        out = ROOT / "notes" / f"test_empty_final_{feats}.json"
        tsha = S.FINAL_PINS["final_submitting.parquet"]["sha256"]
        if out.exists():
            c = json.loads(out.read_text(encoding="utf-8"))
            if c.get("feats_sha") == feats_sha and c.get("final_template_sha256") == tsha:
                print("pack_r2a: empty_on_final from the notes cache (read only)", rel(out), flush=True)
                return c
        f = ROOT / "cache" / f"{feats}.parquet"
        if not f.exists() or S.sha256_file(f)[:16] != feats_sha:
            return {"feats": feats, "feats_sha": feats_sha, "error": f"피처 판 {feats} 의 파일이 없거나 해시가 manifest 와 다르다"}
        from src.batches import hold
        r = S.scan_empty(f, S.final_template_ids(), hold().index.to_numpy())
        print("pack_r2a: empty_on_final scanned fresh (not written)", flush=True)
        return {"feats": feats, "feats_sha": feats_sha, "final_template_sha256": tsha, **r}

    def refuse(*a, **k):
        raise SystemExit("pack_r2a dry replay: a write was attempted")
    S.empty_on_final = ro_empty_on_final
    S.event_append = refuse
    fq.push = fq.build_final = fq.save_result = fq.record = refuse
    return fq.fire_final(name, "pack_r2a dry check (no upload)", "none: dry check of the staged R2a candidate", True, False)


def child_build(parquet: str, outdir: str) -> int:
    """fire_queue.build_final on the parquet with fire_queue.ROOT pointed at outdir (outside the repo), then fire-final's own file checks."""
    import numpy as np
    import pandas as pd
    import src.fire_queue as fq
    from src import shot as S
    o = Path(outdir).resolve()
    assert ROOT.resolve() not in o.parents and o != ROOT.resolve(), o
    (o / "subs").mkdir(parents=True, exist_ok=False)
    fq.ROOT = o
    pred = pd.read_parquet(ROOT / parquet).set_index("MVT_ID_mvt")["pred"]
    sub = fq.build_final(pred)
    assert sub.parent == o / "subs", sub
    tpl = S.final_template_ids()
    vals = S.submission_values(pred, tpl)
    print(json.dumps({"path": str(sub), "sha256": sha256(sub), "final_file_problems": S.final_file_problems(sub, tpl),
                      "values_equal_submission_values": bool(np.array_equal(pd.read_parquet(sub)["TAXITIME_SEC_mvt"].to_numpy(), vals))}))
    return 0


# ---- pack ----

def main() -> int:
    ap = argparse.ArgumentParser(prog="research/f3/pack_r2a.py")
    ap.add_argument("base")
    ap.add_argument("--folder", default="research/f3/queue_ready_r2a")
    ap.add_argument("--tool-out", default="research/r21/f3_pack")
    ap.add_argument("--expect-sha256", default=None, help="sha256 the tool's output must have (e.g. an earlier run of the tool on the same base)")
    a = ap.parse_args()
    assert (ROOT / "src" / "batches.py").exists() and Path.cwd().resolve() == ROOT.resolve(), "run from the live prc-2026 root"
    for f in (RELEASE_CHECKS, RELEASE_SCAN):
        if not f.exists():
            raise SystemExit(f"the release check {rel(f)} is missing: the shippable checks cannot run")
    import numpy as np
    import pandas as pd
    from src import shot as S
    from src.batches import candidate, hold
    from src.ruler import rmse_of

    Q = (ROOT / a.folder).resolve()
    qrel = rel(Q)
    if any(qrel == d or qrel.startswith(d + "/") for d in NO_WRITE) or not qrel.startswith("research/"):
        raise SystemExit(f"refusing folder {qrel}: R2a stages into a folder of its own under research/")
    TO = (ROOT / a.tool_out).resolve()
    if (ROOT / "research" / "r21").resolve() not in TO.parents:
        raise SystemExit("--tool-out must be under research/r21/ (the tool writes only there)")
    me = Path(__file__).resolve()
    rep = {"written": now(), "packer": {"path": rel(me), "sha256": sha256(me)}, "folder": qrel}  # the file that wrote these records

    # 1 base
    BQ, base = find_base(a.base)
    key, bspec = base.split("_", 1)
    name, nspec = f"{key}r_{bspec}r2a", f"{bspec}r2a"
    rep.update(name=name, spec=nspec, base=base, base_dir=rel(BQ))
    used = name_used(f"{key}r")
    if used:
        raise SystemExit(f"{key}r is already used: {used}")
    bp, bj = BQ / f"{base}.parquet", BQ / f"{base}.json"
    bmeta = json.loads(bj.read_text(encoding="utf-8"))
    bmp = ROOT / bmeta["manifest"]
    M = json.loads(bmp.read_text(encoding="utf-8"))
    bsh = {"parquet": sha256(bp), "manifest": sha256(bmp), "json": sha256(bj)}
    S0 = sums(BQ)
    want = {"parquet": S0.get(bp.name), "manifest": S0.get(bmp.name) if bmp.parent == BQ else None, "json": S0.get(bj.name)}
    bad = [k for k in ("parquet", "manifest", "json") if want[k] != bsh[k]]
    if bad or bsh["parquet"][:16] != M["test"]["sha"] or ROOT / M["test"]["path"] != bp:
        raise SystemExit(f"base {base}: SHA256SUMS.txt or manifest test disagree with the files ({bad}, test sha {M['test']['sha']})")
    bc, bwhy = candidate(base, BQ)
    if bwhy:
        raise SystemExit(f"base {base}: candidate() refuses it where it is: {bwhy}")
    mac_int = [h for f, h in S0.items() if f.startswith("spirited-tiger_final.parquet") and (f"for {key})" in f)]
    rep["base_checks"] = {"sha256": bsh, "sha256sums_match": True, "candidate": bc, "base_mac_int_sha256": mac_int[0] if len(mac_int) == 1 else None}

    # 2 R2a
    tsha, lsha, bsha = sha256(TOOL), sha256(LINK), sha256(BUILDER)
    if tsha != TOOL_SHA256 or lsha[:16] != LINK_SHA16 or bsha[:16] != BUILDER_SHA16:
        raise SystemExit(f"R2a tool {tsha[:16]}, link {lsha[:16]} or link builder {bsha[:16]} is not the adopted one "
                         f"({TOOL_SHA256[:16]}, {LINK_SHA16}, {BUILDER_SHA16})")
    tout = TO / f"{name}.parquet"
    if tout.exists() or tout.with_suffix(".parquet.json").exists():
        raise SystemExit(f"{rel(tout)} exists; refusing to overwrite")
    cmd = ["research/r21/r2a_tool.py", "apply", rel(bp), tout.relative_to(ROOT).as_posix()]
    r = child(cmd, timeout=1800)
    rep["tool_run"] = {"command": "python -X utf8 " + " ".join(cmd), "rc": r.returncode, "out": strip_root(r.stdout.strip()[-1500:]),
                       "err": strip_root(r.stderr.strip()[-1500:])}
    if r.returncode != 0 or not tout.exists():
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        raise SystemExit("the R2a tool failed")
    info = json.loads(tout.with_suffix(".parquet.json").read_text(encoding="utf-8"))
    osha = sha256(tout)
    B = pd.read_parquet(bp)
    N_ = pd.read_parquet(tout)
    same_ids = list(N_.columns) == list(B.columns) and np.array_equal(N_["MVT_ID_mvt"].to_numpy(), B["MVT_ID_mvt"].to_numpy())
    ch = N_["pred"].to_numpy() != B["pred"].to_numpy()
    ch_ids = sorted(float(x) for x in N_.loc[ch, "MVT_ID_mvt"])
    month = {float(x["MVT_ID_mvt"]): int(x["month"]) for x in info["rows"]}
    rest_bits = bool(np.array_equal(N_["pred"].to_numpy()[~ch].view(np.int64), B["pred"].to_numpy()[~ch].view(np.int64)))
    all_r2a = set(ch_ids) <= set(month)
    rep["r2a_output"] = {"path": rel(tout), "sha256": osha, "expect_sha256": a.expect_sha256,
                         "equals_expect": None if not a.expect_sha256 else osha == a.expect_sha256,
                         "tool_record_input_sha256_is_base": info["input_sha256"] == bsh["parquet"], "tool_record_output_sha256_ok": info["output_sha256"] == osha,
                         "ids_and_order_same_as_base": bool(same_ids), "dtypes": {c: str(t) for c, t in N_.dtypes.items()},
                         "changed_n": int(ch.sum()), "tool_changed_rows": info["changed_rows"], "changed_ids_all_r2a_rows": all_r2a,
                         "changed_by_month": by_month(ch_ids, month) if all_r2a else None, "changed_ids_sha256": ids_digest(ch_ids),
                         "changed_ids_sha256_of": IDS_RULE, "other_rows_bit_identical": rest_bits, "move_mse_4month": info["move_mse_4month"]}
    ro = rep["r2a_output"]
    if not (ro["tool_record_input_sha256_is_base"] and ro["tool_record_output_sha256_ok"] and same_ids and ro["changed_n"] == info["changed_rows"]
            and all_r2a and rest_bits and ro["equals_expect"] is not False):
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        raise SystemExit("the R2a output failed its checks")

    # 3 stage
    Q.mkdir(parents=True, exist_ok=True)
    qp, mp, jp = Q / f"{name}.parquet", Q / f"manifest_{nspec}.json", Q / f"{name}.json"
    ap_, cp = Q / f"r2a_apply_{name}.json", Q / f"check_{name}.json"
    for p in (qp, mp, jp, ap_, cp):
        if p.exists():
            raise SystemExit(f"{rel(p)} exists; refusing to overwrite")
    shutil.copyfile(tout, qp)
    assert sha256(qp) == osha
    shutil.copyfile(tout.with_suffix(".parquet.json"), ap_)
    code = {}
    for src, h, full in ((TOOL, tsha[:16], tsha), (BUILDER, bsha[:16], bsha)):
        dst = Q / "code" / f"{h}_{src.name}"
        dst.parent.mkdir(exist_ok=True)
        if dst.exists():
            assert sha256(dst) == full, f"{rel(dst)} exists with other bytes"
        else:
            shutil.copyfile(src, dst)
        assert sha256(dst) == full
        code[rel(src)] = {"sha": h, "snapshot": rel(dst)}
    for f, v in (M.get("code_sha_outside_preds") or {}).items():  # the base's kept copies must still be there, unchanged
        assert sha256(ROOT / v["snapshot"])[:16] == v["sha"][:16], f"base snapshot {v['snapshot']} differs from {v['sha']}"
    N = copy.deepcopy(M)
    N["spec"] = N["run_tag"] = nspec
    n_ch = ro["changed_n"]
    N["test"] = {"path": rel(qp), "sha": osha[:16], "rows": int(len(N_)),
                 "what": (f"{key}'s float predictions with R2a applied once as the last step (research/r21/r2a_tool.py apply: LIRF NM-unmatched "
                          f"departures whose linked arrival carries the NM LOBT with 0 < d_lobt <= 3606 s and |SCHED - LOBT_arr| > 3606 s get "
                          f"PI * d_sch + (1 - PI) * pred, PI = 21.5/24); {n_ch} rows changed, every other row bit-identical to {key}; four months, "
                          f"final_submitting order")}
    N["code_sha_outside_preds"] = {**(M.get("code_sha_outside_preds") or {}), **code}
    fo = M.get("final_of") or {}  # the base's values; its Mac int file keys renamed so that they read as the base's
    N["final_of"] = {**{k: v for k, v in fo.items() if not k.startswith("mac_final_file")},
                     **({"base_mac_final_file": fo.get("mac_final_file"), "base_mac_final_file_sha256": fo.get("mac_final_file_sha256")} if fo else {})}
    N["base_final"] = {"queue_name": base, "staged_dir": rel(BQ), "manifest": rel(bmp), "manifest_sha256": bsh["manifest"], "test": rel(bp),
                       "test_sha256": bsh["parquet"], "queue_json": rel(bj), "queue_json_sha256": bsh["json"], "local_rmse": bmeta["local_rmse"],
                       "sha256sums": rel(BQ / "SHA256SUMS.txt")}
    N["post_step"] = {"name": "R2a", "adopted": "2026-10-11 02:4x KST: applied once, as the last step, to the chosen final",
                      "tool": rel(TOOL), "tool_sha256": tsha, "link": rel(LINK), "link_sha256": lsha,
                      "link_builder": rel(BUILDER), "link_builder_sha256": bsha, "pi": info["pi"], "window_s": info["window_s"],
                      "command": rep["tool_run"]["command"], "input": rel(bp), "input_sha256": bsh["parquet"], "output": rel(tout), "output_sha256": osha,
                      "changed_n": n_ch, "changed_by_month": ro["changed_by_month"], "changed_ids_sha256": ro["changed_ids_sha256"],
                      "changed_ids_sha256_of": IDS_RULE, "move_mse_4month": info["move_mse_4month"],
                      "applies_to": ("the final (test) rows only; holdout, forward and es_valid_ids are the base's bytes (R2a is not applied to them); "
                                     "its effect on the 2025 rulers: docs/DESIGN.md section 8.8"),
                      "int_file": ("the int file is the one fire-final builds from test.path (src/fire_queue.py build_final); final_of names the base's "
                                   "Mac int file")}
    N["written_kst"] = now()
    mp.write_bytes(json.dumps(N, ensure_ascii=False, indent=1).encode("utf-8"))
    rm = rmse_of(N["holdout"]["path"], hold())
    meta = {"kind": "CANDIDATE",
            "method": (f"F3 결선 R2a 판 {key}r: 결선 후보 {key}({rel(bp)})의 넉 달 실수 예측에 R2a(research/r21/r2a_tool.py apply, "
                       f"sha256 앞 16자 {tsha[:16]}, 링크 표 research/r21/d3_link4m.parquet {LINK_SHA16}, PI 21.5/24, 창 3606초)를 마지막 "
                       f"단계로 한 번 건 것. 바뀐 행 {n_ch}개(넉 달 이동 MSE {info['move_mse_4month']:.1f}), 나머지 행은 {key} 와 비트까지 같다. "
                       f"val, fwd, es 와 local_rmse 는 {key} 의 것이다(R2a 는 결선 행만 바꾼다). 기준 manifest {rel(bmp)}(sha256 앞 16자 "
                       f"{bsh['manifest'][:16]}). queue/ 에는 넣지 않는다(결선 전환 때 고른다, 10-12)."),
            "local_rmse": round(float(rm), 3), "manifest": rel(mp),
            "note": (f"spec {nspec}, feats {M['feats']} ({M['feats_sha']}), base {base} (test sha256 {bsh['parquet']}), R2a tool sha256 {tsha}, "
                     f"link sha16 {LINK_SHA16}, test sha16 {osha[:16]}, changed rows {n_ch}")}
    jp.write_bytes(json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))

    # 4 checks
    chk = {}
    cand, why = candidate(name, Q)
    chk["candidate"], chk["candidate_problem"] = cand, why
    r = child([rel(Path(__file__)), "_dryfire", qrel, name], timeout=3600)
    lines = [strip_root(x) for x in (r.stdout or "").splitlines()]
    chk["fire_final_dry"] = {"rc": r.returncode, "would_fire": any("(dry) would fire" in x for x in lines),
                             "blocks": [x for x in lines if " BLOCK " in x], "warns": [x for x in lines if " WARN " in x],
                             "out": lines[-14:], "err": strip_root((r.stderr or "").strip()[-1200:])}
    tmp = Path(tempfile.mkdtemp(prefix="pack_r2a_build_"))
    try:
        builds = {}
        for tag, pq_ in (("r2a_1", qp), ("r2a_2", qp), ("base", bp)):
            r = child([rel(Path(__file__)), "_build", rel(pq_), str(tmp / tag)], timeout=1800)
            if r.returncode != 0:
                builds[tag] = {"rc": r.returncode, "err": strip_root((r.stderr or "").strip()[-1200:])}
                continue
            builds[tag] = json.loads(r.stdout.strip().splitlines()[-1])
        ok3 = all("sha256" in builds[t] for t in builds)
        I = {"builds": {t: {k: v for k, v in b.items() if k != "path"} for t, b in builds.items()},
             "reproducible": ok3 and builds["r2a_1"]["sha256"] == builds["r2a_2"]["sha256"]}
        if ok3:
            i1 = pd.read_parquet(builds["r2a_1"]["path"])
            ib = pd.read_parquet(builds["base"]["path"])
            d = i1["TAXITIME_SEC_mvt"].to_numpy() != ib["TAXITIME_SEC_mvt"].to_numpy()
            d_ids = sorted(float(x) for x in i1.loc[d, "MVT_ID_mvt"])
            in_r2a = set(d_ids) <= set(ch_ids)
            eq_mac = None if not rep["base_checks"]["base_mac_int_sha256"] else builds["base"]["sha256"] == rep["base_checks"]["base_mac_int_sha256"]
            I.update(sha256=builds["r2a_1"]["sha256"], base_build_sha256=builds["base"]["sha256"], base_build_equals_base_mac_int=eq_mac,
                     same_ids_as_base_int=bool(np.array_equal(i1["MVT_ID_mvt"].to_numpy(), ib["MVT_ID_mvt"].to_numpy())),
                     float_rows_changed=n_ch, float_rows_changed_by_month=ro["changed_by_month"],
                     int_values_differing_from_base_int=len(d_ids), int_values_differing_by_month=by_month(d_ids, month) if in_r2a else None,
                     int_values_differing_from_base_mac_int=len(d_ids) if eq_mac else None,
                     int_abs_change_sum_s=int(np.abs(i1["TAXITIME_SEC_mvt"].to_numpy(np.int64) - ib["TAXITIME_SEC_mvt"].to_numpy(np.int64)).sum()),
                     differing_ids_are_r2a_changed_rows=in_r2a)
        chk["int_file"] = I
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    chk["build_tmp_removed"] = not tmp.exists()
    I = chk["int_file"]
    chk["pass"] = bool(cand and not why and chk["fire_final_dry"]["rc"] == 0 and chk["fire_final_dry"]["would_fire"] and not chk["fire_final_dry"]["blocks"]
                       and I.get("reproducible") and all(not b.get("final_file_problems") and b.get("values_equal_submission_values") for b in I["builds"].values())
                       and I.get("base_build_equals_base_mac_int") is not False and I.get("same_ids_as_base_int") and I.get("differing_ids_are_r2a_changed_rows"))
    rep["checks"] = chk
    rep["staged"] = {"queue_name": name, "staged_dir": qrel, "parquet_sha256": sha256(qp), "json_sha256": sha256(jp), "manifest_sha256": sha256(mp)}

    # 5 shippable (the manifest at preds/manifest_<spec>r2a.json, the check json at its own path)
    ids = {int(x) for x in S.final_template_ids()}
    rx = release_patterns()
    new = {k: N[k] for k in NEW_KEYS} | {"code_sha_outside_preds (added here)": code}
    dtmp = Path(tempfile.mkdtemp(prefix="pack_r2a_draft_"))  # the check json before its shippable record, outside the repo
    try:
        draft = dtmp / f"check_{name}.json"
        draft.write_bytes(json.dumps(rep, ensure_ascii=False, indent=1, default=str).encode("utf-8"))
        scan = release_scan({f"preds/manifest_{nspec}.json": mp, f"{qrel}/check_{name}.json": draft})
    finally:
        shutil.rmtree(dtmp, ignore_errors=True)
    sh = {"manifest_test_row_ids": test_ids_in(N, ids), "manifest_new_fields_abs_paths": hits(new, [ABS_RX]),
          "manifest_new_fields_private_words": hits(new, rx),
          "manifest_base_keys_with_paths_or_words": hits({k: v for k, v in N.items() if k not in NEW_KEYS and k != "code_sha_outside_preds"}, [ABS_RX] + rx),
          "check_json_test_row_ids": test_ids_in(json.loads(json.dumps(rep, default=str)), ids),
          "check_json_abs_paths": hits(json.loads(json.dumps(rep, default=str)), [ABS_RX]),
          "check_json_private_words": hits(json.loads(json.dumps(rep, default=str)), rx), "release_scan": scan}
    sh["clean"] = bool(not sh["manifest_test_row_ids"] and not sh["manifest_new_fields_abs_paths"] and not sh["manifest_new_fields_private_words"]
                       and not sh["check_json_test_row_ids"] and not sh["check_json_abs_paths"] and not sh["check_json_private_words"] and scan["clean"])
    chk["shippable"] = sh
    chk["pass"] = bool(chk["pass"] and sh["clean"])
    fin = json.loads(json.dumps(rep, default=str))
    left = (test_ids_in(fin, ids), hits(fin, [ABS_RX]), hits(fin, rx))
    assert not any(left), f"the check json with its shippable record is not shippable: {left}"
    cp.write_bytes(json.dumps(rep, ensure_ascii=False, indent=1, default=str).encode("utf-8"))
    mine = [qp, mp, jp, ap_, cp] + [ROOT / v["snapshot"] for v in code.values()]
    lines_ = [f"{sha256(p)}  {Path(os.path.relpath(p, Q)).as_posix()}\n" for p in mine]
    lines_ += [f"{bsh[k]}  {Path(os.path.relpath(p, Q)).as_posix()}\n" for k, p in (("parquet", bp), ("manifest", bmp), ("json", bj))]
    sp = Q / "SHA256SUMS.txt"
    names = {x.split("  ", 1)[1].rstrip("\n") for x in lines_}
    keep = [x for x in (sp.read_text(encoding="utf-8").splitlines(keepends=True) if sp.exists() else [])
            if x.split("  ", 1)[-1].rstrip("\n") not in names]
    sp.write_bytes("".join(keep + lines_).encode("utf-8"))  # LF only, so that sha256sum -c reads it
    print(json.dumps({k: rep[k] for k in ("name", "folder", "base", "staged")}, ensure_ascii=False, indent=1))
    print(json.dumps(rep["r2a_output"], ensure_ascii=False))
    print(json.dumps({"candidate_problem": why, "local_rmse": cand.get("local_rmse"), **{k: v for k, v in chk["fire_final_dry"].items() if k != "out"},
                      "int_file": {k: v for k, v in I.items() if k != "builds"}, "shippable": sh, "pass": chk["pass"]}, ensure_ascii=False, indent=1))
    for x in chk["fire_final_dry"]["out"]:
        print("  dry|", x[:300])
    return 0 if chk["pass"] else 1


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_dryfire":
        raise SystemExit(child_dryfire(sys.argv[2], sys.argv[3]))
    if len(sys.argv) >= 2 and sys.argv[1] == "_build":
        raise SystemExit(child_build(sys.argv[2], sys.argv[3]))
    raise SystemExit(main())
