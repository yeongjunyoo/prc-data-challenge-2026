# SPDX-License-Identifier: GPL-3.0-only
"""F3 L4 final (the G3 of 600 and 602c): research/f3/cmp_manifests_l3.py (9f2a690aa5e3cfd4, unchanged) for L4's names. The reference
manifests are R17-A's gate-only chains of phase f in the clone r17a_l4g_sbx (fetched read-only into <ref preds dir>): every tag carries the
marker "l4gf" after "m6r28" (m6r28l4gfx...), and dropping it gives this root's name (R17-A, 10-10 10:3x). Their val and fwd are R17-A's real L4
files, the same bytes the final root cloned (research/f3/collect_l4.py); their full and test are placeholders (links to 514's learners), so
test files are not compared. For every final manifest matching <glob> with a reference of the same name once the marker is added: holdout,
forward and es_valid_ids equal by sha16; seeds and column equal; config and commands equal once every 16-hex input hash is masked, each
command is cut to its script and arguments, and "m6r28l4gfx" is read as "m6r28x" in the reference. Code shas are listed where they differ.
With --variant V (602c: BbF), the final manifests are the merged variant's (tag x50ws75c3r<V>): each is compared with the reference manifest of
the same stage without the tag (the merge copies the default's val and fwd byte for byte), and the merged corrector stage itself on its files,
seeds and column. Report <out json>.
    python -X utf8 research/f3/cmp_manifests_l4.py <final preds dir> <ref preds dir> <out json> [<glob>, default manifest_m6r28x*ws75*.json] [--variant V]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HEX = re.compile(r"\b[0-9a-f]{16}\b")
MARK_FINAL, MARK_REF = "m6r28x", "m6r28l4gfx"


def norm_cmd(c: str) -> str:
    t = c.split()
    i = next((k for k, x in enumerate(t) if x.endswith(".py") or x.endswith(".sh") or x == "-m"), 0)
    return HEX.sub("<h>", " ".join(t[i:]))


def norm(x):
    if isinstance(x, dict):
        return {k: norm(v) for k, v in x.items()}
    if isinstance(x, list):
        return [norm(v) for v in x]
    if isinstance(x, str):
        return norm_cmd(x) if (".py" in x or ".sh" in x) else HEX.sub("<h>", x)
    return x


def main():
    args = [x for x in sys.argv[1:]]
    V = None
    if "--variant" in args:
        i = args.index("--variant")
        V = args[i + 1]
        del args[i:i + 2]
    fin, ref, outp = Path(args[0]), Path(args[1]), Path(args[2])
    pat = args[3] if len(args) > 3 else "manifest_m6r28x*ws75*.json"
    tag = f"x50ws75c3r{V}" if V else None
    rep = {"glob": pat, "variant": V, "marker": [MARK_FINAL, MARK_REF], "compared": 0, "only_final": [], "fails": {}, "same": [],
           "code_sha_differences": {}}
    for p in sorted(fin.glob(pat)):
        if tag and tag not in p.name:
            continue
        base = p.name.replace(tag, "x50ws75c3") if tag else p.name
        q = ref / base.replace(MARK_FINAL, MARK_REF, 1)
        if not q.exists():
            rep["only_final"].append(p.name)
            continue
        a, b = json.loads(p.read_text(encoding="utf-8")), json.loads(q.read_text(encoding="utf-8"))
        rep["compared"] += 1
        d = {}
        for k in ("holdout", "forward", "es_valid_ids"):
            if a.get(k, {}).get("sha") != b.get(k, {}).get("sha"):
                d[k] = [a.get(k, {}).get("sha"), b.get(k, {}).get("sha")]
        for k in ("seeds", "column"):
            if a.get(k) != b.get(k):
                d[k] = [a.get(k), b.get(k)]
        merged_stage = bool(tag) and p.stem.endswith(tag)
        for k in (() if merged_stage else ("config", "commands")):
            fa = norm(a.get(k))
            fb = json.loads(json.dumps(norm(b.get(k))).replace(MARK_REF, MARK_FINAL))
            if tag:
                fa = json.loads(json.dumps(fa).replace(tag, "x50ws75c3"))
            if fa != fb:
                d[k] = {"final": fa, "ref": fb}
        ca, cb = a.get("code_sha", {}), b.get("code_sha", {})
        cd = {f: [ca.get(f), cb.get(f)] for f in sorted(set(ca) | set(cb)) if ca.get(f) != cb.get(f)}
        if cd:
            rep["code_sha_differences"][p.stem[9:]] = cd
        if d:
            rep["fails"][p.stem[9:]] = d
        else:
            rep["same"].append(p.stem[9:])
    allc = sorted({f for v in rep["code_sha_differences"].values() for f in v})
    rep["code_files_differing_anywhere"] = allc
    rep["pass"] = rep["compared"] > 0 and not rep["fails"]
    rep["summary"] = (f"{rep['compared']} specs in both: {len(rep['same'])} with holdout, forward, es, seeds, column equal and config and commands "
                      f"equal up to input hashes and the marker; {len(rep['fails'])} failing; {len(rep['only_final'])} only in the final root; code "
                      f"files differing somewhere: {allc}")
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(rep["summary"])
    for k, v in list(rep["fails"].items())[:12]:
        print(" FAIL", k, json.dumps(v, ensure_ascii=False)[:600])
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
