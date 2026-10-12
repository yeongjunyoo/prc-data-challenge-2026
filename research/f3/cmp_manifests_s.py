# SPDX-License-Identifier: GPL-3.0-only
"""F3 Package S (the G3 of 603): the final root's chain C stage manifests (the T form, <T_S>adsbgeondctxv3x50ws75c3 and its tails) and 603's
chainavg3 manifest, fetched into <final preds dir>, against F2's S sandbox (fetched read-only into <sandbox preds dir>): its gate-name chain
(T_G = T_S with "sg": research/f2/s_spec.py; full there is a placeholder, val and fwd equal the final names' bit for bit) and its three-chain end
(the L4 sandbox's A and B carry R17-A's "l4gf" marker). research/f3/cmp_manifests_l3.py's rules (9f2a690aa5e3cfd4, not imported): holdout,
forward and es_valid_ids equal by sha16; seeds and column equal; config and commands equal once every 16-hex input hash is masked, each command
is cut to its script and arguments, and the sandbox's names are read as the final root's (T_G -> T_S, m6r28l4gfx -> m6r28x). Test files are not
compared (the sandbox's full is a placeholder). Code shas are listed where they differ.
    python -X utf8 research/f3/cmp_manifests_s.py <598c|l4> <final preds dir> <sandbox preds dir> <out json>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HEX = re.compile(r"\b[0-9a-f]{16}\b")
NAMES = {"598c": dict(T_S="m6r28sapcbrmfw4sv114", T_G="m6r28sgapcbrmfw4sv114", A="m6r28wincbrmfw4sv434tmc", A_sbx="m6r28wincbrmfw4sv434tmc"),
         "l4": dict(T_S="m6r28xsapcbrmfw4sv114", T_G="m6r28xsgapcbrmfw4sv114", A="m6r28xwincbrmfw4sv434tmc", A_sbx="m6r28l4gfxwincbrmfw4sv434tmc")}
MID, GJA = "adsbgeondctxv3x50ws75c3", "amfhgkcxncrdbawgpglgja"
TAILS = ["", "am", "amfh", "amfhgkc", "amfhgkcx", "amfhgkcxncr", "amfhgkcxncrdb", "amfhgkcxncrdbaw", "amfhgkcxncrdbawgp", "amfhgkcxncrdbawgpgl", GJA]


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
    fam, fin, sbx, outp = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    N = NAMES[fam]
    maps = [(N["T_G"], N["T_S"]), ("m6r28l4gfx", "m6r28x")]
    to_final = lambda s: [s := s.replace(a, b) for a, b in maps][-1]  # noqa: E731
    pairs = [(f"manifest_{N['T_S']}{MID}{t}.json", f"manifest_{N['T_G']}{MID}{t}.json") for t in TAILS]
    pairs.append((f"manifest_{N['A']}{MID}{GJA}avskaps.json", f"manifest_{N['A_sbx']}{MID}{GJA}avskaps.json"))
    rep = {"family": fam, "maps_sandbox_to_final": maps, "compared": 0, "missing": [], "fails": {}, "same": [], "code_sha_differences": {}}
    for fn, sn in pairs:
        p, q = fin / fn, sbx / sn
        if not p.exists() or not q.exists():
            rep["missing"].append({"final": fn if not p.exists() else None, "sandbox": sn if not q.exists() else None})
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
        for k in ("config", "commands"):
            fa = norm(a.get(k))
            fb = json.loads(to_final(json.dumps(norm(b.get(k)))))
            if fa != fb:
                d[k] = {"final": fa, "sandbox_mapped": fb}
        ca, cb = a.get("code_sha", {}), b.get("code_sha", {})
        cd = {f: [ca.get(f), cb.get(f)] for f in sorted(set(ca) | set(cb)) if ca.get(f) != cb.get(f)}
        if cd:
            rep["code_sha_differences"][fn[9:-5]] = cd
        if d:
            rep["fails"][fn[9:-5]] = d
        else:
            rep["same"].append(fn[9:-5])
    rep["code_files_differing_anywhere"] = sorted({f for v in rep["code_sha_differences"].values() for f in v})
    rep["pass"] = rep["compared"] == len(pairs) and not rep["fails"]
    rep["summary"] = (f"{rep['compared']} of {len(pairs)} pairs compared: {len(rep['same'])} with holdout, forward, es, seeds, column equal and config "
                      f"and commands equal up to input hashes and the name map; {len(rep['fails'])} failing; missing {rep['missing']}; code files "
                      f"differing somewhere: {rep['code_files_differing_anywhere']}")
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(rep["summary"])
    for k, v in list(rep["fails"].items())[:12]:
        print(" FAIL", k, json.dumps(v, ensure_ascii=False)[:600])
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
