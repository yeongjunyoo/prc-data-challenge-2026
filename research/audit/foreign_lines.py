# SPDX-License-Identifier: GPL-3.0-only
"""Exact-line audit: lines (whitespace-normalised, >= 38 chars, not comments/docstring-only) that occur in both a saved foreign Python file
(or fenced code block of saved foreign markdown) and one of our Python files. Read-only. Usage: python -I foreign_lines.py <root> <out>"""
import os, re, sys, json
from collections import defaultdict
ROOT, OUT = os.path.abspath(sys.argv[1]), sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FOREIGN = [
    "research/r10_a/web/gh_arnavhm_prc_26/analysis_build_pru_calibration.py",
    "research/r10_a/web/gh_arnavhm_prc_26/analysis_oa10_stage1_427.py",
    "research/r10_a/web/gh_arnavhm_prc_26/analysis_oa11_position.py",
    "research/r10_a/web/gh_niclaswue_ATOW-Prediction_download_additional_data.py",
    "research/r14_c/web/ravindra_priors.py",
]
SKIP = {".venv", "__pycache__", "web", ".git", "gpu_run", "_work", "_release_stage", "docs_release", "_backup", "_mac_stage", "cache", "data", "logs", "preds", "subs", "queue", "notes"}
def norm(l):
    l = re.sub(r"\s+", " ", l.strip())
    return l
def lines_of(path):
    out = []
    for l in open(path, encoding="utf-8", errors="replace"):
        n = norm(l)
        if len(n) >= 38 and not n.startswith("#") and not n.startswith('"""') and not n.startswith("print(") and not n.startswith("import ") and not n.startswith("from "):
            out.append(n)
    return out
fl = defaultdict(set)
for f in FOREIGN:
    p = os.path.join(ROOT, f)
    if os.path.exists(p):
        for n in lines_of(p):
            fl[n].add(os.path.basename(f))
res = defaultdict(lambda: defaultdict(list))
for top in ("src", "tools", "research"):
    for dp, dn, fn in os.walk(os.path.join(ROOT, top)):
        dn[:] = [d for d in dn if d not in SKIP]
        for n in fn:
            if not n.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dp, n), ROOT).replace("\\", "/")
            if rel in FOREIGN:
                continue
            for ln in set(lines_of(os.path.join(dp, n))):
                if ln in fl:
                    res[rel][",".join(sorted(fl[ln]))].append(ln)
out = {}
for rel, d in res.items():
    for src, lns in d.items():
        out.setdefault(rel, {})[src] = lns
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
tot = sorted(((sum(len(v) for v in d.values()), rel) for rel, d in out.items()), reverse=True)
for n, rel in tot[:30]:
    print(n, rel, {k: len(v) for k, v in out[rel].items()})
