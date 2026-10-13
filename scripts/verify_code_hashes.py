# SPDX-License-Identifier: GPL-3.0-only
"""Check the source files of this repository against the hashes recorded in the manifests in preds/.

Every manifest lists, under "code_sha", the first 16 hex characters of the SHA-256 of each source file as it was when that chain step ran
(the daily submissions v35 to v39, the first final file in preds/first_final/, the final file in preds/final/); the final file's manifest in preds/ also lists, under
"code_sha_outside_preds", the scripts whose copies were kept beside its queue entry. This repository adds one SPDX license line at the top
of each source file (after the "#!" line where there is one), English docstrings in some files of src/ and tools/, and, in src/s3io.py,
replaces the function that reads the access key. The check removes the SPDX line and puts back the lines that the docstrings replaced,
as recorded in scripts/release_docstrings.json, before hashing; src/s3io.py is reported separately.

    python scripts/verify_code_hashes.py                      # one line per file and a summary
    python scripts/verify_code_hashes.py --final              # only the final file's chain: preds/final/ and the manifest in preds/ with a final_of field
    python scripts/verify_code_hashes.py --as-run src/feats.py    # print a file as it ran (without the release's docstrings and SPDX line)
Exit code 0 if every file matches one of the hashes recorded for it.
"""
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPDX = "# SPDX-License-Identifier: GPL-3.0-only"
PATCH = os.path.join(ROOT, "scripts", "release_docstrings.json")


def load_patches():
    """The release's docstring hunks per file (empty if the record is missing)."""
    if not os.path.isfile(PATCH):
        return {}
    return json.load(open(PATCH, encoding="utf-8"))["files"]


def as_run(path, rel, patches):
    """(text as it ran, newline, problem). Puts back the lines the release's docstrings replaced, then removes the SPDX line."""
    raw = open(path, "rb").read().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw[:2000] else "\n"
    lines = raw.split(nl)
    for h in sorted((patches.get(rel) or {}).get("hunks", []), key=lambda h: -h["line"]):
        i = h["line"] - 1
        here = "\n".join(lines[i:i + h["lines_here"]])
        if hashlib.sha256(here.encode("utf-8")).hexdigest()[:16] != h["here_sha16"]:
            return None, nl, "the docstring lines of %s at line %d are not the recorded ones" % (h["target"], h["line"])
        lines[i:i + h["lines_here"]] = h["as_run"]
    out = [ln for i, ln in enumerate(lines) if not (ln == SPDX and i <= 1)]
    return nl.join(out), nl, None


def main():
    patches = load_patches()
    if "--as-run" in sys.argv[1:]:
        rel = sys.argv[sys.argv.index("--as-run") + 1].replace("\\", "/")
        text, nl, problem = as_run(os.path.join(ROOT, rel), rel, patches)
        if problem:
            print(problem)
            return 1
        sys.stdout.buffer.write(text.encode("utf-8"))
        return 0
    final_only = "--final" in sys.argv[1:]
    want, n_man = {}, 0
    for p in sorted(glob.glob(os.path.join(ROOT, "preds", "**", "manifest_*.json"), recursive=True)):
        m = json.load(open(p, encoding="utf-8"))
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        if final_only and not (rel.startswith("preds/final/") or (rel.count("/") == 1 and "final_of" in m)):
            continue
        n_man += 1
        for f, h in (m.get("code_sha") or {}).items():
            want.setdefault(f, set()).add(h)
        for f, v in (m.get("code_sha_outside_preds") or {}).items():
            want.setdefault(f, set()).add(v["sha"] if isinstance(v, dict) else v)
    bad = restored = 0
    for f in sorted(want):
        p = os.path.join(ROOT, f)
        if not os.path.isfile(p):
            print("MISSING", f)
            bad += 1
            continue
        text, nl, problem = as_run(p, f, patches)
        if problem:
            print("%-6s %-40s %s" % ("PATCH", f, problem))
            bad += 1
            continue
        got = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        ok = got in want[f]
        restored += f in patches
        print("%-6s %-40s %s %s%s" % ("ok" if ok else "DIFF", f, got, ",".join(sorted(want[f])), "  (docstrings put back)" if f in patches else ""))
        if not ok and f != "src/s3io.py":
            bad += 1
    print("manifests:", n_man, "files:", len(want), "problems:", bad)
    print("files whose release docstrings were put back before hashing:", restored, "(scripts/release_docstrings.json)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
