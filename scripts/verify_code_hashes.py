# SPDX-License-Identifier: GPL-3.0-only
"""Check the source files of this repository against the hashes recorded in the manifests in preds/.

Every manifest lists, under "code_sha", the first 16 hex characters of the SHA-256 of each source file as it was when that chain step ran
(the daily submissions v35 to v39, the first final file in preds/first_final/, the final file in preds/final/); the final file's manifest in preds/ also lists, under
"code_sha_outside_preds", the scripts whose copies were kept beside its queue entry. This repository adds one SPDX license line at the top
of each source file (after the "#!" line where there is one) and, in src/s3io.py, replaces the function that reads the access key. The check
removes the SPDX line before hashing; src/s3io.py is reported separately.

    python scripts/verify_code_hashes.py            # one line per file and a summary
    python scripts/verify_code_hashes.py --final    # only the final file's chain: preds/final/ and the manifest in preds/ with a final_of field
Exit code 0 if every file matches one of the hashes recorded for it.
"""
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPDX = "# SPDX-License-Identifier: GPL-3.0-only"


def sha16_without_spdx(path):
    raw = open(path, "rb").read().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw[:2000] else "\n"
    lines = raw.split(nl)
    out = [ln for i, ln in enumerate(lines) if not (ln == SPDX and i <= 1)]
    return hashlib.sha256(nl.join(out).encode("utf-8")).hexdigest()[:16]


def main():
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
    bad = 0
    for f in sorted(want):
        p = os.path.join(ROOT, f)
        if not os.path.isfile(p):
            print("MISSING", f)
            bad += 1
            continue
        got = sha16_without_spdx(p)
        ok = got in want[f]
        print("%-6s %-40s %s %s" % ("ok" if ok else "DIFF", f, got, ",".join(sorted(want[f]))))
        if not ok and f != "src/s3io.py":
            bad += 1
    print("manifests:", n_man, "files:", len(want), "problems:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
