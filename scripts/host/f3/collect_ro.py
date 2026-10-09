# SPDX-License-Identifier: GPL-3.0-only
"""F3 final root: copy (APFS clone, read-only use) the val_ and fwd_ caches and iters_*.json that the full-setting rebuild of 514 reads,
from the Mac sandboxes where the three chains were built. A file present in several sandboxes must be byte-identical there; a conflict is
reported and the file is taken from the chain's own builder sandbox (A: r16b_sbx, B: r10b_sbx, C: r13f3_sbx, legs: r16b_sbx first).
full_ caches are never copied (every full output is rebuilt, so a missed rebuild fails loudly). Tail-stage caches are not copied (rebuilt
in all settings; their val/fwd outputs are then compared with today's as a regression check). Writes research/f3/out/collect_ro.json.
    python -X utf8 research/f3/collect_ro.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

P = Path(os.environ.get("PRC_WORK", "."))  # Mac work folder (the build sandboxes and final_run)
D = P / "final_run" / "cache"
SB = ["r16b_sbx", "r13l_sbx", "r10b_sbx", "r13f3_sbx"]
LEG = r"m6r27(skwin|sk|skap|cbwin|cb|cbap|rmwin|rm|tmcwin|tmc)(g|gx4|x2|x3|x4)?"
REF = r"m6r25(|s|sg|sgoof4|cb|xg|rm)"
BASES = r"m6r27(wincbrmfw4sv434(tmc)?|skcbrmfw4sv434|apcbrmfw4sv114)(adsbgeondctxv3x50w(c3)?)?"
PAT = re.compile(rf"^(val|fwd)_({LEG}|{REF}|{BASES})_s[0-9]\.parquet$")
PREF = {"skcbrmfw4sv434": "r10b_sbx", "apcbrmfw4sv114": "r13f3_sbx", "wincbrmfw4sv434": "r16b_sbx"}


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    found: dict[str, dict[str, str]] = {}
    for s in SB:
        for f in (P / s / "cache").iterdir():
            if PAT.match(f.name) or f.name.startswith("iters_"):
                found.setdefault(f.name, {})[s] = sha16(f)
    rep = {"copied": {}, "conflicts": {}}
    for name, by in sorted(found.items()):
        shas = set(by.values())
        pick = next(iter(by))
        if len(shas) > 1:
            pref = next((v for k, v in PREF.items() if k in name), "r16b_sbx")
            pick = pref if pref in by else pick
            rep["conflicts"][name] = {"by_sandbox": by, "picked": pick}
        dst = D / name
        assert not dst.exists(), f"exists: {dst}"
        subprocess.run(["cp", "-cp", str(P / pick / "cache" / name), str(dst)], check=True)
        rep["copied"][name] = {"from": pick, "sha16": by[pick]}
    (P / "final_run" / "research" / "f3" / "out" / "collect_ro.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(len(rep["copied"]), "copied;", len(rep["conflicts"]), "conflicts")
    for k, v in rep["conflicts"].items():
        print(" conflict", k, v)


if __name__ == "__main__":
    main()
