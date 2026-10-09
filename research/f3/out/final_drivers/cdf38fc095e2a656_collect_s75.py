"""F3 second rebuild, Default final = 531 (coordinator 10-09 23:2x): clone (APFS, read-only use) from var1_sbx, the sandbox where 531 was
built, the val_ and fwd_ caches of the three chains' scale2-0.75 corrector stage (seeds 0, 1, 2) and of its seed average (c3), and the c3
early-stopping id files the manifest step reads. Same rule as research/f3/collect_ro.py: full_ caches are never copied (every full output
is rebuilt on the final data, so a missed rebuild fails loudly); tail-stage caches are not copied (the tails rebuild all settings and their
val, fwd and es outputs are then compared with 531's live files through the manifests). The es files must equal the live root's (sha16 below,
read on the PC 10-09 23:3x). Refuses to overwrite. Writes research/f3/out/collect_s75.json.
    python -X utf8 research/f3/collect_s75.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

P = Path.home() / "nova-adsb" / "prcm"
S = P / "var1_sbx"
D = P / "final_run"
CB = ["m6r27wincbrmfw4sv434tmc", "m6r27skcbrmfw4sv434", "m6r27apcbrmfw4sv114"]
MID = "adsbgeondctxv3x50ws75"
LIVE_ES = {"val": "dd793de93000e14e", "fwd": "925e0b6eceb77d24"}  # prc-2026/preds/es_valid_ids_<CB>adsbgeondctxv3x50ws75c3_<setting>_s0.parquet


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    todo = []
    for c in CB:
        for st in ("val", "fwd"):
            for k in (0, 1, 2):
                todo.append(("cache", f"{st}_{c}{MID}_s{k}.parquet", None))
            todo.append(("cache", f"{st}_{c}{MID}c3_s0.parquet", None))
            todo.append(("preds", f"es_valid_ids_{c}{MID}c3_{st}_s0.parquet", LIVE_ES[st]))
    for sub, name, _ in todo:
        assert (S / sub / name).exists(), f"missing in var1_sbx: {sub}/{name}"
        assert not (D / sub / name).exists(), f"exists in final_run: {sub}/{name}"
    rep = {"from": str(S), "copied": {}}
    for sub, name, want in todo:
        h = sha16(S / sub / name)
        if want is not None:
            assert h == want, f"{name}: var1_sbx sha16 {h} differs from the live root's {want}"
        subprocess.run(["cp", "-cp", str(S / sub / name), str(D / sub / name)], check=True)
        assert sha16(D / sub / name) == h
        rep["copied"][f"{sub}/{name}"] = {"sha16": h, "live_sha16": want}
    (D / "research" / "f3" / "out" / "collect_s75.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(len(rep["copied"]), "copied from var1_sbx (es files equal to the live root's)")


if __name__ == "__main__":
    main()
