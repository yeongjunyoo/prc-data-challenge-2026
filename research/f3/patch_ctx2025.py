# SPDX-License-Identifier: GPL-3.0-only
"""F3 final root: make the 2025 rows of the rebuilt cache/ctx_v2.parquet byte-identical to today's ctx_v2 (sha16 ac01294f15b8a618).
The rebuild differs there in one cell: src/feats_ctx2.py sets st_k = 0 for a stand with no arrival anywhere in data/, and one EDDM stand
used by a 2025-01 departure gains an arrival in the final file's Feb/Jun 2026 rows, so that row's st_k turns from 0 to NaN (no previous
departure). The corrector trains on these 2025 rows and its fit cache keys hash them, so today's values are kept (the 2026 rows stay as
rebuilt). Refuses if any other 2025 cell differs. Writes cache/ctx_v2.parquet in place (old file kept as cache/ctx_v2.unpatched.parquet)
and cache/ctx_v2.json gains a "patched_2025" entry. ctx_v3 is then rebuilt from it with the unchanged src/feats_aiu.py.
    python -X utf8 research/f3/patch_ctx2025.py <today's ctx_v2.parquet> ac01294f15b8a618
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    old_p, old_sha = Path(sys.argv[1]), sys.argv[2]
    assert sha16(old_p) == old_sha
    cp = Path("cache/ctx_v2.parquet")
    N, O = pd.read_parquet(cp), pd.read_parquet(old_p)
    assert list(N.columns) == list(O.columns)
    mN, mO = N["part"].astype(str).str.startswith("2025").to_numpy(), O["part"].astype(str).str.startswith("2025").to_numpy()
    iN, iO = np.nonzero(mN)[0], np.nonzero(mO)[0]
    assert len(iN) == len(iO) and (N["MVT_ID_mvt"].to_numpy()[iN] == O["MVT_ID_mvt"].to_numpy()[iO]).all()
    changed = {}
    for c in N.columns:
        a, b = N[c].to_numpy()[iN], O[c].to_numpy()[iO]
        if a.dtype.kind == "f":
            d = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        else:
            d = a != b
        if d.any():
            changed[c] = {"rows": int(d.sum()), "ids": N["MVT_ID_mvt"].to_numpy()[iN][d][:10].tolist(),
                          "new": [None if (isinstance(x, float) and np.isnan(x)) else x for x in a[d][:10].tolist()],
                          "old": [None if (isinstance(x, float) and np.isnan(x)) else x for x in b[d][:10].tolist()]}
            col = N[c].to_numpy().copy()
            col[iN] = b
            N[c] = pd.Series(col, dtype=N[c].dtype)
    assert set(changed) <= {"st_k"} and sum(v["rows"] for v in changed.values()) <= 1, changed
    os.replace(cp, "cache/ctx_v2.unpatched.parquet")
    N.to_parquet(cp, index=False)
    meta = json.loads(Path("cache/ctx_v2.json").read_text(encoding="utf-8"))
    meta["sha16_unpatched"] = meta["sha16"]
    meta["sha16"] = sha16(cp)
    meta["patched_2025"] = {"from": str(old_p), "from_sha16": old_sha, "changed": changed, "tool": "research/f3/patch_ctx2025.py",
                            "why": "st_k of one 2025-01 EDDM row: 0 in today's table (its stand had no arrival in data/), NaN in the rebuild (the final "
                                   "file adds an arrival on that stand in Feb/Jun 2026); today's value kept so the corrector's training rows and fit cache stay"}
    Path("cache/ctx_v2.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"sha16": meta["sha16"], "changed": changed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
