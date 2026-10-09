# SPDX-License-Identifier: GPL-3.0-only
"""F3 final root: build cache/all.parquet and cache/feats_v25.parquet from data/ (ranking.parquet = final_ranking) with the unchanged
src/load.py and src/feats.py. feats.py refuses to rebuild a frozen name (feats_v25), so the build runs under a temporary name and the file
is renamed; the name only chooses the output path and the v2.6 switch (feats_v25 and the temporary name both take the v2.5 path).
Run from the root (cwd = root):
    python -X utf8 research/f3/build_feats_final.py [--skip_all]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import src.feats as feats  # noqa: E402
from src.load import CACHE, load_all  # noqa: E402

TMP = "feats_f3build_v25"


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    t0 = time.time()
    assert Path("src/feats.py").resolve().parent.parent == Path.cwd().resolve(), "run from the root"
    assert TMP not in feats.V26_NAMES
    if "--skip_all" not in sys.argv:
        assert not (CACHE / "all.parquet").exists(), "cache/all.parquet exists"
        df = load_all(rebuild=True)
        print("all.parquet", df.shape, df.groupby(["src", "PHASE_mvt"], observed=True).size().to_dict(), round(time.time() - t0, 1), "s", flush=True)
        del df
    out = CACHE / "feats_v25.parquet"
    assert not out.exists(), "cache/feats_v25.parquet exists"
    F = feats.build(rebuild=True, name=TMP)
    os.replace(CACHE / f"{TMP}.parquet", out)
    meta = {"name": "feats_v25", "built_as": TMP, "rows": int(len(F)), "rows_src0": int((F["src"] == 0).sum()), "columns": list(F.columns),
            "sha16": sha16(out), "all_sha16": sha16(CACHE / "all.parquet"), "ranking_sha16": sha16(Path("data/ranking.parquet")),
            "code": {p: sha16(Path(p)) for p in ("src/load.py", "src/feats.py", "research/f3/build_feats_final.py")},
            "seconds": round(time.time() - t0, 1), "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (CACHE / "feats_v25.f3build.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "columns"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
