# SPDX-License-Identifier: GPL-3.0-only
"""F3 final root: src/feats_ctx.py (imported, unchanged) with the stand_p vocabulary frozen. src/feats_ctx.py numbers stand_p by sorting
the prefixes of the rows it sees, so the final file's new prefixes (LEMD_PE, LSZH_WE) would shift 88 of the 159 codes and break the
corrector fit cache (its keys hash the exact training matrix). Here the codes of today's ctx_v1 (sha16 given) are kept and any new prefix
gets the next free code, in sorted order. Everything else is feats_ctx.main: --months 1,7 of 2025 plus data/ranking.parquet (= final).
Run from the root (cwd):
    python -X utf8 research/f3/feats_ctx_final.py --name ctx_v1 --months 1,7 --vocab_from <today's ctx_v1.parquet> --vocab_sha 0383b7a4b36c5821
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, ".")
import src.feats_ctx as fc  # noqa: E402

ROOT = fc.ROOT


def main():
    pa_ = argparse.ArgumentParser()
    pa_.add_argument("--name", default="ctx_v1")
    pa_.add_argument("--months", default="1,7")
    pa_.add_argument("--vocab_from", required=True)
    pa_.add_argument("--vocab_sha", required=True)
    a = pa_.parse_args()
    fc.below_normal()
    vp = Path(a.vocab_from)
    if fc.sha16(vp) != a.vocab_sha:
        raise SystemExit(f"{vp} sha {fc.sha16(vp)} != {a.vocab_sha}")
    V = pd.read_parquet(vp, columns=["stand_p", "stand_p_c"]).drop_duplicates()
    assert V["stand_p"].is_unique and V["stand_p_c"].is_unique
    vocab = dict(zip(V["stand_p"], V["stand_p_c"].astype(int)))
    assert sorted(vocab.values()) == list(range(len(vocab)))
    want = {int(m) for m in a.months.split(",")}
    out = []
    for f in sorted(glob.glob(str(ROOT / "data" / "training_*.parquet"))):
        if int(Path(f).name[14:16]) not in want:
            continue
        out.append(fc.build(fc.read(f)))
        print(Path(f).name, len(out[-1]), flush=True)
    d = fc.read(str(ROOT / "data" / "ranking.parquet"))
    out.append(fc.build(d))
    print("ranking", len(out[-1]), flush=True)
    F = pd.concat(out, ignore_index=True)
    assert F["MVT_ID_mvt"].is_unique
    new = sorted(set(F["stand_p"].unique()) - set(vocab))
    for s in new:
        vocab[s] = len(vocab)
    F["stand_p_c"] = F["stand_p"].map(vocab).astype("int32")
    p = ROOT / "cache" / f"{a.name}.parquet"
    assert not p.exists(), f"{p} exists"
    F.to_parquet(p, index=False)
    fill = {part: {c: round(float(g[c].notna().mean()), 4) for c in fc.FEATS} for part, g in F.groupby("part")}
    meta = {"name": a.name, "file": f"cache/{a.name}.parquet", "sha16": fc.sha16(p), "rows": int(len(F)), "parts": sorted(F["part"].unique().tolist()),
            "columns": fc.FEATS + ["stand_p", "stand_p_c", "t", "aobt3", "day", "part"], "stand_p_vocab": len(vocab), "fill": fill,
            "code": "src/feats_ctx.py", "code_sha16": fc.sha16(ROOT / "src" / "feats_ctx.py"),
            "vocab_frozen_from": {"file": str(vp), "sha16": a.vocab_sha, "codes": len(vocab) - len(new)},
            "new_prefixes": {s: {"code": vocab[s], "rows": int((F["stand_p"] == s).sum())} for s in new},
            "builder": "research/f3/feats_ctx_final.py", "builder_sha16": fc.sha16(Path(__file__))}
    (ROOT / "cache" / f"{a.name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("name", "sha16", "rows", "parts", "stand_p_vocab", "new_prefixes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
