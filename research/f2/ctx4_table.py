# SPDX-License-Identifier: GPL-3.0-only
"""F2 (private notes research/2026-10-08-f2-four-months.md 5.7): the context table of the four-month corrector. Every row of --base (ctx_v3), unchanged,
plus the rows of --parts (2025-02, 2025-06) taken from --extra (f2ctx_v3: src/feats_ctx.py --months 1,2,6,7, src/feats_ctx2.py, src/feats_aiu.py
under new names, the table gate 1.4 used), with stand_p_c recoded into the base table's vocabulary (stand_p label -> the base code; labels the base
lacks get new codes after its largest, in sorted order). Every other column of the added rows is the extra table's value. So the base rows (Jan/Jul
2025 and the test rows) are exactly what 514 and 531 read, and the added rows are the rows gate 1.4 measured, up to the code numbers.
Checks: both inputs sha16-checked; same columns and Arrow types; base pairs (stand_p, stand_p_c) one to one, and the extra table's too; no added
MVT_ID in the base; after writing, the base rows read back equal the base file column by column and the added rows equal the extra rows (codes aside).
    python -X utf8 research/f2/ctx4_table.py --base ctx_v3 --base_sha <16> --extra f2ctx_v3 --extra_sha <16> --parts 2025-02,2025-06 --name f2ctx4_v3
Writes cache/<name>.parquet and cache/<name>.json; prints the output sha16 (the corrector's --ctx_sha).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def one_to_one(df: pd.DataFrame, what: str) -> dict:
    pr = df[["stand_p", "stand_p_c"]].drop_duplicates()
    if not (pr["stand_p"].is_unique and pr["stand_p_c"].is_unique):
        raise SystemExit(f"{what}: stand_p and stand_p_c are not one to one")
    return dict(zip(pr["stand_p"].tolist(), pr["stand_p_c"].astype(int).tolist()))


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("--base", required=True)
    pa.add_argument("--base_sha", required=True)
    pa.add_argument("--extra", required=True)
    pa.add_argument("--extra_sha", required=True)
    pa.add_argument("--parts", required=True)
    pa.add_argument("--name", required=True)
    pa.add_argument("--cache", default=str(CACHE), help="folder of the tables (a test uses another one)")
    g = pa.parse_args()
    cdir = Path(g.cache)
    pb, pe, dst = cdir / f"{g.base}.parquet", cdir / f"{g.extra}.parquet", cdir / f"{g.name}.parquet"
    for p, want in ((pb, g.base_sha), (pe, g.extra_sha)):
        if sha16(p) != want:
            raise SystemExit(f"{p.name} sha16 {sha16(p)} != {want}")
    if dst.exists():
        raise SystemExit(f"{dst.name} exists, refusing")
    sb, se = pq.read_schema(pb), pq.read_schema(pe)
    if list(sb.names) != list(se.names) or any(str(sb.field(c).type) != str(se.field(c).type) for c in sb.names):
        raise SystemExit("base and extra tables differ in columns or types")
    B, E = pd.read_parquet(pb), pd.read_parquet(pe)
    parts = sorted(set(g.parts.split(",")))
    if B["part"].isin(parts).any():
        raise SystemExit(f"the base already holds rows of {parts}")
    Ea = E[E["part"].isin(parts)].copy().reset_index(drop=True)
    if sorted(Ea["part"].unique().tolist()) != parts:
        raise SystemExit(f"the extra table lacks some of {parts}")
    if np.isin(Ea["MVT_ID_mvt"].to_numpy(), B["MVT_ID_mvt"].to_numpy()).any():
        raise SystemExit("an added MVT_ID is already a base row")
    vb = one_to_one(B, g.base)
    one_to_one(E, g.extra)
    new = sorted(set(Ea["stand_p"].tolist()) - set(vb))
    vocab = dict(vb)
    for i, s in enumerate(new):
        vocab[s] = max(vb.values()) + 1 + i
    old_codes = Ea["stand_p_c"].to_numpy().copy()
    Ea["stand_p_c"] = Ea["stand_p"].map(vocab).astype(B["stand_p_c"].dtype)
    out = pd.concat([B, Ea], ignore_index=True)
    if list(out.dtypes) != list(B.dtypes):
        raise SystemExit("concat changed a dtype")
    out.to_parquet(dst, index=False)
    R = pd.read_parquet(dst)
    head, tail = R.iloc[:len(B)].reset_index(drop=True), R.iloc[len(B):].reset_index(drop=True)
    bad = [c for c in B.columns if not head[c].equals(B[c])]
    bad += [f"added:{c}" for c in B.columns if c != "stand_p_c" and not tail[c].equals(Ea[c])]
    if bad:
        dst.unlink()
        raise SystemExit(f"read-back differs in {bad}")
    so = pq.read_schema(dst)
    if any(str(so.field(c).type) != str(sb.field(c).type) for c in sb.names):
        dst.unlink()
        raise SystemExit("the written table's Arrow types differ from the base's")
    meta = {"name": g.name, "file": f"cache/{g.name}.parquet", "sha16": sha16(dst), "rows": int(len(R)), "base": g.base, "base_sha16": g.base_sha,
            "base_rows": int(len(B)), "extra": g.extra, "extra_sha16": g.extra_sha, "parts_added": {p: int((Ea["part"] == p).sum()) for p in parts},
            "stand_p_codes_changed_in_added_rows": int((old_codes != Ea["stand_p_c"].to_numpy()).sum()), "new_stand_p_labels": new,
            "code": "research/f2/ctx4_table.py", "code_sha16": sha16(Path(__file__))}
    (cdir / f"{g.name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
