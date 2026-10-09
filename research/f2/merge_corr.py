# SPDX-License-Identifier: GPL-3.0-only
"""F2 final split kit: corrector-stage merge by row group (coordinator 10-09 23:4x; vault research/2026-10-08-f2-four-months.md 5.8; the split is
post hoc, raised after chain A's gate14 numbers, and the coordinator decides). One chain and one setting per call, after research/r10_b/cavg.py,
in the root it runs from (final_run). Row groups of the test rows (coordinator 23:5x): jj = the MVT_ID set of --jj_ids (research/f2/janjul_ids.parquet,
the test ids of the original Jan/Jul ranking.parquet); fu = every other test row without an ADS-B match; fm = with one. The match is the row's MVT_ID_mvt among the matched == True ids of
cache/<adsb>.parquet (the corrector's s1_matched: the table its stage 1 reads); both tables sha16-checked; the ctx part of every row must agree
(jj rows 2026-01 or 2026-07, the others 2026-02 or 2026-06).
  --setting full : cache/full_<tag>_s0.parquet = the default's file (cache/full_<default>_s0.parquet, the cavg tag of the T 0.75 corrector, the
                   531 configuration) with pred taken row by row from the group's source (--jj, --fu, --fm: cavg tags). Every other column comes
                   from the corrector's base and must be equal in all sources (src/lobt_window.project's p_art does not depend on pred); the row
                   order is the default's.
  --setting val, fwd : cache/<setting>_<tag>_s0.parquet = a byte copy of the default's (the variants differ only on Feb/Jun test rows; the tails
                   fit on val, so their fits are the default's), and preds/es_valid_ids_<tag>_<setting>_s0.parquet copied from the default's
                   (as cavg does).
--base/--base_seed name the blend the corrector ran on (artmm, arthist, aiu_shift and gkey read them from the manifest config); checked as in cavg:
mu, g, dsch and y of the written file equal cache/<setting>_<base>_s<base_seed>.parquet.
Variants: D (the null merge) = --jj, --fu, --fm all the default; A = --fm the T 0.5 cavg tag (514 configuration); Ba (594a) = --fu and --fm the
four-month 0.75 cavg tag; Bb (594b) = --fu the four-month 0.75 and --fm the four-month 0.5 cavg tag (the twin from the same stage-2 models). Refuses to overwrite. After writing, the file is read back and every group must equal its source
row by row in every column. Writes notes/merge_<tag>_<setting>.json (groups, counts, sources and their sha16, output sha16).
    python -X utf8 -u research/f2/merge_corr.py --setting <val|fwd|full> --seed 0 --tag <new tag> --base <blend> --base_seed 0
        --default <cavg tag> --jj <cavg tag> --fu <cavg tag> --fm <cavg tag> --ctx ctx_v3 --ctx_sha <16> --adsb adsb_v1nd --adsb_sha <16>
        [--jj_ids research/f2/janjul_ids.parquet] [--expect 2026-02:106542/144158,2026-06:127966/181791] [--root .]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
JJ, FJ = ("2026-01", "2026-07"), ("2026-02", "2026-06")


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


def same(x: np.ndarray, z: np.ndarray) -> bool:
    """Bit-level equality of two columns (NaN at the same places counts as equal)."""
    if x.shape != z.shape or x.dtype != z.dtype:
        return False
    if x.dtype.kind == "f":
        return bool(np.array_equal(np.ascontiguousarray(x).view(f"u{x.itemsize}"), np.ascontiguousarray(z).view(f"u{z.itemsize}")))
    if x.dtype.kind == "O":
        return bool(pd.Series(x).equals(pd.Series(z)))
    return bool(np.array_equal(x, z))


def group_map(root: Path, ids: np.ndarray, ctx: str, ctx_sha: str, adsb: str, adsb_sha: str, jj_ids: str) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    pc, pa_ = root / "cache" / f"{ctx}.parquet", root / "cache" / f"{adsb}.parquet"
    for p, want in ((pc, ctx_sha), (pa_, adsb_sha)):
        got = sha16(p)
        if got != want:
            raise SystemExit(f"{p.name} sha16 {got} != {want}")
    c = pd.read_parquet(pc, columns=["MVT_ID_mvt", "part"])
    c = c[c["MVT_ID_mvt"].isin(ids)]
    if not c["MVT_ID_mvt"].is_unique:
        raise SystemExit(f"{ctx}: test ids appear more than once")
    part = c.set_index("MVT_ID_mvt")["part"].astype(str).reindex(ids)
    if part.isna().any():
        raise SystemExit(f"{ctx}: {int(part.isna().sum())} test rows without a context row")
    part = part.to_numpy()
    m = pq.read_table(pa_, columns=["MVT_ID_mvt"], filters=[("matched", "=", True)]).column(0).to_numpy()
    matched = np.isin(ids, m)
    rk = root / jj_ids
    rids = pd.read_parquet(rk, columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()
    jj = np.isin(ids, rids)
    if not np.isin(part[jj], JJ).all() or not np.isin(part[~jj], FJ).all():
        raise SystemExit(f"{jj_ids} ids and the ctx parts disagree: jj parts {sorted(set(part[jj]))}, other parts {sorted(set(part[~jj]))}")
    return {"jj": jj, "fu": ~jj & ~matched, "fm": ~jj & matched}, part, matched, {"path": jj_ids, "sha16": sha16(rk), "rows": int(len(rids))}


def base_check(root: Path, df: pd.DataFrame, setting: str, base: str, base_seed: int) -> None:
    bb = pd.read_parquet(root / "cache" / f"{setting}_{base}_s{base_seed}.parquet")
    if not np.array_equal(bb["MVT_ID_mvt"].to_numpy(), df["MVT_ID_mvt"].to_numpy()):
        raise SystemExit(f"row order differs from the base {base}")
    for c in ("mu", "g", "dsch", "y"):
        if c in df.columns and not same(df[c].to_numpy(), bb[c].to_numpy()):
            raise SystemExit(f"column {c} differs from the base {base}")


def main() -> int:
    pa = argparse.ArgumentParser()
    for k in ("--tag", "--base", "--default", "--jj", "--fu", "--fm", "--ctx", "--ctx_sha", "--adsb", "--adsb_sha"):
        pa.add_argument(k, required=True)
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--seed", type=int, default=0, choices=[0])
    pa.add_argument("--base_seed", type=int, default=0)
    pa.add_argument("--expect", default="2026-02:106542/144158,2026-06:127966/181791", help="part:matched/rows checks ('' = none)")
    pa.add_argument("--jj_ids", default="research/f2/janjul_ids.parquet", help="parquet whose MVT_ID_mvt set defines the Jan/Jul rows")
    pa.add_argument("--root", default=".")
    a = pa.parse_args()
    t0 = time.time()
    root = Path(a.root).resolve()
    C, PR, N = root / "cache", root / "preds", root / "notes"
    if a.tag in (a.default, a.jj, a.fu, a.fm):
        raise SystemExit("the new tag must differ from every source tag")
    out = C / f"{a.setting}_{a.tag}_s0.parquet"
    rp = N / f"merge_{a.tag}_{a.setting}.json"
    dpath = C / f"{a.setting}_{a.default}_s0.parquet"
    es_src, es_dst = PR / f"es_valid_ids_{a.default}_{a.setting}_s0.parquet", PR / f"es_valid_ids_{a.tag}_{a.setting}_s0.parquet"
    for p in (out, rp) + ((es_dst,) if a.setting != "full" else ()):
        if p.exists():
            raise SystemExit(f"exists, refusing to overwrite: {p}")
    rep = {"what": "F2 final split kit, corrector-stage merge (research/f2/merge_corr.py)", "setting": a.setting, "tag": a.tag, "base": a.base,
           "default": {"tag": a.default, "path": str(dpath.relative_to(root)), "sha16": sha16(dpath)}, "ctx": [a.ctx, a.ctx_sha], "adsb": [a.adsb, a.adsb_sha],
           "code_sha16": sha16(Path(__file__))}
    D = pd.read_parquet(dpath)
    if a.setting != "full":
        base_check(root, D, a.setting, a.base, a.base_seed)
        tmp = out.with_name(out.name + ".tmp")
        shutil.copyfile(dpath, tmp)
        tmp.replace(out)
        if sha16(out) != rep["default"]["sha16"]:
            raise SystemExit(f"{out.name}: the copy differs from the default's bytes")
        if es_src.exists():
            shutil.copyfile(es_src, es_dst)
            rep["es_valid_ids"] = {"from": str(es_src.relative_to(root)), "sha16": sha16(es_dst)}
        else:
            rep["es_valid_ids"] = "absent in the default (not copied)"
        rep["note"] = "byte copy of the default's file"
    else:
        ids = D["MVT_ID_mvt"].to_numpy()
        if not D["MVT_ID_mvt"].is_unique:
            raise SystemExit("default: duplicate MVT_ID_mvt")
        G, part, matched, rep["jj_ids"] = group_map(root, ids, a.ctx, a.ctx_sha, a.adsb, a.adsb_sha, a.jj_ids)
        counts = {p: [int(((part == p) & matched).sum()), int((part == p).sum())] for p in sorted(set(part))}
        rep["counts_by_part"] = {p: f"{m_}/{n_}" for p, (m_, n_) in counts.items()}
        rep["groups"] = {g: int(v.sum()) for g, v in G.items()}
        for e in [x for x in a.expect.split(",") if x]:
            p, mn = e.split(":")
            if rep["counts_by_part"].get(p) != mn:
                raise SystemExit(f"group counts: {p} {rep['counts_by_part'].get(p)} != expected {mn}")
        srcs = {"jj": a.jj, "fu": a.fu, "fm": a.fm}
        frames, rep["sources"] = {a.default: D}, {}
        for g, t in srcs.items():
            if t not in frames:
                sp = C / f"full_{t}_s0.parquet"
                S = pd.read_parquet(sp)
                if list(S.columns) != list(D.columns):
                    raise SystemExit(f"{sp.name}: columns differ from the default's")
                if not np.array_equal(S["MVT_ID_mvt"].to_numpy(), ids):
                    raise SystemExit(f"{sp.name}: row order differs from the default's")
                for c in D.columns:
                    if c != "pred" and not same(S[c].to_numpy(), D[c].to_numpy()):
                        raise SystemExit(f"{sp.name}: column {c} differs from the default's (only pred may differ)")
                frames[t] = S
            rep["sources"][g] = {"tag": t, "sha16": sha16(C / f"full_{t}_s0.parquet")}
        pred = D["pred"].to_numpy(copy=True)
        for g, t in srcs.items():
            pred[G[g]] = frames[t]["pred"].to_numpy()[G[g]]
        res = D.copy()
        res["pred"] = pred
        base_check(root, res, "full", a.base, a.base_seed)
        if not (np.isfinite(pred).all() and (pred >= 0).all()):
            raise SystemExit("merged pred has non-finite or negative values")
        tmp = out.with_name(out.name + ".tmp")
        res.to_parquet(tmp, index=False)
        tmp.replace(out)
        back = pd.read_parquet(out)
        if list(back.columns) != list(D.columns) or [str(x) for x in back.dtypes] != [str(x) for x in D.dtypes]:
            raise SystemExit("written file: columns or dtypes differ from the default's")
        for g, t in srcs.items():
            for c in D.columns:
                if not same(back[c].to_numpy()[G[g]], frames[t][c].to_numpy()[G[g]]):
                    raise SystemExit(f"written file: group {g} column {c} differs from its source {t}")
        dp = back["pred"].to_numpy() - D["pred"].to_numpy()
        rep["changed_rows_vs_default"] = {g: int((dp[G[g]] != 0).sum()) for g in G}
        rep["pred_shift_vs_default"] = {g: {"mean": round(float(dp[G[g]].mean()), 3), "mse": round(float((dp[G[g]] ** 2).mean()), 3)} for g in G}
    rep["out"] = {"path": str(out.relative_to(root)), "sha16": sha16(out)}
    rep["seconds"] = round(time.time() - t0, 1)
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"MERGE {a.tag} {a.setting}: {json.dumps({k: rep[k] for k in ('groups', 'changed_rows_vs_default', 'out') if k in rep}, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
