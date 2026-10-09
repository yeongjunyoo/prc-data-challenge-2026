# SPDX-License-Identifier: GPL-3.0-only
"""F2 final split kit: the five checks and a manifest (coordinator 10-09 23:4x; private notes research/2026-10-08-f2-four-months.md 5.8). Run from the
root (final_run). It can run twice: after the merges (corrector stage only) and again after the variants' tails and chainavg3 (with --finals).
Corrector stage (research/f2/merge_corr.py outputs):
  (ii) every merged tag: cache/full_<tag>_s0.parquet has the default's row order and in every group equals the group's source row by row in every
       column; cache/val_ and fwd_<tag>_s0.parquet are byte copies of the default's; the merge reports exist for val, fwd and full.
  (iv) a null merge (all three sources = its default) equals the default's full file in every column.
Final files (given as manifests):
  (i)   row set and order: every variant final's test file has the default final's MVT_ID_mvt in the same order, unique;
  (iii) Jan/Jul rows of every variant final equal the default final's bit for bit (no tail step pools across months);
  (iv)  a D final, when given, equals the default final bit for bit;
  (v)   src/shot.pred_problems (non-finite, duplicate ids, template coverage) and negatives 0 for each final, with the share above 3,600 s and the
        pred mean next to the default's. The formal readiness check is tools/enqueue.py --readiness-warn at staging.
Manifest research/f2/out/split_check.json: the kit's code sha16, the ctx4 table and fj extras sha16 when present, the group counts against the
expected Feb 106,542 / 144,158 and Jun 127,966 / 181,791, per merged tag and per final the sha16 and results, and pass.
    python -X utf8 -u research/f2/check_split.py --merged <tag>[,<tag>...] [--twins <tag75>:<tag50>,...] [--final_default preds/manifest_<spec>.json]
        [--finals D=preds/manifest_<spec>.json,Ba=...,Bb=...] [--ctx ctx_v3 --ctx_sha <16>] [--root .]
Twins: the four-month corrector's 0.75 and 0.5 runs share one stage-2 output (per seed, notes/ctx_corr_<tag>_full_s<k>_delta.parquet bit-equal).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
EXPECT = {"2026-02": "106542/144158", "2026-06": "127966/181791"}
KIT = ("research/f2/merge_corr.py", "research/f2/check_split.py", "research/f2/ctx_corr4.py", "research/f2/ctx4_table.py")
INPUTS = ("cache/f2ctx4f_v3.parquet", "cache/f2ctx_v3.parquet", "cache/fj_f2wincbrmfw4_s0.parquet", "cache/fj_f2skcbrmfw4_s0.parquet",
          "cache/fj_f2apcbrmfw4_s0.parquet")


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


def same(x: np.ndarray, z: np.ndarray) -> bool:
    if x.shape != z.shape or x.dtype != z.dtype:
        return False
    if x.dtype.kind == "f":
        return bool(np.array_equal(np.ascontiguousarray(x).view(f"u{x.itemsize}"), np.ascontiguousarray(z).view(f"u{z.itemsize}")))
    if x.dtype.kind == "O":
        return bool(pd.Series(x).equals(pd.Series(z)))
    return bool(np.array_equal(x, z))


def parts_of(root: Path, ids: np.ndarray, ctx: str, ctx_sha: str | None) -> np.ndarray:
    p = root / "cache" / f"{ctx}.parquet"
    if ctx_sha and sha16(p) != ctx_sha:
        raise SystemExit(f"{p.name} sha16 differs from {ctx_sha}")
    c = pd.read_parquet(p, columns=["MVT_ID_mvt", "part"])
    c = c[c["MVT_ID_mvt"].isin(ids)].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")["part"].astype(str).reindex(ids)
    if c.isna().any():
        raise SystemExit(f"{ctx}: {int(c.isna().sum())} rows without a part")
    return c.to_numpy()


def check_merged(root: Path, tag: str, fails: list[str]) -> dict:
    N, C = root / "notes", root / "cache"
    r = {}
    reps = {s: N / f"merge_{tag}_{s}.json" for s in ("val", "fwd", "full")}
    miss = [s for s, p in reps.items() if not p.exists()]
    if miss:
        fails.append(f"{tag}: merge report missing for {miss}")
        return {"missing": miss}
    R = {s: json.loads(p.read_text(encoding="utf-8")) for s, p in reps.items()}
    dtag = R["full"]["default"]["tag"]
    for s in ("val", "fwd"):
        a, b = C / f"{s}_{tag}_s0.parquet", C / f"{s}_{dtag}_s0.parquet"
        ok = a.exists() and b.exists() and sha16(a) == sha16(b)
        r[f"{s}_byte_copy"] = ok
        if not ok:
            fails.append(f"{tag}: {s} is not a byte copy of {dtag}")
    D = pd.read_parquet(C / f"full_{dtag}_s0.parquet")
    M = pd.read_parquet(C / f"full_{tag}_s0.parquet")
    ids = D["MVT_ID_mvt"].to_numpy()
    r["row_order_equal_default"] = bool(np.array_equal(M["MVT_ID_mvt"].to_numpy(), ids) and list(M.columns) == list(D.columns))
    if not r["row_order_equal_default"]:
        fails.append(f"{tag}: row order or columns differ from {dtag}")
        return r
    part = parts_of(root, ids, R["full"]["ctx"][0], R["full"]["ctx"][1])
    import pyarrow.parquet as pq
    jpath = R["full"].get("jj_ids", {}).get("path", "research/f2/janjul_ids.parquet")
    if sha16(root / jpath) != R["full"].get("jj_ids", {}).get("sha16"):
        fails.append(f"{tag}: {jpath} sha16 differs from the merge's")
    jjm = np.isin(ids, pd.read_parquet(root / jpath, columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy())
    adsb, adsb_sha = R["full"]["adsb"]
    if sha16(C / f"{adsb}.parquet") != adsb_sha:
        fails.append(f"{tag}: {adsb} sha16 differs from the merge's")
    mid = pq.read_table(C / f"{adsb}.parquet", columns=["MVT_ID_mvt"], filters=[("matched", "=", True)]).column(0).to_numpy()
    matched = np.isin(ids, mid)
    if not (np.isin(part[jjm], ("2026-01", "2026-07")).all() and np.isin(part[~jjm], ("2026-02", "2026-06")).all()):
        fails.append(f"{tag}: Jan/Jul ids and ctx parts disagree")
    G = {"jj": jjm, "fu": ~jjm & ~matched, "fm": ~jjm & matched}
    r["groups"] = {g: int(v.sum()) for g, v in G.items()}
    r["counts_by_part"] = {p: f"{int(((part == p) & matched).sum())}/{int((part == p).sum())}" for p in sorted(set(part))}
    for p, want in EXPECT.items():
        if r["counts_by_part"].get(p) != want:
            fails.append(f"{tag}: {p} counts {r['counts_by_part'].get(p)} != expected {want}")
    if r["groups"] != R["full"]["groups"]:
        fails.append(f"{tag}: group sizes differ from the merge report")
    srcs = {g: R["full"]["sources"][g]["tag"] for g in ("jj", "fu", "fm")}
    r["sources"] = srcs
    r["null_merge"] = all(t == dtag for t in srcs.values())
    for g, t in srcs.items():
        S = D if t == dtag else pd.read_parquet(C / f"full_{t}_s0.parquet")
        if not np.array_equal(S["MVT_ID_mvt"].to_numpy(), ids):
            fails.append(f"{tag}: source {t} row order differs")
            continue
        ok = all(same(M[c].to_numpy()[G[g]], S[c].to_numpy()[G[g]]) for c in D.columns)
        r[f"group_{g}_equals_source"] = ok
        if not ok:
            fails.append(f"{tag}: group {g} differs from its source {t}")
    if r["null_merge"]:
        ok = all(same(M[c].to_numpy(), D[c].to_numpy()) for c in D.columns)
        r["null_merge_equals_default"] = ok
        if not ok:
            fails.append(f"{tag}: null merge differs from {dtag}")
    r["full_sha16"] = sha16(C / f"full_{tag}_s0.parquet")
    return r


def check_twins(root: Path, pair: str, fails: list[str]) -> dict:
    """<tag 0.75>:<tag 0.5> of research/f2/ctx_corr4.py: per seed the stage-1 value and the unscaled stage-2 change (notes/ctx_corr_<tag>_full_s<k>_delta.parquet)
    must be identical bit for bit (one stage-2 output, two scales), and the two cavg files may differ in pred only."""
    t75, t50 = pair.split(":")
    r = {}
    for k in (0, 1, 2):
        pa_, pb_ = root / "notes" / f"ctx_corr_{t75}_full_s{k}_delta.parquet", root / "notes" / f"ctx_corr_{t50}_full_s{k}_delta.parquet"
        if not (pa_.exists() and pb_.exists()):
            r[f"s{k}_same_stage2"] = None
            fails.append(f"twins {pair}: seed {k} delta file missing ({pa_.name} or {pb_.name})")
            continue
        a, b = pd.read_parquet(pa_), pd.read_parquet(pb_)
        ok = list(a.columns) == list(b.columns) and all(same(a[c].to_numpy(), b[c].to_numpy()) for c in a.columns)
        r[f"s{k}_same_stage2"] = ok
        if not ok:
            fails.append(f"twins {pair}: seed {k} stage-1 or unscaled stage-2 output differs")
    A = pd.read_parquet(root / "cache" / f"full_{t75}c3_s0.parquet")
    B = pd.read_parquet(root / "cache" / f"full_{t50}c3_s0.parquet")
    ok = list(A.columns) == list(B.columns) and all(same(A[c].to_numpy(), B[c].to_numpy()) for c in A.columns if c != "pred")
    r["c3_differ_in_pred_only"] = ok
    r["c3_pred_rows_differ"] = int((A["pred"].to_numpy() != B["pred"].to_numpy()).sum())
    if not ok:
        fails.append(f"twins {pair}: the cavg files differ outside pred")
    return r


def test_of(root: Path, manifest: str) -> tuple[pd.DataFrame, dict]:
    m = json.loads((root / manifest).read_text(encoding="utf-8"))
    p = root / m["test"]["path"]
    return pd.read_parquet(p), {"spec": m.get("spec"), "test": m["test"]["path"], "sha16": sha16(p), "manifest_test_sha": m["test"].get("sha")}


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("--merged", default="")
    pa.add_argument("--twins", default="", help="<tag 0.75>:<tag 0.5>[,...] four-month corrector twins (one stage-2 output, two scales)")
    pa.add_argument("--final_default", default="")
    pa.add_argument("--finals", default="")
    pa.add_argument("--ctx", default="ctx_v3")
    pa.add_argument("--ctx_sha", default="")
    pa.add_argument("--root", default=".")
    pa.add_argument("--out", default="research/f2/out/split_check.json")
    a = pa.parse_args()
    t0 = time.time()
    root = Path(a.root).resolve()
    fails: list[str] = []
    rep = {"what": "F2 final split kit checks (research/f2/check_split.py)", "written": time.strftime("%Y-%m-%d %H:%M:%S"),
           "code_sha16": {f: sha16(root / f) for f in KIT if (root / f).exists()},
           "inputs_sha16": {f: sha16(root / f) for f in INPUTS if (root / f).exists()}, "expected_counts": EXPECT, "merged": {}, "finals": {}}
    rep["twins"] = {}
    for pair in [t for t in a.twins.split(",") if t]:
        rep["twins"][pair] = check_twins(root, pair, fails)
        print(f"(twins) {pair}: {rep['twins'][pair]}")
    for tag in [t for t in a.merged.split(",") if t]:
        rep["merged"][tag] = check_merged(root, tag, fails)
        print(f"(ii, iv) {tag}: {json.dumps(rep['merged'][tag], ensure_ascii=False)[:600]}")
    if a.final_default:
        DF, info = test_of(root, a.final_default)
        ids = DF["MVT_ID_mvt"].to_numpy()
        part = parts_of(root, ids, a.ctx, a.ctx_sha or None)
        jj = np.isin(part, ("2026-01", "2026-07"))
        try:
            sys.path.insert(0, str(root))
            import src.shot as S
            probs = lambda p: S.pred_problems(p)  # noqa: E731
        except Exception as e:  # noqa: BLE001
            probs = None
            rep["pred_problems_unavailable"] = repr(e)[:200]
        dpred = DF["pred"].to_numpy(dtype=float)
        info.update(rows=int(len(DF)), unique=bool(DF["MVT_ID_mvt"].is_unique), negatives=int((dpred < 0).sum()), nonfinite=int((~np.isfinite(dpred)).sum()),
                    share_above_3600=round(float((dpred > 3600).mean()), 6), pred_mean=round(float(dpred.mean()), 3),
                    pred_problems=probs(root / info["test"]) if probs else None)
        rep["finals"]["default"] = info
        for e in [x for x in a.finals.split(",") if x]:
            name, mp = e.split("=", 1)
            VF, vi = test_of(root, mp)
            vp = VF["pred"].to_numpy(dtype=float)
            vi["row_order_equal_default"] = bool(np.array_equal(VF["MVT_ID_mvt"].to_numpy(), ids) and VF["MVT_ID_mvt"].is_unique)
            if not vi["row_order_equal_default"]:
                fails.append(f"final {name}: row set or order differs from the default final")
            else:
                vi["janjul_equal_default"] = all(same(VF[c].to_numpy()[jj], DF[c].to_numpy()[jj]) for c in DF.columns if c in VF.columns)
                vi["febjun_changed_rows"] = int((vp[~jj] != dpred[~jj]).sum())
                if not vi["janjul_equal_default"]:
                    fails.append(f"final {name}: Jan/Jul rows differ from the default final")
                if name == "D":
                    vi["equals_default"] = all(same(VF[c].to_numpy(), DF[c].to_numpy()) for c in DF.columns) and list(VF.columns) == list(DF.columns)
                    if not vi["equals_default"]:
                        fails.append("final D (null merge) differs from the default final")
            vi.update(negatives=int((vp < 0).sum()), nonfinite=int((~np.isfinite(vp)).sum()), share_above_3600=round(float((vp > 3600).mean()), 6),
                      pred_mean=round(float(vp.mean()), 3), pred_problems=probs(root / vi["test"]) if probs else None)
            if vi["negatives"] or vi["nonfinite"] or vi["pred_problems"]:
                fails.append(f"final {name}: negatives {vi['negatives']}, non-finite {vi['nonfinite']}, problems {vi['pred_problems']}")
            rep["finals"][name] = vi
            print(f"(i, iii, iv, v) {name}: {json.dumps(vi, ensure_ascii=False)[:600]}")
    rep["fails"] = fails
    rep["pass"] = not fails
    rep["seconds"] = round(time.time() - t0, 1)
    out = root / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"SPLITCHECK pass {rep['pass']}: {fails[:6]} -> {out}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
