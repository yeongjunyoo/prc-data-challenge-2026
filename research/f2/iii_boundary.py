# SPDX-License-Identifier: GPL-3.0-only
"""F2: check (iii) restated at the month boundary (F3 10-10 03:0x, for the coordinator's decision). Read only.
src/aiu_shift.py --group block groups the frame's rows by airport and the UTC date of the estimated off-block time MVT - mn (mn = mu + corrector
pred - base pred for non-G rows), over the whole four-month test frame. So a Jan 31 group also holds early Feb 1 departures, and a Jun 30 group
also holds early Jul 1 departures. The group's c is solved from all its rows, and every shiftable row moves by lam (1 - q) c. Per chain, variant
final against the default final:
  (a) every stage before db (corrector stage included): Jan/Jul rows bit-equal;
  (b) db: every Jan/Jul row that differs is a boundary row (non-G, in one of the ten airports' groups, and its db group also holds a non-G
      Feb/Jun row in the variant's or the default's grouping); within each group the difference divided by lam (1 - q) is one number
      (rows at the LOBT window edge or at 0 left out);
  (c) every later stage (aw, gp, gl, gja): the differing Jan/Jul rows are a subset of db's;
  (d) the final (chainavg3): the differing Jan/Jul rows are a subset of the union of the three chains' db sets.
    python -X utf8 -u research/f2/iii_boundary.py --default preds/manifest_<spec>.json --finals Ba=preds/manifest_<spec>.json[,Bb=...] [--root .]
Writes research/f2/out/iii_boundary.json (or --out).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HALF = 3606.0
APS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
STAGES = ["gja", "gl", "gp", "aw", "db", "xncr", "gkcx", "gkc", "fh", "am"]


def load(root: Path, rel: str) -> dict:
    return json.loads((root / rel).read_text(encoding="utf-8"))


def walk(root: Path, mp: str | None) -> list[tuple[str, dict]]:
    out = []
    while mp and (root / mp).exists() and len(out) < 40:
        m = load(root, mp)
        out.append((mp, m))
        mp = m.get("source", {}).get("manifest")
    return out


def label(m: dict) -> str:
    if "aiu_shift" in m:
        return "db"
    s = m.get("spec", "")
    return next((x for x in STAGES if s.endswith(x)), "corr")


def preds(root: Path, m: dict) -> tuple[np.ndarray, np.ndarray]:
    t = pq.read_table(root / m["test"]["path"], columns=["MVT_ID_mvt", "pred"])
    return t.column("MVT_ID_mvt").to_numpy(), np.ascontiguousarray(np.asarray(t.column("pred").to_numpy(zero_copy_only=False), dtype=np.float64))


def differ(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.view(np.int64) != b.view(np.int64)


def db_keys(root: Path, m: dict, ids: np.ndarray, fx: pd.DataFrame, cx: pd.DataFrame) -> dict:
    """aiu_shift's own block key per row (block_days with by="mn"), for non-G rows; G rows get "G" (their q and mn come from the G model and
    are the same in both finals, so they cannot move a group's c)."""
    a = m["aiu_shift"]
    cf = f"cache/full_{a['corrector']}_s0.parquet"
    bf = next(k for k in a["inputs_sha16"] if k.startswith(f"cache/full_{a['base']}_s"))
    C = pd.read_parquet(root / cf, columns=["MVT_ID_mvt", "pred", "mu", "p_art", "g"]).set_index("MVT_ID_mvt")
    B = pd.read_parquet(root / bf, columns=["MVT_ID_mvt", "pred"]).set_index("MVT_ID_mvt")["pred"]
    frame_ids = C.index.to_numpy()
    days = set(cx["day"].reindex(frame_ids).to_numpy())
    C = C.reindex(ids)
    mn = C["mu"].to_numpy(dtype=float) + (C["pred"].to_numpy(dtype=float) - B.reindex(ids).to_numpy(dtype=float))
    tt = cx["t"].reindex(ids).to_numpy(dtype=float)
    est = pd.to_datetime(tt - np.nan_to_num(mn, nan=0.0), unit="s").strftime("%Y-%m-%d").to_numpy()
    est = np.where(pd.Series(est).isin(days).to_numpy(), est, "out")
    ap = fx["ap"].reindex(ids).to_numpy()
    g = C["g"].to_numpy().astype(bool)
    key = np.where(g, "G", np.where((est == "out") | ~np.isin(ap, APS), "none", pd.Series(ap).astype(str).to_numpy() + "|" + est))
    return {"key": key, "q": C["p_art"].to_numpy(dtype=float), "lam": float(m["config"].get("--lam", "0.5")), "files": [cf, bf]}


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("--default", required=True)
    pa.add_argument("--finals", required=True)
    pa.add_argument("--root", default=".")
    pa.add_argument("--jj", default="research/f2/janjul_ids.parquet")
    pa.add_argument("--out", default="research/f2/out/iii_boundary.json")
    a = pa.parse_args()
    t0 = time.time()
    root = Path(a.root).resolve()
    jj_ids = pd.read_parquet(root / a.jj).iloc[:, 0].to_numpy()
    D0 = load(root, a.default)
    ids0, _ = preds(root, D0)
    fx = pq.read_table(root / "cache/feats_v25.parquet", columns=["MVT_ID_mvt", "src", "ap", "d_lobt"], read_dictionary=["ap"]).to_pandas()
    fx = fx[fx["src"].isin([0, 1, 7]).to_numpy() & np.isin(fx["MVT_ID_mvt"].to_numpy(), ids0)].set_index("MVT_ID_mvt")  # as src/aiu_shift.py inputs()
    fx["ap"] = fx["ap"].astype(str)
    _CX["ids"] = ids0
    rep = {"what": __doc__.split("\n")[0], "default": a.default, "finals": {}}
    fails = []
    for e in [x for x in a.finals.split(",") if x]:
        name, mp = e.split("=", 1)
        V0 = load(root, mp)
        r = {"manifest": mp, "chains": {}}
        ids, fd = preds(root, D0)
        idv, fv = preds(root, V0)
        assert np.array_equal(ids, idv), "final row order differs (check (i))"
        jj = np.isin(ids, jj_ids)
        dfin = differ(fv, fd) & jj
        r["final"] = {"janjul_rows": int(jj.sum()), "janjul_differ": int(dfin.sum()), "febjun_differ": int((differ(fv, fd) & ~jj).sum()),
                      "janjul_differ_rint": int((np.rint(fv) != np.rint(fd))[jj].sum()),
                      "janjul_max_abs": float(np.abs(fv - fd)[dfin].max()) if dfin.any() else 0.0,
                      "janjul_sq_sum": float(((fv - fd)[dfin] ** 2).sum()), "all_rows": int(len(ids))}
        db_union = np.zeros(len(ids), bool)
        for lab_c, sk in zip("ABC", ("source", "source_b", "source_c")):
            vc, dc = walk(root, V0[sk]["manifest"]), walk(root, D0[sk]["manifest"])
            n = min(len(vc), len(dc))
            rc = {"stages": [], "depth": [len(vc), len(dc)]}
            sets = {}
            for k in range(n):
                (vmp, vm), (dmp, dm) = vc[k], dc[k]
                lv, ld = label(vm), label(dm)
                i1, p1 = preds(root, dm)
                i2, p2 = preds(root, vm)
                if not np.array_equal(i1, ids) or not np.array_equal(i2, ids):
                    o = pd.Series(p1, index=i1).reindex(ids).to_numpy()
                    p2 = pd.Series(p2, index=i2).reindex(ids).to_numpy()
                    p1 = np.ascontiguousarray(o, dtype=np.float64)
                    p2 = np.ascontiguousarray(p2, dtype=np.float64)
                dd = differ(p2, p1) & jj
                sets[k] = dd
                rc["stages"].append({"stage": lv, "stage_default": ld, "spec": vm.get("spec"), "default_spec": dm.get("spec"),
                                     "janjul_differ": int(dd.sum()), "febjun_differ": int((differ(p2, p1) & ~jj).sum())})
            labs = [s["stage"] for s in rc["stages"]]
            if "db" not in labs:
                fails.append(f"{name} {lab_c}: no db stage in the chain walk")
                r["chains"][lab_c] = rc
                continue
            kdb = labs.index("db")
            # (a) stages before db (deeper in the walk)
            pre = [(rc["stages"][k]["stage"], rc["stages"][k]["janjul_differ"]) for k in range(kdb + 1, n)]
            rc["a_pre_db_all_equal"] = all(c == 0 for _, c in pre)
            if not rc["a_pre_db_all_equal"]:
                fails.append(f"{name} {lab_c}: Jan/Jul rows differ before db: {[x for x in pre if x[1]]}")
            # (b) db: boundary rows and one c change per group
            ddb = sets[kdb]
            db_union |= ddb
            cxt = CX(root, dc[kdb][1])
            kv = db_keys(root, vc[kdb][1], ids, fx, cxt)
            kd = db_keys(root, dc[kdb][1], ids, fx, cxt)
            fj = ~jj
            own_same = bool(np.array_equal(kv["key"][jj], kd["key"][jj]))
            rc["b_janjul_q_equal"] = bool(np.array_equal(kv["q"][jj], kd["q"][jj], equal_nan=True))
            mem = (set(kv["key"][fj]) | set(kd["key"][fj])) - {"G", "none"}
            bnd = jj & np.isin(kd["key"], list(mem))
            _, pd_db = preds(root, dc[kdb][1])
            _, pv_db = preds(root, vc[kdb][1])
            _, pd_pre = preds(root, dc[kdb + 1][1])  # the stage before db (its input)
            _, pv_pre = preds(root, vc[kdb + 1][1])
            dl = fx["d_lobt"].reindex(ids).to_numpy(dtype=float)
            edge = np.zeros(len(ids), bool)
            for f in (pd_db, pv_db):
                edge |= (f == 0) | (np.isfinite(dl) & ((np.abs(f - (dl - HALF)) < 1e-9) | (np.abs(f - (dl + HALF)) < 1e-9)))
            groups = {}
            for gk in sorted(set(kd["key"][ddb]) | set(kd["key"][bnd])):
                ing = kd["key"] == gk
                b_rows = ing & bnd
                d_rows = ing & ddb
                use = d_rows & ~edge & (kd["q"] < 1)
                rr = (pv_db - pd_db)[use] / (kd["lam"] * (1 - kd["q"][use]))
                cc = []
                for fpre, fdb, kk in ((pd_pre, pd_db, kd), (pv_pre, pv_db, kv)):  # the group's c in each final, from its moved rows
                    mv = (kk["key"] == gk) & ~edge & (kk["q"] < 1) & (fdb != fpre)
                    cc.append(float(np.median((fdb - fpre)[mv] / (kk["lam"] * (1 - kk["q"][mv])))) if mv.any() else None)
                groups[gk] = {"boundary_rows": int(b_rows.sum()), "differ": int(d_rows.sum()), "differ_not_boundary": int((d_rows & ~bnd).sum()),
                              "c_default": cc[0], "c_variant": cc[1],
                              "dc": float(np.median(rr)) if len(rr) else None, "dc_spread": float(rr.max() - rr.min()) if len(rr) else None,
                              "at_edge": int((d_rows & edge).sum())}
            rc["b_own_key_same"] = own_same
            rc["b_boundary_rows"] = int(bnd.sum())
            rc["b_differ"] = int(ddb.sum())
            rc["b_differ_outside_boundary"] = int((ddb & ~bnd).sum())
            rc["b_groups_moved"] = {k: v for k, v in groups.items() if v["differ"]}
            rc["b_groups_boundary_unmoved"] = sorted(k for k, v in groups.items() if not v["differ"])
            rc["b_max_dc_spread"] = max([v["dc_spread"] for v in groups.values() if v["dc_spread"] is not None], default=0.0)
            rc["db_inputs"] = {"variant": kv["files"], "default": kd["files"]}
            if not own_same:
                fails.append(f"{name} {lab_c}: a Jan/Jul row's own db group differs between the finals")
            if rc["b_differ_outside_boundary"]:
                fails.append(f"{name} {lab_c}: {rc['b_differ_outside_boundary']} Jan/Jul rows differ at db outside the boundary groups")
            if rc["b_max_dc_spread"] > 1e-6:
                fails.append(f"{name} {lab_c}: within a group the change over lam (1 - q) is not one number (spread {rc['b_max_dc_spread']:.3g})")
            # (c) later stages
            post = [(rc["stages"][k]["stage"], int((sets[k] & ~ddb).sum())) for k in range(0, kdb)]
            rc["c_post_db_new_rows"] = post
            if any(c for _, c in post):
                fails.append(f"{name} {lab_c}: a stage after db changes other Jan/Jul rows: {[x for x in post if x[1]]}")
            r["chains"][lab_c] = rc
            print(f"(iii') {name} {lab_c}: pre-db equal {rc['a_pre_db_all_equal']}; db differ {rc['b_differ']} "
                  f"(outside boundary {rc['b_differ_outside_boundary']}, boundary rows {rc['b_boundary_rows']}, max spread {rc['b_max_dc_spread']:.2e}); "
                  f"moved groups {[(k, v['differ'], round(v['dc'], 5) if v['dc'] is not None else None) for k, v in rc['b_groups_moved'].items()]}; "
                  f"unmoved boundary groups (rows, c default, c variant) {[(k, groups[k]['boundary_rows'], groups[k]['c_default'] and round(groups[k]['c_default'], 3), groups[k]['c_variant'] and round(groups[k]['c_variant'], 3)) for k in rc['b_groups_boundary_unmoved']]}; "
                  f"moved c {[(k, v['c_default'] and round(v['c_default'], 4), v['c_variant'] and round(v['c_variant'], 4)) for k, v in rc['b_groups_moved'].items()]}; "
                  f"after db new rows {[x for x in post if x[1]]}", flush=True)
        r["final"]["d_outside_db_union"] = int((dfin & ~db_union).sum())
        if r["final"]["d_outside_db_union"]:
            fails.append(f"{name}: {r['final']['d_outside_db_union']} final Jan/Jul rows differ outside the chains' db sets")
        print(f"(iii') {name} final: {r['final']}", flush=True)
        rep["finals"][name] = r
    rep["fails"] = fails
    rep["pass"] = not fails
    rep["seconds"] = round(time.time() - t0, 1)
    out = root / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"IIIB pass {rep['pass']}: {fails[:6]} -> {out}", flush=True)
    return 0 if rep["pass"] else 1


_CX: dict = {}


def CX(root: Path, m: dict) -> pd.DataFrame:
    ctx = m["config"].get("--ctx", "ctx_v3")
    if ctx not in _CX:
        c = pq.read_table(root / "cache" / f"{ctx}.parquet", columns=["MVT_ID_mvt", "day", "t"], read_dictionary=["day"]).to_pandas()
        c = c[np.isin(c["MVT_ID_mvt"].to_numpy(), _CX["ids"])].set_index("MVT_ID_mvt")  # the test frame's rows (the final's ids)
        c["day"] = c["day"].astype(str)
        _CX[ctx] = c
    return _CX[ctx]


if __name__ == "__main__":
    sys.exit(main())
