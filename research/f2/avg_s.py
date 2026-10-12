# SPDX-License-Identifier: GPL-3.0-only
"""F2 Package S averager (research/f2/s_spec.py): explicit seed lists and a seed-0 donor. The rule is 514's research/r10_b/s20_avg_legs.py
(sha16 94154b858cc18e71) as R13-F3's s20_avg_legs3.py, R17-A's avg_r17a.py and avg_l4.py and F3's avg_legs_final.py ran it:
  mode cb  (CatBoost legs): mu_cat on all rows;  mode rm (RealMLP legs): mu_rmlp on all rows;
  mode m3g (a model3 G-band leg): p_art, mu, pred, pred_noG on non-G rows from the model3 leg's seed files; G rows stay seed 0's.
Rows, order and every other column are the donor's (the leg's seed-0 file; for m3g the G-band leg's seed-0 file, whose non-G rows must equal
the model3 seed-0 file's to 1e-9, as s20_avg_legs3.py checks). The averaged column is np.mean(vals, axis=0) over the listed seeds in the
listed order, the donor's own column standing for seed 0 when 0 is listed (bit-identical to the scripts above for lists that start at 0).
New here: the list need not start at 0 (chain C's own RealMLP seeds 4-7 replace the donor's mu_rmlp entirely); settings include fj; every
seed file must hold exactly the donor's ids (none missing, none extra) and, for val, fwd and fj, the donor's early-stopping ids (the new
tag's es file is the donor's, copied). Refuses existing outputs. --verify recomputes in memory and compares with the existing outputs bit
for bit (columns, dtypes, order, es file), writes nothing, exits 1 on any difference.
    python -X utf8 research/f2/avg_s.py <cb|rm|m3g> <donor tag> <seed-file tag> <new tag> <seeds, e.g. 0,1,2> <settings, e.g. val,fwd> [--verify]
(cwd = the root: cache/, preds/ and research/f2/out/ are relative to it.)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

COLS = {"cb": ["mu_cat"], "rm": ["mu_rmlp"], "m3g": ["p_art", "mu", "pred", "pred_noG"]}


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def compute(root: Path, mode: str, donor: str, sfile: str, seeds: list[int], st: str) -> tuple[pd.DataFrame, dict]:
    """The averaged frame and a record; raises SystemExit on any rule break."""
    C, P = root / "cache", root / "preds"
    if mode not in COLS:
        raise SystemExit(f"mode must be cb, rm or m3g, not {mode}")
    if not seeds or seeds != sorted(set(seeds)):
        raise SystemExit(f"seeds must be ascending and distinct: {seeds}")
    if mode == "m3g" and seeds[0] != 0:
        raise SystemExit("m3g averages onto the seed-0 G-band leg: the list must start at 0")
    dp = C / f"{st}_{donor}_s0.parquet"
    b0 = pd.read_parquet(dp)
    ids = b0["MVT_ID_mvt"].to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise SystemExit(f"{dp.name}: duplicate ids")
    g = b0["g"].to_numpy().astype(bool) if mode == "m3g" else np.zeros(len(b0), bool)
    rec = {"setting": st, "mode": mode, "donor": f"cache/{dp.name}", "donor_sha16": sha16(dp), "seed_files": {}, "rows": int(len(b0)),
           "g_rows_kept_seed0": int(g.sum())}
    files = {}
    for k in seeds:  # every listed seed's file is read and checked; seed 0's value itself comes from the donor (as s20_avg_legs.py)
        p = C / f"{st}_{sfile}_s{k}.parquet"
        bk = pd.read_parquet(p)
        kid = bk["MVT_ID_mvt"].to_numpy()
        if len(kid) != len(ids) or len(np.unique(kid)) != len(kid) or not np.array_equal(np.sort(kid), np.sort(ids)):
            raise SystemExit(f"{p.name}: its ids are not exactly the donor's ({len(kid)} vs {len(ids)} rows)")
        rec["seed_files"][f"s{k}"] = {"file": f"cache/{p.name}", "sha16": sha16(p), "same_order": bool(np.array_equal(kid, ids))}
        files[k] = bk.set_index("MVT_ID_mvt").reindex(ids)
    if mode == "m3g" and sfile != donor:  # s20_avg_legs3.py: the G-band leg's non-G rows must be the model3 leg's (seed 0)
        s0 = files[0]
        for c in COLS[mode]:
            if c in b0.columns and c in s0.columns:
                d = float(np.nanmax(np.abs(b0[c].to_numpy(dtype=float)[~g] - s0[c].to_numpy(dtype=float)[~g])))
                if not d <= 1e-9:
                    raise SystemExit(f"{st}: {donor} and {sfile} differ on non-G rows in {c} (max {d})")
        rec["donor_equals_seed0_on_nonG"] = True
    res = b0.copy()
    rec["columns"] = {}
    for c in COLS[mode]:
        if c not in b0.columns:
            continue
        vals = [b0[c].to_numpy(dtype=float)] if seeds[0] == 0 else []
        for k in (seeds[1:] if seeds[0] == 0 else seeds):
            v = files[k][c]
            if v.isna().any():
                raise SystemExit(f"{st} seed {k}: {c} has rows missing")
            vals.append(v.to_numpy(dtype=float))
        m = np.mean(vals, axis=0)
        res[c] = np.where(g, b0[c].to_numpy(dtype=float), m)
        rec["columns"][c] = {"seeds": seeds, "rms_change_vs_donor": round(float(np.sqrt(np.mean((res[c].to_numpy() - b0[c].to_numpy(dtype=float))[~g] ** 2))), 4)}
    if not rec["columns"]:
        raise SystemExit(f"{dp.name} has none of {COLS[mode]}")
    if st != "full":
        e0 = P / f"es_valid_ids_{donor}_{st}_s0.parquet"
        es0 = np.sort(pd.read_parquet(e0)["MVT_ID_mvt"].to_numpy())
        rec["es_donor"] = {"file": f"preds/{e0.name}", "sha16": sha16(e0), "rows": int(len(es0))}
        for k in seeds:
            ek = P / f"es_valid_ids_{sfile}_{st}_s{k}.parquet"
            if not ek.exists():
                raise SystemExit(f"missing {ek.name}")
            if not np.array_equal(np.sort(pd.read_parquet(ek)["MVT_ID_mvt"].to_numpy()), es0):
                raise SystemExit(f"{ek.name}: early-stopping ids differ from the donor's")
        rec["es_seeds_equal_donor"] = True
    return res, rec


def frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    bad = []
    if list(a.columns) != list(b.columns):
        return [f"columns {list(a.columns)} vs {list(b.columns)}"]
    if len(a) != len(b):
        return [f"rows {len(a)} vs {len(b)}"]
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype != y.dtype:
            bad.append(f"{c}: dtype {x.dtype} vs {y.dtype}")
        elif x.dtype.kind == "f":
            if not np.array_equal(x, y, equal_nan=True):
                bad.append(f"{c}: {int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum())} rows differ")
        elif not np.array_equal(x, y):
            bad.append(f"{c}: values differ")
    return bad


def main() -> int:
    a = [x for x in sys.argv[1:] if x != "--verify"]
    verify = "--verify" in sys.argv[1:]
    if len(a) != 6:
        raise SystemExit(__doc__)
    mode, donor, sfile, new, seeds, settings = a[0], a[1], a[2], a[3], [int(s) for s in a[4].split(",")], a[5].split(",")
    root = Path.cwd()
    if new in (donor, sfile) or not set(settings) <= {"val", "fwd", "full", "fj"}:
        raise SystemExit("new tag must differ from the donor and seed-file tags; settings among val, fwd, full, fj")
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    out_dir = root / "research" / "f2" / "out"
    bad_all = []
    for st in settings:
        out = root / "cache" / f"{st}_{new}_s0.parquet"
        esn = root / "preds" / f"es_valid_ids_{new}_{st}_s0.parquet"
        if not verify and (out.exists() or out.is_symlink() or (st != "full" and (esn.exists() or esn.is_symlink()))):
            raise SystemExit(f"exists, refusing: {out.name} or {esn.name}")
        res, rec = compute(root, mode, donor, sfile, seeds, st)
        if verify:
            if not out.exists():
                bad_all.append(f"{st}: {out.name} missing")
                continue
            bad = frames_equal(res, pd.read_parquet(out))
            if st != "full":
                e_new = pd.read_parquet(esn)["MVT_ID_mvt"].to_numpy() if esn.exists() else None
                e_don = pd.read_parquet(root / "preds" / f"es_valid_ids_{donor}_{st}_s0.parquet")["MVT_ID_mvt"].to_numpy()
                if e_new is None or not np.array_equal(e_new, e_don):
                    bad.append("es file is not the donor's")
            print(f"verify {st} {new}: {'bit-identical' if not bad else 'DIFFERS: ' + '; '.join(bad)}", flush=True)
            bad_all += [f"{st}: {b}" for b in bad]
            continue
        res.to_parquet(out, index=False)
        if st != "full":
            shutil.copyfile(root / "preds" / f"es_valid_ids_{donor}_{st}_s0.parquet", esn)
        rec.update({"new": f"cache/{out.name}", "new_sha16": sha16(out), "written": time.strftime("%Y-%m-%d %H:%M:%S")})
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"avg_s_{new}_{st}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"{st} {new}: {mode} seeds {seeds} donor {donor} files {sfile}; " + ", ".join(f"{c} rms change {v['rms_change_vs_donor']}" for c, v in rec["columns"].items())
              + f"; wrote {out.name} {rec['new_sha16']}", flush=True)
    if verify and bad_all:
        print("VERIFY FAILED: " + " | ".join(bad_all), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
