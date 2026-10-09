# SPDX-License-Identifier: GPL-3.0-only
"""A2: the all-row corrector of src/ctx_corr.py (imported, unchanged) with more stage-2 columns. Run it exactly like src/ctx_corr.py, with
--ctx naming a context file built by src/feats_ctx2.py (cache/ctx_v2: the ctx_v1 columns plus nine input-only ones).

Added stage-2 columns (everything else, stage 1, folds, pairwise nesting, stage-2 parameters, --scale2, the window last, is ctx_corr's):
  gate_delay_pm30, plan_ground_aobt3                    R4 lane's two state columns (vault research/2026-10-05-r4-sources.md 5.5)
  st_g1, st_g2, st_g3, st_gp, st_gp2, st_k, st_n_arr_2h R3 lane's stand occupancy (vault research/2026-10-05-r3-structure.md 2.5)
  st_g1_m_pred, st_g2_m_pred, st_g3_m_pred              st_g1..3 minus the stage-1 prediction (made inside, like ctx_corr's plan gaps)
Research-only flags (never in a candidate's command; removed before ctx_corr parses the rest):
  --drop2 <c1,c2,...>   leave these stage-2 columns out (column ablations)
  --fit_cache <dir>     keep every fitted LightGBM model (stage 1 and stage 2) in <dir> under a hash of its exact training matrix, target,
                        settings and the code of src/adsb_corr.py and src/ctx_corr.py, and load it instead of refitting when the same fit comes
                        again. A loaded model gives the same predictions as the fit it stores, so the outputs equal a run without the flag; it lets
                        a run stopped part way resume, and lets the full setting and stage-2 column ablations reuse the fits they share.
    .venv/Scripts/python.exe -u src/ctx_corr2.py --setting val --seed 0 --base m6r25sgoof4 --base_seed 0 --tag m6r25sgoof4adsbgeoctx2_50w
        --feats feats_v25 --adsb adsb_v1 --adsb_sha <16> --geo geo_v1 --geo_sha <16> --ctx ctx_v2 --ctx_sha <16> --folds 10 --rounds 300
        --rounds2 400 --max_pred 3600 --threads 4 --scale2 0.5
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import src.ctx_corr as cc  # noqa: E402

NEW_CTX = ["gate_delay_pm30", "plan_ground_aobt3", "st_g1", "st_g2", "st_g3", "st_gp", "st_gp2", "st_k", "st_n_arr_2h"]
PRED_GAP = {"st_g1_m_pred": "st_g1", "st_g2_m_pred": "st_g2", "st_g3_m_pred": "st_g3"}


class Rows2(cc.Rows):
    def design2(self, s1, delta):
        for name, src in PRED_GAP.items():
            self.cols[name] = self.cols[src] - s1
        return super().design2(s1, delta)


def pop_flag(flag: str) -> str | None:
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    val = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return val


def digest(*parts) -> str:
    h = hashlib.blake2b(digest_size=16)
    for x in parts:
        if isinstance(x, np.ndarray):
            a = np.ascontiguousarray(x)
            h.update(str((a.dtype.str, a.shape)).encode())
            h.update(a.tobytes())
        else:
            h.update(json.dumps(x, sort_keys=True, default=str).encode())
    return h.hexdigest()


def fit_cache(d: Path) -> None:
    """Route ac.fit (stage 1) and cc.fit2 (stage 2) through a model store in d (see --fit_cache)."""
    import lightgbm as lgb
    import src.adsb_corr as ac
    d.mkdir(parents=True, exist_ok=True)
    code = [hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16] for f in ("src/adsb_corr.py", "src/ctx_corr.py")]
    fit1, fit2 = ac.fit, cc.fit2
    stats = {"hit": 0, "fit": 0}

    def store(kind, key, make):
        p = d / f"{kind}_{key}.txt"
        if p.exists():
            stats["hit"] += 1
            return lgb.Booster(model_str=p.read_text(encoding="utf-8"))
        m = make()
        tmp = p.with_suffix(".tmp")
        tmp.write_text(m.model_to_string(), encoding="utf-8")
        tmp.replace(p)
        stats["fit"] += 1
        return m

    def fit1_c(X, seed, rounds, threads):
        cols = ac.BASE_COLS + ac.ADSB_COLS
        key = digest(code, cols, seed, rounds, threads, X[cols].astype(float).to_numpy(), X["y"].to_numpy(float), X["pred"].to_numpy(float))
        return store("s1", key, lambda: fit1(X, seed, rounds, threads))

    def fit2_c(M, target, seed, rounds, threads):
        key = digest(code, cc.X2, cc.CAT2, seed, rounds, threads, np.asarray(M, dtype=float), np.asarray(target, dtype=float))
        return store("s2", key, lambda: fit2(M, target, seed, rounds, threads))

    ac.fit, cc.fit2 = fit1_c, fit2_c
    import atexit
    atexit.register(lambda: print(f"  fit cache {d}: {stats['hit']} loaded, {stats['fit']} fitted", flush=True))


def main():
    drop = pop_flag("--drop2")
    cache = pop_flag("--fit_cache")
    cc.CTX_COLS = cc.CTX_COLS + NEW_CTX
    x2 = cc.X2 + NEW_CTX + list(PRED_GAP)
    if drop:
        gone = set(drop.split(","))
        unknown = gone - set(x2)
        if unknown:
            raise SystemExit(f"--drop2: not stage-2 columns: {sorted(unknown)}")
        x2 = [c for c in x2 if c not in gone]
        print(f"  stage-2 columns without {sorted(gone)}: {len(x2)}", flush=True)
    cc.X2 = x2
    cc.Rows = Rows2
    if cache:
        fit_cache(Path(cache))
    cc.main()


if __name__ == "__main__":
    main()
