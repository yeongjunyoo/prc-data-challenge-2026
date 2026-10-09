"""F2 four-month corrector (vault research/2026-10-08-f2-four-months.md 1.4 and 5.7): src/ctx_corr3.py (imported, unchanged, with src/ctx_corr.py
and src/ctx_corr2.py) whose stage 2 also trains on the rows of an extra base file: --extra_base <tag> reads cache/fj_<tag>_s0.parquet, the
leave-Feb/Jun-out blend of the same chain (rows of February and June 2025), sha16-checked with --extra_sha. main() below is src/ctx_corr.py's main
with the extra rows added to stage-2 training; every helper is ctx_corr's (imported). With --extra_base none it must reproduce ctx_corr3 byte for
byte (checked in 5.7 against the stored corrector caches).
  val : S_k trains on the other folds' rows (pairwise-nested stage-1 values, as ctx_corr) plus every extra row, and is applied to fold k
        (gate14's U_V).
  full: the mean of those ten S_k on the base rows.
  fwd : ctx_corr's January-only stage 2 plus every extra row (February and June precede July). The extra rows' bases come from legs trained on
        ten months that include July 2025, so this ruler has an indirect July channel; information only.
Extra rows: stage-1 value = mean of the ten stage-1 fold models on their matched rows (fwd: the January model), 0 when unmatched (adsb_v1nd has no
Feb/Jun 2025 day); a row enters stage 2 when it has y and its stage-1 prediction is below --max_pred (as gate14). Their context rows come from
--ctx, which must hold them (research/f2/ctx4_table.py). The extra rows are appended after the base rows, in their file order (as gate14).
    python -u research/f2/ctx_corr4.py <src/ctx_corr3.py arguments> --extra_base <tag>|none [--extra_sha <16>] [--fit_cache <dir>]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import src.adsb_corr as ac  # noqa: E402
import src.ctx_corr as cc  # noqa: E402
import src.ctx_corr2 as c2  # noqa: E402
import src.ctx_corr3 as c3  # noqa: E402
from src.common import breakdown, rmse  # noqa: E402
from src.load import CACHE  # noqa: E402
from src.lobt_window import feats_d_lobt, mu_of, project  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def extra_rows(RE, models, a):
    """(stage-2 matrix, target) of the extra rows that enter stage 2; empty when there are none."""
    if RE is None:
        return np.zeros((0, len(cc.X2))), np.zeros(0)
    d1 = np.zeros(RE.n)
    if len(RE.X):
        d1[RE.X["pos"].to_numpy()] = np.mean([ac.predict(m, RE.X) for m in models], axis=0)
    s1 = np.clip(RE.base["pred"] + d1, 0, None)
    y = RE.b["y"].to_numpy(dtype=float)
    ok = np.isfinite(y) & (s1 < a.max_pred)
    return RE.design2(s1, d1)[ok], (y - s1)[ok]


def stage2_val_models4(R, T, fT, fo, nested, a, keep, ME, tE):
    """cc.stage2_val_models with the extra rows (ME, tE) appended to every S_k's training rows."""
    y = R.b["y"].to_numpy(dtype=float)
    fold_row = pd.Series(R.day).map(fo).to_numpy()
    assert not np.isnan(fold_row.astype(float)).any(), "a val day has no fold"
    T_pos = T["pos"].to_numpy()
    assert np.array_equal(fold_row[T_pos].astype(float), fT.astype(float)), "stage-1 and row folds disagree"
    S = []
    for k in range(a.folds):
        dm = np.zeros(len(T))
        for j in range(a.folds):
            if j != k:
                dm[fT == j] = nested[(j, k)]
        delta = R.stage1(dm, T_pos)
        s1 = np.clip(R.base["pred"] + delta, 0, None)
        tr = (fold_row != k) & keep & (s1 < a.max_pred)
        M = R.design2(s1, delta)
        S.append(cc.fit2(np.vstack([M[tr], ME]), np.concatenate([(y - s1)[tr], tE]), a.seed, a.rounds2, a.threads))
        del M
        print(f"  stage-2 fold {k}: {int(tr.sum()):,} rows + {len(tE):,} extra rows", flush=True)
    return S, fold_row


def run(extra: str, extra_sha: str | None) -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", default="val", choices=["val", "fwd", "full"])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--base", required=True)
    pa.add_argument("--base_seed", type=int, default=0)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--feats", default="feats_v25")
    pa.add_argument("--adsb", required=True)
    pa.add_argument("--adsb_sha", required=True)
    pa.add_argument("--geo", default="none")
    pa.add_argument("--geo_sha", default="none")
    pa.add_argument("--ctx", required=True)
    pa.add_argument("--ctx_sha", required=True)
    pa.add_argument("--folds", type=int, default=10)
    pa.add_argument("--rounds", type=int, default=300)
    pa.add_argument("--rounds2", type=int, default=400)
    pa.add_argument("--max_pred", type=float, default=3600.0)
    pa.add_argument("--threads", type=int, default=3)
    pa.add_argument("--scale2", type=float, default=1.0)
    a = pa.parse_args()
    ac.below_normal()
    t0 = time.time()
    A = cc.stage1_inputs(a)
    ctx = cc.checked(a.ctx, a.ctx_sha).set_index("MVT_ID_mvt")
    val = pd.read_parquet(CACHE / f"val_{a.base}_s{a.base_seed}.parquet")
    T = ac.design(val, A)
    T = T[np.isfinite(T["y"].to_numpy())]
    b = val if a.setting == "val" else pd.read_parquet(CACHE / f"{a.setting}_{a.base}_s{a.base_seed}.parquet")
    RV = cc.Rows(val, A, a, ctx)
    assert set(np.unique(RV.day)) == set(T["day"].unique()), "val days with no matched row: the fold map would differ from adsb_corr"
    RB = RV if a.setting == "val" else cc.Rows(b, A, a, ctx)
    X = RB.X
    RE = None
    if extra != "none":
        pe = CACHE / f"fj_{extra}_s0.parquet"
        got = cc.sha16(pe)
        if extra_sha is None or got != extra_sha:
            raise SystemExit(f"{pe.name} sha {got} != --extra_sha {extra_sha}")
        E = pd.read_parquet(pe)
        clash = np.intersect1d(E["MVT_ID_mvt"].to_numpy(), val["MVT_ID_mvt"].to_numpy())
        if len(clash):
            raise SystemExit(f"{len(clash)} extra rows are also val rows")
        RE = cc.Rows(E, A, a, ctx)
    keep_v = np.isfinite(val["y"].to_numpy(dtype=float))
    delta1 = np.zeros(len(b))
    delta2 = np.zeros(len(b))
    if a.setting == "val":
        fo, fT, models = cc.stage1_folds(T, a)
        oof = np.zeros(len(T))
        for j, m in enumerate(models):
            oof[fT == j] = ac.predict(m, T[fT == j])
        delta1[T["pos"].to_numpy()] = oof
        note = f"{a.folds} day folds over {T['day'].nunique()} days, {len(T)} matched rows"
        nested = cc.nested_stage1(T, fT, a)
        ME, tE = extra_rows(RE, models, a)
        S, fold_row = stage2_val_models4(RV, T, fT, fo, nested, a, keep_v, ME, tE)
        s1 = np.clip(RV.base["pred"] + delta1, 0, None)
        M = RV.design2(s1, delta1)
        app = s1 < a.max_pred
        for k, m in enumerate(S):
            sel = (fold_row == k) & app
            delta2[sel] = m.predict(M[sel])
        imp = np.mean([m.feature_importance("gain") for m in S], axis=0)
        note += f"; stage 2 on all rows below {a.max_pred:.0f} s, pairwise-nested stage-1 inputs, plus {len(tE):,} extra rows"
    elif a.setting == "fwd":
        jan_T = (T["day"].str[5:7] == "01").to_numpy()
        mdl = ac.fit(T[jan_T], a.seed, a.rounds, a.threads)
        delta1[X["pos"].to_numpy()] = ac.predict(mdl, X)
        note = f"January only: {T[jan_T]['day'].nunique()} days, {int(jan_T.sum())} matched rows"
        fo = ac.day_folds(T["day"].unique(), a.folds, a.seed)
        fT = T["day"].map(fo).to_numpy()
        dm = np.zeros(len(T))
        for j in range(a.folds):
            inj = jan_T & (fT == j)
            if inj.any():
                m = ac.fit(T[jan_T & (fT != j)], a.seed, a.rounds, a.threads)
                dm[inj] = ac.predict(m, T[inj])
        dv = RV.stage1(dm, T["pos"].to_numpy())
        s1v = np.clip(RV.base["pred"] + dv, 0, None)
        jan = (pd.Series(RV.day).str[5:7] == "01").to_numpy()
        tr = jan & keep_v & (s1v < a.max_pred)
        ME, tE = extra_rows(RE, [mdl], a)
        S = cc.fit2(np.vstack([RV.design2(s1v, dv)[tr], ME]), np.concatenate([(val["y"].to_numpy(dtype=float) - s1v)[tr], tE]),
                    a.seed, a.rounds2, a.threads)
        s1 = np.clip(RB.base["pred"] + delta1, 0, None)
        app = s1 < a.max_pred
        delta2[app] = S.predict(RB.design2(s1, delta1)[app])
        imp = S.feature_importance("gain")
        note += f"; stage 2 on {int(tr.sum()):,} January rows (January-only nested stage-1 inputs) plus {len(tE):,} extra rows"
    else:
        fo, fT, models = cc.stage1_folds(T, a)
        delta1[X["pos"].to_numpy()] = np.mean([ac.predict(m, X) for m in models], axis=0)
        note = f"mean of {a.folds} day-fold correctors over {T['day'].nunique()} days, {len(T)} matched rows"
        nested = cc.nested_stage1(T, fT, a)
        ME, tE = extra_rows(RE, models, a)
        S, _ = stage2_val_models4(RV, T, fT, fo, nested, a, keep_v, ME, tE)
        s1 = np.clip(RB.base["pred"] + delta1, 0, None)
        M = RB.design2(s1, delta1)
        app = s1 < a.max_pred
        delta2[app] = np.mean([m.predict(M[app]) for m in S], axis=0)
        imp = np.mean([m.feature_importance("gain") for m in S], axis=0)
        note += f"; stage 2 = mean of {a.folds} fold models on all rows below {a.max_pred:.0f} s, plus {len(tE):,} extra rows"
    base_pred = b["pred"].to_numpy(dtype=float)
    s1 = np.clip(base_pred + delta1, 0, None)
    pd.DataFrame({"MVT_ID_mvt": b["MVT_ID_mvt"].to_numpy(), "s1": s1, "d2": delta2}).to_parquet(
        CACHE.parent / "notes" / f"ctx_corr_{a.tag}_{a.setting}_s{a.seed}_delta.parquet", index=False)
    delta2 = a.scale2 * delta2
    s2 = np.clip(s1 + delta2, 0, None)
    which, mu = mu_of(b)
    p_art, g, dsch = b["p_art"].to_numpy(dtype=float), b["g"].to_numpy().astype(bool), b["dsch"].to_numpy(dtype=float)
    d = feats_d_lobt(a.feats, b["MVT_ID_mvt"].to_numpy())
    pred_w, p2, _ = project(s2, mu, p_art, g, dsch, d)
    pred_s1w, _, _ = project(s1, mu, p_art, g, dsch, d)
    pred_bw, _, _ = project(base_pred, mu, p_art, g, dsch, d)
    assert np.isfinite(pred_w).all() and (pred_w >= 0).all()
    out = b.copy()
    out["pred"], out["p_art"] = pred_w, p2
    print(f"[{a.tag} {a.setting} seed{a.seed}] base {a.base}_s{a.base_seed} (mu = {which}), extra {extra}, ADS-B {a.adsb}({a.adsb_sha}) geo {a.geo}, "
          f"ctx {a.ctx}, matched {len(X):,} / {len(b):,}, stage-2 rows {int((delta2 != 0).sum()):,}, {note}, {time.time() - t0:.0f} s")
    rep = {"tag": a.tag, "setting": a.setting, "seed": a.seed, "base": a.base, "extra_base": extra, "extra_sha": extra_sha, "adsb": a.adsb,
           "adsb_sha": a.adsb_sha, "geo": a.geo, "ctx": a.ctx, "scale2": a.scale2, "note": note, "rows": int(len(b)), "matched": int(len(X)),
           "stage2_rows": int((delta2 != 0).sum()), "seconds": round(time.time() - t0)}
    sh = pd.Series(imp, index=cc.X2) / np.sum(imp)
    rep["stage2_gain_share"] = {k: round(float(v), 4) for k, v in sh.sort_values(ascending=False).head(20).items()}
    if "y" in b.columns:
        y = b["y"].to_numpy(dtype=float)
        ap = pd.read_parquet(CACHE / f"{a.feats}.parquet", columns=["MVT_ID_mvt", "ap"]).set_index("MVT_ID_mvt")["ap"].reindex(
            b["MVT_ID_mvt"].to_numpy()).astype(str).to_numpy()
        print(f"  base+window {rmse(pred_bw, y):.3f} -> stage1+window {rmse(pred_s1w, y):.3f} -> stage1+stage2+window {rmse(pred_w, y):.3f}")
        print(breakdown(pred_w, y, ap, f"{a.tag} {a.setting}"))
        se_w, se_1 = (pred_w - y) ** 2, (pred_s1w - y) ** 2
        rep.update(rmse_base_window=round(rmse(pred_bw, y), 3), rmse_stage1_window=round(rmse(pred_s1w, y), 3), rmse=round(rmse(pred_w, y), 3),
                   dmse_stage2=round(float(se_w.mean() - se_1.mean()), 1))
    (CACHE.parent / "notes" / f"ctx_corr_{a.tag}_{a.setting}_s{a.seed}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    dst = CACHE / f"{a.setting}_{a.tag}_s{a.seed}.parquet"
    out.to_parquet(dst, index=False)
    if a.setting != "full":
        P = CACHE.parent / "preds"
        shutil.copyfile(P / f"es_valid_ids_{a.base}_{a.setting}_s{a.base_seed}.parquet", P / f"es_valid_ids_{a.tag}_{a.setting}_s{a.seed}.parquet")
    print(f"saved {dst.name}")


def main():
    extra = c2.pop_flag("--extra_base") or "none"
    extra_sha = c2.pop_flag("--extra_sha")
    drop = c2.pop_flag("--drop2")
    cache = c2.pop_flag("--fit_cache")
    if c2.pop_flag("--aiu_date") not in (None, "mvt"):
        raise SystemExit("--aiu_date block is not supported here")
    # src/ctx_corr3.main()'s configuration, verbatim (--aiu_date mvt)
    cc.CTX_COLS = cc.CTX_COLS + c2.NEW_CTX + c3.AIU_CTX
    x2 = cc.X2 + c2.NEW_CTX + list(c2.PRED_GAP) + c3.AIU_CTX + c3.AIU_GAP
    if drop:
        gone = set(drop.split(","))
        unknown = gone - set(x2)
        if unknown:
            raise SystemExit(f"--drop2: not stage-2 columns: {sorted(unknown)}")
        x2 = [c for c in x2 if c not in gone]
        print(f"  stage-2 columns without {sorted(gone)}: {len(x2)}", flush=True)
    cc.X2 = x2
    cc.Rows = c3.Rows3
    if cache:
        c2.fit_cache(Path(cache))
    run(extra, extra_sha)


if __name__ == "__main__":
    main()
