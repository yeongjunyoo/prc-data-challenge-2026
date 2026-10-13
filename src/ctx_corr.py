# SPDX-License-Identifier: GPL-3.0-only
"""All-row second-stage corrector on top of the ADS-B corrector, then the LOBT window (src/lobt_window.project) last.

Stage 1 is src/adsb_corr.py unchanged (imported): matched rows, its columns, day folds, seed, rounds and parameters. With --geo, the six
  stand-geometry columns of the observation lane (research/obs2/corr_plus.py geo_cols("filled"), frozen into cache/<geo>.parquet by
  research/ctx_corr/freeze_geo.py) are added to its ADS-B columns the way research/obs2/geo_corr.py does (the 321 line).
Stage 2 (this file): LightGBM on y - stage1 (clipped +-1,800 s) over every row whose stage-1 prediction is below --max_pred (ADS-B
  matched or not, NM matched or not), with base columns, stage-1 flags and the context columns of src/feats_ctx.py plus columns that need
  the stage-1 prediction (each plan time's gap to it, departures on the ground at the estimated off-block). Rows at or above --max_pred,
  and every row when --ctx is none, keep the stage-1 value.
  The idea of an all-row second stage with context columns is credited to the public team outgoing-boat (README 「외부 아이디어」);
  the code and the column definitions are ours.
Out-of-sample discipline: a stage-2 model never sees a stage-1 value that used labels of the days it is applied to.
  val : day folds (adsb_corr.day_folds with --seed). Fold k rows get the plain stage-1 out-of-fold value (the 291 / 321 value). Stage-2
        model S_k trains on the other folds' rows; their stage-1 values come from stage-1 models fitted without their own fold and
        without fold k (pairwise nested, 45 extra stage-1 fits).
  fwd : stage 1 = one model on the January ruler days applied to the fwd base (as adsb_corr). Stage 2 trains on January rows only, whose
        stage-1 values come from January-only day folds: no July label is used.
  full: stage 1 = mean of the 10 fold models (as adsb_corr); stage 2 = mean of the 10 S_k.
--scale2 multiplies the stage-2 correction before the window (as adsb_corr --scale does for stage 1).
Output: cache/<setting>_<tag>_s<seed>.parquet (columns of the base, pred and p_art replaced), notes/ctx_corr_<tag>_<setting>_s<seed>.json,
  notes/ctx_corr_<tag>_<setting>_s<seed>_delta.parquet (stage-1 prediction and the unscaled stage-2 change per row).
    .venv/Scripts/python.exe -u src/ctx_corr.py --setting val --seed 0 --base m6r25sgcbxr --tag m6r25sgcbxradsbctxw --base_seed 0
        --feats feats_v25 --adsb adsb_v1 --adsb_sha <16> --geo none --geo_sha none --ctx ctx_v1 --ctx_sha <16> --folds 10 --rounds 300
        --rounds2 400 --max_pred 3600 --threads 4 --scale2 0.5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import src.adsb_corr as ac  # noqa: E402
from src.common import breakdown, rmse  # noqa: E402
from src.load import CACHE  # noqa: E402
from src.lobt_window import feats_d_lobt, mu_of, project  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
APTS = ac.APTS
GEO_NEW = ["d_stand_first", "st_known", "d_stand_stop", "d_any_stand", "seg_code", "to_gap"]
CTX_COLS = ["arr_rwy_ti_m15", "arr_rwy_n_m15", "arr_rwy_ti_m60", "arr_ap_ti_m15", "nb_rwy_daobt_m60", "nb_rwy_daobt_p30", "own_m_nb_rwy",
            "own_m_nb_ap", "prev_st_arr_ti", "prev_st_arr_age", "ret_rwy", "ret_ap", "delay_ap_m15", "rwy_share_pm30", "n_dep_rwys_pm30",
            "n_arr_rwys_pm30", "stand_p_c"]
F_COLS = ["nm", "d_sch", "d_aobt", "d_eobt", "d_iobt", "d_lobt", "hour_mvt", "ap"]
PLAN = ["d_sch", "d_lobt", "d_iobt", "d_eobt", "d_aobt"]
X2 = (["pred", "mu", "p_art", "dsch", "g", "apt_code", "hour_mvt", "nm", "d_aobt", "d_eobt", "d_lobt", "d_iobt", "s1_delta", "s1_matched", "fm_ok"]
      + CTX_COLS + ["g_sch", "g_lo", "g_io", "g_eo", "g_ao", "g_min_abs", "n_plans", "n_plans_over_1h", "dep_ground_at_est"])
CAT2 = ["apt_code", "stand_p_c"]
CLIP = 1800.0


def sha16(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def checked(name: str, want: str) -> pd.DataFrame:
    """Table cache/<name>.parquet after checking that its SHA-256 prefix equals want (stops otherwise)."""
    p = CACHE / f"{name}.parquet"
    got = sha16(p)
    if got != want:
        raise SystemExit(f"{p.name} sha {got} != {want}")
    return pd.read_parquet(p)


def stage1_inputs(a) -> pd.DataFrame:
    """ADS-B table of stage 1, with the stand-geometry columns joined when --geo is not none."""
    A = ac.load_adsb(a.adsb, a.adsb_sha)
    if a.geo != "none":
        G = checked(a.geo, a.geo_sha).set_index("MVT_ID_mvt")
        ac.ADSB_COLS = list(ac.ADSB_COLS) + GEO_NEW
        A = A.join(G[GEO_NEW], how="left")
    return A


class Rows:
    """Every row of one setting: base columns, feature-version columns, context columns, and the stage-1 frame of its matched rows."""

    def __init__(self, b: pd.DataFrame, A: pd.DataFrame, a, ctx: pd.DataFrame):
        self.b = b
        ids = b["MVT_ID_mvt"].to_numpy()
        self.n = len(b)
        F = pd.read_parquet(CACHE / f"{a.feats}.parquet", columns=["MVT_ID_mvt"] + F_COLS).set_index("MVT_ID_mvt").reindex(ids)
        C = ctx.reindex(ids)
        assert C["t"].notna().all(), "context rows missing for some base rows"
        self.base = {c: b[c].to_numpy(dtype=float) for c in ("pred", "mu", "p_art", "dsch", "g")}
        self.cols = {c: F[c].to_numpy(dtype=float) for c in F_COLS if c != "ap"}
        self.cols["apt_code"] = F["ap"].astype(str).map({k: i for i, k in enumerate(APTS)}).to_numpy(dtype=float)
        self.ap = F["ap"].astype(str).to_numpy()
        for c in CTX_COLS:
            self.cols[c] = C[c].to_numpy(dtype=float)
        self.t, self.aobt3, self.part = C["t"].to_numpy(float), C["aobt3"].to_numpy(float), C["part"].astype(str).to_numpy()
        self.day = C["day"].astype(str).to_numpy()
        self.X = ac.design(b, A)  # matched rows (pos = position in b)
        self.pos = self.X["pos"].to_numpy()
        self.matched = np.zeros(self.n)
        self.matched[self.pos] = 1.0
        self.fm_ok = np.full(self.n, np.nan)
        self.fm_ok[self.pos] = self.X["fm_ok"].to_numpy(dtype=float)
        # departures of the same airport-month, for the ground count at an estimated off-block time
        self.groups = []
        for (ap, part), idx in pd.Series(np.arange(self.n)).groupby([self.ap, self.part]):
            ii = idx.to_numpy()
            ok = np.isfinite(self.aobt3[ii]) & (self.aobt3[ii] < self.t[ii])
            self.groups.append((ii, np.sort(self.aobt3[ii][ok]), np.sort(self.t[ii][ok])))

    def stage1(self, delta_T: np.ndarray, T_pos: np.ndarray) -> np.ndarray:
        """Stage-1 correction for every row from values aligned with the stage-1 frame T (T_pos = positions of T rows here)."""
        d = np.zeros(self.n)
        d[T_pos] = delta_T
        return d

    def design2(self, s1: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Stage-2 matrix for stage-1 prediction s1 (all rows) with stage-1 correction delta."""
        c = dict(self.cols)
        c.update({k: v for k, v in self.base.items() if k != "pred"})
        c["pred"], c["s1_delta"], c["s1_matched"], c["fm_ok"] = s1, delta, self.matched, self.fm_ok
        gaps = []
        for k, nm in zip(PLAN, ("g_sch", "g_lo", "g_io", "g_eo", "g_ao")):
            c[nm] = self.cols[k] - s1
            gaps.append(c[nm])
        G = np.vstack(gaps)
        with np.errstate(invalid="ignore", all="ignore"):
            c["g_min_abs"] = np.where(np.isfinite(G).any(axis=0), np.nanmin(np.where(np.isfinite(G), np.abs(G), np.inf), axis=0), np.nan)
        c["n_plans"] = np.isfinite(G).sum(axis=0).astype(float)
        c["n_plans_over_1h"] = sum((np.nan_to_num(self.cols[k], nan=-1.0) > 3600).astype(float) for k in PLAN)
        est = self.t - s1
        dg = np.full(self.n, np.nan)
        for ii, ao_s, t_s in self.groups:
            e = est[ii]
            dg[ii] = np.searchsorted(ao_s, e, side="right") - np.searchsorted(t_s, e, side="right")
        c["dep_ground_at_est"] = dg
        return np.column_stack([np.asarray(c[k], dtype=float) for k in X2])


def fit2(M: np.ndarray, target: np.ndarray, seed: int, rounds: int, threads: int):
    """Stage-2 LightGBM on the matrix M (columns X2) and the target clipped to +-1,800 s."""
    import lightgbm as lgb
    params = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=threads, seed=seed, deterministic=True, force_col_wise=True)
    ds = lgb.Dataset(M, np.clip(target, -CLIP, CLIP), feature_name=X2, categorical_feature=CAT2, free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=rounds)


def stage1_folds(T: pd.DataFrame, a):
    """Day folds of stage 1 and one stage-1 model per fold, fitted without that fold's days."""
    fo = ac.day_folds(T["day"].unique(), a.folds, a.seed)
    fT = T["day"].map(fo).to_numpy()
    models = [ac.fit(T[fT != j], a.seed, a.rounds, a.threads) for j in range(a.folds)]
    return fo, fT, models


def nested_stage1(T: pd.DataFrame, fT: np.ndarray, a) -> dict:
    """(j, k) -> stage-1 predictions for T rows of fold j from a model fitted without folds j and k (j != k)."""
    out = {}
    for j in range(a.folds):
        for k in range(j + 1, a.folds):
            m = ac.fit(T[(fT != j) & (fT != k)], a.seed, a.rounds, a.threads)
            out[(j, k)] = ac.predict(m, T[fT == j])
            out[(k, j)] = ac.predict(m, T[fT == k])
    return out


def stage2_val_models(R: "Rows", T, fT, fo, models, nested, a, keep: np.ndarray):
    """S_k for k in folds, trained on rows of the other folds with pairwise-nested stage-1 values. keep: rows with y (finite)."""
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
        S.append(fit2(M[tr], (y - s1)[tr], a.seed, a.rounds2, a.threads))
        del M
        print(f"  stage-2 fold {k}: {int(tr.sum()):,} rows", flush=True)
    return S, fold_row


def main():
    """Command line entry: run stage 1 and stage 2 for one setting and seed and write the outputs of the module docstring."""
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
    pa.add_argument("--ctx", default="none")
    pa.add_argument("--ctx_sha", default="none")
    pa.add_argument("--folds", type=int, default=10)
    pa.add_argument("--rounds", type=int, default=300)
    pa.add_argument("--rounds2", type=int, default=400)
    pa.add_argument("--max_pred", type=float, default=3600.0)
    pa.add_argument("--threads", type=int, default=3)
    pa.add_argument("--scale2", type=float, default=1.0, help="stage-2 correction multiplier before the window (1.0 = as fitted)")
    a = pa.parse_args()
    ac.below_normal()
    t0 = time.time()
    A = stage1_inputs(a)
    ctx = checked(a.ctx, a.ctx_sha).set_index("MVT_ID_mvt") if a.ctx != "none" else None
    val = pd.read_parquet(CACHE / f"val_{a.base}_s{a.base_seed}.parquet")
    T = ac.design(val, A)
    T = T[np.isfinite(T["y"].to_numpy())]
    b = val if a.setting == "val" else pd.read_parquet(CACHE / f"{a.setting}_{a.base}_s{a.base_seed}.parquet")
    two = ctx is not None
    if two:
        RV = Rows(val, A, a, ctx)
        assert set(np.unique(RV.day)) == set(T["day"].unique()), "val days with no matched row: the fold map would differ from adsb_corr"
        RB = RV if a.setting == "val" else Rows(b, A, a, ctx)
        X = RB.X
    else:
        X = T if a.setting == "val" else ac.design(b, A)
    keep_v = np.isfinite(val["y"].to_numpy(dtype=float))
    delta1 = np.zeros(len(b))
    delta2 = np.zeros(len(b))
    imp = None
    if a.setting == "val":
        fo, fT, models = stage1_folds(T, a)
        oof = np.zeros(len(T))
        for j, m in enumerate(models):
            oof[fT == j] = ac.predict(m, T[fT == j])
        delta1[T["pos"].to_numpy()] = oof
        note = f"{a.folds} day folds over {T['day'].nunique()} days, {len(T)} matched rows"
        if two:
            nested = nested_stage1(T, fT, a)
            S, fold_row = stage2_val_models(RV, T, fT, fo, models, nested, a, keep_v)
            s1 = np.clip(RV.base["pred"] + delta1, 0, None)
            M = RV.design2(s1, delta1)
            app = s1 < a.max_pred
            for k, m in enumerate(S):
                sel = (fold_row == k) & app
                delta2[sel] = m.predict(M[sel])
            imp = np.mean([m.feature_importance("gain") for m in S], axis=0)
            note += f"; stage 2 on all rows below {a.max_pred:.0f} s, pairwise-nested stage-1 inputs"
    elif a.setting == "fwd":
        jan_T = (T["day"].str[5:7] == "01").to_numpy()
        mdl = ac.fit(T[jan_T], a.seed, a.rounds, a.threads)
        delta1[X["pos"].to_numpy()] = ac.predict(mdl, X)
        note = f"January only: {T[jan_T]['day'].nunique()} days, {int(jan_T.sum())} matched rows"
        if two:
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
            S = fit2(RV.design2(s1v, dv)[tr], (val["y"].to_numpy(dtype=float) - s1v)[tr], a.seed, a.rounds2, a.threads)
            s1 = np.clip(RB.base["pred"] + delta1, 0, None)
            app = s1 < a.max_pred
            delta2[app] = S.predict(RB.design2(s1, delta1)[app])
            imp = S.feature_importance("gain")
            note += f"; stage 2 on {int(tr.sum()):,} January rows (January-only nested stage-1 inputs)"
    else:
        fo, fT, models = stage1_folds(T, a)
        delta1[X["pos"].to_numpy()] = np.mean([ac.predict(m, X) for m in models], axis=0)
        note = f"mean of {a.folds} day-fold correctors over {T['day'].nunique()} days, {len(T)} matched rows"
        if two:
            nested = nested_stage1(T, fT, a)
            S, _ = stage2_val_models(RV, T, fT, fo, models, nested, a, keep_v)
            s1 = np.clip(RB.base["pred"] + delta1, 0, None)
            M = RB.design2(s1, delta1)
            app = s1 < a.max_pred
            delta2[app] = np.mean([m.predict(M[app]) for m in S], axis=0)
            imp = np.mean([m.feature_importance("gain") for m in S], axis=0)
            note += f"; stage 2 = mean of {a.folds} fold models on all rows below {a.max_pred:.0f} s"
    base_pred = b["pred"].to_numpy(dtype=float)
    s1 = np.clip(base_pred + delta1, 0, None)
    pd.DataFrame({"MVT_ID_mvt": b["MVT_ID_mvt"].to_numpy(), "s1": s1, "d2": delta2}).to_parquet(
        CACHE.parent / "notes" / f"ctx_corr_{a.tag}_{a.setting}_s{a.seed}_delta.parquet", index=False)  # unscaled stage-2 change (diagnostics)
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
    print(f"[{a.tag} {a.setting} seed{a.seed}] base {a.base}_s{a.base_seed} (mu = {which}), ADS-B {a.adsb}({a.adsb_sha}) geo {a.geo}, ctx {a.ctx}, "
          f"matched {len(X):,} / {len(b):,}, stage-1 mean {delta1[X['pos'].to_numpy()].mean():+.1f} s, stage-2 rows {int((delta2 != 0).sum()):,} "
          f"mean {delta2[delta2 != 0].mean() if (delta2 != 0).any() else 0:+.1f} s, |stage 2| median {np.median(np.abs(delta2[delta2 != 0])) if (delta2 != 0).any() else 0:.1f} s, "
          f"{note}, {time.time() - t0:.0f} s")
    rep = {"tag": a.tag, "setting": a.setting, "seed": a.seed, "base": a.base, "adsb": a.adsb, "adsb_sha": a.adsb_sha, "geo": a.geo, "ctx": a.ctx, "scale2": a.scale2,
           "note": note, "rows": int(len(b)), "matched": int(len(X)), "stage2_rows": int((delta2 != 0).sum()), "seconds": round(time.time() - t0)}
    if imp is not None:
        sh = pd.Series(imp, index=X2) / np.sum(imp)
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
        rep["per_airport_dmse_stage2"] = {k: round(float((se_w[ap == k].sum() - se_1[ap == k].sum()) / len(y)), 1) for k in APTS}
        rep["per_airport_rmse"] = {k: [round(rmse(pred_s1w[ap == k], y[ap == k]), 2), round(rmse(pred_w[ap == k], y[ap == k]), 2)] for k in APTS}
        if two:
            dd = pd.DataFrame({"day": RB.day, "d": se_w - se_1}).groupby("day")["d"].mean()
            rep["days_better"] = f"{int((dd < 0).sum())}/{len(dd)}"
            rep["per_day_dmse_median"] = round(float(dd.median()), 1)
        core = y < 3600
        rep["body_rmse"] = [round(rmse(pred_s1w[core], y[core]), 3), round(rmse(pred_w[core], y[core]), 3)]
        print(json.dumps({k: rep[k] for k in ("dmse_stage2", "per_airport_dmse_stage2", "body_rmse") if k in rep} | ({"days_better": rep.get("days_better")}),
                         ensure_ascii=False))
    (CACHE.parent / "notes" / f"ctx_corr_{a.tag}_{a.setting}_s{a.seed}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    dst = CACHE / f"{a.setting}_{a.tag}_s{a.seed}.parquet"
    out.to_parquet(dst, index=False)
    if a.setting != "full":
        P = CACHE.parent / "preds"
        shutil.copyfile(P / f"es_valid_ids_{a.base}_{a.setting}_s{a.base_seed}.parquet", P / f"es_valid_ids_{a.tag}_{a.setting}_s{a.seed}.parquet")
    print(f"saved {dst.name}")


if __name__ == "__main__":
    main()
