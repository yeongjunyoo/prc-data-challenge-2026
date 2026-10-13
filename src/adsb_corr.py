# SPDX-License-Identifier: GPL-3.0-only
"""ADS-B ground-trace corrector (stage 1, docs/DESIGN.md section 7.1), then the LOBT window (src/lobt_window.project).

It corrects the residual of a base mixture prediction (--base) on the departures matched to open ADS-B data (adsb.lol
globe_history, ODbL 1.0) and leaves the other rows as they are. src/ctx_corr.py uses these functions as its stage 1.
Inputs: cache/<setting>_<base>_s<seed>.parquet (the base's val, fwd and full outputs); cache/<adsb>.parquet, the per-departure ADS-B
columns built by research/adsb/build_feats.py (its SHA-256 prefix must equal --adsb_sha); d_lobt of cache/<feats>.parquet.
A LightGBM learns y - pred (clipped to +-1,800 s) on the matched January and July 2025 rows of the base's val output, from the base's
columns (pred, mu, p_art, dsch, g, airport code, hour) and the ADS-B columns:
    val:  --folds day folds; the matched rows of a day are corrected by the model fitted without that day
    fwd:  the matched July rows are corrected by a model fitted on January days only
    full: the matched rows to predict are corrected by the mean of the --folds models of val
Output: cache/<setting>_<tag>_s<seed>.parquet (the base's columns with pred and p_art changed) and a copy of the base's early-stopping
rows. The airport code (APTS) knows the ten airports of the challenge.

    python src/adsb_corr.py --setting val --seed 0 --base <tag> --tag <tag>adsbw --feats feats_v25 --adsb adsb_v1 --adsb_sha <16>
        --folds 10 --rounds 300 --threads 3

Original note (Korean):
ADS-B 지상 궤적 보정기: 바탕 혼합 예측(--base)의 잔차를 adsb.lol 로 매칭된 출발편에서만 고치고, 마지막에 LOBT 창(src/lobt_window.project)을 건다.

입력: cache/<setting>_<base>_s<seed>.parquet(바탕 val, fwd, full), cache/<adsb>.parquet(research/adsb/build_feats.py 가 만든 출발편별 ADS-B 칸,
      --adsb_sha 와 sha256 앞 16자가 같아야 한다), cache/<feats>.parquet 의 d_lobt(창).
보정기: LightGBM 이 매칭된 2025-01, 07 홀드아웃 행(바탕 val)의 y − pred(±1,800초에서 자름)를 바탕 칸(pred, mu, p_art, dsch, g, 공항, 시)과
      ADS-B 칸으로 배운다. 매칭 안 된 행은 바탕 그대로 둔다.
  val : 날 묶음 --folds 겹(날을 섞어 나눔). 각 날의 매칭 행은 그 날이 빠진 보정기로 고친다(밖 표본). 날별, 공항별 표를 notes/adsb_corr_<tag>_val.json 에 쓴다.
  fwd : 바탕 fwd(2025-07)의 매칭 행을 1월 날만으로 배운 보정기로 고친다(7월 정보 없음, 앞으로 가는 자).
  full: 바탕 full(2026 시험)의 매칭 행을 val 의 --folds 개 보정기 평균으로 고친다.
그다음 창: mu 는 바탕의 혼합식 mu(lobt_window.mu_of), d_lobt 는 --feats 의 원래 칸.
출력: cache/<setting>_<tag>_s<seed>.parquet(바탕과 같은 칸, pred 와 p_art 만 바뀜). 조기 종료 행은 바탕 것을 이 태그로 복사(보정기는 바탕 학습을 건드리지 않는다).
    .venv/Scripts/python.exe -u src/adsb_corr.py --setting val --seed 0 --base m6r25sgc75 --tag m6r25sgc75adsbw --feats feats_v25 --adsb adsb_v1 --adsb_sha <16> --folds 10 --rounds 300 --threads 3
자료: adsb.lol globe_history(Open Database License 1.0), README 「외부 데이터」.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.common import breakdown, rmse  # noqa: E402
from src.load import CACHE  # noqa: E402
from src.lobt_window import feats_d_lobt, mu_of, project  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
CLIP = 1800.0
BASE_COLS = ["pred", "mu", "p_art", "dsch", "g", "apt_code", "hour"]
ADSB_COLS = ["est_mvt", "est_to", "est_first", "est_mvt_last", "dt_to", "fm_minus_pred", "first_minus_pred", "last_minus_pred",
             "fm_minus_sched", "first_minus_sched", "ev_first_gs", "ev_first_dist_m", "ev_first_stationary", "ev_stand_seen",
             "ev_stand_dur", "ev_fm_gap", "ev_n_mid_stops", "ev_mid_stop_s", "ev_mid_stop_max", "ev_n_long_stops",
             "ev_fm_last_gap", "ev_last_long_dur", "ev_to_gap", "ev_n_pts", "ev_max_gap_after_fm", "ev_share_mlat", "fm_seen", "fm_ok"]
A_READ = ["MVT_ID_mvt", "day", "ADEP_mvt", "mvt", "matched", "fm_seen", "fm_ok", "dt_to", "est_to", "est_mvt", "est_first",
          "est_mvt_last"] + [c for c in ADSB_COLS if c.startswith("ev_")]


def below_normal() -> None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass.restype = wintypes.BOOL
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS


def sha16(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def load_adsb(name: str, want_sha: str) -> pd.DataFrame:
    """Matched departures of the ADS-B table cache/<name>.parquet, indexed by MVT_ID_mvt, with hour and airport code.

    Stops if the SHA-256 prefix of the table differs from want_sha.
    """
    p = CACHE / f"{name}.parquet"
    got = sha16(p)
    if got != want_sha:
        raise SystemExit(f"{p.name} sha {got} != --adsb_sha {want_sha}")
    import pyarrow.parquet as pq
    A = pq.read_table(p, columns=A_READ, filters=[("matched", "=", True)]).to_pandas(strings_to_categorical=True)
    A["day"] = A["day"].astype(str)
    A["ADEP_mvt"] = A["ADEP_mvt"].astype(str)
    A["hour"] = ((A["mvt"] % 86400) // 3600).astype(float)
    A["apt_code"] = A["ADEP_mvt"].map({k: i for i, k in enumerate(APTS)}).astype(float)
    A["fm_seen"] = A["fm_seen"].astype(float)
    A["fm_ok"] = A["fm_ok"].astype(float)
    return A.set_index("MVT_ID_mvt")


def design(b: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    """Matched rows of base frame b with base and ADS-B columns (index = position in b)."""
    ids = b["MVT_ID_mvt"].to_numpy()
    hit = pd.Index(A.index).get_indexer(ids)
    pos = np.nonzero(hit >= 0)[0]
    X = A.iloc[hit[pos]].reset_index(drop=True)
    for c in ("pred", "mu", "p_art", "dsch", "g"):
        X[c] = b[c].to_numpy(dtype=float)[pos]
    X["fm_minus_pred"] = X["est_mvt"] - X["pred"]
    X["first_minus_pred"] = X["est_first"] - X["pred"]
    X["last_minus_pred"] = X["est_mvt_last"] - X["pred"]
    X["fm_minus_sched"] = X["dsch"] - X["est_mvt"]
    X["first_minus_sched"] = X["dsch"] - X["est_first"]
    X["pos"] = pos
    if "y" in b.columns:
        X["y"] = b["y"].to_numpy(dtype=float)[pos]
    return X


def fit(X: pd.DataFrame, seed: int, rounds: int, threads: int):
    """LightGBM of the clipped residual y - pred on the rows X, from the base and ADS-B columns (BASE_COLS, ADSB_COLS)."""
    import lightgbm as lgb
    params = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=threads, seed=seed,
                  deterministic=True, force_col_wise=True)
    cols = BASE_COLS + ADSB_COLS
    tgt = np.clip(X["y"].to_numpy() - X["pred"].to_numpy(), -CLIP, CLIP)
    return lgb.train(params, lgb.Dataset(X[cols].astype(float).to_numpy(), tgt), num_boost_round=rounds)


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Correction (seconds) of a model of fit for the rows X."""
    return model.predict(X[BASE_COLS + ADSB_COLS].astype(float).to_numpy())


def day_folds(days, k: int, seed: int) -> dict:
    """Fold number of each day: the days in a random order (seed 10,000 + seed) dealt to k folds in turn."""
    perm = np.random.default_rng(10_000 + seed).permutation(sorted(days))
    return {d: i % k for i, d in enumerate(perm)}


def fold_models(T: pd.DataFrame, k: int, seed: int, rounds: int, threads: int):
    """One model per day fold, each fitted without that fold's days; also returns the fold of every row of T."""
    fo = day_folds(T["day"].unique(), k, seed)
    f = T["day"].map(fo).to_numpy()
    return [fit(T[f != j], seed, rounds, threads) for j in range(k)], f


def report(b, pred_w, pred_basew, X, setting, tag) -> dict:
    """Ruler summary of a corrected prediction: RMSE of the base and of the corrected values on all, matched,
    first-movement and unmatched rows, and by airport.
    """
    y = b["y"].to_numpy(dtype=float)
    pos = X["pos"].to_numpy()
    m = np.zeros(len(b), bool)
    m[pos] = True
    fm = np.zeros(len(b), bool)
    fm[pos[X["fm_ok"].to_numpy() > 0]] = True
    out = {"rows": int(len(b)), "matched": int(m.sum()), "fm_ok": int(fm.sum())}
    for name, sel in (("all", np.ones(len(b), bool)), ("matched", m), ("fm_ok", fm), ("unmatched", ~m), ("body_y_le_3600", y <= 3600)):
        out[name] = {"n": int(sel.sum()), "base_window": round(rmse(pred_basew[sel], y[sel]), 3), "adsb_window": round(rmse(pred_w[sel], y[sel]), 3)}
    ap = pd.Series("", index=range(len(b)), dtype=object)
    ap.iloc[pos] = X["ADEP_mvt"].to_numpy()
    days = pd.Series("", index=range(len(b)), dtype=object)
    days.iloc[pos] = X["day"].to_numpy()
    out["per_airport_matched"] = {a: {"n": int((ap == a).sum()),
                                      "base_window": round(rmse(pred_basew[(ap == a).to_numpy()], y[(ap == a).to_numpy()]), 2),
                                      "adsb_window": round(rmse(pred_w[(ap == a).to_numpy()], y[(ap == a).to_numpy()]), 2)}
                                  for a in APTS if (ap == a).sum()}
    out["per_day_matched"] = {d: {"n": int((days == d).sum()),
                                  "base_window": round(rmse(pred_basew[(days == d).to_numpy()], y[(days == d).to_numpy()]), 2),
                                  "adsb_window": round(rmse(pred_w[(days == d).to_numpy()], y[(days == d).to_numpy()]), 2)}
                              for d in sorted(set(X["day"]))}
    return out


def main():
    """Command line entry: correct the base output for one setting and write the outputs of the module docstring."""
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", default="val", choices=["val", "fwd", "full"])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--base", required=True)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--feats", default="feats_v25")
    pa.add_argument("--adsb", required=True)
    pa.add_argument("--adsb_sha", required=True)
    pa.add_argument("--folds", type=int, default=10)
    pa.add_argument("--rounds", type=int, default=300)
    pa.add_argument("--threads", type=int, default=3)
    pa.add_argument("--scale", type=float, default=1.0, help="보정량에 곱하는 값(1.0 = 그대로, 묶음 변형용)")
    a = pa.parse_args()
    below_normal()
    A = load_adsb(a.adsb, a.adsb_sha)
    val = pd.read_parquet(CACHE / f"val_{a.base}_s{a.seed}.parquet")
    T = design(val, A)
    T = T[np.isfinite(T["y"].to_numpy())]
    b = val if a.setting == "val" else pd.read_parquet(CACHE / f"{a.setting}_{a.base}_s{a.seed}.parquet")
    X = design(b, A) if a.setting != "val" else T
    delta = np.zeros(len(b))
    if a.setting == "val":
        models, f = fold_models(T, a.folds, a.seed, a.rounds, a.threads)
        oof = np.zeros(len(T))
        for j, mdl in enumerate(models):
            oof[f == j] = predict(mdl, T[f == j])
        delta[T["pos"].to_numpy()] = oof
        note = f"{a.folds} day folds over {T['day'].nunique()} days, {len(T)} matched rows"
    elif a.setting == "fwd":
        jan = T[T["day"].str[5:7] == "01"]
        mdl = fit(jan, a.seed, a.rounds, a.threads)
        delta[X["pos"].to_numpy()] = predict(mdl, X)
        note = f"January only: {jan['day'].nunique()} days, {len(jan)} matched rows"
    else:
        models, _ = fold_models(T, a.folds, a.seed, a.rounds, a.threads)
        delta[X["pos"].to_numpy()] = np.mean([predict(mdl, X) for mdl in models], axis=0)
        note = f"mean of {a.folds} day-fold correctors over {T['day'].nunique()} days, {len(T)} matched rows"
    delta = a.scale * delta
    base_pred = b["pred"].to_numpy(dtype=float)
    corr = np.clip(base_pred + delta, 0, None)
    which, mu = mu_of(b)
    p_art, g, dsch = b["p_art"].to_numpy(dtype=float), b["g"].to_numpy().astype(bool), b["dsch"].to_numpy(dtype=float)
    d = feats_d_lobt(a.feats, b["MVT_ID_mvt"].to_numpy())
    pred_w, p2, changed = project(corr, mu, p_art, g, dsch, d)
    pred_basew, _, _ = project(base_pred, mu, p_art, g, dsch, d)
    assert np.isfinite(pred_w).all() and (pred_w >= 0).all()
    unmatched = np.ones(len(b), bool)
    unmatched[X["pos"].to_numpy()] = False
    assert np.array_equal(pred_w[unmatched], pred_basew[unmatched]), "매칭 안 된 행이 바탕 + 창과 달라졌다"
    out = b.copy()
    out["pred"], out["p_art"] = pred_w, p2
    print(f"[{a.tag} {a.setting} seed{a.seed}] 바탕 {a.base}(mu = {which}), ADS-B {a.adsb}({a.adsb_sha}), 매칭 행 {len(X):,} / {len(b):,}, "
          f"보정 평균 {delta[X['pos'].to_numpy()].mean():+.1f}초, |보정| 중앙 {np.median(np.abs(delta[X['pos'].to_numpy()])):.1f}초, {note}")
    if "y" in b.columns:
        y = b["y"].to_numpy(dtype=float)
        print(f"  바탕 {rmse(base_pred, y):.3f} → 바탕+창 {rmse(pred_basew, y):.3f} → ADS-B 보정+창 {rmse(pred_w, y):.3f}")
        import pyarrow.parquet as pq
        ap = pq.read_table(CACHE / f"{a.feats}.parquet", columns=["MVT_ID_mvt", "ap"]).to_pandas(strings_to_categorical=True)
        ap = ap.set_index("MVT_ID_mvt")["ap"].reindex(b["MVT_ID_mvt"].to_numpy()).astype(str).to_numpy()
        print(breakdown(pred_w, y, ap, f"{a.tag} {a.setting}"))
        rep = report(b, pred_w, pred_basew, X, a.setting, a.tag)
        rep.update(tag=a.tag, setting=a.setting, base=a.base, adsb=a.adsb, adsb_sha=a.adsb_sha, note=note,
                   rmse_base=round(rmse(base_pred, y), 3), rmse_base_window=round(rmse(pred_basew, y), 3),
                   rmse_adsb_window=round(rmse(pred_w, y), 3))
        (CACHE.parent / "notes" / f"adsb_corr_{a.tag}_{a.setting}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    dst = CACHE / f"{a.setting}_{a.tag}_s{a.seed}.parquet"
    out.to_parquet(dst, index=False)
    if a.setting != "full":
        P = CACHE.parent / "preds"
        shutil.copyfile(P / f"es_valid_ids_{a.base}_{a.setting}_s{a.seed}.parquet", P / f"es_valid_ids_{a.tag}_{a.setting}_s{a.seed}.parquet")
    print(f"saved {dst.name}")


if __name__ == "__main__":
    main()
