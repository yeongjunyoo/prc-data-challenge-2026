# SPDX-License-Identifier: GPL-3.0-only
"""3차 모델: 전역 혼합 + G 그룹(LIRF, NM 미매칭) 전용 기대값.

R13-F3 사본(research/r13_f3/model3ap.py, vault research/2026-10-07-r13-f3.md 6): src/model3.py(sha16 4641d5b5d91abd26)에 --mu_sep all 만 더했다.
  all: 회귀(mu)를 공항마다 따로 학습한다(공항마다 내부 조기 종료, 재학습 x1.1, full 은 공항별 반복 수 iters["reg_<공항>"]). 분류기, G 모형,
  혼합, 창 밖 규칙, 출력은 원본 그대로다. all 이 아니면 원본과 같은 길을 탄다.

전역:  pred = p_art * d_sch + (1 - p_art) * mu
  p_art : BLOCK 이 SCHED 와 60초 안일 확률(LightGBM 이진, --wpow 로 (|d_sch-1000|) 가중 가능)
  mu    : art 도 shift(하루 어긋남)도 아닌 행으로 학습한 L2 회귀
G 그룹: pred = pa*d_sch + ps*(86400*max(dd,1) + mu) + pn*mu   (pa, ps, pn 은 src/sub_g.py 의 lookup+접두사 와 작은 GBM 평균)
direct: 같은 피처로 정답을 바로 맞히는 L2 회귀(성격이 다른 차선)
--eobt_comp: 세 번째 성분. art 가 아닌 몫을 p_e * d_eobt + (1 - p_e) * mu 로 나눈다(p_e: BLOCK 이 EOBT1 과 EOBT_WIN 초 안일 확률,
             EOBT1 이 없는 행은 0). mu 는 이 행들까지 뺀 회귀다.

--setting val : 2025 의 1, 7월을 빼고 학습, 1, 7월로 잰다(로컬 자).
--setting fwd : 2025-01~06 으로 학습, 2025-07 로 잰다(앞으로 가는 자, 부호 점검용).
--setting nd  : 2025-01~10 으로 학습, 2025-11, 12 로 잰다(해가 바뀌는 시험의 옮김을 학습 기간 안에서 재는 둘째 자, 10-03).
--setting full: 2025 전체로 학습해 ranking 출발을 예측한다. 반복 수는 val 의 1.15 배(cache/iters_<tag>.json).
--es internal(기본): 조기 종료를 학습 범위 안의 날짜 블록(UTC 날짜 번호 % 10 == 0, 약 10%)으로 하고 그 날들은 학습에서 뺀다.
                    잴 행(1, 7월)은 조기 종료에 쓰지 않는다.
--es holdout      : 옛 방식(잴 행으로 조기 종료). 기본값을 --es internal 로 바꾼 것은 10-02 18:31 KST(notes/jobs_ruler.txt 생성 시각)이고,
                    그 전 결과(m3, m3w, m4*, m5, m6, m7)는 이 옛 방식이라 낙관적이다.
--refit           : 내부 조기 종료로 찾은 반복 수 x 1.1 로 학습 범위 전체(조기 종료 날 포함)에 다시 맞춰 잰다.
                    전체 데이터 최종 모델(--setting full)과 같은 방식이다. full 은 반복 수 x 1.1 x --full_mult(기본 1.15)를 쓴다.
조기 종료 행은 preds/es_valid_ids_<tag>_<setting>_s<seed>.parquet 에 남긴다.
출력: cache/<setting>_<tag>_s<seed>.parquet (MVT_ID_mvt, pred, direct, p_art, mu, g, [y])
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src import sub_g  # noqa: E402
from src.common import breakdown, feature_cols, rmse  # noqa: E402
from src.feats import CAT_FEATS, build  # noqa: E402
from src.load import CACHE  # noqa: E402
from src.te import add_te  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
HIGH_CARD = {"ades", "stand", "op", "prefix"}
REFIT = 1.1  # 재학습 반복 수 배수: 학습 범위 전체 / 조기 종료 뺀 학습 행 = 약 1.11
EOBT_WIN = 60  # --eobt_comp 의 라벨: |y - d_eobt| <= 60 초(art 의 SCHED 창과 같은 폭)
T0 = time.time()


def log(msg: str):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def to_matrix(F: pd.DataFrame, cols: list[str], rows: np.ndarray) -> np.ndarray:
    X = np.empty((int(rows.sum()), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        s = F[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            v = s.cat.codes.to_numpy()[rows].astype(np.float32)
            v[v < 0] = np.nan
            X[:, j] = v
        else:
            X[:, j] = s.to_numpy(dtype=np.float32, na_value=np.nan)[rows]
    return X


def train(params, dset, rounds, dval=None):
    if dval is None:
        return lgb.train(params, dset, num_boost_round=rounds), rounds
    m = lgb.train(params, dset, num_boost_round=rounds, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    return m, m.best_iteration


def g_probs(Gtr: pd.DataFrame, Gte: pd.DataFrame, seeds=(0, 1, 2)) -> np.ndarray:
    """G 그룹 세 유형 확률: lookup+항공사 키 보정과 작은 GBM(시드 3개) 의 평균. 조기 종료 없음(고정 200회).

    키 칸 이름은 "pfx" 로 고정이다(sub_g.gbm_probs 가 그 이름을 범주로 쓴다). 피처 v2.5 부터 값은 편명 접두사 대신 목적지(ADES)다.
    """
    ytr = Gtr["y"].to_numpy()
    dtr = Gtr["d_sch"].to_numpy().astype(float)
    k = sub_g.labels(ytr, dtr)
    c_tr = sub_g.cell(dtr, Gtr["dd"].to_numpy())
    c_te = sub_g.cell(Gte["d_sch"].to_numpy().astype(float), Gte["dd"].to_numpy())
    P1 = sub_g.pfx_adjust(sub_g.predict_lookup(sub_g.fit_lookup(c_tr, k), c_te), Gte["pfx"].to_numpy(), Gtr["pfx"].to_numpy(), k)
    feat = [c for c in ["d_sch", "dd", "hour_mvt", "hour_sch", "garb", "fl_len", "stand_d"] if c in Gtr.columns]
    cats = pd.CategoricalDtype(sorted(set(Gtr["pfx"]) | set(Gte["pfx"])))
    Xtr = Gtr[feat].astype("float32").copy()
    Xtr["pfx"] = Gtr["pfx"].astype(cats)
    Xte = Gte[feat].astype("float32").copy()
    Xte["pfx"] = Gte["pfx"].astype(cats)
    P2 = np.mean([sub_g.gbm_probs(Xtr, k, Xte, seed=s) for s in seeds], axis=0)
    return 0.5 * P1 + 0.5 * P2


G_BAND = (3600.0, 14400.0)  # --g_band_avg 의 지연(d_sch) 구간


def g_override(P: np.ndarray, d_sch: np.ndarray, dd: np.ndarray, mu_g: np.ndarray, p_art_g: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """G 행의 기대값과 정상 몫 확률. P 는 g_probs 의 [정상, art, shift]. p_art_g 를 주면 지연이 G_BAND 안인 행의 art 확률을
    G 모형과 주 분류기의 평균으로 바꾼다(10-03 열 달 한 달씩 빼기 시험에서 10/10 달 G 제곱오차 감소, research/gmodel/oof_g_art.py)."""
    pa, pn = P[:, 1], P[:, 0]
    if p_art_g is not None:
        band = (d_sch >= G_BAND[0]) & (d_sch < G_BAND[1])
        pa = np.where(band, 0.5 * P[:, 1] + 0.5 * p_art_g, P[:, 1])
        pn = np.where(band, np.clip(1 - pa - P[:, 2], 0, None), P[:, 0])
    g = pa * d_sch + P[:, 2] * (86400 * np.maximum(dd, 1) + np.clip(mu_g, 300, 3000)) + pn * mu_g
    return g, pn


def reg_set(dbase, idx: np.ndarray, target: np.ndarray, anchor: str):
    """회귀 행의 부분 Dataset. --anchor 가 있으면 라벨을 y - 기준값(target)으로 바꾼다. 없으면 바탕의 y 그대로(옛 동작)."""
    d = dbase.subset(idx)
    if anchor:
        d.construct()
        d.set_label(np.nan_to_num(target[idx]))
    return d


def es_day_mask(mvt_ids: np.ndarray) -> np.ndarray:
    """조기 종료용 날짜 블록: 이륙 시각의 UTC 날짜 번호 % 10 == 0."""
    A = pd.read_parquet(CACHE / "all.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "t_mvt"])
    A = A[A["PHASE_mvt"] == "DEP"]
    day = pd.Series((A["t_mvt"].to_numpy() // 86400).astype(np.int64), index=A["MVT_ID_mvt"].to_numpy())
    d = day.reindex(mvt_ids).to_numpy()
    return (d % 10) == 0


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", default="val", choices=["val", "fwd", "nd", "fj", "full"])
    pa.add_argument("--es", default="internal", choices=["internal", "holdout"])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--rounds", type=int, default=6000)
    pa.add_argument("--tag", default="m3")
    pa.add_argument("--iters_tag", default="", help="full 에서 반복 수를 읽을 태그(기본은 --tag)")
    pa.add_argument("--lr", type=float, default=0.05)
    pa.add_argument("--leaves", type=int, default=127)
    pa.add_argument("--mdl", type=int, default=200)
    pa.add_argument("--wpow", type=float, default=0.0)
    pa.add_argument("--wcap", type=float, default=60000.0)
    pa.add_argument("--drop_hc", action="store_true", help="고카디널리티 범주 원본을 빼고 그룹 통계만 쓴다")
    pa.add_argument("--no_direct", action="store_true")
    pa.add_argument("--threads", type=int, default=6)
    pa.add_argument("--g_only", action="store_true", help="G 확률까지만 계산하고 끝낸다(점검용)")
    pa.add_argument("--feats", default="feats", help="cache/<이름>.parquet 피처 판(feats_v2: m3, m3w / feats_v21: m4 계열 / feats_v22: m5, m6 / feats: 최신)")
    pa.add_argument("--drop", default="", help="쉼표로 뺄 피처 칸")
    pa.add_argument("--clf_drop_prefix", default="", help="art 분류기에서만 뺄 칸 접두사(쉼표), 예: m_ 는 METAR")
    pa.add_argument("--lr_reg", type=float, default=0.0, help="회귀만 다른 학습률(0 이면 --lr)")
    pa.add_argument("--refit", action="store_true", help="내부 조기 종료 반복 수로 학습 범위 전체에 다시 맞춰 잰다(val, fwd). full 에서는 효과 없음")
    pa.add_argument("--full_mult", type=float, default=1.15, help="full 반복 수 = 내부 조기 종료 반복 수 x refit_factor x 이 값")
    pa.add_argument("--eobt_comp", action="store_true", help="세 번째 혼합 성분(BLOCK = EOBT1)을 켠다")
    pa.add_argument("--anchor", default="", choices=["", "aobt", "win"],
                    help="aobt: 회귀(mu)의 목표를 y - clip(MVT - AOBT_3, 0, 7200) 로 두고 예측에 그 값을 더한다(AOBT_3 없는 행은 학습 범위 정상 행의 공항별 중앙값). "
                         "win: 기준값을 그 공항별 중앙값을 LOBT 창 [d_lobt - 3606, d_lobt + 3606] 안으로 자른 값으로(LOBT 없는 행은 중앙값). "
                         "근거 vault research/2026-10-06-r5-paradigms.md(창 바닥에 붙는 지연 출발)")
    pa.add_argument("--g_band_avg", action="store_true", help="G 행 가운데 지연 1~4시간 행의 art 확률을 G 모형과 주 분류기의 평균으로(g_override)")
    pa.add_argument("--mu_sep", default="", help="쉼표로 준 공항은 회귀(mu)를 따로 학습한다: 그 공항 행은 그 공항들만의 회귀, 나머지는 그 공항들을 뺀 회귀(예: LIRF)")
    args = pa.parse_args()
    F = build(name=args.feats)
    if args.drop:
        F = F.drop(columns=[c for c in args.drop.split(",") if c in F.columns])
    src = F["src"].to_numpy()
    te = src == 0
    if args.setting == "val":
        fit_rows, eval_rows = ~np.isin(src, [1, 7]) & ~te, np.isin(src, [1, 7])
    elif args.setting == "fwd":
        fit_rows, eval_rows = np.isin(src, [1, 2, 3, 4, 5, 6]), src == 7
    elif args.setting == "fj":  # F2 (vault research/2026-10-08-f2-four-months.md): 2025 의 2, 6월을 빼고 학습, 2, 6월로 잰다
        fit_rows, eval_rows = ~np.isin(src, [2, 6]) & ~te, np.isin(src, [2, 6])
    elif args.setting == "nd":
        fit_rows, eval_rows = np.isin(src, list(range(1, 11))), np.isin(src, [11, 12])
    else:
        fit_rows, eval_rows = ~te, te
    labeled_eval = args.setting in ("val", "fwd", "nd", "fj")
    # 조기 종료 행과 학습 행
    if args.setting == "full":
        es_rows = np.zeros(len(F), bool)
    elif args.es == "internal":
        es_rows = fit_rows & es_day_mask(F["MVT_ID_mvt"].to_numpy())
    else:
        es_rows = eval_rows.copy()
    train_rows = fit_rows & ~es_rows if args.es == "internal" else fit_rows
    if args.setting != "full":  # 조기 종료에 쓴 행을 남긴다(자 행과 겹치지 않는지 도구가 확인)
        pd.DataFrame({"MVT_ID_mvt": F["MVT_ID_mvt"].to_numpy()[es_rows]}).to_parquet(
            CACHE.parent / "preds" / f"es_valid_ids_{args.tag}_{args.setting}_s{args.seed}.parquet", index=False)
    T = add_te(F, fit_rows)
    F = pd.concat([F, T], axis=1)
    del T
    cols = feature_cols(F, HIGH_CARD if args.drop_hc else None)
    cat_idx = [j for j, c in enumerate(cols) if c in CAT_FEATS]
    y_all = F["y"].to_numpy().astype(float)
    dsch_all = F["d_sch"].to_numpy().astype(float)
    art_all = (np.abs(y_all - dsch_all) <= 60)
    shift_all = (y_all > 60000) & ~art_all
    # 세 번째 성분(--eobt_comp): BLOCK 이 EOBT1 과 EOBT_WIN 초 안(art, shift 가 아닌 행). EOBT1 이 없으면 이 성분은 없다
    deobt_all = F["d_eobt"].to_numpy().astype(float) if "d_eobt" in F.columns else np.full(len(F), np.nan)  # --drop 으로 뺐으면 성분도 없다
    assert not (args.eobt_comp and "d_eobt" not in F.columns), "--eobt_comp 에는 d_eobt 칸이 있어야 한다"
    eob_all = (np.abs(y_all - deobt_all) <= EOBT_WIN) & ~art_all & ~shift_all
    g_all = ((F["ap"] == "LIRF") & (F["nm"] == 0)).to_numpy()
    APALL = args.mu_sep == "all"  # R13-F3: 공항별 회귀
    if APALL:
        mus_all = np.zeros(len(F), bool)
        apa_all = F["ap"].astype(str).to_numpy()
    else:
        mus_all = F["ap"].astype(str).isin([a for a in args.mu_sep.split(",") if a]).to_numpy()  # --mu_sep 공항 행
        assert not args.mu_sep or mus_all.any(), f"--mu_sep {args.mu_sep!r} 에 맞는 공항 행이 없다"
    if args.anchor == "aobt":
        da = F["d_aobt"].to_numpy().astype(float)
        apv = F["ap"].astype(str).to_numpy()
        nfit = fit_rows & ~art_all & ~shift_all & (y_all < 3600)
        med = pd.Series(y_all[nfit]).groupby(apv[nfit]).median()
        fb = pd.Series(apv).map(med).fillna(float(np.median(y_all[nfit]))).to_numpy()
        anchor_all = np.where(np.isnan(da), fb, np.clip(da, 0, 7200))
    elif args.anchor == "win":  # LOBT 창 기준(vault research/2026-10-06-r5-paradigms.md): 공항별 정상 학습 행 중앙값을 창 [d_lobt - 3606, d_lobt + 3606] 안으로
        dl = F["d_lobt"].to_numpy().astype(float)
        apv = F["ap"].astype(str).to_numpy()
        nfit = fit_rows & ~art_all & ~shift_all & (y_all < 3600)
        med = pd.Series(y_all[nfit]).groupby(apv[nfit]).median()
        fb = pd.Series(apv).map(med).fillna(float(np.median(y_all[nfit]))).to_numpy()
        anchor_all = np.where(np.isnan(dl), fb, np.clip(fb, dl - 3606.0, dl + 3606.0))
    else:
        anchor_all = np.zeros(len(F))
    meta = pd.DataFrame({"MVT_ID_mvt": F["MVT_ID_mvt"].to_numpy()[eval_rows], "ap": F["ap"].astype(str).to_numpy()[eval_rows]})
    # G 그룹 표(조기 종료가 없으니 학습 범위 전체를 쓴다)
    gkey = "prefix" if "prefix" in F.columns else "ades"  # 피처 v2.5 는 편명 접두사가 없어 목적지로 항공사를 대신한다
    Gcols = [c for c in ["y", "d_sch", "dd", gkey, "hour_mvt", "hour_sch", "garb", "fl_len", "stand"] if c in F.columns]
    G = F.loc[g_all & (fit_rows | eval_rows), Gcols].copy()
    G["pfx"] = G[gkey].astype(str).fillna("nan")
    G["stand_d"] = pd.to_numeric(G["stand"].astype(str).fillna("nan").str[5:6], errors="coerce")
    g_fit_mask = fit_rows[g_all & (fit_rows | eval_rows)]
    Gf = G[g_fit_mask & ~np.isnan(G["y"].to_numpy())]
    Ge = G[~g_fit_mask]
    P = g_probs(Gf, Ge)
    log(f"G probs: fit {len(Gf)} eval {len(Ge)} mean pa {P[:, 1].mean():.3f} ps {P[:, 2].mean():.4f}")
    if args.g_only:
        return
    Xf = to_matrix(F, cols, train_rows)
    Xs = to_matrix(F, cols, es_rows) if es_rows.any() else None
    Xe = to_matrix(F, cols, eval_rows)
    del F
    yf, ys, ye = y_all[train_rows], y_all[es_rows], y_all[eval_rows]
    artf, arts, arte = art_all[train_rows].astype(float), art_all[es_rows].astype(float), art_all[eval_rows].astype(float)
    shf, shs = shift_all[train_rows], shift_all[es_rows]
    dsf, dss, dse = dsch_all[train_rows], dsch_all[es_rows], dsch_all[eval_rows]
    eobf, eobs = eob_all[train_rows], eob_all[es_rows]
    deobf, deobs, deobe = deobt_all[train_rows], deobt_all[es_rows], deobt_all[eval_rows]
    if not args.eobt_comp:  # 성분이 꺼져 있으면 회귀가 빼는 행도 없다(옛 동작 그대로)
        eobf, eobs = np.zeros_like(eobf), np.zeros_like(eobs)
    ge = g_all[eval_rows]
    msf, mss, mse_ = mus_all[train_rows], mus_all[es_rows], mus_all[eval_rows]
    if APALL:
        apa_f, apa_s, apa_e = apa_all[train_rows], apa_all[es_rows], apa_all[eval_rows]
    anf, ans, ane = anchor_all[train_rows], anchor_all[es_rows], anchor_all[eval_rows]
    log(f"[{args.tag} {args.setting} es={args.es} seed{args.seed}] features {len(cols)} train {len(yf)} es {len(ys)} eval {len(ye)}"
        + (f" | mu_sep all: {len(set(apa_f))} airports" if APALL else f" | mu_sep {args.mu_sep}: train {int(msf.sum())} eval {int(mse_.sum())}" if args.mu_sep else ""))
    params = dict(learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=args.mdl, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=args.threads, verbose=-1,
                  cat_smooth=20.0, min_data_per_group=50, max_cat_threshold=64, seed=args.seed, force_col_wise=True,
                  feature_pre_filter=False)
    iters_path = CACHE / f"iters_{args.iters_tag or args.tag}.json"
    fixed = json.loads(iters_path.read_text()) if args.setting == "full" else None
    wfun = (lambda d: (np.clip(np.abs(d - 1000.0), 300.0, args.wcap) / 1000.0) ** args.wpow) if args.wpow > 0 else (lambda d: None)
    wf, ws = wfun(dsf), wfun(dss)
    # 분류기만 일부 칸을 뺄 때(--clf_drop_prefix): 따로 Dataset 을 만든다
    pref = tuple(p for p in args.clf_drop_prefix.split(",") if p)
    c_idx = [j for j, c in enumerate(cols) if not (pref and c.startswith(pref))]
    sep = len(c_idx) < len(cols)
    Xe_c = Xe[:, c_idx] if sep else Xe
    if sep:
        cat_c = [k for k, j in enumerate(c_idx) if j in cat_idx]
        dc = lgb.Dataset(Xf[:, c_idx], artf, weight=wf, categorical_feature=cat_c, free_raw_data=True, params=params)
        dc.construct()
        dvc = None
        if Xs is not None:
            dvc = lgb.Dataset(Xs[:, c_idx], arts, weight=ws, reference=dc, categorical_feature=cat_c, free_raw_data=True,
                              params=params)
            dvc.construct()
        log(f"classifier uses {len(c_idx)} of {len(cols)} features (dropped prefix {pref})")
    base = lgb.Dataset(Xf, np.nan_to_num(yf), categorical_feature=cat_idx, free_raw_data=True, params=params)
    base.construct()
    vbase = None
    if Xs is not None:
        vbase = lgb.Dataset(Xs, np.nan_to_num(ys), reference=base, categorical_feature=cat_idx, free_raw_data=True,
                            params=params)
        vbase.construct()
    if not args.refit:
        del Xf, Xs
    log("datasets built")
    # 1) art 분류
    if not sep:
        dc = base.subset(np.arange(len(yf)))
        dc.construct()
        dc.set_label(artf)
        if wf is not None:
            dc.set_weight(wf)
        dvc = None
        if vbase is not None:
            dvc = vbase.subset(np.arange(len(ys)))
            dvc.construct()
            dvc.set_label(arts)
            if ws is not None:
                dvc.set_weight(ws)
    mult = (fixed.get("refit_factor", 1.0) * args.full_mult) if fixed else 1.0  # full: 내부 조기 종료 반복 수 x 재학습 배수 x full_mult
    rounds_c = int(fixed["clf"] * mult) if fixed else args.rounds
    m_c, it_c = train(dict(params, objective="binary"), dc, rounds_c, dvc)
    p_art = m_c.predict(Xe_c, num_iteration=it_c if fixed is None else None)
    log(f"clf iters {it_c}")
    # 1b) 세 번째 성분: art 아닌 행 중 BLOCK 이 EOBT1 에 붙을 확률(EOBT1 있는 행만 학습, 없는 행은 0)
    it_e = de = dve = None
    p_e = np.zeros(len(ye))
    if args.eobt_comp:
        def eobt_set(dbase, rows_art, rows_sh, rows_de, rows_lab):
            idx = np.where((rows_art == 0) & ~rows_sh & ~np.isnan(rows_de))[0]
            d = dbase.subset(idx)
            d.construct()
            d.set_label(rows_lab[idx].astype(float))
            w = wfun(rows_de[idx])
            if w is not None:
                d.set_weight(w)
            return d
        dc.construct()
        de = eobt_set(dc, artf, shf, deobf, eobf)
        dve = eobt_set(dvc, arts, shs, deobs, eobs) if dvc is not None else None
        rounds_e = int(fixed["eclf"] * mult) if fixed else args.rounds
        m_e, it_e = train(dict(params, objective="binary"), de, rounds_e, dve)
        p_e = np.where(np.isnan(deobe), 0.0, m_e.predict(Xe_c, num_iteration=it_e if fixed is None else None))
        log(f"eobt clf iters {it_e} (rows labelled eobt in train {int(eobf.sum())})")
    # 2) 나머지 회귀(art, shift, 그리고 성분을 켰으면 eobt 행을 뺀 행)
    #    --mu_sep: 그 공항 행(msf)을 뺀 회귀와 그 공항 행만의 회귀 둘. 평가 행은 공항에 따라 둘 중 하나를 쓴다
    #    --anchor: 회귀 라벨은 y - 기준값, 예측은 기준값 + 회귀값(reg_set)
    preg = dict(params, objective="regression")
    if args.lr_reg:
        preg["learning_rate"] = args.lr_reg
    it_r2 = None
    if APALL:  # R13-F3: 공항마다 회귀 하나(조기 종료 행도 그 공항 행)
        dr = dvr = None
        it_r = None
        mu = np.full(len(ye), np.nan)
        it_ap, m_ap = {}, {}
        for a in sorted(set(apa_f)):
            d_a = reg_set(base, np.where((artf == 0) & ~shf & ~eobf & (apa_f == a))[0], yf - anf, args.anchor)
            dv_a = reg_set(vbase, np.where((arts == 0) & ~shs & ~eobs & (apa_s == a))[0], ys - ans, args.anchor) if vbase is not None else None
            m_a, it_a = train(preg, d_a, int(fixed[f"reg_{a}"] * mult) if fixed else args.rounds, dv_a)
            sel = apa_e == a
            mu[sel] = ane[sel] + m_a.predict(Xe[sel], num_iteration=it_a if fixed is None else None)
            it_ap[a], m_ap[a] = it_a, m_a
            log(f"reg {a} iters {it_a}")
            del d_a, dv_a
        assert not np.isnan(mu).any(), "평가 행에 학습 행이 없는 공항이 있다"
        m_r = m_ap[max(m_ap, key=lambda k: int((apa_f == k).sum()))]  # 중요도 출력용: 학습 행이 가장 많은 공항
    else:
        dr = reg_set(base, np.where((artf == 0) & ~shf & ~eobf & ~msf)[0], yf - anf, args.anchor)
        dvr = reg_set(vbase, np.where((arts == 0) & ~shs & ~eobs & ~mss)[0], ys - ans, args.anchor) if vbase is not None else None
        rounds_r = int(fixed["reg"] * mult) if fixed else args.rounds
        preg = dict(params, objective="regression")
        if args.lr_reg:
            preg["learning_rate"] = args.lr_reg
        m_r, it_r = train(preg, dr, rounds_r, dvr)
        mu = ane + m_r.predict(Xe, num_iteration=it_r if fixed is None else None)
        log(f"reg iters {it_r}")
        it_r2 = None
        if args.mu_sep:
            dr2 = reg_set(base, np.where((artf == 0) & ~shf & ~eobf & msf)[0], yf - anf, args.anchor)
            dvr2 = reg_set(vbase, np.where((arts == 0) & ~shs & ~eobs & mss)[0], ys - ans, args.anchor) if vbase is not None else None
            m_r2, it_r2 = train(preg, dr2, int(fixed["reg2"] * mult) if fixed else args.rounds, dvr2)
            mu = np.where(mse_, ane + m_r2.predict(Xe, num_iteration=it_r2 if fixed is None else None), mu)
            log(f"reg2 ({args.mu_sep}) iters {it_r2}")
    it_d = None
    if not args.no_direct:
        rounds_d = int(fixed["direct"] * mult) if fixed else args.rounds
        m_d, it_d = train(dict(params, objective="regression"), base, rounds_d, vbase)
        dpred = m_d.predict(Xe, num_iteration=it_d if fixed is None else None)
        log(f"direct iters {it_d}")
    # 2b) 재학습: 내부 조기 종료로 찾은 반복 수 x REFIT 로 학습 범위 전체(조기 종료 날 포함)에 다시 맞춘다.
    #     전체 데이터 최종 모델(full)과 같은 방식(고정 반복 수, 모든 학습 행)이라 자가 그 모델을 잰다.
    if args.refit and args.setting in ("val", "fwd", "nd", "fj"):
        assert args.es == "internal"
        del base, vbase, dc, dvc, dr, dvr, de, dve
        Xa = np.concatenate([Xf, Xs])
        del Xf, Xs
        ya = np.concatenate([yf, ys])
        arta = np.concatenate([artf, arts])
        sha = np.concatenate([shf, shs])
        wa = wfun(np.concatenate([dsf, dss]))
        if sep:
            dca = lgb.Dataset(Xa[:, c_idx], arta, weight=wa, categorical_feature=cat_c, free_raw_data=True, params=params)
        base_a = lgb.Dataset(Xa, np.nan_to_num(ya), categorical_feature=cat_idx, free_raw_data=True, params=params)
        base_a.construct()
        del Xa
        if not sep:
            dca = base_a.subset(np.arange(len(ya)))
            dca.construct()
            dca.set_label(arta)
            if wa is not None:
                dca.set_weight(wa)
        m_c = lgb.train(dict(params, objective="binary"), dca, num_boost_round=max(1, round(it_c * REFIT)))
        p_art = m_c.predict(Xe_c)
        eoba = np.concatenate([eobf, eobs])
        if args.eobt_comp:
            dca.construct()
            m_e = lgb.train(dict(params, objective="binary"), eobt_set(dca, arta, sha, np.concatenate([deobf, deobs]), eoba),
                            num_boost_round=max(1, round(it_e * REFIT)))
            p_e = np.where(np.isnan(deobe), 0.0, m_e.predict(Xe_c))
        msa = np.concatenate([msf, mss])
        ra = np.concatenate([yf - anf, ys - ans])
        if APALL:  # R13-F3: 공항별 재학습
            apa_a = np.concatenate([apa_f, apa_s])
            for a in sorted(it_ap):
                m_a = lgb.train(preg, reg_set(base_a, np.where((arta == 0) & ~sha & ~eoba & (apa_a == a))[0], ra, args.anchor), num_boost_round=max(1, round(it_ap[a] * REFIT)))
                sel = apa_e == a
                mu[sel] = ane[sel] + m_a.predict(Xe[sel])
                m_ap[a] = m_a
            m_r = m_ap[max(m_ap, key=lambda k: int((apa_a == k).sum()))]
        else:
            m_r = lgb.train(preg, reg_set(base_a, np.where((arta == 0) & ~sha & ~eoba & ~msa)[0], ra, args.anchor), num_boost_round=max(1, round(it_r * REFIT)))
            mu = ane + m_r.predict(Xe)
        if args.mu_sep and not APALL:
            m_r2 = lgb.train(preg, reg_set(base_a, np.where((arta == 0) & ~sha & ~eoba & msa)[0], ra, args.anchor), num_boost_round=max(1, round(it_r2 * REFIT)))
            mu = np.where(mse_, ane + m_r2.predict(Xe), mu)
        if it_d is not None:
            dpred = lgb.train(dict(params, objective="regression"), base_a, num_boost_round=max(1, round(it_d * REFIT))).predict(Xe)
        log(f"refit on all {len(ya)} fit rows with iters x {REFIT}")
    mu_n = p_e * np.nan_to_num(deobe) + (1 - p_e) * mu if args.eobt_comp else mu  # art 아닌 몫(성분이 꺼져 있으면 mu)
    pred = np.clip(p_art * dse + (1 - p_art) * mu_n, 0, None)
    # 3) G 그룹 덮어쓰기
    mu_g = mu[ge]
    g_pred, pn_g = g_override(P, Ge["d_sch"].to_numpy(), Ge["dd"].to_numpy(), mu_g, p_art[ge] if args.g_band_avg else None)
    pred_g = pred.copy()
    pred_g[ge] = np.clip(g_pred, 0, None)
    res = {"MVT_ID_mvt": meta["MVT_ID_mvt"], "pred": pred_g, "pred_noG": pred, "p_art": p_art, "mu": mu, "g": ge.astype(np.int8),
           "dsch": dse}
    iters = {"clf": it_c, "reg": it_r, "es": args.es}
    if args.refit:
        iters["refit_factor"] = REFIT
    if args.eobt_comp:
        res["p_eobt"] = p_e
        iters["eclf"] = it_e
    if args.mu_sep and not APALL:
        iters["reg2"] = it_r2
    if APALL:
        iters.update({f"reg_{a}": it for a, it in it_ap.items()})
    if args.anchor:
        res["anchor"] = ane
    if it_d is not None:
        res["direct"] = np.clip(dpred, 0, None)
        dg = dpred.copy()
        dg[ge] = g_pred - mu_g * pn_g + dpred[ge] * pn_g  # G 는 같은 기대값 식, 정상 몫만 direct 로
        res["direct_G"] = np.clip(dg, 0, None)
        iters["direct"] = it_d
    if labeled_eval:
        res["y"] = ye
        apv = meta["ap"].to_numpy()
        print(breakdown(pred_g, ye, apv, f"{args.tag} {args.setting} mixture+G (clipped)"))
        print(f"  mixture without G override {rmse(pred, ye):.2f} | G rows {int(ge.sum())} MSE contrib noG "
              f"{((pred[ge] - ye[ge]) ** 2).sum() / len(ye):.0f} -> G {((pred_g[ge] - ye[ge]) ** 2).sum() / len(ye):.0f}")
        if "direct" in res:
            print(f"  direct {rmse(res['direct'], ye):.2f} | direct+G {rmse(res['direct_G'], ye):.2f}")
        noise = (ye > 50000) & (dse < 20000)
        print(f"  without {int(noise.sum())} no-signal rows: mixture+G {rmse(pred_g[~noise], ye[~noise]):.2f}")
        if args.eobt_comp:
            eob_e = eob_all[eval_rows]
            print(f"  eobt rows in eval {int(eob_e.sum())} | mean p_eobt on them {p_e[eob_e].mean():.3f}, elsewhere {p_e[~eob_e].mean():.4f}")
        imp = pd.Series(m_c.feature_importance("gain"), index=[cols[j] for j in c_idx]).sort_values(ascending=False)
        print("  clf top gain:", (imp / imp.sum()).head(10).round(3).to_dict())
        imp = pd.Series(m_r.feature_importance("gain"), index=cols).sort_values(ascending=False)
        print("  reg top gain:", (imp / imp.sum()).head(15).round(3).to_dict())
        if args.setting == "val" and args.seed == 0:
            iters_path.write_text(json.dumps(iters))
    out = CACHE / f"{args.setting}_{args.tag}_s{args.seed}.parquet"
    pd.DataFrame(res).to_parquet(out, index=False)
    log(f"saved {out.name}")


if __name__ == "__main__":
    main()
