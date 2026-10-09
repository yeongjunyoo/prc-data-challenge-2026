# SPDX-License-Identifier: GPL-3.0-only
"""R3 측정: 복사 기록(BLOCK = 기준 시각 ±6.5초) 다중 분류기를 자 밖 달로 학습해 자(1, 7월)에서 얼마나 가르나, 321 위에 섞으면 얼마인가.
학습: 2025 의 짝수 달 여섯(2, 4, 6, 8, 10, 12월), 입력 칸은 feats_v25 의 입력 칸 전부(y, src, MVT_ID 뺌). 라벨은 학습 정답에서만.
평가: 자 두 달에서 칸별 AUC, 그리고 321 잔차 위의 선형 섞기 Σ_c β_c q_c (d_c − pred)(날 묶음 10겹, src/adsb_corr.day_folds).
    .venv/Scripts/python.exe -X utf8 research/r3_structure/p2_copy_clf.py [train|eval]
"""
from __future__ import annotations

import ctypes
import json
import sys
from ctypes import wintypes
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if sys.platform == "win32":
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)

W = 6.5
CLS = ["none", "sch", "lobt", "eobt", "aobt"]
CAT = ["ap", "rwy", "stand", "actype", "op", "mseg", "ftype", "wtc", "frule", "ades_reg", "ades"]


def labels(F: pd.DataFrame) -> np.ndarray:
    y = F["y"].to_numpy(float)
    h = {k: np.abs(y - F[c].to_numpy(float)) <= W for k, c in (("sch", "d_sch"), ("lobt", "d_lobt"), ("eobt", "d_eobt"), ("aobt", "d_aobt"))}
    lab = np.zeros(len(F), np.int8)
    lab[h["aobt"]] = 4
    lab[h["eobt"]] = 3
    lab[h["lobt"]] = 2
    lab[h["sch"]] = 1
    return lab


def load(srcs):
    """feats_v25 의 srcs 행: (X float32 행렬, 칸 이름, 범주 칸, 곁 칸 DataFrame[MVT_ID, y, d_sch, d_lobt, d_eobt, d_aobt]).
    범주 칸 코드는 전체 행(2.43M)의 pandas Categorical 코드라 학습, 자, 시험이 같은 코드다."""
    import pyarrow as pa
    path = ROOT / "cache" / "feats_v25.parquet"
    schema = pq.read_schema(path)
    cat = [c for c in CAT if c in schema.names]
    src_all = pq.read_table(path, columns=["src"]).column("src").to_numpy()
    keep = np.isin(src_all, srcs)
    n = int(keep.sum())
    cols = [c for c in schema.names if c not in ("MVT_ID_mvt", "src", "y")]
    X = np.empty((n, len(cols)), dtype="float32")
    for j, c in enumerate(cols):
        col = pq.read_table(path, columns=[c]).column(c)
        if c in cat:
            v = col.to_pandas()
            v = v.astype("category") if v.dtype.name != "category" else v
            codes = v.cat.codes.to_numpy().astype("float32")
            codes[codes < 0] = np.nan
            X[:, j] = codes[keep]
        else:
            X[:, j] = col.to_numpy(zero_copy_only=False).astype("float32")[keep]
        del col
        pa.default_memory_pool().release_unused()
    side = pq.read_table(path, columns=["MVT_ID_mvt", "y", "d_sch", "d_lobt", "d_eobt", "d_aobt"], filters=[("src", "in", srcs)]).to_pandas()
    assert len(side) == n
    return X, cols, cat, side


def train():
    import lightgbm as lgb
    X, cols, cat, side = load([2, 5, 8, 11])
    lab = labels(side)
    print("학습 행", len(side), "라벨 비율", {CLS[i]: round(float((lab == i).mean()), 4) for i in range(5)}, flush=True)
    params = dict(objective="multiclass", num_class=5, learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=3, seed=0, max_cat_to_onehot=8,
                  cat_smooth=20)
    ds = lgb.Dataset(X, lab, feature_name=cols, categorical_feature=cat, free_raw_data=True, params={"max_bin": 127, "verbose": -1})
    ds.construct()
    del X
    m = lgb.train(params, ds, num_boost_round=200)
    m.save_model(str(OUT / "p2_copy_clf.txt"))
    imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("gain top", (imp / imp.sum()).head(20).round(4).to_dict())


def predict_rows(srcs, fname):
    import lightgbm as lgb
    m = lgb.Booster(model_file=str(OUT / "p2_copy_clf.txt"))
    X, cols, cat, F = load(srcs)
    assert cols == m.feature_name()
    P = m.predict(X)
    del X
    out = pd.DataFrame({"MVT_ID_mvt": F["MVT_ID_mvt"].to_numpy()})
    for i, c in enumerate(CLS):
        out[f"q_{c}"] = P[:, i].astype("float32")
    if F["y"].notna().any():
        out["lab"] = labels(F)
    out.to_parquet(OUT / fname)
    return out


def evaluate():
    import src.adsb_corr as ac
    from sklearn.metrics import roc_auc_score
    Q = predict_rows([1, 7], "p2_q_ruler.parquet").set_index("MVT_ID_mvt")
    S = pd.read_parquet(OUT / "e1_rows.parquet")  # 깨끗한 자(세 행 뺌)
    R = pd.read_parquet(ROOT / "research" / "r2_errors" / "d1_rows.parquet")  # 원래 자 전체
    b = pd.read_parquet(ROOT / "cache" / "val_m6r25sgcbxradsbgeow_s0.parquet").set_index("MVT_ID_mvt").reindex(R.index)
    R["p"] = b["pred"].to_numpy()
    R["p_art321"] = b["p_art"].to_numpy()
    R = R.join(Q, how="left")
    F = pq.read_table(ROOT / "cache" / "feats_v25.parquet", columns=["MVT_ID_mvt", "d_iobt"], filters=[("src", "in", [1, 7])]).to_pandas().set_index("MVT_ID_mvt")
    R = R.join(F, how="left")
    res = {}
    for i, c in enumerate(CLS[1:], start=1):
        yy = (R["lab"] == i).to_numpy()
        res[f"auc_{c}"] = round(float(roc_auc_score(yy, R[f"q_{c}"])), 4)
        res[f"rate_{c}"] = round(float(yy.mean()), 4)
    sch = (R["lab"] == 1).to_numpy()
    res["auc_sch_by_p_art321"] = round(float(roc_auc_score(sch, R["p_art321"])), 4)
    # 선형 섞기
    y = R["y"].to_numpy(float)
    p = R["p"].to_numpy(float)
    D = {"sch": R["dsch"].to_numpy(float), "lobt": R["d_lobt"].to_numpy(float), "eobt": R["d_eobt"].to_numpy(float), "aobt": R["d_aobt"].to_numpy(float)}
    X = np.column_stack([np.nan_to_num(R[f"q_{c}"].to_numpy(float) * (D[c] - p), nan=0.0) for c in CLS[1:]])
    gmask = R["g"].to_numpy(bool)
    X[gmask] = 0.0
    days = R["day"].astype(str).to_numpy()
    fo = ac.day_folds(sorted(set(days)), 10, 0)
    fold = pd.Series(days).map(fo).to_numpy()
    corr = np.zeros(len(R))
    betas = []
    clean = ~((~gmask) & (y >= 50000))
    for k in range(10):
        tr = (fold != k) & clean & np.isfinite(y)
        A = X[tr]
        t = np.clip(y[tr] - p[tr], -3600, 3600)
        beta = np.linalg.solve(A.T @ A + 1e3 * np.eye(A.shape[1]), A.T @ t)
        betas.append(beta)
        corr[fold == k] = X[fold == k] @ beta
    new = p + corr
    N = len(R)
    res["beta_mean"] = dict(zip(CLS[1:], np.round(np.mean(betas, axis=0), 3).tolist()))
    res["rmse_321"] = round(float(np.sqrt(((p - y) ** 2).mean())), 3)
    res["rmse_mix"] = round(float(np.sqrt(((new - y) ** 2).mean())), 3)
    res["dmse_raw"] = round(float(((new - y) ** 2).mean() - ((p - y) ** 2).mean()), 1)
    res["dmse_clean"] = round(float((((new - y) ** 2)[clean].sum() - ((p - y) ** 2)[clean].sum()) / N), 1)
    res["per_ap_dmse_clean"] = {ap: round(float((((new - y) ** 2)[clean & (R["ap"] == ap).to_numpy()].sum() - ((p - y) ** 2)[clean & (R["ap"] == ap).to_numpy()].sum()) / N), 1)
                                for ap in sorted(R["ap"].unique())}
    pd.DataFrame({"MVT_ID_mvt": R.index.to_numpy(), "corr": corr}).to_parquet(OUT / "p2_mix_corr.parquet")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    json.dump(res, open(OUT / "p2_copy_clf_eval.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    {"train": train, "eval": evaluate}[sys.argv[1]]()
