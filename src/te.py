# SPDX-License-Identifier: GPL-3.0-only
"""Target encodings: smoothed means of label-derived targets by group, out of fold by month.

For a training row the mean of its group (for example airport and stand) leaves out the row's own month; the other rows (the months
that a setting holds out and the rows to predict) get the mean over all training months. Only training labels enter (a departure to
predict has none). add_te returns the columns that src/model3.py and the other legs append to the feature table.

Original note (Korean):
학습 정답으로 만드는 그룹 통계(target encoding). 학습 행은 달 단위 out-of-fold, 나머지 행은 학습 달 전체로 낸다.

시험 행에는 BLOCK 이 없으므로 이 통계는 학습 데이터로만 계산한다(지시: 공항이나 스탠드별 간격 통계는 학습 데이터로만).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def combo_key(F: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """One integer key per row for the combination of the columns cols (category codes or integer values; missing values count as 0)."""
    key = np.zeros(len(F), dtype=np.int64)
    for c in cols:
        s = F[c]
        codes = s.cat.codes.to_numpy().astype(np.int64) if isinstance(s.dtype, pd.CategoricalDtype) else s.fillna(-1).to_numpy().astype(np.int64)
        codes = codes + 1  # 결측(-1) -> 0
        key = key * (int(codes.max()) + 1) + codes
    return key


def te_oof(key: np.ndarray, target: np.ndarray, month: np.ndarray, train_mask: np.ndarray, k: float = 20.0,
           prior: float | None = None) -> np.ndarray:
    """Smoothed mean of target per key, out of fold by month.

    Rows in train_mask get the mean over the training rows of the other months, the other rows the mean over all training rows. Rows
    whose target is NaN do not enter. Smoothing: k pseudo-rows at the prior (the mean of the training rows unless prior is given).

    Original note (Korean):
    target 이 NaN 인 행은 통계에서 뺀다. train_mask 행은 자기 달을 뺀 값, 나머지 행은 학습 달 전체 값.
    """
    use = train_mask & ~np.isnan(target)
    if prior is None:
        prior = float(np.nanmean(target[use]))
    d = pd.DataFrame({"key": key[use], "m": month[use], "t": target[use]})
    tot = d.groupby("key")["t"].agg(["sum", "count"])
    per = d.groupby(["key", "m"])["t"].agg(["sum", "count"])
    out = np.full(len(key), np.nan)
    # 학습 밖 행: 전체 통계
    s_all = pd.Series(key).map(tot["sum"]).fillna(0).to_numpy()
    n_all = pd.Series(key).map(tot["count"]).fillna(0).to_numpy()
    out[:] = (s_all + prior * k) / (n_all + k)
    # 학습 행: 자기 달을 뺀다
    idx = np.where(train_mask)[0]
    mi = pd.MultiIndex.from_arrays([key[idx], month[idx]])
    s_m = per["sum"].reindex(mi).fillna(0).to_numpy()
    n_m = per["count"].reindex(mi).fillna(0).to_numpy()
    out[idx] = (s_all[idx] - s_m + prior * k) / (n_all[idx] - n_m + k)
    return out.astype("float32")


def add_te(F: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    """Target-encoding columns for the feature table F, with labels taken only from the rows in train_mask (the setting's fitting rows).

    Four targets per group: the share of schedule copies among departures more than 30 minutes late (te_artlate_*), the label clipped to
    0 to 3,600 s (te_ycl_*), the label minus the NM proxy MVT - AOBT_3, that is AOBT_3 - BLOCK, clipped to +-1,800 s (te_off_*), and for
    some groups the share of labels of an hour or more (te_big_*). Groups: the airport with stand, operator, destination, aircraft type,
    runway and some combinations; with feature set 2.5 the flight-number prefix groups give way to destination groups.

    Original note (Korean):
    train_mask: 이 설정에서 정답을 쓸 수 있는 행(검증 설정이면 1, 7월 제외, 최종이면 2025 전체).
    """
    y = F["y"].to_numpy().astype(float)
    dsch = F["d_sch"].to_numpy().astype(float)
    daobt = F["d_aobt"].to_numpy().astype(float)
    month = F["src"].to_numpy().astype(int)
    art = (np.abs(y - dsch) <= 60).astype(float)
    late = dsch > 1800
    t_art_late = np.where(late, art, np.nan)            # 늦은 출발 중 BLOCK=SCHED 비율
    t_ycl = np.clip(y, 0, 3600)                         # 정상 택시 수준
    t_off = np.where(np.isnan(daobt), np.nan, np.clip(y - daobt, -1800, 1800))  # BLOCK - AOBT3 (부호 반대) 의 그룹 평균
    t_big = (y >= 3600).astype(float)
    t_ycl = np.where(np.isnan(y), np.nan, t_ycl)
    t_art_late = np.where(np.isnan(y), np.nan, t_art_late)
    t_big = np.where(np.isnan(y), np.nan, t_big)
    out = {}
    specs = {
        "ap_pfx_nm": ["ap", "prefix", "nm"],
        "ap_stand": ["ap", "stand"],
        "ap_op": ["ap", "op"],
        "ap_ades": ["ap", "ades"],
        "ap_actype": ["ap", "actype"],
        "ap_rwy": ["ap", "rwy"],
        "ap_stand_pfx": ["ap", "stand", "prefix"],
        "ap_stand_rwy": ["ap", "stand", "rwy"],
    }
    if "flight" in F.columns:  # 피처 v2.3 부터: 같은 편명(매일 도는 편)의 택시 수준과 BLOCK-AOBT3 습관
        specs["ap_flight"] = ["ap", "flight"]
    if "prefix" not in F.columns:  # 피처 v2.5: 편명 접두사가 없으니 그 자리를 (공항, 목적지, NM 매칭)과 (공항, 스탠드, 목적지)로
        specs.pop("ap_pfx_nm")
        specs.pop("ap_stand_pfx")
        specs["ap_ades_nm"] = ["ap", "ades", "nm"]
        specs["ap_stand_ades"] = ["ap", "stand", "ades"]
    for name, cols in specs.items():
        key = combo_key(F, cols)
        out[f"te_artlate_{name}"] = te_oof(key, t_art_late, month, train_mask, k=10.0)
        out[f"te_ycl_{name}"] = te_oof(key, t_ycl, month, train_mask, k=30.0)
        out[f"te_off_{name}"] = te_oof(key, t_off, month, train_mask, k=30.0)
        if name in ("ap_pfx_nm", "ap_ades_nm", "ap_stand"):
            out[f"te_big_{name}"] = te_oof(key, t_big, month, train_mask, k=30.0)
    return pd.DataFrame(out, index=F.index)
