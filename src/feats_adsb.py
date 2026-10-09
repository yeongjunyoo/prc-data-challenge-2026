# SPDX-License-Identifier: GPL-3.0-only
"""피처 v2.7 = v2.5 에 ADS-B 지상 궤적 칸(ab_*, adsb.lol globe_history, Open Database License 1.0)을 붙인 판.

ADS-B 칸은 research/adsb/build_feats2.py 의 출발편 표(--adsb, 예 adsb_v2)에서 오고, 정의는 보정기(src/adsb_corr.py)와 같다.
  ab_matched           1 = 그날 ADS-B 이륙 사건과 맞음, 0 = 그날 자료는 있는데 안 맞음, NaN = 그날 자료가 아직 없음
  ab_fm_seen, ab_fm_ok 첫 움직임을 봤는가, 그 움직임의 시작을 봤는가(맞은 행만, 나머지 NaN)
  ab_est_mvt, ab_est_to, ab_est_first, ab_est_mvt_last, ab_dt_to   MVT − 첫 움직임, 이륙 − 첫 움직임, MVT − 첫 지상 점,
                       MVT − 마지막 긴 멈춤 뒤 움직임, ADS-B 이륙 − MVT
  ab_fm_m_sch, ab_first_m_sch      d_sch − ab_est_mvt, d_sch − ab_est_first(보정기의 fm_minus_sched, first_minus_sched)
  ab_fm_m_aobt, ab_first_m_aobt    d_aobt − ab_est_mvt, d_aobt − ab_est_first. 보정기의 fm_minus_pred, first_minus_pred 는 바탕 예측과의
                       차인데, 주 모형에는 바탕 예측이 없어 회귀의 가장 큰 입력인 d_aobt(MVT − AOBT_3)와의 차로 바꿨다.
  ab_first_gs ... ab_share_mlat     사건 칸 16개(ev_* 그대로)
맞지 않은 행은 ab_matched 만 0 이고 나머지는 NaN 이다.

--mode 는 행의 덮임(커버리지)을 어떻게 둘지 정한다. 칸(공항, 달력 달 1 또는 7)마다 2025 자의 비율(맞음 m25, 맞은 행 가운데 첫 움직임 봄
f25)과 2026 시험의 비율(m26, f26)을 입력 표에서 세고, 행마다 MVT_ID 의 고정 해시로 고른다(무작위 씨앗 없음, 다시 만들어도 같다).
  raw(기본): 아무 행도 가리지 않는다(시험은 2026 의 덮임 그대로).
  sim      : 자 행(2025 의 1, 7월)만, 칸 비율이 2026 보다 높으면 2026 비율까지 내린다(덮임 표류 모의, 자에서만 쓰는 판). 가린 맞은 행은
             맞지 않은 행처럼 되고, 첫 움직임만 가린 행은 FM_COLS 만 빈다.
  harmonize: 시험 행만, 칸 비율이 2025 보다 높으면 2025 비율까지 내린다(맞음은 m25/m26 만 남기고, 남은 행의 첫 움직임 칸은 f25/f26 만
             남긴다). 정답은 쓰지 않는다. 첫 움직임 칸만 가린 행은 실제 자료에 드문 조합(주기장에서 멈춘 첫 점은 있는데 첫 움직임은 없음)이
             되므로, 쓰기 전에 그 몫을 따로 본다.
harmonize 판의 시험 행과 sim 판의 자 행은 칸마다 min(2025, 2026) 비율이 되어, sim 판으로 잰 자 값이 harmonize 판 시험의 로컬 짝이다.

    python -m src.feats_adsb feats_v27 --adsb adsb_v2 --mode raw
출력: cache/<이름>.parquet(feats_v25 의 칸과 행 순서 그대로, 뒤에 ab_* 28칸), cache/<이름>.json(칸 목록, 입력 해시, 칸별 비율과 가린 수).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, ".")
from src.load import CACHE  # noqa: E402

BASE = "feats_v25"
EV = ["first_gs", "first_dist_m", "first_stationary", "stand_seen", "stand_dur", "fm_gap", "n_mid_stops", "mid_stop_s",
      "mid_stop_max", "n_long_stops", "fm_last_gap", "last_long_dur", "to_gap", "n_pts", "max_gap_after_fm", "share_mlat"]
EST = ["est_mvt", "est_to", "est_first", "est_mvt_last", "dt_to"]
COLS = (["ab_matched", "ab_fm_seen", "ab_fm_ok"] + [f"ab_{c}" for c in EST]
        + ["ab_fm_m_sch", "ab_first_m_sch", "ab_fm_m_aobt", "ab_first_m_aobt"] + [f"ab_{c}" for c in EV])
# 첫 움직임이 안 보이면 비는 칸(adsb_v2 에서 확인: 맞은 행 가운데 첫 움직임 없는 행의 채움 비율 1% 미만)
FM_COLS = ["ab_est_mvt", "ab_est_to", "ab_est_mvt_last", "ab_fm_m_sch", "ab_fm_m_aobt", "ab_stand_dur", "ab_fm_gap",
           "ab_fm_last_gap", "ab_last_long_dur"]
MONTHS = (1, 7)


def sha16(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def u01(ids: np.ndarray, salt: int) -> np.ndarray:
    """MVT_ID 의 고정 해시(splitmix64) → [0, 1)."""
    with np.errstate(over="ignore"):
        z = ids.astype(np.int64).view(np.uint64) + np.uint64(salt) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def adsb_columns(F: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    """F(MVT_ID_mvt, d_sch, d_aobt) 행 순서의 ab_* 칸. A 에 없는 행(그날 자료 없음)은 ab_matched 까지 NaN."""
    idx = pd.Index(A["MVT_ID_mvt"].to_numpy()).get_indexer(F["MVT_ID_mvt"].to_numpy())
    have = idx >= 0
    a = A.iloc[np.where(have, idx, 0)].reset_index(drop=True)
    matched = a["matched"].to_numpy(dtype=bool) & have
    out = pd.DataFrame(index=F.index)
    out["ab_matched"] = np.where(have, matched.astype(np.float64), np.nan)
    out["ab_fm_seen"] = np.where(matched, a["fm_seen"].to_numpy(dtype=np.float64), np.nan)
    out["ab_fm_ok"] = np.where(matched, a["fm_ok"].to_numpy(dtype=np.float64), np.nan)
    for c in EST:
        out[f"ab_{c}"] = np.where(matched, a[c].to_numpy(dtype=np.float64), np.nan)
    dsch, daobt = F["d_sch"].to_numpy(dtype=np.float64), F["d_aobt"].to_numpy(dtype=np.float64)
    out["ab_fm_m_sch"] = dsch - out["ab_est_mvt"].to_numpy()
    out["ab_first_m_sch"] = dsch - out["ab_est_first"].to_numpy()
    out["ab_fm_m_aobt"] = daobt - out["ab_est_mvt"].to_numpy()
    out["ab_first_m_aobt"] = daobt - out["ab_est_first"].to_numpy()
    for c in EV:
        out[f"ab_{c}"] = np.where(matched, a[f"ev_{c}"].to_numpy(dtype=np.float64), np.nan)
    return out[COLS].astype("float32")


def cell_rates(ap: np.ndarray, year: np.ndarray, month: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """칸(공항, 해, 달)의 행 수와 맞음 비율 m, 맞은 행 가운데 첫 움직임 봄 비율 f(그날 자료가 있는 행만)."""
    ok = X["ab_matched"].notna().to_numpy()
    d = pd.DataFrame({"ap": ap[ok], "year": year[ok], "month": month[ok], "m": X["ab_matched"].to_numpy()[ok],
                      "f": X["ab_fm_seen"].to_numpy()[ok]})
    g = d.groupby(["ap", "year", "month"])
    return pd.DataFrame({"n": g.size(), "m": g["m"].mean(), "f": g["f"].mean()})


def mask(X: pd.DataFrame, rows: np.ndarray, keep_m: np.ndarray, keep_f: np.ndarray, ids: np.ndarray) -> tuple[int, int]:
    """rows 가운데 맞은 행을 keep_m 확률로만 남기고(나머지는 맞지 않은 행처럼), 남은 첫 움직임 행의 FM_COLS 는 keep_f 확률로만 남긴다."""
    m = rows & (X["ab_matched"].to_numpy() == 1)
    drop_m = m & (u01(ids, 1) >= keep_m)
    X.loc[drop_m, [c for c in COLS if c != "ab_matched"]] = np.nan
    X.loc[drop_m, "ab_matched"] = 0.0
    f = rows & (X["ab_fm_seen"].to_numpy() == 1)
    drop_f = f & (u01(ids, 2) >= keep_f)
    X.loc[drop_f, FM_COLS] = np.nan
    X.loc[drop_f, ["ab_fm_seen", "ab_fm_ok"]] = 0.0
    return int(drop_m.sum()), int(drop_f.sum())


def main():
    pa_ = argparse.ArgumentParser()
    pa_.add_argument("name")
    pa_.add_argument("--adsb", default="adsb_v2")
    pa_.add_argument("--mode", default="raw", choices=["raw", "sim", "harmonize"])
    a = pa_.parse_args()
    t0 = dt.datetime.now()
    T = pq.read_table(CACHE / f"{BASE}.parquet")
    assert not any(c.startswith("ab_") for c in T.column_names)
    F = T.select(["MVT_ID_mvt", "src", "ap", "d_sch", "d_aobt"]).to_pandas()
    A = pd.read_parquet(CACHE / f"{a.adsb}.parquet")
    X = adsb_columns(F, A)
    ids = F["MVT_ID_mvt"].to_numpy()
    src = F["src"].to_numpy()
    tm = pd.read_parquet(CACHE / "all.parquet", columns=["MVT_ID_mvt", "t_mvt"]).set_index("MVT_ID_mvt")["t_mvt"].reindex(ids)
    ts = pd.to_datetime(tm.to_numpy(), unit="s")
    year, month, ap = ts.year.to_numpy(), ts.month.to_numpy(), F["ap"].astype(str).to_numpy()
    R = cell_rates(ap, year, month, X)
    cells = {}
    for apc in sorted(set(ap)):
        for mo in MONTHS:
            r25 = R.loc[(apc, 2025, mo)] if (apc, 2025, mo) in R.index else None
            r26 = R.loc[(apc, 2026, mo)] if (apc, 2026, mo) in R.index else None
            if r25 is None or r26 is None:
                continue
            cells[(apc, mo)] = dict(n25=int(r25.n), m25=float(r25.m), f25=float(r25.f) if r25.m > 0 else 0.0,
                                    n26=int(r26.n), m26=float(r26.m), f26=float(r26.f) if r26.m > 0 else 0.0)
    keep_m, keep_f = np.ones(len(F)), np.ones(len(F))
    target = (src == 0) if a.mode == "harmonize" else np.isin(src, MONTHS) if a.mode == "sim" else np.zeros(len(F), bool)
    for (apc, mo), c in cells.items():
        sel = target & (ap == apc) & (month == mo)
        if a.mode == "harmonize":
            km = min(1.0, c["m25"] / c["m26"]) if c["m26"] > 0 else 1.0
            kf = min(1.0, c["f25"] / c["f26"]) if c["f26"] > 0 else 1.0
        else:
            km = min(1.0, c["m26"] / c["m25"]) if c["m25"] > 0 else 1.0
            kf = min(1.0, c["f26"] / c["f25"]) if c["f25"] > 0 else 1.0
        keep_m[sel], keep_f[sel] = km, kf
        c.update(keep_m=round(km, 4), keep_f=round(kf, 4))
    dm, df_ = mask(X, target, keep_m, keep_f, ids) if a.mode != "raw" else (0, 0)
    for (apc, mo), c in cells.items():
        for yr, rows in ((2025, (src == mo)), (2026, (src == 0) & (month == mo))):
            s = rows & (ap == apc) & X["ab_matched"].notna().to_numpy()
            c[f"after_m{yr % 100}"] = round(float(X["ab_matched"].to_numpy()[s].mean()), 4) if s.any() else None
            c[f"after_fm{yr % 100}"] = round(float((X["ab_fm_seen"].to_numpy()[s] == 1).mean()), 4) if s.any() else None
    for c in COLS:
        T = T.append_column(c, pa.array(X[c].to_numpy(), type=pa.float32()))
    out = CACHE / f"{a.name}.parquet"
    pq.write_table(T, out, compression="snappy")
    have = X["ab_matched"].notna().to_numpy()
    fill = {f"{lab}": round(float(X["ab_est_first"].notna().to_numpy()[rows].mean()), 4)
            for lab, rows in (("ruler", np.isin(src, MONTHS)), ("test", src == 0))}
    meta = dict(name=a.name, base=BASE, base_sha16=sha16(CACHE / f"{BASE}.parquet"), adsb=a.adsb, adsb_sha16=sha16(CACHE / f"{a.adsb}.parquet"),
                mode=a.mode, columns=COLS, fm_columns=FM_COLS, rows=int(len(F)), rows_with_day_data=int(have.sum()),
                training_rows_with_day_data=round(float(have[(src != 0) & ~np.isin(src, MONTHS)].mean()), 4),
                masked_matched=dm, masked_fm=df_, est_first_fill=fill,
                cells={f"{k[0]} {k[1]:02d}": v for k, v in cells.items()}, sha16=sha16(out),
                seconds=round((dt.datetime.now() - t0).total_seconds(), 1),
                written_kst=dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST"))
    (CACHE / f"{a.name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("cells", "columns", "fm_columns")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
