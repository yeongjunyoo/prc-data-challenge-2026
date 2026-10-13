# SPDX-License-Identifier: GPL-3.0-only
"""Feature table of the departures, one row per departure, built only from columns that the movements to predict also have.

A departure's BLOCK_TIME and TAXITIME (the label) are never used as inputs; the arrivals' block times are, since the input to predict
has them. Feature groups: identity and category columns (airport, runway, stand, aircraft type, operator, market segment,
destination), the take-off time minus the schedule and the NM times (d_sch, d_aobt, d_eobt, d_iobt, d_lobt and their differences),
local clock and calendar (needs the airport's time zone in TZ), day and duplicate-record statistics, traffic counts and neighbour
averages (congestion), the sequence of movements at the stand (stand_seq) and METAR weather (metar_feats; optional files
data/metar/<ICAO>.csv). Since version 2.4 every window statistic is computed inside the UTC month of the movement, so that the
training months and the months to predict see the same amount of context (docs/DESIGN.md section 5).

    python src/feats.py [name]     # writes cache/<name>.parquet (default name feats)

The frozen names feats_v2 to feats_v26 are not rebuilt (build refuses them): build under another name and rename the file, as
research/f3/build_feats_final.py does for feats_v25. Every name that can be built gives version 2.5, except feats_mask_v26, which gives
version 2.6, the version of the frozen feats_v26 (src/mask_test.py builds it).
To add an airport, add its time zone to TZ; otherwise its local-time columns stay empty and no METAR file is read for it.

Original note (Korean):
출발편 피처. 시험 때 보이는 칸만 쓴다(출발편의 BLOCK 과 TAXITIME 은 어디에도 넣지 않는다).

혼잡도와 주변 통계는 같은 칸막이 안의 행끼리 만든다. v2.4 부터 칸막이는 이륙(착륙) 시각의 UTC 달이다:
학습, 자(2025-01, 07), 시험(2026-01, 07)이 모두 자기 달 안의 사건만 본다(달 경계 가용성을 맞춤).
METAR 는 칸마다 이륙 전 METAR_ASOF_H(3)시간 안의 마지막 값 있는 보고를 쓴다.
v2.5 는 v2.4 에서 편명 파생 칸 prefix, fl_len 을 뺐다(편명이 비면 NaN 이 되는 칸).
v2.6 은 V26_NAMES 이름으로 만들 때만 켠다: 피처 전에 계획 시계를 2025 뜻으로 맞추고(plan_fix) 새 칸 다섯을 붙인다(v26_feats).
v25 이하 이름과 feats 로 만들면 v2.5 그대로다.
결과는 cache/feats.parquet(python src/feats.py [이름] 이면 cache/<이름>.parquet). 고정 판: feats_v2, v21, v22, v23, v24, v25, v26
(가림 시험 notes/mask_test_feats_<판>.json).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.load import CACHE, DATA as ROOT_DATA, load_all  # noqa: E402

TZ = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London", "EHAM": "Europe/Amsterdam",
      "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid", "LFPG": "Europe/Paris", "LIRF": "Europe/Rome",
      "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul"}
CAT_FEATS = ["ap", "rwy", "stand", "actype", "op", "prefix", "mseg", "ftype", "wtc", "frule", "ades_reg", "ades"]
# v2.4 METAR as-of 창(시간). 시험 커버리지를 보기 전에 3시간으로 고정했다.
METAR_ASOF_H = 3.0
# v2.6 을 켜는 이름(고정 판과 그 가림 판). 다른 이름은 v2.5 로 만든다
V26_NAMES = ("feats_v26", "feats_mask_v26")
# v2.6 제빙 위치(연구 노트 2026-10-04 2.4 절): 2 = 오프블록 뒤 원격 제빙(택시아웃 안), 1 = 섞임, 0 = 주로 주기장(오프블록 전)
DEICE_CLS = {"EDDM": 2, "EHAM": 2, "LFPG": 2, "LTFM": 2, "LEMD": 2, "EDDF": 1, "LSZH": 1, "LEBL": 1, "EGLL": 0, "LIRF": 0}


def local_parts(t: pd.Series, ap: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local calendar day (days since 2025-01-01), seconds since local midnight and weekday (0 = Monday) of UTC epoch seconds t.

    The time zone is that of each row's airport ap (TZ). Rows of airports that are not in TZ, and rows without t, get NaN.

    Original note (Korean):
    현지 날짜(2025-01-01 기준 일수), 현지 하루 중 초, 요일.
    """
    day = np.full(len(t), np.nan)
    tod = np.full(len(t), np.nan)
    dow = np.full(len(t), np.nan)
    tv = t.to_numpy()
    apv = ap.astype(str).to_numpy()
    for a, z in TZ.items():
        m = (apv == a) & ~np.isnan(tv)
        if m.any():
            ts = pd.to_datetime(pd.Series(tv[m]), unit="s", utc=True).dt.tz_convert(z).dt.tz_localize(None)
            norm = ts.dt.normalize()
            day[m] = ((norm - pd.Timestamp("2025-01-01")).dt.days).to_numpy()
            tod[m] = ((ts - norm).dt.total_seconds()).to_numpy()
            dow[m] = ts.dt.dayofweek.to_numpy()
    return day, tod, dow


def _cat(s: pd.Series) -> pd.Series:
    return s.astype(str).replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan}).astype("category")


def window_count(sorted_t: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Number of events of the sorted array sorted_t in the window [lo, hi), for each pair of bounds.

    Original note (Korean):
    [lo, hi) 안의 사건 수.
    """
    return np.searchsorted(sorted_t, hi, side="left") - np.searchsorted(sorted_t, lo, side="left")


def window_sum(sorted_t: np.ndarray, csum: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Sum of a quantity over the events of sorted_t in [lo, hi), from its cumulative sum csum (csum[0] = 0, one more entry than sorted_t)."""
    return csum[np.searchsorted(sorted_t, hi, side="left")] - csum[np.searchsorted(sorted_t, lo, side="left")]


def congestion(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    """Traffic around each departure's take-off at its airport, counted inside the same UTC month (column _part).

    Counts of departures and arrivals in windows before and after the take-off (5, 15 and 60 minutes; 10 and 30 minutes on the same
    runway), the gaps to the previous and next take-off on the runway, take-offs during the departure's own taxi (from AOBT_3 or from
    the schedule), aircraft taxiing at the take-off and at AOBT_3, scheduled departures within 30 minutes, and averages over the
    neighbouring departures (take-off minus AOBT_3, take-off minus schedule) and arrivals (taxi-in times, which the input to predict
    also has). Returns float32 columns indexed like dep.

    Original note (Korean):
    같은 공항, 같은 칸막이(_part: v2.4 부터 이륙, 착륙 시각의 UTC 달) 안에서만 센다.

    시험은 2026-01, 07 만 있어 달 경계 너머 사건이 없다. 학습과 자도 같은 창(자기 달)만 보게 해 가용성을 맞춘다.
    """
    out = pd.DataFrame(index=dep.index, dtype="float32")
    cols = {}
    for name in ["c_dep_m15", "c_dep_p15", "c_dep_m60", "c_dep_p60", "c_arr_m15", "c_arr_p15", "c_arr_m60", "c_arr_p60",
                 "c_dep_rwy_m30", "c_dep_rwy_p30", "c_takeoff_in_taxi_aobt", "c_takeoff_in_taxi_sch", "taxiing_now",
                 "queue_at_aobt", "nb_daobt_m60", "nb_dsch_m60", "nb_taxiin_m60", "nb_taxiin_m15p15",
                 "nb_daobt_p30", "c_sched_m30p30", "nb_dsch_p30", "c_dep_rwy_m10", "rwy_gap_prev", "rwy_gap_next", "c_dep_m5"]:
        cols[name] = np.full(len(dep), np.nan, dtype="float64")
    period_dep = dep["_part"].to_numpy()
    period_arr = arr["_part"].to_numpy()
    for ap in dep["ADEP_mvt"].cat.categories:
        for per in np.unique(period_dep):
            md = ((dep["ADEP_mvt"] == ap).to_numpy()) & (period_dep == per)
            ma = ((arr["ADES_mvt"] == ap).to_numpy()) & (period_arr == per)
            if not md.any():
                continue
            d = dep.loc[md]
            a = arr.loc[ma]
            t = d["t_mvt"].to_numpy()
            dm = np.sort(t)
            am = np.sort(a["t_mvt"].to_numpy())
            idx = np.where(md)[0]
            cols["c_dep_m15"][idx] = window_count(dm, t - 900, t)
            cols["c_dep_p15"][idx] = window_count(dm, t + 1, t + 901)
            cols["c_dep_m60"][idx] = window_count(dm, t - 3600, t)
            cols["c_dep_p60"][idx] = window_count(dm, t + 1, t + 3601)
            cols["c_arr_m15"][idx] = window_count(am, t - 900, t)
            cols["c_arr_p15"][idx] = window_count(am, t, t + 900)
            cols["c_arr_m60"][idx] = window_count(am, t - 3600, t)
            cols["c_arr_p60"][idx] = window_count(am, t, t + 3600)
            # 같은 활주로 출발
            rw = d["RUNWAY_mvt"].astype(str).fillna("nan").to_numpy()
            tmp = np.full(len(d), np.nan)
            tmp2 = np.full(len(d), np.nan)
            tmp3 = np.full(len(d), np.nan)
            gp = np.full(len(d), np.nan)
            gn = np.full(len(d), np.nan)
            for r in np.unique(rw):
                mr = rw == r
                rs = np.sort(t[mr])
                tmp[mr] = window_count(rs, t[mr] - 1800, t[mr])
                tmp2[mr] = window_count(rs, t[mr] + 1, t[mr] + 1801)
                tmp3[mr] = window_count(rs, t[mr] - 600, t[mr])
                lo = np.searchsorted(rs, t[mr], side="left")
                hi = np.searchsorted(rs, t[mr], side="right")
                gp[mr] = np.where(lo > 0, t[mr] - rs[np.maximum(lo - 1, 0)], np.nan)
                gn[mr] = np.where(hi < len(rs), rs[np.minimum(hi, len(rs) - 1)] - t[mr], np.nan)
            cols["c_dep_rwy_m30"][idx] = tmp
            cols["c_dep_rwy_p30"][idx] = tmp2
            cols["c_dep_rwy_m10"][idx] = tmp3
            cols["rwy_gap_prev"][idx] = gp
            cols["rwy_gap_next"][idx] = gn
            cols["c_dep_m5"][idx] = window_count(dm, t - 300, t)
            # 내 택시 구간(AOBT3 또는 SCHED ~ 이륙) 동안 이륙한 다른 출발 수
            ao = d["t_aobt3"].to_numpy()
            sc = d["t_sch"].to_numpy()
            ok = ~np.isnan(ao)
            v = np.full(len(d), np.nan)
            v[ok] = window_count(dm, np.minimum(ao[ok], t[ok]), t[ok])
            cols["c_takeoff_in_taxi_aobt"][idx] = v
            cols["c_takeoff_in_taxi_sch"][idx] = window_count(dm, np.minimum(sc, t), t)
            # 지금 택시 중인 항공기(AOBT3 <= x < 이륙). AOBT3 가 없는 출발은 이륙 15분 전으로 세어 누적 어긋남을 막는다
            aos = np.sort(np.where(ok, np.minimum(ao, t), t - 900))
            cols["taxiing_now"][idx] = np.searchsorted(aos, t, side="right") - np.searchsorted(dm, t, side="left")
            v = np.full(len(d), np.nan)
            v[ok] = np.searchsorted(aos, ao[ok], side="right") - np.searchsorted(dm, ao[ok], side="left")
            cols["queue_at_aobt"][idx] = v
            # 이웃 출발의 (이륙 - AOBT3), (이륙 - SCHED) 평균(지난 60분), 다음 30분
            order = np.argsort(t)
            ts_ = t[order]
            daobt = (t - ao)[order]
            okd = ~np.isnan(daobt)
            cs_v = np.concatenate([[0.0], np.cumsum(np.where(okd, np.clip(daobt, -3600, 7200), 0.0))])
            cs_n = np.concatenate([[0.0], np.cumsum(okd.astype(float))])
            s_ = window_sum(ts_, cs_v, t - 3600, t + 1)
            n_ = window_sum(ts_, cs_n, t - 3600, t + 1)
            cols["nb_daobt_m60"][idx] = np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan)
            s_ = window_sum(ts_, cs_v, t, t + 1801)
            n_ = window_sum(ts_, cs_n, t, t + 1801)
            cols["nb_daobt_p30"][idx] = np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan)
            dsch = np.clip((t - sc)[order], -3600, 14400)
            cs_v = np.concatenate([[0.0], np.cumsum(dsch)])
            cs_n = np.arange(len(ts_) + 1, dtype=float)
            s_ = window_sum(ts_, cs_v, t - 3600, t + 1)
            n_ = window_sum(ts_, cs_n, t - 3600, t + 1)
            cols["nb_dsch_m60"][idx] = np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan)
            s_ = window_sum(ts_, cs_v, t, t + 1801)
            n_ = window_sum(ts_, cs_n, t, t + 1801)
            cols["nb_dsch_p30"][idx] = np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan)
            # 예정 출발 수(SCHED 기준) 내 이륙 시각 주변
            ss = np.sort(sc)
            cols["c_sched_m30p30"][idx] = window_count(ss, t - 1800, t + 1800)
            # 도착 택시인(시험에서도 보임) 이웃 평균
            if len(a):
                ao_ = np.argsort(a["t_mvt"].to_numpy())
                ta = a["t_mvt"].to_numpy()[ao_]
                yi = np.clip(a["y"].to_numpy()[ao_], 0, 7200)
                oky = ~np.isnan(yi)
                cs_v = np.concatenate([[0.0], np.cumsum(np.where(oky, yi, 0.0))])
                cs_n = np.concatenate([[0.0], np.cumsum(oky.astype(float))])
                s_ = window_sum(ta, cs_v, t - 3600, t)
                n_ = window_sum(ta, cs_n, t - 3600, t)
                cols["nb_taxiin_m60"][idx] = np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan)
                s_ = window_sum(ta, cs_v, t - 900, t + 900)
                n_ = window_sum(ta, cs_n, t - 900, t + 900)
                cols["nb_taxiin_m15p15"][idx] = np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan)
    for k, v in cols.items():
        out[k] = v.astype("float32")
    return out


def stand_seq(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    """Movements at the departure's stand around its take-off, inside the same UTC month.

    Seconds since the in-block of the previous and the second previous arrival at the stand, until the next arrival's in-block, since
    the previous departure's take-off, until the next departure's schedule, and the number of arrivals that blocked in between the
    departure's schedule and its take-off. Returns float32 columns indexed like dep.

    Original note (Korean):
    같은 스탠드의 도착 블록인 시각(시험에서도 보임)과 다른 출발들. 내 이륙 시각 기준 간격.

    v2.4 부터 같은 칸막이(_part: UTC 달) 안의 사건만 본다. 스탠드에 그 달 도착이 없으면 사이 도착 수는 0 이다.
    """
    out = {k: np.full(len(dep), np.nan) for k in ["st_prev_arr_gap", "st_prev2_arr_gap", "st_next_arr_gap",
                                                  "st_prev_dep_gap", "st_next_dep_sch_gap", "st_arr_between_sch_mvt"]}
    period_dep = dep["_part"].to_numpy()
    period_arr = arr["_part"].to_numpy()
    dkey = dep["ADEP_mvt"].astype(str).fillna("nan") + "|" + dep["STAND_mvt"].astype(str).fillna("nan")
    akey = arr["ADES_mvt"].astype(str).fillna("nan") + "|" + arr["STAND_mvt"].astype(str).fillna("nan")
    for per in np.unique(period_dep):
        md = period_dep == per
        ma = period_arr == per
        dk = dkey[md].to_numpy()
        ak = akey[ma].to_numpy()
        dt_ = dep.loc[md, "t_mvt"].to_numpy()
        dsc = dep.loc[md, "t_sch"].to_numpy()
        ab = arr.loc[ma, "t_blk"].to_numpy()
        didx = np.where(md)[0]
        # 정렬 키: (스탠드, 시각)
        a_ord = np.lexsort((ab, ak))
        ak_s, ab_s = ak[a_ord], ab[a_ord]
        d_ord = np.lexsort((dt_, dk))
        dk_s, dt_s, dsc_s = dk[d_ord], dt_[d_ord], dsc[d_ord]
        ukeys, a_start = np.unique(ak_s, return_index=True)
        a_end = np.append(a_start[1:], len(ak_s))
        amap = {k: (s, e) for k, s, e in zip(ukeys, a_start, a_end)}
        ukeys_d, d_start = np.unique(dk_s, return_index=True)
        d_end = np.append(d_start[1:], len(dk_s))
        for k, s, e in zip(ukeys_d, d_start, d_end):
            tt = dt_s[s:e]
            sc_ = dsc_s[s:e]
            rows = didx[d_ord[s:e]]
            # 같은 스탠드의 앞 출발 이륙 시각
            prev_dep = np.concatenate([[np.nan], tt[:-1]])
            out["st_prev_dep_gap"][rows] = tt - prev_dep
            nxt_sch = np.concatenate([sc_[1:], [np.nan]])
            out["st_next_dep_sch_gap"][rows] = nxt_sch - tt
            if k not in amap:
                out["st_arr_between_sch_mvt"][rows] = 0.0
                continue
            a0, a1 = amap[k]
            ib = ab_s[a0:a1]
            j = np.searchsorted(ib, tt, side="left")  # ib[j-1] < tt <= ib[j]
            p1 = np.where(j >= 1, ib[np.maximum(j - 1, 0)], np.nan)
            p2 = np.where(j >= 2, ib[np.maximum(j - 2, 0)], np.nan)
            nx = np.where(j < len(ib), ib[np.minimum(j, len(ib) - 1)], np.nan)
            out["st_prev_arr_gap"][rows] = tt - p1
            out["st_prev2_arr_gap"][rows] = tt - p2
            out["st_next_arr_gap"][rows] = nx - tt
            lo = np.minimum(sc_, tt)
            out["st_arr_between_sch_mvt"][rows] = np.searchsorted(ib, tt, side="left") - np.searchsorted(ib, lo, side="left")
    return pd.DataFrame({k: v.astype("float32") for k, v in out.items()}, index=dep.index)


def metar_feats(dep: pd.DataFrame) -> pd.DataFrame:
    """Weather at the departure airport from METAR reports (data/metar/<ICAO>.csv, IEM ASOS format, written by src/metar_fetch.py).

    For each column the value of the last report that has one within METAR_ASOF_H (3) hours before the take-off: temperature, dew-point
    spread, humidity, wind, gust, visibility, ceiling and present-weather flags; also counts of snow, freezing and thunderstorm reports
    and of precipitation in the past 3 hours, the age of the last report, the 6-hour minimum temperature and a de-icing flag. Airports
    without a file, or not in TZ, get NaN. Returns float32 columns.

    Original note (Korean):
    공개 METAR(data/metar/<ICAO>.csv, src/metar_fetch.py). 이륙 시각 직전 보고의 값과 지난 3시간의 현상 횟수.

    없으면 빈 칸(NaN)으로 둔다. 학습과 시험이 같은 함수로 만든다(관측 시각 기준, 정답은 쓰지 않는다).
    """
    names = ["m_tmpc", "m_spread", "m_relh", "m_wind", "m_gust", "m_vis", "m_ceil", "m_sn", "m_fz", "m_ra", "m_ts", "m_fg",
             "m_sn3h", "m_fz3h", "m_ts3h", "m_precip3h", "m_age", "m_tmin6h", "m_deice"]
    out = {k: np.full(len(dep), np.nan) for k in names}
    mdir = ROOT_DATA / "metar"
    for ap_ in TZ:
        p = mdir / f"{ap_}.csv"
        m = (dep["ADEP_mvt"].astype(str) == ap_).to_numpy()
        if not p.exists() or not m.any():
            continue
        w = pd.read_csv(p, na_values=["", "M"], low_memory=False)
        w["t"] = (pd.to_datetime(w["valid"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
        w = w.sort_values("t").drop_duplicates("t", keep="last").reset_index(drop=True)
        wx = w["wxcodes"].fillna("").astype(str)
        tc = (pd.to_numeric(w["tmpf"], errors="coerce") - 32) * 5 / 9
        dc = (pd.to_numeric(w["dwpf"], errors="coerce") - 32) * 5 / 9
        ceil = np.where(w["skyc1"].astype(str).isin(["BKN", "OVC", "VV"]), pd.to_numeric(w["skyl1"], errors="coerce"), 99999.0)
        vals = {
            "m_tmpc": tc.to_numpy(), "m_spread": (tc - dc).to_numpy(), "m_relh": pd.to_numeric(w["relh"], errors="coerce").to_numpy(),
            "m_wind": pd.to_numeric(w["sknt"], errors="coerce").to_numpy(),
            "m_gust": pd.to_numeric(w["gust"], errors="coerce").fillna(0.0).to_numpy(),  # 빈 돌풍 칸 = 돌풍 없음
            "m_vis": pd.to_numeric(w["vsby"], errors="coerce").to_numpy(), "m_ceil": ceil,
            "m_sn": wx.str.contains("SN|SG|PL").to_numpy().astype(float), "m_fz": wx.str.contains("FZ").to_numpy().astype(float),
            "m_ra": wx.str.contains("RA|DZ").to_numpy().astype(float), "m_ts": wx.str.contains("TS").to_numpy().astype(float),
            "m_fg": wx.str.contains("FG|BR").to_numpy().astype(float),
        }
        tw = w["t"].to_numpy()
        t = dep.loc[m, "t_mvt"].to_numpy()
        j = np.searchsorted(tw, t, side="right") - 1  # 이륙 시각 이전 마지막 보고
        ok = j >= 0
        jj = np.maximum(j, 0)
        for k, v in vals.items():
            # v2.4 as-of 규칙: 칸마다 이륙 시각 전 METAR_ASOF_H 시간 안의 마지막 '값 있는' 보고. 없으면 NaN
            okv = ~np.isnan(v)
            tv, vv = tw[okv], v[okv]
            jv = np.searchsorted(tv, t, side="right") - 1
            jjv = np.maximum(jv, 0)
            fresh = (jv >= 0) & ((t - tv[jjv]) <= METAR_ASOF_H * 3600)
            out[k][m] = np.where(fresh, vv[jjv], np.nan)
        out["m_age"][m] = np.where(ok, t - tw[jj], np.nan)
        # 지난 3시간 현상 횟수
        for k, src_ in [("m_sn3h", "m_sn"), ("m_fz3h", "m_fz"), ("m_ts3h", "m_ts")]:
            cs = np.concatenate([[0.0], np.cumsum(vals[src_])])
            out[k][m] = cs[np.searchsorted(tw, t, side="right")] - cs[np.searchsorted(tw, t - 10800, side="left")]
        pr = ((vals["m_sn"] + vals["m_fz"] + vals["m_ra"] + vals["m_ts"]) > 0).astype(float)
        cs = np.concatenate([[0.0], np.cumsum(pr)])
        out["m_precip3h"][m] = cs[np.searchsorted(tw, t, side="right")] - cs[np.searchsorted(tw, t - 10800, side="left")]
        # 지난 6시간 최저 기온(서리)
        tser = pd.Series(vals["m_tmpc"], index=pd.to_datetime(tw, unit="s"))
        tmin = tser.rolling("6h").min().to_numpy()
        fresh = ok & ((t - tw[jj]) <= METAR_ASOF_H * 3600)
        out["m_tmin6h"][m] = np.where(fresh, tmin[jj], np.nan)
    o = pd.DataFrame(out, index=dep.index)
    # 제빙 가능성: 기온 3도 이하이면서 강수, 안개, 또는 이슬점 차 2도 이하(서리)
    o["m_deice"] = ((o["m_tmpc"] <= 3) & ((o["m_precip3h"] > 0) | (o["m_fg"] > 0) | (o["m_spread"] <= 2) | (o["m_tmin6h"] <= 0))).astype(float)
    o.loc[o["m_tmpc"].isna(), "m_deice"] = np.nan
    return o.astype("float32")


def plan_fix(dep: pd.DataFrame) -> pd.DataFrame:
    """Version 2.6 only, not used by the final stack: align two NM planning clocks with their 2025 meaning before the features.

    LFPG: an EOBT_1 that is less than 15 minutes after IOBT is set back to IOBT (ARVT_1 moves with it). LTFM: plans whose IOBT is 10
    minutes before the schedule move 10 minutes later. Changes dep in place and returns the changed rows by airport and month.

    Original note (Korean):
    v2.6 계획 시계 보정. 피처를 만들기 전에 dep 의 NM 계획 시각을 고친다(입력만, 학습 달과 시험 달이 같은 길). 바뀐 행 수를 (공항, UTC 달)로 돌려준다.

    LFPG: 0 < EOBT_1 - IOBT < 900초(작은 EOBT 갱신)면 EOBT_1 := IOBT. 2025 각 달 2.4~4.3%, 2026-07 24% 다.
      ARVT_1 은 같은 비행 계획 칸이라 EOBT_1 과 같이 움직였다(목적지별 갱신 없음 행의 dur1 중앙값과 견주면 작은 갱신 행은 -58초,
      IOBT 로 재면 +392초). 그래서 ARVT_1 도 같은 폭만큼 당긴다. 안 당기면 dur1, ddur 이 그 행에서 갱신 폭만큼 어긋난다. LOBT 는 99.4% 가 IOBT 그대로다.
    LTFM: IOBT - SCHED == -600초(2025 각 달 12.7~20.5%, 2026-07 0.13%)면 계획 전체(IOBT, LOBT, EOBT_1, ARVT_1)를 600초 늦춰 IOBT := SCHED.
      그 행의 LOBT, EOBT_1 은 99.7% 가 IOBT 와 같고 ARVT_1 도 600초 이르다. IOBT 만 옮기면 lobt_m_iobt 에 -600 이 새로 생기고
      eobt_m_sch, lobt_m_sch 에는 그 범주가 남는다.
    """
    apv = dep["ADEP_mvt"].astype(str).to_numpy()
    ti = dep["t_iobt"].to_numpy(dtype="float64", copy=True)
    de = dep["t_eobt1"].to_numpy(dtype="float64") - ti
    m1 = (apv == "LFPG") & (de > 0) & (de < 900)
    m2 = (apv == "LTFM") & ((ti - dep["t_sch"].to_numpy(dtype="float64")) == -600)
    dep.loc[m1, "t_arvt1"] = dep.loc[m1, "t_arvt1"].to_numpy() - de[m1]
    dep.loc[m1, "t_eobt1"] = ti[m1]
    for c in ("t_iobt", "t_lobt", "t_eobt1", "t_arvt1"):
        dep.loc[m2, c] = dep.loc[m2, c].to_numpy() + 600.0
    cnt = pd.DataFrame({"ap": apv, "part": dep["_part"].to_numpy(), "LFPG_eobt_to_iobt": m1, "LTFM_plan_plus600": m2})
    cnt = cnt[m1 | m2].groupby(["ap", "part"])[["LFPG_eobt_to_iobt", "LTFM_plan_plus600"]].sum()
    return cnt


def v26_feats(dep: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    """Version 2.6 only, not used by the final stack: five more input-only columns.

    Departures of the day relative to the month's median day, a remote-hold flag (AOBT_3 - IOBT above 30 minutes) and its product with
    the gap to the next arrival at the stand, the airport's de-icing class (DEICE_CLS) and its product with the METAR de-icing flag.

    Original note (Korean):
    v2.6 새 칸. 입력만 쓴다. dep 는 plan_fix 를 거친 것, F 는 st_next_arr_gap 과 m_deice 가 붙은 뒤의 것.
    """
    ap = dep["ADEP_mvt"].astype(str).to_numpy()
    out = pd.DataFrame(index=dep.index)
    # 그 공항의 그 UTC 날짜 출발 수 / 같은 칸막이(UTC 달) 안 그 공항 일별 출발 수의 중앙값. 결항이 많은 날(EHAM 2026-01-03~07)을 잡는다
    k = pd.MultiIndex.from_arrays([ap, dep["_part"].to_numpy(), np.floor(dep["t_mvt"].to_numpy() / 86400.0)], names=["ap", "p", "ud"])
    daily = pd.Series(1, index=k).groupby(level=["ap", "p", "ud"]).size()
    med = daily.groupby(level=["ap", "p"]).median()
    out["dep_day_ratio"] = (daily.reindex(k).to_numpy(dtype="float64")
                            / med.reindex(pd.MultiIndex.from_arrays([ap, dep["_part"].to_numpy()])).to_numpy(dtype="float64")).astype("float32")
    # 원격 대기 문: NM 기준 지연(AOBT_3 - IOBT)이 30분을 넘는가. 둘 중 하나라도 없으면 NaN
    r = (dep["t_aobt3"] - dep["t_iobt"]).to_numpy(dtype="float64")
    rh = np.where(np.isnan(r), np.nan, (r > 1800).astype("float64"))
    out["rh30"] = rh.astype("float32")
    out["rh30_gap"] = (rh * F["st_next_arr_gap"].to_numpy(dtype="float64")).astype("float32")
    dc = pd.Series(ap).map(DEICE_CLS).to_numpy(dtype="float64")
    out["deice_cls"] = dc.astype("float32")
    out["m_deice_x"] = (F["m_deice"].to_numpy(dtype="float64") * dc).astype("float32")
    return out


def build(rebuild: bool = False, name: str = "feats") -> pd.DataFrame:
    """Build or read the feature table cache/<name>.parquet, one row per departure.

    The first columns are MVT_ID_mvt, src (1 to 12 = training month, 0 = to predict) and y (the label, NaN for rows to predict); the
    others are features (src.common.feature_cols leaves out the identifiers). With rebuild=False an existing file is read back.
    Movements come from src.load.load_all (cache/all.parquet).

    Original note (Korean):
    name: cache/<name>.parquet. 판을 고정해 쓰려면 feats_v21 처럼 복사본 이름을 준다(재생성은 feats 만).
    """
    cp = CACHE / f"{name}.parquet"
    if cp.exists() and not rebuild:
        return pd.read_parquet(cp)
    assert name not in ("feats_v2", "feats_v21", "feats_v22", "feats_v23", "feats_v24", "feats_v25", "feats_v26"), "고정해 둔 판은 다시 만들지 않는다"
    df = load_all()
    dep = df[df["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    arr = df[df["PHASE_mvt"] == "ARR"][["ADES_mvt", "STAND_mvt", "t_mvt", "t_blk", "y", "src"]].reset_index(drop=True)
    del df
    # v2.4: 순서, 창 피처의 칸막이 = 이륙(착륙) 시각의 UTC 달. 학습, 자, 시험 모두 자기 달 안의 사건만 본다
    for x in (dep, arr):
        ts = pd.to_datetime(x["t_mvt"], unit="s")
        x["_part"] = (ts.dt.year * 100 + ts.dt.month).astype("int64")
    v26 = name in V26_NAMES
    if v26:
        cnt = plan_fix(dep)
        print("v2.6 plan_fix 바뀐 행 수(공항, UTC 달)\n" + cnt.to_string(), flush=True)
    F = pd.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"].to_numpy(), "src": dep["src"].to_numpy(), "y": dep["y"].to_numpy()})
    ap = dep["ADEP_mvt"].astype(str)
    F["ap"] = _cat(ap)
    F["rwy"] = _cat(ap + "_" + dep["RUNWAY_mvt"].astype(str).fillna("nan"))
    F["stand"] = _cat(ap + "_" + dep["STAND_mvt"].astype(str).fillna("nan"))
    act = dep["AIRCRAFT_TYPE_mvt"].astype(str).where(dep["AIRCRAFT_TYPE_mvt"].notna(), dep["AIRCRAFT_TYPE_flt"].astype(str))
    F["actype"] = _cat(act)
    F["op"] = _cat(dep["AIRCRAFT_OPERATOR_flt"])
    fl = dep["FLIGHT_mvt"].astype(str)
    # v2.5: 편명(FLIGHT_mvt)에서만 만들고 편명이 비면 NaN 이 되는 칸(prefix, fl_len)을 뺐다. 시험의 편명 빈 행이 자보다 많다
    # (notes/readiness_deviation_2026-10-02.json). 항공사 정보는 op(AIRCRAFT_OPERATOR_flt)와 G 모형의 목적지로 남는다.
    # v2.3 의 편명 칸(편명 그룹 통계용)은 v2.4 에서 뺐다: m7 에서 이득이 없었다(levers.md)
    F["mseg"] = _cat(dep["MARKET_SEGMENT_flt"])
    F["ftype"] = _cat(dep["FLIGHT_TYPE_flt"])
    F["wtc"] = _cat(dep["WK_TBL_CAT_flt"])
    F["frule"] = _cat(dep["FLIGHT_RULE_mvt"])
    F["ades_reg"] = _cat(dep["ADES_mvt"].astype(str).str[:2])
    F["ades"] = _cat(dep["ADES_mvt"])
    F["nm"] = dep["FLIGHT_ID_mvt"].notna().astype("int8")
    F["garb"] = ((fl.str.len() >= 5) & ((fl.str[3] == fl.str[2]) | (fl.str[3:5] == fl.str[1:3]) | (fl.str[3:6] == fl.str[0:3]))).astype("int8")
    F["fl_eq_cs"] = (fl == dep["CALLSIGN_flt"].astype(str)).astype("int8")
    F["divert"] = (dep["ADES_flt"].astype(str) != dep["ADES_FILED_flt"].astype(str)).astype("int8")
    F["ades_mismatch"] = (dep["ADES_mvt"].astype(str) != dep["ADES_flt"].astype(str)).astype("int8")
    t = dep["t_mvt"]
    for nm_, c in [("d_sch", "t_sch"), ("d_aobt", "t_aobt3"), ("d_eobt", "t_eobt1"), ("d_iobt", "t_iobt"), ("d_lobt", "t_lobt")]:
        F[nm_] = (t - dep[c]).astype("float32")
    F["aobt_m_sch"] = (dep["t_aobt3"] - dep["t_sch"]).astype("float32")
    F["eobt_m_sch"] = (dep["t_eobt1"] - dep["t_sch"]).astype("float32")
    F["lobt_m_sch"] = (dep["t_lobt"] - dep["t_sch"]).astype("float32")
    F["iobt_m_sch"] = (dep["t_iobt"] - dep["t_sch"]).astype("float32")
    F["aobt_m_eobt"] = (dep["t_aobt3"] - dep["t_eobt1"]).astype("float32")
    F["aobt_m_lobt"] = (dep["t_aobt3"] - dep["t_lobt"]).astype("float32")
    F["lobt_m_iobt"] = (dep["t_lobt"] - dep["t_iobt"]).astype("float32")
    F["eobt_m_lobt"] = (dep["t_eobt1"] - dep["t_lobt"]).astype("float32")
    F["dur1"] = (dep["t_arvt1"] - dep["t_eobt1"]).astype("float32")
    F["dur3"] = (dep["t_arvt3"] - dep["t_aobt3"]).astype("float32")
    F["ddur"] = (F["dur3"] - F["dur1"]).astype("float32")
    F["arvt3_m_mvt"] = (dep["t_arvt3"] - t).astype("float32")
    F["arvt3_m_arvt1"] = (dep["t_arvt3"] - dep["t_arvt1"]).astype("float32")
    F["sec_mvt"] = (t % 60).astype("float32")
    F["sec_sch"] = (dep["t_sch"] % 60).astype("float32")
    F["sec_aobt"] = (dep["t_aobt3"] % 60).astype("float32")
    day_m, tod_m, dow_m = local_parts(t, ap)
    day_s, tod_s, _ = local_parts(dep["t_sch"], ap)
    F["hour_mvt"] = (tod_m / 3600).astype("float32")
    F["hour_sch"] = (tod_s / 3600).astype("float32")
    F["dow"] = dow_m.astype("float32")
    F["dd"] = (day_m - day_s).astype("float32")
    # 같은 날 같은 공항 출발 수(하루 규모)
    F["_day"] = day_m
    F["_part"] = dep["_part"].to_numpy()
    # 현지 날짜가 UTC 달 경계에 걸치면 그 날의 다른 달 쪽 행은 시험에 없으므로 (공항, 현지 날짜, UTC 달)로 묶는다(v2.4)
    F["c_day"] = F.groupby(["ap", "_day", "_part"], observed=True)["src"].transform("size").astype("float32")
    # 하루 단위 상태(제빙, 기상 장애 날): 같은 공항, 같은 현지 날짜의 (이륙-AOBT3), (이륙-SCHED) 평균. 입력만 쓴다
    F["_daobt"] = F["d_aobt"].clip(-1800, 5400)
    F["_dsch"] = F["d_sch"].clip(-3600, 14400)
    grp = F.groupby(["ap", "_day", "_part"], observed=True)
    F["day_daobt"] = grp["_daobt"].transform("mean").astype("float32")
    F["day_dsch"] = grp["_dsch"].transform("mean").astype("float32")
    F["day_daobt_rel"] = (F["day_daobt"] - F.groupby("ap", observed=True)["_daobt"].transform("mean")).astype("float32")
    F = F.drop(columns=["_day", "_daobt", "_dsch", "_part"])
    # 같은 공항, 같은 편명, 같은 SCHED 의 출발 기록이 여럿인가(재기록). 이륙 순서와 앞 기록과의 간격. 같은 UTC 달 안에서만(v2.4)
    key = ap.fillna("nan") + "|" + fl.fillna("nan") + "|" + dep["t_sch"].astype("int64").astype(str) + "|" + dep["_part"].astype(str)
    g = pd.DataFrame({"k": key.to_numpy(), "t": t.to_numpy()})
    F["dup_n"] = g.groupby("k")["t"].transform("size").astype("float32").to_numpy()
    F["dup_rank"] = g.groupby("k")["t"].rank(method="first").astype("float32").to_numpy()
    gs = g.sort_values(["k", "t"])
    prev = gs.groupby("k")["t"].shift()
    F["dup_gap_prev"] = (gs["t"] - prev).reindex(g.index).astype("float32").to_numpy()
    F.loc[F["dup_n"] <= 1, ["dup_rank", "dup_gap_prev"]] = np.nan
    # NM 의 AOBT3 가 '이륙 - 기본 택시' 값인지: (공항, 활주로) 별 최빈 분 값과의 차(입력만으로 계산)
    dmin = np.round(F["d_aobt"].to_numpy() / 60.0)
    tmp = pd.DataFrame({"rwy": F["rwy"].astype(str).to_numpy(), "m": dmin})
    mode = tmp.dropna().groupby("rwy")["m"].agg(lambda s: s.value_counts().index[0])
    F["d_aobt_mode_dev"] = (dmin - tmp["rwy"].map(mode).to_numpy()).astype("float32")
    C = congestion(dep, arr)
    S = stand_seq(dep, arr)
    W = metar_feats(dep)
    F = pd.concat([F, C, S, W], axis=1)
    if v26:
        F = pd.concat([F, v26_feats(dep, F)], axis=1)
    F.to_parquet(cp, index=False)
    return F


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import time
    t0 = time.time()
    F = build(rebuild=True, name=sys.argv[1] if len(sys.argv) > 1 else "feats")
    print(F.shape, round(F.memory_usage(deep=True).sum() / 1e6, 1), "MB", round(time.time() - t0, 1), "s")
    print(F.describe().T.to_string())
