# SPDX-License-Identifier: GPL-3.0-only
"""Context columns for the all-row corrector (src/ctx_corr.py), built from movement inputs only (never a departure's taxi-out).

Idea credit: the public team outgoing-boat (EnioAguiar/prc-taxiout-2026, GPL-3.0) describes, in its README and research notes, a second-stage
corrector over all rows with "context" columns: arrival taxi-in per runway in 15 and 60 minute windows, neighbour departures' (MVT - AOBT_3)
before and after, own minus neighbour, the taxi-in of the previous arrival on the same stand, the stand prefix, gate holding at AOBT_3
(departures past EOBT not yet off-block), the recent off-block delay and the runway configuration. We took the list of ideas from that prose
and wrote every definition and line below ourselves on our own data (no code copied). README 「외부 아이디어」 carries the credit.

Rows: every departure of the 2025 training months and of the 2026 ranking file. Windows never cross a calendar month (the test file holds
two separate months), each airport on its own. For a departure at take-off time t (MVT), AOBT_3 a, EOBT_1 e:
  arr_rwy_ti_m15, arr_rwy_n_m15, arr_rwy_ti_m60  arrivals on the same runway string that landed in (t - 15 min, t] / (t - 60 min, t]:
                                                  mean taxi-in (clipped 0..3600 s) and count
  arr_ap_ti_m15                                   the same over all arrivals at the airport, 15 minutes
  nb_rwy_daobt_m60, nb_rwy_daobt_p30              other departures on the same runway that took off in (t - 3601, t - 1] / (t, t + 1800]:
                                                  mean of clip(MVT - AOBT_3, 0, 7200)
  own_m_nb_rwy, own_m_nb_ap                       own clip(MVT - AOBT_3, 0, 7200) minus the runway / airport mean of the previous hour
  prev_st_arr_ti, prev_st_arr_age                 the last arrival on the same stand with in-block before t (at most 24 h old): taxi-in, t - in-block
  ret_rwy, ret_ap                                 departures (same runway / airport) with EOBT_1 <= a < AOBT_3 and 0 < AOBT_3 - EOBT_1 < 7200 s:
                                                  held past their EOBT at my AOBT_3
  delay_ap_m15                                    mean clip(AOBT_3 - EOBT_1, -3600, 7200) of airport departures whose AOBT_3 is in (a - 900, a]
  rwy_share_pm30, n_dep_rwys_pm30, n_arr_rwys_pm30  departures within t +- 30 min: share on my runway, number of runways used; arrivals: runways used
  stand_p                                         airport + leading letters of the stand name (first character when there are none)
  t, aobt3, day, part                              take-off, AOBT_3 (epoch s), UTC day, YYYY-MM (the corrector derives prediction-dependent columns)
Output: cache/<name>.parquet and cache/<name>.json (sha16, rows, fill share per month).
    .venv/Scripts/python.exe -X utf8 src/feats_ctx.py --name ctx_v1
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
COLS = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt",
        "TAXITIME_SEC_mvt", "AOBT_3_flt", "EOBT_1_flt"]
FEATS = ["arr_rwy_ti_m15", "arr_rwy_n_m15", "arr_rwy_ti_m60", "arr_ap_ti_m15", "nb_rwy_daobt_m60", "nb_rwy_daobt_p30", "own_m_nb_rwy",
         "own_m_nb_ap", "prev_st_arr_ti", "prev_st_arr_age", "ret_rwy", "ret_ap", "delay_ap_m15", "rwy_share_pm30", "n_dep_rwys_pm30",
         "n_arr_rwys_pm30"]


def below_normal() -> None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)


def epoch_s(col: pa.ChunkedArray) -> np.ndarray:
    unit = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[col.type.unit]
    return np.asarray(col.cast(pa.int64()).to_numpy(zero_copy_only=False), dtype="float64") / unit


def read(path: str) -> pd.DataFrame:
    """Movements of one challenge file with the columns the context needs; a departure's taxi-out (the label) is set to NaN."""
    t = pq.read_table(path, columns=COLS)
    d = pd.DataFrame({"MVT_ID_mvt": np.asarray(t.column("MVT_ID_mvt").to_numpy(zero_copy_only=False), dtype="float64")})
    for c in ("ADEP_mvt", "ADES_mvt", "PHASE_mvt", "RUNWAY_mvt", "STAND_mvt"):
        d[c] = t.column(c).to_pandas().astype("object").fillna("nan").astype(str).to_numpy()
    for c, n in (("MVT_TIME_UTC_mvt", "t"), ("BLOCK_TIME_UTC_mvt", "blk"), ("AOBT_3_flt", "ao"), ("EOBT_1_flt", "eo")):
        d[n] = epoch_s(t.column(c))
    d["tx"] = np.asarray(t.column("TAXITIME_SEC_mvt").to_numpy(zero_copy_only=False), dtype="float64")
    d.loc[d["PHASE_mvt"] == "DEP", "tx"] = np.nan  # a departure's taxi-out is the label: never read here
    d["part"] = pd.to_datetime(d["t"], unit="s").dt.strftime("%Y-%m").to_numpy()
    return d


def win(ts_sorted: np.ndarray, csum: np.ndarray, ccnt: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Sum and count of values whose time is in (lo, hi] (ts sorted, csum/ccnt cumulative with a leading 0)."""
    a, b = np.searchsorted(ts_sorted, lo, side="right"), np.searchsorted(ts_sorted, hi, side="right")
    return csum[b] - csum[a], ccnt[b] - ccnt[a]


def cum(ts: np.ndarray, vals: np.ndarray):
    o = np.argsort(ts, kind="stable")
    ts, vals = ts[o], vals[o]
    ok = np.isfinite(vals) & np.isfinite(ts)
    ts = np.where(np.isfinite(ts), ts, np.inf)
    return ts, np.concatenate([[0.0], np.cumsum(np.where(ok, vals, 0.0))]), np.concatenate([[0.0], np.cumsum(ok.astype(float))])


def wmean(ts, vals, lo, hi):
    s, c, n = cum(ts, vals)
    sm, cn = win(s, c, n, lo, hi)
    return np.where(cn > 0, sm / np.maximum(cn, 1), np.nan), cn


def held(e: np.ndarray, o: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Number of departures with e <= at < o (0 < o - e < 7200), at each time in `at` (NaN stays NaN)."""
    ok = np.isfinite(e) & np.isfinite(o) & (o > e) & (o - e < 7200)
    st, en = np.sort(e[ok]), np.sort(o[ok])
    n = np.searchsorted(st, at, side="right") - np.searchsorted(en, at, side="right")
    return np.where(np.isfinite(at), n, np.nan).astype(float)


def stand_prefix(s: str) -> str:
    m = re.match(r"[A-Za-z]+", s)
    return m.group(0).upper() if m else s[:1]


def one_airport(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    """Context columns (FEATS) of the departures of one airport and one UTC month, from its departures and arrivals."""
    n = len(dep)
    out = {k: np.full(n, np.nan) for k in FEATS}
    t, ao, eo = dep["t"].to_numpy(), dep["ao"].to_numpy(), dep["eo"].to_numpy()
    daobt = np.clip(t - ao, 0, 7200)
    rwy = dep["RUNWAY_mvt"].to_numpy()
    ta, ti = arr["t"].to_numpy(), np.clip(arr["tx"].to_numpy(), 0, 3600)
    arwy = arr["RUNWAY_mvt"].to_numpy()
    out["arr_ap_ti_m15"], _ = wmean(ta, ti, t - 900, t)
    m, _ = wmean(t, daobt, t - 3601, t - 1)
    out["own_m_nb_ap"] = daobt - m
    out["ret_ap"] = held(eo, ao, ao)
    dly = np.where(np.isfinite(ao) & np.isfinite(eo), np.clip(ao - eo, -3600, 7200), np.nan)
    out["delay_ap_m15"], _ = wmean(ao, dly, ao - 900, ao)
    # runway configuration around t: departures and arrivals per runway within +- 30 min
    _, tot = wmean(t, np.ones(n), t - 1800, t + 1800)
    nd = np.zeros(n)
    for r in np.unique(rwy):
        tr = np.sort(t[rwy == r])
        c = np.searchsorted(tr, t + 1800, side="right") - np.searchsorted(tr, t - 1800, side="right")
        nd += c > 0
        sel = rwy == r
        out["rwy_share_pm30"][sel] = c[sel] / np.maximum(tot[sel], 1)
    out["n_dep_rwys_pm30"] = nd
    na = np.zeros(n)
    for r in np.unique(arwy):
        tr = np.sort(ta[arwy == r])
        na += (np.searchsorted(tr, t + 1800, side="right") - np.searchsorted(tr, t - 1800, side="right")) > 0
    out["n_arr_rwys_pm30"] = na
    for r in np.unique(rwy):
        sel = np.nonzero(rwy == r)[0]
        tr = t[sel]
        am = arwy == r
        m15, c15 = wmean(ta[am], ti[am], tr - 900, tr)
        m60, _ = wmean(ta[am], ti[am], tr - 3600, tr)
        out["arr_rwy_ti_m15"][sel], out["arr_rwy_n_m15"][sel], out["arr_rwy_ti_m60"][sel] = m15, c15, m60
        mb, _ = wmean(tr, daobt[sel], tr - 3601, tr - 1)
        mf, _ = wmean(tr, daobt[sel], tr, tr + 1800)
        out["nb_rwy_daobt_m60"][sel], out["nb_rwy_daobt_p30"][sel] = mb, mf
        out["own_m_nb_rwy"][sel] = daobt[sel] - mb
        out["ret_rwy"][sel] = held(eo[sel], ao[sel], ao[sel])
    # previous arrival on the same stand (in-block before my take-off, at most 24 h old)
    ab = arr[np.isfinite(arr["blk"].to_numpy())]
    stand = dep["STAND_mvt"].to_numpy()
    for s, idx in pd.Series(np.arange(n)).groupby(stand):
        a = ab[ab["STAND_mvt"].to_numpy() == s]
        if a.empty:
            continue
        o = np.argsort(a["blk"].to_numpy(), kind="stable")
        bk, tx = a["blk"].to_numpy()[o], np.clip(a["tx"].to_numpy()[o], 0, 3600)
        ii = idx.to_numpy()
        j = np.searchsorted(bk, t[ii], side="left") - 1
        ok = j >= 0
        age = np.where(ok, t[ii] - bk[np.maximum(j, 0)], np.nan)
        keep = ok & (age <= 86400)
        out["prev_st_arr_ti"][ii] = np.where(keep, tx[np.maximum(j, 0)], np.nan)
        out["prev_st_arr_age"][ii] = np.where(keep, age, np.nan)
    F = pd.DataFrame(out)
    F.insert(0, "MVT_ID_mvt", dep["MVT_ID_mvt"].to_numpy())
    F["stand_p"] = [f"{a}_{stand_prefix(s)}" for a, s in zip(dep["ADEP_mvt"].to_numpy(), stand)]
    F["t"], F["aobt3"] = t, ao
    F["day"] = pd.to_datetime(t, unit="s").strftime("%Y-%m-%d").to_numpy()
    return F


def build(d: pd.DataFrame) -> pd.DataFrame:
    """Context columns of every departure of the movements d, airport by airport and month by month."""
    parts = []
    for part in sorted(d["part"].unique()):
        dp = d[d["part"] == part]
        dep_all, arr_all = dp[dp["PHASE_mvt"] == "DEP"], dp[dp["PHASE_mvt"] == "ARR"]
        for ap in sorted(dep_all["ADEP_mvt"].unique()):
            dep = dep_all[dep_all["ADEP_mvt"] == ap].reset_index(drop=True)
            arr = arr_all[arr_all["ADES_mvt"] == ap].reset_index(drop=True)
            F = one_airport(dep, arr)
            F["part"] = part
            parts.append(F)
    return pd.concat(parts, ignore_index=True)


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    """Command line entry: read the challenge files, build the context table and write cache/<name>.parquet and its json."""
    pa_ = argparse.ArgumentParser()
    pa_.add_argument("--name", default="ctx_v1")
    pa_.add_argument("--months", default="1,2,3,4,5,6,7,8,9,10,11,12", help="2025 training months to include (the test file is always in)")
    a = pa_.parse_args()
    below_normal()
    want = {int(m) for m in a.months.split(",")}
    out = []
    for f in sorted(glob.glob(str(ROOT / "data" / "training_*.parquet"))):
        if int(Path(f).name[14:16]) not in want:
            continue
        out.append(build(read(f)))
        print(Path(f).name, len(out[-1]), flush=True)
    d = read(str(ROOT / "data" / "ranking.parquet"))
    out.append(build(d))
    print("ranking", len(out[-1]), flush=True)
    F = pd.concat(out, ignore_index=True)
    assert F["MVT_ID_mvt"].is_unique
    vocab = {s: i for i, s in enumerate(sorted(F["stand_p"].unique()))}
    F["stand_p_c"] = F["stand_p"].map(vocab).astype("int32")
    p = ROOT / "cache" / f"{a.name}.parquet"
    F.to_parquet(p, index=False)
    fill = {part: {c: round(float(g[c].notna().mean()), 4) for c in FEATS} for part, g in F.groupby("part")}
    meta = {"name": a.name, "file": f"cache/{a.name}.parquet", "sha16": sha16(p), "rows": int(len(F)), "parts": sorted(F["part"].unique().tolist()),
            "columns": FEATS + ["stand_p", "stand_p_c", "t", "aobt3", "day", "part"], "stand_p_vocab": len(vocab), "fill": fill,
            "code": "src/feats_ctx.py", "code_sha16": sha16(Path(__file__))}
    (ROOT / "cache" / f"{a.name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("name", "sha16", "rows", "parts", "stand_p_vocab")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
