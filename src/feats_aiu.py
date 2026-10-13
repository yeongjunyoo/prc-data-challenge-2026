# SPDX-License-Identifier: GPL-3.0-only
"""Context file v3 for A3 (src/ctx_corr3.py): every column of cache/<base>.parquet (src/feats_ctx2.py, sha-checked, unchanged) plus EUROCONTROL
AIU airport aggregates per departure row, from data/ext/aiu (organizer-approved for the challenge, data/ext/aiu/MANIFEST.json and
organizer_rulings.txt; terms: copy with EUROCONTROL as the source, no commercial use). Only these two kinds of file are read:
  taxi_out_additional_time_<year>.csv   monthly per airport: TF, VALID_FL, TOTAL_REF_NB_FL, TOTAL_REF_TIME_MIN, TOTAL_ADD_TIME_MIN
  all_pre_departure_delays_<year>.csv   daily per airport (FLT_DATE): FLT_DEP_IFR_2, DLY_ALL_PRE_2
A row of year Y (UTC date of MVT) reads the year-Y files: 2025 for training and ruler rows, 2026 for test rows (same definitions both years).
  aiu_m_txo    monthly mean taxi-out, 60 * (TOTAL_REF_TIME_MIN + TOTAL_ADD_TIME_MIN) / TOTAL_REF_NB_FL seconds (flights with an unimpeded reference)
  aiu_m_add    monthly mean additional taxi-out, 60 * TOTAL_ADD_TIME_MIN / TOTAL_REF_NB_FL
  aiu_m_valid  VALID_FL / TF (the file's validity filter, which moves with long-taxi days)
  aiu_d_pdd    daily mean pre-departure delay, 60 * DLY_ALL_PRE_2 / FLT_DEP_IFR_2 seconds. On 2025 it equals our mean of max(0, BLOCK - SCHED)
               over the airport's departures of that UTC day of MVT to 0.8 s (research/ctx_corr/aiu_check.py)
  aiu_d_txo    derived daily mean taxi-out: mean over the airport-day's departures of (MVT - SCHED), minus aiu_d_pdd (= mean of min(MVT - SCHED, y))
These are aggregates of the airport's own off-block records, on the ruler as on the test (the organizer allowed them as inputs, 09-27).
Output: cache/<name>.parquet (rows of the base file) and cache/<name>.json (sha16, input file sha16s, fill per month).
    .venv/Scripts/python.exe -X utf8 src/feats_aiu.py --name ctx_v3 --base ctx_v2 --base_sha ac01294f15b8a618
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
AIU = ROOT / "data" / "ext" / "aiu"
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
NEW = ["aiu_m_txo", "aiu_m_add", "aiu_m_valid", "aiu_d_pdd", "aiu_d_txo"]
FILES = {f"{k}_{y}.csv": None for k in ("taxi_out_additional_time", "all_pre_departure_delays") for y in (2025, 2026)}


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def below_normal() -> None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)


def monthly() -> pd.DataFrame:
    """EUROCONTROL monthly taxi-out figures of the ten airports, 2025 and 2026: mean taxi-out time, mean additional time and valid
    share per airport and month (aiu_m_txo, aiu_m_add, aiu_m_valid; seconds).
    """
    fr = []
    for y in (2025, 2026):
        m = pd.read_csv(AIU / f"taxi_out_additional_time_{y}.csv")
        m = m[m["APT_ICAO"].isin(APTS)].copy()
        m["ym"] = m["YEAR"].astype(str) + "-" + m["MONTH_NUM"].astype(int).map("{:02d}".format)
        m["aiu_m_txo"] = 60 * (m["TOTAL_REF_TIME_MIN"] + m["TOTAL_ADD_TIME_MIN"]) / m["TOTAL_REF_NB_FL"]
        m["aiu_m_add"] = 60 * m["TOTAL_ADD_TIME_MIN"] / m["TOTAL_REF_NB_FL"]
        m["aiu_m_valid"] = m["VALID_FL"] / m["TF"]
        fr.append(m.rename(columns={"APT_ICAO": "ap"})[["ap", "ym", "aiu_m_txo", "aiu_m_add", "aiu_m_valid"]])
    M = pd.concat(fr, ignore_index=True)
    assert not M.duplicated(["ap", "ym"]).any()
    return M


def daily() -> pd.DataFrame:
    """EUROCONTROL daily pre-departure delay of the ten airports, 2025 and 2026: mean delay per departure (aiu_d_pdd, seconds)."""
    fr = []
    for y in (2025, 2026):
        d = pd.read_csv(AIU / f"all_pre_departure_delays_{y}.csv")
        d = d[d["APT_ICAO"].isin(APTS)].copy()
        d["aiu_d_pdd"] = 60 * d["DLY_ALL_PRE_2"] / d["FLT_DEP_IFR_2"]
        fr.append(d.rename(columns={"APT_ICAO": "ap", "FLT_DATE": "day"})[["ap", "day", "aiu_d_pdd"]])
    D = pd.concat(fr, ignore_index=True)
    assert not D.duplicated(["ap", "day"]).any()
    return D


def departures(months: set[str]) -> pd.DataFrame:
    """Departures of the base file's months: MVT_ID, airport, UTC day and month of MVT, MVT - SCHED (inputs only)."""
    fr = []
    files = [p for p in sorted((ROOT / "data").glob("training_*.parquet")) if p.name[9:16] in months] + [ROOT / "data" / "ranking.parquet"]
    for f in files:
        t = pq.read_table(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
        dep = np.asarray(t.column("PHASE_mvt").to_pandas() == "DEP")
        tm = t.column("MVT_TIME_UTC_mvt").to_pandas()
        d = pd.DataFrame({"MVT_ID_mvt": np.asarray(t.column("MVT_ID_mvt").to_numpy(zero_copy_only=False), dtype="float64"),
                          "ap": t.column("ADEP_mvt").to_pandas().astype(str).to_numpy(), "day": tm.dt.strftime("%Y-%m-%d").to_numpy(),
                          "d_sch": (tm - t.column("SCHED_TIME_UTC_mvt").to_pandas()).dt.total_seconds().to_numpy()})[dep]
        fr.append(d)
    D = pd.concat(fr, ignore_index=True)
    D["ym"] = D["day"].str[:7]
    return D


def main():
    """Command line entry: the context table of the base (checked by its SHA-256 prefix) with the AIU columns added."""
    pa_ = argparse.ArgumentParser()
    pa_.add_argument("--name", default="ctx_v3")
    pa_.add_argument("--base", default="ctx_v2")
    pa_.add_argument("--base_sha", required=True)
    a = pa_.parse_args()
    below_normal()
    bp = ROOT / "cache" / f"{a.base}.parquet"
    if sha16(bp) != a.base_sha:
        raise SystemExit(f"{bp.name} sha {sha16(bp)} != {a.base_sha}")
    B = pd.read_parquet(bp)
    months = {p for p in B["part"].astype(str).unique() if p.startswith("2025-")}
    D = departures(months)
    D = D[D["MVT_ID_mvt"].isin(set(B["MVT_ID_mvt"]))].copy()
    g = D.groupby(["ap", "day"])["d_sch"].mean().rename("dsch_day").reset_index()
    D = D.merge(monthly(), on=["ap", "ym"], how="left").merge(daily(), on=["ap", "day"], how="left").merge(g, on=["ap", "day"], how="left")
    D["aiu_d_txo"] = D["dsch_day"] - D["aiu_d_pdd"]
    D = D.set_index("MVT_ID_mvt")
    assert D.index.is_unique
    ids = B["MVT_ID_mvt"].to_numpy()
    for c in NEW:
        B[c] = D[c].reindex(ids).to_numpy()
    p = ROOT / "cache" / f"{a.name}.parquet"
    B.to_parquet(p, index=False)
    fill = {part: {c: round(float(x[c].notna().mean()), 4) for c in NEW} for part, x in B.groupby("part")}
    meta = {"name": a.name, "file": f"cache/{a.name}.parquet", "sha16": sha16(p), "rows": int(len(B)), "base": a.base, "base_sha16": a.base_sha,
            "new_columns": NEW, "inputs_sha16": {f: sha16(AIU / f) for f in FILES}, "fill": fill, "code": "src/feats_aiu.py", "code_sha16": sha16(Path(__file__)),
            "source": "EUROCONTROL Aviation Intelligence Unit (ansperformance.eu), data/ext/aiu/MANIFEST.json"}
    (ROOT / "cache" / f"{a.name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("name", "sha16", "rows", "inputs_sha16")}, ensure_ascii=False))
    print(pd.DataFrame(fill).round(3).to_string())


if __name__ == "__main__":
    main()
