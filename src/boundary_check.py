# SPDX-License-Identifier: GPL-3.0-only
# 시험 피처의 달 경계 결측 점검(G2 ② 준비 점검이 잡은 칸의 원인을 본다). 피처 판 하나에서 시험 행과 자의 행의 결측이 달의 마지막 48시간,
# 첫 48시간에 몰리는지 센다. 시험에는 2026년 1월과 7월만 있어 달 끝 출발편의 「다음 움직임」이 자료 밖에 있다.
#     python -m src.boundary_check feats_v21 [칸 ...]   → notes/boundary_check_<판>.json
import datetime as dt
import glob
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ["st_next_arr_gap", "st_next_dep_sch_gap", "rwy_gap_next", "st_arr_between_sch_mvt", "m_wind"]


def main(fv: str, cols: list[str]) -> None:
    names = set(pq.read_schema(ROOT / "cache" / f"{fv}.parquet").names)
    cols = [c for c in cols if c in names]
    F = pd.read_parquet(ROOT / "cache" / f"{fv}.parquet", columns=["MVT_ID_mvt", *cols])
    sub = pd.read_parquet(ROOT / "data" / "submitting.parquet")
    idc = [c for c in sub.columns if c.startswith("MVT_ID")][0]
    rk = pd.read_parquet(ROOT / "data" / "ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"])
    rk = rk[rk["MVT_ID_mvt"].isin(set(sub[idc]))]
    tr = pd.concat([pd.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]) for f in sorted(glob.glob(str(ROOT / "data" / "training_*.parquet")))])
    tr = tr[(tr["PHASE_mvt"] == "DEP") & tr["MVT_TIME_UTC_mvt"].dt.month.isin([1, 7])]
    out = {"feats": fv, "written_kst": dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"), "rows": {}}
    for name, rows in (("test", rk), ("ruler", tr)):
        d = rows.merge(F, on="MVT_ID_mvt")
        t = d["MVT_TIME_UTC_mvt"].dt.tz_convert("UTC").dt.tz_localize(None)
        start = t.dt.to_period("M").dt.to_timestamp()
        end = (t.dt.to_period("M") + 1).dt.to_timestamp()
        last48, first48 = (end - t).dt.total_seconds() < 48 * 3600, (t - start).dt.total_seconds() < 48 * 3600
        out["rows"][name] = {"n": int(len(d)), "columns": {c: {"nan": int(d[c].isna().sum()), "nan_pct": round(100 * float(d[c].isna().mean()), 3),
                                                               "nan_last48h": int((d[c].isna() & last48).sum()), "nan_first48h": int((d[c].isna() & first48).sum())}
                                                           for c in cols}}
    (ROOT / "notes" / f"boundary_check_{fv}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for name in ("test", "ruler"):
        for c, v in out["rows"][name]["columns"].items():
            print(f"{fv} {name:5s} {c:24s} NaN {v['nan']:5d} ({v['nan_pct']:.3f}%), 마지막 48시간 {v['nan_last48h']:5d}, 첫 48시간 {v['nan_first48h']:4d}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1:] or DEFAULT)
