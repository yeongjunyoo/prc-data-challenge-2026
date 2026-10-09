# SPDX-License-Identifier: GPL-3.0-only
"""학습 12개월과 ranking 을 한 표로 읽는다. 문자열은 category, 시각은 epoch 초(float, 결측 NaN).

메모리를 아끼려고 결과를 cache/all.parquet 에 한 번 저장해 다시 쓴다.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"

SHORT = {"MVT_TIME_UTC_mvt": "t_mvt", "BLOCK_TIME_UTC_mvt": "t_blk", "SCHED_TIME_UTC_mvt": "t_sch", "LOBT_flt": "t_lobt",
         "IOBT_flt": "t_iobt", "EOBT_1_flt": "t_eobt1", "ARVT_1_flt": "t_arvt1", "AOBT_3_flt": "t_aobt3",
         "ARVT_3_flt": "t_arvt3"}
STR_COLS = ["FLIGHT_mvt", "FLIGHT_RULE_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt",
            "STAND_mvt", "CALLSIGN_flt", "ADEP_flt", "ADES_flt", "ADES_FILED_flt", "MARKET_SEGMENT_flt",
            "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt"]
TIMES = list(SHORT.values())


def _read(path: str) -> pd.DataFrame:
    t = pq.read_table(path)
    out = {}
    for c in t.column_names:
        col = t.column(c)
        if c in SHORT:
            us = col.cast(pa.int64()).to_numpy(zero_copy_only=False)  # 결측은 float NaN 으로 온다
            out[SHORT[c]] = np.asarray(us, dtype="float64") / 1e6
        elif c in ("MVT_ID_mvt", "FLIGHT_ID_mvt"):
            out[c] = np.asarray(col.to_numpy(zero_copy_only=False), dtype="float64")
        elif c == "TAXITIME_SEC_mvt":
            out["y"] = np.asarray(col.to_numpy(zero_copy_only=False), dtype="float64")
        else:
            out[c] = np.asarray(col.to_pandas(), dtype=object)
    return pd.DataFrame(out)


def load_all(rebuild: bool = False) -> pd.DataFrame:
    """전체 이동(DEP, ARR)과 ranking. src: 1~12 = 2025 월, 0 = ranking(2026)."""
    CACHE.mkdir(exist_ok=True)
    cp = CACHE / "all.parquet"
    if cp.exists() and not rebuild:
        return pd.read_parquet(cp)
    parts = []
    files = sorted(glob.glob(str(DATA / "training_*.parquet"))) + [str(DATA / "ranking.parquet")]
    for f in files:
        d = _read(f)
        for c in STR_COLS:
            d[c] = d[c].astype("category")
        d["src"] = np.int8(0) if f.endswith("ranking.parquet") else np.int8(int(Path(f).name[14:16]))
        parts.append(d)
    # category 를 전체 기준으로 맞춘다(학습과 시험의 코드가 같게)
    for c in STR_COLS:
        cats = pd.api.types.union_categoricals([p[c] for p in parts]).categories
        for p in parts:
            p[c] = p[c].cat.set_categories(cats)
    df = pd.concat(parts, ignore_index=True)
    del parts
    df.to_parquet(cp, index=False)
    return df


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    d = load_all(rebuild=True)
    print(d.shape, round(d.memory_usage(deep=True).sum() / 1e6, 1), "MB")
    print(d.dtypes.to_string())
    print(d.groupby(["src", "PHASE_mvt"], observed=True).size().unstack())
