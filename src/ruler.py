# SPDX-License-Identifier: GPL-3.0-only
"""The local ruler: the departures of January and July 2025 (RULER_MONTHS), the hash of that row set and the RMSE of a prediction file on it."""
# 로컬 자 하나의 정의(2025-01, 07 출발편), 그 행 해시, 예측 파일의 RMSE. sign_agree 와 모델링의 ruler_checks 가 같은 정의를 쓴다.
# compare.holdout() 도 같은 행(2025 출발편 중 1, 7월)을 고른다.
import glob
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RULER_MONTHS = (1, 7)


def ruler_frame() -> pd.DataFrame:
    """All 2025 departures of data/training_*.parquet, indexed by MVT_ID_mvt, with the columns month and TAXITIME_SEC_mvt.

    The ruler is the rows whose month is in RULER_MONTHS.

    Original note (Korean):
    2025 출발편 전체(index MVT_ID_mvt, 칸 month, TAXITIME_SEC_mvt). 자는 month 가 RULER_MONTHS 인 행이다.
    """
    cols = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
    tr = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(ROOT / "data" / "training_*.parquet")))], ignore_index=True)
    tr = tr[tr["PHASE_mvt"] == "DEP"]
    return tr.assign(month=tr["MVT_TIME_UTC_mvt"].dt.month).set_index("MVT_ID_mvt")


def ruler_hash(ids) -> str:
    """Hash of a row set: the first 16 hex characters of the SHA-256 of the sorted movement ids as int64.

    Original note (Korean):
    자의 행 정의 해시. 정렬한 MVT_ID(int64)의 sha256 앞 16자. ruler_checks.json 의 ruler_hash 와 같아야 한다.
    """
    return hashlib.sha256(np.sort(np.asarray(ids, dtype="int64")).tobytes()).hexdigest()[:16]


def rmse_of(path: str, rows: pd.DataFrame) -> float:
    """RMSE of a prediction file (MVT_ID_mvt, pred) on the labels of rows; stops if a row has no prediction.

    Original note (Korean):
    예측 파일(MVT_ID_mvt, pred)을 rows 의 정답으로 잰 RMSE. 행이 하나라도 빠지면 멈춘다.
    """
    y = rows["TAXITIME_SEC_mvt"].astype(float)
    p = pd.read_parquet(ROOT / path).set_index("MVT_ID_mvt")["pred"].reindex(y.index)
    if p.isna().any():
        raise SystemExit(f"{path}: 측정 행과 안 맞는다(빠진 행 {int(p.isna().sum())})")
    return float(np.sqrt(((p - y) ** 2).mean()))
