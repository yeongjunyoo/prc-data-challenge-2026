# SPDX-License-Identifier: GPL-3.0-only
# ANCHOR(플레이북 §0): 가장 단순한 제출. 2025년 전체 출발편의 공항별 택시아웃 중앙값을 손대지 않고 낸다.
import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.make_sub import build  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
cols = ["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
tr = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/training_*.parquet"))], ignore_index=True)
tr = tr[tr["PHASE_mvt"] == "DEP"]
med = tr.groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].median()

# 같은 방법의 로컬 자: 1월, 7월을 뺀 달로 중앙값을 내 1월, 7월을 잰다(시험 기간과 같은 달)
hold = tr["MVT_TIME_UTC_mvt"].dt.month.isin([1, 7])
med_h = tr[~hold].groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].median()
y = tr.loc[hold, "TAXITIME_SEC_mvt"].astype(float)
local = float(np.sqrt(np.mean((tr.loc[hold, "ADEP_mvt"].map(med_h).astype(float) - y) ** 2)))

rk = pd.read_parquet("data/ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt"])
rk = rk[rk["PHASE_mvt"] == "DEP"].set_index("MVT_ID_mvt")
pred = rk["ADEP_mvt"].map(med).astype(float)
build(pred, "ANCHOR 공항별 중앙값(2025 전체 출발)", local_rmse=local, note="플레이북 §0 ANCHOR, 서버 바닥과 파이프라인 확인")
print("local holdout RMSE (Jan, Jul 2025)", round(local, 2))
