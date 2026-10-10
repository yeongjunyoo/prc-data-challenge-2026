# SPDX-License-Identifier: GPL-3.0-only
# G7 자의 분해능: 로컬 자(2025-01, 07 출발편)의 RMSE 와 두 방법 차이(Δ)의 흔들림을 날짜 블록 부트스트랩으로 잰다.
# 같은 날의 행끼리는 붐빔과 날씨를 같이 타서 행 단위로 뽑으면 흔들림을 작게 본다. 날짜를 통째로 뽑는다.
import glob
import json
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
cols = ["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt"]
tr = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/training_*.parquet"))], ignore_index=True)
tr = tr[tr["PHASE_mvt"] == "DEP"]
hold = tr["MVT_TIME_UTC_mvt"].dt.month.isin([1, 7])
med = tr[~hold].groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].median()
h = tr[hold].copy()
y = h["TAXITIME_SEC_mvt"].astype(float).to_numpy()
a = h["ADEP_mvt"].map(med).astype(float).to_numpy()                       # 방법 A: 공항 중앙값(v1 과 같은 방법)
prox = (h["MVT_TIME_UTC_mvt"] - h["AOBT_3_flt"]).dt.total_seconds()
b = prox.where(prox.notna(), h["ADEP_mvt"].map(med)).clip(lower=0).astype(float).to_numpy()  # 방법 B: 이륙 − AOBT_3
day = h["MVT_TIME_UTC_mvt"].dt.date.to_numpy()
days = np.unique(day)
idx = {d: np.flatnonzero(day == d) for d in days}
se_a, se_b = (a - y) ** 2, (b - y) ** 2
rng = np.random.default_rng(20261002)
ra, rb, rd = [], [], []
for _ in range(300):
    pick = np.concatenate([idx[d] for d in rng.choice(days, size=len(days), replace=True)])
    xa, xb = np.sqrt(se_a[pick].mean()), np.sqrt(se_b[pick].mean())
    ra.append(xa); rb.append(xb); rd.append(xb - xa)
out = {"rows": int(len(y)), "days": int(len(days)), "reps": 300,
       "rmse_A_median": float(np.sqrt(se_a.mean())), "sd_A": float(np.std(ra)),
       "rmse_B_proxy": float(np.sqrt(se_b.mean())), "sd_B": float(np.std(rb)),
       "delta_B_minus_A": float(np.sqrt(se_b.mean()) - np.sqrt(se_a.mean())), "sd_delta": float(np.std(rd)),
       "rule": "증분이 sd_delta 의 2배 미만이면 로컬로 승격하지 않고 쿼터로 판정(private notes G7)"}
json.dump(out, open("notes/resolution.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
