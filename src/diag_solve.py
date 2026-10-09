# SPDX-License-Identifier: GPL-3.0-only
# G2 성분 진단 풀이: 상수 발 둘(v2 상수 0, v3 상수 1000)의 서버 RMSE 로 시험 정답의 평균과 분산을 푼다.
# RMSE²(c) = Var(y) + (mean(y) - c)² 이라 두 식을 빼면 mean 이, 넣으면 Var 가 나온다. 결과는 notes/diag_server_solve.json.
import csv
import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
led = {r["version"]: r for r in csv.DictReader(open(ROOT / "subs" / "ledger.csv", encoding="utf-8"))}
c1, r1 = 0.0, float(led["2"]["server_rmse"])
c2, r2 = 1000.0, float(led["3"]["server_rmse"])
mean = (r1**2 - r2**2 + c2**2 - c1**2) / (2 * (c2 - c1))
var = r1**2 - (mean - c1) ** 2
loc = json.load(open(ROOT / "notes" / "diag_local_constants.json", encoding="utf-8"))
out = {"server": {"v2_c0": r1, "v3_c1000": r2, "test_mean": round(mean, 2), "test_sd": round(math.sqrt(var), 2)},
       "local_holdout": {"mean": round(loc["holdout_mean"], 2), "sd": round(math.sqrt(loc["holdout_var"]), 2)},
       "method": "RMSE²(c) = Var + (mean − c)², c = 0 과 1000 두 식을 연립"}
(ROOT / "notes" / "diag_server_solve.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
