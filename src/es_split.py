# SPDX-License-Identifier: GPL-3.0-only
"""옛 방식과 새 방식의 차이를 둘로 나눈다(본 자 2025-01, 07).

old : 잴 행으로 조기 종료, 학습 범위 전체로 학습(--es holdout)            태그 m4b, m6
new : 학습 범위 안 날짜 블록으로 조기 종료, 그 날은 학습에서 뺌(--es internal)  태그 m4bi, m6i
refit: new 의 반복 수 x 1.1 로 학습 범위 전체에 다시 학습(--es internal --refit) 태그 m4br, m6r
(a) 조기 종료를 잴 행으로 한 낙관 = refit - old
(b) 조기 종료 날(약 10%)을 학습에서 뺀 비용 = new - refit
합 = new - old
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")


def avg(tag, seeds):
    ds = [pd.read_parquet(ROOT / "cache" / f"val_{tag}_s{s}.parquet") for s in seeds]
    y = ds[0]["y"].to_numpy()
    ps = [np.clip(d["pred"].to_numpy(), 0, None) for d in ds]
    return y, ps, np.mean(ps, axis=0)


def r(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


for name, old, new, refit in [("m4b_avg2", "m4b", "m4bi", "m4br"), ("m6_avg2", "m6", "m6i", "m6r")]:
    y, po, ao = avg(old, [0, 1])
    _, pn, an = avg(new, [0, 1])
    _, pr, ar = avg(refit, [0, 1])
    ro, rn, rr = r(ao, y), r(an, y), r(ar, y)
    core = y < 3600
    print(f"{name}: old {ro:.2f} | new(ES days dropped) {rn:.2f} | refit {rr:.2f} || (a) ES optimism refit-old {rr - ro:+.2f} | "
          f"(b) dropped-days cost new-refit {rn - rr:+.2f} | total new-old {rn - ro:+.2f}")
    print(f"   per seed old {[round(r(p, y), 2) for p in po]} new {[round(r(p, y), 2) for p in pn]} refit {[round(r(p, y), 2) for p in pr]}"
          f" | core old {r(ao[core], y[core]):.2f} new {r(an[core], y[core]):.2f} refit {r(ar[core], y[core]):.2f}")
