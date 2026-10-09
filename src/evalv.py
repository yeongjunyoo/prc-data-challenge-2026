# SPDX-License-Identifier: GPL-3.0-only
"""검증 예측 파일(cache/val_*.parquet)을 같은 자로 다시 잰다. 시드 평균, 섞기, 그리고 '보이는 신호 없는 큰 정답' 행을 뺀 값.

사용: python src/evalv.py <col> <file1> [<file2> ...]   (여러 파일이면 시드 평균과 시드별 RMSE 표준편차)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")


def main():
    col, files = sys.argv[1], sys.argv[2:]
    dfs = [pd.read_parquet(ROOT / "cache" / f).set_index("MVT_ID_mvt") for f in files]
    y = dfs[0]["y"].to_numpy()
    dsch = dfs[0]["dsch"].to_numpy() if "dsch" in dfs[0] else None
    rm = [float(np.sqrt(np.mean((d[col].to_numpy() - y) ** 2))) for d in dfs]
    avg = sum(d[col] for d in dfs) / len(dfs)
    r_avg = float(np.sqrt(np.mean((avg.to_numpy() - y) ** 2)))
    print(f"{col}: per-file RMSE {[round(r, 2) for r in rm]} mean {np.mean(rm):.2f} sd {np.std(rm, ddof=1) if len(rm) > 1 else 0:.2f} | avg-pred RMSE {r_avg:.2f}")
    if dsch is not None:
        noise = (y > 50000) & (dsch < 20000)
        print(f"  rows with y>50000 and MVT-SCHED<20000 (no visible signal): {int(noise.sum())}; RMSE without them {np.sqrt(np.mean((avg.to_numpy()[~noise] - y[~noise]) ** 2)):.2f}")


if __name__ == "__main__":
    main()
