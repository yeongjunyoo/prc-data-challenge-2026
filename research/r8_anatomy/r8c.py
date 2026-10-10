# SPDX-License-Identifier: GPL-3.0-only
"""R8 해부 갈래 공용(10-07). 468 사슬과 비교 후보의 자 파일, 혼합 성분(q, mn, d), 입력 칸.
사슬(468): m6r27wincbrmfw4(고정 B 가중 바탕) -> adsbgeond(V5) ctxv3x50(A3 x0.5) w -> am(artmm) -> fh(arthist) -> gkc -> gkcx -> gkcxncr -> db(AIU 날 이동, 오프블록 날짜) -> aw(반대 창).
"""
from __future__ import annotations

import ctypes
import json
import sys
from ctypes import wintypes
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # the repository root (this file is research/r8_anatomy/r8c.py)
OUT = ROOT / "research" / "r8_anatomy"
sys.path.insert(0, str(ROOT))

C468 = "m6r27wincbrmfw4adsbgeondctxv3x50wamfhgkcxncrdbaw"
SPECS = {
    "c321": "m6r25sgcbxradsbgeow",                                  # R3 의 바탕(10-05)
    "v25": "m6r27skcbfw4adsbgeondctxv3x50wgkcxndsaw",               # 413, 서버 234.0647
    "a1": "m6r27skcbrmfw4adsbgeondctxv3x50wamgkcxncrdsaw",          # 456
    "b": "m6r27wincbrmfw4adsbgeondctxv3x50wamgkcxncrdsaw",          # 446
    "c466": "m6r27wincbrmfw4adsbgeondctxv3x50wamfhgkcxncrdsaw",     # 466
    "c468": C468,                                                   # 468
    "c486": "m6r27skcbrmfw4adsbgeondctxv3x50wamfhgkcxncrdbaw",      # 486
}
STAGES468 = {  # 단계 -> spec(468 사슬)
    "base": "m6r27wincbrmfw4", "corr": "m6r27wincbrmfw4adsbgeondctxv3x50w", "am": "m6r27wincbrmfw4adsbgeondctxv3x50wam",
    "fh": "m6r27wincbrmfw4adsbgeondctxv3x50wamfh", "gkc": "m6r27wincbrmfw4adsbgeondctxv3x50wamfhgkc",
    "gkcx": "m6r27wincbrmfw4adsbgeondctxv3x50wamfhgkcx", "xncr": "m6r27wincbrmfw4adsbgeondctxv3x50wamfhgkcxncr",
    "db": "m6r27wincbrmfw4adsbgeondctxv3x50wamfhgkcxncrdb", "aw": C468}
NDEN = {"val": 344419, "fwd": 190713}
KEYPART = {"val": "holdout", "fwd": "forward", "full": "test"}


def nice():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if sys.platform == "win32":
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 500)


def manifest(spec):
    return json.loads((ROOT / "preds" / f"manifest_{spec}.json").read_text(encoding="utf-8"))


def preds(spec, setting):
    """후보 파일의 예측(val 홀드아웃, fwd 7월 전방, full 시험). manifest 가 없으면 cache 의 pred."""
    mp = ROOT / "preds" / f"manifest_{spec}.json"
    if mp.exists():
        rel = manifest(spec)[KEYPART[setting]]["path"]
        return pd.read_parquet(ROOT / rel).set_index("MVT_ID_mvt")["pred"].astype(float)
    return pd.read_parquet(ROOT / "cache" / f"{setting}_{spec}_s0.parquet", columns=["MVT_ID_mvt", "pred"]).set_index("MVT_ID_mvt")["pred"].astype(float)


def cache(spec, setting, cols=None):
    return pd.read_parquet(ROOT / "cache" / f"{setting}_{spec}_s0.parquet", columns=cols).set_index("MVT_ID_mvt")


def rows(setting):
    """자 행의 입력 칸(r2_errors/d1_rows: val 의 344,419 행). fwd 는 그 가운데 7월."""
    R = pd.read_parquet(ROOT / "research" / "r2_errors" / "d1_rows.parquet")
    R = R.drop(columns=[c for c in ("p_base", "p_v16", "p_c272", "p_v14", "p_art", "mu") if c in R.columns])
    if setting == "fwd":
        R = R[R["month"] == 7]
    return R


def boot(d_row, day, nden, B=1000, seed=0):
    """날(UTC 이륙 날짜) 다시 뽑기: 행 Δ 를 날마다 묶어 B 번. 반환 Δ(ΔMSE) 표본."""
    ud, inv = np.unique(np.asarray(day), return_inverse=True)
    dday = np.bincount(inv, weights=np.asarray(d_row, float), minlength=len(ud))
    rng = np.random.default_rng(seed)
    return np.array([(dday * np.bincount(rng.integers(0, len(ud), len(ud)), minlength=len(ud))).sum() for _ in range(B)]) / nden
