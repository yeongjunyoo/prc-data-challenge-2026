# SPDX-License-Identifier: GPL-3.0-only
"""제출 파일을 만들고 검사한 뒤 장부에 적는다. 올리기는 s3io.push 가 한다.

규칙(ranking.html): 템플릿 `submitting.parquet` 의 TAXITIME 칸을 채운다. MVT_ID 불일치, 행 누락, 행 추가는 채점 오류.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "data" / "submitting.parquet"
LEDGER = ROOT / "subs" / "ledger.csv"
TEAM = "spirited-tiger"
FIELDS = ["version", "made_kst", "file", "sha256", "method", "local_rmse", "server_rmse", "server_read_kst", "note"]


def next_version() -> int:
    used = [int(p.stem.split("_v")[1]) for p in (ROOT / "subs").glob(f"{TEAM}_v*.parquet")]
    return max(used, default=0) + 1


def ledger_append(path: Path, method: str, local_rmse: float | None, note: str) -> None:
    """제출 파일 한 줄을 장부에 더한다. 발사기는 올리기가 끝난 뒤에만 부른다(올리지 못한 파일은 장부에 없다)."""
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    new = not LEDGER.exists()
    with LEDGER.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"version": int(path.stem.split("_v")[1]), "made_kst": dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M"),
                    "file": path.name, "sha256": sha[:16], "method": method,
                    "local_rmse": "" if local_rmse is None else round(local_rmse, 2), "server_rmse": "",
                    "server_read_kst": "", "note": note})


def build(pred: pd.Series, method: str, local_rmse: float | None = None, note: str = "", write_ledger: bool = True) -> Path:
    """pred: index = MVT_ID_mvt, 값 = 초. 템플릿 순서대로 맞추고 검사한다."""
    # 템플릿의 정답 칸은 전부 비어 pandas 가 실수형으로 읽는다. 서버가 보는 것은 parquet 스키마라 arrow 스키마로 맞춘다.
    schema = pq.read_schema(TEMPLATE)
    tpl = pd.read_parquet(TEMPLATE)
    vals = pred.reindex(tpl["MVT_ID_mvt"].to_numpy())
    if vals.isna().any():
        raise SystemExit(f"예측 없는 행 {int(vals.isna().sum())}개")
    out = pd.DataFrame({"MVT_ID_mvt": tpl["MVT_ID_mvt"].to_numpy(),
                        "TAXITIME_SEC_mvt": np.clip(np.rint(vals.to_numpy()), 0, None).astype("int32")})
    table = pa.Table.from_pandas(out, schema=schema, preserve_index=False)
    assert table.schema.equals(schema, check_metadata=False) and table.num_rows == len(tpl)
    v = next_version()
    path = ROOT / "subs" / f"{TEAM}_v{v}.parquet"
    pq.write_table(table, path)
    back = pd.read_parquet(path)
    assert back["MVT_ID_mvt"].equals(tpl["MVT_ID_mvt"]) and back["TAXITIME_SEC_mvt"].notna().all()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if write_ledger:
        ledger_append(path, method, local_rmse, note)
    print("built", path.name, "rows", len(out), "mean", round(float(out["TAXITIME_SEC_mvt"].mean()), 1), "sha", sha[:16])
    return path


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
