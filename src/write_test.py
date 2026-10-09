# SPDX-License-Identifier: GPL-3.0-only
"""시험 예측을 preds/test_<tag>_s<seed>.parquet 로 쓴다(MVT_ID_mvt 는 submitting.parquet 와 같은 dtype, 같은 순서, pred 초, 0 아래 자르기).

곁에 json: 짝이 되는 로컬 자 RMSE, 재현 명령, 행 수, pred 평균, 표준편차, 최댓값.
사용: python src/write_test.py <cache_file> <col> <tag> <seed> <holdout_rmse> <holdout_file> "<command>"
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")


def main():
    cache_file, col, tag, seed, hold_rmse, hold_file, cmd = sys.argv[1:8]
    src = pd.read_parquet(ROOT / "cache" / cache_file, columns=["MVT_ID_mvt", col]).set_index("MVT_ID_mvt")[col]
    tpl = pd.read_parquet(ROOT / "data" / "submitting.parquet", columns=["MVT_ID_mvt"])
    assert len(tpl) == 344841 and src.index.is_unique and len(src) == len(tpl), (len(src), len(tpl))
    assert set(src.index) == set(tpl["MVT_ID_mvt"]), "MVT_ID 집합 불일치"
    pred = np.clip(src.reindex(tpl["MVT_ID_mvt"].to_numpy()).to_numpy(dtype="float64"), 0, None)
    assert np.isfinite(pred).all()
    out = pd.DataFrame({"MVT_ID_mvt": tpl["MVT_ID_mvt"].to_numpy(), "pred": pred})
    assert out["MVT_ID_mvt"].dtype == pq.read_schema(ROOT / "data" / "submitting.parquet").field("MVT_ID_mvt").type.to_pandas_dtype()
    name = f"test_{tag}_s{seed}"
    p = ROOT / "preds" / f"{name}.parquet"
    out.to_parquet(p, index=False)
    back = pd.read_parquet(p)
    assert back["MVT_ID_mvt"].equals(tpl["MVT_ID_mvt"]) and len(back) == 344841
    meta = {"file": p.name, "source": f"cache/{cache_file}:{col}", "n_rows": int(len(out)),
            "pred_mean": round(float(pred.mean()), 2), "pred_sd": round(float(pred.std()), 2),
            "pred_max": round(float(pred.max()), 2), "pred_min": round(float(pred.min()), 2),
            "n_pred_over_3600": int((pred > 3600).sum()), "holdout_local_rmse": float(hold_rmse),
            "holdout_file": hold_file, "command": cmd}
    (ROOT / "preds" / f"{name}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
