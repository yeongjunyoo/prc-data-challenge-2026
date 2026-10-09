# SPDX-License-Identifier: GPL-3.0-only
"""cache/full_<tag>_s<seed>.parquet 들(시드 평균)을 preds/<name>.parquet(MVT_ID_mvt, pred 초)로 쓴다.

검사: 행 수 344,841, submitting.parquet 의 MVT_ID 와 같은 집합, 결측 없음.
사용: python src/write_pred.py <name> <col> <file1> [<file2> ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")


def main():
    name, col, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    parts = [pd.read_parquet(ROOT / "cache" / f, columns=["MVT_ID_mvt", col]).set_index("MVT_ID_mvt")[col] for f in files]
    pred = sum(parts) / len(parts)
    tpl = pd.read_parquet(ROOT / "data" / "submitting.parquet", columns=["MVT_ID_mvt"])
    assert len(pred) == len(tpl) == 344841, (len(pred), len(tpl))
    assert set(pred.index) == set(tpl["MVT_ID_mvt"]), "MVT_ID 불일치"
    pred = pred.reindex(tpl["MVT_ID_mvt"].to_numpy())
    assert pred.notna().all()
    out = pd.DataFrame({"MVT_ID_mvt": pred.index.to_numpy(), "pred": np.clip(pred.to_numpy(), 0, None)})
    (ROOT / "preds").mkdir(exist_ok=True)
    p = ROOT / "preds" / f"{name}.parquet"
    out.to_parquet(p, index=False)
    print("wrote", p.name, "rows", len(out), "mean", round(float(out["pred"].mean()), 1), "max", round(float(out["pred"].max()), 1),
          "n>3600", int((out["pred"] > 3600).sum()))


if __name__ == "__main__":
    main()
