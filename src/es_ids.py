# SPDX-License-Identifier: GPL-3.0-only
"""이미 끝난 내부 조기 종료 실행(m4bi, m6i 의 val 시드 0, 1)의 조기 종료 행을 model3 와 같은 함수로 다시 만들어 남긴다.

model3.py 는 10-02 18:45:40 KST(그 수정의 파일 시각) 이후 실행부터 preds/es_valid_ids_<tag>_<setting>_s<seed>.parquet 를 직접 쓴다.
그렇게 쓴 첫 파일은 preds/es_valid_ids_m4bi_fwd_s0.parquet(18:46:02)이다.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.load import CACHE  # noqa: E402
from src.model3 import es_day_mask  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
for tag, feats in [("m4bi", "feats_v21"), ("m6i", "feats_v22")]:
    F = pd.read_parquet(CACHE / f"{feats}.parquet", columns=["MVT_ID_mvt", "src"])
    src = F["src"].to_numpy()
    fit_rows = ~np.isin(src, [1, 7]) & (src != 0)
    es_rows = fit_rows & es_day_mask(F["MVT_ID_mvt"].to_numpy())
    for seed in [0, 1]:
        p = CACHE.parent / "preds" / f"es_valid_ids_{tag}_val_s{seed}.parquet"
        pd.DataFrame({"MVT_ID_mvt": F["MVT_ID_mvt"].to_numpy()[es_rows]}).to_parquet(p, index=False)
        print(p.name, int(es_rows.sum()), "months", sorted(set(src[es_rows].tolist())))
