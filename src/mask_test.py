# SPDX-License-Identifier: GPL-3.0-only
"""가림 시험: 2025-01, 2025-07 출발 행의 t_blk 와 y 를 NaN 으로 가리고 피처를 다시 만든 뒤, 그 행들의 피처가 가리지 않은 판과 같은지 본다.

다른 칸이 하나라도 있으면 검증 행이 같은 기간 출발편의 BLOCK 이나 TAXITIME 을 보고 있다는 뜻이다(시험 행은 못 본다).
도착 행과 다른 달은 건드리지 않는다. 그룹 통계(src/te.py, 모델이 검증 설정에서 만드는 칸)도 같은 방식으로 비교한다.

가린 판:
  v2, v21, v22, v23 -> cache/feats_mask.parquet      (v2.3 코드로 만든 것. 옛 판의 칸은 v2.3 에 같은 정의로 들어 있다)
  v24              -> cache/feats_mask_v24.parquet  (v2.4 코드, 달 칸막이와 METAR as-of)
  v25, v26         -> cache/feats_mask_<판>.parquet (v2.5, v2.6 코드. v2.6 은 이름으로 켜진다: src/feats.py V26_NAMES)
사용: python -m src.mask_test v2 v21 v22 v23 v24   ->  notes/mask_test_feats_<판>.json (feats_sha, masked_sha 포함)
"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import src.feats as feats  # noqa: E402
from src.load import CACHE, load_all as _orig_load  # noqa: E402
from src.te import add_te  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
HOLD = [1, 7]
MASKED = {"v2": "feats_mask", "v21": "feats_mask", "v22": "feats_mask", "v23": "feats_mask", "v24": "feats_mask_v24",
          "v25": "feats_mask_v25", "v26": "feats_mask_v26"}


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def masked_load(rebuild: bool = False):
    df = _orig_load()
    m = (df["PHASE_mvt"] == "DEP").to_numpy() & df["src"].isin(HOLD).to_numpy()
    df.loc[m, "t_blk"] = np.nan
    df.loc[m, "y"] = np.nan
    print("masked DEP holdout rows:", int(m.sum()), flush=True)
    return df


def same(a: pd.Series, b: pd.Series) -> bool:
    if isinstance(a.dtype, pd.CategoricalDtype) or isinstance(b.dtype, pd.CategoricalDtype):
        a = a.astype(str)
        b = b.astype(str)
        return bool(((a == b) | (a.isna() & b.isna())).all())
    x = a.to_numpy(dtype="float64", na_value=np.nan)
    y = b.to_numpy(dtype="float64", na_value=np.nan)
    return bool(((x == y) | (np.isnan(x) & np.isnan(y))).all())


def compare(Fm: pd.DataFrame, Fu: pd.DataFrame, rows: np.ndarray, cols: list[str]) -> list[str]:
    return [c for c in cols if not same(Fm.loc[rows, c].reset_index(drop=True), Fu.loc[rows, c].reset_index(drop=True))]


def main():
    versions = sys.argv[1:] or ["v2", "v21", "v22", "v23", "v24"]
    for v in versions:
        mname = MASKED[v]
        p = CACHE / f"{mname}.parquet"
        if not p.exists():
            assert v in ("v24", "v25", "v26"), "v2~v23 의 가린 판은 v2.3 코드로 만든 feats_mask.parquet 를 쓴다"
            feats.load_all = masked_load  # build 안에서 부르는 load_all 을 바꿔 끼운다
            feats.build(rebuild=True, name=mname)
            feats.load_all = _orig_load
        Fm = pd.read_parquet(p)
        hold = Fm["src"].isin(HOLD).to_numpy()
        up = CACHE / f"feats_{v}.parquet"
        Fu = pd.read_parquet(up)
        assert (Fu["MVT_ID_mvt"].to_numpy() == Fm["MVT_ID_mvt"].to_numpy()).all()
        cols = [c for c in Fu.columns if c != "y"]
        missing = [c for c in cols if c not in Fm.columns]
        diff = compare(Fm, Fu, hold, [c for c in cols if c in Fm.columns])
        # 모델이 검증 설정(1, 7월 제외 학습)에서 만드는 그룹 통계도 같은지
        fit_rows = ~np.isin(Fu["src"].to_numpy(), HOLD + [0])
        Tu = add_te(Fu, fit_rows)
        Tm = add_te(Fm[[c for c in Fu.columns if c in Fm.columns]], fit_rows)
        te_diff = compare(Tm, Tu, hold, list(Tu.columns))
        out = {"feats": f"feats_{v}", "rows": int(hold.sum()), "differing_columns": diff, "missing_in_masked_build": missing,
               "te_columns_checked": len(Tu.columns), "differing_te_columns_val_setting": te_diff,
               "feats_sha": sha16(up), "masked_sha": sha16(p),
               "masked_build": f"cache/{mname}.parquet (2025-01, 07 DEP t_blk and y set to NaN before src/feats.py build)",
               "written_kst": dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")}
        (ROOT / "notes" / f"mask_test_feats_{v}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
