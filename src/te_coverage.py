# SPDX-License-Identifier: GPL-3.0-only
"""그룹 통계(src/te.py) 칸마다 값이 있는 몫(non_null)과, 그 행의 키가 통계를 만든 학습 행(정답 있는 행)에 한 번이라도 나온 몫(key_seen)을 잰다.

te_oof 는 학습 밖 행에 (합 + prior*k)/(수 + k) 를 준다. 처음 보는 키도 prior 로 채워지므로 non_null 은 정의상 1 이고,
자와 시험의 차이는 key_seen(키가 학습 행에 있었나, 없으면 값이 prior 그대로)으로 드러난다.
key_seen 은 te.te_oof 를 감싸 add_te 가 실제로 넘기는 키와 대상(NaN 뺀 학습 행)으로 잰다. 키 정의를 따로 옮겨 적지 않는다.

설정:
  holdout: 2025-01, 07 출발 행. 통계는 검증 설정(1, 7월과 시험 행을 뺀 2025)으로
  test   : ranking 출발(src 0) 중 data/submitting.parquet 에 있는 행. 통계는 full 설정(2025 전체)으로
  pseudo : 달 쌍 (2, 8), (3, 9), (4, 10), (5, 11), (6, 12) 마다 그 두 달 행. 통계는 나머지 2025 달로
사용: python -u src/te_coverage.py [피처 판, 기본 feats_v25]  ->  notes/te_coverage_<피처 판>.json
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
import src.te as te  # noqa: E402
from src.load import CACHE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
PAIRS = [(2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
_orig_te_oof = te.te_oof
_cur = {}


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _spy(key, target, month, train_mask, k=20.0, prior=None):
    out = _orig_te_oof(key, target, month, train_mask, k, prior)
    use = train_mask & ~np.isnan(target)
    _cur["seen"].append(float(np.isin(key[_cur["eval"]], np.unique(key[use])).mean()))
    return out


te.te_oof = _spy  # add_te 는 모듈 전역 te_oof 를 부른다


def coverage(F: pd.DataFrame, fit_rows: np.ndarray, eval_rows: np.ndarray) -> dict:
    _cur["eval"], _cur["seen"] = eval_rows, []
    T = te.add_te(F, fit_rows)
    assert len(_cur["seen"]) == T.shape[1]
    nn = (~np.isnan(T.to_numpy()[eval_rows])).mean(axis=0)
    cols = {c: {"non_null": round(float(n), 4), "key_seen": round(s, 4)} for c, n, s in zip(T.columns, nn, _cur["seen"])}
    return {"rows": int(eval_rows.sum()), "fit_rows": int(fit_rows.sum()), "columns": cols}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "feats_v25"
    p = CACHE / f"{name}.parquet"
    F = pd.read_parquet(p)
    src = F["src"].to_numpy()
    sub = pd.read_parquet(ROOT / "data" / "submitting.parquet", columns=["MVT_ID_mvt"])["MVT_ID_mvt"]
    test_rows = (src == 0) & F["MVT_ID_mvt"].isin(sub).to_numpy()
    out = {"feats": name, "feats_sha": sha16(p), "te_py_sha": sha16(ROOT / "src" / "te.py"),
           "definition": "non_null: share of eval rows with a non-NaN TE value. key_seen: share of eval rows whose group key occurs "
                         "among the fit rows with a non-NaN target for that column (otherwise the value is the prior). "
                         "te_oof fills unseen keys with the prior, so non_null is 1 by construction.",
           "settings": {}}
    out["settings"]["holdout_2025_01_07"] = coverage(F, ~np.isin(src, [1, 7]) & (src != 0), np.isin(src, [1, 7]))
    print("holdout done", flush=True)
    out["settings"]["test_ranking_in_submitting"] = coverage(F, src != 0, test_rows)
    out["settings"]["test_ranking_in_submitting"]["ranking_rows"] = int((src == 0).sum())
    print("test done", flush=True)
    for a, b in PAIRS:
        out["settings"][f"pseudo_{a:02d}_{b:02d}"] = coverage(F, ~np.isin(src, [a, b]) & (src != 0), np.isin(src, [a, b]))
        print(f"pseudo {a} {b} done", flush=True)
    # 칸마다 한 줄 요약: 자, 시험, 가짜 자 다섯의 최소와 최대(key_seen)
    summ = {}
    for c in out["settings"]["holdout_2025_01_07"]["columns"]:
        ps = [out["settings"][f"pseudo_{a:02d}_{b:02d}"]["columns"][c]["key_seen"] for a, b in PAIRS]
        summ[c] = {"holdout": out["settings"]["holdout_2025_01_07"]["columns"][c]["key_seen"],
                   "test": out["settings"]["test_ranking_in_submitting"]["columns"][c]["key_seen"],
                   "pseudo_min": min(ps), "pseudo_max": max(ps)}
    out["key_seen_summary"] = summ
    out["written_kst"] = dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")
    dst = ROOT / "notes" / f"te_coverage_{name}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))
    print("wrote", dst.name, out["written_kst"])


if __name__ == "__main__":
    main()
