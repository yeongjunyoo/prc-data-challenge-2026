# SPDX-License-Identifier: GPL-3.0-only
# G7 승격 판정: 로컬 자(2025-01, 07 출발) 위에서 두 예측의 RMSE 차이 Δ 와 날짜 블록 부트스트랩 표준편차를 낸다.
# |Δ| 가 표준편차의 2배 이상일 때만 로컬로 우열이 갈린다. 아니면 서버 쿼터로 판정한다.
# 이 도구는 분해능만 판정한다. 후보 선택 권한은 notes/sign_agreement.json 의 local_selection_allowed(닫힘 증거와 닫힘 감사)가 정한다(private notes §0 G2).
# 분해능은 날짜 블록 부트스트랩과 시드 잡음(notes/seed_noise.json 의 단일 시드 표준편차 * sqrt(1/ka + 1/kb))을 합친 표준편차다.
#     python -m src.compare preds/holdout_A.parquet preds/holdout_B.parquet [--ka 1] [--kb 2] [--reps 300]   (ka, kb = 평균한 시드 수)
# 예측 파일 칸: MVT_ID_mvt, pred(초). 로컬 자의 행과 하나도 빠짐없이 맞아야 한다.
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def holdout() -> pd.DataFrame:
    cols = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
    tr = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/training_*.parquet"))], ignore_index=True)
    tr = tr[(tr["PHASE_mvt"] == "DEP") & tr["MVT_TIME_UTC_mvt"].dt.month.isin([1, 7])]
    return tr.set_index("MVT_ID_mvt")


def load(path: str, ids: pd.Index) -> np.ndarray:
    s = pd.read_parquet(path).set_index("MVT_ID_mvt")["pred"]
    v = s.reindex(ids)
    if v.isna().any() or len(s) != len(ids):
        raise SystemExit(f"{path}: 로컬 자와 행이 안 맞는다(빠진 행 {int(v.isna().sum())}, 파일 {len(s)} 대 자 {len(ids)})")
    return v.to_numpy(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--ka", type=int, default=1, help="a 가 평균한 시드 수")
    ap.add_argument("--kb", type=int, default=1, help="b 가 평균한 시드 수")
    args = ap.parse_args()
    h = holdout()
    y = h["TAXITIME_SEC_mvt"].to_numpy(float)
    se_a, se_b = (load(args.a, h.index) - y) ** 2, (load(args.b, h.index) - y) ** 2
    day = h["MVT_TIME_UTC_mvt"].dt.date.to_numpy()
    days = np.unique(day)
    idx = {d: np.flatnonzero(day == d) for d in days}
    rng = np.random.default_rng(20261002)
    rd = []
    for _ in range(args.reps):
        pick = np.concatenate([idx[d] for d in rng.choice(days, size=len(days), replace=True)])
        rd.append(np.sqrt(se_b[pick].mean()) - np.sqrt(se_a[pick].mean()))
    ra, rb = float(np.sqrt(se_a.mean())), float(np.sqrt(se_b.mean()))
    sd_boot = float(np.std(rd))
    seed = json.loads(Path("notes/seed_noise.json").read_text(encoding="utf-8"))
    sd_seed = float(seed["sd_single_seed"] * np.sqrt(1 / args.ka + 1 / args.kb))
    sd = float(np.sqrt(sd_boot ** 2 + sd_seed ** 2))
    tag = "분해능 밖"
    out = {"a": args.a, "b": args.b, "rows": int(len(y)), "days": int(len(days)), "reps": args.reps,
           "rmse_a": round(ra, 2), "rmse_b": round(rb, 2), "delta_b_minus_a": round(rb - ra, 2), "sd_delta": round(sd, 2),
           "sd_boot": round(sd_boot, 2), "sd_seed": round(sd_seed, 2), "ka": args.ka, "kb": args.kb,
           "verdict": (f"b 가 낫다({tag})" if rb - ra <= -2 * sd else f"a 가 낫다({tag})" if rb - ra >= 2 * sd
                       else "분해능 안: 서버 쿼터로 판정")}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    stem = lambda p: p.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".parquet")  # noqa: E731
    f = Path("notes") / "compare" / f"{stem(args.a)}__{stem(args.b)}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
