# SPDX-License-Identifier: GPL-3.0-only
"""notes/ruler_checks.json 을 sign_agree 가 읽는 꼴로 쓴다(10-02 18:50 인터페이스).

{"ruler_hash": src.ruler.ruler_hash(2025-01, 07 출발 MVT_ID),
 "forward_preds": {tag: "preds/fwd_<tag>.parquet"},     2025-01~06 학습(내부 조기 종료 + 재학습), 2025-07 출발 전부 예측
 "main_holdouts": {tag: "preds/holdout_...parquet"},     같은 설정의 본 자(2025-01, 07) 예측
 "es_valid_ids": {tag: "preds/es_valid_ids_<tag>.parquet"},  본 자 실행의 조기 종료 행
 "written_kst": ...}
각 태그의 fwd 와 holdout 파일은 cache/<setting>_<run>_s<seed>.parquet 의 시드 평균(0 아래 자르기)에서 만든다.
판정(RMSE, 부호 일치)은 도구가 src.ruler.rmse_of 로 다시 계산한다. 여기서는 참고로 같은 값을 details 에 남긴다.
"""
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.ruler import RULER_MONTHS, ruler_frame, ruler_hash, rmse_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "cache"
sys.stdout.reconfigure(encoding="utf-8")
# 태그 -> (실행 태그, 시드들, 칸, 피처 판). 방식 r: 내부 조기 종료 + 재학습(--refit), i: 내부 조기 종료만(조기 종료 날은 학습에서 빠짐)
SCHEME = sys.argv[1] if len(sys.argv) > 1 else "r"
SPECS = {"m3_directG": (f"m3{SCHEME}", [0], "direct_G", "feats_v2"),
         "m4b_avg2": (f"m4b{SCHEME}", [0, 1], "pred", "feats_v21"),
         "m6_avg2": (f"m6{SCHEME}", [0, 1], "pred", "feats_v22")}
SCHEME_TEXT = {"r": "early stopping on train-range days (UTC day % 10 == 0), then refit on all train-range rows with iterations x 1.1 (src/model3.py --es internal --refit)",
               "i": "early stopping on train-range days (UTC day % 10 == 0); those days are left out of training (src/model3.py --es internal)"}


def seed_avg(setting, run, seeds, col):
    ds = [pd.read_parquet(C / f"{setting}_{run}_s{s}.parquet") for s in seeds]
    ids = ds[0]["MVT_ID_mvt"].to_numpy()
    for d in ds[1:]:
        assert (d["MVT_ID_mvt"].to_numpy() == ids).all()
    return pd.DataFrame({"MVT_ID_mvt": ids, "pred": np.mean([np.clip(d[col].to_numpy(), 0, None) for d in ds], axis=0)})


def from_manifests(specs: list[str]):
    """깨끗한 후보 매니페스트(preds/manifest_<spec>.json)의 파일을 그대로 가리킨다(새 파일을 쓰지 않는다)."""
    rf = ruler_frame()
    ruler = rf[rf["month"].isin(RULER_MONTHS)]
    july = rf[rf["month"] == 7]
    fwd, main_h, esv, det = {}, {}, {}, {}
    for s in specs:
        m = json.loads((ROOT / "preds" / f"manifest_{s}.json").read_text(encoding="utf-8"))
        fwd[s], main_h[s], esv[s] = m["forward"]["path"], m["holdout"]["path"], m["es_valid_ids"]["path"]
        es_ids = pd.read_parquet(ROOT / esv[s])["MVT_ID_mvt"]
        det[s] = {"manifest": f"preds/manifest_{s}.json", "feats": m["feats"], "main_rmse": round(rmse_of(main_h[s], ruler), 2),
                  "forward_rmse": round(rmse_of(fwd[s], july), 2), "es_overlap_with_ruler": int(es_ids.isin(ruler.index).sum())}
    pairs = {}
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            a, b = specs[i], specs[j]
            dm = det[b]["main_rmse"] - det[a]["main_rmse"]
            dfw = det[b]["forward_rmse"] - det[a]["forward_rmse"]
            pairs[f"{a}__{b}"] = {"delta_main": round(dm, 2), "delta_forward": round(dfw, 2), "sign_agrees": bool(np.sign(dm) == np.sign(dfw))}
    out = {"ruler_hash": ruler_hash(ruler.index.to_numpy()), "forward_preds": fwd, "main_holdouts": main_h, "es_valid_ids": esv,
           "details": {"scheme": SCHEME_TEXT["r"] + "; test uses the same iterations (--full_mult 1.0)", "forward": "train 2025-01..06, measure all 2025-07 DEP rows",
                       "models": det, "pairs": pairs},
           "written_kst": dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")}
    (ROOT / "notes" / "ruler_checks.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main():
    if SCHEME == "manifest":
        return from_manifests(sys.argv[2:])
    rf = ruler_frame()
    ruler = rf[rf["month"].isin(RULER_MONTHS)]
    july = rf[rf["month"] == 7]
    h = ruler_hash(ruler.index.to_numpy())
    fwd, main_h, esv, det = {}, {}, {}, {}
    for tag, (run, seeds, col, feats) in SPECS.items():
        f = seed_avg("fwd", run, seeds, col)
        fp = ROOT / "preds" / f"fwd_{tag}.parquet"
        f.to_parquet(fp, index=False)
        m = seed_avg("val", run, seeds, col)
        r_main = float(np.sqrt(((m.set_index("MVT_ID_mvt")["pred"].reindex(ruler.index) - ruler["TAXITIME_SEC_mvt"]) ** 2).mean()))
        mp = ROOT / "preds" / f"holdout_{run}_{tag.split('_', 1)[1]}_rmse{r_main:.2f}.parquet"
        m.to_parquet(mp, index=False)
        ep = ROOT / "preds" / f"es_valid_ids_{run}_{tag.split('_', 1)[1]}.parquet"
        shutil.copyfile(ROOT / "preds" / f"es_valid_ids_{run}_val_s{seeds[0]}.parquet", ep)
        es_ids = pd.read_parquet(ep)["MVT_ID_mvt"]
        overlap = int(es_ids.isin(ruler.index).sum())
        fwd[tag] = str(fp.relative_to(ROOT)).replace("\\", "/")
        main_h[tag] = str(mp.relative_to(ROOT)).replace("\\", "/")
        esv[tag] = str(ep.relative_to(ROOT)).replace("\\", "/")
        det[tag] = {"run": run, "seeds": seeds, "column": col, "feats": feats, "main_rmse": round(rmse_of(main_h[tag], ruler), 2),
                    "forward_rmse": round(rmse_of(fwd[tag], july), 2), "fwd_rows": int(len(f)), "es_ids": int(len(es_ids)),
                    "es_overlap_with_ruler": overlap}
    tags = list(SPECS)
    pairs = {}
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a, b = tags[i], tags[j]
            dm = det[b]["main_rmse"] - det[a]["main_rmse"]
            dfw = det[b]["forward_rmse"] - det[a]["forward_rmse"]
            pairs[f"{a}__{b}"] = {"delta_main": round(dm, 2), "delta_forward": round(dfw, 2), "sign_agrees": bool(np.sign(dm) == np.sign(dfw))}
    out = {"ruler_hash": h, "forward_preds": fwd, "main_holdouts": main_h, "es_valid_ids": esv,
           "details": {"scheme": SCHEME_TEXT[SCHEME], "forward": "train 2025-01..06, measure all 2025-07 DEP rows", "models": det, "pairs": pairs},
           "written_kst": dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")}
    (ROOT / "notes" / "ruler_checks.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
