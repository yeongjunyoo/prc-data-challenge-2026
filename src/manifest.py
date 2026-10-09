# SPDX-License-Identifier: GPL-3.0-only
"""깨끗한 후보 하나의 산출물과 매니페스트를 쓴다(10-02 묶음 규칙).

사용: python -m src.manifest <spec> <run_tag> <col> <seeds 쉼표> <feats> <jobs 파일>
  예: python -m src.manifest m4br_avg2 m4br24 pred 0,1 feats_v24 notes/jobs_v24_batch.txt
쓰는 것:
  preds/holdout_<spec>_rmse<X>.parquet (+json)   본 자 2025-01, 07 출발, cache/val_<run>_s<seed>.parquet 의 시드 평균(0 아래 자르기)
  preds/test_<spec>.parquet (+json)              cache/full_<run>_s<seed>.parquet 의 시드 평균, submitting.parquet 순서와 dtype
  preds/fwd_<spec>.parquet                       cache/fwd_<run>_s<seed>.parquet 의 시드 평균(2025-07 출발 전부)
  preds/es_valid_ids_<spec>.parquet              본 자 실행의 조기 종료 행(시드끼리 같은지, 자와 겹침 0 인지 확인)
  preds/manifest_<spec>.json                     config, commands, 해시, 코드 스냅숏(preds/code/<sha16>_<파일>)
"""
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, ".")
from src.ruler import RULER_MONTHS, ruler_frame, ruler_hash, rmse_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "cache"
P = ROOT / "preds"
CODE = ["src/model3.py", "src/feats.py", "src/te.py", "src/load.py", "src/sub_g.py", "src/common.py", "src/cat_mu.py", "src/blend_mu.py", "src/g_band.py", "src/xgb_mu.py", "src/rmlp_mu.py", "src/lobt_window.py", "src/adsb_corr.py", "src/feats_adsb.py", "src/adsb_geo.py",
        "src/blend_oof.py", "src/ctx_corr.py", "src/feats_ctx.py"]
# 그 후보만의 다른 코드(연구 폴더의 입력 고정 스크립트 등)는 환경 변수 MANIFEST_EXTRA_CODE(쉼표 구분 경로)로 더한다. 없으면 위 목록 그대로다.
CODE += [f.strip() for f in os.environ.get("MANIFEST_EXTRA_CODE", "").split(",") if f.strip()]
sys.stdout.reconfigure(encoding="utf-8")


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def seed_avg(setting, run, seeds, col):
    ds = [pd.read_parquet(C / f"{setting}_{run}_s{s}.parquet", columns=["MVT_ID_mvt", col]) for s in seeds]
    ids = ds[0]["MVT_ID_mvt"].to_numpy()
    for d in ds[1:]:
        assert (d["MVT_ID_mvt"].to_numpy() == ids).all()
    return pd.DataFrame({"MVT_ID_mvt": ids, "pred": np.mean([np.clip(d[col].to_numpy(), 0, None) for d in ds], axis=0)})


def jobs_for(jobs_file: str, run: str):
    out = {"val": [], "full": [], "fwd": []}
    for ln in (ROOT / jobs_file).read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        cmd = ln.split("\t", 1)[1]
        argv = shlex.split(cmd)
        if f"--tag" in argv and argv[argv.index("--tag") + 1] == run:
            out[argv[argv.index("--setting") + 1]].append(cmd)
    return out


def config_of(cmd: str) -> dict:
    argv = shlex.split(cmd)
    argv = argv[next(i for i, a in enumerate(argv) if a.startswith("--")):]  # 실행 파일, -u, src/model3.py 뒤의 플래그
    cfg, i = {}, 0
    while i < len(argv):
        k = argv[i]
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            cfg[k] = argv[i + 1]
            i += 2
        else:
            cfg[k] = True
            i += 1
    for k in ("--setting", "--seed"):
        cfg.pop(k, None)
    return cfg


def main():
    spec, run, col, seeds_s, feats, jobs_file = sys.argv[1:7]
    seeds = [int(s) for s in seeds_s.split(",")]
    rf = ruler_frame()
    ruler = rf[rf["month"].isin(RULER_MONTHS)]
    july = rf[rf["month"] == 7]
    # 코드: 지금 파일이 스냅숏과 같아야 한다
    code_sha = {}
    for f in CODE:
        h = sha16(ROOT / f)
        snap = P / "code" / f"{h}_{Path(f).name}"
        if not snap.exists():
            snap.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / f, snap)
        code_sha[f] = {"sha": h, "snapshot": rel(snap)}
    # 조기 종료 행
    es = [pd.read_parquet(P / f"es_valid_ids_{run}_val_s{s}.parquet")["MVT_ID_mvt"].to_numpy() for s in seeds]
    for e in es[1:]:
        assert np.array_equal(np.sort(e), np.sort(es[0])), "시드끼리 조기 종료 행이 다르다"
    overlap = int(pd.Index(es[0]).isin(ruler.index).sum())
    assert overlap == 0, f"조기 종료 행이 자와 겹친다: {overlap}"
    esp = P / f"es_valid_ids_{spec}.parquet"
    pd.DataFrame({"MVT_ID_mvt": es[0]}).to_parquet(esp, index=False)
    # 본 자
    h = seed_avg("val", run, seeds, col)
    tmp = P / f"_tmp_holdout_{spec}.parquet"
    h.to_parquet(tmp, index=False)
    r_main = rmse_of(rel(tmp), ruler)
    hp = P / f"holdout_{spec}_rmse{r_main:.2f}.parquet"
    tmp.replace(hp)
    # 앞으로 가는 자
    f = seed_avg("fwd", run, seeds, col)
    fp = P / f"fwd_{spec}.parquet"
    f.to_parquet(fp, index=False)
    r_fwd = rmse_of(rel(fp), july)
    # 시험
    t = seed_avg("full", run, seeds, col).set_index("MVT_ID_mvt")["pred"]
    tpl = pd.read_parquet(ROOT / "data" / "submitting.parquet", columns=["MVT_ID_mvt"])
    assert len(t) == len(tpl) and set(t.index) == set(tpl["MVT_ID_mvt"])
    tv = t.reindex(tpl["MVT_ID_mvt"].to_numpy()).to_numpy(dtype="float64")
    assert np.isfinite(tv).all() and (tv >= 0).all()
    test = pd.DataFrame({"MVT_ID_mvt": tpl["MVT_ID_mvt"].to_numpy(), "pred": tv})
    assert test["MVT_ID_mvt"].dtype == pq.read_schema(ROOT / "data" / "submitting.parquet").field("MVT_ID_mvt").type.to_pandas_dtype()
    tp = P / f"test_{spec}.parquet"
    test.to_parquet(tp, index=False)
    jobs = jobs_for(jobs_file, run)
    assert len(jobs["val"]) == len(jobs["full"]) == len(jobs["fwd"]) == len(seeds), jobs
    cfg = config_of(jobs["val"][0])
    for kind in ("val", "full", "fwd"):
        for cmd in jobs[kind]:
            assert config_of(cmd) == cfg, (kind, cmd)
    feats_sha = sha16(C / f"{feats}.parquet")
    now = dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")
    hold_json = {"file": hp.name, "spec": spec, "rows": int(len(h)), "local_rmse": round(r_main, 2), "seeds": seeds, "column": col,
                 "feats": feats, "es_valid_ids": rel(esp), "commands": jobs["val"], "written_kst": now}
    hp.with_suffix(".json").write_text(json.dumps(hold_json, ensure_ascii=False, indent=1), encoding="utf-8")
    test_json = {"file": tp.name, "spec": spec, "n_rows": int(len(test)), "pred_mean": round(float(tv.mean()), 2),
                 "pred_sd": round(float(tv.std()), 2), "pred_max": round(float(tv.max()), 2), "holdout_file": rel(hp),
                 "holdout_local_rmse": round(r_main, 2), "feats": feats, "es_valid_ids": rel(esp), "forward": rel(fp),
                 "forward_rmse_2025_07": round(r_fwd, 2), "command": jobs["full"], "written_kst": now}
    tp.with_suffix(".json").write_text(json.dumps(test_json, ensure_ascii=False, indent=1), encoding="utf-8")
    man = {"spec": spec, "run_tag": run, "column": col, "seeds": seeds, "feats": feats, "feats_sha": feats_sha,
           "config": cfg, "commands": {"holdout": jobs["val"], "test": jobs["full"], "forward": jobs["fwd"]},
           "command": jobs["val"][0],
           "code_sha": {k: v["sha"] for k, v in code_sha.items()}, "code_snapshots": {k: v["snapshot"] for k, v in code_sha.items()},
           "holdout": {"path": rel(hp), "sha": sha16(hp), "rmse": round(r_main, 2), "ruler_hash": ruler_hash(ruler.index.to_numpy())},
           "test": {"path": rel(tp), "sha": sha16(tp)},
           "forward": {"path": rel(fp), "sha": sha16(fp), "rmse_2025_07": round(r_fwd, 2)},
           "es_valid_ids": {"path": rel(esp), "sha": sha16(esp), "rows": int(len(es[0])), "overlap_with_ruler": overlap},
           "iteration_rule": "internal early stopping on train-range days (UTC day % 10 == 0); holdout and forward refit on all train-range rows with iterations x 1.1; test (full 2025) uses the same iterations x 1.1 (--full_mult 1.0)",
           "written_kst": now}
    mp = P / f"manifest_{spec}.json"
    mp.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(man, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
