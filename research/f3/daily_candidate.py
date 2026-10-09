# SPDX-License-Identifier: GPL-3.0-only
"""F3 server check of the first final file (10-08 23:1x): its 344,841 Jan/Jul rows as a daily-format candidate, NOT fired.
In the final root (cwd), from preds/test_<spec>.parquet (the chain's float predictions, 670,790 rows):
  <out dir>/<name>.parquet   MVT_ID_mvt (double), pred (double): the public-phase template's ids in its order (as 514's queue parquet)
  <out dir>/<name>.json      queue-style entry: kind, method, local_rmse (the holdout of the manifest), manifest, note, readiness
  <out dir>/composition_<name>.json  src/composition.py (readiness, G2) run in a sandbox root whose data/ranking and data/submitting are the
                             public-phase files, with the final feats_v27 and the final manifest's holdout
  checks in the json: rows, ids complete and unique, NaN and negatives 0, the move against 514's float test predictions and v35 (int) on the
  same ids, by airport group; the expected daily score: about 228.6 to 228.7 against 514's 228.384, a gap over about 1 point
  means a rebuild error.
    python -X utf8 research/f3/daily_candidate.py <spec> <name> <old ranking> <old submitting> <514 test parquet> <v35 parquet> <out dir>
The second rebuild's daily check (581 = 592's Jan/Jul rows, 10-10 00:2x) uses name prefix f3f2: the references are 531's float test
and v36 in the same argument slots, and TEXTS below gives its labels and texts (f3f1 = the first final file, unchanged).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MINRES = ["EDDM", "EGLL", "LFPG", "LIRF", "LTFM"]
TEXTS = {
    "f3f1": dict(refs=("vs_514_float", "vs_v35_int"), same="514",
                 method=("F3 최종 재구축의 첫 파일(514 를 final_ranking 위에 그대로 다시 지음, research/f3, private notes research/2026-10-08-f3-jitter.md)의 "
                         "1, 7월 344,841행을 공개 단계 틀의 순서로 옮긴 일일 형식 후보. 서버 확인용이고 쏘지 않는다(10-08 23:1x). "
                         "기대: 514 의 228.384 에서 다시 뽑힌 시각만큼 움직인 228.6 ~ 228.7 근처, 1점 넘게 벌어지면 재구축 오류.")),
    "f3f3": dict(refs=("vs_570_float", "vs_570_int_rounded"), same="570",
                 method=("F3 L3 판 결선의 기준 596(L3S = 570 의 꼴을 final_ranking 위에 다시 지음, research/f3, private notes research/2026-10-08-f3-jitter.md "
                         "5.4 절)의 1, 7월 344,841행을 공개 단계 틀의 순서로 옮긴 일일 형식 후보(597). 598a 의 1, 7월은 596 과 경계 오프블록 묶음 밖에서 "
                         "비트까지 같다. 쏘지 않는다(10-10 03:5x). 570 은 아직 쏘지 않아 정수 참조는 570 의 실수 예측을 build_final 식으로 "
                         "반올림한 파일이다.")),
    "f3f2": dict(refs=("vs_531_float", "vs_v36_int"), same="531",
                 method=("F3 두 번째 재구축의 기본안 592(531 을 final_ranking 위에 다시 지음, research/f3, private notes research/2026-10-08-f3-jitter.md)의 "
                         "1, 7월 344,841행을 공개 단계 틀의 순서로 옮긴 일일 형식 후보(581). 결선 꼴 594a 의 1, 7월 행은 592 와 비트까지 같아야 하므로 "
                         "이 한 발이 결선의 1, 7월 절반을 서버에서 확인한다. 쏘지 않는다(10-10 00:2x).")),
}
PY = sys.executable


def sha16(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
    spec, name, old_rk, old_sub, t514, v35, out = sys.argv[1:8]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads(Path(f"preds/manifest_{spec}.json").read_text(encoding="utf-8"))
    T = pd.read_parquet(m["test"]["path"])
    assert sha16(m["test"]["path"]) == m["test"]["sha"], "test file differs from the manifest"
    S = pd.read_parquet(old_sub)
    idc = [c for c in S.columns if c.startswith("MVT_ID")][0]
    ids = S[idc].to_numpy()
    D = T.set_index("MVT_ID_mvt").reindex(ids)
    chk = {"rows": int(len(D)), "template_rows": int(len(ids)), "ids_unique": bool(pd.Index(ids).is_unique),
           "missing": int(D["pred"].isna().sum()), "negative": int((D["pred"] < 0).sum()), "test_rows_in_final_file": int(len(T))}
    assert chk["missing"] == 0 and chk["negative"] == 0 and chk["ids_unique"] and chk["rows"] == 344841, chk
    q = pd.DataFrame({"MVT_ID_mvt": ids.astype("float64"), "pred": D["pred"].to_numpy(dtype="float64")})
    pq_path = out / f"{name}.parquet"
    q.to_parquet(pq_path, index=False)
    # move against 514 (float) and v35 (int) on the same ids
    a = pd.read_parquet(t514).set_index("MVT_ID_mvt").reindex(ids)["pred"].to_numpy(float)
    V = pd.read_parquet(v35)
    vid = [c for c in V.columns if c.startswith("MVT_ID")][0]
    vcol = [c for c in V.columns if c != vid][0]
    v = V.set_index(vid).reindex(ids)[vcol].to_numpy(float)
    ap = pd.read_parquet(old_rk, columns=["MVT_ID_mvt", "ADEP_mvt"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")["ADEP_mvt"].astype(str)
    apv = ap.reindex(ids).to_numpy()
    p = q["pred"].to_numpy()
    mv = {}
    t = TEXTS[name.split("_", 1)[0]]
    for ref, r in zip(t["refs"], (a, v)):
        d = p - r
        mv[ref] = {"mse": round(float(np.mean(d ** 2)), 3), "rms": round(float(np.sqrt(np.mean(d ** 2))), 3), "mean": round(float(np.mean(d)), 3),
                   "max_abs": round(float(np.max(np.abs(d))), 1),
                   "minute_airports_rms": round(float(np.sqrt(np.mean(d[np.isin(apv, MINRES)] ** 2))), 3),
                   "second_airports_rms": round(float(np.sqrt(np.mean(d[~np.isin(apv, MINRES)] ** 2))), 3)}
    chk["move"] = mv
    # composition in a sandbox root with the public-phase ranking and submitting
    sb = Path("research/f3/out/qr_root")
    if sb.exists():
        shutil.rmtree(sb)
    for d in ("data", "cache", "preds", "notes", "qr"):
        (sb / d).mkdir(parents=True)
    shutil.copytree("src", sb / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for f in Path("data").glob("training_*.parquet"):
        os.symlink(f.resolve(), sb / "data" / f.name)
    os.symlink(Path(old_rk).resolve(), sb / "data" / "ranking.parquet")
    os.symlink(Path(old_sub).resolve(), sb / "data" / "submitting.parquet")
    os.symlink(Path(f"cache/{m['feats']}.parquet").resolve(), sb / "cache" / f"{m['feats']}.parquet")
    os.symlink(Path(m["holdout"]["path"]).resolve(), sb / m["holdout"]["path"])
    shutil.copy2(f"preds/manifest_{spec}.json", sb / "preds" / f"manifest_{spec}.json")
    shutil.copy2(pq_path, sb / "qr" / f"{name}.parquet")
    entry = {"kind": "CANDIDATE", "manifest": f"preds/manifest_{spec}.json"}
    (sb / "qr" / f"{name}.json").write_text(json.dumps(entry), encoding="utf-8")
    r = subprocess.run([PY, "-X", "utf8", "-m", "src.composition", "--queue-dir", "qr", "--out", f"notes/composition_{name}.json", "--only", name],
                       cwd=sb, capture_output=True, text=True)
    comp_p = sb / "notes" / f"composition_{name}.json"
    comp = json.loads(comp_p.read_text(encoding="utf-8")) if comp_p.exists() else {"error": r.stdout[-2000:] + r.stderr[-2000:]}
    shutil.copy2(comp_p, out / f"composition_{name}.json") if comp_p.exists() else None
    failed = [{"check": c["check"], "value": c["value"], "band": c["band"]} for c in comp.get("checks", []) if not c["pass"]]
    meta = {"kind": "CANDIDATE",
            "method": t["method"],
            "local_rmse": m["holdout"].get("rmse"), "manifest_final_root": f"<final root>/preds/manifest_{spec}.json",
            "manifest_sha16": sha16(f"preds/manifest_{spec}.json"), "test_sha16_final_root": m["test"]["sha"], "spec": spec,
            "note": f"spec {spec}, feats {m['feats']}, holdout {m['holdout']['path']} ({t['same']} 의 val 캐시 그대로라 로컬 값이 {t['same']} 와 같다)",
            "readiness": f"composition_{name}.json", "readiness_all_pass": comp.get("all_pass"), "readiness_failed": failed,
            "checks": chk, "parquet_sha16": sha16(pq_path), "fired": False}
    (out / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
