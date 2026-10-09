# SPDX-License-Identifier: GPL-3.0-only
"""두 큐 후보(같은 뒤 단계를 거친 사슬 둘)의 최종 예측을 고정 가중으로 섞어 새 후보를 만든다: f = w x A + (1 - w) x B.
근거: private notes `research/2026-10-07-r8-anatomy.md` 6절(사전 등록 R8-P3, 측정
research/r8_anatomy/m4_blend.py). 468(창 기준 다리 넷의 사슬)과 486(기준 없는 다리의 같은 사슬)을 반씩: val 287.141 -> 286.641
(ΔMSE -286.6, 날 다시 뽑기 Δ<0 1.000), fwd 275.902 -> 275.303(-330.4, 1.000). R5-paradigms 5.3 「기준이 다른 모델과 섞기」.
가중 w 는 측정 전에 0.5 로 정했다(고르지 않음). 학습하지 않는다. 두 후보의 행 집합, 피처 판, 자 정의가 같아야 한다.

쓰는 법(저장소에서, memcap 안):
    .venv/Scripts/python.exe -X utf8 tools/chainavg.py build <A 큐 이름> <B 큐 이름> <새 큐 번호> [--w 0.5] [--suffix sk] [--no-enqueue]
        두 원 후보의 manifest 를 검사(src.batches.manifest_of)하고, 자, 7월 전방, 시험 예측을 섞어 preds/ 에 새 파일 셋과 manifest 를 쓰고,
        A 대비, B 대비 ΔMSE 를 찍은 뒤 tools/enqueue.py --readiness-warn 으로 큐에 넣는다. 새 spec 은 A spec 뒤에 "av" + suffix 를 붙인다.
    .venv/Scripts/python.exe -X utf8 tools/chainavg.py --setting {val,fwd,full} --src_a <A manifest> --src_a_sha <16> --src_b <B manifest> --src_b_sha <16> --w 0.5 --tag <새 spec>
        manifest 의 commands 가 적는 재현 명령. 그 설정의 파일 하나만 다시 쓴다.
    .venv/Scripts/python.exe -X utf8 tools/chainavg.py enqueue <새 큐 이름>
        build 의 큐 넣기 단계만 다시 한다(notes/chainavg_<새 큐 이름>.json). 데이터 단계와 따로 띄운다(준비 점검이 메모리를 많이 쓴다).
"""
from __future__ import annotations

import ctypes
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
KEY = {"val": "holdout", "fwd": "forward", "full": "test"}
N_TEST = 344841


def below_normal() -> None:
    if sys.platform == "win32":
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)


def sha16(rel: str | Path) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:16]


def now_kst() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")


def truth() -> pd.Series:
    T = pq.read_table(ROOT / "cache" / "all.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "src", "y"]).to_pandas()
    T = T[(T["PHASE_mvt"].astype(str) == "DEP") & (T["src"] > 0)]
    return T.set_index("MVT_ID_mvt")["y"].astype(float)


def mix(setting: str, ma: dict, mb: dict, w: float) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """A 파일의 행 순서 그대로 w A + (1 - w) B. 두 파일의 행 집합이 같아야 한다."""
    A = pd.read_parquet(ROOT / ma[KEY[setting]]["path"])
    B = pd.read_parquet(ROOT / mb[KEY[setting]]["path"])
    a = A.set_index("MVT_ID_mvt")["pred"].astype(float)
    b = B.set_index("MVT_ID_mvt")["pred"].astype(float)
    if len(a) != len(b) or a.index.has_duplicates or b.index.has_duplicates or not a.index.isin(b.index).all():
        raise SystemExit(f"{setting}: 두 후보의 행 집합이 다르다({len(a)} 대 {len(b)})")
    bb = b.reindex(a.index)
    f = w * a + (1.0 - w) * bb
    if not np.isfinite(f.to_numpy()).all():
        raise SystemExit(f"{setting}: 섞은 예측에 유한하지 않은 값이 있다")
    new = A.copy()
    new["pred"] = f.to_numpy()
    return new, a, bb, f


def write_setting(setting: str, ma: dict, mb: dict, w: float, tag: str) -> tuple[str, pd.Series, pd.Series, pd.Series]:
    new, a, b, f = mix(setting, ma, mb, w)
    if setting == "val":
        y = truth()
        rm = float(np.sqrt(((f - y.reindex(f.index)) ** 2).mean()))
        rel = f"preds/holdout_{tag}_rmse{rm:.2f}.parquet"
    elif setting == "fwd":
        rel = f"preds/fwd_{tag}.parquet"
    else:
        rel = f"preds/test_{tag}.parquet"
    new.to_parquet(ROOT / rel, index=False)
    return rel, a, b, f


def delta(old: pd.Series, new: pd.Series, y: pd.Series) -> dict:
    yy = y.reindex(old.index).to_numpy()
    ok = np.isfinite(yy)
    o, n, yy = old.to_numpy()[ok], new.reindex(old.index).to_numpy()[ok], yy[ok]
    se0, se1 = (o - yy) ** 2, (n - yy) ** 2
    N = len(yy)
    return {"rows": int(N), "rmse_old": round(float(np.sqrt(se0.mean())), 3), "rmse_new": round(float(np.sqrt(se1.mean())), 3),
            "dmse": round(float((se1.sum() - se0.sum()) / N), 1)}


def load_meta(stem: str) -> dict:
    for q in (ROOT / "queue", ROOT / "queue" / "done"):
        p = q / f"{stem}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise SystemExit(f"큐에 {stem} 이 없다(queue/, queue/done/)")


def build(stem_a: str, stem_b: str, num: str, w: float, suffix: str) -> int:
    from src.batches import manifest_of
    from src.ruler import ruler_hash
    meta_a, meta_b = load_meta(stem_a), load_meta(stem_b)
    ma, why = manifest_of(meta_a)
    if why:
        print("거부: A manifest 검사", why)
        return 1
    mb, why = manifest_of(meta_b)
    if why:
        print("거부: B manifest 검사", why)
        return 1
    if (ma["feats"], ma["feats_sha"]) != (mb["feats"], mb["feats_sha"]):
        print("거부: 두 후보의 피처 판이 다르다", ma["feats"], mb["feats"])
        return 1
    if ma["holdout"].get("ruler_hash") != mb["holdout"].get("ruler_hash"):
        print("거부: 두 후보의 자 정의(ruler_hash)가 다르다")
        return 1
    clash = {f: (h, mb["code_sha"][f]) for f, h in ma["code_sha"].items() if f in mb["code_sha"] and mb["code_sha"][f] != h}
    tag = ma["spec"] + "av" + suffix
    name = f"{num}_{tag}"
    if (ROOT / "queue" / f"{name}.json").exists() or (ROOT / f"preds/manifest_{tag}.json").exists():
        print("거부: 같은 이름이 이미 있다", name)
        return 1
    y = truth()
    files, rep = {}, {}
    for setting in ("val", "fwd", "full"):
        rel, a, b, f = write_setting(setting, ma, mb, w, tag)
        files[setting] = rel
        if setting == "full":
            rep[setting] = {"rows": int(len(f)), "move_mse_vs_a": round(float(((f - a) ** 2).sum() / N_TEST), 1),
                            "rows_moved_vs_a": int((np.abs(f - a) > 0.5).sum()), "move_mse_vs_b": round(float(((f - b) ** 2).sum() / N_TEST), 1)}
        else:
            rep[setting] = {"vs_a": delta(a, f, y), "vs_b": delta(b, f, y)}
        print(f"[{name}] {setting}: {rep[setting]}", flush=True)
    # es_valid_ids: 두 원 후보의 조기 종료 행의 합집합(섞기는 학습하지 않는다, 둘 다 자와 겹치지 않는다)
    es = pd.concat([pd.read_parquet(ROOT / ma["es_valid_ids"]["path"]), pd.read_parquet(ROOT / mb["es_valid_ids"]["path"])])
    es = es.drop_duplicates("MVT_ID_mvt").reset_index(drop=True)
    ov = int(es["MVT_ID_mvt"].isin(pd.read_parquet(ROOT / files["val"])["MVT_ID_mvt"]).sum())
    if ov:
        print(f"거부: 조기 종료 행 {ov}개가 자의 행과 겹친다")
        return 1
    es_rel = f"preds/es_valid_ids_{tag}.parquet"
    es.to_parquet(ROOT / es_rel, index=False)
    me = "tools/chainavg.py"
    h = sha16(me)
    snap = f"preds/code/{h}_chainavg.py"
    if not (ROOT / snap).exists():
        shutil.copy2(ROOT / me, ROOT / snap)
    src_a, src_b = meta_a["manifest"], meta_b["manifest"]
    cfg = {"--src_a": src_a, "--src_a_sha": sha16(src_a), "--src_b": src_b, "--src_b_sha": sha16(src_b), "--w": str(w), "--tag": tag}
    flags = " ".join(f"{k} {v}" for k, v in cfg.items())
    cmd = lambda s: f".venv/Scripts/python.exe -X utf8 tools/chainavg.py --setting {s} {flags}"  # noqa: E731
    hold_ids = pd.read_parquet(ROOT / files["val"])["MVT_ID_mvt"].to_numpy()
    code_sha = {**{f: h_ for f, h_ in mb["code_sha"].items() if f not in ma["code_sha"]}, **ma["code_sha"], me: h}
    snaps = {**{f: s for f, s in mb.get("code_snapshots", {}).items() if f not in ma.get("code_snapshots", {})}, **ma.get("code_snapshots", {}), me: snap}
    man = {
        "spec": tag, "run_tag": tag, "column": "pred", "seeds": ma["seeds"], "feats": ma["feats"], "feats_sha": ma["feats_sha"],
        "config": cfg,
        "commands": {"holdout": [cmd("val")], "test": [cmd("full")], "forward": [cmd("fwd")]},
        "command": cmd("val"),
        "code_sha": code_sha,
        "code_snapshots": snaps,
        "holdout": {"path": files["val"], "sha": sha16(files["val"]), "rmse": round(rep["val"]["vs_a"]["rmse_new"], 2), "ruler_hash": ruler_hash(hold_ids)},
        "test": {"path": files["full"], "sha": sha16(files["full"])},
        "forward": {"path": files["fwd"], "sha": sha16(files["fwd"]), "rmse_2025_07": round(rep["fwd"]["vs_a"]["rmse_new"], 2)},
        "es_valid_ids": {"path": es_rel, "sha": sha16(es_rel), "rows": int(len(es)), "overlap_with_ruler": 0},
        "iteration_rule": "A: " + ma.get("iteration_rule", "") + " || B: " + mb.get("iteration_rule", "") +
                          f" || chainavg: w A + (1 - w) B with w = {w} fixed before measuring, no fitting",
        "source": {"queue": stem_a, "manifest": src_a, "manifest_sha": sha16(src_a), "spec": ma["spec"],
                   "holdout_sha": ma["holdout"]["sha"], "test_sha": ma["test"]["sha"], "forward_sha": ma["forward"]["sha"]},
        "source_b": {"queue": stem_b, "manifest": src_b, "manifest_sha": sha16(src_b), "spec": mb["spec"],
                     "holdout_sha": mb["holdout"]["sha"], "test_sha": mb["test"]["sha"], "forward_sha": mb["forward"]["sha"],
                     "code_sha_differs_from_a": {f: {"a": x, "b": z} for f, (x, z) in clash.items()}},
        "chainavg": {"w": w, "report": rep, "basis": "private notes research/2026-10-07-r8-anatomy.md 6 (R8-P3)"},
        "written_kst": now_kst(),
    }
    mp = f"preds/manifest_{tag}.json"
    (ROOT / mp).write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    from src.batches import manifest_of as mo
    _, why = mo({"manifest": mp})
    if why:
        print("거부: 새 manifest 검사", why)
        return 1
    print(f"[{name}] manifest {mp}(manifest_of 통과)", flush=True)
    method = (f"{stem_a} 와 {stem_b} 의 최종 예측을 {w}:{1 - w:g} 로 섞은 판(tools/chainavg.py, 학습 없음, 가중은 측정 전에 정함). "
              "두 사슬은 바탕 다리(LightGBM, CatBoost, RealMLP)의 회귀 기준만 다르고(창 기준 대 없음) 뒤 단계가 같다. "
              f"근거 private notes research/2026-10-07-r8-anatomy.md 6절(R8-P3). 자 ΔMSE(A 대비) {rep['val']['vs_a']['dmse']}, "
              f"7월 전방 ΔMSE(A 대비) {rep['fwd']['vs_a']['dmse']}, 시험 이동(A 대비) {rep['full']['move_mse_vs_a']} MSE, "
              "10-05 단일 발 체제라 준비 점검은 경고(--readiness-warn)")
    (ROOT / "notes" / f"chainavg_{name}.json").write_text(json.dumps({"name": name, "manifest": mp, "source_a": stem_a, "source_b": stem_b, "method": method,
                                                                      "report": rep, "written_kst": now_kst()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def enqueue(name: str) -> int:
    info = json.loads((ROOT / "notes" / f"chainavg_{name}.json").read_text(encoding="utf-8"))
    r = subprocess.run([sys.executable, "-X", "utf8", "tools/enqueue.py", info["manifest"], name, info["method"], "--readiness-warn"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1500:], r.stderr[-800:], flush=True)
    return r.returncode


def opt(a: list[str], k: str, default: str) -> str:
    return a[a.index(k) + 1] if k in a else default


def main() -> int:
    below_normal()
    a = sys.argv[1:]
    if a and a[0] == "_files":
        return build(a[1], a[2], a[3], float(opt(a, "--w", "0.5")), opt(a, "--suffix", "sk"))
    if a and a[0] == "enqueue":
        return enqueue(a[1])
    if a and a[0] == "build":
        if len(a) < 4:
            print("usage: chainavg.py build <A 큐 이름> <B 큐 이름> <새 큐 번호> [--w 0.5] [--suffix sk] [--no-enqueue]")
            return 2
        w, suffix = opt(a, "--w", "0.5"), opt(a, "--suffix", "sk")
        me = [sys.executable, "-X", "utf8", str(Path(__file__).resolve())]
        rc = subprocess.run(me + ["_files", a[1], a[2], a[3], "--w", w, "--suffix", suffix], cwd=ROOT).returncode
        if rc != 0 or "--no-enqueue" in a:
            return rc
        spec = json.loads((ROOT / load_meta(a[1])["manifest"]).read_text(encoding="utf-8"))["spec"]
        return subprocess.run(me + ["enqueue", f"{a[3]}_{spec}av{suffix}"], cwd=ROOT).returncode
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--seed", default="0")
    pa.add_argument("--src_a", required=True)
    pa.add_argument("--src_a_sha", required=True)
    pa.add_argument("--src_b", required=True)
    pa.add_argument("--src_b_sha", required=True)
    pa.add_argument("--w", required=True, type=float)
    pa.add_argument("--tag", required=True)
    g = pa.parse_args(a)
    if sha16(g.src_a) != g.src_a_sha or sha16(g.src_b) != g.src_b_sha:
        print("거부: 원 manifest 해시가 다르다")
        return 1
    ma = json.loads((ROOT / g.src_a).read_text(encoding="utf-8"))
    mb = json.loads((ROOT / g.src_b).read_text(encoding="utf-8"))
    rel, _, _, _ = write_setting(g.setting, ma, mb, g.w, g.tag)
    print(rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
