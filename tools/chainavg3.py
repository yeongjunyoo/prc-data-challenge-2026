# SPDX-License-Identifier: GPL-3.0-only
"""세 큐 후보(같은 뒤 단계를 거친 사슬 셋)의 최종 예측을 1/3 씩 섞어 새 후보를 만든다: f = (A + B + C) / 3.
tools/chainavg.py(두 사슬, sha16 565d51c881797751)를 셋으로 늘린 새 파일이다(공유 도구는 바꾸지 않음). 근거: vault
`01 Projects/공모전 파이프라인/착수/PRC Data Challenge 2026/research/2026-10-07-r13-f3.md` 5 ~ 6절(R13-F3, 셋째 가족 ap: 공항별 회귀
LightGBM 과 CatBoost 다리의 사슬). 가중 1/3 은 측정 전에 정했다(고르지 않음). 학습하지 않는다. 세 후보의 행 집합, 피처 판, 자 정의가 같아야 한다.

쓰는 법(저장소 또는 모래 상자에서, memcap 안):
    .venv/Scripts/python.exe -X utf8 tools/chainavg3.py build <A 큐 이름> <B 큐 이름> <C 큐 이름> <새 큐 번호> [--suffix skap] [--no-enqueue]
        세 원 후보의 manifest 를 검사(src.batches.manifest_of)하고, 자, 7월 전방, 시험 예측을 섞어 preds/ 에 새 파일 셋과 manifest 를 쓰고,
        A, B, C 대비 ΔMSE 를 찍는다. 새 spec 은 A spec 뒤에 "av" + suffix 를 붙인다. --no-enqueue 가 없으면 tools/enqueue.py --readiness-warn.
    .venv/Scripts/python.exe -X utf8 tools/chainavg3.py --setting {val,fwd,full} --src_a <A manifest> --src_a_sha <16> --src_b <B manifest>
        --src_b_sha <16> --src_c <C manifest> --src_c_sha <16> --tag <새 spec>
        manifest 의 commands 가 적는 재현 명령. 그 설정의 파일 하나만 다시 쓴다.
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
W = "1/3"


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


def mix(setting: str, ms: list[dict]) -> tuple[pd.DataFrame, list[pd.Series], pd.Series]:
    """A 파일의 행 순서 그대로 (A + B + C) / 3. 세 파일의 행 집합이 같아야 한다."""
    A = pd.read_parquet(ROOT / ms[0][KEY[setting]]["path"])
    s = [A.set_index("MVT_ID_mvt")["pred"].astype(float)]
    for m in ms[1:]:
        x = pd.read_parquet(ROOT / m[KEY[setting]]["path"]).set_index("MVT_ID_mvt")["pred"].astype(float)
        if len(x) != len(s[0]) or x.index.has_duplicates or s[0].index.has_duplicates or not s[0].index.isin(x.index).all():
            raise SystemExit(f"{setting}: 후보들의 행 집합이 다르다({len(s[0])} 대 {len(x)})")
        s.append(x.reindex(s[0].index))
    f = (s[0] + s[1] + s[2]) / 3.0
    if not np.isfinite(f.to_numpy()).all():
        raise SystemExit(f"{setting}: 섞은 예측에 유한하지 않은 값이 있다")
    new = A.copy()
    new["pred"] = f.to_numpy()
    return new, s, f


def write_setting(setting: str, ms: list[dict], tag: str) -> tuple[str, list[pd.Series], pd.Series]:
    new, s, f = mix(setting, ms)
    if setting == "val":
        y = truth()
        rm = float(np.sqrt(((f - y.reindex(f.index)) ** 2).mean()))
        rel = f"preds/holdout_{tag}_rmse{rm:.2f}.parquet"
    elif setting == "fwd":
        rel = f"preds/fwd_{tag}.parquet"
    else:
        rel = f"preds/test_{tag}.parquet"
    new.to_parquet(ROOT / rel, index=False)
    return rel, s, f


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


def build(stems: list[str], num: str, suffix: str) -> int:
    from src.batches import manifest_of
    from src.ruler import ruler_hash
    metas = [load_meta(x) for x in stems]
    ms = []
    for lab, meta in zip("ABC", metas):
        m, why = manifest_of(meta)
        if why:
            print(f"거부: {lab} manifest 검사", why)
            return 1
        ms.append(m)
    if len({(m["feats"], m["feats_sha"]) for m in ms}) != 1:
        print("거부: 후보들의 피처 판이 다르다", [m["feats"] for m in ms])
        return 1
    if len({m["holdout"].get("ruler_hash") for m in ms}) != 1:
        print("거부: 후보들의 자 정의(ruler_hash)가 다르다")
        return 1
    tag = ms[0]["spec"] + "av" + suffix
    name = f"{num}_{tag}"
    if (ROOT / "queue" / f"{name}.json").exists() or (ROOT / f"preds/manifest_{tag}.json").exists():
        print("거부: 같은 이름이 이미 있다", name)
        return 1
    y = truth()
    files, rep = {}, {}
    for setting in ("val", "fwd", "full"):
        rel, s, f = write_setting(setting, ms, tag)
        files[setting] = rel
        if setting == "full":
            rep[setting] = {"rows": int(len(f)), **{f"move_mse_vs_{lab}": round(float(((f - x) ** 2).sum() / N_TEST), 1) for lab, x in zip("abc", s)},
                            "rows_moved_vs_a": int((np.abs(f - s[0]) > 0.5).sum())}
        else:
            rep[setting] = {f"vs_{lab}": delta(x, f, y) for lab, x in zip("abc", s)}
        print(f"[{name}] {setting}: {rep[setting]}", flush=True)
    es = pd.concat([pd.read_parquet(ROOT / m["es_valid_ids"]["path"]) for m in ms]).drop_duplicates("MVT_ID_mvt").reset_index(drop=True)
    ov = int(es["MVT_ID_mvt"].isin(pd.read_parquet(ROOT / files["val"])["MVT_ID_mvt"]).sum())
    if ov:
        print(f"거부: 조기 종료 행 {ov}개가 자의 행과 겹친다")
        return 1
    es_rel = f"preds/es_valid_ids_{tag}.parquet"
    es.to_parquet(ROOT / es_rel, index=False)
    me = "tools/chainavg3.py"
    h = sha16(me)
    snap = f"preds/code/{h}_chainavg3.py"
    if not (ROOT / snap).exists():
        shutil.copy2(ROOT / me, ROOT / snap)
    srcs = [meta["manifest"] for meta in metas]
    cfg = {}
    for lab, sm in zip("abc", srcs):
        cfg[f"--src_{lab}"], cfg[f"--src_{lab}_sha"] = sm, sha16(sm)
    cfg["--tag"] = tag
    flags = " ".join(f"{k} {v}" for k, v in cfg.items())
    cmd = lambda st: f".venv/Scripts/python.exe -X utf8 tools/chainavg3.py --setting {st} {flags}"  # noqa: E731
    hold_ids = pd.read_parquet(ROOT / files["val"])["MVT_ID_mvt"].to_numpy()
    code_sha, snaps, clash = {}, {}, {}
    for m in reversed(ms):  # A wins on a clash, as in chainavg
        for f_, h_ in m["code_sha"].items():
            if f_ in code_sha and code_sha[f_] != h_:
                clash[f_] = clash.get(f_, []) + [h_]
            code_sha[f_] = h_
        snaps.update(m.get("code_snapshots", {}))
    code_sha[me], snaps[me] = h, snap
    man = {
        "spec": tag, "run_tag": tag, "column": "pred", "seeds": ms[0]["seeds"], "feats": ms[0]["feats"], "feats_sha": ms[0]["feats_sha"],
        "config": cfg,
        "commands": {"holdout": [cmd("val")], "test": [cmd("full")], "forward": [cmd("fwd")]},
        "command": cmd("val"),
        "code_sha": code_sha,
        "code_snapshots": snaps,
        "holdout": {"path": files["val"], "sha": sha16(files["val"]), "rmse": round(rep["val"]["vs_a"]["rmse_new"], 2), "ruler_hash": ruler_hash(hold_ids)},
        "test": {"path": files["full"], "sha": sha16(files["full"])},
        "forward": {"path": files["fwd"], "sha": sha16(files["fwd"]), "rmse_2025_07": round(rep["fwd"]["vs_a"]["rmse_new"], 2)},
        "es_valid_ids": {"path": es_rel, "sha": sha16(es_rel), "rows": int(len(es)), "overlap_with_ruler": 0},
        "iteration_rule": " || ".join(f"{lab}: " + m.get("iteration_rule", "") for lab, m in zip("ABC", ms)) +
                          " || chainavg3: (A + B + C) / 3 with weights 1/3 fixed before measuring, no fitting",
        "source": {"queue": stems[0], "manifest": srcs[0], "manifest_sha": sha16(srcs[0]), "spec": ms[0]["spec"],
                   "holdout_sha": ms[0]["holdout"]["sha"], "test_sha": ms[0]["test"]["sha"], "forward_sha": ms[0]["forward"]["sha"]},
        "source_b": {"queue": stems[1], "manifest": srcs[1], "manifest_sha": sha16(srcs[1]), "spec": ms[1]["spec"],
                     "holdout_sha": ms[1]["holdout"]["sha"], "test_sha": ms[1]["test"]["sha"], "forward_sha": ms[1]["forward"]["sha"]},
        "source_c": {"queue": stems[2], "manifest": srcs[2], "manifest_sha": sha16(srcs[2]), "spec": ms[2]["spec"],
                     "holdout_sha": ms[2]["holdout"]["sha"], "test_sha": ms[2]["test"]["sha"], "forward_sha": ms[2]["forward"]["sha"]},
        "code_sha_clash": clash,
        "chainavg3": {"w": W, "report": rep, "basis": "vault research/2026-10-07-r13-f3.md 5 ~ 6 (R13-F3)"},
        "written_kst": now_kst(),
    }
    mp = f"preds/manifest_{tag}.json"
    (ROOT / mp).write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    _, why = manifest_of({"manifest": mp})
    if why:
        print("거부: 새 manifest 검사", why)
        return 1
    print(f"[{name}] manifest {mp}(manifest_of 통과)", flush=True)
    (ROOT / "notes" / f"chainavg3_{name}.json").write_text(json.dumps({"name": name, "manifest": mp, "sources": stems, "report": rep,
                                                                       "written_kst": now_kst()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def opt(a: list[str], k: str, default: str) -> str:
    return a[a.index(k) + 1] if k in a else default


def main() -> int:
    below_normal()
    a = sys.argv[1:]
    if a and a[0] == "build":
        if len(a) < 5:
            print("usage: chainavg3.py build <A 큐 이름> <B 큐 이름> <C 큐 이름> <새 큐 번호> [--suffix skap] [--no-enqueue]")
            return 2
        suffix = opt(a, "--suffix", "skap")
        rc = build(a[1:4], a[4], suffix)
        if rc != 0 or "--no-enqueue" in a:
            return rc
        spec_a = json.loads((ROOT / load_meta(a[1])["manifest"]).read_text(encoding="utf-8"))["spec"]
        info = json.loads((ROOT / "notes" / f"chainavg3_{a[4]}_{spec_a}av{suffix}.json").read_text(encoding="utf-8"))
        r = subprocess.run([sys.executable, "-X", "utf8", "tools/enqueue.py", info["manifest"], info["name"], "chainavg3 (A + B + C) / 3", "--readiness-warn"],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(r.stdout[-1500:], r.stderr[-800:], flush=True)
        return r.returncode
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--seed", default="0")
    for lab in "abc":
        pa.add_argument(f"--src_{lab}", required=True)
        pa.add_argument(f"--src_{lab}_sha", required=True)
    pa.add_argument("--tag", required=True)
    g = pa.parse_args(a)
    ms = []
    for lab in "abc":
        p, h = getattr(g, f"src_{lab}"), getattr(g, f"src_{lab}_sha")
        if sha16(p) != h:
            print(f"거부: 원 manifest {lab} 해시가 다르다")
            return 1
        ms.append(json.loads((ROOT / p).read_text(encoding="utf-8")))
    rel, _, _ = write_setting(g.setting, ms, g.tag)
    print(rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
