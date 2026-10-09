# SPDX-License-Identifier: GPL-3.0-only
"""반대 창 두 규칙(R1, R2)을 큐 후보 하나 위에 마지막으로 얹어 새 후보를 만든다.
근거: vault `01 Projects/공모전 파이프라인/착수/PRC Data Challenge 2026/research/2026-10-05-r3-structure.md` 2.4(연구 스크립트
research/r3_structure/s5_link.py, e11_antiwin_rules.py 와 같은 정의).

이음: 출발(목적지가 10곳 안)을 목적 공항의 도착 기록과 (출발 공항, 도착 공항, 공항 편명)으로 묶고, 이륙 300초 뒤부터 8시간 안의 첫 착륙과 잇는다.
  NM 미매칭 출발(FLIGHT_ID 없음)인데 이어진 도착 기록이 NM 매칭이면 그 도착 기록의 LOBT, AOBT_3(그 비행의 출발 쪽 시각)를 쓴다.
  d_lobt = MVT − LOBT_arr, d_aobt = MVT − AOBT3_arr, d_sch = MVT − SCHED, sch_m_lobt = SCHED − LOBT_arr. 입력 칸만 쓴다.
  2025 에서 이 행의 97% 가 |BLOCK − LOBT_arr| > 3,606 이다: 매칭기가 LOBT 창으로 버린 출발이라 정답은 그 창 밖에 있다.
R1(G 꼴: LIRF, d_sch ≥ 3,600): |sch_m_lobt| ≤ 3,606 이고 d_lobt > 3,606 이면 예측 = d_aobt + c(SCHED 복사가 불가능한 진짜 지연).
R2(LIRF, LTFM 만): 0 < d_lobt ≤ 3,606 이면 예측 = max(그 후보의 예측, d_lobt + 3,606 + m)(정답이 창 위쪽 가지에 있어야 한다).
c, m: 평가 달 밖 2025 행의 중앙값(c: R1 행의 y − d_aobt, m: R2 행의 y − (d_lobt + 3,606)). val(자 1, 7월)은 1, 7월 밖, fwd(2025-07)는
  7월 밖, nd(11, 12월)는 11, 12월 밖, full(시험)은 열두 달 전부. 새로 고르는 값은 없다(연구에서 정한 식 그대로).
규칙이 바꾸는 행은 출발 쪽 LOBT 가 없는 행이라 LOBT 창과 겹치지 않는다. 그래서 순서상 맨 마지막에 둔다.

쓰는 법(저장소에서, memcap 안):
    .venv/Scripts/python.exe -X utf8 tools/antiwin.py build <원 큐 이름> <새 큐 번호> [--no-enqueue]
        원 후보의 manifest 에서 자, 7월 전방, 시험 예측을 읽어 규칙을 얹고, preds/ 에 새 파일 셋과 manifest 를 쓰고, 자, 전방, 둘째 자(원 후보의
        cache/nd_<run_tag>_s0.parquet 가 있을 때)의 ΔMSE 와 바뀐 행(좋아진 / 전체)을 찍고, 시험에서 바뀐 행이 규칙 행 안에만 있는지 확인한 뒤
        tools/enqueue.py --readiness-warn 으로 큐에 넣는다. 새 spec 은 원 spec 뒤에 "aw" 를 붙인다.
    .venv/Scripts/python.exe -X utf8 tools/antiwin.py --setting {val,fwd,full} --src <원 manifest> --src_sha <16> --link antiwin_link_v1 --link_sha <16> --tag <새 spec>
        manifest 의 commands 가 적는 재현 명령. 그 설정의 파일 하나만 다시 쓴다.
    .venv/Scripts/python.exe -X utf8 tools/antiwin.py link
        이음 표 cache/antiwin_link_v1.parquet(+json)만 만든다(이미 있으면 그대로).
    .venv/Scripts/python.exe -X utf8 tools/antiwin.py enqueue <새 큐 이름>
        build 의 큐 넣기 단계만 다시 한다(notes/antiwin_<새 큐 이름>.json 의 manifest 와 설명). build 는 데이터 단계(_files)와 이 단계를 따로 띄운다:
        준비 점검(src.composition)이 3 GB 상한을 거의 다 쓰므로 데이터를 든 프로세스와 겹치지 않게 한다.
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

APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
LINK = "antiwin_link_v1"
WIN = 3606.0
R2_APTS = ("LIRF", "LTFM")
EXCL = {"val": (1, 7), "fwd": (7,), "nd": (11, 12), "full": ()}
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


def build_link(force: bool = False) -> pd.DataFrame:
    """이어진 NM 미매칭 출발(도착 쪽 NM 있음) 표. 학습 행은 정답 y 를 함께 둔다(c, m 을 학습 달에서만 정할 때 쓴다)."""
    out = ROOT / "cache" / f"{LINK}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    sc = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt"]
    cols = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "src", "y", "t_mvt", "t_sch", "t_lobt", "t_aobt3"] + sc
    X = pq.read_table(ROOT / "cache" / "all.parquet", columns=cols, read_dictionary=sc).to_pandas()
    okfl = X["FLIGHT_mvt"].notna().to_numpy()
    isdep = X["PHASE_mvt"].eq("DEP").to_numpy()
    isarr = X["PHASE_mvt"].eq("ARR").to_numpy()
    dep = X[isdep & X["ADES_mvt"].isin(APTS).to_numpy() & okfl].copy()
    arr = X[isarr & X["ADEP_mvt"].isin(APTS).to_numpy() & okfl].copy()
    del X
    for d in (dep, arr):
        d["k"] = d["ADEP_mvt"].astype(str) + "|" + d["ADES_mvt"].astype(str) + "|" + d["FLIGHT_mvt"].astype(str)
    arr = arr.sort_values(["k", "t_mvt"]).reset_index(drop=True)
    a_k, a_t = arr["k"].to_numpy(), arr["t_mvt"].to_numpy()
    uk, st = np.unique(a_k, return_index=True)
    en = np.append(st[1:], len(a_k))
    amap = {k: (s, e) for k, s, e in zip(uk, st, en)}
    dep = dep[dep["FLIGHT_ID_mvt"].isna()].reset_index(drop=True)  # NM 미매칭 출발만 쓴다
    link = np.full(len(dep), -1, np.int64)
    d_k, d_t = dep["k"].to_numpy(), dep["t_mvt"].to_numpy()
    for i in range(len(dep)):
        r = amap.get(d_k[i])
        if r is None:
            continue
        s, e = r
        j = s + np.searchsorted(a_t[s:e], d_t[i] + 300, side="left")
        if j < e and a_t[j] - d_t[i] <= 8 * 3600:
            link[i] = j
    ok = link >= 0
    A = arr.iloc[link[ok]].reset_index(drop=True)
    L = dep.loc[ok, ["MVT_ID_mvt", "src", "y", "ADEP_mvt", "t_mvt", "t_sch"]].reset_index(drop=True)
    L["ADEP_mvt"] = L["ADEP_mvt"].astype(str)
    L["a_nm"] = A["FLIGHT_ID_mvt"].notna().to_numpy()
    L["a_t_lobt"] = A["t_lobt"].to_numpy()
    L["a_t_aobt3"] = A["t_aobt3"].to_numpy()
    L = L[L["a_nm"]].drop(columns="a_nm").reset_index(drop=True)
    L["d_lobt"] = L["t_mvt"] - L["a_t_lobt"]
    L["d_aobt"] = L["t_mvt"] - L["a_t_aobt3"]
    L["d_sch"] = L["t_mvt"] - L["t_sch"]
    L["sch_m_lobt"] = L["t_sch"] - L["a_t_lobt"]
    L["r1"] = ((L["ADEP_mvt"] == "LIRF") & (L["d_sch"] >= 3600) & (np.abs(L["sch_m_lobt"]) <= WIN) & (L["d_lobt"] > WIN)
               & np.isfinite(L["d_aobt"]))
    L["r2"] = L["ADEP_mvt"].isin(R2_APTS) & (L["d_lobt"] > 0) & (L["d_lobt"] <= WIN)
    L.to_parquet(out, index=False)
    info = {"rows": int(len(L)), "train_rows": int((L["src"] > 0).sum()), "test_rows": int((L["src"] == 0).sum()),
            "r1_train": int((L["r1"] & (L["src"] > 0)).sum()), "r2_train": int((L["r2"] & (L["src"] > 0)).sum()),
            "r1_test": int((L["r1"] & (L["src"] == 0)).sum()), "r2_test": int((L["r2"] & (L["src"] == 0)).sum()),
            "sha16": sha16(out.relative_to(ROOT)), "built_kst": now_kst(), "code": "tools/antiwin.py build_link"}
    out.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print("이음 표", out.name, info, flush=True)
    return L


def now_kst() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")


def constants(L: pd.DataFrame, setting: str) -> tuple[float, float]:
    tr = L[(L["src"] > 0) & ~L["src"].isin(EXCL[setting])]
    c = float(np.median((tr["y"] - tr["d_aobt"])[tr["r1"]]))
    m = float(np.median((tr["y"] - (tr["d_lobt"] + WIN))[tr["r2"]]))
    return c, m


def apply_rules(pred: pd.Series, L: pd.DataFrame, setting: str) -> tuple[pd.Series, dict]:
    """pred: MVT_ID 색인의 예측. 규칙 행만 바꾼 새 Series 와 규칙별 바뀐 행 수."""
    c, m = constants(L, setting)
    Lx = L.set_index("MVT_ID_mvt")
    Lx = Lx[Lx.index.isin(pred.index)]
    new = pred.copy()
    r1 = Lx.index[Lx["r1"].to_numpy()]
    r2 = Lx.index[Lx["r2"].to_numpy()]
    new.loc[r1] = (Lx.loc[r1, "d_aobt"] + c).to_numpy()
    lb = (Lx.loc[r2, "d_lobt"] + WIN + m).to_numpy()
    new.loc[r2] = np.maximum(new.loc[r2].to_numpy(), lb)
    ch = new.to_numpy() != pred.to_numpy()
    info = {"c": round(c, 2), "m": round(m, 2), "r1_rows": int(len(r1)), "r2_rows": int(len(r2)), "changed": int(ch.sum()),
            "changed_r1": int(np.isin(pred.index[ch], r1).sum()), "changed_r2": int(np.isin(pred.index[ch], r2).sum())}
    return new, info


def truth() -> pd.Series:
    T = pq.read_table(ROOT / "cache" / "all.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "src", "y"]).to_pandas()
    T = T[(T["PHASE_mvt"].astype(str) == "DEP") & (T["src"] > 0)]
    return T.set_index("MVT_ID_mvt")["y"].astype(float)


def delta(old: pd.Series, new: pd.Series, y: pd.Series) -> dict:
    yy = y.reindex(old.index).to_numpy()
    ok = np.isfinite(yy)
    o, n, yy = old.to_numpy()[ok], new.to_numpy()[ok], yy[ok]
    se0, se1 = (o - yy) ** 2, (n - yy) ** 2
    ch = n != o
    N = len(yy)
    return {"rows": int(N), "rmse_old": round(float(np.sqrt(se0.mean())), 3), "rmse_new": round(float(np.sqrt(se1.mean())), 3),
            "dmse": round(float((se1.sum() - se0.sum()) / N), 1), "changed": int(ch.sum()), "improved": int((se1 < se0)[ch].sum())}


def read_pred(rel: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / rel)


def write_setting(setting: str, src_manifest: dict, tag: str, L: pd.DataFrame) -> tuple[str, pd.DataFrame, pd.DataFrame, dict]:
    """한 설정의 원 예측에 규칙을 얹어 새 파일로 쓴다. 돌려주는 값: 새 파일 경로, 원 표, 새 표, 규칙 정보."""
    key = {"val": "holdout", "fwd": "forward", "full": "test"}[setting]
    old = read_pred(src_manifest[key]["path"])
    p = old.set_index("MVT_ID_mvt")["pred"].astype(float)
    newp, info = apply_rules(p, L, setting)
    new = old.copy()
    new["pred"] = newp.to_numpy()
    if setting == "val":
        y = truth()
        rm = float(np.sqrt(((newp - y.reindex(newp.index)) ** 2).mean()))
        rel = f"preds/holdout_{tag}_rmse{rm:.2f}.parquet"
    elif setting == "fwd":
        rel = f"preds/fwd_{tag}.parquet"
    else:
        rel = f"preds/test_{tag}.parquet"
    new.to_parquet(ROOT / rel, index=False)
    return rel, old, new, info


def build(src_stem: str, num: str) -> int:
    from src.batches import manifest_of
    from src.ruler import ruler_hash
    meta = json.loads((ROOT / "queue" / f"{src_stem}.json").read_text(encoding="utf-8"))
    sm, why = manifest_of(meta)
    if why:
        print("거부: 원 후보 manifest 검사", why)
        return 1
    src_mp = meta["manifest"]
    tag = sm["spec"] + "aw"
    name = f"{num}_{tag}"
    if (ROOT / "queue" / f"{name}.json").exists() or (ROOT / f"preds/manifest_{tag}.json").exists():
        print("거부: 같은 이름이 이미 있다", name)
        return 1
    L = build_link()
    link_rel = f"cache/{LINK}.parquet"
    link_sha = sha16(link_rel)
    y = truth()
    rep = {}
    # 시험부터: 바뀐 행이 규칙 행 안에만 있는지, 수가 예상(30 안팎) 안인지 확인하고 나서야 파일을 쓴다
    old_t = read_pred(sm["test"]["path"]).set_index("MVT_ID_mvt")["pred"].astype(float)
    new_t, info_t = apply_rules(old_t, L, "full")
    ch = old_t.index[new_t.to_numpy() != old_t.to_numpy()]
    rule_ids = set(L.loc[(L["src"] == 0) & (L["r1"] | L["r2"]), "MVT_ID_mvt"])
    outside = [i for i in ch if i not in rule_ids]
    print(f"[{name}] 시험: 규칙 행 R1 {info_t['r1_rows']}, R2 {info_t['r2_rows']}, 바뀐 행 {len(ch)}(R1 {info_t['changed_r1']}, R2 {info_t['changed_r2']}), "
          f"규칙 밖에서 바뀐 행 {len(outside)}, 이동 MSE(시험 N) {((new_t - old_t) ** 2).sum() / N_TEST:,.1f}, c {info_t['c']}, m {info_t['m']}", flush=True)
    if outside or not (10 <= len(ch) <= 60):
        print("거부: 시험에서 바뀐 행이 예상과 다르다(규칙 밖이거나 10~60 행 밖)")
        return 1
    files = {}
    for setting in ("val", "fwd", "full"):
        rel, old, new, info = write_setting(setting, sm, tag, L)
        files[setting] = rel
        o = old.set_index("MVT_ID_mvt")["pred"].astype(float)
        n = new.set_index("MVT_ID_mvt")["pred"].astype(float)
        if setting == "full":
            assert np.array_equal(n.to_numpy(), new_t.reindex(n.index).to_numpy())
            rep[setting] = {**info, "move_mse_test_n": round(float(((n - o) ** 2).sum() / N_TEST), 1)}
        else:
            rep[setting] = {**info, **delta(o, n, y)}
        print(f"[{name}] {setting}: {rep[setting]}", flush=True)
    nd_path = ROOT / "cache" / f"nd_{sm.get('run_tag', sm['spec'])}_s0.parquet"
    if nd_path.exists():
        nd = pd.read_parquet(nd_path).set_index("MVT_ID_mvt")["pred"].astype(float)
        ndn, info = apply_rules(nd, L, "nd")
        rep["nd"] = {**info, **delta(nd, ndn, y)}
    else:
        rep["nd"] = f"원 후보의 둘째 자 예측이 없다({nd_path.relative_to(ROOT).as_posix()})"
    print(f"[{name}] nd: {rep['nd']}", flush=True)
    # es_valid_ids 는 원 후보의 것을 그대로(규칙은 학습을 하지 않는다)
    es_rel = f"preds/es_valid_ids_{tag}.parquet"
    shutil.copy2(ROOT / sm["es_valid_ids"]["path"], ROOT / es_rel)
    # 코드 사본
    me = "tools/antiwin.py"
    h = sha16(me)
    snap = f"preds/code/{h}_antiwin.py"
    if not (ROOT / snap).exists():
        shutil.copy2(ROOT / me, ROOT / snap)
    cfg = {"--src": src_mp, "--src_sha": sha16(src_mp), "--link": LINK, "--link_sha": link_sha, "--tag": tag}
    flags = " ".join(f"{k} {v}" for k, v in cfg.items())
    cmd = lambda s: f".venv/Scripts/python.exe -X utf8 tools/antiwin.py --setting {s} {flags}"  # noqa: E731
    hold_ids = pd.read_parquet(ROOT / files["val"])["MVT_ID_mvt"].to_numpy()
    man = {
        "spec": tag, "run_tag": tag, "column": "pred", "seeds": sm["seeds"], "feats": sm["feats"], "feats_sha": sm["feats_sha"],
        "config": cfg,
        "commands": {"holdout": [cmd("val")], "test": [cmd("full")], "forward": [cmd("fwd")]},
        "command": cmd("val"),
        "code_sha": {**sm["code_sha"], me: h},
        "code_snapshots": {**sm.get("code_snapshots", {}), me: snap},
        "holdout": {"path": files["val"], "sha": sha16(files["val"]), "rmse": round(rep["val"]["rmse_new"], 2), "ruler_hash": ruler_hash(hold_ids)},
        "test": {"path": files["full"], "sha": sha16(files["full"])},
        "forward": {"path": files["fwd"], "sha": sha16(files["fwd"]), "rmse_2025_07": round(rep["fwd"]["rmse_new"], 2)},
        "es_valid_ids": {**sm["es_valid_ids"], "path": es_rel, "sha": sha16(es_rel)},
        "iteration_rule": sm.get("iteration_rule", "") + " | antiwin: no fitting; c and m are medians over 2025 months outside the evaluated months",
        "source": {"queue": src_stem, "manifest": src_mp, "manifest_sha": sha16(src_mp), "spec": sm["spec"],
                   "holdout_sha": sm["holdout"]["sha"], "test_sha": sm["test"]["sha"], "forward_sha": sm["forward"]["sha"]},
        "antiwin": {"link": link_rel, "link_sha": link_sha, "rules": "R1 G-type LIRF d_sch>=3600, |SCHED-LOBT_arr|<=3606, d_lobt>3606 -> d_aobt+c; "
                    "R2 LIRF/LTFM 0<d_lobt<=3606 -> max(pred, d_lobt+3606+m)", "report": rep},
        "written_kst": now_kst(),
    }
    mp = f"preds/manifest_{tag}.json"
    (ROOT / mp).write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{name}] manifest {mp}", flush=True)
    method = (f"{src_stem} 위에 반대 창 두 규칙을 맨 끝에 얹은 판(tools/antiwin.py, 이음 표 {link_rel} sha {link_sha}). "
              "NM 미매칭 출발을 목적 공항(10곳 안)의 NM 매칭 도착 기록과 편명과 시각으로 이어 그 비행의 LOBT, AOBT_3 를 얻고, "
              "매칭기가 LOBT 창으로 버린 행이라 정답이 창 밖에 있다는 것을 쓴다. R1(G 꼴): SCHED 가 LOBT 1시간 안이고 MVT − LOBT > 3,606 이면 "
              "MVT − AOBT_3 + c, R2(LIRF, LTFM): 0 < MVT − LOBT ≤ 3,606 이면 max(예측, MVT − LOBT + 3,606 + m). c, m 은 평가 달 밖 2025 중앙값, 새 적합 없음. "
              f"근거 vault research/2026-10-05-r3-structure.md 2.4. 자 ΔMSE {rep['val']['dmse']}({rep['val']['improved']}/{rep['val']['changed']}), "
              f"7월 전방 ΔMSE {rep['fwd']['dmse']}({rep['fwd']['improved']}/{rep['fwd']['changed']}), 시험 바뀐 행 {rep['full']['changed']}, "
              "10-05 단일 발 체제라 준비 점검은 경고(--readiness-warn)")
    (ROOT / "notes" / f"antiwin_{name}.json").write_text(json.dumps({"name": name, "manifest": mp, "source": src_stem, "method": method, "report": rep,
                                                                     "written_kst": now_kst()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def enqueue(name: str) -> int:
    """build 가 남긴 notes/antiwin_<name>.json 의 manifest 와 설명으로 tools/enqueue.py --readiness-warn 을 부른다(데이터를 읽지 않는 프로세스에서)."""
    info = json.loads((ROOT / "notes" / f"antiwin_{name}.json").read_text(encoding="utf-8"))
    r = subprocess.run([sys.executable, "-X", "utf8", "tools/enqueue.py", info["manifest"], name, info["method"], "--readiness-warn"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1500:], r.stderr[-800:], flush=True)
    return r.returncode


def main() -> int:
    below_normal()
    a = sys.argv[1:]
    if a and a[0] == "link":
        build_link(force="--force" in a)
        return 0
    if a and a[0] == "_files":
        return build(a[1], a[2])
    if a and a[0] == "enqueue":
        return enqueue(a[1])
    if a and a[0] == "build":
        if len(a) < 3:
            print("usage: antiwin.py build <원 큐 이름> <새 큐 번호> [--no-enqueue]")
            return 2
        me = [sys.executable, "-X", "utf8", str(Path(__file__).resolve())]
        rc = subprocess.run(me + ["_files", a[1], a[2]], cwd=ROOT).returncode  # 데이터 단계는 따로 끝내 메모리를 돌려준다
        if rc != 0 or "--no-enqueue" in a:
            return rc
        meta = json.loads((ROOT / "queue" / f"{a[1]}.json").read_text(encoding="utf-8"))
        spec = json.loads((ROOT / meta["manifest"]).read_text(encoding="utf-8"))["spec"]
        return subprocess.run(me + ["enqueue", f"{a[2]}_{spec}aw"], cwd=ROOT).returncode
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--seed", default="0")
    pa.add_argument("--src", required=True)
    pa.add_argument("--src_sha", required=True)
    pa.add_argument("--link", default=LINK)
    pa.add_argument("--link_sha", required=True)
    pa.add_argument("--tag", required=True)
    g = pa.parse_args(a)
    if sha16(g.src) != g.src_sha:
        print("거부: 원 manifest 해시가 다르다")
        return 1
    if sha16(f"cache/{g.link}.parquet") != g.link_sha:
        print("거부: 이음 표 해시가 다르다")
        return 1
    sm = json.loads((ROOT / g.src).read_text(encoding="utf-8"))
    rel, _, _, info = write_setting(g.setting, sm, g.tag, pd.read_parquet(ROOT / "cache" / f"{g.link}.parquet"))
    print(rel, info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
