# SPDX-License-Identifier: GPL-3.0-only
"""운항사-달 도착 복사 몫이 크게 움직인 공항-운항사-달 칸에서 주 분류기 p_art 의 칸 평균을 옮긴다(G 아닌 출발만). A3 보정기 출력 바로 뒤, gkey 앞에 둔다.
순서: 바탕 다리 → V5 기하 → A3 x0.5 → 이 단계 → gkey(avg) → gkeyx → gkeyxn → AIU 날 이동 → 반대 창(맨 끝).
근거: vault `01 Projects/공모전 파이프라인/착수/PRC Data Challenge 2026/research/2026-10-06-r6-g.md` 의 비G 확장(연구 스크립트 research/r6_g/p1_check.py,
p4_cell_lomo.py, p5_mm_thresh.py). 생각의 출처: 우리 기록 감식(같은 운항사의 그 달 도착 기록이 SCHED 복사이면 출발 기록도 복사다. 2025 LIRF 운항사-달 안
상관 0.77, 기울기 0.93). 남의 코드 없음. 입력만 쓴다(도착 BLOCK 은 시험에도 있는 입력 칸). 정답은 κ 를 재는 학습 달 출발 칸 평균에만 쓴다.

정의:
  운항사 키 = 편명(FLIGHT_mvt) 앞 세 글자가 영문이면 그것, 아니면 앞 두 글자(EDDF, EHAM, LSZH 는 IATA 판매 편명). 같은 공항의 출발과 도착이 같은 키를 쓴다.
  달 = 2025 은 src(1 ~ 12), 2026 은 100 + 이륙(착륙) UTC 달.
  ca(공항, 운항사, 달) = 그 공항 도착 가운데 |BLOCK − SCHED| ≤ 6.5초 몫. ca_ref = 그 설정의 학습 달(val 은 1, 7월 밖, fwd 는 1 ~ 6월, full 은 2025 전부) 도착 전체의 몫.
  Δa = ca − ca_ref, 칸 조건: 그 달 도착 100행 이상, 기준 도착 300행 이상, |Δa| ≥ 0.10.
  κ = 학습 달 칸(출발 100, 도착 100 이상, 공항-운항사 세 달 이상)에서 잰 「운항사 안」 기울기(출발 art 몫 대 도착 복사 몫, 고정 효과, 출발 행 가중).
  칸마다 로짓 이동 b 하나: 그 칸의 바꿀 행(G 아님, p_art > 0) 평균 σ(logit p + b) = 평균 p + κ Δa(0.0001 ~ 0.999 로 자름).
  예측: Δ = (p' − p)(d − mn), mn = (pred_보정기 − p d)/(1 − p)(p 는 0.0001 ~ 0.999). 새 예측 = 원 후보 예측 + Δ 를 LOBT 창 [d_lobt − 3,606, d_lobt + 3,606]
  (feats_v25 d_lobt, 창이 있는 행만) 안으로, 0 아래는 0. p_art = 0 인 행(LOBT 창 규칙 1로 art 가 불가능한 행), G 행, 칸 밖 행은 그대로(바이트까지 같은지 확인).
  문턱 0.10 은 연구에서 정했다(0.05 는 val, fwd 에서 나빴다. 그 비교를 본 뒤 고른 값이라 자 고르기 위험이 있다. 노트 참조).
AIU 날 이동이 쓰도록 보정기 캐시와 같은 칸의 cache/<setting>_<새 spec>_s0.parquet 를 쓴다(p_art 만 p' 로, pred 와 mu 는 보정기 그대로라 AIU 의 정상 평균
  mn = mu + (pred − base) 가 그대로다). manifest 의 --base 는 원 보정기의 것을 그대로 두어 gkey 의 G 다리 찾기와 AIU 의 바탕 읽기가 같게 간다.

쓰는 법(저장소에서, memcap 3 GB 안):
    .venv/Scripts/python.exe -X utf8 tools/artmm.py table [--force]      표 cache/artmm_v1.parquet(출발 키), cache/artmm_cells_v1.parquet(칸)과 json
    .venv/Scripts/python.exe -X utf8 tools/artmm.py check <A3 판 manifest 또는 큐 이름>   세 설정의 ΔMSE, 칸, 이동만 찍는다(표 말고는 쓰지 않는다)
    .venv/Scripts/python.exe -X utf8 tools/artmm.py build <A3 판 큐 이름 또는 manifest> <새 큐 번호> [--no-enqueue]
        preds/ 에 새 파일 셋, cache/ 에 val, fwd, full 캐시, manifest 를 쓰고(새 spec 은 원 spec + "am"), notes/artmm_<새 큐 이름>.json 을 남긴다.
    .venv/Scripts/python.exe -X utf8 tools/artmm.py enqueue <새 큐 이름>
    .venv/Scripts/python.exe -X utf8 tools/artmm.py --setting {val,fwd,full} --base <바탕> --base_seed <s> --src <원 manifest> --src_sha <16> --corr <보정기 태그>
        --table artmm_v1 --table_sha <16> --cells artmm_cells_v1 --cells_sha <16> --th 0.10 --tag <새 spec>      (manifest 의 재현 명령)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

TABLE, CELLS = "artmm_v1", "artmm_cells_v1"
TH = 0.10
MIN_ARR, MIN_REF, MIN_KD, MIN_KA, MIN_KM = 100, 300, 100, 100, 3
SUFFIX = "am"
HALF = 3606.0
N_TEST = 344841
KEYPART = {"val": "holdout", "fwd": "forward", "full": "test"}
FITM = {"val": [m for m in range(1, 13) if m not in (1, 7)], "fwd": [1, 2, 3, 4, 5, 6], "full": list(range(1, 13))}


def sha16(rel: str | Path) -> str:
    h = hashlib.sha256()
    with open(ROOT / rel, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def now_kst() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")


def below_normal() -> None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)


def carkey(cats) -> dict:
    out = {}
    for s in cats:
        if not isinstance(s, str) or len(s) < 2:
            out[s] = None
        elif len(s) >= 3 and s[:3].isalpha():
            out[s] = s[:3]
        else:
            out[s] = s[:2]
    return out


# ---- 표 ----

def build_table(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    import pyarrow.parquet as pq
    o1, o2 = ROOT / "cache" / f"{TABLE}.parquet", ROOT / "cache" / f"{CELLS}.parquet"
    if o1.exists() and o2.exists() and not force:
        return pd.read_parquet(o1), pd.read_parquet(o2)
    t = pq.read_table(ROOT / "cache" / "all.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "FLIGHT_ID_mvt",
                                                              "t_mvt", "t_sch", "t_blk", "y", "src"])
    D = t.to_pandas()
    del t
    dep = (D["PHASE_mvt"].astype(str) == "DEP").to_numpy()
    D["ap"] = np.where(dep, D["ADEP_mvt"].astype(str), D["ADES_mvt"].astype(str))
    fl = D["FLIGHT_mvt"]
    D["car"] = fl.astype(object).map(carkey(fl.cat.categories if hasattr(fl, "cat") else pd.unique(fl.dropna())))
    src = D["src"].to_numpy()
    mth = pd.to_datetime(D["t_mvt"], unit="s").dt.month.to_numpy()
    D["mon"] = np.where(src == 0, 100 + mth, src).astype(np.int16)
    K = D.loc[dep, ["MVT_ID_mvt", "ap", "car", "mon"]].reset_index(drop=True)
    A = D[~dep & D["car"].notna().to_numpy() & np.isfinite(D["t_blk"].to_numpy(dtype=float))]
    ca = A.assign(cp=(np.abs(A["t_blk"].to_numpy(dtype=float) - A["t_sch"].to_numpy(dtype=float)) <= 6.5).astype(float)) \
        .groupby(["ap", "car", "mon"])["cp"].agg(arr_n="size", arr_copy="sum").reset_index()
    g = dep & (D["ap"].to_numpy() == "LIRF") & D["FLIGHT_ID_mvt"].isna().to_numpy()
    T = D[dep & ~g & (src > 0) & D["car"].notna().to_numpy()]
    art = (np.abs(T["y"].to_numpy(dtype=float) - (T["t_mvt"].to_numpy(dtype=float) - T["t_sch"].to_numpy(dtype=float))) <= 60).astype(float)
    cd = T.assign(art=art).groupby(["ap", "car", "mon"])["art"].agg(dep_n="size", dep_art="sum").reset_index()
    C = ca.merge(cd, on=["ap", "car", "mon"], how="left")
    del D, A, T
    K.to_parquet(o1, index=False)
    C.to_parquet(o2, index=False)
    for o, extra in ((o1, {"rows": int(len(K)), "kind": "departure keys (all departures, 2025 and 2026)"}),
                     (o2, {"rows": int(len(C)), "kind": "airport x carrier x month: arrivals and arrival copies (inputs), 2025 non-G departures and art (training labels)"})):
        info = {**extra, "sha16": sha16(o.relative_to(ROOT)), "source": "cache/all.parquet", "source_sha16": sha16("cache/all.parquet"),
                "built_kst": now_kst(), "code": "tools/artmm.py table", "code_sha16": sha16("tools/artmm.py")}
        o.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        print("표", o.name, json.dumps(info, ensure_ascii=False), flush=True)
    return K, C


# ---- 칸 이동 ----

def kappa(C: pd.DataFrame, months: list[int]) -> float:
    z = C[C["mon"].isin(months) & (C["arr_n"] >= MIN_KA) & (C["dep_n"] >= MIN_KD)].copy()
    z = z[z.groupby(["ap", "car"])["mon"].transform("size") >= MIN_KM]
    z["ca"] = z["arr_copy"] / z["arr_n"]
    z["da"] = z["dep_art"] / z["dep_n"]
    for c in ("ca", "da"):
        z[c + "_d"] = z[c] - z.groupby(["ap", "car"])[c].transform("mean")
    w = z["dep_n"].to_numpy(float)
    return float((w * z["ca_d"] * z["da_d"]).sum() / (w * z["ca_d"] ** 2).sum())


def solve_b(lo: np.ndarray, target: float) -> float:
    a, b = -8.0, 8.0
    for _ in range(60):
        m = 0.5 * (a + b)
        if (1 / (1 + np.exp(-(lo + m)))).mean() < target:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def shift(setting: str, corr: pd.DataFrame, K: pd.DataFrame, C: pd.DataFrame, th: float = TH) -> tuple[np.ndarray, np.ndarray, list, float]:
    """corr: 보정기 캐시(MVT_ID_mvt, pred, mu, p_art, g, dsch). 돌려주는 값: 새 p_art(p'), Δ 예측, 칸 정보, κ."""
    kap = kappa(C, FITM[setting])
    ref = C[C["mon"].isin(FITM[setting])].groupby(["ap", "car"])[["arr_copy", "arr_n"]].sum()
    cm = C.set_index(["ap", "car", "mon"])
    k = K.set_index("MVT_ID_mvt").reindex(corr["MVT_ID_mvt"].to_numpy())
    p0 = corr["p_art"].to_numpy(dtype=float)
    elig = (~corr["g"].to_numpy().astype(bool)) & (p0 > 0) & k["car"].notna().to_numpy()
    p = np.clip(p0, 1e-4, 0.999)
    lo = np.log(p / (1 - p))
    d, f = corr["dsch"].to_numpy(dtype=float), corr["pred"].to_numpy(dtype=float)
    mn = (f - p * d) / (1 - p)
    p_new = p0.copy()
    rows = []
    E = pd.DataFrame({"ap": k["ap"].to_numpy(), "car": k["car"].to_numpy(), "mon": k["mon"].to_numpy()})[elig]
    for (ap, car, mon), ii in E.groupby(["ap", "car", "mon"]).indices.items():
        idx = np.flatnonzero(elig)[ii]
        if (ap, car, mon) not in cm.index or (ap, car) not in ref.index:
            continue
        n_a, n_r = float(cm.loc[(ap, car, mon), "arr_n"]), float(ref.loc[(ap, car), "arr_n"])
        if n_a < MIN_ARR or n_r < MIN_REF:
            continue
        da = float(cm.loc[(ap, car, mon), "arr_copy"] / n_a - ref.loc[(ap, car), "arr_copy"] / n_r)
        if abs(da) < th:
            continue
        pm = float(p[idx].mean())
        b = solve_b(lo[idx], float(np.clip(pm + kap * da, 1e-4, 0.999)))
        p_new[idx] = 1 / (1 + np.exp(-(lo[idx] + b)))
        rows.append({"ap": ap, "car": car, "mon": int(mon), "rows": int(len(idx)), "da": round(da, 4), "p_mean": round(pm, 4),
                     "p_mean_new": round(float(p_new[idx].mean()), 4), "b": round(b, 4)})
    delta = np.where(p_new != p0, (np.clip(p_new, 1e-4, 0.999) - p) * (d - mn), 0.0)
    return p_new, delta, rows, kap


def d_lobt_of(ids: np.ndarray) -> np.ndarray:
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "cache" / "feats_v25.parquet", columns=["MVT_ID_mvt", "d_lobt"])
    s = pd.Series(np.asarray(t.column("d_lobt").to_numpy(zero_copy_only=False), dtype=float), index=t.column("MVT_ID_mvt").to_numpy())
    return s.reindex(ids).to_numpy()


def run_setting(setting: str, sm: dict, K: pd.DataFrame, C: pd.DataFrame, th: float = TH):
    """(새 후보 표, 새 캐시 표, 원 예측, 새 예측, 정보)."""
    corr_tag = sm.get("run_tag", sm["spec"])
    cache = pd.read_parquet(ROOT / "cache" / f"{setting}_{corr_tag}_s0.parquet")
    p_new, delta, rows, kap = shift(setting, cache, K, C, th)
    ids = cache["MVT_ID_mvt"].to_numpy()
    cand = pd.read_parquet(ROOT / sm[KEYPART[setting]]["path"])
    old = cand.set_index("MVT_ID_mvt")["pred"].astype(float)
    if set(old.index) != set(ids):
        raise SystemExit(f"{setting}: 후보 파일과 보정기 캐시의 행이 다르다")
    ch = delta != 0
    o = old.reindex(ids).to_numpy()
    nv = o + delta
    dl = d_lobt_of(ids[ch])
    lo_w, hi_w = dl - HALF, dl + HALF
    v = nv[ch]
    v = np.where(np.isfinite(dl), np.clip(v, lo_w, hi_w), v)
    nv[ch] = np.clip(v, 0, None)
    new = old.copy()
    new.loc[ids[ch]] = nv[ch]
    same = new.to_numpy() == old.to_numpy()
    untouched = ~new.index.isin(ids[ch])
    if not same[untouched].all():
        raise SystemExit(f"{setting}: 손대지 않을 행이 바뀌었다")
    g = cache["g"].to_numpy().astype(bool)
    if np.any(ch & g):
        raise SystemExit(f"{setting}: G 행이 바뀌었다")
    out = cand.copy()
    out["pred"] = new.reindex(cand["MVT_ID_mvt"].to_numpy()).to_numpy()
    cache_new = cache.copy()
    cache_new["p_art"] = p_new
    info = {"kappa": round(kap, 4), "cells": len(rows), "changed": int((new.to_numpy() != old.to_numpy()).sum()), "g_changed": 0,
            "move_mse": round(float(((new - old) ** 2).sum()), 1), "cells_detail": rows}
    return out, cache_new, old, new, info


def source(arg: str, strict: bool = True) -> tuple[dict, str, str | None]:
    from src.batches import manifest_of
    if arg.endswith(".json"):
        meta, stem = {"manifest": arg}, None
    else:
        meta, stem = json.loads((ROOT / "queue" / f"{arg}.json").read_text(encoding="utf-8")), arg
    sm, why = manifest_of(meta)
    if why:
        raise SystemExit(f"거부: 원 후보 manifest 검사 {why}")
    cfg = sm.get("config", {})
    tag = sm.get("run_tag", sm["spec"])
    if "--leg" in cfg or "--base" not in cfg or sm["spec"].endswith("aw") or sm["spec"].endswith("ds") or sm["spec"].endswith(SUFFIX):
        raise SystemExit("거부: 원 후보는 A3 보정기 출력(--base 가 있고 gkey 류, AIU, 반대 창, 이 단계가 아닌 판)이어야 한다")
    for s in ("val", "fwd", "full"):
        if not (ROOT / "cache" / f"{s}_{tag}_s0.parquet").exists():
            raise SystemExit(f"거부: 보정기 캐시 cache/{s}_{tag}_s0.parquet 가 없다")
    return sm, meta["manifest"], stem


def truth() -> pd.Series:
    import pyarrow.parquet as pq
    T = pq.read_table(ROOT / "cache" / "all.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "src", "y"]).to_pandas()
    T = T[(T["PHASE_mvt"].astype(str) == "DEP") & (T["src"] > 0)]
    return T.set_index("MVT_ID_mvt")["y"].astype(float)


def delta_rep(old: pd.Series, new: pd.Series, y: pd.Series) -> dict:
    yy = y.reindex(old.index).to_numpy()
    ok = np.isfinite(yy)
    o, n, yy = old.to_numpy()[ok], new.to_numpy()[ok], yy[ok]
    se0, se1 = (o - yy) ** 2, (n - yy) ** 2
    ch = n != o
    N = len(yy)
    return {"rows": int(N), "rmse_old": round(float(np.sqrt(se0.mean())), 3), "rmse_new": round(float(np.sqrt(se1.mean())), 3),
            "dmse": round(float((se1.sum() - se0.sum()) / N), 1), "changed": int(ch.sum()), "improved": int((se1 < se0)[ch].sum())}


def check(arg: str) -> int:
    sm, src_mp, _ = source(arg, strict=False)
    K, C = build_table()
    y = truth()
    print(f"[check] 원 {src_mp}, 보정기 {sm.get('run_tag', sm['spec'])}, 표 {TABLE} {sha16(f'cache/{TABLE}.parquet')}, {CELLS} {sha16(f'cache/{CELLS}.parquet')}", flush=True)
    for setting in ("full", "val", "fwd"):
        _, _, old, new, info = run_setting(setting, sm, K, C)
        rep = {k: v for k, v in info.items() if k != "cells_detail"}
        if setting == "full":
            rep["move_mse_test_n"] = round(info["move_mse"] / N_TEST, 1)
        else:
            rep.update(delta_rep(old, new, y))
        print(f"[check] {setting}: {rep}", flush=True)
        print(f"[check] {setting} 칸: {info['cells_detail']}", flush=True)
    return 0


def build(arg: str, num: str) -> int:
    from src.ruler import ruler_hash
    sm, src_mp, stem = source(arg)
    tag = sm["spec"] + SUFFIX
    name = f"{num}_{tag}"
    if (ROOT / "queue" / f"{name}.json").exists() or (ROOT / f"preds/manifest_{tag}.json").exists():
        print("거부: 같은 이름이 이미 있다", name)
        return 1
    K, C = build_table()
    y = truth()
    rep, files = {}, {}
    for setting in ("full", "val", "fwd"):
        out, cache_new, old, new, info = run_setting(setting, sm, K, C)
        if setting == "val":
            rm = float(np.sqrt(((new - y.reindex(new.index)) ** 2).mean()))
            rel = f"preds/holdout_{tag}_rmse{rm:.2f}.parquet"
        elif setting == "fwd":
            rel = f"preds/fwd_{tag}.parquet"
        else:
            rel = f"preds/test_{tag}.parquet"
        r = {k: v for k, v in info.items() if k != "move_mse"}
        if setting == "full":
            r["move_mse_test_n"] = round(info["move_mse"] / N_TEST, 1)
        else:
            r.update(delta_rep(old, new, y))
        rep[setting] = r
        out.to_parquet(ROOT / rel, index=False)
        cache_new.to_parquet(ROOT / "cache" / f"{setting}_{tag}_s0.parquet", index=False)
        files[setting] = rel
        print(f"[{name}] {setting}: {({k: v for k, v in r.items() if k != 'cells_detail'})}", flush=True)
    es_rel = f"preds/es_valid_ids_{tag}.parquet"
    shutil.copy2(ROOT / sm["es_valid_ids"]["path"], ROOT / es_rel)
    me = "tools/artmm.py"
    h = sha16(me)
    snap = f"preds/code/{h}_artmm.py"
    if not (ROOT / snap).exists():
        shutil.copy2(ROOT / me, ROOT / snap)
    c0 = sm["config"]
    cfg = {"--base": c0["--base"], "--base_seed": c0.get("--base_seed", "0"), "--src": src_mp, "--src_sha": sha16(src_mp), "--corr": sm.get("run_tag", sm["spec"]),
           "--table": TABLE, "--table_sha": sha16(f"cache/{TABLE}.parquet"), "--cells": CELLS, "--cells_sha": sha16(f"cache/{CELLS}.parquet"),
           "--th": f"{TH:.2f}", "--tag": tag}
    flags = " ".join(f"{k} {v}" for k, v in cfg.items())
    cmd = lambda s: f".venv/Scripts/python.exe -X utf8 tools/artmm.py --setting {s} {flags}"  # noqa: E731
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
        "iteration_rule": sm.get("iteration_rule", "") + " | artmm: p_art cell means moved by kappa x (carrier-month arrival copy share minus its training-month "
                          "share) where that change is at least 0.10; kappa from training-month cells only; threshold fixed in research",
        "source": {"queue": stem, "manifest": src_mp, "manifest_sha": sha16(src_mp), "spec": sm["spec"],
                   "holdout_sha": sm["holdout"]["sha"], "test_sha": sm["test"]["sha"], "forward_sha": sm["forward"]["sha"]},
        "artmm": {"corrector": sm.get("run_tag", sm["spec"]), "table": f"cache/{TABLE}.parquet", "table_sha": cfg["--table_sha"], "cells": f"cache/{CELLS}.parquet",
                  "cells_sha": cfg["--cells_sha"], "threshold": TH, "caches": {s: f"cache/{s}_{tag}_s0.parquet" for s in ("val", "fwd", "full")},
                  "model": "per airport x carrier x month cell with |arrival copy share change| >= 0.10: one logit shift so that the cell's mean p_art moves by "
                           "kappa x change; prediction moved by (p' - p)(d - mn); LOBT window and 0 floor on changed rows; G rows and p_art = 0 rows untouched; "
                           "the cache files carry p' for the AIU day shift; gkey and later steps go after this",
                  "caveats": ["the 0.10 threshold was chosen after seeing that 0.05 lost on val (+25) and fwd (+75); the label-only cell LOMO does not separate "
                              "0.05 from 0.10 (d_sch^2-weighted -10.9% vs -10.3%)",
                              "the val (-17) and fwd (-1) gains are noise-sized; the strong ruler evidence (nd -325) is a single event, THY at LIRF in "
                              "Nov-Dec 2025"],
                  "report": rep},
        "written_kst": now_kst(),
    }
    mp = f"preds/manifest_{tag}.json"
    (ROOT / mp).write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{name}] manifest {mp}", flush=True)
    cells_txt = ", ".join(f"{c['ap']} {c['car']} {c['mon']}(Δa {c['da']:+.3f}, {c['rows']}행)" for c in rep["full"]["cells_detail"])
    method = (f"{stem or src_mp} 의 G 아닌 출발 가운데, 같은 공항 같은 운항사의 그 달 도착 기록 SCHED 복사 몫이 학습 달보다 0.10 넘게 움직인 칸에서 주 분류기 p_art "
              f"평균을 κ × 그 변화만큼 옮긴 판(tools/artmm.py, 표 {TABLE} {cfg['--table_sha']}, {CELLS} {cfg['--cells_sha']}). 예측은 (p' − p)(d − mn) 만큼, "
              "LOBT 창 안으로. G 행과 칸 밖 행은 그대로. κ 는 학습 달 칸에서만 쟀다. 근거 vault research/2026-10-06-r6-g.md. gkey 와 그 뒤 단계는 이 뒤에 둔다. "
              f"시험 칸: {cells_txt}. 자 ΔMSE {rep['val']['dmse']}({rep['val']['improved']}/{rep['val']['changed']}), 7월 전방 ΔMSE {rep['fwd']['dmse']}"
              f"({rep['fwd']['improved']}/{rep['fwd']['changed']}), 시험 바뀐 행 {rep['full']['changed']}, 이동 MSE {rep['full']['move_mse_test_n']}. "
              "약점 둘: 문턱 0.10 은 0.05 가 val(+25)과 fwd(+75)에서 진 것을 본 뒤 골랐고(정답만 쓴 칸 단위 한 달 빼기는 둘을 가르지 못한다), "
              "val 과 fwd 이득은 잡음 크기이며 강한 자 증거(nd -325)는 THY(LIRF, 2025 11 ~ 12월) 사건 하나다. "
              "10-05 단일 발 체제라 준비 점검은 경고(--readiness-warn)")
    (ROOT / "notes" / f"artmm_{name}.json").write_text(json.dumps({"name": name, "manifest": mp, "source": stem or src_mp, "method": method, "report": rep,
                                                                   "written_kst": now_kst()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def enqueue(name: str) -> int:
    info = json.loads((ROOT / "notes" / f"artmm_{name}.json").read_text(encoding="utf-8"))
    r = subprocess.run([sys.executable, "-X", "utf8", "tools/enqueue.py", info["manifest"], name, info["method"], "--readiness-warn"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1500:], r.stderr[-800:], flush=True)
    return r.returncode


def main() -> int:
    below_normal()
    a = sys.argv[1:]
    if a and a[0] == "table":
        build_table(force="--force" in a)
        return 0
    if a and a[0] == "check":
        return check(a[1])
    if a and a[0] == "_files":
        return build(a[1], a[2])
    if a and a[0] == "enqueue":
        return enqueue(a[1])
    if a and a[0] == "build":
        if len(a) < 3:
            print("usage: artmm.py build <A3 판 큐 이름 또는 manifest> <새 큐 번호> [--no-enqueue]")
            return 2
        me = [sys.executable, "-X", "utf8", str(Path(__file__).resolve())]
        rc = subprocess.run(me + ["_files", a[1], a[2]], cwd=ROOT).returncode
        if rc != 0 or "--no-enqueue" in a:
            return rc
        sm, _, _ = source(a[1])
        return subprocess.run(me + ["enqueue", f"{a[2]}_{sm['spec']}{SUFFIX}"], cwd=ROOT).returncode
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--seed", default="0")
    for k in ("--base", "--base_seed", "--src", "--src_sha", "--corr", "--table", "--table_sha", "--cells", "--cells_sha", "--th", "--tag"):
        pa.add_argument(k, required=True)
    g = pa.parse_args(a)
    if sha16(g.src) != g.src_sha or sha16(f"cache/{g.table}.parquet") != g.table_sha or sha16(f"cache/{g.cells}.parquet") != g.cells_sha:
        print("거부: 원 manifest 나 표 해시가 다르다")
        return 1
    if abs(float(g.th) - TH) > 1e-12:
        print("거부: 문턱이 도구의 것과 다르다")
        return 1
    sm = json.loads((ROOT / g.src).read_text(encoding="utf-8"))
    if sm.get("run_tag", sm["spec"]) != g.corr:
        print("거부: 보정기 태그가 원 manifest 와 다르다")
        return 1
    K, C = pd.read_parquet(ROOT / "cache" / f"{g.table}.parquet"), pd.read_parquet(ROOT / "cache" / f"{g.cells}.parquet")
    out, cache_new, _, _, info = run_setting(g.setting, sm, K, C)
    rel = sorted((ROOT / "preds").glob(f"holdout_{g.tag}_rmse*.parquet"))[0].relative_to(ROOT) if g.setting == "val" else \
        Path(f"preds/{'fwd' if g.setting == 'fwd' else 'test'}_{g.tag}.parquet")
    out.to_parquet(ROOT / rel, index=False)
    cache_new.to_parquet(ROOT / "cache" / f"{g.setting}_{g.tag}_s0.parquet", index=False)
    print(rel, {k: v for k, v in info.items() if k != "cells_detail"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
