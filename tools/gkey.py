# SPDX-License-Identifier: GPL-3.0-only
"""G 행(LIRF 출발 가운데 NM 미매칭)만 새 G 모형으로 다시 계산해 큐 후보 하나 위에 얹는다. 반대 창(tools/antiwin.py) 앞에 둔다.
근거: vault `01 Projects/공모전 파이프라인/착수/PRC Data Challenge 2026/research/2026-10-06-r5-analogs.md` 4.5(연구 스크립트
research/r5_analogs/g_carrier2.py), R3 2.5(research/r3_structure/p5_g_rep.py). 생각의 출처: Pushback to the Future 우승 팀들의 항공사 코드,
OSN Discord 「the echo of the planned time ... is mostly a carrier thing when there's no NM match」(README 「외부 아이디어」).

새 G 모형: src/model3.g_probs 와 같은 식(칸 빈도 + 키 보정과 작은 다중분류 GBM 세 시드를 반반)을 키 둘로 두 번 불러 세 유형 확률을 반씩 평균한다.
  키 1: 목적지(ADES, 지금 G 모형의 키. 피처 판에 prefix 가 있으면 g_band 처럼 그것), 키 2: 편명 앞 ICAO 세 글자(없으면 앞 두 글자, 편명이 비면 "D:" + 키 1).
  두 번 모두 GBM 쪽에 R3 스탠드, 이음 칸 아홉을 더한다: st_arr_between_sch_mvt, st_next_arr_gap, st_prev_arr_gap, st_prev2_arr_gap, st_prev_dep_gap
  (피처 판), k_since_prevdep, arr_after_sch(아래 표의 g1 과 d_sch), link_nm, link_in_win(반대 창 이음 표의 SCHED − LOBT_arr).
  띠 규칙(model3.g_override: 지연 1~4시간 행의 art 확률을 주 분류기 p_art 와 반반)과 mu 는 G 다리(src/g_band.py 출력)의 것을 그대로 쓴다.
옛 G 는 같은 다리의 설정(키 1, 칸 없음)으로 다시 계산해 다리 파일의 G 값과 같은지(최대 차 1e-3 초) 먼저 확인한다. 어긋나면 멈춘다.
덮기: 후보 예측의 G 행에 (새 G − 옛 G)를 더한다(G 행에 걸린 뒤 단계의 보정은 그대로 둔다). 0 아래는 0.
학습 행: 설정마다 g_band 와 같다(val 은 2025 의 1, 7월 밖, fwd 는 1~6월, full 은 2025 전체). 평균 가중 0.5, 칸 아홉, 키는 연구에서 정한 그대로이고
  자로 고르지 않는다. 이 단계에는 적합되는 상수 c, m 이 없다(그것은 반대 창의 몫).
반대 창의 R1 행은 G 행이다. 반대 창은 R1 행을 덮어쓰고 R2 행에 하한을 거니, 이 덮기는 반드시 반대 창 앞에 두고 반대 창을 맨 끝에 둔다.
표 cache/gkey_v1.parquet(+json): G 행마다 운항사 글자(car), k_since_prevdep, g1(src/feats_ctx2.stand_columns 와 같은 정의를 LIRF 에만,
  달 경계를 넘는다), sch_m_lobt(cache/antiwin_link_v1.parquet). 입력 칸만 쓴다. 만들 때 R3 의 표(s2_stand, s5_link)와 행마다 견준 결과를 json 에 둔다.

쓰는 법(저장소에서, memcap 안, 반대 창과 같은 꼴):
    .venv/Scripts/python.exe -X utf8 tools/gkey.py table [--force]
    .venv/Scripts/python.exe -X utf8 tools/gkey.py build <원 큐 이름(반대 창 전)> <새 큐 번호> [--no-enqueue] [--variant {avg_new,avg}] [--leg <g_band 태그> --leg_feats <피처 판>]
        --variant avg_new(기본, spec 끝 gk): 측정한 그대로 두 키 평균 + 칸 아홉. avg(spec 끝 gkc): 두 키 평균만(칸 없음).
        원 후보 manifest 의 자, 7월 전방, 시험 예측에 덮기를 얹어 preds/ 에 새 파일 셋과 manifest 를 쓰고, 자와 전방의 ΔMSE, 바뀐 행(좋아진 / 전체),
        시험의 바뀐 행과 이동 MSE 를 찍은 뒤 tools/enqueue.py --readiness-warn 으로 넣는다(데이터 단계와 큐 단계는 다른 프로세스). 새 spec 은 원 spec + "gk".
        G 다리는 manifest 의 --base 를 따라 내려가 manifest 가 없는 첫 태그로 정하고, 그 태그의 logs/<태그>_val_s0.log 첫 줄이 g_band 명령인지 본다.
    .venv/Scripts/python.exe -X utf8 tools/gkey.py enqueue <새 큐 이름>
    .venv/Scripts/python.exe -X utf8 tools/gkey.py --setting {val,fwd,full} --src <원 manifest> --src_sha <16> --leg <태그> --leg_feats <판>
        --table gkey_v1 --table_sha <16> --tag <새 spec>      (manifest 의 commands 가 적는 재현 명령, 그 설정의 파일 하나만 다시 쓴다)
"""
from __future__ import annotations

import ctypes
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

TABLE = "gkey_v1"
ST_FEATS = ["st_arr_between_sch_mvt", "st_next_arr_gap", "st_prev_arr_gap", "st_prev2_arr_gap", "st_prev_dep_gap"]
NEW = ST_FEATS + ["k_since_prevdep", "arr_after_sch", "link_nm", "link_in_win"]
WIN = 3606.0
N_TEST = 344841
FIT = {"val": lambda s: ~np.isin(s, [1, 7]) & (s != 0), "fwd": lambda s: np.isin(s, [1, 2, 3, 4, 5, 6]), "full": lambda s: s != 0}
EVAL = {"val": lambda s: np.isin(s, [1, 7]), "fwd": lambda s: s == 7, "full": lambda s: s == 0}
KEYPART = {"val": "holdout", "fwd": "forward", "full": "test"}
SUFFIX = {"avg_new": "gk", "avg": "gkc"}  # avg_new: 측정한 그대로(두 키 + 칸 아홉), avg: 두 키 평균만(칸 없음)


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


# ---- 표 ----

def lirf_stand_cols() -> pd.DataFrame:
    """src/feats_ctx2.stand_columns 의 st_g1, st_k 와 같은 정의를 LIRF 출발에만(2025, 2026 모든 달, 달 경계를 넘는다)."""
    t = ds.dataset(ROOT / "cache" / "all.parquet").to_table(
        columns=["MVT_ID_mvt", "PHASE_mvt", "STAND_mvt", "t_mvt", "t_blk"],
        filter=((pc.field("PHASE_mvt") == "DEP") & (pc.field("ADEP_mvt") == "LIRF")) | ((pc.field("PHASE_mvt") == "ARR") & (pc.field("ADES_mvt") == "LIRF")))
    X = t.to_pandas()
    X["stand"] = X["STAND_mvt"].astype(object)
    D = X[(X["PHASE_mvt"] == "DEP") & X["stand"].notna()][["MVT_ID_mvt", "stand", "t_mvt"]].reset_index(drop=True)
    A = X[(X["PHASE_mvt"] == "ARR") & X["stand"].notna() & np.isfinite(X["t_blk"].to_numpy(dtype=float))][["stand", "t_blk"]].reset_index(drop=True)
    g1 = np.full(len(D), np.nan)
    kk = np.full(len(D), np.nan)
    amap = {s: np.sort(g["t_blk"].to_numpy(dtype=float)) for s, g in A.groupby("stand")}
    for s, g in D.groupby("stand"):
        o = np.argsort(g["t_mvt"].to_numpy(dtype=float), kind="stable")
        rows = g.index.to_numpy()[o]
        tt = g["t_mvt"].to_numpy(dtype=float)[o]
        prev = np.concatenate([[np.nan], tt[:-1]])
        if s not in amap:
            kk[rows] = 0
            continue
        ib = amap[s]
        j = np.searchsorted(ib, tt, side="left")
        g1[rows] = np.where(j >= 1, tt - ib[np.maximum(j - 1, 0)], np.nan)
        jp = np.searchsorted(ib, np.where(np.isfinite(prev), prev, -np.inf), side="left")
        kk[rows] = np.where(np.isfinite(prev), j - jp, np.nan)
    return pd.DataFrame({"MVT_ID_mvt": D["MVT_ID_mvt"].to_numpy(), "g1": g1, "k_since_prevdep": kk})


def build_table(force: bool = False) -> pd.DataFrame:
    out = ROOT / "cache" / f"{TABLE}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    G = pd.read_parquet(ROOT / "cache" / "feats_v25.parquet", columns=["MVT_ID_mvt", "src", "ap", "nm"], filters=[("ap", "==", "LIRF"), ("nm", "==", 0)])
    G = G[["MVT_ID_mvt", "src"]].reset_index(drop=True)
    fl = ds.dataset(ROOT / "cache" / "all.parquet").to_table(columns=["MVT_ID_mvt", "FLIGHT_mvt"],
                                                             filter=(pc.field("PHASE_mvt") == "DEP") & (pc.field("ADEP_mvt") == "LIRF")).to_pandas()
    fl = fl.set_index("MVT_ID_mvt")["FLIGHT_mvt"].astype(object).reindex(G["MVT_ID_mvt"].to_numpy())
    f = pd.Series(fl.to_numpy(), dtype=object).astype(str)
    car = f.str.extract(r"^([A-Z]{3})")[0].fillna(f.str.extract(r"^([A-Z0-9]{2})")[0])
    car[pd.Series(fl.to_numpy()).isna().to_numpy()] = np.nan
    G["car"] = car.to_numpy()
    S = lirf_stand_cols().set_index("MVT_ID_mvt")
    G["k_since_prevdep"] = S["k_since_prevdep"].reindex(G["MVT_ID_mvt"].to_numpy()).to_numpy()
    G["g1"] = S["g1"].reindex(G["MVT_ID_mvt"].to_numpy()).to_numpy()
    sys.path.insert(0, str(ROOT / "tools"))
    from antiwin import build_link  # 반대 창과 같은 이음 표(이미 있으면 그 파일)
    L = build_link().set_index("MVT_ID_mvt")
    G["sch_m_lobt"] = L["sch_m_lobt"].reindex(G["MVT_ID_mvt"].to_numpy()).to_numpy()
    G.to_parquet(out, index=False)
    cmp = compare_r3(G)
    info = {"rows": int(len(G)), "train_rows": int((G["src"] > 0).sum()), "test_rows": int((G["src"] == 0).sum()),
            "car_fill": {"2025": round(float(G.loc[G["src"] > 0, "car"].notna().mean()), 4), "2026": round(float(G.loc[G["src"] == 0, "car"].notna().mean()), 4)},
            "link_fill": {"2025": round(float(G.loc[G["src"] > 0, "sch_m_lobt"].notna().mean()), 4),
                          "2026": round(float(G.loc[G["src"] == 0, "sch_m_lobt"].notna().mean()), 4)},
            "check_vs_r3": cmp, "sha16": sha16(out.relative_to(ROOT)), "link_table": "cache/antiwin_link_v1.parquet",
            "link_sha16": sha16("cache/antiwin_link_v1.parquet"), "built_kst": now_kst(), "code": "tools/gkey.py table", "code_sha16": sha16("tools/gkey.py")}
    out.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print("표", out.name, json.dumps(info, ensure_ascii=False), flush=True)
    return G


def compare_r3(G: pd.DataFrame) -> dict:
    """R3 연구 표(research/r3_structure/s2_stand_*, s5_link_*_unmatched)와 G 행마다 견준다(측정 때 쓴 값과 같은지)."""
    R3 = ROOT / "research" / "r3_structure"
    fs = [R3 / "s2_stand_train.parquet", R3 / "s2_stand_test.parquet"]
    ls = [R3 / "s5_link_train_unmatched.parquet", R3 / "s5_link_test_unmatched.parquet"]
    if not all(f.exists() for f in fs + ls):
        return {"status": "R3 files not present"}
    S = pd.concat([pd.read_parquet(f, columns=["MVT_ID_mvt", "g1", "k_since_prevdep"]) for f in fs]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    Ltr = pd.read_parquet(ls[0], columns=["MVT_ID_mvt", "sch_m_lobt"])
    Lte = pd.read_parquet(ls[1])
    Lte = pd.DataFrame({"MVT_ID_mvt": Lte["MVT_ID_mvt"], "sch_m_lobt": Lte["t_sch"] - Lte["a_t_lobt"]})
    L = pd.concat([Ltr, Lte]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    ids = G["MVT_ID_mvt"].to_numpy()
    res = {}
    for mine, theirs in (("g1", S["g1"].reindex(ids)), ("k_since_prevdep", S["k_since_prevdep"].reindex(ids)), ("sch_m_lobt", L["sch_m_lobt"].reindex(ids))):
        a, b = G[mine].to_numpy(float), theirs.to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        res[mine] = {"nan_mismatch": int((np.isfinite(a) != np.isfinite(b)).sum()),
                     "max_abs_diff": float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0, "rows_both": int(both.sum())}
    return res


# ---- G 모형 ----

def find_leg(sm: dict) -> tuple[str, str]:
    """manifest 의 --base 를 따라 내려가 manifest 가 없는 첫 태그(G 다리)와 그 g_band 명령의 --feats 를 돌려준다."""
    tag = sm["config"].get("--base")
    seen = set()
    while tag and (ROOT / "preds" / f"manifest_{tag}.json").exists() and tag not in seen:
        seen.add(tag)
        tag = json.loads((ROOT / "preds" / f"manifest_{tag}.json").read_text(encoding="utf-8"))["config"].get("--base")
    lg = ROOT / "logs" / f"{tag}_val_s0.log"
    if not tag or not lg.exists():
        raise SystemExit(f"G 다리를 찾지 못했다(태그 {tag}). --leg, --leg_feats 를 주라")
    first = lg.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    if "src/g_band.py" not in first or f"--tag {tag}" not in first:
        raise SystemExit(f"{lg.name} 첫 줄이 g_band 명령이 아니다: {first[:200]}")
    m = re.search(r"--feats (\S+)", first)
    return tag, (m.group(1) if m else "feats_v25")


def g_frame(feats: str) -> pd.DataFrame:
    """g_band 와 같은 G 표(LIRF, nm 0) + 스탠드 칸 다섯. 행 순서는 피처 판 순서."""
    fp = ROOT / "cache" / f"{feats}.parquet"
    names = set(pq.read_schema(fp).names)
    gkey = "prefix" if "prefix" in names else "ades"
    want = ["MVT_ID_mvt", "src", "ap", "nm", "y", "d_sch", "dd", gkey, "hour_mvt", "hour_sch", "garb", "fl_len", "stand"] + ST_FEATS
    G = pd.read_parquet(fp, columns=[c for c in dict.fromkeys(want) if c in names], filters=[("ap", "==", "LIRF"), ("nm", "==", 0)])
    G = G.reset_index(drop=True)
    G["pfx"] = G[gkey].astype(str).fillna("nan")
    G["stand_d"] = pd.to_numeric(G["stand"].astype(str).fillna("nan").str[5:6], errors="coerce")
    return G


def add_new(G: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    T = T.set_index("MVT_ID_mvt")
    ids = G["MVT_ID_mvt"].to_numpy()
    miss = int((~pd.Index(ids).isin(T.index)).sum())
    if miss:
        raise SystemExit(f"표 {TABLE} 에 없는 G 행이 {miss} 개다(tools/gkey.py table --force)")
    H = G.copy()
    H["k_since_prevdep"] = T["k_since_prevdep"].reindex(ids).to_numpy()
    g1 = T["g1"].reindex(ids).to_numpy()
    H["arr_after_sch"] = ((g1 < H["d_sch"].to_numpy(dtype=float) - 300) & (g1 > 900)).astype(float)
    s = T["sch_m_lobt"].reindex(ids).to_numpy(dtype=float)
    H["link_nm"] = np.isfinite(s).astype(float)
    H["link_in_win"] = np.where(np.isfinite(s), (np.abs(s) <= WIN).astype(float), np.nan)
    car = T["car"].reindex(ids).to_numpy(dtype=object)
    pfx = H["pfx"].to_numpy(dtype=object)
    H["k_car"] = [("C:" + str(c)) if isinstance(c, str) else ("D:" + str(p)) for c, p in zip(car, pfx)]
    return H


def probs(Gtr: pd.DataFrame, Gte: pd.DataFrame, extra: list[str], seeds=(0, 1, 2)) -> np.ndarray:
    """src/model3.g_probs 와 같은 식에 GBM 쪽 칸 extra 를 더한 것(extra 가 비면 g_probs 와 같다). 키 칸은 "pfx"."""
    import src.sub_g as sub_g
    ytr = Gtr["y"].to_numpy()
    dtr = Gtr["d_sch"].to_numpy().astype(float)
    k = sub_g.labels(ytr, dtr)
    c_tr = sub_g.cell(dtr, Gtr["dd"].to_numpy())
    c_te = sub_g.cell(Gte["d_sch"].to_numpy().astype(float), Gte["dd"].to_numpy())
    P1 = sub_g.pfx_adjust(sub_g.predict_lookup(sub_g.fit_lookup(c_tr, k), c_te), Gte["pfx"].to_numpy(), Gtr["pfx"].to_numpy(), k)
    feat = [c for c in ["d_sch", "dd", "hour_mvt", "hour_sch", "garb", "fl_len", "stand_d"] if c in Gtr.columns] + extra
    cats = pd.CategoricalDtype(sorted(set(Gtr["pfx"]) | set(Gte["pfx"])))
    Xtr = Gtr[feat].astype("float32").copy()
    Xtr["pfx"] = Gtr["pfx"].astype(cats)
    Xte = Gte[feat].astype("float32").copy()
    Xte["pfx"] = Gte["pfx"].astype(cats)
    P2 = np.mean([sub_g.gbm_probs(Xtr, k, Xte, seed=s) for s in seeds], axis=0)
    return 0.5 * P1 + 0.5 * P2


def g_values(setting: str, leg: str, feats: str, T: pd.DataFrame, variant: str = "avg_new") -> pd.DataFrame:
    """그 설정의 평가 G 행마다 옛 G(다리 재현), 새 G, 다리 파일 값. 재현이 어긋나면 멈춘다."""
    from src.model3 import g_override, g_probs
    G = g_frame(feats)
    s = G["src"].to_numpy()
    fit, ev = FIT[setting](s), EVAL[setting](s)
    Gf = G[fit & ~np.isnan(G["y"].to_numpy(dtype=float))]
    Ge = G[ev]
    b = pd.read_parquet(ROOT / "cache" / f"{setting}_{leg}_s0.parquet", columns=["MVT_ID_mvt", "pred", "mu", "p_art", "g"]).set_index("MVT_ID_mvt")
    gid = b.index[b["g"].to_numpy().astype(bool)]
    if set(gid) != set(Ge["MVT_ID_mvt"]):
        raise SystemExit(f"{setting}: 다리 {leg} 의 G 행과 피처 판의 G 행이 다르다({len(gid)} 대 {len(Ge)})")
    bb = b.reindex(Ge["MVT_ID_mvt"].to_numpy())
    mu_g, pa_g = bb["mu"].to_numpy(dtype=float), bb["p_art"].to_numpy(dtype=float)
    d, dd = Ge["d_sch"].to_numpy(dtype=float), Ge["dd"].to_numpy()
    g_old = np.clip(g_override(g_probs(Gf, Ge), d, dd, mu_g, pa_g)[0], 0, None)
    err = float(np.max(np.abs(g_old - bb["pred"].to_numpy(dtype=float)))) if len(Ge) else 0.0
    print(f"  {setting}: 옛 G 재현, 다리 {leg} 의 G {len(Ge)} 행과 최대 차 {err:.3e} 초", flush=True)
    if err >= 1e-3:
        raise SystemExit(f"{setting}: 옛 G 재현이 다리 파일과 다르다(최대 차 {err})")
    Hf, He = add_new(Gf, T), add_new(Ge, T)
    extra = NEW if variant == "avg_new" else []
    Pa = probs(Hf, He, extra)
    Hf2, He2 = Hf.copy(), He.copy()
    Hf2["pfx"], He2["pfx"] = Hf2["k_car"], He2["k_car"]
    Pc = probs(Hf2, He2, extra)
    g_new = np.clip(g_override(0.5 * (Pa + Pc), d, dd, mu_g, pa_g)[0], 0, None)
    return pd.DataFrame({"g_old": g_old, "g_new": g_new, "leg": bb["pred"].to_numpy(dtype=float)}, index=Ge["MVT_ID_mvt"].to_numpy())


def overlay(pred: pd.Series, V: pd.DataFrame) -> tuple[pd.Series, dict]:
    """pred: MVT_ID 색인의 후보 예측. G 행에 (새 G − 옛 G)를 더한 새 Series 와 정보."""
    miss = int((~V.index.isin(pred.index)).sum())
    if miss:
        raise SystemExit(f"후보 예측에 없는 G 행이 {miss} 개다")
    new = pred.copy()
    old_g = pred.reindex(V.index).to_numpy(dtype=float)
    later = old_g - V["g_old"].to_numpy()
    new.loc[V.index] = np.clip(old_g + (V["g_new"].to_numpy() - V["g_old"].to_numpy()), 0, None)
    ch = new.to_numpy() != pred.to_numpy()
    info = {"g_rows": int(len(V)), "changed": int(ch.sum()), "outside_g": int((~pred.index[ch].isin(V.index)).sum()),
            "later_stage_rows": int((np.abs(later) > 1e-6).sum()), "later_stage_mean_abs": round(float(np.mean(np.abs(later))), 2),
            "mean_g_old": round(float(V["g_old"].mean()), 1), "mean_g_new": round(float(V["g_new"].mean()), 1)}
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


def write_setting(setting: str, sm: dict, tag: str, leg: str, feats: str, T: pd.DataFrame, variant: str = "avg_new") -> tuple[str, pd.Series, pd.Series, dict]:
    old_df = pd.read_parquet(ROOT / sm[KEYPART[setting]]["path"])
    old = old_df.set_index("MVT_ID_mvt")["pred"].astype(float)
    V = g_values(setting, leg, feats, T, variant)
    new, info = overlay(old, V)
    out = old_df.copy()
    out["pred"] = new.reindex(old_df["MVT_ID_mvt"].to_numpy()).to_numpy()
    if setting == "val":
        y = truth()
        rm = float(np.sqrt(((new - y.reindex(new.index)) ** 2).mean()))
        rel = f"preds/holdout_{tag}_rmse{rm:.2f}.parquet"
    elif setting == "fwd":
        rel = f"preds/fwd_{tag}.parquet"
    else:
        rel = f"preds/test_{tag}.parquet"
    out.to_parquet(ROOT / rel, index=False)
    return rel, old, new, info


def build(src_stem: str, num: str, leg: str | None, leg_feats: str | None, variant: str = "avg_new") -> int:
    from src.batches import manifest_of
    from src.ruler import ruler_hash
    meta = json.loads((ROOT / "queue" / f"{src_stem}.json").read_text(encoding="utf-8"))
    sm, why = manifest_of(meta)
    if why:
        print("거부: 원 후보 manifest 검사", why)
        return 1
    if sm["spec"].endswith("aw"):
        print("거부: 원 후보가 반대 창 판이다. 이 덮기는 반대 창 앞에 둔다(반대 창 전 후보를 주라)")
        return 1
    for f in ("src/model3.py", "src/sub_g.py"):  # 옛 G 재현과 새 G 가 쓰는 코드는 원 후보의 것과 같아야 한다(재현 확인이 한 번 더 잡는다)
        if f in sm["code_sha"] and sha16(f) != sm["code_sha"][f]:
            print(f"거부: {f} 가 원 후보 manifest 의 사본과 다르다")
            return 1
    src_mp = meta["manifest"]
    tag = sm["spec"] + SUFFIX[variant]
    name = f"{num}_{tag}"
    if (ROOT / "queue" / f"{name}.json").exists() or (ROOT / f"preds/manifest_{tag}.json").exists():
        print("거부: 같은 이름이 이미 있다", name)
        return 1
    if leg is None:
        leg, leg_feats = find_leg(sm)
    print(f"[{name}] 원 {src_stem}, G 다리 {leg}(피처 {leg_feats})", flush=True)
    T = build_table()
    tab_rel = f"cache/{TABLE}.parquet"
    tab_sha = sha16(tab_rel)
    y = truth()
    rep = {}
    files = {}
    for setting in ("full", "val", "fwd"):  # 시험부터: 바뀐 행이 G 행 안에만 있는지, 수가 G 행 수 안인지 확인하고 나서 나머지를 쓴다
        rel, old, new, info = write_setting(setting, sm, tag, leg, leg_feats, T, variant)
        files[setting] = rel
        if setting == "full":
            rep[setting] = {**info, "move_mse_test_n": round(float(((new - old) ** 2).sum() / N_TEST), 1)}
            print(f"[{name}] full: {rep[setting]}", flush=True)
            if info["outside_g"] or not (100 <= info["changed"] <= info["g_rows"]):
                (ROOT / rel).unlink()
                print("거부: 시험에서 바뀐 행이 예상과 다르다(G 행 밖이거나 100 ~ G 행 수 밖)")
                return 1
        else:
            rep[setting] = {**info, **delta(old, new, y)}
            print(f"[{name}] {setting}: {rep[setting]}", flush=True)
    es_rel = f"preds/es_valid_ids_{tag}.parquet"
    shutil.copy2(ROOT / sm["es_valid_ids"]["path"], ROOT / es_rel)
    me = "tools/gkey.py"
    h = sha16(me)
    snap = f"preds/code/{h}_gkey.py"
    if not (ROOT / snap).exists():
        shutil.copy2(ROOT / me, ROOT / snap)
    aw = "tools/antiwin.py"  # 표의 이음 칸을 만드는 build_link
    ha = sha16(aw)
    snap_aw = f"preds/code/{ha}_antiwin.py"
    if not (ROOT / snap_aw).exists():
        shutil.copy2(ROOT / aw, ROOT / snap_aw)
    cfg = {"--src": src_mp, "--src_sha": sha16(src_mp), "--leg": leg, "--leg_feats": leg_feats, "--table": TABLE, "--table_sha": tab_sha,
           "--variant": variant, "--tag": tag}
    flags = " ".join(f"{k} {v}" for k, v in cfg.items())
    cmd = lambda s: f".venv/Scripts/python.exe -X utf8 tools/gkey.py --setting {s} {flags}"  # noqa: E731
    hold_ids = pd.read_parquet(ROOT / files["val"])["MVT_ID_mvt"].to_numpy()
    man = {
        "spec": tag, "run_tag": tag, "column": "pred", "seeds": sm["seeds"], "feats": sm["feats"], "feats_sha": sm["feats_sha"],
        "config": cfg,
        "commands": {"holdout": [cmd("val")], "test": [cmd("full")], "forward": [cmd("fwd")]},
        "command": cmd("val"),
        "code_sha": {**sm["code_sha"], me: h, aw: ha},
        "code_snapshots": {**sm.get("code_snapshots", {}), me: snap, aw: snap_aw},
        "holdout": {"path": files["val"], "sha": sha16(files["val"]), "rmse": round(rep["val"]["rmse_new"], 2), "ruler_hash": ruler_hash(hold_ids)},
        "test": {"path": files["full"], "sha": sha16(files["full"])},
        "forward": {"path": files["fwd"], "sha": sha16(files["fwd"]), "rmse_2025_07": round(rep["fwd"]["rmse_new"], 2)},
        "es_valid_ids": {**sm["es_valid_ids"], "path": es_rel, "sha": sha16(es_rel)},
        "iteration_rule": sm.get("iteration_rule", "") + " | gkey: G model refit per setting on the g_band training rows (2025 months outside the evaluated ones); "
                          "key average 0.5 and the nine columns fixed in research, nothing chosen on the ruler",
        "source": {"queue": src_stem, "manifest": src_mp, "manifest_sha": sha16(src_mp), "spec": sm["spec"],
                   "holdout_sha": sm["holdout"]["sha"], "test_sha": sm["test"]["sha"], "forward_sha": sm["forward"]["sha"]},
        "gkey": {"leg": leg, "leg_feats": leg_feats, "table": tab_rel, "table_sha": tab_sha, "new_columns": NEW,
                 "variant": variant,
                 "model": "mean of src/model3.g_probs-form probabilities with key ADES and key ICAO carrier"
                          + (", both with the nine columns; " if variant == "avg_new" else ", no extra columns; ") +
                          "band rule and mu of the g_band leg; added to the candidate as (new G - reproduced old G); antiwin must stay after this",
                 "report": rep},
        "written_kst": now_kst(),
    }
    mp = f"preds/manifest_{tag}.json"
    (ROOT / mp).write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{name}] manifest {mp}", flush=True)
    method = (f"{src_stem} 의 G 행(LIRF, NM 미매칭)만 새 G 모형으로 바꾼 판(tools/gkey.py, 표 {tab_rel} sha {tab_sha}, G 다리 {leg}). "
              "model3.g_probs 꼴의 세 유형 확률을 목적지 키와 편명 앞 ICAO 운항사 키로 두 번 내어 반씩 평균하고, GBM 쪽에 R3 스탠드, 이음 칸 아홉을 더했다. "
              "띠 규칙과 mu 는 다리의 것, 후보에는 (새 G − 재현한 옛 G)를 더했다. 학습 행은 설정마다 g_band 와 같고 자로 고른 값은 없다. "
              "근거 vault research/2026-10-06-r5-analogs.md 4.5. 반대 창은 이 뒤에 둔다. "
              f"자 ΔMSE {rep['val']['dmse']}({rep['val']['improved']}/{rep['val']['changed']}), 7월 전방 ΔMSE {rep['fwd']['dmse']}"
              f"({rep['fwd']['improved']}/{rep['fwd']['changed']}), 시험 바뀐 행 {rep['full']['changed']}, 이동 MSE {rep['full']['move_mse_test_n']}, "
              "10-05 단일 발 체제라 준비 점검은 경고(--readiness-warn)")
    (ROOT / "notes" / f"gkey_{name}.json").write_text(json.dumps({"name": name, "manifest": mp, "source": src_stem, "method": method, "report": rep,
                                                                  "written_kst": now_kst()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def enqueue(name: str) -> int:
    """build 가 남긴 notes/gkey_<name>.json 의 manifest 와 설명으로 tools/enqueue.py --readiness-warn 을 부른다(데이터를 든 프로세스 밖에서)."""
    info = json.loads((ROOT / "notes" / f"gkey_{name}.json").read_text(encoding="utf-8"))
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
    if a and a[0] == "_files":
        leg = a[a.index("--leg") + 1] if "--leg" in a else None
        lf = a[a.index("--leg_feats") + 1] if "--leg_feats" in a else None
        var = a[a.index("--variant") + 1] if "--variant" in a else "avg_new"
        return build(a[1], a[2], leg, lf, var)
    if a and a[0] == "enqueue":
        return enqueue(a[1])
    if a and a[0] == "build":
        if len(a) < 3:
            print("usage: gkey.py build <원 큐 이름> <새 큐 번호> [--no-enqueue] [--leg <태그> --leg_feats <판>]")
            return 2
        if ("--leg" in a) != ("--leg_feats" in a):
            print("--leg 와 --leg_feats 는 함께 준다")
            return 2
        me = [sys.executable, "-X", "utf8", str(Path(__file__).resolve())]
        extra = [x for x in a[3:] if x != "--no-enqueue"]
        rc = subprocess.run(me + ["_files", a[1], a[2]] + extra, cwd=ROOT).returncode  # 데이터 단계는 따로 끝내 메모리를 돌려준다
        if rc != 0 or "--no-enqueue" in a:
            return rc
        meta = json.loads((ROOT / "queue" / f"{a[1]}.json").read_text(encoding="utf-8"))
        spec = json.loads((ROOT / meta["manifest"]).read_text(encoding="utf-8"))["spec"]
        var = a[a.index("--variant") + 1] if "--variant" in a else "avg_new"
        return subprocess.run(me + ["enqueue", f"{a[2]}_{spec}{SUFFIX[var]}"], cwd=ROOT).returncode
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--seed", default="0")
    pa.add_argument("--src", required=True)
    pa.add_argument("--src_sha", required=True)
    pa.add_argument("--leg", required=True)
    pa.add_argument("--leg_feats", required=True)
    pa.add_argument("--table", default=TABLE)
    pa.add_argument("--table_sha", required=True)
    pa.add_argument("--variant", default="avg_new", choices=list(SUFFIX))
    pa.add_argument("--tag", required=True)
    g = pa.parse_args(a)
    if sha16(g.src) != g.src_sha:
        print("거부: 원 manifest 해시가 다르다")
        return 1
    if sha16(f"cache/{g.table}.parquet") != g.table_sha:
        print("거부: 표 해시가 다르다")
        return 1
    sm = json.loads((ROOT / g.src).read_text(encoding="utf-8"))
    rel, _, _, info = write_setting(g.setting, sm, g.tag, g.leg, g.leg_feats, pd.read_parquet(ROOT / "cache" / f"{g.table}.parquet"), g.variant)
    print(rel, info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
