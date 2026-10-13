# SPDX-License-Identifier: GPL-3.0-only
"""Copy-probability step for LIRF departures with an NM record (docs/DESIGN.md section 8.3), right after artmm.

p_art moves in log-odds by one coefficient beta times the difference between the flight number's copy share in the fitting months
(smoothed toward its route and carrier) and the month's mean p_art of that flight number. beta is fitted on the ruler rows with
day-fold nesting (val), on January (fwd) or on both months (full). The prediction moves with p_art and stays inside the LOBT window.

    python tools/arthist.py table                          # cache/arthist_v1.parquet
    python tools/arthist.py --setting {val,fwd,full} --base <base> --base_seed 0 --src <manifest> --src_sha <16> --corr <corrector tag>
        --table arthist_v1 --table_sha <16> --tag <spec>

Original note (Korean):
LIRF G 아닌 출발의 주 분류기 p_art 를 같은 편명의 학습 달 art 몫(정답 이력) 쪽으로 로짓 계수 하나만큼 옮긴다. artmm 바로 뒤, gkey 앞에 둔다.
순서: 바탕 다리 → V5 기하 → A3 x0.5 → artmm → 이 단계 → gkey(avg) → gkeyx → gkeyxn → AIU 날 이동 → 반대 창(맨 끝).
근거: vault `01 Projects/공모전 파이프라인/착수/PRC Data Challenge 2026/research/2026-10-07-r7-u.md`(연구 스크립트 research/r7_u/c4_param.py 의 P1h,
c6_lomo.py). 생각의 출처: R5 기록 감식(LIRF 에서만 편명의 다른 달 복사 비율이 복사를 가른다, 공항 안 AUC 0.667)과 R6 의 G 칸(같은 달 편명 이력).
남의 코드 없음. 입력은 편명, 목적지, 날짜이고, 정답은 학습 달 출발의 art 여부(이력)와 계수 β 를 맞추는 자 행에만 쓴다.

정의:
  편명 키 fc = tools/gkeyx.canon(FLIGHT_mvt)(겹친 글자 지움), 운항사 = fc 앞 세 글자(영문 대문자일 때), 노선 = 목적지(ADES_mvt).
  달 = 2025 은 src(1 ~ 12), 2026 은 100 + 이륙 UTC 달. art = |y − (MVT − SCHED)| ≤ 60초.
  이력(설정의 학습 달 = val 은 1, 7월 밖, fwd 는 1 ~ 6월, full 은 2025 전부, LIRF NM 매칭 출발만):
    c = 운항사 art 몫, r = (노선 art 수 + 50 c) / (노선 행 + 50), h = (편명 art 수 + 20 r) / (편명 행 + 20)
  x = logit(h) − logit(p̄), p̄ = 그 편명의 그 달 평가 행 평균 p_art(입력만). 둘 다 0.001 ~ 0.999 로 자른다. x 가 없으면 0.
  p' = σ(logit p + β x), p 는 원 후보(artmm 판)의 캐시 p_art. p = 0 인 행(LOBT 창 규칙 1로 art 가 불가능한 행)과 G 행은 그대로.
  β: 자 행(val 1, 7월, 원 후보 val 캐시의 p 와 val 이력)에서 로그 손실 + 0.01 β² n/1000 을 L-BFGS 로. val 은 날 열 겹(src.adsb_corr.day_folds,
     씨앗 0)마다 다른 겹으로, fwd 는 1월 행으로, full 은 두 달 전부로 맞춘다(A3 보정기 2단계와 같은 꼴).
  예측: Δ = (p' − p)(d − mn), mn = (f_corr − p_corr d)/(1 − p_corr)(보정기 캐시의 pred 와 p_art, p 는 0.0001 ~ 0.999). 새 예측 = 원 후보 예측 + Δ 를
  LOBT 창 [d_lobt − 3,606, d_lobt + 3,606](feats_v25 d_lobt, 창이 있는 행만) 안으로, 0 아래는 0. 바꾸지 않은 행은 바이트까지 같은지 확인한다.
AIU 날 이동이 쓰도록 원 후보 캐시와 같은 칸의 cache/<setting>_<새 spec>_s0.parquet 를 쓴다(p_art 만 p', pred 와 mu 는 그대로라 AIU 의 정상 평균
  mn = mu + (pred − base) 가 그대로다). manifest 의 --base 는 원 보정기의 것을 그대로 두어 gkey 의 G 다리 찾기와 AIU 의 바탕 읽기가 같게 간다.

쓰는 법(저장소에서, memcap 2 GB 안):
    .venv/Scripts/python.exe -X utf8 tools/arthist.py table [--force]      표 cache/arthist_v1.parquet(LIRF 출발: 편명 키, 운항사, 노선, 달, 날, art 여부)와 json
    .venv/Scripts/python.exe -X utf8 tools/arthist.py check <artmm 판 manifest 또는 큐 이름>   세 설정의 ΔMSE, β, 이동만 찍는다(표 말고는 쓰지 않는다)
    .venv/Scripts/python.exe -X utf8 tools/arthist.py build <artmm 판 큐 이름 또는 manifest> <새 큐 번호> [--no-enqueue]
        preds/ 에 새 파일 셋, cache/ 에 val, fwd, full 캐시, manifest 를 쓰고(새 spec 은 원 spec + "fh"), notes/arthist_<새 큐 이름>.json 을 남긴다.
    .venv/Scripts/python.exe -X utf8 tools/arthist.py enqueue <새 큐 이름>
    .venv/Scripts/python.exe -X utf8 tools/arthist.py --setting {val,fwd,full} --base <바탕> --base_seed <s> --src <원 manifest> --src_sha <16>
        --corr <보정기 태그> --table arthist_v1 --table_sha <16> --tag <새 spec>      (manifest 의 재현 명령)
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

TABLE = "arthist_v1"
SUFFIX = "fh"
K_FC, K_RT = 20.0, 50.0
FOLDS, FOLD_SEED = 10, 0
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


def split_car(s):
    if not isinstance(s, str) or len(s) < 4 or not s[:3].isalpha() or not s[:3].isupper():
        return None
    return s[:3]


# ---- 표 ----

def build_table(force: bool = False) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    from tools.gkeyx import canon
    out = ROOT / "cache" / f"{TABLE}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    t = ds.dataset(ROOT / "cache" / "all.parquet").to_table(
        columns=["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADES_mvt", "t_mvt", "t_sch", "y", "src"],
        filter=(pc.field("PHASE_mvt") == "DEP") & (pc.field("ADEP_mvt") == "LIRF"))
    D = t.to_pandas()
    del t
    fl = D["FLIGHT_mvt"].astype(object)
    cm = {x: canon(x) for x in pd.unique(fl.dropna())}
    T = pd.DataFrame({"MVT_ID_mvt": D["MVT_ID_mvt"].to_numpy()})
    T["src"] = D["src"].to_numpy().astype(np.int16)
    mth = pd.to_datetime(D["t_mvt"].astype(float), unit="s").dt.month.to_numpy()
    T["mon"] = np.where(T["src"].to_numpy() == 0, 100 + mth, T["src"].to_numpy()).astype(np.int16)
    T["day"] = pd.to_datetime(D["t_mvt"].astype(float), unit="s").dt.strftime("%Y-%m-%d").to_numpy()
    T["nm"] = D["FLIGHT_ID_mvt"].notna().to_numpy().astype(np.int8)
    T["fc"] = fl.map(cm).to_numpy()
    sc = {x: split_car(x) for x in pd.unique(T["fc"].dropna())}
    T["car"] = T["fc"].map(sc).to_numpy()
    T["dst"] = D["ADES_mvt"].astype(str).to_numpy()
    y = D["y"].astype(float).to_numpy()
    d = D["t_mvt"].astype(float).to_numpy() - D["t_sch"].astype(float).to_numpy()
    art = np.abs(y - d) <= 60
    k = np.where(art, 1.0, np.where((y > 60000) & ~art, 2.0, 0.0))
    k[(T["src"].to_numpy() == 0) | ~np.isfinite(y)] = np.nan
    T["k"] = k
    del D
    T.to_parquet(out, index=False)
    info = {"rows": int(len(T)), "kind": "LIRF departures (2025 and 2026): canonical flight key, carrier, destination, month, UTC day, NM match, "
                                          "2025 class (1 art, 2 day push, 0 normal; training labels, history only)",
            "sha16": sha16(out.relative_to(ROOT)), "source": "cache/all.parquet", "source_sha16": sha16("cache/all.parquet"), "built_kst": now_kst(),
            "code": "tools/arthist.py table", "code_sha16": sha16("tools/arthist.py")}
    out.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print("표", out.name, json.dumps(info, ensure_ascii=False), flush=True)
    return T


# ---- 이력 칸과 β ----

def hist_x(Z: pd.DataFrame, T: pd.DataFrame, months: list[int]) -> np.ndarray:
    H = T[T["mon"].isin(months) & (T["nm"] == 1) & T["k"].notna()]
    a = (H["k"].to_numpy() == 1).astype(float)
    H = H.assign(a=a)
    car = H.groupby("car")["a"].mean()
    cr = car.reindex(Z["car"].to_numpy()).fillna(float(a.mean())).to_numpy()
    rt = H.groupby(["car", "dst"])["a"].agg(["sum", "size"])
    kk = pd.MultiIndex.from_arrays([Z["car"], Z["dst"]])
    rs, rn = rt["sum"].reindex(kk).fillna(0).to_numpy(), rt["size"].reindex(kk).fillna(0).to_numpy()
    r_s = (rs + K_RT * cr) / (rn + K_RT)
    h = H.groupby("fc")["a"].agg(["sum", "size"])
    fc = Z["fc"].astype(str).to_numpy()
    hs, hn = h["sum"].reindex(fc).fillna(0).to_numpy(), h["size"].reindex(fc).fillna(0).to_numpy()
    h_h = np.clip((hs + K_FC * r_s) / (hn + K_FC), 1e-3, 1 - 1e-3)
    pbar = np.clip(Z.groupby([Z["fc"].astype(str), Z["mon"]])["p"].transform("mean").to_numpy(), 1e-3, 1 - 1e-3)
    return np.nan_to_num(np.log(h_h / (1 - h_h)) - np.log(pbar / (1 - pbar)))


def fit_beta(x: np.ndarray, lp: np.ndarray, a: np.ndarray) -> float:
    """로그 손실 Σ[log(1 + e^z) − a z] + 0.01 β² n/1000, z = logit p + β x 의 최소점(뉴턴, 계수 하나)."""
    lam = 0.01 * len(a) / 1000
    b = 0.0
    for _ in range(100):
        s = 1 / (1 + np.exp(-(lp + b * x)))
        g = float(x @ (s - a)) + 2 * lam * b
        h = float((x * x) @ (s * (1 - s))) + 2 * lam
        step = g / h
        b -= step
        if abs(step) < 1e-12:
            break
    return b


def frame(setting: str, src_tag: str, corr_tag: str, T: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(원 후보 캐시 전체, 그 가운데 LIRF G 아닌 행의 표: p, p_corr, d, f_corr, x[, art], day)."""
    c = pd.read_parquet(ROOT / "cache" / f"{setting}_{src_tag}_s0.parquet")
    pc_ = pd.read_parquet(ROOT / "cache" / f"{setting}_{corr_tag}_s0.parquet", columns=["MVT_ID_mvt", "p_art"]).set_index("MVT_ID_mvt")["p_art"]
    Tl = T[T["nm"] == 1].set_index("MVT_ID_mvt")
    sel = c["MVT_ID_mvt"].isin(Tl.index).to_numpy() & ~c["g"].to_numpy().astype(bool)
    Z = pd.DataFrame({"pos": np.flatnonzero(sel)}, index=c["MVT_ID_mvt"].to_numpy()[sel])
    Z["p"] = c["p_art"].to_numpy(dtype=float)[sel]
    Z["p_corr"] = pc_.reindex(Z.index).to_numpy(dtype=float)
    Z["d"] = c["dsch"].to_numpy(dtype=float)[sel]
    Z["f_corr"] = c["pred"].to_numpy(dtype=float)[sel]
    for k in ("fc", "car", "dst", "mon", "day"):
        Z[k] = Tl[k].reindex(Z.index).to_numpy()
    if np.isnan(Z["p_corr"].to_numpy()).any():
        raise SystemExit(f"{setting}: 보정기 캐시 {corr_tag} 에 없는 행이 있다")
    Z["x"] = hist_x(Z, T, FITM[setting])
    p = np.clip(Z["p"].to_numpy(), 1e-4, 0.999)
    Z["lp"] = np.log(p / (1 - p))
    if "y" in c.columns:
        Z["art"] = (np.abs(c["y"].to_numpy(dtype=float)[sel] - Z["d"].to_numpy()) <= 60).astype(float)
    return c, Z


def betas(src_tag: str, corr_tag: str, T: pd.DataFrame) -> dict:
    """자 행에서 맞춘 β: 날 겹마다(val), 1월(fwd), 두 달 전부(full)."""
    from src.adsb_corr import day_folds
    _, Zv = frame("val", src_tag, corr_tag, T)
    ok = Zv["p"].to_numpy() > 0
    fo = day_folds(np.unique(Zv["day"]), FOLDS, FOLD_SEED)
    fold = pd.Series(Zv["day"].to_numpy()).map(fo).to_numpy()
    x, lp, a = Zv["x"].to_numpy(), Zv["lp"].to_numpy(), Zv["art"].to_numpy()
    out = {"fold": {}, "fold_of_day": fo}
    for j in range(FOLDS):
        m = ok & (fold != j)
        out["fold"][j] = fit_beta(x[m], lp[m], a[m])
    jan = ok & (Zv["mon"].to_numpy() == 1)
    out["jan"] = fit_beta(x[jan], lp[jan], a[jan])
    out["all"] = fit_beta(x[ok], lp[ok], a[ok])
    return out


def shift(setting: str, src_tag: str, corr_tag: str, T: pd.DataFrame, B: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """(원 캐시, 새 p_art(캐시 순서), Δ 예측(캐시 순서), 정보)."""
    c, Z = frame(setting, src_tag, corr_tag, T)
    if setting == "val":
        beta = pd.Series(Z["day"].to_numpy()).map(B["fold_of_day"]).map(B["fold"]).to_numpy(dtype=float)
    else:
        beta = np.full(len(Z), B["jan"] if setting == "fwd" else B["all"])
    p0 = Z["p"].to_numpy()
    on = p0 > 0
    pn = np.where(on, 1 / (1 + np.exp(-(Z["lp"].to_numpy() + beta * Z["x"].to_numpy()))), p0)
    p = np.clip(p0, 1e-4, 0.999)
    pcr = np.clip(Z["p_corr"].to_numpy(), 1e-4, 0.999)
    d = Z["d"].to_numpy()
    mn = (Z["f_corr"].to_numpy() - pcr * d) / (1 - pcr)
    dz = np.where(on, (np.clip(pn, 1e-4, 0.999) - p) * (d - mn), 0.0)
    p_new = c["p_art"].to_numpy(dtype=float).copy()
    delta = np.zeros(len(c))
    pos = Z["pos"].to_numpy()
    p_new[pos] = pn
    delta[pos] = dz
    info = {"rows_lirf_nong": int(len(Z)), "rows_moved": int(on.sum()), "beta": (sorted(set(np.round(beta, 6).tolist())) if setting == "val" else float(beta[0]) if len(beta) else None),
            "mean_p_old": round(float(p0.mean()), 5), "mean_p_new": round(float(pn.mean()), 5)}
    return c, p_new, delta, info


def d_lobt_of(ids: np.ndarray) -> np.ndarray:
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "cache" / "feats_v25.parquet", columns=["MVT_ID_mvt", "d_lobt"])
    s = pd.Series(np.asarray(t.column("d_lobt").to_numpy(zero_copy_only=False), dtype=float), index=t.column("MVT_ID_mvt").to_numpy())
    return s.reindex(ids).to_numpy()


def run_setting(setting: str, sm: dict, corr_tag: str, T: pd.DataFrame, B: dict):
    """(새 후보 표, 새 캐시 표, 원 예측, 새 예측, 정보)."""
    src_tag = sm.get("run_tag", sm["spec"])
    cache, p_new, delta, info = shift(setting, src_tag, corr_tag, T, B)
    ids = cache["MVT_ID_mvt"].to_numpy()
    cand = pd.read_parquet(ROOT / sm[KEYPART[setting]]["path"])
    old = cand.set_index("MVT_ID_mvt")["pred"].astype(float)
    if set(old.index) != set(ids):
        raise SystemExit(f"{setting}: 후보 파일과 원 캐시의 행이 다르다")
    ch = delta != 0
    o = old.reindex(ids).to_numpy()
    nv = o + delta
    dl = d_lobt_of(ids[ch])
    v = nv[ch]
    v = np.where(np.isfinite(dl), np.clip(v, dl - HALF, dl + HALF), v)
    nv[ch] = np.clip(v, 0, None)
    new = old.copy()
    new.loc[ids[ch]] = nv[ch]
    untouched = ~new.index.isin(ids[ch])
    if not (new.to_numpy() == old.to_numpy())[untouched].all():
        raise SystemExit(f"{setting}: 손대지 않을 행이 바뀌었다")
    if np.any(ch & cache["g"].to_numpy().astype(bool)):
        raise SystemExit(f"{setting}: G 행이 바뀌었다")
    out = cand.copy()
    out["pred"] = new.reindex(cand["MVT_ID_mvt"].to_numpy()).to_numpy()
    cache_new = cache.copy()
    cache_new["p_art"] = p_new
    info.update({"changed": int((new.to_numpy() != old.to_numpy()).sum()), "g_changed": 0, "move_mse": round(float(((new - old) ** 2).sum()), 1)})
    return out, cache_new, old, new, info


def source(arg: str) -> tuple[dict, str, str | None, str]:
    from src.batches import manifest_of
    if arg.endswith(".json"):
        meta, stem = {"manifest": arg}, None
    else:
        q = ROOT / "queue" / f"{arg}.json"
        meta, stem = json.loads((q if q.exists() else ROOT / "queue" / "done" / f"{arg}.json").read_text(encoding="utf-8")), arg
    sm, why = manifest_of(meta)
    if why:
        raise SystemExit(f"거부: 원 후보 manifest 검사 {why}")
    cfg = sm.get("config", {})
    tag = sm.get("run_tag", sm["spec"])
    if "--leg" in cfg or "--variant" in cfg or "--base" not in cfg or sm["spec"].endswith(SUFFIX):
        raise SystemExit("거부: 원 후보는 A3 보정기 출력이나 그 위 artmm 판(--base 가 있고 gkey 류, AIU, 반대 창, 이 단계가 아닌 판)이어야 한다")
    corr = cfg.get("--corr", tag)
    for s in ("val", "fwd", "full"):
        for t_ in (tag, corr):
            if not (ROOT / "cache" / f"{s}_{t_}_s0.parquet").exists():
                raise SystemExit(f"거부: 캐시 cache/{s}_{t_}_s0.parquet 가 없다")
    return sm, meta["manifest"], stem, corr


def truth() -> pd.Series:
    import pyarrow.parquet as pq
    T = pq.read_table(ROOT / "cache" / "all.parquet", columns=["MVT_ID_mvt", "y"], filters=[("PHASE_mvt", "==", "DEP"), ("src", ">", 0)]).to_pandas()
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


def beta_rep(B: dict) -> dict:
    return {"fold": {str(k): round(v, 6) for k, v in B["fold"].items()}, "jan": round(B["jan"], 6), "all": round(B["all"], 6)}


def check(arg: str) -> int:
    sm, src_mp, _, corr = source(arg)
    T = build_table()
    y = truth()
    B = betas(sm.get("run_tag", sm["spec"]), corr, T)
    print(f"[check] 원 {src_mp}, 원 캐시 {sm.get('run_tag', sm['spec'])}, 보정기 {corr}, 표 {TABLE} {sha16(f'cache/{TABLE}.parquet')}, β {beta_rep(B)}", flush=True)
    for setting in ("full", "val", "fwd"):
        _, _, old, new, info = run_setting(setting, sm, corr, T, B)
        rep = dict(info)
        if setting == "full":
            rep["move_mse_test_n"] = round(info["move_mse"] / N_TEST, 1)
        else:
            rep.update(delta_rep(old, new, y))
        rep.pop("beta", None) if setting == "val" else None
        print(f"[check] {setting}: {rep}", flush=True)
    return 0


def build(arg: str, num: str) -> int:
    from src.ruler import ruler_hash
    sm, src_mp, stem, corr = source(arg)
    tag = sm["spec"] + SUFFIX
    name = f"{num}_{tag}"
    if (ROOT / "queue" / f"{name}.json").exists() or (ROOT / f"preds/manifest_{tag}.json").exists():
        print("거부: 같은 이름이 이미 있다", name)
        return 1
    T = build_table()
    y = truth()
    B = betas(sm.get("run_tag", sm["spec"]), corr, T)
    rep, files = {"beta": beta_rep(B)}, {}
    for setting in ("full", "val", "fwd"):
        out, cache_new, old, new, info = run_setting(setting, sm, corr, T, B)
        if setting == "val":
            rm = float(np.sqrt(((new - y.reindex(new.index)) ** 2).mean()))
            rel = f"preds/holdout_{tag}_rmse{rm:.2f}.parquet"
        elif setting == "fwd":
            rel = f"preds/fwd_{tag}.parquet"
        else:
            rel = f"preds/test_{tag}.parquet"
        r = {k: v for k, v in info.items() if k not in ("move_mse", "beta")}
        if setting == "full":
            r["move_mse_test_n"] = round(info["move_mse"] / N_TEST, 1)
        else:
            r.update(delta_rep(old, new, y))
        rep[setting] = r
        out.to_parquet(ROOT / rel, index=False)
        cache_new.to_parquet(ROOT / "cache" / f"{setting}_{tag}_s0.parquet", index=False)
        files[setting] = rel
        print(f"[{name}] {setting}: {r}", flush=True)
    es_rel = f"preds/es_valid_ids_{tag}.parquet"
    shutil.copy2(ROOT / sm["es_valid_ids"]["path"], ROOT / es_rel)
    me = "tools/arthist.py"
    h = sha16(me)
    snap = f"preds/code/{h}_arthist.py"
    if not (ROOT / snap).exists():
        shutil.copy2(ROOT / me, ROOT / snap)
    c0 = sm["config"]
    cfg = {"--base": c0["--base"], "--base_seed": c0.get("--base_seed", "0"), "--src": src_mp, "--src_sha": sha16(src_mp), "--corr": corr,
           "--table": TABLE, "--table_sha": sha16(f"cache/{TABLE}.parquet"), "--tag": tag}
    flags = " ".join(f"{k} {v}" for k, v in cfg.items())
    cmd = lambda s: f".venv/Scripts/python.exe -X utf8 tools/arthist.py --setting {s} {flags}"  # noqa: E731
    hold_ids = pd.read_parquet(ROOT / files["val"])["MVT_ID_mvt"].to_numpy()
    caveats = ["the hierarchical prior (flight -> carrier-route -> carrier, P1h) was chosen over the carrier-only prior (P1) after both were measured; "
               "both won val (day folds), January<->July, July forward and Nov-Dec",
               "the 12-month leave-one-month-out is a label-only proxy against a carrier-level baseline (row-weighted -1.15%, d_sch^2-weighted -2.88%, "
               "12/12 months); the main p_art exists only for the ruler months, so the shift itself was measured on val, fwd and nd",
               "beta is fitted on the val ruler rows (day folds for val, January for fwd, both months for test), the way the A3 corrector's stage 2 is fitted",
               "the history uses 2025 departure labels of the training months; 2026 flights with little 2025 history fall back to the carrier-route and "
               "carrier shares",
               "after seeing that about 40% of the test move sits on a dozen rare-carrier rows, a variant that also shrinks the carrier share toward the LIRF "
               "share was measured and lost all four rulers (val -33, Jan<->Jul -29, fwd +6, Nov-Dec -36 against -45, -40, -21, -51); kept as built",
               "by month the gain is July (val -42 to -45, fwd -21) and Nov-Dec (-21, -30); January is flat (+2.0 to +2.3); within val July one carrier-month "
               "(WMT) carries about two thirds"]
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
        "iteration_rule": sm.get("iteration_rule", "") + " | arthist: LIRF non-G p_art moved in log-odds by beta x (flight-history art share vs the flight-month "
                          "mean p_art); one coefficient beta fitted on the val ruler rows (day folds for val, January for fwd, both months for test)",
        "source": {"queue": stem, "manifest": src_mp, "manifest_sha": sha16(src_mp), "spec": sm["spec"],
                   "holdout_sha": sm["holdout"]["sha"], "test_sha": sm["test"]["sha"], "forward_sha": sm["forward"]["sha"]},
        "arthist": {"source_cache": sm.get("run_tag", sm["spec"]), "corrector": corr, "table": f"cache/{TABLE}.parquet", "table_sha": cfg["--table_sha"],
                    "shrink": {"flight": K_FC, "route": K_RT}, "folds": FOLDS, "fold_seed": FOLD_SEED,
                    "inputs_sha16": {f"cache/{s}_{t_}_s0.parquet": sha16(f"cache/{s}_{t_}_s0.parquet") for s in ("val", "fwd", "full")
                                     for t_ in dict.fromkeys((sm.get("run_tag", sm["spec"]), corr))},
                    "caches": {s: f"cache/{s}_{tag}_s0.parquet" for s in ("val", "fwd", "full")},
                    "model": "LIRF non-G rows with p_art > 0: p' = sigmoid(logit p + beta x), x = logit(h) - logit(mean p of the flight in that month), "
                             "h = flight art share in the training months shrunk 20 rows toward the carrier-route share (itself shrunk 50 rows toward the carrier "
                             "share); prediction moved by (p' - p)(d - mn) with mn from the corrector's pred and p_art; LOBT window and 0 floor on changed rows; "
                             "G rows and p_art = 0 rows untouched; the cache files carry p' for the AIU day shift; gkey and later steps go after this",
                    "caveats": caveats, "report": rep},
        "written_kst": now_kst(),
    }
    mp = f"preds/manifest_{tag}.json"
    (ROOT / mp).write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{name}] manifest {mp}", flush=True)
    method = (f"{stem or src_mp} 의 LIRF G 아닌 출발에서 주 분류기 p_art 를 같은 편명의 2025 학습 달 art 몫(편명 20행 -> 같은 운항사 같은 노선 50행 -> 운항사 "
              f"몫으로 수축) 쪽으로 로짓 계수 하나 β 만큼 옮긴 판(tools/arthist.py, 표 {TABLE} {cfg['--table_sha']}). β 는 자 행에서 맞췄다(val 은 날 열 겹, "
              f"fwd 는 1월, 시험은 두 달: {rep['beta']['all']}). 예측은 (p' − p)(d − mn) 만큼, LOBT 창 안으로. G 행과 p_art 0 행은 그대로. "
              "근거 vault research/2026-10-07-r7-u.md. gkey 와 그 뒤 단계는 이 뒤에 둔다. "
              f"자 ΔMSE {rep['val']['dmse']}({rep['val']['improved']}/{rep['val']['changed']}), 7월 전방 ΔMSE {rep['fwd']['dmse']}"
              f"({rep['fwd']['improved']}/{rep['fwd']['changed']}), 시험 바뀐 행 {rep['full']['changed']}, 이동 MSE {rep['full']['move_mse_test_n']}. "
              "약점: 계층 수축 판은 운항사 수축 판과 둘 다 잰 뒤 골랐고(둘 다 네 자에서 이김), 열두 달 한 달 빼기는 정답만 쓴 대리다. "
              "이득은 7월과 11, 12월이고 1월은 0 안팎이다(+2). 시험 이동의 약 40% 가 드문 운항사 십여 행이다(운항사 몫까지 수축한 판은 네 자 모두 졌다). "
              "10-05 단일 발 체제라 준비 점검은 경고(--readiness-warn)")
    (ROOT / "notes" / f"arthist_{name}.json").write_text(json.dumps({"name": name, "manifest": mp, "source": stem or src_mp, "method": method, "report": rep,
                                                                     "written_kst": now_kst()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def enqueue(name: str) -> int:
    info = json.loads((ROOT / "notes" / f"arthist_{name}.json").read_text(encoding="utf-8"))
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
            print("usage: arthist.py build <artmm 판 큐 이름 또는 manifest> <새 큐 번호> [--no-enqueue]")
            return 2
        me = [sys.executable, "-X", "utf8", str(Path(__file__).resolve())]
        rc = subprocess.run(me + ["_files", a[1], a[2]], cwd=ROOT).returncode
        if rc != 0 or "--no-enqueue" in a:
            return rc
        sm, _, _, _ = source(a[1])
        return subprocess.run(me + ["enqueue", f"{a[2]}_{sm['spec']}{SUFFIX}"], cwd=ROOT).returncode
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--seed", default="0")
    for k in ("--base", "--base_seed", "--src", "--src_sha", "--corr", "--table", "--table_sha", "--tag"):
        pa.add_argument(k, required=True)
    g = pa.parse_args(a)
    if sha16(g.src) != g.src_sha or sha16(f"cache/{g.table}.parquet") != g.table_sha:
        print("거부: 원 manifest 나 표 해시가 다르다")
        return 1
    sm = json.loads((ROOT / g.src).read_text(encoding="utf-8"))
    if sm.get("config", {}).get("--corr", sm.get("run_tag", sm["spec"])) != g.corr:
        print("거부: 보정기 태그가 원 manifest 와 다르다")
        return 1
    T = pd.read_parquet(ROOT / "cache" / f"{g.table}.parquet")
    B = betas(sm.get("run_tag", sm["spec"]), g.corr, T)
    out, cache_new, _, _, info = run_setting(g.setting, sm, g.corr, T, B)
    rel = sorted((ROOT / "preds").glob(f"holdout_{g.tag}_rmse*.parquet"))[0].relative_to(ROOT) if g.setting == "val" else \
        Path(f"preds/{'fwd' if g.setting == 'fwd' else 'test'}_{g.tag}.parquet")
    out.to_parquet(ROOT / rel, index=False)
    cache_new.to_parquet(ROOT / "cache" / f"{g.setting}_{g.tag}_s0.parquet", index=False)
    print(rel, {k: v for k, v in info.items() if k != "beta"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
