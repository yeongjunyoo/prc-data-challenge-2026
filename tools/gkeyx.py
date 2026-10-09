# SPDX-License-Identifier: GPL-3.0-only
"""gkey avg 판(tools/gkey.py --variant avg)의 G 행(LIRF 출발 가운데 NM 미매칭)에 기록 감식 칸 열둘을 더한 G 모형을 얹는다.
순서: 원 후보 → gkey(--variant avg) → 이 단계 → (AIU 날 이동) → 반대 창(맨 끝).
근거: vault `01 Projects/공모전 파이프라인/착수/PRC Data Challenge 2026/research/2026-10-06-r5-records.md`(연구 스크립트
research/r5_records/a7_g_month.py, a14_g_guard.py, b1_gkx.py). 생각의 출처: 우리 기록 감식(G 행의 복사 여부는 그 편명이 그 달 얼마나 자주
뜨는지와 LOBT 를 지연에 맞춰 고쳤는지를 따른다). 남의 코드 없음.

칸(입력만, LIRF 출발 전부로 센다. 키는 정규화 편명 x 달 mon(2025 은 src 1~12, 2026 은 100 + 이륙 UTC 달), 자기 제외):
  fm_n 같은 편명 다른 출발 수, fm_nm 그 가운데 NM 매칭 수, fm_gshare 그 가운데 G 몫, fm_upd 매칭 가운데 LOBT − SCHED > 900 몫,
  fm_unupd 매칭 지연(MVT − SCHED ≥ 1,800) 가운데 AOBT_3 − LOBT > 900 몫, fm_lsmed 매칭 LOBT − SCHED 중앙, fm_dsch MVT − SCHED 중앙,
  ffmt_empty, ffmt_icao_alnum, fl_suffix_alpha 편명 꼴, a_unupd 같은 목적지 x 달의 고치지 않은 지연 몫(최소 3), al_unupd 같은 운항사 세 글자 x 달(최소 5).
  정규화 편명: 앞 세 글자 뒤에 겹쳐 적힌 글자를 지운다(WMTMT306 → WMT306, AEZZ2994 → AEZ2994).
새 G: gkey avg 와 같은 식(model3.g_probs 꼴 확률을 목적지 키와 운항사 키로 두 번 내어 반씩 평균, 띠 규칙과 mu 는 G 다리의 것)에서 GBM 쪽에 칸 열둘을
  더한 확률을 MVT − SCHED < 43,200 인 G 행에만 쓰고, 나머지 G 행은 gkey avg 확률 그대로 둔다. 연구에서 정한 그대로이고 자로 고른 값은 없다.
덮기: 원 후보(gkey avg 판)의 G 행에 (새 G − gkey avg G)를 더하고 0 아래는 0. G 아닌 행은 그대로(바이트까지 같은지 확인한다).
  gkey avg G 재현은 gkey 파일을 그 원본에서 1e-3 초 안에 다시 만드는지 먼저 확인하고, 어긋나면 멈춘다.
AIU 날 이동(src/aiu_shift.py)은 이 단계의 manifest 를 보고 classes() 로 G 행의 정상 확률과 mu 를 받는다.

쓰는 법(저장소에서, memcap 안):
    .venv/Scripts/python.exe -X utf8 tools/gkeyx.py table [--force]          표 cache/gkeyx_v1.parquet(+json)
    .venv/Scripts/python.exe -X utf8 tools/gkeyx.py check <gkey avg 판 manifest>  세 설정을 계산해 ΔMSE 와 이동만 찍는다(아무 파일도 쓰지 않는다)
    .venv/Scripts/python.exe -X utf8 tools/gkeyx.py build <gkey avg 판 manifest 또는 큐 이름> <새 큐 번호> [--no-enqueue]
        preds/ 에 새 파일 셋과 manifest 를 쓰고(새 spec 은 원 spec + "x"), notes/gkeyx_<새 큐 이름>.json 을 남긴다. --no-enqueue 가 없으면 큐에 넣는다.
    .venv/Scripts/python.exe -X utf8 tools/gkeyx.py enqueue <새 큐 이름>
    .venv/Scripts/python.exe -X utf8 tools/gkeyx.py --setting {val,fwd,full} --src <원 manifest> --src_sha <16> --leg <태그> --leg_feats <판>
        --table <gkey 표> --table_sha <16> --xtable gkeyx_v1 --xtable_sha <16> --variant avg_x --tag <새 spec>   (manifest 의 재현 명령)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

XTABLE = "gkeyx_v1"
XCOLS = ["fm_n", "fm_nm", "fm_gshare", "fm_upd", "fm_unupd", "fm_lsmed", "fm_dsch", "ffmt_empty", "ffmt_icao_alnum", "fl_suffix_alpha", "a_unupd", "al_unupd"]
CUT = 43200.0
VARIANT = "avg_x"
SUFFIX = "x"
N_TEST = 344841
KEYPART = {"val": "holdout", "fwd": "forward", "full": "test"}


def gkey():
    spec = importlib.util.spec_from_file_location("gkey", ROOT / "tools" / "gkey.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha16(rel: str | Path) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:16]


def now_kst() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")


# ---- 표 ----

def canon(s):
    if not isinstance(s, str) or len(s) < 5:
        return s if isinstance(s, str) else None
    if s[3:5] == s[1:3]:
        return s[:3] + s[5:]
    if s[3] == s[2]:
        return s[:3] + s[4:]
    if s[3:6] == s[0:3]:
        return s[:3] + s[6:]
    return s


PAT = [("empty", re.compile(r"^$")), ("icao_num", re.compile(r"^[A-Z]{3}\d{1,4}$")), ("icao_alnum", re.compile(r"^[A-Z]{3}\d{1,3}[A-Z]{1,2}$")),
       ("iata_num", re.compile(r"^[A-Z0-9]{2}\d{1,4}[A-Z]?$"))]


def ffmt(s: str) -> str:
    if s and len(s) >= 5 and (s[3] == s[2] or s[3:5] == s[1:3] or s[3:6] == s[0:3]):
        return "garb"
    for n, p in PAT:
        if p.match(s):
            return n
    return "other"


def build_table(force: bool = False) -> pd.DataFrame:
    out = ROOT / "cache" / f"{XTABLE}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    t = ds.dataset(ROOT / "cache" / "all.parquet").to_table(
        columns=["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADES_mvt", "src", "t_mvt", "t_sch", "t_lobt", "t_aobt3"],
        filter=(pc.field("PHASE_mvt") == "DEP") & (pc.field("ADEP_mvt") == "LIRF"))
    D = t.to_pandas()
    del t
    src = D["src"].to_numpy()
    mth = pd.to_datetime(D["t_mvt"], unit="s").dt.month.to_numpy()
    D["mon"] = np.where(src == 0, 100 + mth, src).astype(np.int16)
    m = D["FLIGHT_ID_mvt"].notna().to_numpy()
    sch, lob, aob = (D[c].to_numpy(dtype=float) for c in ("t_sch", "t_lobt", "t_aobt3"))
    D["d_sch"] = D["t_mvt"].to_numpy(dtype=float) - sch
    fl_obj = D["FLIGHT_mvt"].astype(object)
    u = pd.unique(fl_obj.dropna())
    D["fc"] = fl_obj.map({x: canon(x) for x in u})
    D["_m"] = m.astype(float)
    D["_upd"] = (m & ((lob - sch) > 900)).astype(float)
    dl = m & (D["d_sch"].to_numpy() >= 1800)
    D["_dl"] = dl.astype(float)
    D["_dlun"] = (dl & ((aob - lob) > 900)).astype(float)
    D["_ls"] = np.where(m, lob - sch, np.nan)
    D["_g"] = (~m).astype(float)
    k = D["fc"].astype(object).where(D["fc"].notna(), "__none__").astype(str) + "|" + D["mon"].astype(str)
    gr = D.groupby(k)
    n = gr["_m"].transform("size") - 1
    nm = gr["_m"].transform("sum") - D["_m"]
    D["fm_n"] = n
    D["fm_nm"] = nm
    D["fm_gshare"] = (gr["_g"].transform("sum") - D["_g"]) / n.where(n >= 2)
    D["fm_upd"] = (gr["_upd"].transform("sum") - D["_upd"]) / nm.where(nm >= 2)
    dn = gr["_dl"].transform("sum") - D["_dl"]
    D["fm_unupd"] = (gr["_dlun"].transform("sum") - D["_dlun"]) / dn.where(dn >= 2)
    D["fm_lsmed"] = gr["_ls"].transform("median")
    D["fm_dsch"] = gr["d_sch"].transform("median")
    none = D["fc"].isna()
    D.loc[none, ["fm_n", "fm_nm", "fm_gshare", "fm_upd", "fm_unupd", "fm_lsmed", "fm_dsch"]] = np.nan
    fl = D["FLIGHT_mvt"].astype(str).where(D["FLIGHT_mvt"].notna(), "")
    uf = pd.unique(fl)
    fmt = fl.map({x: ffmt(x) for x in uf})
    D["ffmt_empty"] = (fmt == "empty").astype(float)
    D["ffmt_icao_alnum"] = (fmt == "icao_alnum").astype(float)
    D["fl_suffix_alpha"] = fl.str.match(r"^.*\d[A-Z]+$").astype(int).astype(float)
    ga = D.groupby(["ADES_mvt", "mon"], observed=True)
    den = ga["_dl"].transform("sum") - D["_dl"]
    D["a_unupd"] = (ga["_dlun"].transform("sum") - D["_dlun"]) / den.where(den >= 3)
    D["al"] = D["fc"].where(D["fc"].astype(str).str.match(r"^[A-Z]{3}"), None).astype(str).str[:3]
    gl = D.groupby(["al", "mon"])
    den = gl["_dl"].transform("sum") - D["_dl"]
    D["al_unupd"] = (gl["_dlun"].transform("sum") - D["_dlun"]) / den.where(den >= 5)
    D.loc[D["al"] == "Non", "al_unupd"] = np.nan
    G = D.loc[~m, ["MVT_ID_mvt", "src", "mon"] + XCOLS].reset_index(drop=True)
    G.to_parquet(out, index=False)
    s = G["src"].to_numpy()
    info = {"rows": int(len(G)), "train_rows": int((s > 0).sum()), "test_rows": int((s == 0).sum()),
            "fill": {c: [round(float(G.loc[s > 0, c].notna().mean()), 4), round(float(G.loc[s == 0, c].notna().mean()), 4)] for c in XCOLS},
            "mean_2025_1_7_vs_test": {c: [round(float(G.loc[np.isin(s, [1, 7]), c].mean()), 4), round(float(G.loc[s == 0, c].mean()), 4)] for c in XCOLS},
            "sha16": sha16(out.relative_to(ROOT)), "source": "cache/all.parquet", "source_sha16": sha16("cache/all.parquet"),
            "built_kst": now_kst(), "code": "tools/gkeyx.py table", "code_sha16": sha16("tools/gkeyx.py")}
    out.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print("표", out.name, json.dumps({k: info[k] for k in ("rows", "train_rows", "test_rows", "sha16")}), flush=True)
    return G


# ---- G 모형 ----

def with_x(H: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    H = H.copy()
    ids = H["MVT_ID_mvt"].to_numpy()
    miss = int((~pd.Index(ids).isin(X.index)).sum())
    if miss:
        raise SystemExit(f"표 {XTABLE} 에 없는 G 행이 {miss} 개다(tools/gkeyx.py table --force)")
    for c in XCOLS:
        H[c] = X[c].reindex(ids).to_numpy(dtype=float)
    return H


def gx_values(setting: str, cfg: dict, T: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """cfg: gkey avg 판 manifest 의 config(--leg, --leg_feats). 평가 G 행마다 옛 G(다리 재현), gkey avg G, 새 G, 새 정상 확률, mu."""
    gk = gkey()
    from src.model3 import g_override, g_probs
    G = gk.g_frame(cfg["--leg_feats"])
    s = G["src"].to_numpy()
    Gf = G[gk.FIT[setting](s) & ~np.isnan(G["y"].to_numpy(dtype=float))]
    Ge = G[gk.EVAL[setting](s)]
    b = pd.read_parquet(ROOT / "cache" / f"{setting}_{cfg['--leg']}_s0.parquet", columns=["MVT_ID_mvt", "pred", "mu", "p_art", "g"]).set_index("MVT_ID_mvt")
    gid = b.index[b["g"].to_numpy().astype(bool)]
    if set(gid) != set(Ge["MVT_ID_mvt"]):
        raise SystemExit(f"{setting}: 다리 {cfg['--leg']} 의 G 행과 피처 판의 G 행이 다르다({len(gid)} 대 {len(Ge)})")
    bb = b.reindex(Ge["MVT_ID_mvt"].to_numpy())
    mu_g, pa_g = bb["mu"].to_numpy(dtype=float), bb["p_art"].to_numpy(dtype=float)
    d, dd = Ge["d_sch"].to_numpy(dtype=float), Ge["dd"].to_numpy()
    g_old = np.clip(g_override(g_probs(Gf, Ge), d, dd, mu_g, pa_g)[0], 0, None)
    err = float(np.max(np.abs(g_old - bb["pred"].to_numpy(dtype=float)))) if len(Ge) else 0.0
    if err >= 1e-3:
        raise SystemExit(f"{setting}: 옛 G 재현이 다리 파일과 다르다(최대 차 {err})")
    Hf, He = gk.add_new(Gf, T), gk.add_new(Ge, T)
    Hf2, He2 = Hf.copy(), He.copy()
    Hf2["pfx"], He2["pfx"] = Hf2["k_car"], He2["k_car"]
    P_av = 0.5 * (gk.probs(Hf, He, []) + gk.probs(Hf2, He2, []))
    P_x = 0.5 * (gk.probs(with_x(Hf, X), with_x(He, X), XCOLS) + gk.probs(with_x(Hf2, X), with_x(He2, X), XCOLS))
    P = np.where((d < CUT)[:, None], P_x, P_av)
    g_av = np.clip(g_override(P_av, d, dd, mu_g, pa_g)[0], 0, None)
    g_new, pn_new = g_override(P, d, dd, mu_g, pa_g)
    return pd.DataFrame({"g_old": g_old, "g_av": g_av, "g_new": np.clip(g_new, 0, None), "pn_new": pn_new, "mu_g": mu_g, "d_sch": d},
                        index=Ge["MVT_ID_mvt"].to_numpy())


def load_pred(rel: str) -> pd.Series:
    return pd.read_parquet(ROOT / rel).set_index("MVT_ID_mvt")["pred"].astype(float)


def check_gkey(setting: str, gm: dict, V: pd.DataFrame) -> float:
    """gkey avg G 재현이 gkey 파일의 G 행을 그 원본에서 다시 만드는가(1e-3 초)."""
    src_m = json.loads((ROOT / gm["source"]["manifest"]).read_text(encoding="utf-8"))
    f_src = load_pred(src_m[KEYPART[setting]]["path"])
    f_gk = load_pred(gm[KEYPART[setting]]["path"])
    ids = V.index.to_numpy()
    want = np.clip(f_src.reindex(ids).to_numpy() + V["g_av"].to_numpy() - V["g_old"].to_numpy(), 0, None)
    err = float(np.max(np.abs(want - f_gk.reindex(ids).to_numpy()))) if len(ids) else 0.0
    if not err < 1e-3:
        raise SystemExit(f"{setting}: gkey avg G 재현이 gkey 파일과 다르다(최대 차 {err})")
    return err


def overlay(pred: pd.Series, V: pd.DataFrame) -> tuple[pd.Series, dict]:
    miss = int((~V.index.isin(pred.index)).sum())
    if miss:
        raise SystemExit(f"후보 예측에 없는 G 행이 {miss} 개다")
    new = pred.copy()
    old_g = pred.reindex(V.index).to_numpy(dtype=float)
    new.loc[V.index] = np.clip(old_g + (V["g_new"].to_numpy() - V["g_av"].to_numpy()), 0, None)
    ch = new.to_numpy() != pred.to_numpy()
    nong = ~pred.index.isin(V.index)
    if not np.array_equal(new.to_numpy()[nong], pred.to_numpy()[nong]):
        raise SystemExit("G 아닌 행이 바뀌었다")
    info = {"g_rows": int(len(V)), "changed": int(ch.sum()), "outside_g": int((ch & nong).sum()), "guarded_rows": int((V["d_sch"] >= CUT).sum()),
            "mean_g_av": round(float(V["g_av"].mean()), 1), "mean_g_new": round(float(V["g_new"].mean()), 1)}
    return new, info


def truth() -> pd.Series:
    return gkey().truth()


def delta(old: pd.Series, new: pd.Series, y: pd.Series) -> dict:
    return gkey().delta(old, new, y)


def classes(setting: str, gm: dict, f_src: pd.Series, f_gx: pd.Series) -> pd.DataFrame:
    """src/aiu_shift.py 가 부른다: 이 단계 manifest gm 의 G 모형에서 G 행의 정상 확률(pn)과 mu_g. 먼저 이 단계 파일(f_gx)을 그 원본(f_src)에서
    1e-3 초 안에 다시 만드는지 확인한다."""
    cfg = gm["config"]
    for k, tb in (("--table_sha", cfg["--table"]), ("--xtable_sha", cfg["--xtable"])):
        if sha16(f"cache/{tb}.parquet") != cfg[k]:
            raise SystemExit(f"cache/{tb}.parquet sha16 differs from the gkeyx manifest's {cfg[k]}")
    T = pd.read_parquet(ROOT / "cache" / f"{cfg['--table']}.parquet")
    X = pd.read_parquet(ROOT / "cache" / f"{cfg['--xtable']}.parquet").set_index("MVT_ID_mvt")
    V = gx_values(setting, cfg, T, X)
    ids = V.index.to_numpy()
    want = np.clip(f_src.reindex(ids).to_numpy(dtype=float) + V["g_new"].to_numpy() - V["g_av"].to_numpy(), 0, None)
    err = float(np.max(np.abs(want - f_gx.reindex(ids).to_numpy(dtype=float)))) if len(ids) else 0.0
    print(f"  {setting}: gkeyx G model recomputed for {len(ids)} G rows; against the gkeyx file max diff {err:.2e} s", flush=True)
    if not err < 1e-3:
        raise SystemExit(f"{setting}: the recomputed gkeyx G model does not reproduce the gkeyx file (max diff {err})")
    return pd.DataFrame({"pn": V["pn_new"].to_numpy(), "mu_g": V["mu_g"].to_numpy()}, index=ids)


def source(arg: str, strict: bool = True) -> tuple[dict, str, str | None]:
    """원 후보: manifest 경로(.json) 또는 큐 이름. (manifest, manifest 경로, 큐 이름). strict 가 아니면(check) 코드 사본 차이는 경고만 하고,
    옛 G 와 gkey avg G 의 재현 확인(1e-3 초)이 지킨다."""
    from src.batches import manifest_of
    if arg.endswith(".json"):
        meta, stem = {"manifest": arg}, None
    else:
        meta, stem = json.loads((ROOT / "queue" / f"{arg}.json").read_text(encoding="utf-8")), arg
    sm, why = manifest_of(meta)
    if why:
        raise SystemExit(f"거부: 원 후보 manifest 검사 {why}")
    cfg = sm.get("config", {})
    if cfg.get("--variant") != "avg" or "--leg" not in cfg:
        raise SystemExit("거부: 원 후보가 tools/gkey.py --variant avg 판이 아니다")
    if sm["spec"].endswith("aw"):
        raise SystemExit("거부: 원 후보가 반대 창 판이다")
    for f in ("src/model3.py", "src/sub_g.py", "tools/gkey.py"):
        if f in sm["code_sha"] and sha16(f) != sm["code_sha"][f]:
            if strict:
                raise SystemExit(f"거부: {f} 가 원 후보 manifest 의 사본과 다르다")
            print(f"경고: {f} 가 원 후보 manifest 의 사본과 다르다(check 라 계속, 재현 확인이 지킨다)", flush=True)
    return sm, meta["manifest"], stem


def run_setting(setting: str, sm: dict, T: pd.DataFrame, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    V = gx_values(setting, sm["config"], T, X)
    err = check_gkey(setting, sm, V)
    old_df = pd.read_parquet(ROOT / sm[KEYPART[setting]]["path"])
    old = old_df.set_index("MVT_ID_mvt")["pred"].astype(float)
    new, info = overlay(old, V)
    info["gkey_repro_max_diff"] = float(f"{err:.3e}")
    out = old_df.copy()
    out["pred"] = new.reindex(old_df["MVT_ID_mvt"].to_numpy()).to_numpy()
    return out, old, new, info


def tables(sm: dict) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    cfg = sm["config"]
    if sha16(f"cache/{cfg['--table']}.parquet") != cfg["--table_sha"]:
        raise SystemExit("거부: gkey 표 해시가 원 후보 manifest 와 다르다")
    T = pd.read_parquet(ROOT / "cache" / f"{cfg['--table']}.parquet")
    build_table()
    xrel = f"cache/{XTABLE}.parquet"
    return T, pd.read_parquet(ROOT / xrel).set_index("MVT_ID_mvt"), xrel, sha16(xrel)


def check(arg: str) -> int:
    sm, src_mp, _ = source(arg, strict=False)
    T, X, xrel, xsha = tables(sm)
    y = truth()
    print(f"[check] 원 {src_mp}, 다리 {sm['config']['--leg']}, 표 {xrel} {xsha}", flush=True)
    for setting in ("full", "val", "fwd"):
        _, old, new, info = run_setting(setting, sm, T, X)
        if setting == "full":
            info["move_mse_test_n"] = round(float(((new - old) ** 2).sum() / N_TEST), 1)
        else:
            info.update(delta(old, new, y))
        print(f"[check] {setting}: {info}", flush=True)
    return 0


def build(arg: str, num: str) -> int:
    from src.ruler import ruler_hash
    sm, src_mp, stem = source(arg)
    tag = sm["spec"] + SUFFIX
    name = f"{num}_{tag}"
    if (ROOT / "queue" / f"{name}.json").exists() or (ROOT / f"preds/manifest_{tag}.json").exists():
        print("거부: 같은 이름이 이미 있다", name)
        return 1
    T, X, xrel, xsha = tables(sm)
    y = truth()
    rep, files = {}, {}
    for setting in ("full", "val", "fwd"):
        out, old, new, info = run_setting(setting, sm, T, X)
        if setting == "val":
            rm = float(np.sqrt(((new - y.reindex(new.index)) ** 2).mean()))
            rel = f"preds/holdout_{tag}_rmse{rm:.2f}.parquet"
        elif setting == "fwd":
            rel = f"preds/fwd_{tag}.parquet"
        else:
            rel = f"preds/test_{tag}.parquet"
        if setting == "full":
            rep[setting] = {**info, "move_mse_test_n": round(float(((new - old) ** 2).sum() / N_TEST), 1)}
            if info["outside_g"] or not (100 <= info["changed"] <= info["g_rows"]):
                print("거부: 시험에서 바뀐 행이 예상과 다르다(G 행 밖이거나 100 ~ G 행 수 밖)", info)
                return 1
        else:
            rep[setting] = {**info, **delta(old, new, y)}
        out.to_parquet(ROOT / rel, index=False)
        files[setting] = rel
        print(f"[{name}] {setting}: {rep[setting]}", flush=True)
    es_rel = f"preds/es_valid_ids_{tag}.parquet"
    shutil.copy2(ROOT / sm["es_valid_ids"]["path"], ROOT / es_rel)
    me = "tools/gkeyx.py"
    h = sha16(me)
    snap = f"preds/code/{h}_gkeyx.py"
    if not (ROOT / snap).exists():
        shutil.copy2(ROOT / me, ROOT / snap)
    c0 = sm["config"]
    cfg = {"--src": src_mp, "--src_sha": sha16(src_mp), "--leg": c0["--leg"], "--leg_feats": c0["--leg_feats"], "--table": c0["--table"],
           "--table_sha": c0["--table_sha"], "--xtable": XTABLE, "--xtable_sha": xsha, "--variant": VARIANT, "--tag": tag}
    flags = " ".join(f"{k} {v}" for k, v in cfg.items())
    cmd = lambda s: f".venv/Scripts/python.exe -X utf8 tools/gkeyx.py --setting {s} {flags}"  # noqa: E731
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
        "iteration_rule": sm.get("iteration_rule", "") + " | gkeyx: G model refit per setting on the g_band training rows; twelve record columns, "
                          "the 43,200 s d_sch guard and the 0.5 key average fixed in research, nothing chosen on the ruler",
        "source": {"queue": stem, "manifest": src_mp, "manifest_sha": sha16(src_mp), "spec": sm["spec"],
                   "holdout_sha": sm["holdout"]["sha"], "test_sha": sm["test"]["sha"], "forward_sha": sm["forward"]["sha"]},
        "gkeyx": {"leg": c0["--leg"], "leg_feats": c0["--leg_feats"], "table": f"cache/{c0['--table']}.parquet", "table_sha": c0["--table_sha"],
                  "xtable": xrel, "xtable_sha": xsha, "columns": XCOLS, "d_sch_guard": CUT, "variant": VARIANT,
                  "model": "gkey avg (ADES key and ICAO carrier key, mean) with the twelve record columns on the GBM side, used on G rows with "
                           "d_sch < 43,200 s; band rule and mu of the g_band leg; added to the gkey avg candidate as (new G - reproduced gkey avg G); "
                           "antiwin must stay after this",
                  "report": rep},
        "written_kst": now_kst(),
    }
    mp = f"preds/manifest_{tag}.json"
    (ROOT / mp).write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{name}] manifest {mp}", flush=True)
    method = (f"{stem or src_mp} 의 G 행(LIRF, NM 미매칭)에 기록 감식 칸 열둘을 더한 G 모형을 얹은 판(tools/gkeyx.py, 표 {xrel} sha {xsha}, "
              f"G 다리 {c0['--leg']}). gkey avg(목적지 키와 운항사 키 평균) 식의 GBM 쪽에 같은 편명의 그 달 기록 이력 일곱, 편명 꼴 셋, 목적지와 운항사의 "
              "그 달 고치지 않은 지연 몫 둘을 더하고 MVT − SCHED < 43,200 행에만 쓴다. 띠 규칙과 mu 는 다리의 것, 후보에는 (새 G − 재현한 gkey avg G)를 "
              "더했다. 학습 행은 설정마다 g_band 와 같고 자로 고른 값은 없다. 근거 vault research/2026-10-06-r5-records.md. 반대 창은 이 뒤에 둔다. "
              f"자 ΔMSE {rep['val']['dmse']}({rep['val']['improved']}/{rep['val']['changed']}), 7월 전방 ΔMSE {rep['fwd']['dmse']}"
              f"({rep['fwd']['improved']}/{rep['fwd']['changed']}), 시험 바뀐 행 {rep['full']['changed']}, 이동 MSE {rep['full']['move_mse_test_n']}, "
              "10-05 단일 발 체제라 준비 점검은 경고(--readiness-warn)")
    (ROOT / "notes" / f"gkeyx_{name}.json").write_text(json.dumps({"name": name, "manifest": mp, "source": stem or src_mp, "method": method,
                                                                   "report": rep, "written_kst": now_kst()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def enqueue(name: str) -> int:
    info = json.loads((ROOT / "notes" / f"gkeyx_{name}.json").read_text(encoding="utf-8"))
    r = subprocess.run([sys.executable, "-X", "utf8", "tools/enqueue.py", info["manifest"], name, info["method"], "--readiness-warn"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1500:], r.stderr[-800:], flush=True)
    return r.returncode


def main() -> int:
    gkey().below_normal()
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
            print("usage: gkeyx.py build <gkey avg 판 manifest 또는 큐 이름> <새 큐 번호> [--no-enqueue]")
            return 2
        me = [sys.executable, "-X", "utf8", str(Path(__file__).resolve())]
        rc = subprocess.run(me + ["_files", a[1], a[2]], cwd=ROOT).returncode  # 데이터 단계는 따로 끝내 메모리를 돌려준다
        if rc != 0 or "--no-enqueue" in a:
            return rc
        sm, _, _ = source(a[1])
        return subprocess.run(me + ["enqueue", f"{a[2]}_{sm['spec']}{SUFFIX}"], cwd=ROOT).returncode
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--seed", default="0")
    for k in ("--src", "--src_sha", "--leg", "--leg_feats", "--table", "--table_sha", "--xtable", "--xtable_sha", "--tag"):
        pa.add_argument(k, required=True)
    pa.add_argument("--variant", default=VARIANT, choices=[VARIANT])
    g = pa.parse_args(a)
    if sha16(g.src) != g.src_sha:
        print("거부: 원 manifest 해시가 다르다")
        return 1
    if sha16(f"cache/{g.table}.parquet") != g.table_sha or sha16(f"cache/{g.xtable}.parquet") != g.xtable_sha:
        print("거부: 표 해시가 다르다")
        return 1
    sm = json.loads((ROOT / g.src).read_text(encoding="utf-8"))
    T = pd.read_parquet(ROOT / "cache" / f"{g.table}.parquet")
    X = pd.read_parquet(ROOT / "cache" / f"{g.xtable}.parquet").set_index("MVT_ID_mvt")
    out, _, _, info = run_setting(g.setting, sm, T, X)
    rel = sorted((ROOT / "preds").glob(f"holdout_{g.tag}_rmse*.parquet"))[0].relative_to(ROOT) if g.setting == "val" else \
        Path(f"preds/{'fwd' if g.setting == 'fwd' else 'test'}_{g.tag}.parquet")
    out.to_parquet(ROOT / rel, index=False)
    print(rel, info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
