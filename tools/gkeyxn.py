# SPDX-License-Identifier: GPL-3.0-only
"""G model of tools/gkeyx.py with more GBM columns: the nine stand and link columns of tools/gkey.py and, in variant avg_xncr (the
final stack), the carrier-month share of arrivals recorded as the schedule and the same share for the carrier's route at LIRF.
Used for G rows with d_sch below 43,200 s; the other G rows keep the gkeyx value (docs/DESIGN.md section 8.4).

    python tools/gkeyxn.py table --version gkeyxn_v2      # cache/gkeyxn_v2.parquet
    python tools/gkeyxn.py --setting {val,fwd,full} --src <manifest> --src_sha <16> --leg <g_band tag> --leg_feats <feature table>
        --table gkey_v1 --table_sha <16> --xtable gkeyx_v1 --xtable_sha <16> --ytable gkeyxn_v2 --ytable_sha <16> --variant avg_xncr --tag <spec>

Original note (Korean):
gkeyx 판(tools/gkeyx.py, --variant avg_x)의 G 행(LIRF 출발 가운데 NM 미매칭)에 칸을 더한 G 모형을 얹는다. 두 판:
  avg_xnc(기본, spec 끝 "nc"): R3 스탠드, 이음 칸 아홉 + 운항사-달 도착 복사 몫(ca_copy_mon) 하나.
  avg_xn (spec 끝 "n")       : R3 스탠드, 이음 칸 아홉만(10-06 18:2x 에 처음 보낸 판, 재현과 AIU 를 위해 그대로 둔다).
  avg_xncr(spec 끝 "ncr")    : avg_xnc + 같은 운항사, 같은 노선의 그 달 LIRF 도착 복사 몫 rt_copy_mon(표 cache/gkeyxn_v2.parquet). 10-06 20:3x 에 더했다.
순서: 원 후보 → gkey(--variant avg) → gkeyx → 이 단계 → (AIU 날 이동) → 반대 창(맨 끝).
근거: vault `01 Projects/공모전 파이프라인/착수/PRC Data Challenge 2026/research/2026-10-06-r6-g.md`(연구 스크립트 research/r6_g/s9_variant.py,
s12_more_feats.py, s17_arr_dep_comove.py, r6e.py). 생각의 출처: R5 기록 감식 8.2 의 1(「gkey avg_new 위에 칸 열둘, 가드 43,200」), R3 스탠드 칸,
우리 기록 감식(운항사의 그 달 도착 기록이 SCHED 복사이면 출발 기록도 복사다: 2025 LIRF 운항사-달 안 상관 0.77). 남의 코드 없음.

새 G: gkeyx 와 같은 식(model3.g_probs 꼴 확률을 목적지 키와 운항사 키로 두 번 내어 반씩 평균, 띠 규칙과 mu 는 G 다리의 것)에서 GBM 쪽 칸을
  gkeyx 의 열둘(gkeyx.XCOLS)에 gkey 의 R3 칸 아홉(gkey.NEW: st_arr_between_sch_mvt, st_next_arr_gap, st_prev_arr_gap, st_prev2_arr_gap,
  st_prev_dep_gap, k_since_prevdep, arr_after_sch, link_nm, link_in_win)과 (avg_xnc 면) ca_copy_mon 을 더한 것으로 바꾼다.
  MVT − SCHED < 43,200 인 G 행에만 쓰고, 나머지 G 행은 gkeyx 의 값을 그대로 둔다(그 행의 차는 0). 연구에서 정한 그대로이고 자로 고른 값은 없다.
ca_copy_mon(표 cache/gkeyxn_v1.parquet, 입력만): 같은 운항사(정규화 편명 앞 ICAO 세 글자, gkeyx.canon)의 같은 달(2025 은 src, 2026 은 100 + UTC 달)
  LIRF 도착 가운데 |BLOCK − SCHED| ≤ 6.5초 몫, 도착 5행 이상일 때만. 도착 BLOCK 은 시험에도 있는 입력 칸이다.
덮기: 원 후보(gkeyx 판)의 G 행에 (새 G − 재현한 gkeyx G)를 더하고 0 아래는 0. G 아닌 행은 그대로(바이트까지 같은지 확인한다).
  먼저 gkeyx G 를 다시 계산해 gkeyx 파일을 그 원본(gkey avg 판)에서 1e-3 초 안에 다시 만드는지 확인하고, 어긋나면 멈춘다.
AIU 날 이동(src/aiu_shift.py)은 이 단계의 manifest(--variant avg_xn 또는 avg_xnc)를 보고 classes() 로 G 행의 정상 확률과 mu 를 받는다.

쓰는 법(저장소에서, memcap 2.5 GB 안):
    .venv/Scripts/python.exe -X utf8 tools/gkeyxn.py table [--force] [--version gkeyxn_v2]   표 cache/gkeyxn_v1(또는 v2).parquet(+json). 없으면 build, check 가 만든다
    .venv/Scripts/python.exe -X utf8 tools/gkeyxn.py check <gkeyx 판 manifest 또는 gkey avg 판 manifest> [--variant avg_xn]
        세 설정을 계산해 ΔMSE 와 이동만 찍는다(표 말고는 아무 파일도 쓰지 않는다). gkey avg 판을 주면 gkeyx 를 메모리에서 얹은 뒤 이 단계를 잰다.
    .venv/Scripts/python.exe -X utf8 tools/gkeyxn.py build <gkeyx 판 manifest 또는 큐 이름> <새 큐 번호> [--no-enqueue] [--variant avg_xn]
        preds/ 에 새 파일 셋과 manifest 를 쓰고(새 spec 은 원 spec + "nc", avg_xn 이면 + "n"), notes/gkeyxn_<새 큐 이름>.json 을 남긴다.
        --no-enqueue 가 없으면 큐에 넣는다.
    .venv/Scripts/python.exe -X utf8 tools/gkeyxn.py enqueue <새 큐 이름>
    .venv/Scripts/python.exe -X utf8 tools/gkeyxn.py --setting {val,fwd,full} --src <gkeyx manifest> --src_sha <16> --leg <태그> --leg_feats <판>
        --table <gkey 표> --table_sha <16> --xtable <gkeyx 표> --xtable_sha <16> [--ytable gkeyxn_v1 --ytable_sha <16>] --variant {avg_xnc,avg_xn}
        --tag <새 spec>   (manifest 의 재현 명령)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
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

VARIANTS = {"avg_xnc": {"suffix": "nc", "ycols": ["ca_copy_mon"], "ytable": "gkeyxn_v1"},
            "avg_xncr": {"suffix": "ncr", "ycols": ["ca_copy_mon", "rt_copy_mon"], "ytable": "gkeyxn_v2"},
            "avg_xn": {"suffix": "n", "ycols": [], "ytable": None}}
DEFAULT = "avg_xnc"
YTABLE = "gkeyxn_v1"  # avg_xnc 의 표(옛 이름 그대로). avg_xncr 는 gkeyxn_v2
N_TEST = 344841
KEYPART = {"val": "holdout", "fwd": "forward", "full": "test"}


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha16(rel: str | Path) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:16]


def now_kst() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")


def load_pred(rel: str) -> pd.Series:
    return pd.read_parquet(ROOT / rel).set_index("MVT_ID_mvt")["pred"].astype(float)


# ---- 표 ----

def build_table(force: bool = False, version: str = YTABLE) -> pd.DataFrame:
    """G 행(LIRF 출발, FLIGHT_ID 없음, 2025 와 2026 모든 달)마다 ca_copy_mon(gkeyxn_v1), 그리고 gkeyxn_v2 는 rt_copy_mon 도. 입력 칸만 쓴다.
    rt_copy_mon: 같은 운항사(정규화 편명 길이 4 이상, 앞 세 글자 영문 대문자), 같은 노선(도착의 ADEP = 이 출발의 ADES), 같은 달 LIRF 도착의 SCHED 복사 몫(3행 이상)."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    if version not in ("gkeyxn_v1", "gkeyxn_v2"):
        raise SystemExit(f"모르는 표 판 {version}")
    out = ROOT / "cache" / f"{version}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    gx = load_tool("gkeyx")
    t = ds.dataset(ROOT / "cache" / "all.parquet").to_table(
        columns=["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "PHASE_mvt", "src", "t_mvt", "t_blk", "t_sch"] + (["ADEP_mvt", "ADES_mvt"] if version == "gkeyxn_v2" else []),
        filter=((pc.field("PHASE_mvt") == "DEP") & (pc.field("ADEP_mvt") == "LIRF")) | ((pc.field("PHASE_mvt") == "ARR") & (pc.field("ADES_mvt") == "LIRF")))
    D = t.to_pandas()
    del t
    src = D["src"].to_numpy()
    mth = pd.to_datetime(D["t_mvt"], unit="s").dt.month.to_numpy()
    D["mon"] = np.where(src == 0, 100 + mth, src).astype(np.int16)
    fl = D["FLIGHT_mvt"].astype(object)
    cm = {x: gx.canon(x) for x in pd.unique(fl.dropna())}
    fc = fl.map(cm)
    D["car"] = [s[:3] if isinstance(s, str) and len(s) >= 3 and s[:3].isalpha() and s[:3].isupper() else None for s in fc]
    arr = (D["PHASE_mvt"].astype(str) == "ARR").to_numpy()
    A = D[arr & D["car"].notna().to_numpy() & np.isfinite(D["t_blk"].to_numpy(dtype=float))]
    cp = (np.abs(A["t_blk"].to_numpy(dtype=float) - A["t_sch"].to_numpy(dtype=float)) <= 6.5).astype(float)
    agg = pd.DataFrame({"car": A["car"].to_numpy(), "mon": A["mon"].to_numpy(), "cp": cp}).groupby(["car", "mon"])["cp"].agg(["sum", "count"])
    G = D[(~arr) & D["FLIGHT_ID_mvt"].isna().to_numpy()].reset_index(drop=True)
    k = pd.MultiIndex.from_arrays([G["car"], G["mon"]])
    s, c = agg["sum"].reindex(k).to_numpy(), agg["count"].reindex(k).to_numpy()
    G["ca_copy_mon"] = np.where(c >= 5, s / np.where(c > 0, c, 1), np.nan)
    G.loc[G["car"].isna(), "ca_copy_mon"] = np.nan
    ycols = ["ca_copy_mon"]
    if version == "gkeyxn_v2":
        car2 = [s_[:3] if isinstance(s_, str) and len(s_) >= 4 and s_[:3].isalpha() and s_[:3].isupper() else None for s_ in fc]
        D["car2"] = car2
        A2 = D[arr & D["car2"].notna().to_numpy() & np.isfinite(D["t_blk"].to_numpy(dtype=float))]
        cp2 = (np.abs(A2["t_blk"].to_numpy(dtype=float) - A2["t_sch"].to_numpy(dtype=float)) <= 6.5).astype(float)
        rt = pd.DataFrame({"car2": A2["car2"].to_numpy(), "org": A2["ADEP_mvt"].astype(str).to_numpy(), "mon": A2["mon"].to_numpy(), "cp": cp2}) \
            .groupby(["car2", "org", "mon"])["cp"].agg(["sum", "count"])
        Gd = D[(~arr) & D["FLIGHT_ID_mvt"].isna().to_numpy()].reset_index(drop=True)
        k2 = pd.MultiIndex.from_arrays([Gd["car2"], Gd["ADES_mvt"].astype(str), Gd["mon"]])
        s2, c2 = rt["sum"].reindex(k2).to_numpy(), rt["count"].reindex(k2).to_numpy()
        G["rt_copy_mon"] = np.where(c2 >= 3, s2 / np.where(c2 > 0, c2, 1), np.nan)
        ycols = ["ca_copy_mon", "rt_copy_mon"]
    G = G[["MVT_ID_mvt", "src", "mon", "car"] + ycols]
    G.to_parquet(out, index=False)
    s = G["src"].to_numpy()
    info = {"rows": int(len(G)), "train_rows": int((s > 0).sum()), "test_rows": int((s == 0).sum()),
            "fill": {c: [round(float(G.loc[s > 0, c].notna().mean()), 4), round(float(G.loc[s == 0, c].notna().mean()), 4)] for c in ycols}
            if version == "gkeyxn_v2" else [round(float(G.loc[s > 0, "ca_copy_mon"].notna().mean()), 4), round(float(G.loc[s == 0, "ca_copy_mon"].notna().mean()), 4)],
            "mean_2025_1_7_vs_test": {c: [round(float(G.loc[np.isin(s, [1, 7]), c].mean()), 4), round(float(G.loc[s == 0, c].mean()), 4)] for c in ycols}
            if version == "gkeyxn_v2" else [round(float(G.loc[np.isin(s, [1, 7]), "ca_copy_mon"].mean()), 4), round(float(G.loc[s == 0, "ca_copy_mon"].mean()), 4)],
            "arrival_rows_used": int(len(A)), "sha16": sha16(out.relative_to(ROOT)), "source": "cache/all.parquet", "source_sha16": sha16("cache/all.parquet"),
            "built_kst": now_kst(), "code": "tools/gkeyxn.py table", "code_sha16": sha16("tools/gkeyxn.py")}
    out.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print("표", out.name, json.dumps({k: info[k] for k in ("rows", "train_rows", "test_rows", "fill", "sha16")}), flush=True)
    return G


# ---- G 모형 ----

def with_y(H: pd.DataFrame, Y: pd.DataFrame | None, ycols: list[str]) -> pd.DataFrame:
    if not ycols:
        return H
    H = H.copy()
    ids = H["MVT_ID_mvt"].to_numpy()
    miss = int((~pd.Index(ids).isin(Y.index)).sum())
    if miss:
        raise SystemExit(f"이 단계 표에 없는 G 행이 {miss} 개다(tools/gkeyxn.py table --force)")
    for c in ycols:
        H[c] = Y[c].reindex(ids).to_numpy(dtype=float)
    return H


def xn_values(setting: str, cfg: dict, T: pd.DataFrame, X: pd.DataFrame, Y: pd.DataFrame | None = None, variant: str = "avg_xn") -> pd.DataFrame:
    """cfg: --leg, --leg_feats. 평가 G 행마다 gkeyx 의 값(gkeyx.gx_values: 옛 G 재현 확인 포함)과 이 단계의 새 G, 새 정상 확률."""
    gx = load_tool("gkeyx")
    gk = gx.gkey()
    from src.model3 import g_override
    ycols = VARIANTS[variant]["ycols"]
    V = gx.gx_values(setting, cfg, T, X)
    G = gk.g_frame(cfg["--leg_feats"])
    s = G["src"].to_numpy()
    Gf = G[gk.FIT[setting](s) & ~np.isnan(G["y"].to_numpy(dtype=float))]
    Ge = G[gk.EVAL[setting](s)]
    if not np.array_equal(Ge["MVT_ID_mvt"].to_numpy(), V.index.to_numpy()):
        raise SystemExit(f"{setting}: G 행 순서가 gkeyx 값과 다르다")
    b = pd.read_parquet(ROOT / "cache" / f"{setting}_{cfg['--leg']}_s0.parquet", columns=["MVT_ID_mvt", "mu", "p_art"]).set_index("MVT_ID_mvt")
    bb = b.reindex(Ge["MVT_ID_mvt"].to_numpy())
    mu_g, pa_g = bb["mu"].to_numpy(dtype=float), bb["p_art"].to_numpy(dtype=float)
    d, dd = Ge["d_sch"].to_numpy(dtype=float), Ge["dd"].to_numpy()
    Hf, He = with_y(gk.add_new(Gf, T), Y, ycols), with_y(gk.add_new(Ge, T), Y, ycols)
    Hf2, He2 = Hf.copy(), He.copy()
    Hf2["pfx"], He2["pfx"] = Hf2["k_car"], He2["k_car"]
    cols = gx.XCOLS + list(gk.NEW) + ycols
    P = 0.5 * (gk.probs(gx.with_x(Hf, X), gx.with_x(He, X), cols) + gk.probs(gx.with_x(Hf2, X), gx.with_x(He2, X), cols))
    g_n, pn_n = g_override(P, d, dd, mu_g, pa_g)
    keep = d >= gx.CUT  # 가드 행: gkeyx 값 그대로(차 0)
    g_n = np.where(keep, V["g_new"].to_numpy(), np.clip(g_n, 0, None))
    pn_n = np.where(keep, V["pn_new"].to_numpy(), pn_n)
    return pd.DataFrame({"g_old": V["g_old"].to_numpy(), "g_av": V["g_av"].to_numpy(), "g_x": V["g_new"].to_numpy(), "pn_x": V["pn_new"].to_numpy(),
                         "g_n": g_n, "pn_n": pn_n, "mu_g": mu_g, "d_sch": d}, index=V.index)


def check_gkeyx(setting: str, gm: dict, V: pd.DataFrame, f_gx: pd.Series | None = None) -> float:
    """다시 계산한 gkeyx G 가 gkeyx 파일의 G 행을 그 원본(gkey avg 판)에서 1e-3 초 안에 다시 만드는가. f_gx 를 주면 그것을 gkeyx 파일로 본다."""
    src_m = json.loads((ROOT / gm["source"]["manifest"]).read_text(encoding="utf-8"))
    f_src = load_pred(src_m[KEYPART[setting]]["path"])
    f_gx = load_pred(gm[KEYPART[setting]]["path"]) if f_gx is None else f_gx
    ids = V.index.to_numpy()
    want = np.clip(f_src.reindex(ids).to_numpy() + V["g_x"].to_numpy() - V["g_av"].to_numpy(), 0, None)
    err = float(np.max(np.abs(want - f_gx.reindex(ids).to_numpy()))) if len(ids) else 0.0
    if not err < 1e-3:
        raise SystemExit(f"{setting}: 다시 계산한 gkeyx G 가 gkeyx 파일과 다르다(최대 차 {err})")
    return err


def overlay(pred: pd.Series, V: pd.DataFrame) -> tuple[pd.Series, dict]:
    miss = int((~V.index.isin(pred.index)).sum())
    if miss:
        raise SystemExit(f"후보 예측에 없는 G 행이 {miss} 개다")
    new = pred.copy()
    old_g = pred.reindex(V.index).to_numpy(dtype=float)
    new.loc[V.index] = np.clip(old_g + (V["g_n"].to_numpy() - V["g_x"].to_numpy()), 0, None)
    ch = new.to_numpy() != pred.to_numpy()
    nong = ~pred.index.isin(V.index)
    if not np.array_equal(new.to_numpy()[nong], pred.to_numpy()[nong]):
        raise SystemExit("G 아닌 행이 바뀌었다")
    gx = load_tool("gkeyx")
    info = {"g_rows": int(len(V)), "changed": int(ch.sum()), "outside_g": int((ch & nong).sum()), "guarded_rows": int((V["d_sch"] >= gx.CUT).sum()),
            "mean_g_x": round(float(V["g_x"].mean()), 1), "mean_g_n": round(float(V["g_n"].mean()), 1),
            "mean_pn_x": round(float(V["pn_x"].mean()), 4), "mean_pn_n": round(float(V["pn_n"].mean()), 4)}
    return new, info


def load_tables(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """이 단계 manifest 의 config 로 표 셋을 읽고 해시를 확인한다(classes, 재현 명령)."""
    for k, tb in (("--table_sha", "--table"), ("--xtable_sha", "--xtable"), ("--ytable_sha", "--ytable")):
        if tb in cfg and sha16(f"cache/{cfg[tb]}.parquet") != cfg[k]:
            raise SystemExit(f"cache/{cfg[tb]}.parquet sha16 differs from the gkeyxn manifest's {cfg[k]}")
    T = pd.read_parquet(ROOT / "cache" / f"{cfg['--table']}.parquet")
    X = pd.read_parquet(ROOT / "cache" / f"{cfg['--xtable']}.parquet").set_index("MVT_ID_mvt")
    Y = pd.read_parquet(ROOT / "cache" / f"{cfg['--ytable']}.parquet").set_index("MVT_ID_mvt") if "--ytable" in cfg else None
    return T, X, Y


def classes(setting: str, gm: dict, f_src: pd.Series, f_xn: pd.Series) -> pd.DataFrame:
    """src/aiu_shift.py 가 부른다: 이 단계 manifest gm 의 G 모형에서 G 행의 정상 확률(pn)과 mu_g. 먼저 이 단계 파일(f_xn)을 그 원본(gkeyx 파일 f_src)에서
    1e-3 초 안에 다시 만드는지 확인한다."""
    cfg = gm["config"]
    variant = cfg["--variant"]
    if bool(VARIANTS[variant]["ycols"]) != ("--ytable" in cfg):
        raise SystemExit(f"gkeyxn manifest: variant {variant} and --ytable do not match")
    T, X, Y = load_tables(cfg)
    V = xn_values(setting, cfg, T, X, Y, variant)
    ids = V.index.to_numpy()
    want = np.clip(f_src.reindex(ids).to_numpy(dtype=float) + V["g_n"].to_numpy() - V["g_x"].to_numpy(), 0, None)
    err = float(np.max(np.abs(want - f_xn.reindex(ids).to_numpy(dtype=float)))) if len(ids) else 0.0
    print(f"  {setting}: gkeyxn ({variant}) G model recomputed for {len(ids)} G rows; against the gkeyxn file max diff {err:.2e} s", flush=True)
    if not err < 1e-3:
        raise SystemExit(f"{setting}: the recomputed gkeyxn G model does not reproduce the gkeyxn file (max diff {err})")
    return pd.DataFrame({"pn": V["pn_n"].to_numpy(), "mu_g": V["mu_g"].to_numpy()}, index=ids)


# ---- 원본, 표 ----

def source(arg: str, strict: bool = True, allow_gkc: bool = False) -> tuple[dict, str, str | None]:
    """원 후보: manifest 경로(.json) 또는 큐 이름. gkeyx 판(--variant avg_x)이어야 한다(allow_gkc 면 gkey avg 판도, check 전용).
    strict 가 아니면(check) 코드 사본 차이는 경고만 하고 재현 확인(1e-3 초)이 지킨다."""
    from src.batches import manifest_of
    if arg.endswith(".json"):
        meta, stem = {"manifest": arg}, None
    else:
        meta, stem = json.loads((ROOT / "queue" / f"{arg}.json").read_text(encoding="utf-8")), arg
    sm, why = manifest_of(meta)
    if why:
        raise SystemExit(f"거부: 원 후보 manifest 검사 {why}")
    cfg = sm.get("config", {})
    ok = {"avg_x"} | ({"avg"} if allow_gkc else set())
    if cfg.get("--variant") not in ok or "--leg" not in cfg:
        raise SystemExit(f"거부: 원 후보가 tools/gkeyx.py 판(--variant avg_x)이 아니다({cfg.get('--variant')})")
    if sm["spec"].endswith("aw") or sm["spec"].endswith("ds"):
        raise SystemExit("거부: 원 후보가 반대 창 판이거나 AIU 날 이동 판이다(이 단계는 그 앞에 둔다)")
    for f in ("src/model3.py", "src/sub_g.py", "tools/gkey.py", "tools/gkeyx.py"):
        if f in sm["code_sha"] and sha16(f) != sm["code_sha"][f]:
            if strict:
                raise SystemExit(f"거부: {f} 가 원 후보 manifest 의 사본과 다르다")
            print(f"경고: {f} 가 원 후보 manifest 의 사본과 다르다(check 라 계속, 재현 확인이 지킨다)", flush=True)
    return sm, meta["manifest"], stem


def tables(cfg: dict, variant: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict]:
    """gkey 표는 원 manifest 의 해시와 같아야 한다. gkeyx 표는 원 manifest 에 있으면 그 해시와 같아야 하고, 없으면(gkey avg 판, check) 지금 파일.
    이 단계 표(avg_xnc)는 없으면 만든다. 돌려주는 dict 는 manifest config 에 들어갈 표 이름과 해시."""
    if sha16(f"cache/{cfg['--table']}.parquet") != cfg["--table_sha"]:
        raise SystemExit("거부: gkey 표 해시가 원 후보 manifest 와 다르다")
    gx = load_tool("gkeyx")
    xt = cfg.get("--xtable", gx.XTABLE)
    xsha = sha16(f"cache/{xt}.parquet")
    if "--xtable_sha" in cfg and xsha != cfg["--xtable_sha"]:
        raise SystemExit("거부: gkeyx 표 해시가 원 후보 manifest 와 다르다")
    tb = {"--table": cfg["--table"], "--table_sha": cfg["--table_sha"], "--xtable": xt, "--xtable_sha": xsha}
    Y = None
    if VARIANTS[variant]["ycols"]:
        yt = VARIANTS[variant]["ytable"]
        Y = build_table(version=yt).set_index("MVT_ID_mvt")
        tb.update({"--ytable": yt, "--ytable_sha": sha16(f"cache/{yt}.parquet")})
    T = pd.read_parquet(ROOT / "cache" / f"{cfg['--table']}.parquet")
    X = pd.read_parquet(ROOT / "cache" / f"{xt}.parquet").set_index("MVT_ID_mvt")
    return T, X, Y, tb


def run_setting(setting: str, sm: dict, T: pd.DataFrame, X: pd.DataFrame, Y: pd.DataFrame | None, variant: str, gx_in_memory: bool = False):
    """sm: gkeyx manifest(또는 gx_in_memory 면 gkey avg 판 manifest: gkeyx 를 메모리에서 얹는다). (새 표, gkeyx 예측, 새 예측, 정보)."""
    V = xn_values(setting, sm["config"], T, X, Y, variant)
    if gx_in_memory:
        gx = load_tool("gkeyx")
        f_gkc = load_pred(sm[KEYPART[setting]]["path"])
        Vx = V.rename(columns={"g_x": "g_new", "pn_x": "pn_new"})
        gx.check_gkey(setting, sm, Vx)  # gkey avg 판 재현(gkeyx 의 확인 그대로)
        old, _ = gx.overlay(f_gkc, Vx)  # 메모리 안 gkeyx 판
        err = 0.0
        old_df = pd.read_parquet(ROOT / sm[KEYPART[setting]]["path"])
        old_df["pred"] = old.reindex(old_df["MVT_ID_mvt"].to_numpy()).to_numpy()
    else:
        err = check_gkeyx(setting, sm, V)
        old_df = pd.read_parquet(ROOT / sm[KEYPART[setting]]["path"])
        old = old_df.set_index("MVT_ID_mvt")["pred"].astype(float)
    new, info = overlay(old, V)
    info["gkeyx_repro_max_diff"] = float(f"{err:.3e}")
    out = old_df.copy()
    out["pred"] = new.reindex(old_df["MVT_ID_mvt"].to_numpy()).to_numpy()
    return out, old, new, info


def check(arg: str, variant: str) -> int:
    sm, src_mp, _ = source(arg, strict=False, allow_gkc=True)
    in_mem = sm["config"]["--variant"] == "avg"
    T, X, Y, tb = tables(sm["config"], variant)
    gx = load_tool("gkeyx")
    y = gx.truth()
    print(f"[check] {variant}: 원 {src_mp}({'gkey avg 판, gkeyx 를 메모리에서 얹음' if in_mem else 'gkeyx 판'}), 다리 {sm['config']['--leg']}, 표 {tb}", flush=True)
    for setting in ("full", "val", "fwd"):
        _, old, new, info = run_setting(setting, sm, T, X, Y, variant, gx_in_memory=in_mem)
        if setting == "full":
            info["move_mse_test_n"] = round(float(((new - old) ** 2).sum() / N_TEST), 1)
        else:
            info.update(gx.delta(old, new, y))
        print(f"[check] {setting}: {info}", flush=True)
    return 0


def build(arg: str, num: str, variant: str) -> int:
    from src.ruler import ruler_hash
    sm, src_mp, stem = source(arg)
    tag = sm["spec"] + VARIANTS[variant]["suffix"]
    name = f"{num}_{tag}"
    if (ROOT / "queue" / f"{name}.json").exists() or (ROOT / f"preds/manifest_{tag}.json").exists():
        print("거부: 같은 이름이 이미 있다", name)
        return 1
    T, X, Y, tb = tables(sm["config"], variant)
    gx = load_tool("gkeyx")
    y = gx.truth()
    rep, files = {}, {}
    for setting in ("full", "val", "fwd"):
        out, old, new, info = run_setting(setting, sm, T, X, Y, variant)
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
            rep[setting] = {**info, **gx.delta(old, new, y)}
        out.to_parquet(ROOT / rel, index=False)
        files[setting] = rel
        print(f"[{name}] {setting}: {rep[setting]}", flush=True)
    es_rel = f"preds/es_valid_ids_{tag}.parquet"
    shutil.copy2(ROOT / sm["es_valid_ids"]["path"], ROOT / es_rel)
    me = "tools/gkeyxn.py"
    h = sha16(me)
    snap = f"preds/code/{h}_gkeyxn.py"
    if not (ROOT / snap).exists():
        shutil.copy2(ROOT / me, ROOT / snap)
    c0 = sm["config"]
    cfg = {"--src": src_mp, "--src_sha": sha16(src_mp), "--leg": c0["--leg"], "--leg_feats": c0["--leg_feats"], **tb, "--variant": variant, "--tag": tag}
    flags = " ".join(f"{k} {v}" for k, v in cfg.items())
    cmd = lambda s: f".venv/Scripts/python.exe -X utf8 tools/gkeyxn.py --setting {s} {flags}"  # noqa: E731
    hold_ids = pd.read_parquet(ROOT / files["val"])["MVT_ID_mvt"].to_numpy()
    gk = gx.gkey()
    ycols = VARIANTS[variant]["ycols"]
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
        "iteration_rule": sm.get("iteration_rule", "") + f" | gkeyxn ({variant}): G model refit per setting on the g_band training rows; the nine R3 stand "
                          "and link columns" + (" and the carrier-month arrival copy share" if ycols else "") + " added to gkeyx's twelve on the GBM side, "
                          "gkeyx's 43,200 s d_sch guard and 0.5 key average kept, nothing chosen on the ruler",
        "source": {"queue": stem, "manifest": src_mp, "manifest_sha": sha16(src_mp), "spec": sm["spec"],
                   "holdout_sha": sm["holdout"]["sha"], "test_sha": sm["test"]["sha"], "forward_sha": sm["forward"]["sha"]},
        "gkeyxn": {"leg": c0["--leg"], "leg_feats": c0["--leg_feats"], "table": f"cache/{tb['--table']}.parquet", "table_sha": tb["--table_sha"],
                   "xtable": f"cache/{tb['--xtable']}.parquet", "xtable_sha": tb["--xtable_sha"],
                   **({"ytable": f"cache/{tb['--ytable']}.parquet", "ytable_sha": tb["--ytable_sha"]} if ycols else {}),
                   "columns": gx.XCOLS + list(gk.NEW) + ycols, "d_sch_guard": gx.CUT, "variant": variant,
                   "model": "gkeyx (gkey avg keys, twelve record columns) with the nine R3 stand and link columns"
                            + (" and ca_copy_mon (same carrier, same month LIRF arrivals with BLOCK = SCHED within 6.5 s)" if ycols else "")
                            + " also on the GBM side, used on G rows with d_sch < 43,200 s (guarded rows keep gkeyx); band rule and mu of the g_band leg; "
                              "added to the gkeyx candidate as (new G - reproduced gkeyx G); AIU day shift and antiwin must stay after this",
                   "report": rep},
        "written_kst": now_kst(),
    }
    mp = f"preds/manifest_{tag}.json"
    (ROOT / mp).write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{name}] manifest {mp}", flush=True)
    ytxt = f", 이 단계 표 {tb['--ytable']} {tb['--ytable_sha']}" if ycols else ", 새 표 없음"
    method = (f"{stem or src_mp} 의 G 행(LIRF, NM 미매칭)에 칸을 더한 G 모형을 얹은 판(tools/gkeyxn.py {variant}, gkey 표 {tb['--table']} {tb['--table_sha']}, "
              f"gkeyx 표 {tb['--xtable']} {tb['--xtable_sha']}{ytxt}, G 다리 {c0['--leg']}). gkeyx 식(목적지 키와 운항사 키 평균, 칸 열둘)의 GBM 쪽에 "
              "스탠드 앞뒤 도착, 출발 간격과 반대 창 이음 칸 아홉" + ("과 같은 운항사의 그 달 LIRF 도착 가운데 BLOCK = SCHED 복사 몫" if ycols else "")
              + "을 더하고 MVT − SCHED < 43,200 행에만 쓴다(나머지는 gkeyx 그대로). 띠 규칙과 mu 는 다리의 것, 후보에는 (새 G − 재현한 gkeyx G)를 더했다. "
              "학습 행은 설정마다 g_band 와 같고 자로 고른 값은 없다. 근거 vault research/2026-10-06-r6-g.md. AIU 날 이동과 반대 창은 이 뒤에 둔다. "
              f"자 ΔMSE {rep['val']['dmse']}({rep['val']['improved']}/{rep['val']['changed']}), 7월 전방 ΔMSE {rep['fwd']['dmse']}"
              f"({rep['fwd']['improved']}/{rep['fwd']['changed']}), 시험 바뀐 행 {rep['full']['changed']}, 이동 MSE {rep['full']['move_mse_test_n']}, "
              "10-05 단일 발 체제라 준비 점검은 경고(--readiness-warn)")
    (ROOT / "notes" / f"gkeyxn_{name}.json").write_text(json.dumps({"name": name, "manifest": mp, "source": stem or src_mp, "method": method,
                                                                    "report": rep, "written_kst": now_kst()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def enqueue(name: str) -> int:
    info = json.loads((ROOT / "notes" / f"gkeyxn_{name}.json").read_text(encoding="utf-8"))
    r = subprocess.run([sys.executable, "-X", "utf8", "tools/enqueue.py", info["manifest"], name, info["method"], "--readiness-warn"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1500:], r.stderr[-800:], flush=True)
    return r.returncode


def arg_variant(a: list[str]) -> str:
    v = a[a.index("--variant") + 1] if "--variant" in a else DEFAULT
    if v not in VARIANTS:
        raise SystemExit(f"--variant 는 {sorted(VARIANTS)} 가운데 하나다")
    return v


def main() -> int:
    load_tool("gkey").below_normal()
    a = sys.argv[1:]
    if a and a[0] == "table":
        build_table(force="--force" in a, version=a[a.index("--version") + 1] if "--version" in a else YTABLE)
        return 0
    if a and a[0] == "check":
        return check(a[1], arg_variant(a))
    if a and a[0] == "_files":
        return build(a[1], a[2], arg_variant(a))
    if a and a[0] == "enqueue":
        return enqueue(a[1])
    if a and a[0] == "build":
        if len(a) < 3:
            print("usage: gkeyxn.py build <gkeyx 판 manifest 또는 큐 이름> <새 큐 번호> [--no-enqueue] [--variant avg_xn]")
            return 2
        variant = arg_variant(a)
        me = [sys.executable, "-X", "utf8", str(Path(__file__).resolve())]
        rc = subprocess.run(me + ["_files", a[1], a[2], "--variant", variant], cwd=ROOT).returncode  # 데이터 단계는 따로 끝내 메모리를 돌려준다
        if rc != 0 or "--no-enqueue" in a:
            return rc
        sm, _, _ = source(a[1])
        return subprocess.run(me + ["enqueue", f"{a[2]}_{sm['spec']}{VARIANTS[variant]['suffix']}"], cwd=ROOT).returncode
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--seed", default="0")
    for k in ("--src", "--src_sha", "--leg", "--leg_feats", "--table", "--table_sha", "--xtable", "--xtable_sha", "--tag"):
        pa.add_argument(k, required=True)
    pa.add_argument("--ytable")
    pa.add_argument("--ytable_sha")
    pa.add_argument("--variant", required=True, choices=list(VARIANTS))
    g = pa.parse_args(a)
    if sha16(g.src) != g.src_sha:
        print("거부: 원 manifest 해시가 다르다")
        return 1
    cfg = {"--table": g.table, "--table_sha": g.table_sha, "--xtable": g.xtable, "--xtable_sha": g.xtable_sha}
    if VARIANTS[g.variant]["ycols"]:
        if not g.ytable or not g.ytable_sha:
            print("거부: avg_xnc 는 --ytable, --ytable_sha 가 있어야 한다")
            return 1
        cfg.update({"--ytable": g.ytable, "--ytable_sha": g.ytable_sha})
    T, X, Y = load_tables(cfg)
    sm = json.loads((ROOT / g.src).read_text(encoding="utf-8"))
    out, _, _, info = run_setting(g.setting, sm, T, X, Y, g.variant)
    rel = sorted((ROOT / "preds").glob(f"holdout_{g.tag}_rmse*.parquet"))[0].relative_to(ROOT) if g.setting == "val" else \
        Path(f"preds/{'fwd' if g.setting == 'fwd' else 'test'}_{g.tag}.parquet")
    out.to_parquet(ROOT / rel, index=False)
    print(rel, info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
