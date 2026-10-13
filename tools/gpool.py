# SPDX-License-Identifier: GPL-3.0-only
"""Sharper type probabilities for the G rows (docs/DESIGN.md section 8.5): the four components of the G model of tools/gkeyxn.py
(two keys, each a lookup table and a GBM) are combined by a log pool instead of their mean, P' proportional to the components'
geometric mean raised to a power a. One exponent a in [0.25, 4] minimises the three-class log loss of out-of-month probabilities on
the fitting months. G rows with d_sch of 43,200 s or more, rows changed by antiwin and rows that are not G stay byte for byte.

    python tools/gpool.py --setting {val,fwd,full} --src <manifest> --src_sha <16> --tag <spec>

Original note (Korean):
G 행의 세 유형 확률을 G 모형 성분 넷의 로그 풀로 날카롭게 한다(R9-S1). 반대 창으로 끝나는 큐 후보 하나 위 맨 끝에 얹는다(G 행만).
근거: vault `01 Projects/공모전 파이프라인/착수/PRC Data Challenge 2026/research/2026-10-07-r9-lirf-coverage.md` 7절(사전 등록 R9-S1, 측정
research/r9_lirf/c7_gpool.py). R8 해부(research/2026-10-07-r8-anatomy.md 5절)가 G 정상 확률이 양 끝에서 무디다고 봤고(lomo N0 0.7 ~ 0.8 → 실제 0.938),
그 까닭으로 성분을 평균하는 꼴을 짚었다. 남의 코드 없음.

G 모형(tools/gkeyxn.py, 사슬의 gkeyxn 단계 manifest 의 표와 변형 그대로)은 키 둘(목적지 pfx, 운항사 k_car) x {찾기 표 P1(pfx_adjust), GBM P2(시드 0 ~ 2
평균)} 넷을 같은 무게로 평균한다(선형 풀). 이 단계는 그 넷을 로그 풀로 합친다:
  P'_c ∝ exp((a / 4) Σ_성분 log max(P_성분,c, 1e-4)), c = 정상, art, shift 로 다시 정규화.
  a 하나를 세 유형 로그 손실(src.sub_g.labels, d < 43,200, 정답 있음)을 [0.25, 4] 에서 황금 분할로 최소화해 맞춘다. a 를 맞추는 행의 성분 확률은
  그 행의 달을 뺀 G 모형에서 온다: val 은 학습 열 달(1, 7월 밖)의 달 j 마다 나머지 아홉 달로 학습, fwd 는 1 ~ 6월의 j 마다 나머지 다섯 달,
  full(시험) 은 2025 열두 달의 j 마다 나머지 열한 달.
G 값: g' = clip(src.model3.g_override(P', d, dd, mu_g, pa_g), 0)(띠 평균 그대로, 사슬의 G 다리 --leg 의 mu, p_art), 원래 g 는 선형 풀의 같은 식이고
  gkeyxn 의 G 값과 같은지 먼저 확인한다(1e-3 초 넘게 다르면 멈춘다). 새 예측 = clip(원 예측 + g' − g, 0). 가드 행(d ≥ 43,200), 반대 창이 바꾼 행,
  G 아닌 행은 바이트까지 그대로(확인).
측정(사전 등록 관문 넷 모두 넘음, 490 꼴: 468, 486 에 걸고 반씩): lomo-12 G SSE −3.45%(10/12 달, 날 다시 뽑기 Δ<0 1.000), 가린 7월(R9 덮임 맞춘 자)
  ΔMSE −433(0.945), 보통 val −520(0.975), fwd −1,541(0.982), 시험 이동 487.5 MSE(G 344 행).
주의: 이 도구의 성분 확률(gkeyxn 의 지금 표)로 다시 만든 선형 풀은 val, fwd, full 에서 gkeyxn 파일의 G 값과 0 차로 같지만, R6 의 연구 파일
  research/r6_g/s9_E4_camon_rt.parquet 의 lomo 확률과는 달에 따라 최대 0.32 다르다(R6 의 옛 표나 옛 행 정보로 보인다). lomo 자는 이 도구의 성분으로 잰 것이다.

쓰는 법(저장소에서, memcap 안):
    .venv/Scripts/python.exe -X utf8 tools/gpool.py check <원 큐 이름>
        세 설정의 a, 원 후보 대비 ΔMSE(val, fwd), 시험 이동만 찍는다(아무 파일도 쓰지 않는다).
    .venv/Scripts/python.exe -X utf8 tools/gpool.py build <원 큐 이름> <새 큐 번호> [--no-enqueue]
        원 후보(반대 창으로 끝나는 판)의 자, 7월 전방, 시험 예측에 얹어 preds/ 에 새 파일 셋과 manifest 를 쓰고 notes/gpool_<새 큐 이름>.json 을 남긴 뒤
        tools/enqueue.py --readiness-warn 으로 큐에 넣는다. 새 spec 은 원 spec 뒤에 "gp" 를 붙인다.
    .venv/Scripts/python.exe -X utf8 tools/gpool.py --setting {val,fwd,full} --src <원 manifest> --src_sha <16> --tag <새 spec>
        manifest 의 commands 가 적는 재현 명령. 그 설정의 a 를 같은 길로 다시 맞추고 그 설정의 파일 하나만 다시 쓴다.
    .venv/Scripts/python.exe -X utf8 tools/gpool.py enqueue <새 큐 이름>
        build 의 큐 넣기 단계만 다시 한다(notes/gpool_<새 큐 이름>.json). 데이터 단계와 따로 띄운다(준비 점검이 메모리를 많이 쓴다).
"""
from __future__ import annotations

import ctypes
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
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
KEY = {"val": "holdout", "fwd": "forward", "full": "test"}
N_TEST = 344841
SUFFIX = "gp"
VARIANTS_OK = ("avg_xncr",)  # 이 규칙을 잰 gkeyxn 변형
FITM = {"val": [2, 3, 4, 5, 6, 8, 9, 10, 11, 12], "fwd": [1, 2, 3, 4, 5, 6], "full": list(range(1, 13))}
A_LO, A_HI, EPS = 0.25, 4.0, 1e-4
CODE_SAME = ("tools/gkey.py", "tools/gkeyx.py", "tools/gkeyxn.py", "src/model3.py", "src/sub_g.py")
CAVEAT = ("성분 확률로 다시 만든 선형 풀은 val, fwd, full 에서 gkeyxn 파일의 G 값과 0 차로 같지만 R6 의 연구 파일 research/r6_g/s9_E4_camon_rt.parquet 의 "
          "lomo 확률과는 달에 따라 최대 0.32 다르다(옛 표나 옛 행 정보로 보인다). 사전 등록의 lomo-12 자는 이 도구의 성분으로 잰 것이다.")


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


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_meta(stem: str) -> dict:
    for q in (ROOT / "queue", ROOT / "queue" / "done"):
        p = q / f"{stem}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise SystemExit(f"큐에 {stem} 이 없다(queue/, queue/done/)")


def load_man(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def gk_manifest(sm: dict) -> dict:
    """원 후보의 사슬에서 gkeyxn 단계 manifest(config 에 --variant 가 avg_xn* 인 첫 판)."""
    m = sm
    for _ in range(40):
        if m.get("config", {}).get("--variant", "").startswith("avg_xn"):
            return m
        nxt = m.get("source", {}).get("manifest")
        if not nxt:
            break
        m = load_man(nxt)
    raise SystemExit("원 후보의 사슬에 gkeyxn 단계가 없다")


class GModel:
    """gkeyxn 단계 config 의 표와 변형으로 G 모형 성분 넷을 다시 만든다(tools/gkey.probs 의 안쪽을 성분별로)."""

    def __init__(self, cfg: dict):
        if cfg["--variant"] not in VARIANTS_OK:
            raise SystemExit(f"gkeyxn 변형 {cfg['--variant']} 은 이 규칙을 잰 변형({VARIANTS_OK})이 아니다")
        self.cfg = cfg
        self.GN = load_tool("gkeyxn")
        self.GX = self.GN.load_tool("gkeyx")
        self.GK = self.GX.gkey()
        import src.sub_g as sub_g
        from src.model3 import g_override
        self.sub_g, self.g_override = sub_g, g_override
        self.T, self.X, self.Y = self.GN.load_tables(cfg)
        ycols = self.GN.VARIANTS[cfg["--variant"]]["ycols"]
        self.cols = self.GX.XCOLS + list(self.GK.NEW) + ycols
        G = self.GK.g_frame(cfg["--leg_feats"])
        self.H1 = self.GX.with_x(self.GN.with_y(self.GK.add_new(G, self.T), self.Y, ycols), self.X)
        self.H2 = self.H1.copy()
        self.H2["pfx"] = self.H2["k_car"]
        self.s = G["src"].to_numpy()
        self.y = G["y"].to_numpy(dtype=float)
        self.d = G["d_sch"].to_numpy(dtype=float)
        self.dd = G["dd"].to_numpy()
        self.ids = G["MVT_ID_mvt"].to_numpy()
        self.fin = np.isfinite(self.y)
        self.CUT = self.GX.CUT

    def comps1(self, Hf: pd.DataFrame, He: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        sg = self.sub_g
        ytr, dtr = Hf["y"].to_numpy(), Hf["d_sch"].to_numpy().astype(float)
        k = sg.labels(ytr, dtr)
        c_tr, c_te = sg.cell(dtr, Hf["dd"].to_numpy()), sg.cell(He["d_sch"].to_numpy().astype(float), He["dd"].to_numpy())
        P1 = sg.pfx_adjust(sg.predict_lookup(sg.fit_lookup(c_tr, k), c_te), He["pfx"].to_numpy(), Hf["pfx"].to_numpy(), k)
        feat = [c for c in ["d_sch", "dd", "hour_mvt", "hour_sch", "garb", "fl_len", "stand_d"] if c in Hf.columns] + self.cols
        cats = pd.CategoricalDtype(sorted(set(Hf["pfx"]) | set(He["pfx"])))
        Xtr = Hf[feat].astype("float32").copy()
        Xtr["pfx"] = Hf["pfx"].astype(cats)
        Xte = He[feat].astype("float32").copy()
        Xte["pfx"] = He["pfx"].astype(cats)
        P2 = np.mean([sg.gbm_probs(Xtr, k, Xte, seed=sd) for sd in (0, 1, 2)], axis=0)
        return P1, P2

    def comps(self, fit_mask: np.ndarray, eval_mask: np.ndarray) -> list[np.ndarray]:
        a1, b1 = self.comps1(self.H1[fit_mask], self.H1[eval_mask])
        a2, b2 = self.comps1(self.H2[fit_mask], self.H2[eval_mask])
        return [a1, b1, a2, b2]

    def fit_a(self, setting: str) -> dict:
        """그 설정의 a: 맞춤 달 j 마다 (설정의 학습 달 − j)로 학습한 성분의 j 행 확률로 로그 손실 최소."""
        months = FITM[setting]
        Cs, ks = [[], [], [], []], []
        for j in months:
            fm = np.isin(self.s, months) & (self.s != j) & self.fin
            em = (self.s == j) & self.fin & (self.d < self.CUT)
            C = self.comps(fm, em)
            for i in range(4):
                Cs[i].append(C[i])
            ks.append(self.sub_g.labels(self.y[em], self.d[em]))
        Cs = [np.vstack(c) for c in Cs]
        kk = np.concatenate(ks)

        def loss(a):
            P = logpool(Cs, a)
            return float(-np.mean(np.log(np.maximum(P[np.arange(len(kk)), kk], 1e-12))))
        lo, hi = A_LO, A_HI
        gr = (np.sqrt(5) - 1) / 2
        x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
        f1, f2 = loss(x1), loss(x2)
        for _ in range(60):
            if f1 < f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - gr * (hi - lo)
                f1 = loss(x1)
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + gr * (hi - lo)
                f2 = loss(x2)
        a = 0.5 * (lo + hi)
        return {"a": round(float(a), 6), "logloss_a1": round(loss(1.0), 5), "logloss_a": round(loss(a), 5), "fit_rows": int(len(kk)), "months": months}

    def delta(self, setting: str, a: float) -> tuple[pd.Series, float]:
        """그 설정 평가 G 행의 g' − g(가드 행 0)와 선형 풀 재현 차(gkeyxn 의 G 값 대비)."""
        em = self.GK.EVAL[setting](self.s)
        fm = self.GK.FIT[setting](self.s) & self.fin
        gids = self.ids[em]
        C = self.comps(fm, em)
        b = pd.read_parquet(ROOT / "cache" / f"{setting}_{self.cfg['--leg']}_s0.parquet", columns=["MVT_ID_mvt", "mu", "p_art"]).set_index("MVT_ID_mvt").reindex(gids)
        mu_g, pa_g = b["mu"].to_numpy(dtype=float), b["p_art"].to_numpy(dtype=float)
        d, dd = self.d[em], self.dd[em]
        g0 = np.clip(self.g_override(lin(C), d, dd, mu_g, pa_g)[0], 0, None)
        g1 = np.clip(self.g_override(logpool(C, a), d, dd, mu_g, pa_g)[0], 0, None)
        guard = d >= self.CUT
        V = self.GN.xn_values(setting, self.cfg, self.T, self.X, self.Y, self.cfg["--variant"])
        err = float(np.max(np.abs(g0[~guard] - V["g_n"].reindex(gids).to_numpy(dtype=float)[~guard]))) if (~guard).any() else 0.0
        if not err < 1e-3:
            raise SystemExit(f"{setting}: 선형 풀 재현이 gkeyxn 의 G 값과 다르다(최대 차 {err})")
        return pd.Series(np.where(guard, 0.0, g1 - g0), index=gids), err


def lin(C: list[np.ndarray]) -> np.ndarray:
    return 0.5 * (0.5 * C[0] + 0.5 * C[1]) + 0.5 * (0.5 * C[2] + 0.5 * C[3])


def logpool(C: list[np.ndarray], a: float) -> np.ndarray:
    L = (a / 4.0) * sum(np.log(np.maximum(c, EPS)) for c in C)
    L = L - L.max(axis=1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(axis=1, keepdims=True)


def apply_setting(setting: str, sm: dict, pre: dict, dl: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """원 후보 파일에 G 이동을 얹는다. 반대 창이 바꾼 행(원 후보와 그 바로 앞 판이 다른 행)과 가드 행은 그대로."""
    old = pd.read_parquet(ROOT / sm[KEY[setting]]["path"])
    f = old.set_index("MVT_ID_mvt")["pred"].astype(float)
    fp = pd.read_parquet(ROOT / pre[KEY[setting]]["path"]).set_index("MVT_ID_mvt")["pred"].astype(float).reindex(f.index)
    aw = np.abs(f.to_numpy() - fp.to_numpy()) > 1e-6
    dlt = dl.reindex(f.index).fillna(0.0).to_numpy()
    ch = (dlt != 0) & ~aw
    nv = f.to_numpy().copy()
    nv[ch] = np.clip(nv[ch] + dlt[ch], 0, None)
    new_s = pd.Series(nv, index=f.index)
    gset = set(dl.index)
    outside = int((~f.index.isin(list(gset)) & (nv != f.to_numpy())).sum())
    if outside:
        raise SystemExit(f"{setting}: G 아닌 행 {outside} 개가 바뀌었다")
    new = old.copy()
    new["pred"] = new_s.reindex(old["MVT_ID_mvt"].to_numpy()).to_numpy()
    info = {"g_rows": int(len(dl)), "g_rows_moving": int((dl != 0).sum()), "antiwin_rows_kept": int((aw & (dlt != 0)).sum()),
            "changed": int((nv != f.to_numpy()).sum()), "move_mse_rows_n": round(float(np.mean((nv - f.to_numpy()) ** 2)), 1)}
    return new, f, new_s, info


def delta(old: pd.Series, new: pd.Series, y: pd.Series) -> dict:
    yy = y.reindex(old.index).to_numpy()
    ok = np.isfinite(yy)
    o, n, yy = old.to_numpy()[ok], new.reindex(old.index).to_numpy()[ok], yy[ok]
    se0, se1 = (o - yy) ** 2, (n - yy) ** 2
    N = len(yy)
    ch = n != o
    return {"rows": int(N), "rmse_old": round(float(np.sqrt(se0.mean())), 3), "rmse_new": round(float(np.sqrt(se1.mean())), 3),
            "dmse": round(float((se1.sum() - se0.sum()) / N), 1), "changed": int(ch.sum()), "improved": int((se1 < se0)[ch].sum())}


def out_rel(setting: str, tag: str, new_s: pd.Series, y: pd.Series | None) -> str:
    if setting == "val":
        rm = float(np.sqrt(((new_s - y.reindex(new_s.index)) ** 2).mean()))
        return f"preds/holdout_{tag}_rmse{rm:.2f}.parquet"
    return f"preds/fwd_{tag}.parquet" if setting == "fwd" else f"preds/test_{tag}.parquet"


def source(sm: dict) -> tuple[dict, dict]:
    if not sm["spec"].endswith("aw"):
        raise SystemExit("원 후보가 반대 창으로 끝나는 판이 아니다(이 단계는 맨 끝, 반대 창 뒤에 둔다)")
    pre = load_man(sm["source"]["manifest"])
    gk = gk_manifest(sm)
    for f in CODE_SAME:
        if f in sm["code_sha"] and sha16(f) != sm["code_sha"][f]:
            raise SystemExit(f"{f} 가 원 후보 manifest 의 사본과 다르다")
    return pre, gk


def run(sm: dict, settings=("full", "val", "fwd")) -> dict:
    pre, gk = source(sm)
    M = GModel(gk["config"])
    out = {}
    for st in settings:
        fa = M.fit_a(st)
        dl, err = M.delta(st, fa["a"])
        new, f, new_s, info = apply_setting(st, sm, pre, dl)
        out[st] = {"a": fa, "repro_err": err, "new": new, "old_s": f, "new_s": new_s, "info": info}
        print(f"  {st}: a {fa['a']:.4f}(로그 손실 a=1 {fa['logloss_a1']} → {fa['logloss_a']}, 맞춤 행 {fa['fit_rows']}), 선형 풀 재현 차 {err:.2e}, {info}", flush=True)
    return out


def check(stem: str) -> int:
    from src.batches import manifest_of
    sm, why = manifest_of(load_meta(stem))
    if why:
        print("거부: 원 후보 manifest 검사", why)
        return 1
    y = truth()
    R = run(sm)
    for st in ("val", "fwd"):
        print(f"[check {stem}] {st}: {delta(R[st]['old_s'], R[st]['new_s'], y)}", flush=True)
    print(f"[check {stem}] full: 이동 MSE(시험 N) {float(((R['full']['new_s'] - R['full']['old_s']) ** 2).sum() / N_TEST):.1f}", flush=True)
    return 0


def build(stem: str, num: str) -> int:
    from src.batches import manifest_of
    from src.ruler import ruler_hash
    meta = load_meta(stem)
    sm, why = manifest_of(meta)
    if why:
        print("거부: 원 후보 manifest 검사", why)
        return 1
    src_mp = meta["manifest"]
    tag = sm["spec"] + SUFFIX
    name = f"{num}_{tag}"
    if (ROOT / "queue" / f"{name}.json").exists() or (ROOT / f"preds/manifest_{tag}.json").exists():
        print("거부: 같은 이름이 이미 있다", name)
        return 1
    y = truth()
    R = run(sm)
    files, rep = {}, {}
    for st in ("val", "fwd", "full"):
        r = R[st]
        rel = out_rel(st, tag, r["new_s"], y if st == "val" else None)
        r["new"].to_parquet(ROOT / rel, index=False)
        files[st] = rel
        if st == "full":
            rep[st] = {**r["info"], "a": r["a"], "move_mse_test_n": round(float(((r["new_s"] - r["old_s"]) ** 2).sum() / N_TEST), 1),
                       "share_over_3600_old": round(float((r["old_s"] > 3600).mean()), 5), "share_over_3600_new": round(float((r["new_s"] > 3600).mean()), 5)}
        else:
            rep[st] = {**r["info"], "a": r["a"], **delta(r["old_s"], r["new_s"], y)}
        print(f"[{name}] {st}: {json.dumps({k: v for k, v in rep[st].items() if k != 'a'}, ensure_ascii=False)}", flush=True)
    es_rel = f"preds/es_valid_ids_{tag}.parquet"
    shutil.copy2(ROOT / sm["es_valid_ids"]["path"], ROOT / es_rel)
    me = "tools/gpool.py"
    h = sha16(me)
    snap = f"preds/code/{h}_gpool.py"
    if not (ROOT / snap).exists():
        shutil.copy2(ROOT / me, ROOT / snap)
    cfg = {"--src": src_mp, "--src_sha": sha16(src_mp), "--tag": tag}
    flags = " ".join(f"{k} {v}" for k, v in cfg.items())
    cmd = lambda s: f".venv/Scripts/python.exe -X utf8 tools/gpool.py --setting {s} {flags}"  # noqa: E731
    hold_ids = pd.read_parquet(ROOT / files["val"])["MVT_ID_mvt"].to_numpy()
    gk = gk_manifest(sm)
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
        "iteration_rule": sm.get("iteration_rule", "") + " | gpool: G class probabilities = log pool of the four G components with one exponent a, "
                          "fitted by 3-class log loss on component probabilities from G models without the row's month (val: nested over the 10 "
                          "training months, fwd: over Jan-Jun, full: the 12 leave-one-month-out probabilities)",
        "source": {"queue": stem, "manifest": src_mp, "manifest_sha": cfg["--src_sha"], "spec": sm["spec"],
                   "holdout_sha": sm["holdout"]["sha"], "test_sha": sm["test"]["sha"], "forward_sha": sm["forward"]["sha"]},
        "gpool": {"gkeyxn_manifest_spec": gk["spec"], "gkeyxn_config": gk["config"], "a": {st: R[st]["a"] for st in R},
                  "repro_err_vs_gkeyxn": {st: R[st]["repro_err"] for st in R}, "report": rep, "caveats": [CAVEAT],
                  "basis": "vault research/2026-10-07-r9-lirf-coverage.md 7 (R9-S1, pre-registered 03:5x KST 10-07)"},
        "written_kst": now_kst(),
    }
    mp = f"preds/manifest_{tag}.json"
    (ROOT / mp).write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    _, why = manifest_of({"manifest": mp})
    if why:
        print("거부: 새 manifest 검사", why)
        return 1
    print(f"[{name}] manifest {mp}(manifest_of 통과)", flush=True)
    method = (f"{stem} 의 G 행(LIRF NM 미매칭) 세 유형 확률을 G 모형 성분 넷(키 둘 x 찾기 표, GBM)의 평균 대신 로그 풀 P' ∝ exp((a/4) Σ log P_k)로 바꾼 판"
              f"(tools/gpool.py, 반대 창 뒤 맨 끝, G 행만). a 는 그 행의 달을 뺀 성분 확률의 세 유형 로그 손실로 맞춘다(val {R['val']['a']['a']:.3f}, "
              f"fwd {R['fwd']['a']['a']:.3f}, 시험 {R['full']['a']['a']:.3f}). 근거 vault research/2026-10-07-r9-lirf-coverage.md 7절(사전 등록 R9-S1: "
              "lomo-12 G SSE −3.45%, 가린 7월 −433, 보통 val −520, fwd −1,541, 490 꼴). "
              f"자 ΔMSE {rep['val']['dmse']}({rep['val']['improved']}/{rep['val']['changed']}), 7월 전방 ΔMSE {rep['fwd']['dmse']}"
              f"({rep['fwd']['improved']}/{rep['fwd']['changed']}), 시험 이동 {rep['full']['move_mse_test_n']} MSE(G {rep['full']['changed']} 행). "
              "가드 행, 반대 창 행, G 아닌 행은 그대로. 10-05 단일 발 체제라 준비 점검은 경고(--readiness-warn)")
    (ROOT / "notes" / f"gpool_{name}.json").write_text(json.dumps({"name": name, "manifest": mp, "source": stem, "method": method, "report": rep,
                                                                   "caveats": [CAVEAT], "written_kst": now_kst()}, ensure_ascii=False, indent=1, default=str),
                                                        encoding="utf-8")
    return 0


def enqueue(name: str) -> int:
    info = json.loads((ROOT / "notes" / f"gpool_{name}.json").read_text(encoding="utf-8"))
    r = subprocess.run([sys.executable, "-X", "utf8", "tools/enqueue.py", info["manifest"], name, info["method"], "--readiness-warn"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1500:], r.stderr[-800:], flush=True)
    return r.returncode


def main() -> int:
    below_normal()
    a = sys.argv[1:]
    if a and a[0] == "check":
        return check(a[1])
    if a and a[0] == "_files":
        return build(a[1], a[2])
    if a and a[0] == "enqueue":
        return enqueue(a[1])
    if a and a[0] == "build":
        if len(a) < 3:
            print("usage: gpool.py build <원 큐 이름> <새 큐 번호> [--no-enqueue]")
            return 2
        me = [sys.executable, "-X", "utf8", str(Path(__file__).resolve())]
        rc = subprocess.run(me + ["_files", a[1], a[2]], cwd=ROOT).returncode  # 데이터 단계는 따로 끝내 메모리를 돌려준다
        if rc != 0 or "--no-enqueue" in a:
            return rc
        spec = load_man(load_meta(a[1])["manifest"])["spec"]
        return subprocess.run(me + ["enqueue", f"{a[2]}_{spec}{SUFFIX}"], cwd=ROOT).returncode
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--seed", default="0")
    pa.add_argument("--src", required=True)
    pa.add_argument("--src_sha", required=True)
    pa.add_argument("--tag", required=True)
    g = pa.parse_args(a)
    if sha16(g.src) != g.src_sha:
        print("거부: 원 manifest 해시가 다르다")
        return 1
    sm = load_man(g.src)
    R = run(sm, settings=(g.setting,))
    r = R[g.setting]
    rel = out_rel(g.setting, g.tag, r["new_s"], truth() if g.setting == "val" else None)
    r["new"].to_parquet(ROOT / rel, index=False)
    print(rel, r["info"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
