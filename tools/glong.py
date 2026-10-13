# SPDX-License-Identifier: GPL-3.0-only
"""Long-delay correction for G rows (docs/DESIGN.md section 8.5): for G rows with d_sch of 30,000 s or more the normal-type probability
is scaled by r = min(1, (k + 0.5) / (E + 0.5)), with k the number of normal labels and E the summed out-of-month normal probability on
the fitting rows with such delays (in 2025 none of the 48 such G rows had a normal label); the freed share goes to the other two types.

    python tools/glong.py --setting {val,fwd,full} --src <manifest> --src_sha <16> --tag <spec>

Original note (Korean):
긴 지연 G 행(LIRF 출발 가운데 NM 미매칭, d_sch = MVT − SCHED ≥ 30,000초)의 정상 확률을 보정한다(R10-D-G1). G 행만 바꾼다.
근거: vault `01 Projects/공모전 파이프라인/착수/PRC Data Challenge 2026/research/2026-10-07-r10-d-types.md` 3절(사전 등록 R10-D-G1, 05:22 KST,
측정 research/r10_d/m1_glong.py). 2025 G 행 가운데 d ≥ 30,000 48행에 정상(제대로 적힌 오프블록)이 하나도 없다(art 37, shift 11). 사슬의 G 확률
(d < 43,200 은 gpool 로그 풀 S1, d ≥ 43,200 은 gkeyx 가드가 남긴 gkey avg)은 그런 행에 정상 확률을 0.02 ~ 0.17 준다. 남의 코드 없음.

규칙(그 설정의 맞춤 달만 쓴다: val 은 1, 7월 밖 열 달, fwd 는 1 ~ 6월, full(시험)은 2025 열두 달):
  사슬 확률 P = (정상, art, shift): d < 43,200 은 gpool 과 같은 성분 넷의 로그 풀(원 사슬 gpool manifest 의 그 설정 a), d ≥ 43,200 은 gkey avg
    (두 키 평균, 칸 없음, tools/gkeyx.gx_values 의 P_av). 먼저 재현을 확인한다: 가드 행 G 값이 gkeyx 의 g_new 와, 나머지 행의 gpool 이동이 원 사슬의
    gpool 단계 파일과 1e-3 초 안에서 같아야 한다.
  대상 행(d ≥ 30,000): P'_n = r P_n, 비운 몫 (1 − r) P_n 은 art 와 shift 에 그 둘의 비로 나눈다.
  r = min(1, (k + 0.5)/(E + 0.5)): 맞춤 달의 대상 행(정답 있음)에서 k = 정상 수, E = 그 행들의 P_n 합. 그 P_n 은 그 행의 달을 뺀 맞춤 달로 학습한
    G 모형의 밖 표본 확률(같은 a, 가드 행은 gkey avg)이다.
  G 값 g' = clip(src.model3.g_override(P', d, dd, mu_g, pa_g), 0), g = 같은 식의 P 값(사슬 G 다리 --leg 의 mu, p_art). 새 예측 = clip(원 예측 + g' − g, 0).
  반대 창이 바꾼 행, d < 30,000 G 행, G 아닌 행은 바이트까지 그대로(확인한다). 원 사슬의 gpool 단계 뒤에 G 행을 바꾼 단계가 있으면 멈춘다
  (G 아닌 행만 바꾼 단계는 앞뒤 어디에 있어도 된다. 두 단계가 바꾸는 행이 겹치지 않는다).
493 꼴(두 사슬 반씩)은 491(468gp), 492(486gp)에 각각 짓고 tools/chainavg.py 로 섞는다(gpool 과 같은 길).

쓰는 법(저장소에서, memcap 2 GB 안):
    .venv/Scripts/python.exe -X utf8 tools/glong.py check <원 큐 이름 또는 manifest>
        세 설정의 r, 재현 차, 원 후보 대비 ΔMSE(val, fwd), 시험 이동만 찍는다(아무 파일도 쓰지 않는다).
    .venv/Scripts/python.exe -X utf8 tools/glong.py build <원 큐 이름 또는 manifest> <새 큐 번호> [--no-enqueue] [--outdir <저장소 안 폴더>] [--method_dir <폴더>]
        새 파일 셋과 manifest 를 쓰고(새 spec = 원 spec + "gl"), notes/glong_<새 큐 이름>.json 을 남긴 뒤 tools/enqueue.py --readiness-warn 으로 넣는다.
        --outdir 를 주면 예측 파일, es_valid_ids, manifest, 방법 json 을 그 폴더에 쓰고 큐에 넣지 않는다(연구 사본, 코드 사본만 preds/code 에).
        --method_dir 를 주면 preds/ 에 쓰되 방법 json 만 그 폴더에 쓰고 큐에 넣지 않는다(큐 밖에서 준비해 사람이 옮길 때).
    .venv/Scripts/python.exe -X utf8 tools/glong.py --setting {val,fwd,full} --src <원 manifest> --src_sha <16> --tag <새 spec> [--outdir <폴더>]
        manifest 의 commands 가 적는 재현 명령. 그 설정의 파일 하나만 다시 쓴다.
    .venv/Scripts/python.exe -X utf8 tools/glong.py enqueue <새 큐 이름>
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
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
KEY = {"val": "holdout", "fwd": "forward", "full": "test"}
N_TEST = 344841
SUFFIX = "gl"
D0 = 30000.0
FITM = {"val": [2, 3, 4, 5, 6, 8, 9, 10, 11, 12], "fwd": [1, 2, 3, 4, 5, 6], "full": list(range(1, 13))}
CODE_SAME = ("tools/gkey.py", "tools/gkeyx.py", "tools/gkeyxn.py", "tools/gpool.py", "src/model3.py", "src/sub_g.py")
BASIS = "vault research/2026-10-07-r10-d-types.md 3 (R10-D-G1, pre-registered 05:22 KST 10-07)"


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha16(rel: str | Path) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:16]


def now_kst() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")


def truth() -> pd.Series:
    T = pq.read_table(ROOT / "cache" / "all.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "src", "y"]).to_pandas()
    T = T[(T["PHASE_mvt"].astype(str) == "DEP") & (T["src"] > 0)]
    return T.set_index("MVT_ID_mvt")["y"].astype(float)


def load_man(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def load_meta(arg: str) -> tuple[dict, str | None]:
    if arg.endswith(".json"):
        return {"manifest": arg}, None
    for q in (ROOT / "queue", ROOT / "queue" / "done"):
        p = q / f"{arg}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")), arg
    raise SystemExit(f"큐에 {arg} 이 없다(queue/, queue/done/)")


def pred(rel: str) -> pd.Series:
    return pd.read_parquet(ROOT / rel).set_index("MVT_ID_mvt")["pred"].astype(float)


def chain_links(sm: dict) -> list[dict]:
    links = [sm]
    for _ in range(60):
        nxt = links[-1].get("source", {}).get("manifest")
        if not nxt:
            break
        links.append(load_man(nxt))
    return links


def stages(sm: dict) -> tuple[dict, dict, dict, dict]:
    """원 사슬에서 gpool 단계, 그 원본(반대 창 판), 그 반대 창 단계의 원본(반대 창 앞 판), gkeyxn 단계 manifest."""
    links = chain_links(sm)
    i = next((k for k, x in enumerate(links) if "gpool" in x), None)
    if i is None:
        raise SystemExit("원 사슬에 gpool 단계가 없다(이 단계는 gpool 위에서 잰 규칙이다)")
    gp, aw = links[i], links[i + 1]
    if not aw["spec"].endswith("aw"):
        raise SystemExit("gpool 단계의 원본이 반대 창 판이 아니다")
    pre_aw = links[i + 2]
    gk = next((x for x in links[i:] if x.get("config", {}).get("--variant", "").startswith("avg_xn")), None)
    if gk is None or gk["config"]["--variant"] != "avg_xncr":
        raise SystemExit("원 사슬에 gkeyxn avg_xncr 단계가 없다")
    for f in CODE_SAME:
        if f in sm["code_sha"] and sha16(f) != sm["code_sha"][f]:
            raise SystemExit(f"{f} 가 원 후보 manifest 의 사본과 다르다")
    return gp, aw, pre_aw, gk


class Step:
    def __init__(self, gp: dict, gk: dict):
        self.GP = load_tool("gpool")
        self.M = self.GP.GModel(gk["config"])
        self.GK, self.GX = self.M.GK, self.M.GX
        self.cfg = gk["config"]
        self.a = {st: float(gp["gpool"]["a"][st]["a"]) for st in ("val", "fwd", "full")}
        import src.sub_g as sub_g
        from src.model3 import g_override
        self.sub_g, self.g_override = sub_g, g_override

    def p_av(self, fm, em):
        M = self.M
        return 0.5 * (self.GK.probs(M.H1[fm], M.H1[em], []) + self.GK.probs(M.H2[fm], M.H2[em], []))

    def chain_p(self, fm, em, a):
        M = self.M
        C = M.comps(fm, em)
        P = self.GP.logpool(C, a)
        g = M.d[em] >= M.CUT
        P[g] = self.p_av(fm, em)[g]
        return P, C

    def fit_r(self, setting: str) -> dict:
        M, months = self.M, FITM[setting]
        k, E, n = 0, 0.0, 0
        for j in months:
            fm = np.isin(M.s, months) & (M.s != j) & M.fin
            em = (M.s == j) & M.fin
            P, _ = self.chain_p(fm, em, self.a[setting])
            band = M.d[em] >= D0
            lab = self.sub_g.labels(M.y[em], M.d[em])
            k += int((lab[band] == 0).sum())
            E += float(P[band, 0].sum())
            n += int(band.sum())
        return {"r": min(1.0, (k + 0.5) / (E + 0.5)), "k": k, "E": round(E, 6), "n": n, "months": months}

    @staticmethod
    def apply_r(P: np.ndarray, d: np.ndarray, r: float) -> np.ndarray:
        pn, pa, ps = P[:, 0], P[:, 1], P[:, 2]
        band = d >= D0
        tot = pa + ps
        wa = np.where(tot > 0, pa / np.where(tot > 0, tot, 1), 0.5)
        free = (1 - r) * pn
        return np.column_stack([np.where(band, r * pn, pn), np.where(band, pa + free * wa, pa), np.where(band, ps + free * (1 - wa), ps)])

    def delta(self, setting: str, gp_file: pd.Series, gp_src: pd.Series, aw_rows: set) -> tuple[pd.Series, dict]:
        """그 설정 평가 G 행의 g' − g(대상 행 밖 0)와 재현 차."""
        M = self.M
        em = self.GK.EVAL[setting](M.s)
        fm = self.GK.FIT[setting](M.s) & M.fin
        gids = M.ids[em]
        d, dd = M.d[em], M.dd[em]
        P, C = self.chain_p(fm, em, self.a[setting])
        b = pd.read_parquet(ROOT / "cache" / f"{setting}_{self.cfg['--leg']}_s0.parquet", columns=["MVT_ID_mvt", "mu", "p_art"]).set_index("MVT_ID_mvt").reindex(gids)
        mu_g, pa_g = b["mu"].to_numpy(float), b["p_art"].to_numpy(float)
        g0 = np.clip(self.g_override(P, d, dd, mu_g, pa_g)[0], 0, None)
        guard = d >= M.CUT
        # 재현 1: 가드 행 = gkeyx g_new
        T, X, _ = self.M.GN.load_tables(self.cfg)
        V = self.GX.gx_values(setting, self.cfg, T, X)
        err_g = float(np.max(np.abs(g0[guard] - V["g_new"].reindex(gids).to_numpy(float)[guard]))) if guard.any() else 0.0
        # 재현 2: gpool 이동(S1 − 선형 풀) = gpool 단계 파일 − 그 원본(반대 창 행, 가드 행 빼고)
        g_lin = np.clip(self.g_override(self.GP.lin(C), d, dd, mu_g, pa_g)[0], 0, None)
        src_g = gp_src.reindex(gids).to_numpy(float)
        gpf = gp_file.reindex(gids).to_numpy(float)
        awg = np.array([i in aw_rows for i in gids])
        chk = ~guard & ~awg
        want = np.clip(src_g + (g0 - g_lin), 0, None)  # gpool 은 원본 + 이동을 0 에서 자른다
        err_p = float(np.max(np.abs(gpf[chk] - want[chk]))) if chk.any() else 0.0
        if not (err_g < 1e-3 and err_p < 1e-3):
            raise SystemExit(f"{setting}: 사슬 G 확률 재현 실패(가드 {err_g:.3e}, gpool 이동 {err_p:.3e})")
        fr = self.fit_r(setting)
        P1 = self.apply_r(P, d, fr["r"])
        g1 = np.clip(self.g_override(P1, d, dd, mu_g, pa_g)[0], 0, None)
        dl = pd.Series(np.where(d >= D0, g1 - g0, 0.0), index=gids)
        tgt = d >= D0
        info = {"fit": fr, "a": self.a[setting], "g_rows": int(len(gids)), "target_rows": int(tgt.sum()), "guard_rows": int(guard.sum()),
                "repro_err_guard_vs_gkeyx": float(f"{err_g:.3e}"), "repro_err_gpool_move": float(f"{err_p:.3e}"),
                "mean_pn_target": round(float(P[tgt, 0].mean()), 5) if tgt.any() else None,
                "mean_pn_target_new": round(float(P1[tgt, 0].mean()), 5) if tgt.any() else None}
        return dl, info


def apply_setting(setting: str, sm: dict, dl: pd.Series, aw_rows: set) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    old = pd.read_parquet(ROOT / sm[KEY[setting]]["path"])
    f = old.set_index("MVT_ID_mvt")["pred"].astype(float)
    dlt = dl.reindex(f.index).fillna(0.0).to_numpy()
    awm = f.index.isin(list(aw_rows))
    ch = (dlt != 0) & ~awm
    nv = f.to_numpy().copy()
    nv[ch] = np.clip(nv[ch] + dlt[ch], 0, None)
    new_s = pd.Series(nv, index=f.index)
    outside = int((~f.index.isin(list(dl.index[dl.to_numpy() != 0])) & (nv != f.to_numpy())).sum())
    if outside:
        raise SystemExit(f"{setting}: 대상 G 행 밖 {outside} 행이 바뀌었다")
    new = old.copy()
    new["pred"] = new_s.reindex(old["MVT_ID_mvt"].to_numpy()).to_numpy()
    info = {"target_rows_moving": int((dl != 0).sum()), "antiwin_rows_kept": int((awm & (dlt != 0)).sum()), "changed": int((nv != f.to_numpy()).sum()),
            "move_mse_rows_n": round(float(np.mean((nv - f.to_numpy()) ** 2)), 3)}
    return new, f, new_s, info


def delta_mse(old: pd.Series, new: pd.Series, y: pd.Series) -> dict:
    yy = y.reindex(old.index).to_numpy()
    ok = np.isfinite(yy)
    o, n, yy = old.to_numpy()[ok], new.reindex(old.index).to_numpy()[ok], yy[ok]
    se0, se1 = (o - yy) ** 2, (n - yy) ** 2
    N = len(yy)
    ch = n != o
    return {"rows": int(N), "rmse_old": round(float(np.sqrt(se0.mean())), 3), "rmse_new": round(float(np.sqrt(se1.mean())), 3),
            "dmse": round(float((se1.sum() - se0.sum()) / N), 2), "changed": int(ch.sum()), "improved": int((se1 < se0)[ch].sum())}


def prepare(sm: dict) -> tuple[Step, dict, dict, set]:
    gp, aw, pre_aw, gk = stages(sm)
    st = Step(gp, gk)
    aws = {}
    for setting in ("val", "fwd", "full"):
        fa, fp = pred(aw[KEY[setting]]["path"]), pred(pre_aw[KEY[setting]]["path"]).reindex(pred(aw[KEY[setting]]["path"]).index)
        aws[setting] = set(fa.index[np.abs(fa.to_numpy() - fp.to_numpy()) > 1e-6])
        # 원 후보의 G 행이 gpool 단계 파일의 G 행과 같아야 한다(뒤 단계가 G 행을 바꾸지 않았다)
        f_src, f_gp = pred(sm[KEY[setting]]["path"]), pred(gp[KEY[setting]]["path"])
        gids = st.M.ids[st.GK.EVAL[setting](st.M.s)]
        if not np.array_equal(f_src.reindex(gids).to_numpy(), f_gp.reindex(gids).to_numpy()):
            raise SystemExit(f"{setting}: 원 후보의 G 행이 gpool 단계 파일과 다르다(gpool 뒤에 G 행을 바꾼 단계가 있다)")
    return st, gp, aw, aws


def run(sm: dict, settings=("full", "val", "fwd")) -> dict:
    st, gp, aw, aws = prepare(sm)
    out = {}
    for s_ in settings:
        dl, info = st.delta(s_, pred(gp[KEY[s_]]["path"]), pred(aw[KEY[s_]]["path"]), aws[s_])
        new, f, new_s, ainfo = apply_setting(s_, sm, dl, aws[s_])
        out[s_] = {"new": new, "old_s": f, "new_s": new_s, "info": {**info, **ainfo}}
        print(f"  {s_}: r {info['fit']['r']:.4f}(k {info['fit']['k']}, E {info['fit']['E']:.3f}, n {info['fit']['n']}), a {info['a']:.4f}, "
              f"재현 차 가드 {info['repro_err_guard_vs_gkeyx']:.1e} / gpool {info['repro_err_gpool_move']:.1e}, 대상 {info['target_rows']}, 바뀐 행 {ainfo['changed']}, "
              f"pn {info['mean_pn_target']} → {info['mean_pn_target_new']}", flush=True)
    return out


def out_rel(setting: str, tag: str, new_s: pd.Series, y: pd.Series | None, outdir: str) -> str:
    base = outdir.rstrip("/") if outdir else "preds"
    if setting == "val":
        rm = float(np.sqrt(((new_s - y.reindex(new_s.index)) ** 2).mean()))
        return f"{base}/holdout_{tag}_rmse{rm:.2f}.parquet"
    return f"{base}/fwd_{tag}.parquet" if setting == "fwd" else f"{base}/test_{tag}.parquet"


def check(arg: str) -> int:
    from src.batches import manifest_of
    meta, _ = load_meta(arg)
    sm, why = manifest_of(meta)
    if why:
        print("거부: 원 후보 manifest 검사", why)
        return 1
    y = truth()
    R = run(sm)
    for s_ in ("val", "fwd"):
        print(f"[check {arg}] {s_}: {delta_mse(R[s_]['old_s'], R[s_]['new_s'], y)}", flush=True)
    print(f"[check {arg}] full: 시험 이동 MSE {float(((R['full']['new_s'] - R['full']['old_s']) ** 2).sum() / N_TEST):.1f}", flush=True)
    return 0


def build(arg: str, num: str, outdir: str, method_dir: str = "") -> int:
    from src.batches import manifest_of
    from src.ruler import ruler_hash
    meta, stem = load_meta(arg)
    sm, why = manifest_of(meta)
    if why:
        print("거부: 원 후보 manifest 검사", why)
        return 1
    src_mp = meta["manifest"]
    tag = sm["spec"] + SUFFIX
    name = f"{num}_{tag}"
    mdir = outdir.rstrip("/") if outdir else "preds"
    mp = f"{mdir}/manifest_{tag}.json"
    if (ROOT / mp).exists() or (not outdir and (ROOT / "queue" / f"{name}.json").exists()):
        print("거부: 같은 이름이 이미 있다", mp)
        return 1
    if outdir:
        (ROOT / mdir).mkdir(parents=True, exist_ok=True)
    y = truth()
    R = run(sm)
    files, rep = {}, {}
    for s_ in ("val", "fwd", "full"):
        r = R[s_]
        rel = out_rel(s_, tag, r["new_s"], y if s_ == "val" else None, outdir)
        r["new"].to_parquet(ROOT / rel, index=False)
        files[s_] = rel
        if s_ == "full":
            rep[s_] = {**{k: v for k, v in r["info"].items()}, "move_mse_test_n": round(float(((r["new_s"] - r["old_s"]) ** 2).sum() / N_TEST), 2),
                       "share_over_3600_old": round(float((r["old_s"] > 3600).mean()), 6), "share_over_3600_new": round(float((r["new_s"] > 3600).mean()), 6)}
        else:
            rep[s_] = {**r["info"], **delta_mse(r["old_s"], r["new_s"], y)}
        print(f"[{name}] {s_}: {json.dumps({k: v for k, v in rep[s_].items() if k != 'fit'}, ensure_ascii=False)}", flush=True)
    es_rel = f"{mdir}/es_valid_ids_{tag}.parquet"
    shutil.copy2(ROOT / sm["es_valid_ids"]["path"], ROOT / es_rel)
    me = "tools/glong.py"
    h = sha16(me)
    snap = f"preds/code/{h}_glong.py"
    if not (ROOT / snap).exists():
        shutil.copy2(ROOT / me, ROOT / snap)
    cfg = {"--src": src_mp, "--src_sha": sha16(src_mp), "--tag": tag}
    if outdir:
        cfg["--outdir"] = mdir
    flags = " ".join(f"{k} {v}" for k, v in cfg.items())
    cmd = lambda s: f".venv/Scripts/python.exe -X utf8 tools/glong.py --setting {s} {flags}"  # noqa: E731
    hold_ids = pd.read_parquet(ROOT / files["val"])["MVT_ID_mvt"].to_numpy()
    gp, _, _, gk = stages(sm)
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
        "iteration_rule": sm.get("iteration_rule", "") + " | glong: on G rows with d_sch >= 30,000 s the chain's normal-class probability is scaled by "
                          "r = min(1, (k + 0.5)/(E + 0.5)) (k normal labels, E summed out-of-month chain probabilities on the setting's fit-month rows "
                          "with d_sch >= 30,000), the freed mass split to art and day push in their ratio",
        "source": {"queue": stem, "manifest": src_mp, "manifest_sha": cfg["--src_sha"], "spec": sm["spec"],
                   "holdout_sha": sm["holdout"]["sha"], "test_sha": sm["test"]["sha"], "forward_sha": sm["forward"]["sha"]},
        "glong": {"gpool_spec": gp["spec"], "gkeyxn_spec": gk["spec"], "d0": D0, "report": rep, "basis": BASIS},
        "written_kst": now_kst(),
    }
    (ROOT / mp).write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _, why = manifest_of({"manifest": mp})
    if why:
        print("거부: 새 manifest 검사", why)
        return 1
    print(f"[{name}] manifest {mp}(manifest_of 통과)", flush=True)
    method = (f"{stem or src_mp} 의 G 행(LIRF NM 미매칭) 가운데 d_sch ≥ 30,000초 행의 정상 확률을 r 배로(비운 몫은 art 와 shift 에 그 비로) 보정한 판"
              f"(tools/glong.py, G 행만, 반대 창과 gpool 뒤 맨 끝). r 는 그 설정 맞춤 달의 같은 행에서 (정상 수 + 0.5)/(밖 표본 정상 확률 합 + 0.5)"
              f"(val {rep['val']['fit']['r']:.3f}, fwd {rep['fwd']['fit']['r']:.3f}, 시험 {rep['full']['fit']['r']:.3f}). 근거 {BASIS}: 2025 G 의 d ≥ 30,000 48행에 정상 0. "
              f"자 ΔMSE {rep['val']['dmse']}({rep['val']['improved']}/{rep['val']['changed']}), 7월 전방 ΔMSE {rep['fwd']['dmse']}"
              f"({rep['fwd']['improved']}/{rep['fwd']['changed']}), 시험 이동 {rep['full']['move_mse_test_n']} MSE(G {rep['full']['changed']} 행). "
              "반대 창 행, d < 30,000 G 행, G 아닌 행은 그대로. G 행만 바꾸므로 홀로 쏘지 않고 바탕이 바뀐 판과 함께 낸다. 10-05 단일 발 체제라 준비 점검은 경고(--readiness-warn)")
    nrel = f"{method_dir.rstrip('/')}/glong_{name}.json" if method_dir else (f"{mdir}/glong_{name}.json" if outdir else f"notes/glong_{name}.json")
    (ROOT / nrel).write_text(json.dumps({"name": name, "manifest": mp, "source": stem or src_mp, "method": method, "report": rep, "written_kst": now_kst()},
                                        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[{name}] 방법 json {nrel}", flush=True)
    return 0


def enqueue(name: str) -> int:
    info = json.loads((ROOT / "notes" / f"glong_{name}.json").read_text(encoding="utf-8"))
    r = subprocess.run([sys.executable, "-X", "utf8", "tools/enqueue.py", info["manifest"], name, info["method"], "--readiness-warn"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1500:], r.stderr[-800:], flush=True)
    return r.returncode


def main() -> int:
    load_tool("gpool").below_normal()
    a = sys.argv[1:]
    outdir = a[a.index("--outdir") + 1] if "--outdir" in a else ""
    method_dir = a[a.index("--method_dir") + 1] if "--method_dir" in a else ""
    if a and a[0] == "check":
        return check(a[1])
    if a and a[0] == "_files":
        return build(a[1], a[2], outdir, method_dir)
    if a and a[0] == "enqueue":
        return enqueue(a[1])
    if a and a[0] == "build":
        if len(a) < 3:
            print("usage: glong.py build <원 큐 이름 또는 manifest> <새 큐 번호> [--no-enqueue] [--outdir <폴더>]")
            return 2
        me = [sys.executable, "-X", "utf8", str(Path(__file__).resolve())]
        rc = subprocess.run(me + ["_files", a[1], a[2]] + (["--outdir", outdir] if outdir else []) + (["--method_dir", method_dir] if method_dir else []), cwd=ROOT).returncode
        if rc != 0 or "--no-enqueue" in a or outdir or method_dir:
            return rc
        meta, _ = load_meta(a[1])
        spec = load_man(meta["manifest"])["spec"]
        return subprocess.run(me + ["enqueue", f"{a[2]}_{spec}{SUFFIX}"], cwd=ROOT).returncode
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--seed", default="0")
    pa.add_argument("--src", required=True)
    pa.add_argument("--src_sha", required=True)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--outdir", default="")
    g = pa.parse_args(a)
    if sha16(g.src) != g.src_sha:
        print("거부: 원 manifest 해시가 다르다")
        return 1
    sm = load_man(g.src)
    R = run(sm, settings=(g.setting,))
    r = R[g.setting]
    rel = out_rel(g.setting, g.tag, r["new_s"], truth() if g.setting == "val" else None, g.outdir)
    r["new"].to_parquet(ROOT / rel, index=False)
    print(rel, {k: v for k, v in r["info"].items() if k != "fit"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
