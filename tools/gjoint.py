# SPDX-License-Identifier: GPL-3.0-only
"""G 하위 모형 다시 맞춤(R11-G). glong(G1) 단계 뒤 맨 끝에 얹는다. G 행(LIRF 출발, NM 미매칭)만 바꾼다.
근거: vault `01 Projects/공모전 파이프라인/착수/PRC Data Challenge 2026/research/2026-10-07-r11-g.md`(측정 research/r11_g/). 남의 코드 없음.

판(--mode):
  a  다리 p_art 띠 평균 빼기(사전 등록 R11-G-A, 06:47 KST 10-07, 새 달 열 자에서 넘음: 2 ~ 6, 8 ~ 12월 lomo ΔSSE −3.06%, 날 다시 뽑기
     P(Δ<0) 0.9995, 나은 달 9/10, 상수 mu 확인 −2.60%). 사슬의 G 값(src.model3.g_override)은 d_sch ∈ [3,600, 14,400) 행의 art 확률을
     0.5 G 모형 + 0.5 다리 p_art 로 쓴다(10-03 도입). 이 판은 그 행에서 G 모형 확률(gpool S1, G1 뒤)을 그대로 쓴다. 성분(정상 mu_g,
     shift 의 택시 clip(mu_g, 300, 3,000))은 사슬 그대로다.
  c  a + 정상 몫을 G 전용 회귀로(사후 판, 후보 아님): 정상 = mu_n(맞춤 달 G 정상 tz 아닌 행의 가법 수축 회귀, y 6,000 자름), tz 정상 =
     mu_n + 로마 UTC 차(운항사 수축 확률 τ_c), shift 택시 clip(mu_n, 300, 3,000). 사전 등록 J1(이 성분 + 가중 다시 맞춤)은 fwd 에서 졌다.
규칙(a): g_A = clip(P1_a d + P1_s (86,400 max(dd, 1) + clip(mu_g, 300, 3,000)) + P1_n mu_g, 0), g = 사슬의 G1 뒤 G 값
  clip(g_override(P1, d, dd, mu_g, pa_g), 0)(tools/glong 의 g1). 새 예측 = clip(원 예측 + g_A − g, 0). 띠 밖 행은 g_A = g 라 그대로다.
  반대 창이 바꾼 행, G 아닌 행은 바이트까지 그대로(확인한다). 다리 p_art 의 띠 몫 0.5 pa_g (d − mu_g) 는 P 를 바꾸는 G 단계의 차에서 지워지므로
  원 파일에 그대로 남아 있다(research/r11_g 진단: 파일과 다시 셈한 g 의 차가 그 몫과 상관 0.02 ~ 0.10). 이 판이 그 몫을 뺀다.
확인: glong 을 원 사슬(gpool 판)에 다시 돌려 glong 단계 파일과 값이 같아야 하고(사슬 확률 재현은 glong 이 가드 행과 gpool 이동으로 확인한다),
  입력 사슬의 G 행이 glong 단계 파일의 G 행과 같아야 한다(glong 뒤에 G 행을 바꾼 단계가 없다). G 아닌 행만 바꾼 단계는 앞뒤 어디에 있어도 된다.
G1 과의 짜임: G1(d ≥ 30,000 정상 확률 r 배)은 띠 밖이라 a 판과 겹치는 행이 없다. a 판은 G1 이 끝난 확률을 입력으로 쓴다.
496 꼴(두 사슬 반씩)은 494(491 + G1), 495(492 + G1)에 각각 짓고 tools/chainavg.py 식으로 섞는다.

쓰는 법(저장소에서, memcap 2.5 GB 안):
    .venv/Scripts/python.exe -X utf8 tools/gjoint.py check <큐 이름 또는 manifest> [--mode a]
        세 설정의 원 후보 대비 ΔMSE(val, fwd), 시험 이동만 찍는다(아무 파일도 쓰지 않는다).
    .venv/Scripts/python.exe -X utf8 tools/gjoint.py build <큐 이름 또는 manifest> <새 큐 번호> [--mode a] [--no-enqueue] [--outdir <폴더>] [--method_dir <폴더>]
        새 파일 셋과 manifest 를 쓰고(새 spec = 원 spec + "gj" + 판), notes/gjoint_<새 큐 이름>.json 을 남긴 뒤 tools/enqueue.py --readiness-warn 으로 넣는다.
        --outdir 는 예측과 manifest 를 그 폴더에(큐에 넣지 않음), --method_dir 는 preds/ 에 쓰되 방법 json 만 그 폴더에(큐에 넣지 않음).
    .venv/Scripts/python.exe -X utf8 tools/gjoint.py --setting {val,fwd,full} --src <원 manifest> --src_sha <16> --tag <새 spec> --mode a [--outdir <폴더>]
        manifest 의 commands 가 적는 재현 명령. 그 설정의 파일 하나만 다시 쓴다.
    .venv/Scripts/python.exe -X utf8 tools/gjoint.py enqueue <새 큐 이름>
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
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
KEY = {"val": "holdout", "fwd": "forward", "full": "test"}
N_TEST = 344841
MODES = ("a", "c")
FITM = {"val": [2, 3, 4, 5, 6, 8, 9, 10, 11, 12], "fwd": [1, 2, 3, 4, 5, 6], "full": list(range(1, 13))}
CODE_SAME = ("tools/gkey.py", "tools/gkeyx.py", "tools/gkeyxn.py", "tools/gpool.py", "tools/glong.py", "src/model3.py", "src/sub_g.py")
BASIS = {"a": "vault research/2026-10-07-r11-g.md 4 (R11-G-A, pre-registered 06:47 KST 10-07, fresh 10-month ruler passed)",
         "c": "vault research/2026-10-07-r11-g.md 3 (components-only, post-hoc report variant; not pre-registered)"}
# c 판의 정상 회귀(research/r11_g/jm.py 와 같은 값)
HB = [6, 9, 12, 15, 18, 21]
DB = [0, 1800, 3600, 7200, 14400]
ALPHA = {"car": 10.0, "hb": 20.0, "sea": 30.0, "db": 20.0, "ddf": 20.0}
YCAP, TAU_A, TZ_LO, TZ_HI = 6000.0, 5.0, 200.0, 2400.0


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


def gl_stage(sm: dict, GL) -> tuple[dict, dict]:
    """입력 사슬에서 glong 단계 manifest 와 그 원본(gpool 판) manifest."""
    links = GL.chain_links(sm)
    i = next((k for k, x in enumerate(links) if "glong" in x), None)
    if i is None:
        raise SystemExit("입력 사슬에 glong 단계가 없다(이 단계는 G1 뒤에 얹는 규칙이다)")
    for f in CODE_SAME:
        if f in sm["code_sha"] and sha16(f) != sm["code_sha"][f]:
            raise SystemExit(f"{f} 가 입력 후보 manifest 의 사본과 다르다")
    return links[i], links[i + 1]


class Joint:
    def __init__(self, sm: dict, mode: str):
        if mode not in MODES:
            raise SystemExit(f"판 {mode} 이 없다({MODES})")
        self.mode = mode
        self.GL = load_tool("glong")
        self.sm = sm
        self.gl, self.src = gl_stage(sm, self.GL)
        self.st, self.gp, self.aw, self.aws = self.GL.prepare(self.src)
        from src.model3 import G_BAND, g_override
        self.G_BAND, self.g_override = G_BAND, g_override
        M = self.st.M
        self.leg = self.st.cfg["--leg"]
        if mode == "c":
            self.frame()

    def verify(self, settings) -> dict:
        """glong 다시 돌리기 = glong 단계 파일(값), 입력 사슬 G 행 = glong 단계 파일 G 행."""
        R = self.GL.run(self.src, settings=settings)
        out = {}
        M = self.st.M
        for s_ in settings:
            f_gl = pred(self.gl[KEY[s_]]["path"])
            re = R[s_]["new_s"].reindex(f_gl.index).to_numpy()
            if not np.array_equal(re, f_gl.to_numpy()):
                raise SystemExit(f"{s_}: glong 을 다시 돌린 값이 glong 단계 파일과 다르다(최대 차 {float(np.nanmax(np.abs(re - f_gl.to_numpy()))):.3e})")
            gids = M.ids[self.st.GK.EVAL[s_](M.s)]
            f_in = pred(self.sm[KEY[s_]]["path"])
            if not np.array_equal(f_in.reindex(gids).to_numpy(), f_gl.reindex(gids).to_numpy()):
                raise SystemExit(f"{s_}: 입력 사슬의 G 행이 glong 단계 파일과 다르다(glong 뒤에 G 행을 바꾼 단계가 있다)")
            out[s_] = {"glong_rebuilt_equal": True, "input_g_rows_equal_glong": True, "r": R[s_]["info"]["fit"]["r"]}
        return out

    # ---- c 판 재료 ----
    def frame(self) -> None:
        M = self.st.M
        t = ds.dataset(ROOT / "cache" / "all.parquet").to_table(columns=["MVT_ID_mvt", "t_mvt"], filter=(pc.field("PHASE_mvt") == "DEP") & (pc.field("ADEP_mvt") == "LIRF"))
        tm = t.to_pandas().drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")["t_mvt"].reindex(M.ids).to_numpy(float)
        tsu = pd.to_datetime(tm, unit="s", utc=True)
        loc = tsu.tz_convert("Europe/Rome").tz_localize(None)
        off = np.asarray((loc - tsu.tz_localize(None)).total_seconds(), dtype=float)
        y, d = M.y, M.d
        art = M.fin & (np.abs(y - d) <= 60)
        shift = M.fin & (y > 60000) & ~art
        normal = M.fin & ~art & ~shift
        tz = normal & (y - off >= TZ_LO) & (y - off <= TZ_HI)
        self.F = pd.DataFrame({"s": M.s, "y": y, "off": off, "car": M.H1["k_car"].astype(str).to_numpy(),
                               "hb": np.digitize((loc.hour + loc.minute / 60.0).to_numpy(), HB), "sea": off.astype(int),
                               "db": np.digitize(d, DB), "ddf": (M.dd >= 1).astype(int), "normal": normal, "tz": tz, "nontz": normal & ~tz},
                              index=M.ids)

    def fit_mu(self, months):
        T = self.F[self.F["s"].isin(months) & self.F["nontz"]]
        y = T["y"].to_numpy(float)
        yw = np.minimum(y, YCAP)
        cw = float(np.mean(y - yw))
        m = float(yw.mean())
        fac = list(ALPHA)
        con = {f: np.zeros(len(T)) for f in fac}
        eff = {}
        for _ in range(10):
            for f in fac:
                r = yw - m - sum(con[g] for g in fac if g != f)
                g = pd.DataFrame({"k": T[f].to_numpy(), "r": r}).groupby("k")["r"].agg(["sum", "count"])
                eff[f] = g["sum"] / (g["count"] + ALPHA[f])
                con[f] = eff[f].reindex(T[f].to_numpy()).to_numpy()
            m = float(np.mean(yw - sum(con[f] for f in fac)))
        return m, eff, cw

    def mu_tau(self, setting, gids):
        m, eff, cw = self.fit_mu(FITM[setting])
        R = self.F.loc[gids]
        mu = np.full(len(R), m + cw)
        for f in ALPHA:
            mu += eff[f].reindex(R[f].to_numpy()).fillna(0.0).to_numpy()
        mu = np.clip(mu, 200.0, 8000.0)
        T = self.F[self.F["s"].isin(FITM[setting]) & self.F["normal"]]
        g = pd.DataFrame({"c": T["car"].to_numpy(), "t": T["tz"].to_numpy(float)}).groupby("c")["t"].agg(["sum", "count"])
        t0 = float(T["tz"].mean())
        tc = g["sum"].reindex(R["car"].to_numpy()).fillna(0).to_numpy()
        mc = g["count"].reindex(R["car"].to_numpy()).fillna(0).to_numpy()
        return mu, (tc + TAU_A * t0) / (mc + TAU_A), R["off"].to_numpy(float)

    def delta(self, setting: str) -> tuple[pd.Series, dict]:
        """그 설정 평가 G 행의 g_new − g(사슬 G1 뒤 G 값)."""
        st, M = self.st, self.st.M
        em = st.GK.EVAL[setting](M.s)
        fm = st.GK.FIT[setting](M.s) & M.fin
        gids = M.ids[em]
        d, dd = M.d[em], M.dd[em]
        P, _ = st.chain_p(fm, em, st.a[setting])
        fr = st.fit_r(setting)
        P1 = st.apply_r(P, d, fr["r"])
        b = pd.read_parquet(ROOT / "cache" / f"{setting}_{self.leg}_s0.parquet", columns=["MVT_ID_mvt", "mu", "p_art"]).set_index("MVT_ID_mvt").reindex(gids)
        mu_g, pa_g = b["mu"].to_numpy(float), b["p_art"].to_numpy(float)
        g = np.clip(self.g_override(P1, d, dd, mu_g, pa_g)[0], 0, None)
        band = (d >= self.G_BAND[0]) & (d < self.G_BAND[1])
        if self.mode == "a":
            S = 86400.0 * np.maximum(dd, 1) + np.clip(mu_g, 300.0, 3000.0)
            gn = np.clip(P1[:, 1] * d + P1[:, 2] * S + P1[:, 0] * mu_g, 0, None)
            out_band = float(np.max(np.abs((gn - g)[~band]))) if (~band).any() else 0.0
            if out_band > 1e-6:
                raise SystemExit(f"{setting}: 띠 밖 행에서 a 판 G 값이 사슬과 다르다({out_band:.3e})")
            extra = {}
        else:
            mu_n, tau, off = self.mu_tau(setting, gids)
            S = 86400.0 * np.maximum(dd, 1) + np.clip(mu_n, 300.0, 3000.0)
            gn = np.clip(P1[:, 1] * d + P1[:, 2] * S + P1[:, 0] * (mu_n + tau * off), 0, None)
            extra = {"mu_n_mean": round(float(mu_n.mean()), 1), "tau_mean": round(float(tau.mean()), 5)}
        dl = pd.Series(gn - g, index=gids)
        info = {"mode": self.mode, "g_rows": int(len(gids)), "band_rows": int(band.sum()), "r": round(float(fr["r"]), 6), "a": st.a[setting],
                "mean_pa_chain_band": round(float((0.5 * P1[band, 1] + 0.5 * pa_g[band]).mean()), 5) if band.any() else None,
                "mean_pa_new_band": round(float(P1[band, 1].mean()), 5) if band.any() else None, **extra}
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
    if int((awm & (nv != f.to_numpy())).sum()):
        raise SystemExit(f"{setting}: 반대 창 행이 바뀌었다")
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


def run(sm: dict, mode: str, settings=("full", "val", "fwd")) -> dict:
    J = Joint(sm, mode)
    ver = J.verify(settings)
    out = {}
    for s_ in settings:
        dl, info = J.delta(s_)
        new, f, new_s, ainfo = apply_setting(s_, sm, dl, J.aws[s_])
        out[s_] = {"new": new, "old_s": f, "new_s": new_s, "info": {**info, **ainfo, **ver[s_]}}
        print(f"  {s_}: {json.dumps(out[s_]['info'], ensure_ascii=False)}", flush=True)
    return out


def out_rel(setting: str, tag: str, new_s: pd.Series, y: pd.Series | None, outdir: str) -> str:
    base = outdir.rstrip("/") if outdir else "preds"
    if setting == "val":
        rm = float(np.sqrt(((new_s - y.reindex(new_s.index)) ** 2).mean()))
        return f"{base}/holdout_{tag}_rmse{rm:.2f}.parquet"
    return f"{base}/fwd_{tag}.parquet" if setting == "fwd" else f"{base}/test_{tag}.parquet"


def opt(a: list, k: str, default: str = "") -> str:
    return a[a.index(k) + 1] if k in a else default


def check(arg: str, mode: str) -> int:
    from src.batches import manifest_of
    meta, _ = load_meta(arg)
    sm, why = manifest_of(meta)
    if why:
        print("거부: 원 후보 manifest 검사", why)
        return 1
    y = truth()
    R = run(sm, mode)
    for s_ in ("val", "fwd"):
        print(f"[check {arg} {mode}] {s_}: {delta_mse(R[s_]['old_s'], R[s_]['new_s'], y)}", flush=True)
    print(f"[check {arg} {mode}] full: 시험 이동 MSE {float(((R['full']['new_s'] - R['full']['old_s']) ** 2).sum() / N_TEST):.2f}", flush=True)
    return 0


def build(arg: str, num: str, mode: str, outdir: str, method_dir: str = "") -> int:
    from src.batches import manifest_of
    from src.ruler import ruler_hash
    meta, stem = load_meta(arg)
    sm, why = manifest_of(meta)
    if why:
        print("거부: 원 후보 manifest 검사", why)
        return 1
    src_mp = meta["manifest"]
    tag = sm["spec"] + "gj" + mode
    name = f"{num}_{tag}"
    mdir = outdir.rstrip("/") if outdir else "preds"
    mp = f"{mdir}/manifest_{tag}.json"
    if (ROOT / mp).exists() or (not outdir and (ROOT / "queue" / f"{name}.json").exists()):
        print("거부: 같은 이름이 이미 있다", mp)
        return 1
    if outdir:
        (ROOT / mdir).mkdir(parents=True, exist_ok=True)
    y = truth()
    R = run(sm, mode)
    files, rep = {}, {}
    for s_ in ("val", "fwd", "full"):
        r = R[s_]
        rel = out_rel(s_, tag, r["new_s"], y if s_ == "val" else None, outdir)
        if (ROOT / rel).exists():
            print("거부: 같은 예측 파일이 이미 있다", rel)
            return 1
        r["new"].to_parquet(ROOT / rel, index=False)
        files[s_] = rel
        if s_ == "full":
            rep[s_] = {**r["info"], "move_mse_test_n": round(float(((r["new_s"] - r["old_s"]) ** 2).sum() / N_TEST), 2),
                       "share_over_3600_old": round(float((r["old_s"] > 3600).mean()), 6), "share_over_3600_new": round(float((r["new_s"] > 3600).mean()), 6)}
        else:
            rep[s_] = {**r["info"], **delta_mse(r["old_s"], r["new_s"], y)}
        print(f"[{name}] {s_}: {json.dumps(rep[s_], ensure_ascii=False)}", flush=True)
    es_rel = f"{mdir}/es_valid_ids_{tag}.parquet"
    shutil.copy2(ROOT / sm["es_valid_ids"]["path"], ROOT / es_rel)
    me = "tools/gjoint.py"
    h = sha16(me)
    snap = f"preds/code/{h}_gjoint.py"
    if not (ROOT / snap).exists():
        shutil.copy2(ROOT / me, ROOT / snap)
    cfg = {"--src": src_mp, "--src_sha": sha16(src_mp), "--tag": tag, "--mode": mode}
    if outdir:
        cfg["--outdir"] = mdir
    flags = " ".join(f"{k} {v}" for k, v in cfg.items())
    cmd = lambda s: f".venv/Scripts/python.exe -X utf8 tools/gjoint.py --setting {s} {flags}"  # noqa: E731
    hold_ids = pd.read_parquet(ROOT / files["val"])["MVT_ID_mvt"].to_numpy()
    gl, src = gl_stage(sm, load_tool("glong"))
    rule = {"a": "on G rows with d_sch in [3,600, 14,400) the art probability is the G model's (gpool S1 after G1) instead of 0.5 G model + "
                 "0.5 leg p_art; other components unchanged",
            "c": "a, plus the normal component replaced by a G-only shrunk additive regression mu_n plus a carrier-shrunk tz-offset component "
                 "(post-hoc variant)"}[mode]
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
        "iteration_rule": sm.get("iteration_rule", "") + f" | gjoint {mode}: {rule}",
        "source": {"queue": stem, "manifest": src_mp, "manifest_sha": cfg["--src_sha"], "spec": sm["spec"],
                   "holdout_sha": sm["holdout"]["sha"], "test_sha": sm["test"]["sha"], "forward_sha": sm["forward"]["sha"]},
        "gjoint": {"mode": mode, "glong_spec": gl["spec"], "gpool_spec": src["spec"], "report": rep, "basis": BASIS[mode]},
        "written_kst": now_kst(),
    }
    (ROOT / mp).write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _, why = manifest_of({"manifest": mp})
    if why:
        print("거부: 새 manifest 검사", why)
        return 1
    print(f"[{name}] manifest {mp}(manifest_of 통과)", flush=True)
    what = {"a": "다리 p_art 띠 평균을 뺀 판(d 1 ~ 4시간 G 행의 art 확률을 0.5 G 모형 + 0.5 다리 p_art 대신 G 모형 그대로, 성분은 사슬 그대로)",
            "c": "띠 평균을 빼고 정상 몫을 G 전용 회귀와 tz 성분으로 바꾼 판(사후 판)"}[mode]
    method = (f"{stem or src_mp} 의 G 행(LIRF NM 미매칭)에 {what}(tools/gjoint.py --mode {mode}, G 행만, glong 뒤 맨 끝). 근거 {BASIS[mode]}. "
              f"자 ΔMSE {rep['val']['dmse']}({rep['val']['improved']}/{rep['val']['changed']}), 7월 전방 ΔMSE {rep['fwd']['dmse']}"
              f"({rep['fwd']['improved']}/{rep['fwd']['changed']}), 시험 이동 {rep['full']['move_mse_test_n']} MSE(G {rep['full']['changed']} 행). "
              "반대 창 행, 띠 밖 G 행, G 아닌 행은 그대로. G 행만 바꾸므로 홀로 쏘지 않고 바탕이 바뀐 판과 함께 낸다. 10-05 단일 발 체제라 준비 점검은 경고(--readiness-warn)")
    nrel = f"{method_dir.rstrip('/')}/gjoint_{name}.json" if method_dir else (f"{mdir}/gjoint_{name}.json" if outdir else f"notes/gjoint_{name}.json")
    (ROOT / nrel).write_text(json.dumps({"name": name, "manifest": mp, "source": stem or src_mp, "method": method, "report": rep, "written_kst": now_kst()},
                                        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[{name}] 방법 json {nrel}", flush=True)
    return 0


def enqueue(name: str) -> int:
    info = json.loads((ROOT / "notes" / f"gjoint_{name}.json").read_text(encoding="utf-8"))
    r = subprocess.run([sys.executable, "-X", "utf8", "tools/enqueue.py", info["manifest"], name, info["method"], "--readiness-warn"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1500:], r.stderr[-800:], flush=True)
    return r.returncode


def main() -> int:
    load_tool("gpool").below_normal()
    a = sys.argv[1:]
    outdir, method_dir, mode = opt(a, "--outdir"), opt(a, "--method_dir"), opt(a, "--mode", "a")
    if a and a[0] == "check":
        return check(a[1], mode)
    if a and a[0] == "_files":
        return build(a[1], a[2], mode, outdir, method_dir)
    if a and a[0] == "enqueue":
        return enqueue(a[1])
    if a and a[0] == "build":
        if len(a) < 3:
            print("usage: gjoint.py build <원 큐 이름 또는 manifest> <새 큐 번호> [--mode a] [--no-enqueue] [--outdir <폴더>] [--method_dir <폴더>]")
            return 2
        me = [sys.executable, "-X", "utf8", str(Path(__file__).resolve())]
        rc = subprocess.run(me + ["_files", a[1], a[2], "--mode", mode] + (["--outdir", outdir] if outdir else []) + (["--method_dir", method_dir] if method_dir else []),
                            cwd=ROOT).returncode
        if rc != 0 or "--no-enqueue" in a or outdir or method_dir:
            return rc
        meta, _ = load_meta(a[1])
        spec = load_man(meta["manifest"])["spec"]
        return subprocess.run(me + ["enqueue", f"{a[2]}_{spec}gj{mode}"], cwd=ROOT).returncode
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "full"])
    pa.add_argument("--seed", default="0")
    pa.add_argument("--src", required=True)
    pa.add_argument("--src_sha", required=True)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--mode", required=True, choices=list(MODES))
    pa.add_argument("--outdir", default="")
    g = pa.parse_args(a)
    if sha16(g.src) != g.src_sha:
        print("거부: 원 manifest 해시가 다르다")
        return 1
    sm = load_man(g.src)
    R = run(sm, g.mode, settings=(g.setting,))
    r = R[g.setting]
    rel = out_rel(g.setting, g.tag, r["new_s"], truth() if g.setting == "val" else None, g.outdir)
    r["new"].to_parquet(ROOT / rel, index=False)
    print(rel, {k: v for k, v in r["info"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
