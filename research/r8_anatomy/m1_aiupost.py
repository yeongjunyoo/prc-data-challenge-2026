# SPDX-License-Identifier: GPL-3.0-only
"""M1: AIU 날 합으로 큰 d - mn 행의 유형(복사 0 기여 대 정상) 사후 확률을 고친다(사전 등록 R8-P1, 볼트 노트 research/2026-10-07-r8-anatomy.md 4절).
468 위(반대 창 뒤 맨 끝)에서 잰다. 읽기만(a1_rows, a0_mix, ctx_v3, AIU 두 파일, feats_v25 dd, 옛 nd 다리).

규칙(등록 글 그대로):
  날 묶음과 g, 공항 치우침 off 는 src/aiu_shift.py --group block 과 같다(옮길 행 = UTC 날짜(t - mn_post), 맞춤 행 = UTC 날짜(t - y),
  val 은 다른 달, fwd 는 val 1월, nd 와 시험은 val 두 달). mn_post = db 이동 뒤 정상 몫(G 아닌 행 mn + (f_db - f_xncr)/(1 - q), G 행 mu_g).
  e = g(d - mn_post), 행 기대 기여 E = (1 - q) e. 반대 창이 바꾼 행(aw)은 E = max(0, d - f_aw) 고정.
  후보 C = {e >= 1,800, 0.001 < q < 0.999, aw 아님}. 날마다 r = (T - off) - Σ_모든 행 E, V_day = Σ_C q(1 - q) e².
  행 i ∈ C: V_i = s²_ap + V_day - q_i(1 - q_i)e_i², log LR_i = [(r + (1 - q_i)e_i)² - (r - q_i e_i)²] / (2 V_i)(±30 에서 자름),
  정상 사후 π_i = (1 - q_i)e^LR / ((1 - q_i)e^LR + q_i), q'_i = 1 - π_i.
  s²_ap = 맞춤 날(같은 규칙의 r)의 (r - Σ_C (Z - (1 - q))e)² 평균(Z = 참 정상, G 아닌 행 |y - d| > 60, G 행 |y - d| > 60 이고 y <= 60,000), 공항별.
  예측: G 아닌 행 f' = f + (q' - q)(d - m), m = (f - q d)/(1 - q), LOBT 창, 0 바닥. G 행 pn' = π, 옮기는 몫은 pa 와(모자라면 ps 에서) 주고받는다,
  f' = f + (pn' - pn) mu_g + (pa' - pa) d + (ps' - ps) S, 0 바닥.
위약: P1 r := 0(날 정보 없이 다듬기만), P2 AIU 합을 공항-달 안에서 섞기(씨앗 다섯 평균).
    run.sh 3500 2.5 m1_aiupost.py            -> m1_aiupost.txt (val, fwd, 위약)
    run.sh 3500 2.5 m1_aiupost.py nd         -> nd 대리(옛 다리, 표준 db 와 반대 창 흉내 위)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from r8c import NDEN, OUT, ROOT, boot, nice

nice()
import src.aiu_shift as A  # noqa: E402

E_MIN, QLO, QHI, LRCAP = 1800.0, 0.001, 0.999, 30.0
cx, fx, tot = A.inputs("ctx_v3", "0b139ddbd6e70573", {"--aiu25_sha": "d742eeb626e5f9f7", "--aiu26_sha": "5952d20ef5f9c424"})
t_all = A.ctx_t("ctx_v3")


def frame_from_a1(st):
    S = pd.read_parquet(OUT / f"a1_rows_{st}.parquet")
    M = pd.read_parquet(OUT / f"a0_mix_{st}.parquet").set_index("MVT_ID_mvt").reindex(S.index)
    g = S["g"].to_numpy(bool)
    q = np.where(g, 1 - S["G_pn"].to_numpy(float), S["q"].to_numpy(float))
    mfr = S["mn"].to_numpy(float)
    with np.errstate(all="ignore"):
        mpost = np.where(g, S["G_mu_g"].to_numpy(float), mfr + (S["s_db"].to_numpy(float) - S["f464"].to_numpy(float)) / (1 - q))
    mpost = np.where(np.isfinite(mpost), mpost, mfr)
    f = S["f_c468"].to_numpy(float)
    D = pd.DataFrame({"id": S.index.to_numpy(), "y": S["y"].to_numpy(float), "d": S["d"].to_numpy(float), "g": g, "ap": S["ap"].astype(str).to_numpy(),
                      "day": cx["day"].reindex(S.index).to_numpy(), "q": q, "mfr": mfr, "mpost": mpost, "f": f, "dl": M["dl"].to_numpy(float),
                      "aw": S["aw"].to_numpy(bool) | (np.abs(f - S["s_db"].to_numpy(float)) > 1e-6),
                      "pa": S["G_pa"].to_numpy(float), "ps": S["G_ps"].to_numpy(float), "pn": S["G_pn"].to_numpy(float), "S": S["G_S"].to_numpy(float),
                      "mu_g": S["G_mu_g"].to_numpy(float)})
    return D


def fit(V):
    """g, off(aiu_shift 와 같은 맞춤) 과 s²_ap(같은 규칙의 r 로, 맞춤 행의 정답 사용)."""
    Vf = V.assign(mn=V["mfr"])
    g = A.fit_g(Vf)
    off = A.offsets(Vf.assign(day=A.block_days(Vf, t_all, "y")), tot)
    T = terms(V, g, off)
    y, d, q, e = V["y"].to_numpy(), V["d"].to_numpy(), V["q"].to_numpy(), T["e"]
    art = np.abs(y - d) <= 60
    Z = (~art) & ~(V["g"].to_numpy() & (y > 60000))
    dev = np.where(T["cand"], (Z.astype(float) - (1 - q)) * e, 0.0)
    s2 = {}
    k = T["key"]
    ok = T["T"] == T["T"]
    df = pd.DataFrame({"k": k[ok], "dev": dev[ok]}).groupby("k")["dev"].sum()
    rd = pd.Series(T["r_row"][ok], index=k[ok]).groupby(level=0).first()
    nz = (rd - df.reindex(rd.index).fillna(0.0))
    for ap, x in nz.groupby(nz.index.str[:4]):
        s2[ap] = float(np.mean(x.to_numpy() ** 2))
    return g, off, s2


def terms(D, g, off, placebo=None, seed=0):
    """행마다 e, E, 후보, 날 열쇠, 그 날의 r 과 V_day(행에 펼침), T(없거나 out 이면 NaN)."""
    Db = D.assign(mn=D["mpost"])
    day_b = A.block_days(Db, t_all, "mn")
    d, q, mp = D["d"].to_numpy(), D["q"].to_numpy(), D["mpost"].to_numpy()
    e = g(d - mp)
    aw = D["aw"].to_numpy()
    E = np.where(aw, np.maximum(0.0, d - D["f"].to_numpy()), (1 - q) * e)
    cand = (e >= E_MIN) & (q > QLO) & (q < QHI) & ~aw & np.isfinite(e)
    key = (D["ap"].to_numpy().astype(object) + "|" + day_b.astype(object)).astype(str)
    uk = pd.unique(key)
    Tmap = {k: (tot.get(k, np.nan) - off.get(k[:4], 0.0)) if not k.endswith("|out") else np.nan for k in uk}
    if placebo == "P2":
        rng = np.random.default_rng(seed)
        kk = pd.Series([k for k in uk if np.isfinite(Tmap[k])])
        grp = kk.str[:4] + kk.str[10:12]  # 공항 + 달
        newT = {}
        for _, ks in kk.groupby(grp.to_numpy()):
            vals = np.array([Tmap[k] + off.get(k[:4], 0.0) for k in ks])
            perm = rng.permutation(len(vals))
            for k, v in zip(ks, vals[perm]):
                newT[k] = v - off.get(k[:4], 0.0)
        Tmap.update(newT)
    Tk = np.array([Tmap[k] for k in key], dtype=float)
    sE = pd.Series(E).groupby(key).sum()
    vC = pd.Series(np.where(cand, q * (1 - q) * e ** 2, 0.0)).groupby(key).sum()
    r_row = Tk - sE.reindex(key).to_numpy()
    if placebo == "P1":
        r_row = np.where(np.isfinite(Tk), 0.0, np.nan)
    return {"e": e, "E": E, "cand": cand, "key": key, "T": Tk, "r_row": r_row, "vday": vC.reindex(key).to_numpy()}


def apply(D, g, off, s2, placebo=None, seed=0):
    T = terms(D, g, off, placebo, seed)
    q, e, cand = D["q"].to_numpy(), T["e"], T["cand"] & np.isfinite(T["r_row"])
    s2r = np.array([s2.get(a, np.nan) for a in D["ap"].to_numpy()])
    cand &= np.isfinite(s2r)
    r = T["r_row"]
    Vi = s2r + T["vday"] - q * (1 - q) * e ** 2
    with np.errstate(all="ignore"):
        lr = np.clip(((r + (1 - q) * e) ** 2 - (r - q * e) ** 2) / (2 * Vi), -LRCAP, LRCAP)
        pi = (1 - q) * np.exp(lr) / ((1 - q) * np.exp(lr) + q)
    qn = np.where(cand, 1 - pi, q)
    f = D["f"].to_numpy().copy()
    d = D["d"].to_numpy()
    gmask = D["g"].to_numpy()
    out = f.copy()
    ng = cand & ~gmask
    with np.errstate(all="ignore"):
        m = (f - q * d) / (1 - q)
    out[ng] = f[ng] + (qn[ng] - q[ng]) * (d[ng] - m[ng])
    dl = D["dl"].to_numpy()
    win = ng & np.isfinite(dl)
    out[win] = np.clip(out[win], dl[win] - 3606.0, dl[win] + 3606.0)
    gg = cand & gmask
    pn, pa, ps = D["pn"].to_numpy(), D["pa"].to_numpy(), D["ps"].to_numpy()
    pn2 = 1 - qn
    pa2 = pa - (pn2 - pn)
    ps2 = ps + np.minimum(pa2, 0.0)
    pa2 = np.maximum(pa2, 0.0)
    ps2 = np.maximum(ps2, 0.0)
    out[gg] = f[gg] + (pn2[gg] - pn[gg]) * D["mu_g"].to_numpy()[gg] + (pa2[gg] - pa[gg]) * d[gg] + (ps2[gg] - ps[gg]) * D["S"].to_numpy()[gg]
    out = np.where(cand, np.maximum(out, 0.0), f)
    return out, {"cand": int(cand.sum()), "cand_g": int(gg.sum()), "changed": int((np.abs(out - f) > 1e-9).sum()), "qn": qn}


def report(tag, D, fnew, nden, extra=""):
    y = D["y"].to_numpy()
    f = D["f"].to_numpy()
    ok = np.isfinite(y)
    dr = np.where(ok, (fnew - y) ** 2 - (f - y) ** 2, 0.0)
    bs = boot(dr[ok], D["day"].to_numpy()[ok], nden)
    gm = D["g"].to_numpy()
    li = D["ap"].to_numpy() == "LIRF"
    ch = np.abs(fnew - f) > 1e-9
    imp = int(((fnew - y) ** 2 < (f - y) ** 2)[ch & ok].sum())
    print(f"  [{tag}] ΔMSE {dr.sum() / nden:+.1f} ± {bs.std():.1f} (Δ<0 {float((bs < 0).mean()):.3f}) | G {dr[gm].sum() / nden:+.1f}, G 아님 LIRF {dr[~gm & li].sum() / nden:+.1f}, "
          f"G 아님 밖 {dr[~gm & ~li].sum() / nden:+.1f} | 바뀐 행 {int(ch.sum())}(좋아짐 {imp}) | 이동 {np.sum((fnew - f) ** 2) / nden:.1f} {extra}", flush=True)
    return dr


def months(D, dr, nden):
    mo = pd.Series(D["day"].to_numpy()).str[5:7].to_numpy()
    return ", ".join(f"{m}월 {dr[mo == m].sum() / nden:+.1f}" for m in sorted(set(mo)))


def main_valfwd():
    V = frame_from_a1("val")
    jan = (V["day"].str[5:7] == "01").to_numpy()
    for st in ("val", "fwd"):
        D = V if st == "val" else frame_from_a1("fwd")
        nden = NDEN[st]
        parts = [(jan, ~jan), (~jan, jan)] if st == "val" else [(np.ones(len(D), bool), None)]
        res = {}
        for pl, seeds in ((None, [0]), ("P1", [0]), ("P2", [0, 1, 2, 3, 4])):
            outs = []
            for sd in seeds:
                fnew = D["f"].to_numpy().copy()
                info = []
                for msk, fitm in parts:
                    Vfit = V[fitm].reset_index(drop=True) if fitm is not None else V[jan].reset_index(drop=True)
                    g, off, s2 = fit(Vfit)
                    Dm = D[msk].reset_index(drop=True)
                    o, inf = apply(Dm, g, off, s2, pl, sd)
                    fnew[msk] = o
                    info.append((inf, {k: round(np.sqrt(v)) for k, v in s2.items()}))
                outs.append(fnew)
            if pl is None:
                print(f"\n==== {st}: 후보 행 {sum(i[0]['cand'] for i in info)}(G {sum(i[0]['cand_g'] for i in info)}), s_ap(초) {info[0][1]}", flush=True)
            drs = [report(f"{st} {'실제' if pl is None else pl}{'' if len(seeds) == 1 else f' 씨앗 {s}'}", D, o, nden) for s, o in zip(seeds, outs)]
            res[pl] = np.mean([x.sum() / nden for x in drs])
            if pl is None:
                print("    달별: " + months(D, drs[0], nden), flush=True)
                np.save(OUT / f"m1_fnew_{st}.npy", outs[0])
        print(f"  요약 {st}: 실제 {res[None]:+.1f}, P1 {res['P1']:+.1f}, P2 평균 {res['P2']:+.1f}", flush=True)


def main_nd():
    """nd 대리: 옛 다리 nd_m6r25 위에 표준 db(λ 0.5, ±60) 와 반대 창을 흉내 낸 뒤(R7 e7 와 같은 길), 그 위에 이 규칙. 맞춤은 val 두 달(468 의 val 프레임)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("antiwin", str(ROOT / "tools" / "antiwin.py"))
    aw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aw)
    LINK = aw.build_link()
    Atab = pq.read_table(ROOT / "cache" / "all.parquet", columns=["MVT_ID_mvt", "t_mvt"], filters=[("PHASE_mvt", "==", "DEP"), ("src", "in", [11, 12])]).to_pandas()
    TND = Atab.set_index("MVT_ID_mvt")["t_mvt"].astype(float)
    global t_all
    t_all = pd.concat([t_all, TND])
    t_all = t_all[~t_all.index.duplicated()]
    C464 = "m6r27wincbrmfw4adsbgeondctxv3x50wamfhgkcxncr"
    ch = A.chain(f"preds/manifest_{C464}.json")
    Vfr = A.frame("val", ch, cx, fx)
    g0 = A.fit_g(Vfr)
    off0 = A.offsets(Vfr.assign(day=A.block_days(Vfr, t_all, "y")), tot)
    b = pd.read_parquet(ROOT / "cache" / "nd_m6r25_s0.parquet")
    t = pq.read_table(ROOT / "cache" / "feats_v25.parquet", columns=["MVT_ID_mvt", "src", "ap", "d_lobt", "dd"], filters=[("src", "in", [11, 12])]).to_pandas().set_index("MVT_ID_mvt")
    ids = b["MVT_ID_mvt"].to_numpy()
    N = pd.DataFrame({"id": ids, "fc": b["pred"].to_numpy(), "f": b["pred"].to_numpy(), "p": b["p_art"].to_numpy(), "mu": b["mu"].to_numpy(),
                      "g": b["g"].to_numpy().astype(bool), "d": b["dsch"].to_numpy(), "base": b["pred"].to_numpy(), "y": b["y"].to_numpy()})
    N["ap"] = t["ap"].astype(str).reindex(ids).to_numpy()
    N["day"] = pd.to_datetime(TND.reindex(ids).to_numpy(), unit="s").strftime("%Y-%m-%d")
    N["dl"] = t["d_lobt"].reindex(ids).to_numpy(dtype=float)
    N["aw"] = A.antiwin_mask("nd", pd.Series(N["f"].to_numpy(), index=ids)).reindex(ids).to_numpy()
    N["m"] = N["mu"]
    N = A.add_classes(N, t["dd"].astype(float))
    NB = N.assign(day=A.block_days(N, t_all, "mn"))
    fdb, _ = A.shift_days(NB, g0, off0, tot, 0.5, 60.0)
    faw = aw.apply_rules(pd.Series(fdb, index=ids), LINK, "nd")[0].reindex(ids).to_numpy()
    # G 성분(add_classes 와 같은 식): pa = p, pn 은 기록된 pred 에서, ps = 1 - pa - pn
    dd = t["dd"].astype(float).reindex(ids).to_numpy()
    p, d, mu, base = N["p"].to_numpy(), N["d"].to_numpy(), N["mu"].to_numpy(), N["base"].to_numpy()
    Sg = 86400 * np.maximum(dd, 1) + np.clip(mu, 300, 3000)
    with np.errstate(all="ignore"):
        pn = np.clip((p * d + (1 - p) * Sg - base) / (Sg - mu), 0, 1 - p)
    q = N["q"].to_numpy()
    mfr = N["mn"].to_numpy()
    gm = N["g"].to_numpy()
    with np.errstate(all="ignore"):
        mpost = np.where(gm, mu, mfr + (fdb - N["f"].to_numpy()) / (1 - q))
    mpost = np.where(np.isfinite(mpost), mpost, mfr)
    D = pd.DataFrame({"id": ids, "y": N["y"].to_numpy(float), "d": d, "g": gm, "ap": N["ap"].to_numpy(), "day": N["day"].to_numpy(), "q": q, "mfr": mfr,
                      "mpost": mpost, "f": faw, "dl": N["dl"].to_numpy(), "aw": N["aw"].to_numpy() | (np.abs(faw - fdb) > 1e-6), "pa": np.where(gm, p, np.nan),
                      "ps": np.where(gm, 1 - p - pn, np.nan), "pn": np.where(gm, pn, np.nan), "S": Sg, "mu_g": mu})
    V = frame_from_a1("val")
    g, off, s2 = fit(V)
    print(f"nd: 행 {len(D)}, 정답 있는 행 {int(np.isfinite(D['y']).sum())}, s_ap {({k: round(np.sqrt(v)) for k, v in s2.items()})}", flush=True)
    yy = D["y"].to_numpy()
    ok = np.isfinite(yy)
    print(f"  nd 흉내 바탕(db + 반대 창) RMSE {np.sqrt(np.mean((faw[ok] - yy[ok]) ** 2)):.3f}", flush=True)
    res = {}
    for pl, seeds in ((None, [0]), ("P1", [0]), ("P2", [0, 1, 2, 3, 4])):
        vals = []
        for sd in seeds:
            o, inf = apply(D, g, off, s2, pl, sd)
            dr = report(f"nd {'실제' if pl is None else pl}{'' if len(seeds) == 1 else f' 씨앗 {sd}'}", D, o, NDEN.get("nd", 328009), f"후보 {inf['cand']}(G {inf['cand_g']})")
            vals.append(dr.sum() / 328009)
            if pl is None:
                print("    달별: " + months(D, dr, 328009), flush=True)
        res[pl] = float(np.mean(vals))
    print(f"  요약 nd: 실제 {res[None]:+.1f}, P1 {res['P1']:+.1f}, P2 평균 {res['P2']:+.1f}", flush=True)


if __name__ == "__main__":
    if "nd" in sys.argv[1:]:
        main_nd()
    else:
        main_valfwd()
