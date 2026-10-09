# SPDX-License-Identifier: GPL-3.0-only
"""M2: G 정상 확률 보정(사전 등록 R8-P2, 볼트 노트 research/2026-10-07-r8-anatomy.md). 468 의 G 값(gkeyxn avg_xncr) 위, 반대 창 행과 가드 행(d >= 43,200) 밖.
확률: research/r6_g/s9_E4_camon_rt.parquet(avg_xncr 확률 N0 정상, N1 art, N2 shift, 시드 0-2, 설정 val, fwd, nd, full, lomo1..12)와 s5_P.parquet(행 정보 d, dd, y, mu, pa_main).
보정(주 판 A): logit(N0') = a logit(N0) + β (x - c), 절편 없음. x = log(그날 LIRF AIU 출발 전 지연 / IFR 출발 수), 날 = UTC 날짜(t_mvt - mu)(어림 오프블록),
  c = 맞춤 행의 x 평균. 0 기여 몫은 N1:N2 비를 지킨다. 맞춤 = 로지스틱 손실(뉴턴), 맞춤 행은 lomo 행(각 행은 제 달을 뺀 모형의 확률):
  lomo m 은 다른 열한 달, val 은 1, 7월 밖 열 달, fwd 는 1 ~ 6월, nd 는 1 ~ 10월, full(시험) 은 열두 달.
보조 판 B: β = 0(온도만).
G 값: g = clip(model3.g_override(P, d, dd, mu, pa_main), 0)(띠 평균 그대로). 468 위 f' = f + (g' - g)(G, 반대 창 아님, 가드 아님).
보고: lomo 열두 달 G SSE(sub_g.expected, R6 r6e 와 같은 식), val, fwd(468 대비, 날 다시 뽑기), nd(G 값끼리), 시험 이동.
    run.sh 3000 2.0 m2_gcal.py -> m2_gcal.txt, m2_gcal_full.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from r8c import NDEN, OUT, ROOT, boot, nice, preds

nice()
import src.sub_g as sub_g  # noqa: E402
from src.model3 import g_override  # noqa: E402

CUT = 43200.0
R6 = ROOT / "research" / "r6_g"
import json  # noqa: E402
import sys  # noqa: E402

BR = sys.argv[1] if len(sys.argv) > 1 else "468"  # 468(B 가 이길 때) 또는 486(B 가 질 때)
C468 = {"468": "m6r27wincbrmfw4adsbgeondctxv3x50wamfhgkcxncrdbaw", "486": "m6r27skcbrmfw4adsbgeondctxv3x50wamfhgkcxncrdbaw"}[BR]
C467 = C468[:-2]
LEG468 = json.loads((ROOT / "preds" / f"manifest_{C468[:-4]}.json").read_text(encoding="utf-8"))["config"]["--leg"]
print("갈래", BR, C468, "G 다리", LEG468, flush=True)
P = pd.read_parquet(R6 / "s5_P.parquet", columns=["setting", "MVT_ID_mvt", "src", "d", "dd", "y", "mu", "pa_main"])
E = pd.read_parquet(R6 / "s9_E4_camon_rt.parquet")
R = P.merge(E, on=["setting", "MVT_ID_mvt"], how="inner")
assert len(R) == len(P) == len(E), (len(R), len(P), len(E))
T = pd.read_parquet(R6 / "s4_lirf_dep.parquet", columns=["MVT_ID_mvt", "t_mvt"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")["t_mvt"]
R["t"] = T.reindex(R["MVT_ID_mvt"].to_numpy()).to_numpy(float)
assert np.isfinite(R["t"]).all(), int((~np.isfinite(R["t"])).sum())
a = pd.concat([pd.read_csv(ROOT / "data" / "ext" / "aiu" / f"all_pre_departure_delays_{yr}.csv") for yr in (2025, 2026)])
a = a[a["APT_ICAO"] == "LIRF"]
pdd = pd.Series((60.0 * a["DLY_ALL_PRE_2"] / a["FLT_DEP_IFR_2"]).to_numpy(), index=a["FLT_DATE"].astype(str).to_numpy())
pdd = pdd[~pdd.index.duplicated()]
dob = pd.to_datetime(R["t"].to_numpy() - R["mu"].to_numpy(float), unit="s").strftime("%Y-%m-%d")
R["x"] = np.log(pdd.reindex(dob).to_numpy(float))
y, d = R["y"].to_numpy(float), R["d"].to_numpy(float)
art = np.abs(y - d) <= 60
sh = (y > 60000) & ~art
R["Z"] = np.where(np.isfinite(y), (~art & ~sh).astype(float), np.nan)
R["day"] = pd.to_datetime(R["t"].to_numpy(), unit="s").strftime("%Y-%m-%d")
EPS = 1e-4


def lg(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit(F, use_x=True):
    """logit(N0') = a z + b (x - c), 절편 없음, 로지스틱 손실 뉴턴(작은 L2 1e-6)."""
    F = F[(F["d"] < CUT) & np.isfinite(F["Z"])]
    z = lg(F["N0"].to_numpy())
    x = F["x"].to_numpy()
    c = float(np.nanmean(x))
    xc = np.where(np.isfinite(x), x - c, 0.0)
    X = np.column_stack([z, xc]) if use_x else z[:, None]
    Z = F["Z"].to_numpy()
    w = np.zeros(X.shape[1])
    w[0] = 1.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-(X @ w)))
        gr = X.T @ (p - Z) + 1e-6 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, gr)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return {"a": float(w[0]), "b": float(w[1]) if use_x else 0.0, "c": c, "n": len(F)}


def apply(S, prm):
    """새 확률(N0', N1', N2'). 가드 행은 그대로."""
    N = S[["N0", "N1", "N2"]].to_numpy(float).copy()
    x = S["x"].to_numpy()
    xc = np.where(np.isfinite(x), x - prm["c"], 0.0)
    n0 = 1 / (1 + np.exp(-(prm["a"] * lg(N[:, 0]) + prm["b"] * xc)))
    z0 = N[:, 1] + N[:, 2]
    with np.errstate(all="ignore"):
        f1 = np.where(z0 > 0, N[:, 1] / z0, 0.5)
    new = np.column_stack([n0, (1 - n0) * f1, (1 - n0) * (1 - f1)])
    keep = S["d"].to_numpy() >= CUT
    new[keep] = N[keep]
    return new


def gval(S, Pm):
    pa = S["pa_main"].to_numpy(float)
    return np.clip(g_override(Pm, S["d"].to_numpy(float), S["dd"].to_numpy(float), S["mu"].to_numpy(float), None if np.isnan(pa).all() else pa)[0], 0, None)


L = R[R["setting"].str.startswith("lomo")].copy()
L["m"] = L["setting"].str[4:].astype(int)
FITM = {"val": [m for m in range(1, 13) if m not in (1, 7)], "fwd": list(range(1, 7)), "nd": list(range(1, 11)), "full": list(range(1, 13))}
res = {}
for vname, use_x in (("A 온도 + 날 지연", True), ("B 온도만", False)):
    print(f"\n######## 판 {vname}", flush=True)
    # lomo 열두 달
    sse0, sse1 = np.zeros(12), np.zeros(12)
    prms = []
    for i, m in enumerate(range(1, 13)):
        prm = fit(L[L["m"] != m], use_x)
        prms.append(prm)
        S = L[L["m"] == m]
        dd_, mu = S["dd"].to_numpy(float), S["mu"].to_numpy(float)
        sse0[i] = ((sub_g.expected(S[["N0", "N1", "N2"]].to_numpy(float), S["d"].to_numpy(float), dd_, mu, mu) - S["y"].to_numpy()) ** 2).sum()
        sse1[i] = ((sub_g.expected(apply(S, prm), S["d"].to_numpy(float), dd_, mu, mu) - S["y"].to_numpy()) ** 2).sum()
    print(f"  lomo G SSE {sse0.sum():.4e} -> {sse1.sum():.4e} ({(sse1.sum() / sse0.sum() - 1) * 100:+.2f}%), 이긴 달 {int((sse1 < sse0 - 1e-6).sum())}/12, "
          f"1, 7월 몫(val 분모) {(sse1[[0, 6]].sum() - sse0[[0, 6]].sum()) / NDEN['val']:+,.0f}", flush=True)
    print("  달별 %: " + ", ".join(f"{m} {(sse1[i] / sse0[i] - 1) * 100:+.1f}" for i, m in enumerate(range(1, 13))), flush=True)
    print("  lomo 매개변수 a: " + ", ".join(f"{p['a']:.2f}" for p in prms) + " | b: " + ", ".join(f"{p['b']:.2f}" for p in prms), flush=True)
    out = {"lomo_pct": (sse1.sum() / sse0.sum() - 1) * 100, "lomo_wins": int((sse1 < sse0 - 1e-6).sum())}
    for st in ("val", "fwd", "nd", "full"):
        prm = fit(L[L["m"].isin(FITM[st])], use_x)
        S = R[R["setting"] == st].reset_index(drop=True)
        ids = S["MVT_ID_mvt"].to_numpy()
        if st != "nd" and LEG468 != "m6r27skg":  # 468 사슬의 G 다리(gkey --leg)의 mu, p_art 로 바꾼다(s5_P 는 m6r27skg 다리 값)
            lb = pd.read_parquet(ROOT / "cache" / f"{st}_{LEG468}_s0.parquet", columns=["MVT_ID_mvt", "mu", "p_art"]).set_index("MVT_ID_mvt").reindex(ids)
            S["mu"], S["pa_main"] = lb["mu"].to_numpy(float), lb["p_art"].to_numpy(float)
        g0 = gval(S, S[["N0", "N1", "N2"]].to_numpy(float))
        g1 = gval(S, apply(S, prm))
        if st in ("val", "fwd"):
            gtool = preds(C468[:-4], st).reindex(ids).to_numpy(float)  # gkeyxn 파일의 G 행 = clip(g_n)
            print(f"  {st}: 맞춤 a {prm['a']:.3f}, b {prm['b']:.3f}, c {prm['c']:.3f}(행 {prm['n']}) | g 재현 대 gkeyxn 파일 최대 차 {np.nanmax(np.abs(g0 - gtool)):.2e}", flush=True)
            f = preds(C468, st).reindex(ids).to_numpy(float)
            aw = np.abs(f - preds(C467, st).reindex(ids).to_numpy(float)) > 1e-6
            keep = aw | (S["d"].to_numpy() >= CUT)
            fn = np.where(keep, f, np.clip(f + (g1 - g0), 0, None))
            yy = S["y"].to_numpy(float)
            dr = (fn - yy) ** 2 - (f - yy) ** 2
            bs = boot(dr, S["day"].to_numpy(), NDEN[st])
            mo = S["day"].str[5:7].to_numpy()
            print(f"  [{st}] ΔMSE {dr.sum() / NDEN[st]:+.1f} ± {bs.std():.1f} (Δ<0 {float((bs < 0).mean()):.3f}) | 바뀐 행 {int((np.abs(fn - f) > 1e-9).sum())}"
                  f"(좋아짐 {int((dr < -1e-6).sum())}, 나빠짐 {int((dr > 1e-6).sum())}) | 이동 {np.sum((fn - f) ** 2) / NDEN[st]:.1f} | 반대 창 행 {int(aw.sum())}, 가드 {int((S['d'] >= CUT).sum())}"
                  + (" | 달별 " + ", ".join(f"{m}월 {dr[mo == m].sum() / NDEN[st]:+.1f}" for m in sorted(set(mo))) if st == "val" else ""), flush=True)
            out[st] = dr.sum() / NDEN[st]
            out[st + "_neg"] = float((bs < 0).mean())
        elif st == "nd":
            yy = S["y"].to_numpy(float)
            dr = (g1 - yy) ** 2 - (g0 - yy) ** 2
            bs = boot(dr, S["day"].to_numpy(), 328009)
            print(f"  [nd] 맞춤 a {prm['a']:.3f}, b {prm['b']:.3f} | G 값끼리 ΔMSE {dr.sum() / 328009:+.1f} ± {bs.std():.1f} (Δ<0 {float((bs < 0).mean()):.3f}), 행 {len(S)}", flush=True)
            out["nd"] = dr.sum() / 328009
        else:
            ft = preds(C468, "full").reindex(ids).to_numpy(float)
            f7 = preds(C467, "full").reindex(ids).to_numpy(float)
            aw = np.abs(ft - f7) > 1e-6
            keep = aw | (S["d"].to_numpy() >= CUT)
            fn = np.where(keep, ft, np.clip(ft + (g1 - g0), 0, None))
            mv = np.sum((fn - ft) ** 2) / 344841
            mo = S["day"].str[5:7].to_numpy()
            print(f"  [시험] 맞춤 a {prm['a']:.3f}, b {prm['b']:.3f}, c {prm['c']:.3f} | 바뀐 행 {int((np.abs(fn - ft) > 1e-9).sum())}, 이동 {mv:.1f}"
                  f" (1월 {np.sum(((fn - ft) ** 2)[mo == '01']) / 344841:.1f}, 7월 {np.sum(((fn - ft) ** 2)[mo == '07']) / 344841:.1f}) | 평균 이동 {np.mean(fn - ft):+.0f}초"
                  f" | 평균 N0 {S['N0'].mean():.3f} -> {apply(S, prm)[:, 0].mean():.3f}", flush=True)
            out["full_move"] = mv
            if use_x:
                pd.DataFrame({"MVT_ID_mvt": ids, "f468": ft, "f_new": fn, "N0": S["N0"].to_numpy(), "N0_new": apply(S, prm)[:, 0], "x": S["x"].to_numpy()}).to_parquet(OUT / f"m2_gcal_full_{BR}.parquet", index=False)
    res[vname] = out
print("\n요약:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res.items()}, flush=True)
