# SPDX-License-Identifier: GPL-3.0-only
"""R13-F3 feasibility (private notes research/2026-10-07-r13-f3.md 1): would a third, AOBT-anchored (--anchor aobt) v2.7 chain family beat the two-family
average (500 = 0.5 win chain + 0.5 sk chain)? Reads existing predictions only, no fitting, no test rows.
  L  leg level, LightGBM: v2.7 win (m6r27skwing) and none (m6r27skg) with their Mac seed-1 runs; v2.6 none (m6r26sg) and aobt (m6r26as8g).
     The v2.6 anchoring effect d_aobt = as8g - sg is transplanted onto the v2.7 none leg (C* = skg + d_aobt). Placebo third members with the same
     noise structure: the none leg's seed 1 (skg + (sk s1 - sk s0)) and the win leg's seed 1.
  B  blend stage (before the corrector) and C corrector stage: 500's two chains (wincbrmfw4sv114 / skcbrmfw4sv114, corrector c3) and the v2.6
     pair m6r26sgc75 / m6r26as8gc75 (corrector stage: the old ADS-B corrector, adsbw). Transplant C* = B + (aobt26 - none26) at the same stage.
     Placebo: B's 497 build (seed-0 legs, corrector seed 0) as the third member.
  F  final stage (500's A and B after the whole tail): the 'as good and as different' estimate, a third member with the mean of A's and B's
     error variance and the A-B error covariance with both.
  H  R5 harness (20 % of days, lr 0.1; base = no anchor, a_win, a_aobt, seeds 0 and 1): 3-way of two-seed families against the 2-way
     base + a_win, with the expected seed-noise part of the gain (equal model count is impossible) subtracted.
Rows: body = y <= 3,600 and not G (G = LIRF without NM flight match), the R10-A transfer rows; clean = all rows but non-G y > 50,000 (R8).
dMSE = (SSE_3way - SSE_2way) / all rows of the ruler (negative = the third family helps). Day resample: UTC MVT day of cache/ctx_v3, 2,000 draws.
    python research/r13_f3/f1_feas.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "r13_f3"
MAC = ROOT / "research" / "r10_b" / "mac"
H = ROOT / "research" / "r5_paradigms" / "preds"
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if sys.platform == "win32":
    import ctypes
    ctypes.WinDLL("kernel32").SetPriorityClass(ctypes.WinDLL("kernel32").GetCurrentProcess(), 0x00004000)

t = pq.read_table(ROOT / "cache" / "all.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "y"], filters=[("PHASE_mvt", "==", "DEP")],
                  read_dictionary=["ADEP_mvt"])
INFO = pd.DataFrame({"ap": t.column("ADEP_mvt").to_pandas().astype(str).to_numpy(), "nm": t.column("FLIGHT_ID_mvt").is_valid().to_numpy(zero_copy_only=False),
                     "y": t.column("y").to_numpy().astype(float)}, index=t.column("MVT_ID_mvt").to_numpy())
del t
DAY = pd.read_parquet(ROOT / "cache" / "ctx_v3.parquet", columns=["MVT_ID_mvt", "day"]).set_index("MVT_ID_mvt")["day"].astype(str)
RES: dict = {}


def cpred(path: Path, ids: np.ndarray) -> np.ndarray:
    s = pd.read_parquet(path, columns=["MVT_ID_mvt", "pred"]).set_index("MVT_ID_mvt")["pred"].astype(float).reindex(ids)
    assert not s.isna().any(), f"{path.name}: rows missing"
    return s.to_numpy()


def boot(d: np.ndarray, day: np.ndarray, n_all: int) -> tuple[float, float]:
    ud, inv = np.unique(day, return_inverse=True)
    per = np.bincount(inv, weights=d, minlength=len(ud))
    cnt = np.bincount(inv, minlength=len(ud)).astype(float)
    dr = np.random.default_rng(0).integers(0, len(ud), size=(2000, len(ud)))
    bt = per[dr].sum(1) / cnt[dr].sum(1)
    return float(bt.std(ddof=1)), float((bt < 0).mean())


def stats(name: str, st: str, P: dict, y: np.ndarray, masks: dict, day: np.ndarray, base3: tuple, thirds: dict) -> None:
    """P: name -> pred. base3 = (A, B). For each third member C: dMSE of (A+B+C)/3 vs (A+B)/2 on each mask, with day resample."""
    n = len(y)
    A, B = P[base3[0]], P[base3[1]]
    two = 0.5 * (A + B)
    rep = {}
    for mk, m in masks.items():
        e = {k: (v - y)[m] for k, v in P.items()}
        r = {"mse": {k: round(float((ev ** 2).sum() / n), 1) for k, ev in e.items()}}
        keys = list(e)
        r["corr"] = {f"{a}~{b}": round(float(np.corrcoef(e[a], e[b])[0, 1]), 5) for i, a in enumerate(keys) for b in keys[i + 1:]}
        r["two_vs_A"] = round(float((((two - y)[m]) ** 2).sum() / n - (e[base3[0]] ** 2).sum() / n), 1)
        r["three"] = {}
        for cn, C in thirds.items():
            d = np.where(m, ((A + B + C) / 3 - y) ** 2 - (two - y) ** 2, 0.0)
            sd, p = boot(d, day, n)
            r["three"][cn] = {"dmse": round(float(d.sum() / n), 1), "sd": round(sd, 1), "p_below0": round(p, 4)}
        rep[mk] = r
        print(f"[{name} {st} {mk}] mse {r['mse']} | two-way vs {base3[0]} {r['two_vs_A']:+.1f}", flush=True)
        print(f"[{name} {st} {mk}] corr {r['corr']}", flush=True)
        for cn, v in r["three"].items():
            print(f"[{name} {st} {mk}]   3-way with {cn}: dMSE vs 2-way {v['dmse']:+.1f} (sd {v['sd']:.1f}, P<0 {v['p_below0']:.3f})", flush=True)
    RES.setdefault(name, {})[st] = rep


def analytic(name: str, st: str, A: np.ndarray, B: np.ndarray, y: np.ndarray, masks: dict) -> None:
    """Third member as good as the mean of A and B and as different from each as A is from B (equal pairwise covariance)."""
    n = len(y)
    out = {}
    for mk, m in masks.items():
        ea, eb = (A - y)[m], (B - y)[m]
        va, vb, cab = (ea ** 2).sum() / n, (eb ** 2).sum() / n, (ea * eb).sum() / n
        vc = 0.5 * (va + vb)
        two = 0.25 * (va + vb + 2 * cab)
        three = (va + vb + vc + 2 * cab + 2 * cab + 2 * cab) / 9
        out[mk] = {"two": round(float(two), 1), "three": round(float(three), 1), "d": round(float(three - two), 1),
                   "corr": round(float(cab / np.sqrt(va * vb)), 5), "two_vs_A": round(float(two - va), 1)}
        print(f"[{name} {st} {mk}] as-good-as-different third: 3-way - 2-way {three - two:+.1f} (2-way vs A {two - va:+.1f}, corr {out[mk]['corr']})", flush=True)
    RES.setdefault(name, {})[st] = out


def main() -> int:
    for st in ("val", "fwd"):
        c = lambda tag, s=0: ROOT / "cache" / f"{st}_{tag}_s{s}.parquet"  # noqa: E731
        ids = pd.read_parquet(c("m6r27skwing"), columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()
        M = INFO.reindex(ids)
        y = M["y"].to_numpy()
        G = (M["ap"].to_numpy() == "LIRF") & ~M["nm"].to_numpy()
        body = (y <= 3600) & ~G
        clean_nonG = ~G & (y <= 50000)
        day = DAY.reindex(ids).to_numpy()
        masks = {"body": body, "nonG_clean": clean_nonG}
        print(f"=== {st}: rows {len(y)}, body {int(body.sum())}, G {int(G.sum())}", flush=True)
        # L: LightGBM legs (G-band legs; non-G rows are the model3 leg's)
        P = {"win27": cpred(c("m6r27skwing"), ids), "none27": cpred(c("m6r27skg"), ids)}
        win1 = cpred(MAC / f"{st}_m6r27skwin_s1.parquet", ids)
        none1 = cpred(MAC / f"{st}_m6r27sk_s1.parquet", ids)
        s26, a26 = cpred(c("m6r26sg"), ids), cpred(c("m6r26as8g"), ids)
        none0_raw = cpred(c("m6r27sk"), ids)
        P["none27_s1"] = P["none27"] + (none1 - none0_raw)
        P["win27_s1"] = np.where(G, P["win27"], win1)
        P["aobt_tr"] = P["none27"] + (a26 - s26)
        P["none26"], P["aobt26"] = s26, a26
        d_win, d_aobt, d_seed = (P["win27"] - P["none27"])[body], (a26 - s26)[body], (none1 - none0_raw)[body]
        cr = {"win~aobt": float(np.corrcoef(d_win, d_aobt)[0, 1]), "win~seed": float(np.corrcoef(d_win, d_seed)[0, 1]),
              "aobt~seed": float(np.corrcoef(d_aobt, d_seed)[0, 1]),
              "rms_win": float(np.sqrt(np.mean(d_win ** 2))), "rms_aobt": float(np.sqrt(np.mean(d_aobt ** 2))), "rms_seed_none": float(np.sqrt(np.mean(d_seed ** 2))),
              "rms_seed_win": float(np.sqrt(np.mean((win1 - P["win27"])[body] ** 2)))}
        RES.setdefault("L_diff", {})[st] = {k: round(v, 4) for k, v in cr.items()}
        print(f"[L {st} body] anchor-effect vectors: {RES['L_diff'][st]}", flush=True)
        stats("L", st, P, y, masks, day, ("win27", "none27"), {"aobt_transplant": P["aobt_tr"], "placebo_none_s1": P["none27_s1"], "placebo_win_s1": P["win27_s1"]})
        del P
        # B: blend stage, C: corrector stage
        for nm_, a27, b27, s26t, a26t, b497 in (("B", "m6r27wincbrmfw4sv114", "m6r27skcbrmfw4sv114", "m6r26sgc75", "m6r26as8gc75", "m6r27skcbrmfw4"),
                                                ("C", "m6r27wincbrmfw4sv114adsbgeondctxv3x50wc3", "m6r27skcbrmfw4sv114adsbgeondctxv3x50wc3",
                                                 "m6r26sgc75adsbw", "m6r26as8gc75adsbw", "m6r27skcbrmfw4adsbgeondctxv3x50w")):
            P = {"A": cpred(c(a27), ids), "B": cpred(c(b27), ids)}
            s26, a26 = cpred(c(s26t), ids), cpred(c(a26t), ids)
            P["aobt_tr"] = P["B"] + (a26 - s26)
            P["none26"], P["aobt26"] = s26, a26
            P["B_497"] = cpred(c(b497), ids)
            dd = {"win~aobt": float(np.corrcoef((P["A"] - P["B"])[body], (a26 - s26)[body])[0, 1]),
                  "rms_AB": float(np.sqrt(np.mean((P["A"] - P["B"])[body] ** 2))), "rms_aobt26": float(np.sqrt(np.mean((a26 - s26)[body] ** 2))),
                  "rms_B_vs_497": float(np.sqrt(np.mean((P["B_497"] - P["B"])[body] ** 2)))}
            RES.setdefault(nm_ + "_diff", {})[st] = {k: round(v, 4) for k, v in dd.items()}
            print(f"[{nm_} {st} body] {RES[nm_ + '_diff'][st]}", flush=True)
            stats(nm_, st, P, y, masks, day, ("A", "B"), {"aobt_transplant": P["aobt_tr"], "placebo_B497": P["B_497"]})
            analytic(nm_ + "_asgood", st, P["A"], P["B"], y, masks)
            del P
        # F: final stage of 500's two chains
        mA = json.loads((ROOT / "preds" / "manifest_m6r27wincbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja.json").read_text(encoding="utf-8"))
        mB = json.loads((ROOT / "preds" / "manifest_m6r27skcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja.json").read_text(encoding="utf-8"))
        key = "holdout" if st == "val" else "forward"
        analytic("F_asgood", st, cpred(ROOT / mA[key]["path"], ids), cpred(ROOT / mB[key]["path"], ids), y, {**masks, "all": np.ones(len(y), bool)})
    # H: R5 harness, two-seed families
    for r in ("val", "fwd", "jj"):
        def load(cfg, s):
            if r == "jj":
                return pd.concat([load_r(cfg, "jb", s), load_r(cfg, "ja", s)], ignore_index=True)
            return load_r(cfg, r, s)

        def load_r(cfg, rr, s):
            return pd.read_parquet(H / f"{cfg}_{rr}_s{s}.parquet", columns=["MVT_ID_mvt", "day", "g", "y", "pred"])
        fr = {cfg: [load(cfg, s) for s in (0, 1)] for cfg in ("base", "a_win", "a_aobt")}
        b0 = fr["base"][0]
        for cfg in fr:
            for f in fr[cfg]:
                assert np.array_equal(f["MVT_ID_mvt"].to_numpy(), b0["MVT_ID_mvt"].to_numpy())
        y, g, day = b0["y"].to_numpy(), b0["g"].to_numpy().astype(bool), b0["day"].astype(str).to_numpy()
        body = (y <= 3600) & ~g
        nog = ~g & (y <= 50000)
        n = len(y)
        two_seed = {cfg: 0.5 * (f[0]["pred"].to_numpy() + f[1]["pred"].to_numpy()) for cfg, f in fr.items()}
        v = {cfg: 0.5 * (f[0]["pred"].to_numpy() - f[1]["pred"].to_numpy()) ** 2 for cfg, f in fr.items()}  # per-row single-seed noise variance estimate
        out = {}
        for mk, m in (("body", body), ("nog", nog)):
            two = 0.5 * (two_seed["base"] + two_seed["a_win"])
            three = (two_seed["base"] + two_seed["a_win"] + two_seed["a_aobt"]) / 3
            d = np.where(m, (three - y) ** 2 - (two - y) ** 2, 0.0)
            sd, p = boot(d, day, n)
            noise = float(((v["base"] + v["a_win"] + v["a_aobt"]) / 18 - (v["base"] + v["a_win"]) / 8)[m].sum() / n)
            alt = {}
            for pr, (x1, x2) in {"win+aobt": ("a_win", "a_aobt"), "base+aobt": ("base", "a_aobt")}.items():
                tw = 0.5 * (two_seed[x1] + two_seed[x2])
                alt[pr] = round(float(np.where(m, (tw - y) ** 2 - (two - y) ** 2, 0.0).sum() / n), 1)
            out[mk] = {"d3": round(float(d.sum() / n), 1), "sd": round(sd, 1), "p_below0": round(p, 4), "seed_noise_part": round(noise, 1),
                       "d3_minus_noise": round(float(d.sum() / n) - noise, 1), "other_pairs_vs_base+win": alt,
                       "mse_two": round(float(((two - y)[m] ** 2).sum() / n), 1)}
            print(f"[H {r} {mk}] 3-way(base,win,aobt) - 2-way(base,win): {out[mk]['d3']:+.1f} (sd {sd:.1f}, P<0 {p:.3f}), expected seed-noise part "
                  f"{noise:+.1f}, rest {out[mk]['d3_minus_noise']:+.1f} | other 2-ways vs base+win {alt}", flush=True)
        RES.setdefault("H", {})[r] = out
    (OUT / "f1_feas.json").write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote f1_feas.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
