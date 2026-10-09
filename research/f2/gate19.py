# SPDX-License-Identifier: GPL-3.0-only
"""F2 gate 1.9 (vault research/2026-10-08-f2-four-months.md 1.9, registered 2026-10-10 06:10:55 KST before any number; rules from the master's
「다음 지렛대 등록」 06:01): do ADS-B matches on the four-month corrector's extra rows (Feb/Jun 2025) help the held-out Feb/Jun month?

gate14.py's world and code (blend world, chains A, B, C, corrector seed 0, float64, 514's arguments, context f2ctx_v3, fit store = a clone of
fitcache_g14 at --threads 2), with the stage-1 tables of research/f2/fj_tables.py (today's rows unchanged plus the fj rows). Month held out:
the scored month's rows are never in stage-2 training; every row of the other F month is an extra row. Scored rows get stage 1 from the new
tables (mean of the ten fold models on matched rows), as the final's Feb/Jun 2026 rows do.
Forms (judged in the final's own form, the mean of the ten S_k; S_k trains on V's other folds with pairwise-nested stage-1 inputs, as
research/f2/ctx_corr4.stage2_val_models4, plus the extra rows):
  T  : S_k on V only (no extra rows)
  L0 : S_k + the other month's rows WITHOUT stage-1 inputs (the old tables have no Feb/Jun 2025 day: today's final)
  Lm : S_k + the other month's rows WITH stage-1 inputs and match columns from the new tables (the lever)
  information: gate 1.4's one-model forms T1, L0_1 (= gate14's L), Lm_1
scale2 per scored row: 0.5 when matched in the new tables, 0.75 otherwise (Bb); LOBT window last.
Three-chain average (per-row mean), F-row denominator, day bootstrap (UTC MVT day, 2,000, seed 0); matched/unmatched/month/chain on own rows.
Rules: adopt iff Lm - L0 <= -20 with P(<0) >= 0.9 and the matched-row point estimate of Lm - L0 <= 0.
       safety: on matched rows, L - T > 0 with P(<0) <= 0.1, with L = Lm if adopted else L0 -> the final's matched Feb/Jun rows back to T at 0.5.
Implementation checks before any number counts (failure: no numbers, exit 3):
  (1) stage 1, T's S_k, T1 and gate 1.4's one-model L load from the store (no fit);
  (2) corrected 07:4x before any number: the same models on the old tables' design of the scored rows at scale2 0.5 reproduce
      gate14_<chain>_F.parquet's T, T1, L on every F row bit for bit (the first form, "unmatched rows equal", was wrong: aiu_m_gap and
      aiu_d_gap are airport-month/day means of s1, so matched rows' stage 1 moves unmatched rows' inputs).
    PYTHONDONTWRITEBYTECODE=1 python -u research/f2/gate19.py --adsb adsb_v1ndfj --adsb_sha <16> --geo geo_v1ndfj --geo_sha <16>
        --ctx f2ctx_v3 --ctx_sha 531e8113e7f26fff --threads 2 --fit_cache research/f2/fitcache_g19
Writes research/f2/out/gate19.json and gate19_<chain>_F.parquet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "research/f2")
import src.adsb_corr as ac  # noqa: E402
import src.ctx_corr as cc  # noqa: E402
import src.ctx_corr2 as c2  # noqa: E402
from src.load import CACHE  # noqa: E402
import m3_corr as m3  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
T0 = time.time()
OUT = Path("research/f2/out")
DEFAULT = "A:m6r27wincbrmfw4sv434tmc:f2wincbrmfw4,B:m6r27skcbrmfw4sv434:f2skcbrmfw4,C:m6r27apcbrmfw4sv114:f2apcbrmfw4"
CALLS: list = []


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def counted(store: Path):
    """Wrap the fit-cache-routed ac.fit / cc.fit2 so every call records (label, fitted)."""
    f1, f2 = ac.fit, cc.fit2
    state = {"label": ""}

    def n():
        return len(os.listdir(store))

    def w1(*x, **kw):
        b = n()
        m = f1(*x, **kw)
        CALLS.append((state["label"] + ":s1", n() > b))
        return m

    def w2(*x, **kw):
        b = n()
        m = f2(*x, **kw)
        CALLS.append((state["label"], n() > b))
        return m
    ac.fit, cc.fit2 = w1, w2
    return state


def chain(name, vbase, fbase, a, A_new, A_old, ctx, lab, g14):
    V = pd.read_parquet(CACHE / f"val_{vbase}_s0.parquet")
    F = pd.read_parquet(CACHE / f"fj_{fbase}_s0.parquet")
    T = ac.design(V, A_new)
    T = T[np.isfinite(T["y"].to_numpy())]
    assert len(ac.design(V, A_old)) == len(ac.design(V, A_new)), "V's matched rows differ between the tables"
    RV, RF, RF0 = cc.Rows(V, A_new, a, ctx), cc.Rows(F, A_new, a, ctx), cc.Rows(F, A_old, a, ctx)
    assert len(RF0.X) == 0, "the old tables match some F rows"
    assert set(np.unique(RV.day)) == set(T["day"].unique())
    yV, yF = V["y"].to_numpy(dtype=float), F["y"].to_numpy(dtype=float)
    keepV, keepF = np.isfinite(yV), np.isfinite(yF)
    lab["label"] = f"{name}:stage1"
    fo, fT, models = cc.stage1_folds(T, a)
    oof = np.zeros(len(T))
    for j, m in enumerate(models):
        oof[fT == j] = ac.predict(m, T[fT == j])
    T_pos = T["pos"].to_numpy()
    d1V = np.zeros(len(V))
    d1V[T_pos] = oof
    d1F = np.zeros(len(F))
    d1F[RF.X["pos"].to_numpy()] = np.mean([ac.predict(m, RF.X) for m in models], axis=0)
    zF = np.zeros(len(F))
    s1V = np.clip(V["pred"].to_numpy(dtype=float) + d1V, 0, None)
    s1F = np.clip(F["pred"].to_numpy(dtype=float) + d1F, 0, None)
    s1F0 = np.clip(F["pred"].to_numpy(dtype=float) + zF, 0, None)
    MV, MF, MF0 = RV.design2(s1V, d1V), RF.design2(s1F, d1F), RF0.design2(s1F0, zF)
    appV, appF, appF0 = s1V < a.max_pred, s1F < a.max_pred, s1F0 < a.max_pred
    tV, tF, tF0 = yV - s1V, yF - s1F, yF - s1F0
    fold_row = pd.Series(RV.day).map(fo).to_numpy()
    monF = pd.Series(RF.day).str[5:7].astype(int).to_numpy()
    matched = RF.matched == 1
    log(f"{name}: V {len(V):,} (matched {len(T):,}), F {len(F):,} (matched {int(matched.sum()):,}); nested stage 1 next")
    nested = cc.nested_stage1(T, fT, a)
    fit = lambda M, t: cc.fit2(M, t, a.seed, a.rounds2, a.threads)  # noqa: E731
    ex0 = {o: keepF & appF0 & (monF == o) for o in (2, 6)}
    exm = {o: keepF & appF & (monF == o) for o in (2, 6)}
    sel = {h: appF & (monF == h) for h in (2, 6)}
    S, pL0, pLm = [], {2: [], 6: []}, {2: [], 6: []}
    for k in range(a.folds):
        dm = np.zeros(len(T))
        for j in range(a.folds):
            if j != k:
                dm[fT == j] = nested[(j, k)]
        delta = RV.stage1(dm, T_pos)
        s1 = np.clip(V["pred"].to_numpy(dtype=float) + delta, 0, None)
        tr = (fold_row != k) & keepV & (s1 < a.max_pred)
        Mk = RV.design2(s1, delta)
        lab["label"] = f"{name}:T_Sk"
        S.append(fit(Mk[tr], (yV - s1)[tr]))
        for h, o in ((2, 6), (6, 2)):
            lab["label"] = f"{name}:L0_Sk"
            m0 = fit(np.vstack([Mk[tr], MF0[ex0[o]]]), np.concatenate([(yV - s1)[tr], tF0[ex0[o]]]))
            pL0[h].append(m0.predict(MF[sel[h]]))
            lab["label"] = f"{name}:Lm_Sk"
            mm = fit(np.vstack([Mk[tr], MF[exm[o]]]), np.concatenate([(yV - s1)[tr], tF[exm[o]]]))
            pLm[h].append(mm.predict(MF[sel[h]]))
        del Mk
        log(f"{name}: fold {k} done")
    xT, xL0, xLm = np.zeros(len(F)), np.zeros(len(F)), np.zeros(len(F))
    xT[appF] = np.mean([m.predict(MF[appF]) for m in S], axis=0)
    for h in (2, 6):
        xL0[sel[h]] = np.mean(pL0[h], axis=0)
        xLm[sel[h]] = np.mean(pLm[h], axis=0)
    trV = keepV & appV
    lab["label"] = f"{name}:T1"
    m1 = fit(MV[trV], tV[trV])
    x1, xL01, xLm1 = np.zeros(len(F)), np.zeros(len(F)), np.zeros(len(F))
    x1[appF] = m1.predict(MF[appF])
    mL01 = {}
    for h, o in ((2, 6), (6, 2)):
        lab["label"] = f"{name}:L0_1"
        mL01[h] = fit(np.vstack([MV[trV], MF0[ex0[o]]]), np.concatenate([tV[trV], tF0[ex0[o]]]))
        xL01[sel[h]] = mL01[h].predict(MF[sel[h]])
        lab["label"] = f"{name}:Lm_1"
        xLm1[sel[h]] = fit(np.vstack([MV[trV], MF[exm[o]]]), np.concatenate([tV[trV], tF[exm[o]]])).predict(MF[sel[h]])
    sc = np.where(matched, 0.5, 0.75)
    w = lambda x, s=sc: m3.windowed(F, np.clip(s1F + s * x, 0, None), a.feats)  # noqa: E731
    # implementation check (2), corrected 07:4x before any number (vault 1.9): the same models on the OLD tables' design of the scored rows
    # (no stage 1, matched 0, s1 = base; aiu_m_gap/aiu_d_gap are airport-month/day means of s1, so they must be the old frame's) at scale2
    # 0.5 reproduce gate14's T, T1 and L on every F row bit for bit.
    ids = F["MVT_ID_mvt"].to_numpy()
    G = g14.set_index("MVT_ID_mvt").reindex(ids)
    sel0 = {h: appF0 & (monF == h) for h in (2, 6)}
    xT0, x10, xL010 = np.zeros(len(F)), np.zeros(len(F)), np.zeros(len(F))
    xT0[appF0] = np.mean([m.predict(MF0[appF0]) for m in S], axis=0)
    x10[appF0] = m1.predict(MF0[appF0])
    for h in (2, 6):
        xL010[sel0[h]] = mL01[h].predict(MF0[sel0[h]])
    w0 = lambda x: m3.windowed(F, np.clip(s1F0 + 0.5 * x, 0, None), a.feats)  # noqa: E731
    chk = {}
    for mine, theirs in ((xT0, "T"), (x10, "T1"), (xL010, "L")):
        a05, b05 = w0(mine), G[theirs].to_numpy(dtype=float)
        chk[theirs] = int((a05.view(np.int64) != b05.view(np.int64)).sum())
    # information: unmatched rows whose airport-day and airport-month hold no matched row keep gate14's inputs under the new tables too
    gd = pd.Series(RF.ap) + "|" + pd.Series(RF.day)
    gm = pd.Series(RF.ap) + "|" + pd.Series(RF.part)
    pure = (~gd.isin(set(gd[matched])) & ~gm.isin(set(gm[matched]))).to_numpy()
    info = {"pure_unmatched_rows": int(pure.sum())}
    for mine, theirs in ((xT, "T"), (x1, "T1"), (xL01, "L")):
        a05, b05 = w(mine, 0.5)[pure], G[theirs].to_numpy(dtype=float)[pure]
        info[theirs] = int((a05.view(np.int64) != b05.view(np.int64)).sum())
    chk["info_pure_unmatched_new_tables"] = info
    PF = pd.DataFrame({"MVT_ID_mvt": ids, "y": yF, "day": RF.day, "matched": matched, "s1w": m3.windowed(F, s1F, a.feats),
                       "T": w(xT), "L0": w(xL0), "Lm": w(xLm), "T1": w(x1), "L0_1": w(xL01), "Lm_1": w(xLm1)})
    return PF, chk


def contrasts(PF: pd.DataFrame) -> dict:
    se = lambda k: (PF[k].to_numpy() - PF["y"].to_numpy()) ** 2  # noqa: E731
    day, mt = PF["day"].to_numpy(), PF["matched"].to_numpy().astype(bool)
    mon = PF["day"].str[5:7].to_numpy()
    r = {}
    for nm, (x, y) in {"Lm_minus_L0": ("Lm", "L0"), "L0_minus_T": ("L0", "T"), "Lm_minus_T": ("Lm", "T"),
                       "Lm1_minus_L01": ("Lm_1", "L0_1"), "L01_minus_T1": ("L0_1", "T1"), "Lm1_minus_T1": ("Lm_1", "T1"), "T_minus_T1": ("T", "T1")}.items():
        d = se(x) - se(y)
        r[nm] = {"all": m3.boot(d, day), "matched": m3.boot(d[mt], day[mt]), "unmatched": m3.boot(d[~mt], day[~mt]),
                 **{f"m{mm}": m3.boot(d[mon == mm], day[mon == mm]) for mm in ("02", "06")},
                 **{f"m{mm}_matched": m3.boot(d[(mon == mm) & mt], day[(mon == mm) & mt]) for mm in ("02", "06")}}
    for k in ("T", "L0", "Lm"):
        r[f"{k}_vs_stage1"] = m3.boot(se(k) - se("s1w"), day)
    r["rows"] = {"all": int(len(PF)), "matched": int(mt.sum()), "m02": int((mon == "02").sum()), "m06": int((mon == "06").sum())}
    return r


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--adsb", required=True)
    pa.add_argument("--adsb_sha", required=True)
    pa.add_argument("--geo", required=True)
    pa.add_argument("--geo_sha", required=True)
    pa.add_argument("--ctx", required=True)
    pa.add_argument("--ctx_sha", required=True)
    pa.add_argument("--chains", default=DEFAULT)
    pa.add_argument("--threads", type=int, default=2)
    pa.add_argument("--fit_cache", default="research/f2/fitcache_g19")
    pa.add_argument("--out", default="research/f2/out")
    g = pa.parse_args()
    global OUT
    OUT = Path(g.out)
    OUT.mkdir(parents=True, exist_ok=True)
    a = SimpleNamespace(adsb=g.adsb, adsb_sha=g.adsb_sha, geo=g.geo, geo_sha=g.geo_sha, feats="feats_v25", folds=10, rounds=300, rounds2=400,
                        max_pred=3600.0, threads=g.threads, seed=0, scale2=0.5)
    ac.below_normal()
    m3.setup()
    store = Path(g.fit_cache)
    assert store.exists() and any(store.iterdir()), "the fit store (a clone of fitcache_g14) is missing"
    c2.fit_cache(store)
    lab = counted(store)
    A_new = cc.stage1_inputs(a)  # reads and sha-checks the new tables once (stage1_inputs extends ac.ADSB_COLS, so it is not called twice)
    fj = set()
    for spec in g.chains.split(","):
        fj |= set(pd.read_parquet(CACHE / f"fj_{spec.split(':')[2]}_s0.parquet", columns=["MVT_ID_mvt"])["MVT_ID_mvt"])
    A_old = A_new[~A_new.index.isin(list(fj))]  # = today's tables' matched rows (fj_tables.py checked they are unchanged)
    ctx = cc.checked(g.ctx, g.ctx_sha).set_index("MVT_ID_mvt")
    rep = {"registered": "vault note 1.9, 2026-10-10 06:10:55 KST; rules master 06:01", "tables": {"adsb": [g.adsb, g.adsb_sha], "geo": [g.geo, g.geo_sha]},
           "ctx": [g.ctx, g.ctx_sha], "fit_cache": str(store), "threads": g.threads, "chains": {}}
    PFs, checks = [], {}
    for spec in g.chains.split(","):
        name, vb, fb = spec.split(":")
        g14 = pd.read_parquet(OUT / f"gate14_{name}_F.parquet")
        PF, chk = chain(name, vb, fb, a, A_new, A_old, ctx, lab, g14)
        checks[name] = chk
        PFs.append(PF.set_index("MVT_ID_mvt"))
        log(f"{name}: check (2) bits differing (old-table design, all rows) {chk}")
    must = [c for c in CALLS if c[0].split(":")[1] in ("stage1", "T_Sk", "T1", "L0_1")]
    fitted_must = [c[0] for c in must if c[1]]
    rep["fit_calls"] = {k: [sum(1 for c in CALLS if c[0] == k and not c[1]), sum(1 for c in CALLS if c[0] == k and c[1])]
                        for k in sorted({c[0] for c in CALLS})}
    rep["check1_must_load"] = {"calls": len(must), "fitted": fitted_must, "pass": not fitted_must}
    rep["check2_old_design_equals_gate14"] = {"bits_differing": checks,
                                              "pass": all(c[k] == 0 for c in checks.values() for k in ("T", "T1", "L"))}
    if not (rep["check1_must_load"]["pass"] and rep["check2_old_design_equals_gate14"]["pass"]):
        rep["result"] = "implementation check failed: numbers withheld"
        (OUT / "gate19.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"GATE 1.9 IMPLEMENTATION CHECK FAILED: {rep['check1_must_load']}, {rep['check2_old_design_equals_gate14']}")
        sys.exit(3)
    for P_, spec in zip(PFs, g.chains.split(",")):
        P_.reset_index().to_parquet(OUT / f"gate19_{spec.split(':')[0]}_F.parquet", index=False)
    for P_, spec in zip(PFs, g.chains.split(",")):
        rep["chains"][spec.split(":")[0]] = contrasts(P_.reset_index())
    PF = PFs[0].copy()
    assert all(np.array_equal(p["matched"].reindex(PF.index).to_numpy(), PF["matched"].to_numpy()) for p in PFs)
    for k in ("s1w", "T", "L0", "Lm", "T1", "L0_1", "Lm_1"):
        PF[k] = np.mean([p[k].reindex(PF.index).to_numpy() for p in PFs], axis=0)
    avg = contrasts(PF.reset_index())
    rep["three_chain_average"] = avg
    ad = avg["Lm_minus_L0"]
    adopt = bool(ad["all"]["dmse"] <= -20 and ad["all"]["p_lt0"] >= 0.9 and ad["matched"]["dmse"] <= 0)
    form = "Lm" if adopt else "L0"
    sf = avg[f"{form}_minus_T"]["matched"]
    safety = bool(sf["dmse"] > 0 and sf["p_lt0"] <= 0.1)
    rep["decision"] = {"adopt": adopt, "adopt_rule": {"Lm_minus_L0_all": ad["all"], "Lm_minus_L0_matched": ad["matched"]},
                       "final_form": form, "safety_fires": safety, "safety_rule": {f"{form}_minus_T_matched": sf},
                       "safety_both_forms_info": {"L0_minus_T_matched": avg["L0_minus_T"]["matched"], "Lm_minus_T_matched": avg["Lm_minus_T"]["matched"]},
                       "seed_noise_note": "corrector seed 0 only; seed_spread (note 5.6): fwd-form three-chain sd 17 to 21"}
    rep["seconds"] = round(time.time() - T0)
    (OUT / "gate19.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"GATE 1.9 adopt {adopt} (Lm-L0 {ad['all']}, matched {ad['matched']['dmse']}); safety on {form} {safety} ({sf})")


if __name__ == "__main__":
    main()
