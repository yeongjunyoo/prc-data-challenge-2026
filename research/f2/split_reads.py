# SPDX-License-Identifier: GPL-3.0-only
"""F2 reads for the final split (coordinator 10-09 23:4x; vault research/2026-10-08-f2-four-months.md 5.8). No fits; reads only; run on the Mac
from ~/nova-adsb/prcm/f2_run, one thread.
(1) Matched-row scale2, airport split (registered 23:44:30 before any of these numbers): July matched rows, 0.75 - 0.5, three-chain average, for the
    airports January 2025 covers (matched share > 0 among January val rows, expected EDDF, EDDM, EGLL, EHAM, LEBL, LSZH) and the rest, per ruler:
    LV2 July and LV4 July (research/f2/out/scale2_<chain>_V.parquet), fwd 531 - 514 per corrector seed (the per-seed outputs research/f2/seed_spread.py
    reads); I_V July as information. Day-block bootstrap as research/f2/m3_corr.boot. Rule: 0.75 for matched rows iff LV2 July, LV4 July and the fwd
    seed mean are all <= max(0, sd over seeds of fwd) at the covered airports; else 0.5; not computable: 0.5.
(2) L's added rows: the three fj files' sha16 (the gate's), their months, and no id among the matched ids of f2_run's adsb_v1nd (0d03ff06267e0dee) or
    final_run's adsb_v1nd (07df03ca69833082); the two tables' matched id sets on 2025 rows compared.
(3) Groups as the kit will build them: Jan/Jul = the test ids in final_run/data/ranking.parquet, matched = final_run adsb_v1nd matched == True;
    expected 344,841 / 91,441 / 234,508; the ctx part of every group checked.
    python -u research/f2/split_reads.py  ->  research/f2/out/split_reads.json
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

P = Path.home() / "nova-adsb/prcm"
RUN, FR = P / "f2_run", P / "final_run"
CH = {"A": "m6r27wincbrmfw4sv434tmc", "B": "m6r27skcbrmfw4sv434", "C": "m6r27apcbrmfw4sv114"}
SBX514 = {"A": "f2_sbx", "B": "r10b_sbx", "C": "r13f3_sbx"}
EXTRA = {"A": ("f2wincbrmfw4", "bbdfbccf5821da14"), "B": ("f2skcbrmfw4", "cd2cf1ea95631aad"), "C": ("f2apcbrmfw4", "c0901a4fd4121cc7")}


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


def boot(d, day, n_boot=2000, seed=0):
    codes, days = pd.factorize(day)
    s = np.bincount(codes, weights=d)
    n = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    est = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return {"dmse": round(float(d.mean()), 2), "sd": round(float(est.std()), 2), "p_lt0": round(float((est < 0).mean()), 4), "days": int(len(s)), "rows": int(len(d))}


def find(name: str, sbx: str) -> Path:
    p = P / sbx / "cache" / name
    if p.exists():
        return p
    alt = sorted(P.glob(f"*/cache/{name}"))
    assert alt, name
    return alt[0]


rep = {"registered": "vault 5.8, 2026-10-09 23:44:30 KST"}
m_run = pq.read_table(RUN / "cache/adsb_v1nd.parquet", columns=["MVT_ID_mvt"], filters=[("matched", "=", True)]).column(0).to_numpy()
rep["f2_run_adsb_v1nd_sha16"] = sha16(RUN / "cache/adsb_v1nd.parquet")
ap = pd.read_parquet(RUN / "cache/feats_v25.parquet", columns=["MVT_ID_mvt", "ap"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")["ap"].astype(str)

# (1) airport split
V = {ch: pd.read_parquet(RUN / f"research/f2/out/scale2_{ch}_V.parquet").set_index("MVT_ID_mvt") for ch in CH}
ids = V["A"].index
for ch in ("B", "C"):
    V[ch] = V[ch].reindex(ids)
y = V["A"]["y"].to_numpy(float)
day = V["A"]["day"].astype(str).to_numpy()
mon = np.array([d[5:7] for d in day])
apt = ap.reindex(ids).to_numpy()
matched = np.isin(ids.to_numpy(), m_run)
jan_share = {a: round(float(matched[(mon == "01") & (apt == a)].mean()), 4) for a in sorted(set(apt))}
jul_share = {a: round(float(matched[(mon == "07") & (apt == a)].mean()), 4) for a in sorted(set(apt))}
covered = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LSZH"]  # the six named in the registration (23:44:30); LEMD 0.03% and LFPG 0.01% January matched shares are not coverage
rep["jan_matched_share"], rep["jul_matched_share"], rep["covered"] = jan_share, jul_share, covered
cov = np.isin(apt, covered)
res = {}
for form in ("LV2", "LV4", "I_V"):
    p75 = np.mean([V[c][f"{form}_s75"].to_numpy(float) for c in CH], axis=0)
    p50 = np.mean([V[c][f"{form}_s50"].to_numpy(float) for c in CH], axis=0)
    d = (p75 - y) ** 2 - (p50 - y) ** 2
    r = {}
    for gname, gm in (("covered", cov), ("uncovered", ~cov)):
        s = (mon == "07") & matched & gm & np.isfinite(d)
        r[gname] = boot(d[s], day[s])
    r["by_airport"] = {a: round(float(d[(mon == "07") & matched & (apt == a)].mean()), 1) for a in sorted(set(apt)) if ((mon == "07") & matched & (apt == a)).any()}
    sj = (mon == "01") & matched & np.isfinite(d)
    r["january_matched_all"] = boot(d[sj], day[sj])
    res[form + "_july"] = r
ctx = pd.read_parquet(P / "f2_sbx/cache/ctx_v3.parquet", columns=["MVT_ID_mvt", "day"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")["day"].astype(str)
fw = {"covered": [], "uncovered": [], "by_airport": {}}
for s in (0, 1, 2):
    pr, fids, fy = {}, None, None
    for v, suf, sb in (("514", "adsbgeondctxv3x50w", None), ("531", "adsbgeondctxv3x50ws75", "var1_sbx")):
        for ch, t in CH.items():
            df = pd.read_parquet(find(f"fwd_{t}{suf}_s{s}.parquet", sb or SBX514[ch]), columns=["MVT_ID_mvt", "pred", "y"])
            if fids is None:
                fids, fy = df["MVT_ID_mvt"].to_numpy(), df["y"].to_numpy(float)
            pr[(v, ch)] = df.set_index("MVT_ID_mvt").reindex(fids)["pred"].to_numpy(float)
    d = (np.mean([pr[("531", c)] for c in CH], axis=0) - fy) ** 2 - (np.mean([pr[("514", c)] for c in CH], axis=0) - fy) ** 2
    fday = ctx.reindex(fids).to_numpy()
    fapt = ap.reindex(fids).to_numpy()
    fm = np.isin(fids, m_run)
    fc = np.isin(fapt, covered)
    for gname, gm in (("covered", fc), ("uncovered", ~fc)):
        sel = fm & gm & np.isfinite(d)
        fw[gname].append(boot(d[sel], fday[sel]))
    fw["by_airport"][f"s{s}"] = {a: round(float(d[fm & (fapt == a)].mean()), 1) for a in sorted(set(fapt)) if (fm & (fapt == a)).any()}
fw["covered_seed_values"] = [x["dmse"] for x in fw["covered"]]
fw["covered_seed_mean"] = round(float(np.mean(fw["covered_seed_values"])), 2)
fw["covered_seed_sd"] = round(float(np.std(fw["covered_seed_values"], ddof=1)), 2)
res["fwd_531_minus_514"] = fw
thr = max(0.0, fw["covered_seed_sd"])
vals = {"LV2_july": res["LV2_july"]["covered"]["dmse"], "LV4_july": res["LV4_july"]["covered"]["dmse"], "fwd_seed_mean": fw["covered_seed_mean"]}
rep["split"] = res
rep["decision"] = {"values_at_covered": vals, "threshold": thr, "penalty_gone": all(v <= thr for v in vals.values()),
                   "matched_scale2": 0.75 if all(v <= thr for v in vals.values()) else 0.5}

# (2) L's added rows
fr_sha = sha16(FR / "cache/adsb_v1nd.parquet")
m_fr = pq.read_table(FR / "cache/adsb_v1nd.parquet", columns=["MVT_ID_mvt"], filters=[("matched", "=", True)]).column(0).to_numpy()
ex = {}
f2ctx = pd.read_parquet(RUN / "cache/f2ctx_v3.parquet", columns=["MVT_ID_mvt", "part"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")["part"].astype(str)
for ch, (t, want) in EXTRA.items():
    p = RUN / f"cache/fj_{t}_s0.parquet"
    eids = pd.read_parquet(p, columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()
    parts = f2ctx.reindex(eids)
    ex[ch] = {"file": p.name, "sha16": sha16(p), "gate_sha16": want, "rows": int(len(eids)), "parts": parts.value_counts(dropna=False).to_dict(),
              "matched_in_f2_run_adsb_v1nd": int(np.isin(eids, m_run).sum()), "matched_in_final_run_adsb_v1nd": int(np.isin(eids, m_fr).sum())}
rep["extras"] = ex
rep["final_run_adsb_v1nd_sha16"] = fr_sha
v_ids = ids.to_numpy()
rep["stage1_2025_matched_ids_equal"] = bool(np.array_equal(np.sort(v_ids[np.isin(v_ids, m_run)]), np.sort(v_ids[np.isin(v_ids, m_fr)])))

# (3) groups as the kit builds them
g = pd.read_parquet(RUN / "research/f2/out/final_groups.parquet")
rk = RUN / "research/f2/janjul_ids.parquet"  # final_run/data/ranking.parquet is the four-month final ranking; the Jan/Jul ids come from the original ranking.parquet (9db75e3a56fa09b8)
if rk.exists():
    rids = pd.read_parquet(rk, columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()
    jj = np.isin(g["MVT_ID_mvt"].to_numpy(), rids)
    mt = np.isin(g["MVT_ID_mvt"].to_numpy(), m_fr)
    part = g["part"].astype(str).to_numpy()
    rep["groups_ranking_def"] = {"ranking_rows": int(len(rids)), "jj": int(jj.sum()), "fu": int((~jj & ~mt).sum()), "fm": int((~jj & mt).sum()),
                                 "jj_parts": pd.Series(part[jj]).value_counts().to_dict(), "rest_parts": pd.Series(part[~jj]).value_counts().to_dict(),
                                 "matched_equals_23_33_map": bool(np.array_equal(mt, g["matched"].to_numpy())), "ranking_sha16": sha16(rk)}
else:
    rep["groups_ranking_def"] = f"missing {rk}"
out = RUN / "research/f2/out/split_reads.json"
out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"covered": covered, "decision": rep["decision"], "LV2_july": res["LV2_july"]["covered"], "LV4_july": res["LV4_july"]["covered"],
                  "fwd_cov": fw["covered"], "fwd_uncov": fw["uncovered"], "LV2_unc": res["LV2_july"]["uncovered"], "LV4_unc": res["LV4_july"]["uncovered"],
                  "extras": {k: {x: v[x] for x in ("sha16", "gate_sha16", "rows", "parts", "matched_in_f2_run_adsb_v1nd", "matched_in_final_run_adsb_v1nd")} for k, v in ex.items()},
                  "stage1_ids_equal": rep["stage1_2025_matched_ids_equal"], "groups": rep["groups_ranking_def"]}, ensure_ascii=False, indent=1))
