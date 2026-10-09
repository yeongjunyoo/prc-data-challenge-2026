# SPDX-License-Identifier: GPL-3.0-only
"""F2 information for the coordinator's row-split spec, the scale2 cell of the Feb/Jun rows with an ADS-B match (private note 1.8 addendum,
registered before these numbers existed; post hoc, raised after chain A's gate14 numbers; not a gate). scale2_gate gave 0.75 - 0.5 = -368.5 (T)
and -467.0 (L) on the Feb/Jun 2025 rows, none of which has a match, while the fwd form (stage 2 fitted on January, applied to July 2025, 60% of
rows matched) gave +114 to +148 per corrector seed. Fits nothing; split by match (id matched in cache/adsb_v1nd.parquet):
  (a) scale2's whole-month Jan/Jul forms from research/f2/out/scale2_<chain>_V.parquet: LV4 (the four-month form), LV2, I_V, U_V, each
      <form>_s75 - <form>_s50, three-chain average (per-row mean, 1/3 each) and per chain;
  (b) 531 - 514 at the corrector stage, per corrector seed 0-2, settings fwd and val, three-chain average and per chain, from the per-seed
      outputs research/f2/seed_spread.py reads (514 <chain>adsbgeondctxv3x50w, 531 <chain>adsbgeondctxv3x50ws75; looked up under --root).
On all, matched and unmatched rows, and per month; mean over the subset's own rows; day-block bootstrap (research/f2/m3_corr.boot copied: UTC
day, 2,000 draws, seed 0).
    python -u research/f2/s75_matched.py [--root <mac-workdir>/prcm] [--out research/f2/out/s75_matched.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
CHAINS = {"A": "m6r27wincbrmfw4sv434tmc", "B": "m6r27skcbrmfw4sv434", "C": "m6r27apcbrmfw4sv114"}
SBX514 = {"A": "f2_sbx", "B": "r10b_sbx", "C": "r13f3_sbx"}
FORMS = ("LV4", "LV2", "I_V", "U_V")


def boot(d: np.ndarray, day: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """research/f2/m3_corr.boot verbatim: day-block bootstrap of mean(d): point, sd, P(<0)."""
    codes, days = pd.factorize(day)
    s = np.bincount(codes, weights=d)
    n = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    est = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return {"dmse": round(float(d.mean()), 2), "sd": round(float(est.std()), 2), "p_lt0": round(float((est < 0).mean()), 4), "days": int(len(s))}


def split(dd: np.ndarray, matched: np.ndarray, day: np.ndarray) -> dict:
    mon = np.array([x[5:7] for x in day])
    ok = np.isfinite(dd)
    r = {}
    for name, s in (("all", ok), ("matched", ok & matched), ("unmatched", ok & ~matched)):
        r[name] = boot(dd[s], day[s])
        for mm in sorted(set(mon[ok])):
            t = s & (mon == mm)
            if t.any():
                r[f"{name}_{mm}"] = boot(dd[t], day[t])
    return r


def find(root: Path, sbx: str, name: str) -> Path:
    p = root / sbx / "cache" / name
    if p.exists():
        return p
    alt = sorted(root.glob(f"*/cache/{name}"))
    assert alt, f"{name} not found under {root}"
    return alt[0]


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--root", default=str(Path(__import__("os").environ.get("PRC_WORK", "."))))
    pa.add_argument("--dir", default="research/f2/out")
    pa.add_argument("--adsb", default="cache/adsb_v1nd.parquet")
    pa.add_argument("--ctx", default=None, help="table with MVT_ID_mvt and day for (b) (default <root>/f2_sbx/cache/ctx_v3.parquet)")
    pa.add_argument("--out", default="research/f2/out/s75_matched.json")
    g = pa.parse_args()
    root = Path(g.root).expanduser()
    ids_m = pq.read_table(g.adsb, columns=["MVT_ID_mvt"], filters=[("matched", "=", True)]).column(0).to_numpy()
    rep = {"what": __doc__.split("\n")[0], "a_whole_month": {}, "b_corrector_531_minus_514": {}}
    # (a) whole-month Jan/Jul forms of scale2_gate
    P = {ch: pd.read_parquet(Path(g.dir) / f"scale2_{ch}_V.parquet").set_index("MVT_ID_mvt") for ch in CHAINS}
    idx = P["A"].index
    for ch in ("B", "C"):
        assert len(P[ch]) == len(idx) and P[ch].index.isin(idx).all(), f"scale2 {ch}: row set differs"
        P[ch] = P[ch].reindex(idx)
    y = P["A"]["y"].to_numpy(float)
    day = P["A"]["day"].astype(str).to_numpy()
    matched = np.asarray(idx.isin(ids_m))
    for form in FORMS:
        if f"{form}_s75" not in P["A"].columns:
            continue
        r = {}
        for ch in list(CHAINS) + ["avg"]:
            if ch == "avg":
                p75 = np.mean([P[c][f"{form}_s75"].to_numpy(float) for c in CHAINS], axis=0)
                p50 = np.mean([P[c][f"{form}_s50"].to_numpy(float) for c in CHAINS], axis=0)
            else:
                p75, p50 = P[ch][f"{form}_s75"].to_numpy(float), P[ch][f"{form}_s50"].to_numpy(float)
            r[ch] = split((p75 - y) ** 2 - (p50 - y) ** 2, matched, day)
        rep["a_whole_month"][form] = r
        a = r["avg"]
        print(f"(a) {form}_s75 - {form}_s50 avg: all {a['all']['dmse']:+.1f} (P {a['all']['p_lt0']:.3f}), matched {a['matched']['dmse']:+.1f} "
              f"(sd {a['matched']['sd']:.1f}, P {a['matched']['p_lt0']:.3f}), unmatched {a['unmatched']['dmse']:+.1f} (sd {a['unmatched']['sd']:.1f}, "
              f"P {a['unmatched']['p_lt0']:.3f}); per chain matched {[r[c]['matched']['dmse'] for c in CHAINS]}")
    # (b) corrector-stage 531 - 514 per seed
    ctx = Path(g.ctx) if g.ctx else root / "f2_sbx/cache/ctx_v3.parquet"
    dayof = pd.read_parquet(ctx, columns=["MVT_ID_mvt", "day"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")["day"].astype(str)
    for st in ("fwd", "val"):
        R = {}
        for s in (0, 1, 2):
            pr, ids, yy = {}, None, None
            for v, suf in (("514", "adsbgeondctxv3x50w"), ("531", "adsbgeondctxv3x50ws75")):
                for ch, base in CHAINS.items():
                    f = find(root, SBX514[ch] if v == "514" else "var1_sbx", f"{st}_{base}{suf}_s{s}.parquet")
                    df = pd.read_parquet(f, columns=["MVT_ID_mvt", "pred", "y"])
                    if ids is None:
                        ids, yy = df["MVT_ID_mvt"].to_numpy(), df["y"].to_numpy(float)
                    df = df.set_index("MVT_ID_mvt")
                    assert len(df) == len(ids) and df.index.isin(ids).all(), f"{f.name}: row set differs"
                    pr[(v, ch)] = df.reindex(ids)["pred"].to_numpy(float)
            dd = dayof.reindex(ids)
            assert dd.notna().all(), f"{st}: rows without a day"
            dd = dd.to_numpy()
            mt = np.isin(ids, ids_m)
            R[f"s{s}"] = {}
            for ch in list(CHAINS) + ["avg"]:
                if ch == "avg":
                    p531 = np.mean([pr[("531", c)] for c in CHAINS], axis=0)
                    p514 = np.mean([pr[("514", c)] for c in CHAINS], axis=0)
                else:
                    p531, p514 = pr[("531", ch)], pr[("514", ch)]
                R[f"s{s}"][ch] = split((p531 - yy) ** 2 - (p514 - yy) ** 2, mt, dd)
            a = R[f"s{s}"]["avg"]
            print(f"(b) {st} seed {s} 531-514 avg: all {a['all']['dmse']:+.1f} (P {a['all']['p_lt0']:.3f}), matched {a['matched']['dmse']:+.1f} "
                  f"(sd {a['matched']['sd']:.1f}, P {a['matched']['p_lt0']:.3f}), unmatched {a['unmatched']['dmse']:+.1f} (sd {a['unmatched']['sd']:.1f}, "
                  f"P {a['unmatched']['p_lt0']:.3f}); matched share {mt.mean():.4f}")
        rep["b_corrector_531_minus_514"][st] = R
    out = Path(g.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"S75MATCHED written {out}")


if __name__ == "__main__":
    main()
