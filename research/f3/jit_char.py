# SPDX-License-Identifier: GPL-3.0-only
"""F3 step 1: characterize the organizer's timestamp jitter in data/final_phase/final_ranking.parquet against data/ranking.parquet.
Read only. Writes research/f3/out/jit_char.json and prints a summary.
    python -X utf8 research/f3/jit_char.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("research/f3/out")
OUT.mkdir(parents=True, exist_ok=True)
TS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt"]
res: dict = {}


def secs(s: pd.Series) -> np.ndarray:
    """timestamp -> float seconds since epoch (NaN for NaT)"""
    v = s.astype("int64").to_numpy().astype(float) / 1e6 if s.dt.tz is None else s.dt.tz_convert("UTC").dt.tz_localize(None).astype("int64").to_numpy().astype(float)
    return v


def to_s(s: pd.Series) -> np.ndarray:
    out = np.full(len(s), np.nan)
    m = s.notna().to_numpy()
    out[m] = (s[m].dt.tz_convert("UTC").dt.tz_localize(None) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    return out


def desc(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    vals, cnt = np.unique(np.round(x, 6), return_counts=True)
    d = {"n": int(len(x)), "changed": int((x != 0).sum()), "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
         "sd": float(x.std()), "integer": bool(np.all(np.abs(x - np.round(x)) < 1e-6))}
    if len(vals) <= 60:
        d["hist"] = {str(v): int(c) for v, c in zip(vals, cnt)}
    return d


def main():
    r = pd.read_parquet("data/ranking.parquet")
    f = pd.read_parquet("data/final_phase/final_ranking.parquet")
    n0 = len(r)
    res["rows"] = {"ranking": n0, "final": len(f)}
    head = f.iloc[:n0].reset_index(drop=True)
    res["same_ids_order"] = bool((head["MVT_ID_mvt"].to_numpy() == r["MVT_ID_mvt"].to_numpy()).all())
    # columns identical?
    same = {}
    for c in r.columns:
        a, b = r[c], head[c]
        if c in TS:
            eq = ((a == b) | (a.isna() & b.isna())).all()
        else:
            eq = a.equals(b) or bool(((a == b) | (a.isna() & b.isna())).all())
        same[c] = bool(eq)
    res["identical_columns"] = same
    ph = r["PHASE_mvt"].to_numpy()
    res["phase_counts"] = {str(k): int(v) for k, v in pd.Series(ph).value_counts().items()}
    # null pattern per phase for key columns in both files
    nulls = {}
    for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]:
        for p in ["DEP", "ARR"]:
            m = ph == p
            nulls[f"{c}|{p}|ranking"] = int(r.loc[m, c].isna().sum())
            nulls[f"{c}|{p}|final_head"] = int(head.loc[m, c].isna().sum())
    fph = f["PHASE_mvt"].to_numpy()
    for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]:
        for p in ["DEP", "ARR"]:
            m = fph == p
            nulls[f"{c}|{p}|final_all"] = int(f.loc[m, c].isna().sum())
    res["nulls"] = nulls
    # timestamp deltas
    d = {}
    for c in TS:
        a, b = to_s(r[c]), to_s(head[c])
        d[c] = b - a
        for p in ["DEP", "ARR"]:
            m = ph == p
            res[f"delta|{c}|{p}"] = desc(d[c][m])
    dm, db = d["MVT_TIME_UTC_mvt"], d["BLOCK_TIME_UTC_mvt"]
    arr = ph == "ARR"
    dep = ph == "DEP"
    # ARR block vs MVT jitter
    both = arr & np.isfinite(dm) & np.isfinite(db)
    res["arr_block_plus_mvt"] = desc(db[both] + dm[both])
    res["arr_corr_block_mvt"] = float(np.corrcoef(db[both], dm[both])[0, 1])
    # joint table of (dm, db) for ARR
    jt = pd.crosstab(dm[both], db[both])
    jt.to_csv(OUT / "arr_joint_dm_db.csv")
    # taxi relations
    tx_r = r["TAXITIME_SEC_mvt"].to_numpy(float)
    tx_f = head["TAXITIME_SEC_mvt"].to_numpy(float)
    res["delta|TAXI|ARR"] = desc((tx_f - tx_r)[arr])
    res["delta|TAXI|DEP"] = desc((tx_f - tx_r)[dep])
    bm_r = to_s(r["BLOCK_TIME_UTC_mvt"]) - to_s(r["MVT_TIME_UTC_mvt"])
    bm_f = to_s(head["BLOCK_TIME_UTC_mvt"]) - to_s(head["MVT_TIME_UTC_mvt"])
    for nm, bm, tx in [("ranking", bm_r, tx_r), ("final_head", bm_f, tx_f)]:
        for p, m in [("ARR", arr), ("DEP", dep)]:
            ok = m & np.isfinite(bm) & np.isfinite(tx)
            res[f"taxi_eq_block_minus_mvt|{nm}|{p}"] = {"n": int(ok.sum()), "eq": int((bm[ok] == tx[ok]).sum()),
                                                      "eq_neg": int((-bm[ok] == tx[ok]).sum())}
    # sub-second structure of the original and new timestamps
    for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]:
        for nm, frame in [("ranking", r), ("final_head", head)]:
            v = to_s(frame[c])
            ok = np.isfinite(v)
            frac = v[ok] - np.floor(v[ok])
            res[f"subsecond|{c}|{nm}"] = {"n": int(ok.sum()), "nonzero_fraction": int((frac > 1e-9).sum())}
            sec = np.mod(v[ok], 60).astype(int)
            res[f"second_of_minute_hist|{c}|{nm}"] = np.bincount(sec, minlength=60).tolist()
    # dependence of the MVT jitter on group variables
    g = pd.DataFrame({"dm": dm, "db": db, "phase": ph, "ap": np.where(dep, r["ADEP_mvt"], r["ADES_mvt"]),
                      "month": pd.to_datetime(r["MVT_TIME_UTC_mvt"]).dt.month.to_numpy(),
                      "hour": pd.to_datetime(r["MVT_TIME_UTC_mvt"]).dt.hour.to_numpy(),
                      "sec0": np.mod(to_s(r["MVT_TIME_UTC_mvt"]), 60)})
    g = g[np.isfinite(g["dm"])]

    def grp(col):
        t = g.groupby(col)["dm"].agg(n="size", mean="mean", sd="std", changed=lambda x: float((x != 0).mean()), mn="min", mx="max")
        return {str(k): {kk: (float(vv) if kk != "n" else int(vv)) for kk, vv in row.items()} for k, row in t.iterrows()}
    res["dm_by_phase"] = grp("phase")
    res["dm_by_airport"] = grp("ap")
    res["dm_by_month"] = grp("month")
    res["dm_by_hour"] = grp("hour")
    res["dm_by_orig_second"] = grp(g["sec0"].round().astype(int).rename("sec0r"))
    # independence: same-flight DEP/ARR pairs (FLIGHT_ID shared), adjacent rows in file order, adjacent in time per airport
    fid = r["FLIGHT_ID_mvt"].to_numpy()
    df = pd.DataFrame({"fid": fid, "dm": dm, "ph": ph})
    pr = df[df["ph"] == "DEP"].dropna().drop_duplicates("fid").merge(df[df["ph"] == "ARR"].dropna().drop_duplicates("fid"), on="fid", suffixes=("_d", "_a"))
    res["corr_same_flight_dep_arr"] = {"n": int(len(pr)), "corr": float(np.corrcoef(pr["dm_d"], pr["dm_a"])[0, 1]) if len(pr) > 10 else None}
    ok = np.isfinite(dm)
    x = dm[ok]
    res["lag1_corr_file_order"] = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    o = g.assign(t=to_s(r["MVT_TIME_UTC_mvt"])[np.isfinite(dm)]).sort_values(["ap", "t"])
    same_ap = o["ap"].to_numpy()[1:] == o["ap"].to_numpy()[:-1]
    xx = o["dm"].to_numpy()
    res["lag1_corr_time_order_same_airport"] = float(np.corrcoef(xx[:-1][same_ap], xx[1:][same_ap])[0, 1])
    # correlation with the original value's second-of-minute and with the taxi value
    res["corr_dm_orig_second"] = float(np.corrcoef(g["dm"], g["sec0"])[0, 1])
    t_ok = arr & np.isfinite(dm) & np.isfinite(tx_r)
    res["corr_dm_taxi_arr"] = float(np.corrcoef(dm[t_ok], tx_r[t_ok])[0, 1])
    # order swaps: per airport and phase, adjacent pairs in original time order whose order flips
    sw = {}
    for p in ["DEP", "ARR"]:
        m = (ph == p) & np.isfinite(dm)
        t0 = to_s(r["MVT_TIME_UTC_mvt"])[m]
        t1 = to_s(head["MVT_TIME_UTC_mvt"])[m]
        ap = (r["ADEP_mvt"] if p == "DEP" else r["ADES_mvt"]).to_numpy()[m]
        oo = np.lexsort((t0, ap))
        a0, a1, aa = t0[oo], t1[oo], ap[oo]
        same = aa[1:] == aa[:-1]
        sw[p] = {"pairs": int(same.sum()), "ties_orig": int(((a0[1:] == a0[:-1]) & same).sum()),
                 "flipped": int(((a1[1:] < a1[:-1]) & (a0[1:] > a0[:-1]) & same).sum()),
                 "gap_le_24s": int((((a0[1:] - a0[:-1]) <= 24) & same).sum())}
    res["order_swaps"] = sw
    # fit check: rounded triangular(-12, 0, 12) and sum of two uniforms
    x = dm[np.isfinite(dm)]
    vals = np.arange(-12, 13)
    obs = np.array([(x == v).sum() for v in vals], float)
    n = obs.sum()

    def tri_cdf(z):
        z = np.clip(z, -12, 12)
        return np.where(z < 0, (z + 12) ** 2 / 288.0, 1 - (12 - z) ** 2 / 288.0)
    p_tri = tri_cdf(vals + 0.5) - tri_cdf(vals - 0.5)
    p_tri = p_tri / p_tri.sum()
    # discrete: sum of two uniform integers in [-6, 6]
    u = np.ones(13) / 13
    p_du = np.convolve(u, u)  # support -12..12
    chi_tri = float(((obs - n * p_tri) ** 2 / (n * p_tri)).sum())
    chi_du = float(((obs - n * p_du) ** 2 / (n * p_du)).sum())
    res["fit"] = {"values": vals.tolist(), "obs": obs.astype(int).tolist(), "exp_rounded_tri": (n * p_tri).round(1).tolist(),
                  "exp_sum_two_int_uniform": (n * p_du).round(1).tolist(), "chi2_rounded_tri": chi_tri, "chi2_sum_two_int_uniform": chi_du, "df": 24}
    # new months: second-of-minute pattern and DEP/ARR counts by month
    fm = pd.to_datetime(f["MVT_TIME_UTC_mvt"]).dt.month.to_numpy()
    res["final_rows_by_month_phase"] = {f"{int(m)}|{p}": int(((fm == m) & (fph == p)).sum()) for m in np.unique(fm[~np.isnan(fm)]) for p in ["DEP", "ARR"]}
    sub = pd.read_parquet("data/final_phase/final_submitting.parquet")
    s0 = pd.read_parquet("data/submitting.parquet")
    res["final_submitting"] = {"rows": len(sub), "head_same_ids": bool((sub["MVT_ID_mvt"].to_numpy()[:len(s0)] == s0["MVT_ID_mvt"].to_numpy()).all()),
                               "taxi_null": int(sub["TAXITIME_SEC_mvt"].isna().sum())}
    tail = f.iloc[n0:]
    tdep = tail[tail["PHASE_mvt"] == "DEP"]["MVT_ID_mvt"].to_numpy()
    res["final_submitting"]["tail_equals_final_dep_order"] = bool(len(tdep) == len(sub) - len(s0) and (sub["MVT_ID_mvt"].to_numpy()[len(s0):] == tdep).all())
    # training files: any sign of jitter? (second-of-minute uniformity, sub-second)
    tr = pd.concat([pd.read_parquet(p, columns=["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "PHASE_mvt", "TAXITIME_SEC_mvt"]) for p in sorted(Path("data").glob("training_*.parquet"))])
    for c in ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]:
        v = to_s(tr[c])
        res[f"second_of_minute_hist|{c}|training"] = np.bincount(np.mod(v[np.isfinite(v)], 60).astype(int), minlength=60).tolist()
    for nm, frame in [("training", tr), ("final_tail", tail)]:
        for p in ["DEP", "ARR"]:
            m = frame["PHASE_mvt"].to_numpy() == p
            bm = to_s(frame.loc[m, "BLOCK_TIME_UTC_mvt"]) - to_s(frame.loc[m, "MVT_TIME_UTC_mvt"])
            tx = frame.loc[m, "TAXITIME_SEC_mvt"].to_numpy(float)
            ok = np.isfinite(bm) & np.isfinite(tx)
            res[f"taxi_eq_block_minus_mvt|{nm}|{p}"] = {"n": int(ok.sum()), "eq": int((bm[ok] == tx[ok]).sum()), "eq_neg": int((-bm[ok] == tx[ok]).sum())}
    (OUT / "jit_char.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    keys = [k for k in res if not k.startswith("second_of_minute_hist") and not k.startswith("dm_by_")]
    for k in keys:
        print(k, json.dumps(res[k], ensure_ascii=False, default=str)[:900])


if __name__ == "__main__":
    sys.exit(main())
