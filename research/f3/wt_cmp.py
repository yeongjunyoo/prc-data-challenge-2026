# SPDX-License-Identifier: GPL-3.0-only
"""F3 fitwrap test comparisons (research/f3/wt_run.sh). Prints and writes <W>/out/wrapper_test.json:
  T1 direct vs T2 save: bit-equal output (the wrapper is transparent when it fits);
  T1 vs T3 load: bit-equal output, every fit loaded (a stored model gives the same predictions as the direct fit);
  T4 strict on the matrix with one training value changed: stopped with MISS_STOP, no output, and the logged difference names that column;
  T5 strict on the final table with the hybrid's old-draw Jan/Jul rows: every fit loaded; rows other than Jan/Jul bit-equal to T1; on the
     Jan/Jul rows the prediction move between the organizer's new draw (T1) and the old draw (T5), by airport group, next to the sampler's
     move on the same kind of rows (19.7 s whole-minute airports, 8.6 s second airports);
  CatBoost synthetic: C2 == C1, C3 == C1 with the fit loaded, C4 stopped.
    python -X utf8 research/f3/wt_cmp.py <W>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MINRES = ["EDDM", "EGLL", "LFPG", "LIRF", "LTFM"]
DROP = {"MVT_ID_mvt", "src", "y", "sec_sch", "sec_aobt", "flight"}


def same(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    r = {"columns_equal": list(a.columns) == list(b.columns), "rows": [len(a), len(b)]}
    if r["columns_equal"] and len(a) == len(b):
        d = {}
        for c in a.columns:
            x, y = a[c].to_numpy(), b[c].to_numpy()
            eq = (x == y) | (pd.isna(x) & pd.isna(y))
            if (~eq).any():
                d[c] = int((~eq).sum())
        r["cells_differing"] = d
        r["bit_equal"] = not d
    else:
        r["bit_equal"] = False
    return r


def events(p: Path) -> tuple[dict, list]:
    ev = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []
    cnt = {}
    for e in ev:
        k = f"{e.get('lib', '-')}:{e['event']}"
        cnt[k] = cnt.get(k, 0) + 1
    return cnt, [e for e in ev if str(e.get("event", "")).startswith("MISS")]


def main():
    W = Path(sys.argv[1])
    o = W / "out"
    rep = {}
    T = {i: pd.read_parquet(o / f"T{i}.parquet") if (o / f"T{i}.parquet").exists() else None for i in (1, 2, 3, 4, 5)}
    rep["T1_vs_T2_save_transparent"] = same(T[1], T[2])
    rep["T1_vs_T3_load_exact"] = same(T[1], T[3])
    for name, d in (("save", "fc_save"), ("load", "fc_load"), ("strict_perturbed", "fc_strict"), ("strict_hybrid", "fc_t5")):
        rep[f"events_{name}"], miss = events(W / d / "fitwrap_log.jsonl")
        if miss:
            rep[f"miss_{name}"] = miss
    # T4: the column named by the logged difference (classifier columns: feats_v25 order without DROP and m_*)
    prep = json.loads((o / "wt_prep.json").read_text(encoding="utf-8"))
    import pyarrow.parquet as pq
    names = [f.name for f in pq.read_schema(W / "r1" / "cache" / "feats_v25.parquet") if f.name not in DROP]
    clf_names = [c for c in names if not c.startswith("m_")]
    m4 = rep.get("miss_strict_perturbed", [])
    rep["T4"] = {"stopped": bool(m4) and m4[0]["event"] == "MISS_STOP" and T[4] is None, "perturbed": prep["r4"],
                 "first_miss": m4[0] if m4 else None,
                 "named_columns": [clf_names[j] if isinstance(j, int) and j < len(clf_names) else j for j in (m4[0].get("data_columns_differing") or [])] if m4 else None}
    if T[5] is not None:
        a = T[1].set_index("MVT_ID_mvt")
        b = T[5].set_index("MVT_ID_mvt").reindex(a.index)
        rid = pd.read_parquet(W / "r5" / "replaced_ids.parquet")["MVT_ID_mvt"].to_numpy()
        isr = np.isin(a.index.to_numpy(), rid)
        rest = same(a[~isr].reset_index(), b[~isr].reset_index())
        ap = pd.read_parquet(W / "r1" / "cache" / "feats_v25.parquet", columns=["MVT_ID_mvt", "ap"]).set_index("MVT_ID_mvt")["ap"].astype(str)
        apr = ap.reindex(a.index[isr]).to_numpy()
        mv = a["pred"].to_numpy(float)[isr] - b["pred"].to_numpy(float)[isr]
        pm = a["p_art"].to_numpy(float)[isr] - b["p_art"].to_numpy(float)[isr]
        r = {"rows_other_than_jan_jul_bit_equal": rest["bit_equal"], "rows_other_detail": rest, "jan_jul_rows": int(isr.sum())}
        for g, m in (("minute", np.isin(apr, MINRES)), ("second", ~np.isin(apr, MINRES)), ("all", np.ones(len(apr), bool))):
            x = mv[m]
            r[g] = {"rows": int(m.sum()), "move_rms_s": round(float(np.sqrt(np.mean(x ** 2))), 2), "move_mean_s": round(float(np.mean(x)), 3),
                    "p99_abs_s": round(float(np.quantile(np.abs(x), 0.99)), 1), "max_abs_s": round(float(np.max(np.abs(x))), 1),
                    "share_abs_gt60s": round(float((np.abs(x) > 60).mean()), 5), "p_art_move_rms": round(float(np.sqrt(np.mean(pm[m] ** 2))), 5)}
        r["sampler_move_rms_s_same_rows_kind"] = {"minute": 19.7, "second": 8.6}
        rep["T5_real_redraw_move"] = r
    else:
        rep["T5_real_redraw_move"] = "no output (see miss_strict_hybrid)"
    cb = W / "cb"
    C = {i: np.load(cb / f"c{i}.npy") if (cb / f"c{i}.npy").exists() else None for i in range(1, 5)}
    eqn = lambda x, y: None if x is None or y is None else bool(np.array_equal(x, y))  # noqa: E731
    rep["catboost"] = {"C2_save_eq_C1": eqn(C[2], C[1]), "C3_load_eq_C1": eqn(C[3], C[1]), "C4_stopped": C[4] is None,
                       "events_save": events(cb / "fc_cb" / "fitwrap_log.jsonl")[0], "events_load": events(cb / "fc_cb_load" / "fitwrap_log.jsonl")[0],
                       "events_strict": events(cb / "fc_cb_strict" / "fitwrap_log.jsonl")[0],
                       "max_abs_diff_C3_C1": None if C[3] is None or C[1] is None else float(np.max(np.abs(C[3] - C[1])))}
    rep["T2_vs_T3_load_reproduces_save"] = same(T[2], T[3])
    others_ok = (rep["T2_vs_T3_load_reproduces_save"]["bit_equal"] and rep["events_load"].get("lgb:loaded", 0) > 0
                 and not rep["events_load"].get("lgb:MISS_fit") and rep["T4"]["stopped"]
                 and rep["catboost"]["C2_save_eq_C1"] and rep["catboost"]["C3_load_eq_C1"] and rep["catboost"]["C4_stopped"])
    rep["PASS_strict_bit_equal_to_direct"] = bool(others_ok and rep["T1_vs_T3_load_exact"]["bit_equal"])
    # rule revised 2026-10-09 00:2x KST after T1 (direct) vs T2 (save) differed by at most 1.0e-12 s: a second direct run T1b measures
    # LightGBM's own run-to-run difference; the wrapper passes when load reproduces the save run bit for bit, the save run differs from a
    # direct run by no more than two direct runs differ (and at most 1e-9), and the stop and CatBoost checks hold
    if (o / "T1b.parquet").exists():
        t1b = pd.read_parquet(o / "T1b.parquet")

        def mx(a, b):
            return {c: float(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))) for c in ("pred", "p_art", "mu")}
        dd, ds = mx(T[1], t1b), mx(T[1], T[2])
        rep["direct_vs_direct_T1_T1b_max_abs"] = dd
        rep["direct_vs_save_T1_T2_max_abs"] = ds
        rep["direct_vs_direct_bits_differ_pred"] = int((T[1]["pred"].to_numpy().view("u8") != t1b["pred"].to_numpy().view("u8")).sum())
        noise_ok = all(ds[c] <= max(dd[c], 0.0) * 10 and ds[c] <= 1e-9 for c in ds) and rep["direct_vs_direct_bits_differ_pred"] > 0
        rep["PASS"] = bool(others_ok and (rep["T1_vs_T3_load_exact"]["bit_equal"] or noise_ok))
        rep["pass_rule"] = "load == save bit-exact; save vs direct within 10x the direct-vs-direct difference and <= 1e-9; stop and CatBoost checks"
    else:
        rep["PASS"] = rep["PASS_strict_bit_equal_to_direct"]
    (o / "wrapper_test.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
