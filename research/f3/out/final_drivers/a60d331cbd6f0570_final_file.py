"""F3 final rebuild: the submission file and its checks (no ledger, no upload; firing is not this lane's).
File: <out> in data/submitting.parquet's row order and arrow schema, TAXITIME_SEC_mvt = clip(rint(pred), 0) as int32 (src/make_sub.build's
rule), pred from preds/test_<spec>.parquet of the root (cwd).
Checks (research/f3/out/final_file_<spec>.json):
  1. 670,790 rows, MVT_ID order = the template's, no missing value, no negative, int32, schema equal to the template's;
  2. Jan/Jul rows against v35 (subs/spirited-tiger_v35.parquet of prc-2026, the int file 514 fired as) and against 514's float predictions:
     move MSE, RMS, max |move|, by airport, and the move against |ΔMVT| (ΔMVT = final minus old MVT of the same id);
  3. Feb/Jun sanity: mean, median and p90 of the prediction by airport and month against the 2025 label of the same calendar month
     (training files, DEP rows), and the same for Jan/Jul.
    python -X utf8 research/f3/final_file.py <spec> <out parquet> <v35 parquet> <514 test parquet> <old ranking parquet>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def secs(s: pd.Series) -> np.ndarray:
    return (s.dt.tz_convert("UTC").dt.tz_localize(None) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()


def main():
    spec, out_p, v35_p, t514_p, old_rank = sys.argv[1:6]
    T = pd.read_parquet(f"preds/test_{spec}.parquet")
    tpl_p = Path("data/submitting.parquet")
    schema = pq.read_schema(tpl_p)
    tpl = pd.read_parquet(tpl_p)
    vals = T.set_index("MVT_ID_mvt")["pred"].reindex(tpl["MVT_ID_mvt"].to_numpy())
    assert not vals.isna().any(), f"missing predictions: {int(vals.isna().sum())}"
    out = pd.DataFrame({"MVT_ID_mvt": tpl["MVT_ID_mvt"].to_numpy(), "TAXITIME_SEC_mvt": np.clip(np.rint(vals.to_numpy()), 0, None).astype("int32")})
    table = pa.Table.from_pandas(out, schema=schema, preserve_index=False)
    assert table.schema.equals(schema, check_metadata=False)
    op = Path(out_p)
    assert not op.exists(), f"{op} exists"
    op.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, op)
    back = pd.read_parquet(op)
    rep = {"spec": spec, "file": str(op), "sha16": sha16(op), "rows": int(len(back)),
           "refs": {"int_file (the v35 slot)": v35_p, "float_test (the 514 slot)": t514_p},
           "check1": {"rows_670790": len(back) == 670790, "order_equals_template": bool(back["MVT_ID_mvt"].equals(tpl["MVT_ID_mvt"])),
                      "no_missing": bool(back["TAXITIME_SEC_mvt"].notna().all()), "no_negative": bool((back["TAXITIME_SEC_mvt"] >= 0).all()),
                      "int32": str(back["TAXITIME_SEC_mvt"].dtype) == "int32", "schema_equal": bool(pq.read_schema(op).equals(schema, check_metadata=False))}}
    rep["check1"]["pass"] = all(rep["check1"].values())
    # 2) Jan/Jul against v35 and 514
    fr = pd.read_parquet("data/ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    fr = fr[fr["PHASE_mvt"] == "DEP"].set_index("MVT_ID_mvt")
    orr = pd.read_parquet(old_rank, columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"])
    orr = orr[orr["PHASE_mvt"] == "DEP"].set_index("MVT_ID_mvt")
    v35 = pd.read_parquet(v35_p).set_index("MVT_ID_mvt")["TAXITIME_SEC_mvt"].astype(float)
    t514 = pd.read_parquet(t514_p).set_index("MVT_ID_mvt")["pred"].astype(float)
    ids = v35.index.to_numpy()
    new_i = back.set_index("MVT_ID_mvt")["TAXITIME_SEC_mvt"].astype(float).reindex(ids).to_numpy()
    new_f = T.set_index("MVT_ID_mvt")["pred"].reindex(ids).to_numpy()
    dmvt = secs(fr["MVT_TIME_UTC_mvt"].reindex(ids)) - secs(orr["MVT_TIME_UTC_mvt"].reindex(ids))
    ap = fr["ADEP_mvt"].reindex(ids).astype(str).to_numpy()
    mv_i, mv_f = new_i - v35.to_numpy(), new_f - t514.reindex(ids).to_numpy()
    rep["check2"] = {"rows": int(len(ids)), "move_mse_vs_v35_int": round(float(np.mean(mv_i ** 2)), 2), "move_rms_vs_v35": round(float(np.sqrt(np.mean(mv_i ** 2))), 3),
                     "move_max_abs_vs_v35": float(np.max(np.abs(mv_i))), "move_mse_vs_514_float": round(float(np.mean(mv_f ** 2)), 2),
                     "move_max_abs_vs_514_float": round(float(np.max(np.abs(mv_f))), 2),
                     "move_mse_by_airport": {a: round(float(np.mean(mv_f[ap == a] ** 2)), 1) for a in sorted(set(ap))},
                     "move_mse_by_abs_dmvt": {f"{lo}-{hi}": round(float(np.mean(mv_f[(np.abs(dmvt) >= lo) & (np.abs(dmvt) <= hi)] ** 2)), 1)
                                              for lo, hi in ((0, 0), (1, 4), (5, 8), (9, 12))},
                     "share_rows_moved_gt_60s": round(float((np.abs(mv_f) > 60).mean()), 5),
                     "mean_move_vs_514": round(float(np.mean(mv_f)), 3)}
    # 3) means by airport and month against 2025 labels of the same calendar month
    tr = pd.concat([pd.read_parquet(p, columns=["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
                    for p in sorted(Path("data").glob("training_2025-0[1267]-*.parquet"))])
    tr = tr[tr["PHASE_mvt"] == "DEP"]
    tr["m"] = tr["MVT_TIME_UTC_mvt"].dt.month
    lab = tr.groupby(["ADEP_mvt", "m"])["TAXITIME_SEC_mvt"].agg(["mean", "median", lambda s: s.quantile(0.9)]).rename(columns={"<lambda_0>": "p90"})
    pr = pd.DataFrame({"ap": fr["ADEP_mvt"].reindex(back["MVT_ID_mvt"]).astype(str).to_numpy(),
                       "m": fr["MVT_TIME_UTC_mvt"].reindex(back["MVT_ID_mvt"]).dt.month.to_numpy(), "p": back["TAXITIME_SEC_mvt"].to_numpy(float)})
    pg = pr.groupby(["ap", "m"])["p"].agg(["size", "mean", "median", lambda s: s.quantile(0.9)]).rename(columns={"<lambda_0>": "p90"})
    tab = {}
    for (a, m), r in pg.iterrows():
        l = lab.loc[(a, m)] if (a, m) in lab.index else None
        tab[f"{a} {m:02d}"] = {"n": int(r["size"]), "pred_mean": round(r["mean"], 1), "pred_median": round(r["median"], 1), "pred_p90": round(r["p90"], 1),
                               "label2025_mean": round(float(l["mean"]), 1) if l is not None else None,
                               "label2025_median": round(float(l["median"]), 1) if l is not None else None}
    rep["check3"] = tab
    rep["check3_month_means"] = {int(m): {"pred_mean": round(float(pr.loc[pr.m == m, "p"].mean()), 1),
                                          "label2025_mean": round(float(tr.loc[tr.m == m, "TAXITIME_SEC_mvt"].mean()), 1),
                                          "rows": int((pr.m == m).sum())} for m in (1, 2, 6, 7)}
    Path(f"research/f3/out/final_file_{spec}.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "check3"}, indent=1, default=str))


if __name__ == "__main__":
    main()
