# SPDX-License-Identifier: GPL-3.0-only
"""Write the predictions of a model run as a file in the order of the template, with the rule of the challenge files.

A run of src/model3.py (or of another leg) with --setting full writes cache/full_<tag>_s<seed>.parquet, which holds one row per
departure of data/ranking.parquet: MVT_ID_mvt and the prediction in seconds (column pred). This script puts the predictions in the
order of data/submitting.parquet, rounds them to the nearest integer, clips them at 0 and writes them as int32 with the template's
schema, the same rule as src/make_sub.py and the final file (docs/REPRODUCE.md section 6), without the contest's file naming and ledger.

    python examples/write_predictions.py cache/full_demo_s0.parquet predictions.parquet
    python examples/write_predictions.py cache/full_demo_s0.parquet predictions.parquet --col pred_noG
Several input files (for example the seeds of one leg) are averaged before rounding.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description="Write predictions in the order and format of data/submitting.parquet.")
    ap.add_argument("inputs", nargs="+", help="prediction files (MVT_ID_mvt and the column --col), then the output file")
    ap.add_argument("--col", default="pred", help="column to write (default pred)")
    ap.add_argument("--template", default=str(ROOT / "data" / "submitting.parquet"))
    a = ap.parse_args()
    if len(a.inputs) < 2:
        ap.error("give at least one prediction file and the output file")
    files, out = a.inputs[:-1], Path(a.inputs[-1])
    parts = [pd.read_parquet(f, columns=["MVT_ID_mvt", a.col]).set_index("MVT_ID_mvt")[a.col] for f in files]
    pred = sum(parts) / len(parts)
    schema = pq.read_schema(a.template)
    ids = pd.read_parquet(a.template, columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()
    vals = pred.reindex(ids)
    if vals.isna().any():
        raise SystemExit("%d departures of the template have no prediction" % int(vals.isna().sum()))
    table = pa.Table.from_pandas(pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": np.clip(np.rint(vals.to_numpy()), 0, None).astype("int32")}),
                                 schema=schema, preserve_index=False)
    pq.write_table(table, out)
    v = table.column("TAXITIME_SEC_mvt").to_numpy()
    print("wrote %s: %d departures, mean %.1f s, median %.0f s, max %d s" % (out, table.num_rows, v.mean(), np.median(v), v.max()))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
