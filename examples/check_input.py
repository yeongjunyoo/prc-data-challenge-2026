# SPDX-License-Identifier: GPL-3.0-only
"""Check the files in data/ against the input format of docs/USAGE.md section 2 before running the pipeline on them.

The pipeline itself checks little: src/load.py reads every timestamp as microseconds, so it reads a file written in seconds or
milliseconds as dates in 1970 without an error, and an empty take-off or schedule time stops it only in src/feats.py. This script
reads the files one at a time and changes nothing. It prints a PROBLEM line for each error of format or value that it finds:
  - file names: ranking.parquet, submitting.parquet and training files whose month src/load.py reads from characters 15 and 16
    of the name (training_YYYY-MM...), at most one file per month, and most of a file's movements in that month (UTC)
  - the 30 columns and their types; timestamps must have the unit us (src/load.py reads every timestamp as microseconds)
  - values that must be present: MVT_ID_mvt, PHASE_mvt (DEP or ARR) and MVT_TIME_UTC_mvt on every row, SCHED_TIME_UTC_mvt and ADEP_mvt
    on every departure, ADES_mvt on every arrival, block time and label on every training departure
  - values of every timestamp column between the years 2000 and 2100; labels equal to take-off minus off-block (in-block minus
    landing for arrivals); movement ids unique across the files; no block time or label on the departures to predict; a template
    with the two columns of the challenge template that lists exactly the departures to predict
and a NOTE line for each of these points of docs/USAGE.md sections 2, 4.2 and 4.3: timestamps without a time zone (read as UTC),
number columns of another type, template columns of another type, a file of movements without arrivals or with more than 1% of its
arrivals without block time or taxi-in time, airports without a time zone in src/feats.py, and each setting of src/model3.py that
cannot run on these files as the code stands (no departures to score, or no G rows among the rows it fits on or scores). It checks
the format and these rules, not whether the values are right: times of the right type and years can still be wrong.

    python examples/check_input.py                    # exit code 1 if there is a PROBLEM line, else 0
    python examples/check_input.py --data <folder>    # files in another folder (the pipeline itself reads data/ only)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from examples.make_sample import SCHEMA, TEMPLATE_SCHEMA  # noqa: E402
from src.feats import TZ  # noqa: E402

TS = [f.name for f in SCHEMA if pa.types.is_timestamp(f.type)]
UNIT = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9}
YEARS = (2000, 2100)
# months (column src; 0 = the movements to predict) that each setting of src/model3.py fits on and scores
SETTINGS = {"val": ([2, 3, 4, 5, 6, 8, 9, 10, 11, 12], [1, 7]), "fwd": ([1, 2, 3, 4, 5, 6], [7]),
            "nd": (list(range(1, 11)), [11, 12]), "full": (list(range(1, 13)), [0])}
CONVERT = ("in pandas, for each timestamp column: df[c] = pd.to_datetime(df[c], utc=True).astype('datetime64[us, UTC]'), "
           "or for epoch seconds stored as numbers pd.to_datetime(df[c], unit='s', utc=True).astype('datetime64[us, UTC]')")


def is_text(ty: pa.DataType) -> bool:
    """True for the arrow types that src/load.py reads as text: string, large string, string view and their dictionaries."""
    if pa.types.is_dictionary(ty):
        ty = ty.value_type
    return pa.types.is_string(ty) or pa.types.is_large_string(ty) or pa.types.is_string_view(ty)


def is_number(ty: pa.DataType) -> bool:
    """True for integer and floating-point arrow types."""
    return pa.types.is_integer(ty) or pa.types.is_floating(ty)


def text(col: pa.ChunkedArray) -> pa.ChunkedArray:
    """A text column as plain strings (dictionaries decoded); all empty if the column holds no text."""
    if not is_text(col.type):
        return pa.chunked_array([pa.nulls(len(col), pa.string())])
    return col.cast(pa.string())


def mask(col: pa.ChunkedArray, value: str) -> np.ndarray:
    """Boolean array: the text column equals value (False where empty)."""
    return pc.fill_null(pc.equal(text(col), value), False).to_numpy(zero_copy_only=False)


def seconds(col: pa.ChunkedArray) -> np.ndarray:
    """UTC epoch seconds of a timestamp column of any unit (NaN where empty); all NaN for a column that is not a timestamp."""
    if not pa.types.is_timestamp(col.type):
        return np.full(len(col), np.nan)
    return col.cast(pa.int64()).to_numpy(zero_copy_only=False).astype("float64") / UNIT[col.type.unit]


def check_file(path: Path, month: int, problems: list) -> dict | None:
    """Check one file of movements (month: from the file name, 0 for ranking.parquet); appends to problems.

    Returns what the checks across files need (movement ids, the ids of its departures, its number of departures and of G rows,
    the airports of its departures) and its notes, or None if columns are missing."""
    t = pq.read_table(path)
    name = path.name
    have = {f.name: f.type for f in t.schema}
    missing = [f.name for f in SCHEMA if f.name not in have]
    if missing:
        problems.append("%s: columns missing: %s" % (name, ", ".join(missing)))
        return None
    bad_ts, naive, numbers, notes = [], [], [], []
    for f in SCHEMA:
        ty = have[f.name]
        if pa.types.is_null(ty) or t.column(f.name).null_count == t.num_rows:  # no value at all: src/load.py reads it as empty
            continue
        if f.name in TS:
            if not pa.types.is_timestamp(ty) or ty.unit != "us":
                bad_ts.append("%s (%s)" % (f.name, ty))
            elif ty.tz is None:
                naive.append(f.name)
        elif pa.types.is_string(f.type):
            if not is_text(ty):
                problems.append("%s: %s is %s, expected text" % (name, f.name, ty))
        elif ty != f.type:
            if is_number(ty):
                numbers.append("%s %s (the challenge files: %s)" % (f.name, ty, f.type))
            else:
                problems.append("%s: %s is %s, expected %s" % (name, f.name, ty, f.type))
    if bad_ts:
        kinds = sorted({b.split(" (", 1)[1][:-1] for b in bad_ts})
        problems.append("%s: timestamp columns not stored as timestamp[us]: %s" % (name, "%d columns of type %s" % (len(bad_ts), kinds[0]) if len(kinds) == 1
                                                                       else ", ".join(bad_ts)))
    dep, arr = mask(t.column("PHASE_mvt"), "DEP"), mask(t.column("PHASE_mvt"), "ARR")
    if (~(dep | arr)).any():
        problems.append("%s: PHASE_mvt empty or other than DEP and ARR on %d rows" % (name, int((~(dep | arr)).sum())))

    def empty(col):
        return t.column(col).is_null().to_numpy(zero_copy_only=False)
    every = np.ones(t.num_rows, bool)
    for col, rows, what in (("MVT_ID_mvt", every, "rows"), ("MVT_TIME_UTC_mvt", every, "rows"), ("SCHED_TIME_UTC_mvt", dep, "departures"),
                            ("ADEP_mvt", dep, "departures"), ("ADES_mvt", arr, "arrivals")):
        n = int((empty(col) & rows).sum())
        if n:
            problems.append("%s: %s empty on %d %s" % (name, col, n, what))
    for col in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"):
        n = int((empty(col) & dep).sum())
        if month and n:
            problems.append("%s: %s empty on %d training departures" % (name, col, n))
        if not month and n != int(dep.sum()):
            problems.append("%s: %s filled on %d departures to predict (they must be empty)" % (name, col, int(dep.sum()) - n))
    n = int((arr & (empty("BLOCK_TIME_UTC_mvt") | empty("TAXITIME_SEC_mvt"))).sum())
    if dep.any() and not arr.any():
        notes.append("%s: no arrivals; the features built from them (stand occupancy, taxi-in times) stay empty for its departures, "
                     "while in the challenge files every month has its arrivals" % name)
    elif n > 0.01 * arr.sum():
        notes.append("%s: %d of %d arrivals without block time or taxi-in time; the features built from them (stand occupancy, taxi-in "
                     "times) then differ from those of files whose arrivals have both, as almost all arrivals of the challenge files do"
                     % (name, n, int(arr.sum())))
    out = []
    for c in TS:
        v = seconds(t.column(c))
        if np.isfinite(v).any():
            yr = np.array([np.nanmin(v), np.nanmax(v)]).astype("int64").astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
            if yr[0] < YEARS[0] or yr[1] > YEARS[1]:
                out.append("%s (%d to %d)" % (c, yr[0], yr[1]))
    if out:
        problems.append("%s: years outside %d to %d, probably converted with a wrong unit: %s" % (name, YEARS[0], YEARS[1], ", ".join(out)))
    mvt = seconds(t.column("MVT_TIME_UTC_mvt"))
    if month and np.isfinite(mvt).any() and not any(o.startswith("MVT_TIME") for o in out):
        m = mvt[np.isfinite(mvt)].astype("int64").astype("datetime64[s]").astype("datetime64[M]").astype(int) % 12 + 1
        if float((m == month).mean()) < 0.5:
            top = int(np.bincount(m).argmax())
            problems.append("%s: the name gives month %d, but %.0f%% of its movements are in month %d (UTC)"
                            % (name, month, 100 * float((m == top).mean()), top))
    if month and is_number(have["TAXITIME_SEC_mvt"]):
        blk = seconds(t.column("BLOCK_TIME_UTC_mvt"))
        y = t.column("TAXITIME_SEC_mvt").to_numpy(zero_copy_only=False).astype("float64")
        bad = (dep | arr) & (np.abs(np.where(dep, mvt - blk, blk - mvt) - y) > 1)  # False where a value is missing
        if bad.any():
            problems.append("%s: TAXITIME_SEC_mvt differs from the take-off minus off-block time (in-block minus landing for arrivals) "
                            "on %d rows" % (name, int(bad.sum())))
    adep = text(t.column("ADEP_mvt"))
    g = dep & mask(adep, "LIRF") & empty("FLIGHT_ID_mvt")
    ids = t.column("MVT_ID_mvt").to_numpy(zero_copy_only=False).astype("float64") if is_number(have["MVT_ID_mvt"]) else np.full(t.num_rows, np.nan)
    aps = {d["values"] for d in pc.value_counts(adep.filter(pa.array(dep))).to_pylist() if d["values"] is not None}
    return {"ids": ids, "dep_ids": ids[dep], "deps": int(dep.sum()), "g": int(g.sum()), "aps": aps, "naive": naive, "numbers": numbers,
            "notes": notes}


def main() -> int:
    ap = argparse.ArgumentParser(description="Check the input files against docs/USAGE.md section 2.")
    ap.add_argument("--data", default=str(ROOT / "data"), help="folder of the files (default data/, the folder that the pipeline reads)")
    data = Path(ap.parse_args().data)
    problems, notes = [], []
    train = sorted(data.glob("training_*.parquet"))
    if not train:
        problems.append("no training_*.parquet in %s" % data)
    files = []
    for p in train:
        mm = p.name[14:16]
        if mm.isdigit() and 1 <= int(mm) <= 12:
            files.append((p, int(mm)))
        else:
            problems.append("%s: characters 15 and 16 of the name, %r, are not a month 01 to 12 (src/load.py reads the month there)" % (p.name, mm))
    months = [m for _, m in files]
    for m in sorted(set(months)):
        if months.count(m) > 1:
            problems.append("%d training files give month %d: %s" % (months.count(m), m, ", ".join(p.name for p, mo in files if mo == m)))
    rp = data / "ranking.parquet"
    if rp.exists():
        files.append((rp, 0))
    else:
        problems.append("missing %s" % rp)
    seen = {}
    for p, m in files:
        r = check_file(p, m, problems)
        if r is not None:
            seen[p.name] = (m, r)
            notes += r["notes"]
    ids = np.concatenate([r["ids"] for _, r in seen.values()]) if seen else np.array([])
    ids = ids[~np.isnan(ids)]
    if len(np.unique(ids)) < len(ids):
        problems.append("%d movement ids appear more than once in the files" % (len(ids) - len(np.unique(ids))))
    tp = data / "submitting.parquet"
    if not tp.exists():
        problems.append("missing %s" % tp)
    else:
        tpl = pq.read_table(tp)
        if tpl.column_names != TEMPLATE_SCHEMA.names:
            problems.append("submitting.parquet: the columns must be MVT_ID_mvt and TAXITIME_SEC_mvt, in this order")
        elif "ranking.parquet" in seen:
            if tpl.schema.types != TEMPLATE_SCHEMA.types:
                notes.append("submitting.parquet: the challenge template has the types double and int32, this one %s; "
                             "examples/write_predictions.py writes the template's types" % ", ".join(str(x) for x in tpl.schema.types))
            want = seen["ranking.parquet"][1]["dep_ids"]
            got = tpl.column("MVT_ID_mvt").to_numpy(zero_copy_only=False).astype("float64")
            if len(got) != len(want) or not np.array_equal(np.sort(got), np.sort(want)):
                problems.append("submitting.parquet: its %d ids are not the %d departures of ranking.parquet" % (len(got), len(want)))

    def in_files(names):
        return "1 file, %s" % names[0] if len(names) == 1 else "%d files" % len(names)
    naive = [(name, r["naive"]) for name, (_, r) in sorted(seen.items()) if r["naive"]]
    if naive:
        notes.append("timestamps without a time zone, read as UTC (docs/USAGE.md section 2): %d columns in %s"
                     % (len({c for _, cs in naive for c in cs}), in_files([n for n, _ in naive])))
    for key, text_ in (("numbers", "number column of another type, read as a float by src/load.py: %s in %s"),
                       ("aps", "airport without a time zone in src/feats.py TZ, whose local-time columns stay empty (docs/USAGE.md section 4.2): "
                               "%s in %s")):
        found = {}
        for name, (_, r) in sorted(seen.items()):
            for x in r[key]:
                if key != "aps" or x not in TZ:
                    found.setdefault(x, []).append(name)
        notes += [text_ % (x, in_files(f)) for x, f in sorted(found.items())]
    if seen:
        deps = {m: r["deps"] for m, r in seen.values()}
        g = {m: r["g"] for m, r in seen.values()}
        why = {}
        for s, (fit, scored) in SETTINGS.items():
            if not sum(deps.get(m, 0) for m in scored):
                if s != "full":
                    notes.append("--setting %s has no departures to score (%s %s) and stops (docs/USAGE.md section 4.3)"
                                 % (s, "months" if len(scored) > 1 else "month", " and ".join(map(str, scored))))
                continue
            gf, gs = sum(g.get(m, 0) for m in fit), sum(g.get(m, 0) for m in scored)
            if not (gf and gs):
                why.setdefault("fits on and scores" if not (gf or gs) else "fits on" if not gf else "scores", []).append(s)
        for w, ss in why.items():
            notes.append("--setting %s: no G rows (LIRF departures without an NM record) among the rows %s %s; src/model3.py stops there "
                         "unless you add the line of docs/USAGE.md section 4.2"
                         % (", ".join(ss[:-1]) + " and " + ss[-1] if len(ss) > 1 else ss[0], "they" if len(ss) > 1 else "it",
                            w.replace("fits", "fit").replace("scores", "score") if len(ss) > 1 else w))
    for p in problems:
        print("PROBLEM", p)
    if any("timestamp[us]" in p or "wrong unit" in p for p in problems):
        print("        ", CONVERT)
    for n in notes:
        print("NOTE   ", n)
    print("checked %d files: %d problems, %d notes" % (len(files) + tp.exists(), len(problems), len(notes)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
