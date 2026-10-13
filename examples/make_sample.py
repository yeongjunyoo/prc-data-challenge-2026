# SPDX-License-Identifier: GPL-3.0-only
"""Write a small synthetic data set in the format of the challenge files, so that the pipeline can be tried without the organizer data.

The organizer data has no published license and is not part of this repository. This script invents departures and arrivals with the
30 columns and types of the challenge files (docs/USAGE.md section 2 lists them) at two of the ten airports, EHAM and LIRF, for every
day of 2025 (twelve training files with labels) and for January 2026 (an input file whose departures have no label, and a template to
fill). LIRF is included because src/model3.py fits a separate model for LIRF departures without a Network Manager flight record.
The values are random draws from a few simple rules (schedule, delay, runway, stand, congestion, how the off-block time was recorded,
and the NM matcher's window: a departure keeps its NM record only if |BLOCK - LOBT| <= 3,606 s); they describe no real airport, and a
model fitted on them says nothing about real taxi times. The draws are seeded, so every run writes the same files.

    python examples/make_sample.py             # writes data/training_2025-MM-01_....parquet (12 files), data/ranking.parquet and
                                               # data/submitting.parquet: 31,531 movements, 2.7 MB, a few seconds
    python examples/make_sample.py --force     # replace files of the same names

It stops without writing if data/ already holds files with these names, so that the sample never mixes with real data.
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
DATA = ROOT / "data"

TS = pa.timestamp("us", tz="UTC")
STR = pa.string()
# names, order and types of the columns of the challenge files (training_*.parquet, ranking.parquet, final_ranking.parquet)
SCHEMA = pa.schema([
    ("MVT_ID_mvt", pa.float64()), ("FLIGHT_ID_mvt", pa.float64()), ("FLIGHT_mvt", STR), ("FLIGHT_RULE_mvt", STR), ("ADEP_mvt", STR),
    ("ADES_mvt", STR), ("PHASE_mvt", STR), ("MVT_TIME_UTC_mvt", TS), ("BLOCK_TIME_UTC_mvt", TS), ("SCHED_TIME_UTC_mvt", TS),
    ("AIRCRAFT_TYPE_mvt", STR), ("RUNWAY_mvt", STR), ("STAND_mvt", STR), ("TAXITIME_SEC_mvt", pa.int32()), ("LOBT_flt", TS),
    ("CALLSIGN_flt", STR), ("ADEP_flt", STR), ("ADES_flt", STR), ("ADES_FILED_flt", STR), ("MARKET_SEGMENT_flt", STR), ("IOBT_flt", TS),
    ("FLIGHT_RULE_flt", STR), ("FLIGHT_TYPE_flt", STR), ("AIRCRAFT_TYPE_flt", STR), ("WK_TBL_CAT_flt", STR), ("AIRCRAFT_OPERATOR_flt", STR),
    ("EOBT_1_flt", TS), ("ARVT_1_flt", TS), ("AOBT_3_flt", TS), ("ARVT_3_flt", TS),
])
TEMPLATE_SCHEMA = pa.schema([("MVT_ID_mvt", pa.float64()), ("TAXITIME_SEC_mvt", pa.int32())])

# invented airport rules: time zone, departures and arrivals per day, departure runways with their mean taxi-out time (s) and their
# share in each of two runway configurations, arrival runways, share of departures without a Network Manager (NM) flight record
AIRPORTS = {
    "EHAM": {"tz": "Europe/Amsterdam", "dep": 22, "arr": 22, "rwy": {"18L": (1150, 0.6, 0.1), "24": (820, 0.3, 0.3), "36L": (700, 0.1, 0.6)},
             "arr_rwy": ["18R", "06", "36R"], "no_nm": 0.01},
    "LIRF": {"tz": "Europe/Rome", "dep": 18, "arr": 18, "rwy": {"25": (780, 0.7, 0.2), "16R": (650, 0.3, 0.8)}, "arr_rwy": ["16L", "34R"],
             "no_nm": 0.06},
}
OPERATORS = ["SYA", "SYB", "SYC", "SYD", "SYE", "SYF"]
SEGMENT = {"SYA": "Mainline", "SYB": "Mainline", "SYC": "Lowcost", "SYD": "Lowcost", "SYE": "Regional", "SYF": "Cargo"}
COPY_SHARE = {"SYA": 0.04, "SYB": 0.04, "SYC": 0.06, "SYD": 0.06, "SYE": 0.30, "SYF": 0.25}  # off-block time recorded as the schedule
TYPES = {"A320": "M", "A321": "M", "B738": "M", "E190": "M", "A333": "H", "B77W": "H"}
OTHER_AIRPORTS = ["EGKK", "LFMN", "LEPA", "EDDT", "LOWW", "EKCH", "ESSA", "LPPT", "LGAV", "EIDW", "EBBR", "LKPR"]
STAND_EFFECT = {"A": -60.0, "B": 0.0, "C": 150.0}  # pier A near the runways, C remote


def local_day(t: np.ndarray, zone: str) -> np.ndarray:
    """Local calendar date (as days since 1970-01-01) of UTC epoch seconds t at an airport."""
    loc = pd.to_datetime(pd.Series(t), unit="s", utc=True).dt.tz_convert(zone).dt.tz_localize(None).dt.normalize()
    return ((loc - pd.Timestamp("1970-01-01")).dt.days).to_numpy()


def utc_of_local(day: pd.Timestamp, hours: np.ndarray, zone: str) -> np.ndarray:
    """UTC epoch seconds of local clock times (hours after local midnight of a calendar day) at an airport."""
    loc = (day + pd.to_timedelta(hours, unit="h")).tz_localize(zone, nonexistent="shift_forward", ambiguous=False)
    return (loc.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds().to_numpy().astype(np.int64)


def day_movements(rng: np.random.Generator, ap: str, day: pd.Timestamp, ids: list, config: int, bad_day: float) -> list[dict]:
    """Departures and arrivals of one airport and one local day, with their labels (taxi times in seconds)."""
    rule = AIRPORTS[ap]
    rows = []
    # arrivals: landing time, taxi-in, stand
    n_arr = rng.poisson(rule["arr"])
    land = utc_of_local(day, rng.uniform(6.0, 24.0, n_arr), rule["tz"])
    for i in range(n_arr):
        taxi_in = int(np.clip(rng.normal(420, 150) * bad_day, 90, 1800))
        stand = rng.choice(list(STAND_EFFECT)) + "%02d" % rng.integers(1, 21)
        op = rng.choice(OPERATORS)
        actype = rng.choice(list(TYPES))
        flt = "%s%d" % (op, rng.integers(100, 9999))
        org = rng.choice(OTHER_AIRPORTS)
        ids[0] += 1
        rows.append({"MVT_ID_mvt": float(ids[0]), "FLIGHT_ID_mvt": float(ids[0] + 5 * 10**8), "FLIGHT_mvt": flt, "FLIGHT_RULE_mvt": "I",
                     "ADEP_mvt": org, "ADES_mvt": ap, "PHASE_mvt": "ARR", "MVT_TIME_UTC_mvt": int(land[i]),
                     "BLOCK_TIME_UTC_mvt": int(land[i]) + taxi_in, "SCHED_TIME_UTC_mvt": (int(land[i]) + int(rng.normal(300, 600))) // 60 * 60,
                     "AIRCRAFT_TYPE_mvt": actype, "RUNWAY_mvt": rng.choice(rule["arr_rwy"]), "STAND_mvt": stand, "TAXITIME_SEC_mvt": taxi_in,
                     "CALLSIGN_flt": flt, "ADEP_flt": org, "ADES_flt": ap, "ADES_FILED_flt": ap, "MARKET_SEGMENT_flt": SEGMENT[op],
                     "FLIGHT_RULE_flt": "I", "FLIGHT_TYPE_flt": "S", "AIRCRAFT_TYPE_flt": actype, "WK_TBL_CAT_flt": TYPES[actype],
                     "AIRCRAFT_OPERATOR_flt": op})
    # departures: schedule on the minute, off-block delay, runway of the day's configuration, taxi-out time
    n_dep = rng.poisson(rule["dep"])
    sched = utc_of_local(day, np.floor(rng.uniform(6.0, 23.75, n_dep) * 60) / 60, rule["tz"])
    delay = np.clip(rng.normal(240, 420, n_dep), 0, None) + (rng.random(n_dep) < 0.08) * rng.exponential(2400, n_dep)
    delay = (delay * bad_day).astype(np.int64)
    aobt = sched + delay + rng.integers(0, 60, n_dep)  # the true off-block time
    next_day = local_day(aobt, rule["tz"]) > local_day(sched, rule["tz"])  # left the stand after local midnight
    rwys = list(rule["rwy"])
    share = np.array([rule["rwy"][r][1 + config] for r in rwys])
    for i in range(n_dep):
        rwy = str(rng.choice(rwys, p=share / share.sum()))
        stand = rng.choice(list(STAND_EFFECT)) + "%02d" % rng.integers(1, 21)
        op = str(rng.choice(OPERATORS))
        actype = str(rng.choice(list(TYPES)))
        busy = int(np.sum(np.abs(aobt - aobt[i]) <= 900)) - 1  # other departures leaving their stands within 15 minutes
        taxi = rule["rwy"][rwy][0] + STAND_EFFECT[stand[0]] + (60 if TYPES[actype] == "H" else 0) + 35 * busy
        taxi = int(np.clip(rng.normal(taxi, 140) * bad_day, 180, 3600))
        mvt = int(aobt[i] + taxi)  # take-off time
        nm = rng.random() >= rule["no_nm"]
        # how the airport recorded the off-block time: a copy of the schedule; after local midnight, sometimes the right clock time on
        # the scheduled date (a label about 86,400 s too long); otherwise the measured time
        p_copy = COPY_SHARE[op] if nm else 0.35
        u = rng.random()
        if u < p_copy:
            block = int(sched[i])
        elif next_day[i] and u < p_copy + (0.5 if nm else 0.8):
            block = int(aobt[i]) - 86400
        else:
            block = int(aobt[i])
        flt = "%s%d" % (op, rng.integers(100, 9999))
        ades = str(rng.choice(OTHER_AIRPORTS))
        dur = int(rng.uniform(1.0, 3.5) * 3600)
        lobt = int(np.clip(sched[i] + delay[i] // 300 * 300, aobt[i] - 3600, aobt[i] + 3600))
        # as in the challenge data, a departure keeps its NM flight record only if |BLOCK - LOBT| <= 3,606 s (docs/DESIGN.md section 2.1)
        nm = nm and abs(block - lobt) <= 3606
        ids[0] += 1
        row = {"MVT_ID_mvt": float(ids[0]), "FLIGHT_mvt": flt, "FLIGHT_RULE_mvt": "I", "ADEP_mvt": ap, "ADES_mvt": ades, "PHASE_mvt": "DEP",
               "MVT_TIME_UTC_mvt": mvt, "BLOCK_TIME_UTC_mvt": block, "SCHED_TIME_UTC_mvt": int(sched[i]), "AIRCRAFT_TYPE_mvt": actype,
               "RUNWAY_mvt": rwy, "STAND_mvt": stand, "TAXITIME_SEC_mvt": mvt - block}
        if nm:  # the Network Manager's flight record: planned and actual times as the NM saw them
            eobt = lobt + int(abs(rng.normal(0, 240))) // 60 * 60
            aobt3 = mvt - int(rng.normal(900, 60))  # the NM derives its off-block time as take-off minus a default taxi time
            row.update({"FLIGHT_ID_mvt": float(ids[0] + 5 * 10**8), "LOBT_flt": lobt, "CALLSIGN_flt": flt, "ADEP_flt": ap, "ADES_flt": ades,
                        "ADES_FILED_flt": ades, "MARKET_SEGMENT_flt": SEGMENT[op], "IOBT_flt": lobt, "FLIGHT_RULE_flt": "I",
                        "FLIGHT_TYPE_flt": "S", "AIRCRAFT_TYPE_flt": actype, "WK_TBL_CAT_flt": TYPES[actype], "AIRCRAFT_OPERATOR_flt": op,
                        "EOBT_1_flt": eobt, "ARVT_1_flt": eobt + dur + 600, "AOBT_3_flt": aobt3, "ARVT_3_flt": aobt3 + dur + 600})
        rows.append(row)
    return rows


def frame(rows: list[dict], labelled_departures: bool) -> pa.Table:
    """Rows as an Arrow table with the challenge schema (times in seconds become UTC timestamps; missing values become nulls)."""
    df = pd.DataFrame(rows)
    if not labelled_departures:  # the input of the month to predict: departures have no off-block time and no taxi time
        dep = df["PHASE_mvt"] == "DEP"
        df.loc[dep, "BLOCK_TIME_UTC_mvt"] = np.nan
        df.loc[dep, "TAXITIME_SEC_mvt"] = np.nan
    cols = {}
    for f in SCHEMA:
        s = df[f.name] if f.name in df else pd.Series([None] * len(df))
        if f.type == TS:
            cols[f.name] = pa.array(pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit="s", utc=True), type=TS, from_pandas=True)
        elif f.type == pa.int32():
            cols[f.name] = pa.array(pd.to_numeric(s, errors="coerce").astype("Int32"), type=pa.int32(), from_pandas=True)
        elif f.type == pa.float64():
            cols[f.name] = pa.array(pd.to_numeric(s, errors="coerce"), type=pa.float64(), from_pandas=True)
        else:
            cols[f.name] = pa.array([None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in s], type=STR)
    return pa.Table.from_pydict(cols, schema=SCHEMA)


def main() -> int:
    ap_ = argparse.ArgumentParser(description="Write a small synthetic data set in the format of the challenge files into data/.")
    ap_.add_argument("--force", action="store_true", help="replace data files of the same names")
    ap_.add_argument("--seed", type=int, default=2026)
    args = ap_.parse_args()
    starts = list(pd.date_range("2025-01-01", "2026-01-01", freq="MS"))  # twelve training months, then January 2026 to predict
    ends = starts[1:] + [pd.Timestamp("2026-02-01")]
    names = ["training_%s_%s.parquet" % (a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")) for a, b in zip(starts[:12], ends[:12])]
    names += ["ranking.parquet", "submitting.parquet"]
    there = [n for n in names if (DATA / n).exists()]
    if there and not args.force:
        print("data/ already holds %d files of the names this script writes (for example %s); nothing written "
              "(use --force to replace them)" % (len(there), there[0]))
        return 1
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ids = [9 * 10**8]  # movement ids of the sample start at 900,000,001 (the challenge files use other ranges)
    for k, (a, b) in enumerate(zip(starts, ends)):
        rows = []
        for day in pd.date_range(a, b - pd.Timedelta(days=1), freq="D"):
            bad = 1.0 + (rng.random() < 0.06) * rng.uniform(0.3, 0.8)  # a few days with weather that slows everything
            for ap in AIRPORTS:
                rows += day_movements(rng, ap, day, ids, config=int(rng.random() < 0.4), bad_day=bad)
        # a training file holds the movements of its UTC month (the challenge files are cut by UTC time)
        lo, hi = a.tz_localize("UTC").value // 10**9, b.tz_localize("UTC").value // 10**9
        keep = [r for r in rows if lo <= r["MVT_TIME_UTC_mvt"] < hi]
        keep.sort(key=lambda r: r["MVT_TIME_UTC_mvt"])
        test = k == 12
        t = frame(keep, labelled_departures=not test)
        name = names[k] if not test else "ranking.parquet"
        pq.write_table(t, DATA / name)
        is_dep = [p == "DEP" for p in t.column("PHASE_mvt").to_pylist()]
        print("wrote data/%s: %d movements, %d departures%s" % (name, t.num_rows, sum(is_dep), " without labels" if test else ""))
        if test:  # the template to fill: the departures of the input file, in its order
            dep_ids = [i for i, d in zip(t.column("MVT_ID_mvt").to_pylist(), is_dep) if d]
            tpl = pa.Table.from_pydict({"MVT_ID_mvt": pa.array(dep_ids, type=pa.float64()),
                                        "TAXITIME_SEC_mvt": pa.array([None] * len(dep_ids), type=pa.int32())}, schema=TEMPLATE_SCHEMA)
            pq.write_table(tpl, DATA / "submitting.parquet")
            print("wrote data/submitting.parquet: %d departures to predict" % tpl.num_rows)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
