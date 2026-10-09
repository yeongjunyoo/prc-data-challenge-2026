# SPDX-License-Identifier: GPL-3.0-only
"""Cut the PRC departure rows of the ADS-B pilot days into one small parquet for the Mac.

Usage (repo root): PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/adsb/prep_days.py 2025-01-15 2025-07-15 2026-01-15 2026-07-15
Writes research/adsb/prc_days.parquet (departures of those UTC days, identifiers, times, label when the
file has one, and the hold-out prediction of cache/val_m6r25sgc75w_s0.parquet when the row is in it).
Light: reads only the needed columns with a row filter.
"""
import os
import sys

import pandas as pd
import pyarrow.dataset as ds

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COLS = ['MVT_ID_mvt', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt',
        'SCHED_TIME_UTC_mvt', 'AIRCRAFT_TYPE_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'TAXITIME_SEC_mvt', 'LOBT_flt',
        'CALLSIGN_flt', 'AIRCRAFT_TYPE_flt', 'AIRCRAFT_OPERATOR_flt', 'IOBT_flt', 'EOBT_1_flt', 'AOBT_3_flt']
VAL = os.path.join(ROOT, 'cache', 'val_m6r25sgc75w_s0.parquet')


def src_for(day):
    if day.startswith('2026'):
        return os.path.join(ROOT, 'data', 'ranking.parquet')
    y, m = int(day[:4]), int(day[5:7])
    y2, m2 = (y + 1, 1) if m == 12 else (y, m + 1)
    return os.path.join(ROOT, 'data', f'training_{y}-{m:02d}-01_{y2}-{m2:02d}-01.parquet')


def main(days):
    parts = []
    for day in days:
        d0 = pd.Timestamp(day, tz='UTC')
        d1 = d0 + pd.Timedelta(days=1)
        dset = ds.dataset(src_for(day), format='parquet')
        f = (ds.field('PHASE_mvt') == 'DEP') & (ds.field('MVT_TIME_UTC_mvt') >= d0) & (ds.field('MVT_TIME_UTC_mvt') < d1)
        t = dset.to_table(columns=COLS, filter=f).to_pandas()
        t['day'] = day
        parts.append(t)
        print(day, len(t), 'departures', 'label non-null', int(t.TAXITIME_SEC_mvt.notna().sum()), flush=True)
    df = pd.concat(parts, ignore_index=True)
    v = pd.read_parquet(VAL)
    df = df.merge(v.rename(columns={'y': 'y_val'}), on='MVT_ID_mvt', how='left')
    print('rows with hold-out pred:', int(df.pred.notna().sum()), 'of', len(df))
    lab = df[df.pred.notna()]
    if len(lab):
        print('label == y_val on joined rows:', bool(((lab.TAXITIME_SEC_mvt - lab.y_val).abs() < 1e-6).all()))
    for c in ('CALLSIGN_flt', 'FLIGHT_mvt'):
        print(c, 'null share by day:', df.groupby('day')[c].apply(lambda s: round(float(s.isna().mean()), 4)).to_dict())
    print(df[['FLIGHT_mvt', 'CALLSIGN_flt', 'AIRCRAFT_OPERATOR_flt', 'RUNWAY_mvt', 'STAND_mvt']].head(8).to_string())
    out = os.path.join(ROOT, 'research', 'adsb', 'prc_days.parquet')
    df.to_parquet(out, index=False)
    print('wrote', out, len(df))


if __name__ == '__main__':
    main(sys.argv[1:])
