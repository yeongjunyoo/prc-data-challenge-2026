# SPDX-License-Identifier: GPL-3.0-only
"""PRC departure rows of every day (2025 training and the 2026 test), with only the columns the ADS-B matcher reads.

Usage (repo root): PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/adsb/prep_all.py
Writes research/adsb/prc_all.parquet sorted by UTC day: MVT_ID_mvt, day, src (file month, 0 = ranking), ADEP_mvt,
MVT_TIME_UTC_mvt, CALLSIGN_flt, AIRCRAFT_TYPE_mvt, RUNWAY_mvt. Same raw columns and types as prep_days.py, so
adsb_match.match_day sees the same input. Checks that the rows are exactly the departures of cache/feats_v25.parquet.
"""
import glob
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COLS = ['MVT_ID_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'CALLSIGN_flt', 'AIRCRAFT_TYPE_mvt', 'RUNWAY_mvt']


def main():
    files = sorted(glob.glob(os.path.join(ROOT, 'data', 'training_*.parquet'))) + [os.path.join(ROOT, 'data', 'ranking.parquet')]
    parts = []
    for f in files:
        t = ds.dataset(f, format='parquet').to_table(columns=COLS, filter=ds.field('PHASE_mvt') == 'DEP')
        src = 0 if f.endswith('ranking.parquet') else int(os.path.basename(f)[14:16])
        t = t.append_column('src', pa.array(np.full(len(t), src, dtype=np.int8)))
        parts.append(t)
        print(os.path.basename(f), len(t), flush=True)
    t = pa.concat_tables(parts)
    day = pc.strftime(t['MVT_TIME_UTC_mvt'], format='%Y-%m-%d')
    t = t.append_column('day', day)
    t = t.take(pc.sort_indices(t, sort_keys=[('day', 'ascending'), ('MVT_ID_mvt', 'ascending')]))
    t = t.select(['MVT_ID_mvt', 'day', 'src', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'CALLSIGN_flt', 'AIRCRAFT_TYPE_mvt', 'RUNWAY_mvt'])
    ids = np.sort(t['MVT_ID_mvt'].to_numpy())
    fv = np.sort(pq.read_table(os.path.join(ROOT, 'cache', 'feats_v25.parquet'), columns=['MVT_ID_mvt'])['MVT_ID_mvt'].to_numpy())
    assert len(np.unique(ids)) == len(ids), 'duplicate MVT_ID_mvt'
    assert len(ids) == len(fv) and (ids == fv).all(), f'rows differ from feats_v25: {len(ids)} vs {len(fv)}'
    out = os.path.join(ROOT, 'research', 'adsb', 'prc_all.parquet')
    pq.write_table(t, out, compression='zstd')
    d = t['day'].to_numpy(zero_copy_only=False)
    print('wrote', out, len(t), 'rows,', len(np.unique(d)), 'days', d.min(), '..', d.max(), '(same departures as feats_v25)')


if __name__ == '__main__':
    main()
