# SPDX-License-Identifier: GPL-3.0-only
"""R17-A copy of research/adsb/prep_all.py (private notes research/2026-10-08-r17-a-adsb-fill.md) for the final-phase test file: PRC departure rows of
data/final_phase/final_ranking.parquet (2026-01, 02, 06, 07; MVT_TIME_UTC_mvt jittered by the organiser) with only the columns the ADS-B
matcher reads, the same raw columns and types as prep_all.py, so adsb_match.match_day sees the same kind of input. The UTC day comes from the
final file's own (jittered) MVT time. No check against feats_v25 (these rows are not in it).

Usage (sandbox or repo root): python research/r17_a/prep_final_r17a.py
Writes research/r17_a/prc_final.parquet sorted by UTC day: MVT_ID_mvt, day, src (0 = test), ADEP_mvt, MVT_TIME_UTC_mvt, CALLSIGN_flt,
AIRCRAFT_TYPE_mvt, RUNWAY_mvt.
"""
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COLS = ['MVT_ID_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'CALLSIGN_flt', 'AIRCRAFT_TYPE_mvt', 'RUNWAY_mvt']


def main():
    f = os.path.join(ROOT, 'data', 'final_phase', 'final_ranking.parquet')
    t = ds.dataset(f, format='parquet').to_table(columns=COLS, filter=ds.field('PHASE_mvt') == 'DEP')
    t = t.append_column('src', pa.array(np.zeros(len(t), dtype=np.int8)))
    day = pc.strftime(t['MVT_TIME_UTC_mvt'], format='%Y-%m-%d')
    t = t.append_column('day', day)
    t = t.take(pc.sort_indices(t, sort_keys=[('day', 'ascending'), ('MVT_ID_mvt', 'ascending')]))
    t = t.select(['MVT_ID_mvt', 'day', 'src', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'CALLSIGN_flt', 'AIRCRAFT_TYPE_mvt', 'RUNWAY_mvt'])
    ids = t['MVT_ID_mvt'].to_numpy()
    assert len(np.unique(ids)) == len(ids), 'duplicate MVT_ID_mvt'
    out = os.path.join(ROOT, 'research', 'r17_a', 'prc_final.parquet')
    pq.write_table(t, out, compression='zstd')
    d = t['day'].to_numpy(zero_copy_only=False)
    months, n = np.unique(np.array([x[:7] for x in d]), return_counts=True)
    print('wrote', out, len(t), 'departures,', len(np.unique(d)), 'days', d.min(), '..', d.max(), dict(zip(months.tolist(), n.tolist())))


if __name__ == '__main__':
    main()
