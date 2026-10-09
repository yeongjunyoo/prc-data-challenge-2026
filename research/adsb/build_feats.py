# SPDX-License-Identifier: GPL-3.0-only
"""Build the per-departure ADS-B feature table (cache/<name>.parquet) from adsb.lol day extracts.

Usage (repo root):
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/adsb/build_feats.py --name adsb_v1 --out-dir <adsb out dir> \
      --prc research/adsb/prc_days.parquet [--days-file research/adsb/days_routeA_all.txt]
Inputs: <out-dir>/<day>/events.parquet (adsb_day.py), PRC departures of those days (prep_days.py).
Output: cache/<name>.parquet, one row per PRC departure of the covered days (matched or not), keyed by MVT_ID_mvt, and
        cache/<name>.json (days, row counts, match rates by airport and month, sha16, source and licence).
Source: adsb.lol globe_history (GitHub releases adsblol/globe_history_2025, _2026), Open Database License 1.0.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from adsb_match import match_day  # noqa: E402

KEEP_EV = ['ev_t_to', 'ev_to_gap', 'ev_roll_trk', 'ev_seg_reason', 'ev_t_first', 'ev_first_gs', 'ev_first_dist_m',
           'ev_first_stationary', 'ev_stand_seen', 'ev_stand_dur', 'ev_t_fm', 'ev_fm_gap', 'ev_n_mid_stops',
           'ev_mid_stop_s', 'ev_mid_stop_max', 'ev_n_long_stops', 'ev_t_fm_last', 'ev_fm_last_gap', 'ev_last_long_dur',
           'ev_n_pts', 'ev_max_gap_after_fm', 'ev_share_mlat', 'ev_icao', 'ev_typ']
KEEP = ['MVT_ID_mvt', 'day', 'ADEP_mvt', 'mvt', 'how', 'matched', 'fm_seen', 'fm_ok', 'first_moving', 'dt_to', 'est_to',
        'est_mvt', 'est_first', 'est_mvt_last'] + KEEP_EV


def sha16(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='adsb_v1')
    ap.add_argument('--out-dir', default=os.environ.get('PRC_ADSB_OUT', 'adsb_out'))
    ap.add_argument('--prc', default=str(HERE / 'prc_days.parquet'))
    ap.add_argument('--days-file', default='')
    a = ap.parse_args()
    prc = pd.read_parquet(a.prc, columns=['MVT_ID_mvt', 'day', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'CALLSIGN_flt',
                                          'AIRCRAFT_TYPE_mvt', 'RUNWAY_mvt'])
    days = sorted(prc.day.unique())
    if a.days_file:
        want = set(open(a.days_file, encoding='utf-8').read().split())
        days = [d for d in days if d in want]
    parts, missing = [], []
    for d in days:
        evf = os.path.join(a.out_dir, d, 'events.parquet')
        if not os.path.exists(evf):
            missing.append(d)
            continue
        m = match_day(prc[prc.day == d], pd.read_parquet(evf))
        for c in ('ev_first_stationary', 'ev_stand_seen'):
            m[c] = m[c].astype(float)
        parts.append(m[KEEP])
    t = pd.concat(parts, ignore_index=True)
    assert t.MVT_ID_mvt.is_unique
    for c in ('matched', 'fm_seen', 'fm_ok', 'first_moving'):
        t[c] = t[c].astype(bool)
    out = ROOT / 'cache' / f'{a.name}.parquet'
    t.to_parquet(out, index=False)
    t['ym'] = t.day.str[:7]
    rates = (t.groupby(['ym', 'ADEP_mvt']).matched.mean() * 100).round(1)
    meta = dict(name=a.name, file=f'cache/{a.name}.parquet', sha16=sha16(out), rows=int(len(t)), days=len(parts),
                days_missing=missing, matched=int(t.matched.sum()), fm_ok=int(t.fm_ok.sum()),
                match_rate_by_month={ym: float(round(100 * g.matched.mean(), 2)) for ym, g in t.groupby('ym')},
                match_rate_by_month_airport={f'{k[0]} {k[1]}': float(v) for k, v in rates.items()},
                source='adsb.lol globe_history (GitHub releases adsblol/globe_history_2025, adsblol/globe_history_2026; '
                       'one release per day as listed in PREFERRED_RELEASES.txt, 2025-12-14..17 mlatonly entries replaced by prod)',
                licence='Open Database License (ODbL) 1.0; feeder contributions CC0 1.0',
                code=['research/adsb/adsb_day.py', 'research/adsb/adsb_match.py', 'research/adsb/build_feats.py'],
                written_kst=dt.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST'))
    (ROOT / 'cache' / f'{a.name}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in meta.items() if k != 'match_rate_by_month_airport'}, ensure_ascii=False, indent=1))



if __name__ == '__main__':
    main()
