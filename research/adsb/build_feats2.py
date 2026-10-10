# SPDX-License-Identifier: GPL-3.0-only
"""Per-departure ADS-B table for every extracted day (route A and route B): cache/<name>.parquet, same columns and the same
matcher as build_feats.py (adsb_v1), one row per PRC departure of each extracted day, keyed by MVT_ID_mvt.

Usage (repo root):
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/adsb/build_feats2.py --name adsb_v2 [--out-dir D:/prc-adsb/out]
Inputs: research/adsb/prc_all.parquet (prep_all.py), <out-dir>/<day>/events.parquet of days whose meta.json exists
        (adsb_day.py writes meta.json last, so a day still being extracted is skipped).
Output: cache/<name>.parquet and cache/<name>.json (days used, days missing per month, match and start-observed rates by
        month and airport, sha16). Departures of days not yet extracted are not in the table (the feature build marks them
        unknown, not unmatched).
Source: adsb.lol globe_history (Open Database License 1.0), README 「외부 데이터」.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from adsb_match import match_day  # noqa: E402
from build_feats import KEEP, sha16  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='adsb_v2')
    ap.add_argument('--out-dir', default='D:/prc-adsb/out')
    ap.add_argument('--prc', default=str(HERE / 'prc_all.parquet'))
    a = ap.parse_args()
    t0 = time.time()
    t = pq.read_table(a.prc)
    day = t['day'].to_numpy(zero_copy_only=False)
    days, start = np.unique(day, return_index=True)
    stop = np.r_[start[1:], len(day)]
    parts, missing = [], []
    for d, s, e in zip(days, start, stop):
        evf, mf = os.path.join(a.out_dir, d, 'events.parquet'), os.path.join(a.out_dir, d, 'meta.json')
        if not (os.path.exists(evf) and os.path.exists(mf)):
            missing.append(str(d))
            continue
        deps = t.slice(int(s), int(e - s)).to_pandas()
        m = match_day(deps, pd.read_parquet(evf))
        for c in ('ev_first_stationary', 'ev_stand_seen'):
            m[c] = m[c].astype(float)
        parts.append(m[KEEP])
    r = pd.concat(parts, ignore_index=True)
    assert r.MVT_ID_mvt.is_unique
    for c in ('matched', 'fm_seen', 'fm_ok', 'first_moving'):
        r[c] = r[c].astype(bool)
    out = ROOT / 'cache' / f'{a.name}.parquet'
    r.to_parquet(out, index=False)
    r['ym'] = r.day.str[:7]
    used = sorted(set(r.day))
    miss_by_month = pd.Series([m[:7] for m in missing]).value_counts().sort_index().to_dict() if missing else {}
    rates = r.groupby(['ym', 'ADEP_mvt']).agg(n=('matched', 'size'), matched=('matched', 'mean'), fm_seen=('fm_seen', 'mean'))
    meta = dict(name=a.name, file=f'cache/{a.name}.parquet', sha16=sha16(out), rows=int(len(r)), days=len(used),
                days_by_month=pd.Series([u[:7] for u in used]).value_counts().sort_index().to_dict(),
                days_missing_by_month=miss_by_month, days_missing=missing,
                matched=int(r.matched.sum()), fm_seen=int(r.fm_seen.sum()), fm_ok=int(r.fm_ok.sum()),
                match_rate_by_month={ym: float(round(100 * g.matched.mean(), 2)) for ym, g in r.groupby('ym')},
                by_month_airport={f'{k[0]} {k[1]}': dict(n=int(v.n), matched=round(100 * float(v.matched), 1), fm_seen=round(100 * float(v.fm_seen), 1))
                                  for k, v in rates.iterrows()},
                source='adsb.lol globe_history (GitHub releases adsblol/globe_history_2025, adsblol/globe_history_2026; '
                       'one release per day as listed in PREFERRED_RELEASES.txt, 2025-12-14..17 mlatonly entries replaced by prod, '
                       '2026-01-29 from prod-0)',
                licence='Open Database License (ODbL) 1.0; feeder contributions CC0 1.0',
                code=['research/adsb/adsb_day.py', 'research/adsb/adsb_match.py', 'research/adsb/build_feats.py', 'research/adsb/build_feats2.py',
                      'research/adsb/prep_all.py'],
                seconds=round(time.time() - t0, 1),
                written_kst=dt.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST'))
    (ROOT / 'cache' / f'{a.name}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in meta.items() if k not in ('by_month_airport', 'days_missing')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
