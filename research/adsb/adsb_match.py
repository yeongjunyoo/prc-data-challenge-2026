# SPDX-License-Identifier: GPL-3.0-only
"""Match ADS-B takeoff events (adsb_day.py) to the PRC departures of the same UTC day.

Usage: .venv/bin/python adsb_match.py --prc prc_days.parquet --out out 2025-01-15 2025-07-15 ...
Writes out/<day>/matched.parquet (one row per PRC departure, event columns prefixed ev_) and
out/match_summary.json (per day and airport: departures, matched, how, stand seen, first seen moving,
takeoff time agreement). Rules, in order, each greedy by |t_to - MVT|, one event per departure:
  1. callsign: same airport, CALLSIGN_flt == ADS-B callsign, |t_to - MVT| <= 1200 s
  2. type+time: same airport, AIRCRAFT_TYPE_mvt == ADS-B type, |t_to - MVT| <= 120 s, and the take-off
     roll track within 30 deg of the runway designator when both are known
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

APTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
CS_WIN = 1200.0
TT_WIN = 120.0


def rwy_heading(r):
    m = re.match(r'^(\d{2})', str(r or ''))
    return float(m.group(1)) * 10.0 if m else np.nan


def angdiff(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def greedy(cand, deps_idx_taken, ev_taken):
    pairs = []
    for r in cand.sort_values('adt').itertuples():
        if r.dep in deps_idx_taken or r.ev in ev_taken:
            continue
        pairs.append((r.dep, r.ev))
        deps_idx_taken.add(r.dep)
        ev_taken.add(r.ev)
    return pairs


def match_day(deps, ev):
    deps = deps.copy().reset_index(drop=True)
    ev = ev.copy().reset_index(drop=True)
    deps['mvt'] = deps.MVT_TIME_UTC_mvt.astype('int64') / 1e9 if deps.MVT_TIME_UTC_mvt.dtype.kind == 'M' else np.nan
    if str(deps.MVT_TIME_UTC_mvt.dtype).startswith('datetime64[us'):
        deps['mvt'] = deps.MVT_TIME_UTC_mvt.astype('int64') / 1e6
    elif str(deps.MVT_TIME_UTC_mvt.dtype).startswith('datetime64[ms'):
        deps['mvt'] = deps.MVT_TIME_UTC_mvt.astype('int64') / 1e3
    else:
        deps['mvt'] = (deps.MVT_TIME_UTC_mvt - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds()
    deps['ev'] = -1
    deps['how'] = ''
    taken_d, taken_e = set(), set()
    cs = deps[['mvt', 'ADEP_mvt', 'CALLSIGN_flt']].reset_index().rename(columns={'index': 'dep'})
    cs = cs[cs.CALLSIGN_flt.notna()]
    e1 = ev[['apt', 'cs', 't_to']].reset_index().rename(columns={'index': 'ev'})
    c = cs.merge(e1, left_on=['ADEP_mvt', 'CALLSIGN_flt'], right_on=['apt', 'cs'])
    c['adt'] = (c.t_to - c.mvt).abs()
    for d, e in greedy(c[c.adt <= CS_WIN], taken_d, taken_e):
        deps.at[d, 'ev'] = e
        deps.at[d, 'how'] = 'callsign'
    rest = deps[deps.ev < 0][['mvt', 'ADEP_mvt', 'AIRCRAFT_TYPE_mvt', 'RUNWAY_mvt']].reset_index().rename(columns={'index': 'dep'})
    e2 = ev[~ev.index.isin(taken_e)][['apt', 'typ', 't_to', 'roll_trk']].reset_index().rename(columns={'index': 'ev'})
    c = rest.merge(e2, left_on=['ADEP_mvt', 'AIRCRAFT_TYPE_mvt'], right_on=['apt', 'typ'])
    c['adt'] = (c.t_to - c.mvt).abs()
    hd = c.RUNWAY_mvt.map(rwy_heading)
    ok_rwy = hd.isna() | c.roll_trk.isna() | (angdiff(hd, c.roll_trk) <= 30)
    for d, e in greedy(c[(c.adt <= TT_WIN) & ok_rwy], taken_d, taken_e):
        deps.at[d, 'ev'] = e
        deps.at[d, 'how'] = 'type_time'
    j = ev.add_prefix('ev_')
    deps = deps.join(j, on='ev')
    deps.loc[deps.ev < 0, [k for k in deps.columns if k.startswith('ev_')]] = np.nan
    deps['matched'] = deps.ev >= 0
    deps['fm_seen'] = deps.matched & deps.ev_t_fm.notna()
    deps['first_moving'] = deps.matched & deps.ev_t_fm.isna() & (deps.ev_seg_reason != 'landing')
    deps['fm_ok'] = deps.fm_seen & (deps.ev_fm_gap <= 120)   # the start of the motion itself was observed
    deps['dt_to'] = deps.ev_t_to - deps.mvt
    deps['est_to'] = deps.ev_t_to - deps.ev_t_fm       # ADS-B takeoff minus ADS-B first motion
    deps['est_mvt'] = deps.mvt - deps.ev_t_fm          # MVT_TIME minus ADS-B first motion
    deps['est_first'] = deps.mvt - deps.ev_t_first     # MVT_TIME minus first ground point
    t_last = deps.ev_t_fm_last.where(deps.ev_t_fm_last.notna(), deps.ev_t_fm)
    deps['est_mvt_last'] = deps.mvt - t_last           # MVT_TIME minus motion after the last stop of 10 min or more
    return deps


def summarize(m):
    rows = {}
    for ap in APTS + ['ALL']:
        g = m if ap == 'ALL' else m[m.ADEP_mvt == ap]
        n = len(g)
        mm = g[g.matched]
        r = dict(deps=n,
                 matched=int(g.matched.sum()), match_rate=round(float(g.matched.mean()), 4) if n else None,
                 by_callsign=int((g.how == 'callsign').sum()), by_type_time=int((g.how == 'type_time').sum()),
                 fm_seen=int(g.fm_seen.sum()), fm_rate=round(float(g.fm_seen.mean()), 4) if n else None,
                 fm_ok=int(g.fm_ok.sum()), fm_ok_rate=round(float(g.fm_ok.mean()), 4) if n else None,
                 first_moving=int(g.first_moving.sum()),
                 landing_no_stand=int((g.matched & g.ev_t_fm.isna() & (g.ev_seg_reason == 'landing')).sum()))
        if len(mm):
            r['dt_to_median'] = round(float(mm.dt_to.median()), 1)
            r['dt_to_abs_le60'] = round(float((mm.dt_to.abs() <= 60).mean()), 4)
            r['dt_to_abs_p90'] = round(float(mm.dt_to.abs().quantile(0.9)), 1)
            f = mm[mm.fm_seen]
            if len(f):
                r['est_to_median'] = round(float(f.est_to.median()), 1)
                r['est_mvt_median'] = round(float(f.est_mvt.median()), 1)
                r['stand_dur_median'] = round(float(f.ev_stand_dur.median()), 1)
                r['fm_gap_median'] = round(float(f.ev_fm_gap.median()), 1)
            r['first_gs_le1_share'] = round(float((mm.ev_first_gs <= 1).mean()), 4)
            r['first_dist_median_m'] = round(float(mm.ev_first_dist_m.median()), 1)
        rows[ap] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prc', default='prc_days.parquet')
    ap.add_argument('--out', default='out')
    ap.add_argument('days', nargs='+')
    a = ap.parse_args()
    prc = pd.read_parquet(a.prc)
    sumf = os.path.join(a.out, 'match_summary.json')
    allsum = json.load(open(sumf)) if os.path.exists(sumf) else {}
    for day in a.days:
        deps = prc[prc.day == day]
        evf = os.path.join(a.out, day, 'events.parquet')
        ev = pd.read_parquet(evf)
        m = match_day(deps, ev)
        m.to_parquet(os.path.join(a.out, day, 'matched.parquet'), index=False)
        s = summarize(m)
        allsum[day] = s
        print(day)
        print(pd.DataFrame(s).T[['deps', 'matched', 'match_rate', 'by_callsign', 'by_type_time', 'fm_seen', 'fm_rate',
                                 'fm_ok_rate', 'first_moving', 'dt_to_median', 'dt_to_abs_le60']].to_string())
    json.dump(allsum, open(sumf, 'w'), indent=1)


if __name__ == '__main__':
    main()
