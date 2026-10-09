# SPDX-License-Identifier: GPL-3.0-only
"""R17-A: the final-phase ADS-B table (cache/adsb_f1.parquet = research/adsb/build_feats2.py on research/r17_a/prc_final.parquet, MVT times
of data/final_phase/final_ranking.parquet, jittered by the organiser) against the public-phase table (cache/adsb_v2.parquet, MVT times of
data/ranking.parquet) on the 2026-01 and 2026-07 departures both files hold, plus matched counts per month and airport of the final table.
Private notes research/2026-10-08-r17-a-adsb-fill.md. Read-only on its inputs; prints a report and writes research/r17_a/cmp_final_r17a.json.
    python research/r17_a/cmp_final_r17a.py
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C = os.path.join(ROOT, 'cache')
EST = ['dt_to', 'est_to', 'est_mvt', 'est_first', 'est_mvt_last']


def main():
    f = pd.read_parquet(os.path.join(C, 'adsb_f1.parquet'))
    v = pd.read_parquet(os.path.join(C, 'adsb_v2.parquet'))
    v = v[v.day.str[:4] == '2026']
    out = {}
    f['ym'] = f.day.str[:7]
    g = f.groupby(['ym', 'ADEP_mvt'])
    tab = pd.DataFrame({'deps': g.size(), 'matched': g.matched.sum(), 'match_rate': (100 * g.matched.mean()).round(1),
                        'fm_seen_rate': (100 * g.fm_seen.mean()).round(1)})
    print('final table: departures and matches per month and airport (days with ADS-B data only)')
    print(tab.to_string())
    tm = f.groupby('ym').agg(deps=('matched', 'size'), matched=('matched', 'sum'), match_rate=('matched', 'mean'))
    print(tm.to_string())
    out['by_month_airport'] = {f'{k[0]} {k[1]}': {kk: (float(vv) if isinstance(vv, (np.floating, float)) else int(vv)) for kk, vv in r.items()}
                               for k, r in tab.iterrows()}
    out['by_month'] = {k: {'deps': int(r.deps), 'matched': int(r.matched), 'match_rate': round(float(r.match_rate), 4)} for k, r in tm.iterrows()}
    # Jan/Jul: the same departures matched on the final (jittered) and the public (ranking.parquet) MVT times
    j = f.merge(v, on='MVT_ID_mvt', suffixes=('_f', '_r'))
    print(f'\nJan/Jul departures in both tables: {len(j)} (final table rows in 2026-01/07: {int(f.ym.isin(["2026-01", "2026-07"]).sum())}, '
          f'public table 2026 rows: {len(v)})')
    jit = j.mvt_f - j.mvt_r
    both = j.matched_f & j.matched_r
    same_ev = both & (j.ev_t_to_f == j.ev_t_to_r) & (j.ev_icao_f == j.ev_icao_r)
    day_moved = j.day_f != j.day_r
    rep = {'rows_both': int(len(j)), 'jitter_s': {'mean': round(float(jit.mean()), 3), 'sd': round(float(jit.std()), 3),
                                                  'min': float(jit.min()), 'max': float(jit.max()), 'share_nonzero': round(float((jit != 0).mean()), 4)},
           'utc_day_moved': int(day_moved.sum()),
           'matched_final': int(j.matched_f.sum()), 'matched_public': int(j.matched_r.sum()), 'matched_both': int(both.sum()),
           'matched_final_only': int((j.matched_f & ~j.matched_r).sum()), 'matched_public_only': int((~j.matched_f & j.matched_r).sum()),
           'same_event_when_both': int(same_ev.sum()), 'different_event_when_both': int((both & ~same_ev).sum()),
           'how_changed': int((both & (j.how_f != j.how_r)).sum())}
    se = j[same_ev]
    for c in EST:
        d = (se[f'{c}_f'] - se[f'{c}_r']).to_numpy(dtype=float)
        ok = np.isfinite(d)
        if ok.any():
            sign = -1.0 if c == 'dt_to' else 1.0
            resid = d[ok] - sign * (se.mvt_f - se.mvt_r).to_numpy(dtype=float)[ok]
            rep[f'{c}_diff'] = {'n': int(ok.sum()), 'max_abs_diff': round(float(np.abs(d[ok]).max()), 3),
                                'max_abs_diff_minus_jitter': round(float(np.abs(resid).max()), 6)}
    for k in ('2026-01', '2026-07'):
        s = j[j.ym == k] if 'ym' in j.columns else j[j.day_f.str[:7] == k]
        rep[k] = {'rows': int(len(s)), 'matched_final': int(s.matched_f.sum()), 'matched_public': int(s.matched_r.sum()),
                  'agree': round(float((s.matched_f == s.matched_r).mean()), 6) if len(s) else None}
    print(json.dumps(rep, indent=1))
    out['jan_jul_vs_public'] = rep
    open(os.path.join(ROOT, 'research', 'r17_a', 'cmp_final_r17a.json'), 'w', encoding='utf-8').write(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
