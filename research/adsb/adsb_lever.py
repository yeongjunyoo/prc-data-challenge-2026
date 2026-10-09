# SPDX-License-Identifier: GPL-3.0-only
"""Size the ADS-B lever on the labelled pilot days (2025) with the hold-out predictions of our model.

Usage: .venv/bin/python adsb_lever.py --out out --test-days 2026-01-15 2026-07-15 2025-01-15 2025-07-15
Reads out/<day>/matched.parquet (adsb_match.py). Writes out/lever.json and prints the tables.

Row sets (2025 days):  FM = matched with ADS-B first motion,  M = matched (any takeoff),  ALL = all departures
  clean = body rows, y <= 3600 s
Estimators: est_to = t_to - t_fm, est_mvt = MVT - t_fm, est_first = MVT - t_first (lower bound when first seen
moving), pred = hold-out prediction (cache/val_m6r25sgc75w_s0).
Lever: (1) cross-day linear stack on FM rows, (2) fixed blend rule, (3) LightGBM residual corrector with
grouped 5-fold CV over both days, with and without the ADS-B columns (the difference is the ADS-B share).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

APTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
SERVER_BEST = 267.84


def st(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    ok = ~(np.isnan(y) | np.isnan(x))
    y, x = y[ok], x[ok]
    if len(y) < 3:
        return dict(n=int(len(y)))
    e = x - y
    r = float(np.corrcoef(y, x)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
    return dict(n=int(len(y)), r=round(r, 4), rmse=round(float(np.sqrt(np.mean(e ** 2))), 1),
                bias=round(float(e.mean()), 1), mae=round(float(np.mean(np.abs(e))), 1),
                medae=round(float(np.median(np.abs(e))), 1))


def load(out, days):
    parts = []
    for d in days:
        f = os.path.join(out, d, 'matched.parquet')
        if os.path.exists(f):
            parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def add_cols(m):
    m = m.copy()
    m['y'] = m.TAXITIME_SEC_mvt
    blk = m.BLOCK_TIME_UTC_mvt
    sch = m.SCHED_TIME_UTC_mvt
    m['sched_copy'] = (blk - sch).dt.total_seconds().abs() <= 60
    m['clean'] = m.y <= 3600   # body rows (the research notes' 3,600 s body/tail split)
    m['hour'] = m.MVT_TIME_UTC_mvt.dt.hour
    m['fm_minus_pred'] = m.est_mvt - m.pred
    m['first_minus_pred'] = m.est_first - m.pred
    m['last_minus_pred'] = m.est_mvt_last - m.pred
    sch_s = (m.SCHED_TIME_UTC_mvt - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds()
    m['fm_minus_sched'] = m.ev_t_fm - sch_s
    m['first_minus_sched'] = m.ev_t_first - sch_s
    return m


def block_feats(m, out):
    """Airport state from all ADS-B takeoffs of the day (matched to PRC or not): for each departure, the
    ADS-B taxi durations (t_to - t_fm, start observed) of the other takeoffs at the same airport within
    +-30 min of its MVT. The row's own event is left out."""
    m = m.copy()
    for c in ('blk_med', 'blk_mean', 'blk_n', 'blk_hold', 'blk_n_to'):
        m[c] = np.nan
    for day, g in m.groupby('day'):
        ev = pd.read_parquet(os.path.join(out, day, 'events.parquet')).reset_index(drop=True)
        ev['taxi'] = ev.t_to - ev.t_fm
        ok = ev.t_fm.notna() & (ev.fm_gap <= 120) & (ev.taxi > 0) & (ev.taxi < 7200)
        for apn, ga in g.groupby('ADEP_mvt'):
            e = ev[ev.apt == apn]
            if e.empty:
                continue
            tt = e.t_to.to_numpy()
            tx = e.taxi.to_numpy()
            okk = ok[e.index].to_numpy()
            hold = e.mid_stop_s.to_numpy()
            eid = e.index.to_numpy()
            for ix, r in zip(ga.index, ga.itertuples()):
                w = np.abs(tt - r.mvt) <= 1800
                own = (eid == r.ev) if r.ev >= 0 else np.zeros(len(eid), bool)
                w_all = w & ~own
                w_ok = w_all & okk
                m.at[ix, 'blk_n_to'] = int(w_all.sum())
                if w_ok.sum():
                    m.at[ix, 'blk_med'] = float(np.median(tx[w_ok]))
                    m.at[ix, 'blk_mean'] = float(np.mean(tx[w_ok]))
                    m.at[ix, 'blk_n'] = int(w_ok.sum())
                    m.at[ix, 'blk_hold'] = float(np.mean(hold[w_ok]))
    m['blk_minus_pred'] = m.blk_med - m.pred
    return m


BLOCK = ['blk_med', 'blk_mean', 'blk_n', 'blk_hold', 'blk_n_to', 'blk_minus_pred']

ADSB = ['est_mvt', 'est_to', 'est_first', 'est_mvt_last', 'dt_to', 'fm_minus_pred', 'first_minus_pred',
        'last_minus_pred', 'fm_minus_sched', 'first_minus_sched', 'ev_first_gs', 'ev_first_dist_m',
        'ev_first_stationary', 'ev_stand_seen', 'ev_stand_dur', 'ev_fm_gap', 'ev_n_mid_stops', 'ev_mid_stop_s',
        'ev_mid_stop_max', 'ev_n_long_stops', 'ev_fm_last_gap', 'ev_last_long_dur', 'ev_to_gap', 'ev_n_pts',
        'ev_max_gap_after_fm', 'ev_share_mlat', 'matched', 'fm_seen', 'fm_ok']
BASE = ['pred', 'mu', 'p_art', 'dsch', 'g', 'apt_code', 'hour']


def lgb_cv(m, feats, seed, rounds=300, lodo=False):
    """Out-of-fold corrected prediction. Folds: random day x airport x 3-hour blocks (5 folds), or with
    lodo=True one fold per day (train on the other days only, as when 2025 days correct 2026 days)."""
    import lightgbm as lgb
    X = m[feats].astype(float).to_numpy()
    tgt = np.clip((m.y - m.pred).to_numpy(), -1800, 1800)
    if lodo:
        days = sorted(m.day.unique())
        fold = m.day.map({d: i for i, d in enumerate(days)}).to_numpy()
        nf = len(days)
    else:
        grp = (m.day + '_' + m.ADEP_mvt + '_' + (m.hour // 3).astype(str)).to_numpy()
        ug = np.unique(grp)
        rng = np.random.default_rng(seed)
        fold_of = dict(zip(ug, rng.integers(0, 5, len(ug))))
        fold = np.array([fold_of[g] for g in grp])
        nf = 5
    oof = np.zeros(len(m))
    params = dict(objective='regression', learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, lambda_l2=10.0, verbose=-1,
                  num_threads=3, seed=seed)
    for k in range(nf):
        tr, te = fold != k, fold == k
        b = lgb.train(params, lgb.Dataset(X[tr], tgt[tr]), num_boost_round=rounds)
        oof[te] = b.predict(X[te])
    return m.pred.to_numpy() + oof


def sse(y, p):
    return float(np.sum((np.asarray(y) - np.asarray(p)) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out')
    ap.add_argument('--label-days', nargs='+', default=['2025-01-15', '2025-07-15'])
    ap.add_argument('--test-days', nargs='+', default=['2026-01-15', '2026-07-15'])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--lodo', action='store_true', help='leave-one-day-out folds instead of random blocks')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    res = {}
    m = add_cols(load(a.out, a.label_days))
    m['apt_code'] = m.ADEP_mvt.map({k: i for i, k in enumerate(APTS)})
    print('label rows', len(m), 'matched', int(m.matched.sum()), 'fm', int(m.fm_seen.sum()))
    # check the mixture identity of the hold-out file
    res['pred_identity_rmse'] = float(np.sqrt(np.mean((m.pred - (m.p_art * m.dsch + (1 - m.p_art) * m.mu)) ** 2)))
    # 1. correlations
    tab = {}
    for name, sel in (('FM', m.fm_seen), ('FM_clean', m.fm_seen & m.clean), ('FM_ok', m.fm_ok),
                      ('FM_ok_clean', m.fm_ok & m.clean), ('M', m.matched),
                      ('M_clean', m.matched & m.clean), ('M_first_moving', m.first_moving),
                      ('M_first_moving_clean', m.first_moving & m.clean)):
        g = m[sel]
        tab[name] = dict(rows=int(len(g)), est_to=st(g.y, g.est_to), est_mvt=st(g.y, g.est_mvt),
                         est_mvt_last=st(g.y, g.est_mvt_last), est_first=st(g.y, g.est_first), pred=st(g.y, g.pred))
    res['corr'] = tab
    print('\n== correlations (y vs estimator) ==')
    for k, v in tab.items():
        print(k, v['rows'])
        for e in ('est_to', 'est_mvt', 'est_mvt_last', 'est_first', 'pred'):
            print('   ', e, v[e])
    per = {}
    for apn in APTS:
        g = m[m.fm_seen & (m.ADEP_mvt == apn)]
        gc = g[g.clean]
        per[apn] = dict(fm=int(len(g)), est_mvt=st(g.y, g.est_mvt), est_to=st(g.y, g.est_to), pred=st(g.y, g.pred),
                        clean_est_mvt=st(gc.y, gc.est_mvt), clean_pred=st(gc.y, gc.pred))
    res['per_airport_fm'] = per
    print('\n== per airport, FM rows: n / r / rmse ==')
    for apn, v in per.items():
        if v['fm']:
            print(f"{apn} n={v['fm']:4d}  est_mvt r={v['est_mvt'].get('r')} rmse={v['est_mvt'].get('rmse')}  "
                  f"est_to r={v['est_to'].get('r')} rmse={v['est_to'].get('rmse')}  pred r={v['pred'].get('r')} "
                  f"rmse={v['pred'].get('rmse')} | clean est_mvt rmse={v['clean_est_mvt'].get('rmse')} "
                  f"pred rmse={v['clean_pred'].get('rmse')}")
    # 2. cross-day linear stack on FM rows
    lin = {}
    fm = m[m.fm_seen].copy()
    for tr_day, te_day in ((a.label_days[0], a.label_days[1]), (a.label_days[1], a.label_days[0])):
        tr = fm[fm.day == tr_day]
        te = fm[fm.day == te_day]
        X = np.c_[np.ones(len(tr)), tr.pred, tr.est_mvt, tr.est_to]
        coef = np.linalg.lstsq(X, tr.y.to_numpy(), rcond=None)[0]
        Xt = np.c_[np.ones(len(te)), te.pred, te.est_mvt, te.est_to]
        ph = Xt @ coef
        lin[te_day] = dict(coef=[round(float(c), 4) for c in coef], n=int(len(te)),
                           rmse_pred=round(float(np.sqrt(np.mean((te.y - te.pred) ** 2))), 1),
                           rmse_stack=round(float(np.sqrt(np.mean((te.y - ph) ** 2))), 1),
                           dSSE=round(sse(te.y, ph) - sse(te.y, te.pred)))
    res['linear_stack_fm'] = lin
    print('\n== cross-day linear stack on FM rows ==', json.dumps(lin))
    # 3. fixed blend rule (prior session's best): blend 50/50 when |est_mvt - pred| < 600
    rule = {}
    for d in a.label_days:
        g = m[m.day == d]
        p2 = g.pred.copy()
        sel = g.fm_seen & ((g.est_mvt - g.pred).abs() < 600)
        p2[sel] = 0.5 * g.pred[sel] + 0.5 * g.est_mvt[sel]
        rule[d] = dict(changed=int(sel.sum()), dSSE=round(sse(g.y, p2) - sse(g.y, g.pred)))
    res['blend_rule'] = rule
    print('\n== blend 50/50 if |est_mvt - pred| < 600 ==', json.dumps(rule))
    # 4. LightGBM residual corrector, grouped CV: base columns only, + per-flight ADS-B, + airport state
    m = block_feats(m, a.out)
    bl = m[m.blk_med.notna()]
    res['block_corr'] = dict(rows=int(len(bl)), resid_vs_blk_minus_pred=st(bl.y - bl.pred, bl.blk_med - bl.pred),
                             y_vs_blk_med=st(bl.y, bl.blk_med))
    print('\n== airport state (other takeoffs +-30 min) ==', json.dumps(res['block_corr']))
    variants = {'base': BASE, 'flight': BASE + ADSB, 'flight_block': BASE + ADSB + BLOCK, 'block': BASE + BLOCK}
    mm = m.copy()
    for c in ('matched', 'fm_seen', 'fm_ok', 'ev_first_stationary', 'ev_stand_seen'):
        mm[c] = mm[c].astype(float)
    out = {}
    N = len(mm)
    yall = mm.y.to_numpy()
    sets = (('ALL', np.ones(N, bool)), ('M', mm.matched.to_numpy() > 0), ('FM', mm.fm_seen.to_numpy() > 0),
            ('unmatched', mm.matched.to_numpy() == 0), ('blk', mm.blk_med.notna().to_numpy()),
            ('clean', mm.clean.to_numpy()), ('M_clean', (mm.matched.to_numpy() > 0) & mm.clean.to_numpy()))
    for seed in range(a.seeds):
        preds = {k: lgb_cv(mm, f, seed, lodo=a.lodo) for k, f in variants.items()}
        r = {}
        for name, sel in sets:
            yv = yall[sel]
            row = dict(n=int(sel.sum()), rmse_pred=round(float(np.sqrt(np.mean((yv - mm.pred.to_numpy()[sel]) ** 2))), 1))
            for k, p in preds.items():
                row[f'rmse_{k}'] = round(float(np.sqrt(np.mean((yv - p[sel]) ** 2))), 1)
            for k in ('flight', 'flight_block', 'block'):
                row[f'dSSE_{k}_vs_base'] = round(sse(yv, preds[k][sel]) - sse(yv, preds['base'][sel]))
            r[name] = row
        per_ap = {}
        for apn in APTS:
            sel = (mm.ADEP_mvt == apn).to_numpy()
            if sel.sum():
                yv = yall[sel]
                per_ap[apn] = dict(n=int(sel.sum()), matched=int((mm.matched.to_numpy() > 0)[sel].sum()),
                                   **{f'dSSE_{k}_vs_base': round(sse(yv, preds[k][sel]) - sse(yv, preds['base'][sel]))
                                      for k in ('flight', 'flight_block', 'block')})
        r['per_airport'] = per_ap
        out[f'seed{seed}'] = r
        print(f'\n== LightGBM residual corrector seed {seed} ==')
        print(pd.DataFrame({k: v for k, v in r.items() if k != 'per_airport'}).T.to_string())
        print(pd.DataFrame(per_ap).T.to_string())
    res['lgb'] = out
    # which columns carry the corrector (gain importance, one fit on all label rows)
    import lightgbm as lgb
    fx = variants['flight']
    b = lgb.train(dict(objective='regression', learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
                       feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, lambda_l2=10.0, verbose=-1,
                       num_threads=3, seed=0),
                  lgb.Dataset(mm[fx].astype(float).to_numpy(), np.clip((mm.y - mm.pred).to_numpy(), -1800, 1800)),
                  num_boost_round=300)
    imp = sorted(zip(fx, b.feature_importance('gain')), key=lambda z: -z[1])
    tot = sum(v for _, v in imp) or 1.0
    res['importance_flight'] = {k: round(float(v / tot), 4) for k, v in imp}
    print('\n== gain importance (flight corrector) ==', json.dumps(dict(list(res['importance_flight'].items())[:15])))
    # extrapolation: gain per matched row by airport (2025 pilot days) carried to the 2026 match counts
    t = load(a.out, a.test_days)
    ext = {}
    mt_ap = {apn: int(((mm.ADEP_mvt == apn) & (mm.matched > 0)).sum()) for apn in APTS}
    for k in ('flight', 'flight_block', 'block'):
        key = f'dSSE_{k}_vs_base'
        g_ap = {apn: float(np.mean([out[s]['per_airport'].get(apn, {}).get(key, 0) for s in out])) for apn in APTS}
        u_ap = {apn: (g_ap[apn] / mt_ap[apn]) if mt_ap[apn] >= 30 else np.nan for apn in APTS}
        u_glob = sum(g_ap.values()) / max(sum(mt_ap.values()), 1)
        d25 = float(np.mean([out[s]['ALL'][key] for s in out])) / N
        e = dict(dMSE_2025_pilot=round(d25, 1),
                 server_if_2025_gain=round(float(np.sqrt(max(SERVER_BEST ** 2 + d25, 0))), 2),
                 gain_per_matched_row_2025={kk: (round(v, 1) if not np.isnan(v) else None) for kk, v in u_ap.items()},
                 gain_per_matched_row_pooled=round(u_glob, 1))
        if len(t):
            n26 = len(t)
            m26 = {apn: int(((t.ADEP_mvt == apn) & t.matched).sum()) for apn in APTS}
            cons = sum((0.0 if np.isnan(u_ap[apn]) else u_ap[apn]) * m26[apn] for apn in APTS) / n26
            pooled = u_glob * sum(m26.values()) / n26
            e.update(test_rows=n26, test_matched=m26, dMSE_test_by_airport=round(cons, 1),
                     dMSE_test_pooled=round(pooled, 1),
                     server_if_by_airport=round(float(np.sqrt(max(SERVER_BEST ** 2 + cons, 0))), 2),
                     server_if_pooled=round(float(np.sqrt(max(SERVER_BEST ** 2 + pooled, 0))), 2))
        ext[k] = e
    res['extrapolation'] = ext
    print('\n== extrapolation ==', json.dumps(ext, indent=1))
    res['cv'] = 'leave-one-day-out' if a.lodo else 'random day x airport x 3h blocks, 5 folds'
    json.dump(res, open(os.path.join(a.out, f'lever{a.tag}.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
