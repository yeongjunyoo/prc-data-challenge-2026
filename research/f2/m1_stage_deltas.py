# SPDX-License-Identifier: GPL-3.0-only
"""F2 ruler M1: per-stage val (in-month) and fwd (Jan-fitted, applied to Jul) delta MSE along 514's three chains, read from manifests."""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT = 'preds/manifest_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgjaavskap.json'
def load(p):
    return json.load(open(p, encoding='utf-8'))
def find_report(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            r = v.get('report') if 'report' in v else None
            if isinstance(r, dict) and ('val' in r or 'fwd' in r):
                out[k] = r
    return out
def chain(path):
    rows = []
    while path and os.path.exists(path):
        d = load(path)
        reps = find_report(d)
        rec = {'spec': d['spec'], 'holdout_rmse': d.get('holdout', {}).get('rmse'), 'fwd_rmse': d.get('forward', {}).get('rmse_2025_07')}
        for k, r in reps.items():
            for st in ('val', 'fwd', 'full'):
                if isinstance(r.get(st), dict):
                    x = r[st]
                    rec[f'{k}.{st}'] = {kk: x.get(kk) for kk in ('dmse', 'changed', 'improved', 'rows', 'move_mse', 'move_mse_test_n', 'rows_moved') if kk in x}
        rows.append(rec)
        src = d.get('source')
        path = src.get('manifest') if isinstance(src, dict) else None
    return rows
r = load(ROOT)
for lab in ('source', 'source_b', 'source_c'):
    print('=====', lab)
    rows = chain(r[lab]['manifest'])
    prev = None
    for rec in reversed(rows):
        sp = rec['spec']
        tag = sp[len(prev):] if prev and sp.startswith(prev) else sp
        hv, fw = rec['holdout_rmse'], rec['fwd_rmse']
        extra = {k: v for k, v in rec.items() if '.' in k}
        print(f"{tag:>28}  val_rmse {hv}  fwd_rmse {fw}  " + json.dumps(extra, ensure_ascii=False)[:600])
        prev = sp
