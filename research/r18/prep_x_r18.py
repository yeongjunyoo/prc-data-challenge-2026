# SPDX-License-Identifier: GPL-3.0-only
"""R18 copy of research/r13_l/prep_x.py with one more setting, fj (F2 M3 ruler: fit on 2025 minus February and June, evaluate February
and June 2025), written under research/r18/x/<setting>. Everything else is the original:

R13-L: the leg input of src/rmlp_mu.py, built once per setting and stored as memory-mappable arrays, so a new learner family can train
from them with little host memory (PC RAM is tight) and every candidate and seed sees the same rows, features and targets.

Same as src/rmlp_mu.py (feats_v27, add_te on the setting's fit rows, feature_cols, CAT_FEATS, regression rows = fit rows minus art and
shift, early-stopping days = es_day_mask, anchor "win" and none, medians over the training rows (regression rows minus es days), a missing
flag for every column with a missing value). Numeric columns (non CAT_FEATS) are float32 filled with the median; CAT_FEATS are integer codes
with missing as one extra category (rmlp_mu does the same).
Output research/r13_l/x/<setting>/:
  X.npy   (n_num, N) float32, column-major (one row per numeric feature, all N feature-table rows), missing filled with the median
  NA.npy  (n_na, N) uint8 missing flags of the na_cols
  C.npy   (n_cat, N) int32 category codes, missing = cardinality - 1
  rows.npz  ids (float64), y, dsch, anchor_win, ap (int8 code), masks fit, eval, es, reg (bool), tr_idx, va_idx, ev_idx (int64)
  meta.json columns, na_cols, cats, cardinalities, medians, counts
    .venv/Scripts/python.exe -X utf8 research/r13_l/prep_x.py --setting val
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common import feature_cols  # noqa: E402
from src.feats import CAT_FEATS, build  # noqa: E402
from src.model3 import es_day_mask  # noqa: E402
from src.te import add_te  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if sys.platform == "win32":  # BelowNormal (r10_common compute rule)
    import ctypes
    ctypes.WinDLL("kernel32").SetPriorityClass(ctypes.WinDLL("kernel32").GetCurrentProcess(), 0x00004000)
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", required=True, choices=["val", "fwd", "fj", "full"])
    pa.add_argument("--feats", default="feats_v27")
    a = pa.parse_args()
    out = ROOT / "research" / "r18" / "x" / a.setting
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists():
        raise SystemExit(f"exists: {out / 'meta.json'}")
    F = build(name=a.feats)
    log(f"features table {F.shape}")
    src = F["src"].to_numpy()
    te = src == 0
    if a.setting == "val":
        fit_rows, eval_rows = ~np.isin(src, [1, 7]) & ~te, np.isin(src, [1, 7])
    elif a.setting == "fwd":
        fit_rows, eval_rows = np.isin(src, [1, 2, 3, 4, 5, 6]), src == 7
    elif a.setting == "fj":  # F2 M3 (research/f2/model3_fj.py): 2025 without February and June, scored on February and June 2025
        fit_rows, eval_rows = ~np.isin(src, [2, 6]) & ~te, np.isin(src, [2, 6])
    else:
        fit_rows, eval_rows = ~te, te
    ids = F["MVT_ID_mvt"].to_numpy()
    es_rows = fit_rows & es_day_mask(ids)
    T = add_te(F, fit_rows).astype(np.float32)
    F = pd.concat([F, T], axis=1)
    del T
    gc.collect()
    cols = feature_cols(F, None)
    cats = [c for c in cols if c in CAT_FEATS and isinstance(F[c].dtype, pd.CategoricalDtype)]
    nums = [c for c in cols if c not in cats]
    y_all = F["y"].to_numpy().astype(float)
    dsch_all = F["d_sch"].to_numpy().astype(float)
    art_all = np.abs(y_all - dsch_all) <= 60
    shift_all = (y_all > 60000) & ~art_all
    reg_all = fit_rows & ~art_all & ~shift_all
    dl = F["d_lobt"].to_numpy().astype(float)
    apv = F["ap"].astype(str).to_numpy()
    nfit = fit_rows & ~art_all & ~shift_all & (y_all < 3600)
    amed = pd.Series(y_all[nfit]).groupby(apv[nfit]).median()
    fb = pd.Series(apv).map(amed).fillna(float(np.median(y_all[nfit]))).to_numpy()
    anchor_win = np.where(np.isnan(dl), fb, np.clip(fb, dl - 3606.0, dl + 3606.0))
    tr = reg_all & ~es_rows
    va = reg_all & es_rows
    N = len(F)
    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float32, shape=(len(nums), N))
    na_cols, na_bits, med = [], [], {}
    for j, c in enumerate(nums):
        s = F[c]
        v = (np.where(s.cat.codes.to_numpy() < 0, np.nan, s.cat.codes.to_numpy()).astype(np.float32) if isinstance(s.dtype, pd.CategoricalDtype)
             else s.to_numpy(dtype=np.float32, na_value=np.nan))
        nan = np.isnan(v)
        if nan.any():
            na_cols.append(c)
            na_bits.append(np.packbits(nan))
        m_ = np.nanmedian(v[tr]) if np.isfinite(v[tr]).any() else 0.0
        med[c] = float(m_) if np.isfinite(m_) else 0.0
        X[j] = np.where(nan, med[c], v)
    X.flush()
    del X
    NA = np.lib.format.open_memmap(out / "NA.npy", mode="w+", dtype=np.uint8, shape=(len(na_cols), N))
    for j, b in enumerate(na_bits):
        NA[j] = np.unpackbits(b)[:N]
    NA.flush()
    del NA, na_bits
    C = np.lib.format.open_memmap(out / "C.npy", mode="w+", dtype=np.int32, shape=(len(cats), N))
    card = {}
    for j, c in enumerate(cats):
        v = F[c].cat.codes.to_numpy()
        card[c] = int(len(F[c].cat.categories) + 1)
        C[j] = np.where(v < 0, card[c] - 1, v).astype(np.int32)
    C.flush()
    del C
    ap_code = F["ap"].cat.codes.to_numpy().astype(np.int8)
    np.savez(out / "rows.npz", ids=ids, y=y_all, dsch=dsch_all, anchor_win=anchor_win, ap=ap_code, fit=fit_rows, eval=eval_rows, es=es_rows,
             reg=reg_all, tr_idx=np.where(tr)[0], va_idx=np.where(va)[0], ev_idx=np.where(eval_rows)[0])
    meta = {"setting": a.setting, "feats": a.feats, "N": int(N), "nums": nums, "na_cols": na_cols, "cats": cats, "card": card, "med": med,
            "ap_categories": [str(x) for x in F["ap"].cat.categories], "n_tr": int(tr.sum()), "n_va": int(va.sum()), "n_ev": int(eval_rows.sum()),
            "n_reg": int(reg_all.sum()), "n_features_rmlp_equiv": len(nums) + len(na_cols) + len(cats)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{a.setting}: N {N}, numeric {len(nums)}, na flags {len(na_cols)}, cats {len(cats)} {card}, train {meta['n_tr']} es {meta['n_va']} "
        f"eval {meta['n_ev']} (rmlp-equivalent features {meta['n_features_rmlp_equiv']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
