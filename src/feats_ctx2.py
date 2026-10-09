# SPDX-License-Identifier: GPL-3.0-only
"""Context columns v2 for the all-row corrector A2 (src/ctx_corr2.py): every column of cache/<base>.parquet (src/feats_ctx.py, sha-checked,
unchanged) plus nine input-only columns. A departure's taxi-out (the label) and a departure's BLOCK time are never read.

Two state columns from the R4 lane's test on 316's residual (research/r4_sources/state_capture.py; vault research/2026-10-05-r4-sources.md 5.5),
same definitions, departures of 2025-01, 2025-07 and the 2026 test file:
  gate_delay_pm30    mean of clip(AOBT_3 - EOBT_1, -3600, 7200) over the airport's departures with MVT in [t - 1800, t + 1800] (self included,
                     departures without both times left out)
  plan_ground_aobt3  NM-matched departures only (FLIGHT_ID present): NM-matched departures of the same airport and UTC day of MVT with
                     EOBT_1 <= my AOBT_3 < their MVT (self included); NaN on NM-unmatched rows
Seven stand-occupancy columns from the R3 lane (research/r3_structure/s2_stand.py stand_events; vault research/2026-10-05-r3-structure.md 2.5),
same definitions over every month in data/ (windows may cross a month edge, as there), per (airport, STAND_mvt), t = MVT of the departure:
  st_gp, st_gp2       t - take-off of the previous / second previous departure from that stand
  st_g1, st_g2, st_g3 t - in-block of the latest, second and third latest arrival on that stand before t
  st_k                arrivals on that stand with in-block in [previous departure's take-off, t)
  st_n_arr_2h         arrivals on that stand with in-block in [t - 7200, t)
When research/r3_structure/s2_stand_{train,test}.parquet exist, the seven are compared with them row by row and the result goes in the json.
Output: cache/<name>.parquet (the rows of the base file) and cache/<name>.json (sha16, fill share per month, the comparison).
    .venv/Scripts/python.exe -X utf8 src/feats_ctx2.py --name ctx_v2 --base ctx_v1 --base_sha 0383b7a4b36c5821
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
R4 = ["gate_delay_pm30", "plan_ground_aobt3"]
ST = ["st_g1", "st_g2", "st_g3", "st_gp", "st_gp2", "st_k", "st_n_arr_2h"]
NEW = R4 + ST


def below_normal() -> None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def epoch_s(col: pa.ChunkedArray) -> np.ndarray:
    unit = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[col.type.unit]
    return np.asarray(col.cast(pa.int64()).to_numpy(zero_copy_only=False), dtype="float64") / unit


def files() -> list[str]:
    return sorted(glob.glob(str(ROOT / "data" / "training_*.parquet"))) + [str(ROOT / "data" / "ranking.parquet")]


class Vocab:
    """Strings to stable integer ids across files (missing -> -1)."""

    def __init__(self):
        self.ids: dict[str, int] = {}

    def codes(self, col: pa.ChunkedArray) -> np.ndarray:
        enc = col.combine_chunks().dictionary_encode()
        dic = [self.ids.setdefault(s, len(self.ids)) for s in enc.dictionary.to_pylist()]
        idx = enc.indices.to_numpy(zero_copy_only=False)
        out = np.full(len(idx), -1, dtype=np.int64)
        ok = enc.indices.is_valid().to_numpy(zero_copy_only=False)
        out[ok] = np.asarray(dic, dtype=np.int64)[idx[ok].astype(np.int64)]
        return out


# ---- R4 state columns ----

def r4_columns(want_months: set[int]) -> pd.DataFrame:
    parts = []
    for f in files():
        name = Path(f).name
        if name.startswith("training_") and int(name[14:16]) not in want_months:
            continue
        t = pq.read_table(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "FLIGHT_ID_mvt"])
        dep = np.asarray(t.column("PHASE_mvt").to_pandas() == "DEP")
        d = pd.DataFrame({"MVT_ID_mvt": np.asarray(t.column("MVT_ID_mvt").to_numpy(zero_copy_only=False), dtype="float64"),
                          "ap": t.column("ADEP_mvt").to_pandas().astype(str).to_numpy(),
                          "t": epoch_s(t.column("MVT_TIME_UTC_mvt")), "ao": epoch_s(t.column("AOBT_3_flt")),
                          "eo": epoch_s(t.column("EOBT_1_flt")),
                          "nm": t.column("FLIGHT_ID_mvt").is_valid().to_numpy(zero_copy_only=False)})[dep]
        d["part"] = name
        parts.append(d)
    D = pd.concat(parts, ignore_index=True)
    gate = np.full(len(D), np.nan)
    plan = np.full(len(D), np.nan)
    for (_, _), g in D.groupby(["part", "ap"]):
        g = g.sort_values("t", kind="stable")
        ix = g.index.to_numpy()
        tt = g["t"].to_numpy()
        gd = g["ao"].to_numpy() - g["eo"].to_numpy()
        ok = np.isfinite(gd)
        cs = np.concatenate([[0.0], np.cumsum(np.where(ok, np.clip(gd, -3600, 7200), 0.0))])
        cn = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
        lo, hi = np.searchsorted(tt, tt - 1800, "left"), np.searchsorted(tt, tt + 1800, "right")
        n = cn[hi] - cn[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            gate[ix] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    M = D[D["nm"]].copy()
    M["day"] = np.floor(M["t"] / 86400).astype(np.int64)
    for (_, _, _), g in M.groupby(["part", "ap", "day"]):
        A, T, E = g["ao"].to_numpy(), g["t"].to_numpy(), g["eo"].to_numpy()
        cnt = np.empty(len(g))
        for s in range(0, len(g), 2048):
            a = A[s:s + 2048, None]
            cnt[s:s + 2048] = (np.isfinite(E)[None, :] & (E[None, :] <= a) & (T[None, :] > a)).sum(1)
        plan[g.index.to_numpy()] = cnt
    return pd.DataFrame({"MVT_ID_mvt": D["MVT_ID_mvt"].to_numpy(), "gate_delay_pm30": gate, "plan_ground_aobt3": plan})


# ---- R3 stand columns ----

def stand_columns() -> pd.DataFrame:
    ap_v, st_v = Vocab(), Vocab()
    dep_parts, arr_parts = [], []
    for f in files():
        t = pq.read_table(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"])
        dep = np.asarray(t.column("PHASE_mvt").to_pandas() == "DEP")
        stand = st_v.codes(t.column("STAND_mvt"))
        a_dep, a_arr = ap_v.codes(t.column("ADEP_mvt")), ap_v.codes(t.column("ADES_mvt"))
        ids = np.asarray(t.column("MVT_ID_mvt").to_numpy(zero_copy_only=False), dtype="float64")
        tm, tb = epoch_s(t.column("MVT_TIME_UTC_mvt")), epoch_s(t.column("BLOCK_TIME_UTC_mvt"))
        keep_d = dep & (stand >= 0)
        dep_parts.append(pd.DataFrame({"MVT_ID_mvt": ids[keep_d], "key": a_dep[keep_d] * 1_000_000 + stand[keep_d], "t": tm[keep_d]}))
        keep_a = ~dep & (stand >= 0) & np.isfinite(tb)
        arr_parts.append(pd.DataFrame({"key": a_arr[keep_a] * 1_000_000 + stand[keep_a], "tin": tb[keep_a]}))
        del t
    D = pd.concat(dep_parts, ignore_index=True)
    A = pd.concat(arr_parts, ignore_index=True)
    out = {k: np.full(len(D), np.nan) for k in ST}
    a_ord = np.lexsort((A["tin"].to_numpy(), A["key"].to_numpy()))
    ak, ab = A["key"].to_numpy()[a_ord], A["tin"].to_numpy()[a_ord]
    uk, a0 = np.unique(ak, return_index=True)
    amap = dict(zip(uk.tolist(), zip(a0.tolist(), np.append(a0[1:], len(ak)).tolist())))
    dk, tm = D["key"].to_numpy(), D["t"].to_numpy()
    d_ord = np.lexsort((tm, dk))
    ud, d0 = np.unique(dk[d_ord], return_index=True)
    for k, s, e in zip(ud.tolist(), d0.tolist(), np.append(d0[1:], len(d_ord)).tolist()):
        rows = d_ord[s:e]
        tt = tm[rows]
        prev = np.concatenate([[np.nan], tt[:-1]])
        prev2 = np.concatenate([[np.nan, np.nan], tt[:-2]])[: len(tt)]
        out["st_gp"][rows] = tt - prev
        out["st_gp2"][rows] = tt - prev2
        if k not in amap:
            out["st_k"][rows] = 0
            out["st_n_arr_2h"][rows] = 0
            continue
        i0, i1 = amap[k]
        ib = ab[i0:i1]
        j = np.searchsorted(ib, tt, side="left")
        out["st_g1"][rows] = np.where(j >= 1, tt - ib[np.maximum(j - 1, 0)], np.nan)
        out["st_g2"][rows] = np.where(j >= 2, tt - ib[np.maximum(j - 2, 0)], np.nan)
        out["st_g3"][rows] = np.where(j >= 3, tt - ib[np.maximum(j - 3, 0)], np.nan)
        jp = np.searchsorted(ib, np.where(np.isfinite(prev), prev, -np.inf), side="left")
        out["st_k"][rows] = np.where(np.isfinite(prev), j - jp, np.nan)
        out["st_n_arr_2h"][rows] = j - np.searchsorted(ib, tt - 7200, side="left")
    S = pd.DataFrame(out)
    S.insert(0, "MVT_ID_mvt", D["MVT_ID_mvt"].to_numpy())
    return S


def compare_r3(S: pd.DataFrame) -> dict:
    """Row-by-row comparison with the R3 lane's files (same definitions): max abs difference and NaN agreement per column."""
    R3 = ROOT / "research" / "r3_structure"
    fs = [R3 / "s2_stand_train.parquet", R3 / "s2_stand_test.parquet"]
    if not all(f.exists() for f in fs):
        return {"status": "R3 files not present"}
    names = {"st_g1": "g1", "st_g2": "g2", "st_g3": "g3", "st_gp": "gp", "st_gp2": "gp2", "st_k": "k_since_prevdep", "st_n_arr_2h": "n_arr_2h"}
    R = pd.concat([pd.read_parquet(f, columns=["MVT_ID_mvt"] + list(names.values())) for f in fs], ignore_index=True).set_index("MVT_ID_mvt")
    mine = S.set_index("MVT_ID_mvt").reindex(R.index)
    res = {"rows": int(len(R))}
    for m, r in names.items():
        a, b = mine[m].to_numpy(float), R[r].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        res[m] = {"nan_mismatch": int((np.isfinite(a) != np.isfinite(b)).sum()), "max_abs_diff": float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0}
    return res


def main():
    pa_ = argparse.ArgumentParser()
    pa_.add_argument("--name", default="ctx_v2")
    pa_.add_argument("--base", default="ctx_v1")
    pa_.add_argument("--base_sha", required=True)
    a = pa_.parse_args()
    below_normal()
    bp = ROOT / "cache" / f"{a.base}.parquet"
    if sha16(bp) != a.base_sha:
        raise SystemExit(f"{bp.name} sha {sha16(bp)} != {a.base_sha}")
    B = pd.read_parquet(bp)
    months = sorted({int(p[5:7]) for p in B["part"].astype(str).unique() if p.startswith("2025-")})
    r4 = r4_columns(set(months)).set_index("MVT_ID_mvt")
    print("R4 columns", len(r4), flush=True)
    S = stand_columns()
    print("stand columns", len(S), flush=True)
    cmp = compare_r3(S)
    print("vs R3 files:", json.dumps(cmp), flush=True)
    S = S.set_index("MVT_ID_mvt")
    ids = B["MVT_ID_mvt"].to_numpy()
    for c in R4:
        B[c] = r4[c].reindex(ids).to_numpy()
    for c in ST:
        B[c] = S[c].reindex(ids).to_numpy()
    assert B["MVT_ID_mvt"].is_unique and r4.index.is_unique and S.index.is_unique
    p = ROOT / "cache" / f"{a.name}.parquet"
    B.to_parquet(p, index=False)
    fill = {part: {c: round(float(g[c].notna().mean()), 4) for c in NEW} for part, g in B.groupby("part")}
    meta = {"name": a.name, "file": f"cache/{a.name}.parquet", "sha16": sha16(p), "rows": int(len(B)), "base": a.base, "base_sha16": a.base_sha,
            "new_columns": NEW, "fill": fill, "check_vs_r3_stand_files": cmp, "code": "src/feats_ctx2.py", "code_sha16": sha16(Path(__file__))}
    (ROOT / "cache" / f"{a.name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("name", "sha16", "rows")}, ensure_ascii=False))
    print(pd.DataFrame(fill).round(3).to_string())


if __name__ == "__main__":
    main()
