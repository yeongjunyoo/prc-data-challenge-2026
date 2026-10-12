# SPDX-License-Identifier: GPL-3.0-only
"""R21 R2a: art-aware refinement of antiwin R2 on LIRF NM-unmatched departures. Deterministic, inputs only, no fitting at apply time.

Why (2025 facts, vault research/2026-10-11-r21-extreme-rows.md): an NM-unmatched LIRF departure whose arrival record at a ten-airport destination
carries the NM LOBT, with 0 < d_lobt = MVT - LOBT_arr <= 3606, cannot be a normal record with a normal taxi (it would have matched NM:
|BLOCK - LOBT| <= 3606 holds on all 2,062,577 matched 2025 departures). antiwin R2 therefore floors the prediction at d_lobt + 3606 + m.
Where art (BLOCK = SCHED) is possible, i.e. |SCHED - LOBT_arr| > 3606, 21 of the 23 such 2025 rows are art (y = MVT - SCHED);
the floor alone leaves them short. R2a: new = PI * d_sch + (1 - PI) * current, on those rows only.

Pinned: PI = (21 + 0.5) / (23 + 1) = 0.8958 (2025, all twelve months, LIRF antiwin-R2 rows with |SCHED - LOBT_arr| > 3606; out-of-month
values 0.886 to 0.929). LINK = research/r21/d3_link4m.parquet (the tools/antiwin.build_link definition re-run read-only on
data/final_phase/final_ranking.parquet; its Jan/Jul rows equal cache/antiwin_link_v1.parquet's 221 test rows), sha16 pinned below.
Window 3606 s is the 2025 matching property, not a tuned value.

    .venv/Scripts/python.exe -X utf8 research/r21/r2a_tool.py apply <final-form parquet: MVT_ID_mvt, pred> <out parquet>
        writes <out parquet> (same columns and row order) and <out parquet>.json (input and output sha256, changed rows, constants, code sha16)
    .venv/Scripts/python.exe -X utf8 research/r21/r2a_tool.py selftest
        1 PI from 2025 labels equals the pinned value; 2 the pinned link table equals a fresh rebuild from final_ranking.parquet;
        3 gate numbers reproduced from the tool's own outputs on the rulers (val, fwd, fj, month-held-out PI);
        4 apply on the staged final 598c twice gives the same bytes and changes only the R2a rows.
Writes only under research/r21/.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
PI = 21.5 / 24.0
WIN = 3606.0
LINK = HERE / "d3_link4m.parquet"
LINK_SHA16 = "38c129a4bb1c18dc"
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
FINAL_598C = ROOT / "research/f3/queue_ready_final/598c_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3rBbFamfhgkcxncrdbawgpglgjaavskapff9.parquet"


def sha(p: Path, n: int = 64) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:n]


def now_kst() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")


def r2a_rows(link: pd.DataFrame) -> pd.DataFrame:
    """LIRF antiwin-R2 rows where art is possible. link needs ADEP_mvt, d_lobt, sch_m_lobt, d_sch (and MVT_ID_mvt)."""
    m = (link["ADEP_mvt"].astype(str) == "LIRF") & (link["d_lobt"] > 0) & (link["d_lobt"] <= WIN) & (link["sch_m_lobt"].abs() > WIN)
    return link[m]


def apply_r2a(pred: pd.Series, rows: pd.DataFrame, pi: float | pd.Series) -> pd.Series:
    """pred indexed by MVT_ID_mvt; rows indexed by MVT_ID_mvt with d_sch. Returns the new series (same index and order)."""
    ids = rows.index.intersection(pred.index)
    p = pi.reindex(ids).to_numpy(float) if isinstance(pi, pd.Series) else float(pi)
    out = pred.copy()
    out.loc[ids] = p * rows.loc[ids, "d_sch"].to_numpy(float) + (1 - p) * pred.loc[ids].to_numpy(float)
    return out


def load_link() -> pd.DataFrame:
    got = sha(LINK, 16)
    if got != LINK_SHA16:
        raise SystemExit(f"link table sha16 {got} != pinned {LINK_SHA16}")
    return pd.read_parquet(LINK)


def final_inputs() -> pd.DataFrame:
    R = pq.read_table(ROOT / "data/final_phase/final_ranking.parquet",
                      columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]).to_pandas()
    D = R[R["PHASE_mvt"] == "DEP"].copy()
    D["d_sch"] = (D["MVT_TIME_UTC_mvt"].astype("int64") - D["SCHED_TIME_UTC_mvt"].astype("int64")) / 1e6
    return D.set_index("MVT_ID_mvt")


def cmd_apply(src: str, dst: str) -> dict:
    src_p, dst_p = Path(src), Path(dst)
    if HERE not in dst_p.resolve().parents:
        raise SystemExit("output must be under research/r21/")
    F = pd.read_parquet(src_p)
    if list(F.columns) != ["MVT_ID_mvt", "pred"]:
        raise SystemExit(f"unexpected columns {list(F.columns)}")
    D = final_inputs()
    if len(F) != len(D) or not np.array_equal(F["MVT_ID_mvt"].to_numpy(), D.index.to_numpy()):
        raise SystemExit("input rows are not the final_ranking DEP rows in order")
    L = load_link().set_index("MVT_ID_mvt")
    rows = r2a_rows(L)
    chk = np.abs(rows["d_sch"].to_numpy() - D.loc[rows.index, "d_sch"].to_numpy())
    if chk.max() > 0.5:
        raise SystemExit(f"link d_sch disagrees with final_ranking by {chk.max()} s")
    if (D.loc[rows.index, "FLIGHT_ID_mvt"].notna()).any():
        raise SystemExit("an R2a row has a FLIGHT_ID (must be NM-unmatched)")
    pred = F.set_index("MVT_ID_mvt")["pred"].astype(float)
    new = apply_r2a(pred, rows, PI)
    out = pd.DataFrame({"MVT_ID_mvt": F["MVT_ID_mvt"].to_numpy(), "pred": new.reindex(F["MVT_ID_mvt"]).to_numpy()})
    assert np.isfinite(out["pred"]).all() and (out["pred"] >= 0).all()
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst_p, index=False)
    ch = out["pred"].to_numpy() != F["pred"].to_numpy()
    month = pd.to_datetime(D.loc[rows.index, "MVT_TIME_UTC_mvt"]).dt.month
    lst = [{"MVT_ID_mvt": float(i), "month": int(month.loc[i]), "d_sch": float(rows.loc[i, "d_sch"]), "d_lobt": float(rows.loc[i, "d_lobt"]),
            "sch_m_lobt": float(rows.loc[i, "sch_m_lobt"]), "old": float(pred.loc[i]), "new": float(new.loc[i])} for i in rows.index]
    info = {"tool": "research/r21/r2a_tool.py", "tool_sha16": sha(Path(__file__), 16), "written_kst": now_kst(),
            "input": str(src_p), "input_sha256": sha(src_p), "output": str(dst_p), "output_sha256": sha(dst_p),
            "pi": PI, "window_s": WIN, "link": str(LINK.relative_to(ROOT)), "link_sha16": LINK_SHA16,
            "changed_rows": int(ch.sum()), "rows": lst,
            "move_mse_4month": float(((out["pred"] - F["pred"]) ** 2).sum() / len(F))}
    dst_p.with_suffix(dst_p.suffix + ".json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"apply: {src_p.name} -> {dst_p.name}, changed {int(ch.sum())} rows, move MSE (4-month N) {info['move_mse_4month']:.2f}, out sha256 {info['output_sha256'][:16]}")
    return info


def rebuild_link() -> pd.DataFrame:
    """tools/antiwin.build_link definition on final_ranking.parquet (read-only), same as research/r21/d3_link4m.py."""
    R = pq.read_table(ROOT / "data/final_phase/final_ranking.parquet",
                      columns=["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
                               "SCHED_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt"]).to_pandas()
    for c, s in (("MVT_TIME_UTC_mvt", "t_mvt"), ("SCHED_TIME_UTC_mvt", "t_sch"), ("LOBT_flt", "t_lobt"), ("AOBT_3_flt", "t_aobt3")):
        v = R[c]
        R[s] = np.where(v.notna(), v.astype("int64") / 1e6, np.nan)
    ok = R["FLIGHT_mvt"].notna()
    dep = R[(R["PHASE_mvt"] == "DEP") & R["ADES_mvt"].isin(APTS) & ok].copy()
    arr = R[(R["PHASE_mvt"] == "ARR") & R["ADEP_mvt"].isin(APTS) & ok].copy()
    for d in (dep, arr):
        d["k"] = d["ADEP_mvt"].astype(str) + "|" + d["ADES_mvt"].astype(str) + "|" + d["FLIGHT_mvt"].astype(str)
    arr = arr.sort_values(["k", "t_mvt"]).reset_index(drop=True)
    a_k, a_t = arr["k"].to_numpy(), arr["t_mvt"].to_numpy()
    uk, st = np.unique(a_k, return_index=True)
    en = np.append(st[1:], len(a_k))
    amap = {k: (s, e) for k, s, e in zip(uk, st, en)}
    dep = dep[dep["FLIGHT_ID_mvt"].isna()].reset_index(drop=True)
    link = np.full(len(dep), -1, np.int64)
    for i, (k, t) in enumerate(zip(dep["k"].to_numpy(), dep["t_mvt"].to_numpy())):
        r = amap.get(k)
        if r is None:
            continue
        s, e = r
        j = s + np.searchsorted(a_t[s:e], t + 300, side="left")
        if j < e and a_t[j] - t <= 8 * 3600:
            link[i] = j
    okl = link >= 0
    A = arr.iloc[link[okl]].reset_index(drop=True)
    L = dep.loc[okl, ["MVT_ID_mvt", "ADEP_mvt", "t_mvt", "t_sch"]].reset_index(drop=True)
    L["a_nm"] = A["FLIGHT_ID_mvt"].notna().to_numpy()
    L["a_t_lobt"] = A["t_lobt"].to_numpy()
    L["a_t_aobt3"] = A["t_aobt3"].to_numpy()
    L = L[L["a_nm"]].reset_index(drop=True)
    L["d_lobt"] = L["t_mvt"] - L["a_t_lobt"]
    L["d_aobt"] = L["t_mvt"] - L["a_t_aobt3"]
    L["d_sch"] = L["t_mvt"] - L["t_sch"]
    L["sch_m_lobt"] = L["t_sch"] - L["a_t_lobt"]
    return L


def selftest() -> int:
    bad = []
    out_dir = HERE / "r2a_selftest"
    out_dir.mkdir(exist_ok=True)
    # 1 PI from 2025 labels
    L25 = pd.read_parquet(ROOT / "cache/antiwin_link_v1.parquet")
    R25 = r2a_rows(L25[L25["src"] > 0]).copy()
    art = (np.abs(R25["y"] - R25["d_sch"]) <= 60)
    pi_full = (art.sum() + 0.5) / (len(R25) + 1.0)
    print(f"1 PI from 2025 labels: {art.sum()} art of {len(R25)} rows -> {pi_full:.6f} (pinned {PI:.6f})")
    if abs(pi_full - PI) > 1e-12:
        bad.append("PI")
    # 2 link table equals a fresh rebuild
    L4 = load_link()
    Lb = rebuild_link()
    cols = ["MVT_ID_mvt", "ADEP_mvt", "t_mvt", "t_sch", "a_t_lobt", "a_t_aobt3", "d_lobt", "d_aobt", "d_sch", "sch_m_lobt"]
    same = len(L4) == len(Lb) and all(np.allclose(L4[c].to_numpy(float), Lb[c].to_numpy(float), equal_nan=True) if c != "ADEP_mvt"
                                      else np.array_equal(L4[c].astype(str).to_numpy(), Lb[c].astype(str).to_numpy()) for c in cols)
    old = L25[L25["src"] == 0]
    jj = L4[pd.to_datetime(L4["t_mvt"], unit="s").dt.month.isin([1, 7])]
    print(f"2 link: pinned rows {len(L4)}, rebuild rows {len(Lb)}, identical {same}; Jan/Jul ids equal to cache/antiwin_link_v1 test rows: "
          f"{set(jj['MVT_ID_mvt']) == set(old['MVT_ID_mvt'])}")
    if not same or set(jj["MVT_ID_mvt"]) != set(old["MVT_ID_mvt"]):
        bad.append("link")
    # 3 gate numbers from the tool's own outputs (month-held-out PI)
    alld = pq.read_table(ROOT / "cache/all.parquet", columns=["MVT_ID_mvt", "t_mvt", "src", "PHASE_mvt", "y"],
                         filters=[("PHASE_mvt", "==", "DEP")]).to_pandas().set_index("MVT_ID_mvt")
    alld["day"] = pd.to_datetime(alld["t_mvt"], unit="s").dt.strftime("%Y-%m-%d")
    R25i = R25.set_index("MVT_ID_mvt")
    R25i = R25i.assign(art=(np.abs(R25i["y"] - R25i["d_sch"]) <= 60))

    def pi_ex(m):
        x = R25i[R25i["src"] != m]
        return (x["art"].sum() + 0.5) / (len(x) + 1.0)
    allr2 = L25[L25["r2"] & (L25["src"] > 0)]
    fj = [pd.read_parquet(ROOT / f"research/f2/out_mac/gate19_{c}_F.parquet", columns=["MVT_ID_mvt", "Lm"]).set_index("MVT_ID_mvt")["Lm"] for c in "ABC"]
    rulers = {
        "val": (pd.read_parquet(ROOT / "preds/holdout_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap_rmse281.37.parquet")
                .set_index("MVT_ID_mvt")["pred"].astype(float), [1, 7], 344419, False, -52.3),
        "fwd": (pd.read_parquet(ROOT / "preds/fwd_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.parquet")
                .set_index("MVT_ID_mvt")["pred"].astype(float), [7], 190713, False, -85.9),
        "fj": ((fj[0] + fj[1] + fj[2]) / 3, [2, 6], 326873, True, -2877.9),
    }
    for name, (P, months, N, floor, expect) in rulers.items():
        base = P.copy()
        rows = R25i[R25i["src"].isin(months)]
        if floor:   # fj is the corrector-stage Lm: emulate antiwin R2's floor first (m out of month), as research/r21/g5_r2a_crossfit.py
            for m in months:
                ids = rows.index[rows["src"] == m]
                mm = float(np.median(allr2[allr2["src"] != m]["y"] - (allr2[allr2["src"] != m]["d_lobt"] + WIN)))
                base.loc[ids] = np.maximum(base.loc[ids].to_numpy(), rows.loc[ids, "d_lobt"].to_numpy() + WIN + mm)
        pis = pd.Series({i: pi_ex(rows.loc[i, "src"]) for i in rows.index})
        new = apply_r2a(base, rows, pis)
        fp = out_dir / f"{name}_r2a.parquet"
        pd.DataFrame({"MVT_ID_mvt": new.index.to_numpy(), "pred": new.to_numpy()}).to_parquet(fp, index=False)
        got = pd.read_parquet(fp).set_index("MVT_ID_mvt")["pred"]   # numbers come from the tool's own output file
        y = alld["y"].reindex(got.index)
        ok = y.notna()
        dse = ((got[ok] - y[ok]) ** 2 - (base[ok] - y[ok]) ** 2)
        d = float(dse.sum() / N)
        days = alld[alld["src"].isin(months)]["day"].unique()
        dd = dse.groupby(alld["day"].reindex(dse.index)).sum().reindex(days).fillna(0).to_numpy()
        b = np.random.default_rng(0).integers(0, len(dd), (4000, len(dd)))
        print(f"3 {name}: rows changed {int((got != base).sum())}, dMSE {d:+.1f} (expected {expect:+.1f}), day-boot P(<0) {np.mean(dd[b].sum(1) < 0):.3f}, file sha16 {sha(fp, 16)}")
        if abs(d - expect) > 0.1:
            bad.append(f"gate {name}")
    # 4 the staged final, twice
    i1 = cmd_apply(str(FINAL_598C), str(out_dir / "598c_r2a_run1.parquet"))
    i2 = cmd_apply(str(FINAL_598C), str(out_dir / "598c_r2a_run2.parquet"))
    by_month = pd.Series([r["month"] for r in i1["rows"]]).value_counts().sort_index().to_dict()
    print(f"4 598c: changed {i1['changed_rows']} rows by month {by_month}, run1 == run2 bytes: {i1['output_sha256'] == i2['output_sha256']}")
    if i1["output_sha256"] != i2["output_sha256"] or i1["changed_rows"] != 13:
        bad.append("598c apply")
    print("SELFTEST", "PASS" if not bad else f"FAIL {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "apply" and len(a) == 3:
        cmd_apply(a[1], a[2])
    elif a and a[0] == "selftest":
        raise SystemExit(selftest())
    else:
        raise SystemExit(__doc__)
