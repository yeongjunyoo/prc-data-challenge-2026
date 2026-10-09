# SPDX-License-Identifier: GPL-3.0-only
# ADS-B 오프블록 덧씌우기 규칙의 값을 두 정답 날(2025-01-15, 2025-07-15)에서 잰다. 모델은 봉인된 m6r_avg2 홀드아웃(자 1, 7월)이다.
# 공항별 보정(정답 - ADS-B 추정의 중앙값)은 다른 날에서 구해 바꿔 끼운다(같은 날 정답으로 맞추지 않는다).
# 읽기만 하고 결과는 research/offblock/overlay_eval.json 에 쓴다.
#     .venv/Scripts/python.exe -X utf8 research/offblock/overlay_eval.py
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DAYS = ["2025-01-15", "2025-07-15"]
pred = pd.read_parquet(ROOT / "preds" / "holdout_m6r_avg2_rmse317.74.parquet")
N_RULER = len(pred)
lab = pd.read_parquet(ROOT / "cache" / "feats_v24.parquet", columns=["MVT_ID_mvt", "y", "src"])
lab = lab[lab["src"].isin([1, 7])]
ruler = pred.merge(lab[["MVT_ID_mvt", "y"]], on="MVT_ID_mvt")
SSE_RULER = float(((ruler["pred"] - ruler["y"]) ** 2).sum())
print(f"ruler rows {N_RULER}, RMSE {np.sqrt(SSE_RULER / N_RULER):.3f}")

M = {}
for day in DAYS:
    m = pd.read_parquet(ROOT / "data_ext" / "adsblol" / day / "matched_departures.parquet")
    m = m.merge(pred, on="MVT_ID_mvt", how="inner")
    m["fb"] = m["BLOCK_TIME_UTC_mvt"] == m["SCHED_TIME_UTC_mvt"]
    M[day] = m
    print(f"{day}: departures {len(m)}, usable {int(m['usable'].sum())}, first_seen_moving {int((m['ev_first_seen_moving'] == True).sum())}")


def offsets(m: pd.DataFrame) -> dict:
    u = m[m["usable"] & ~m["fb"] & (m["TAXITIME_SEC_mvt"] < 3600)]
    return (u["TAXITIME_SEC_mvt"] - u["est_mvt"]).groupby(u["ADEP_mvt"]).median().to_dict() | {"_all": float((u["TAXITIME_SEC_mvt"] - u["est_mvt"]).median())}


res = {"ruler_rmse": round(float(np.sqrt(SSE_RULER / N_RULER)), 3), "days": {}}
rules = {}
for i, day in enumerate(DAYS):
    m = M[day].copy()
    off = offsets(M[DAYS[1 - i]])
    m["off"] = m["ADEP_mvt"].map(off).fillna(off["_all"])
    m["est"] = np.clip(m["est_mvt"] + m["off"], 0, None)
    y, p = m["TAXITIME_SEC_mvt"].to_numpy(float), m["pred"].to_numpy(float)
    u = m["usable"].to_numpy(bool)
    est = m["est"].to_numpy(float)
    diff = np.abs(est - p)
    base = (p - y) ** 2
    out = {}
    for T in (60, 120, 180, 300, 600, 1e9):
        q = np.where(u & (diff <= T), est, p)
        out[f"replace_if_within_{int(T) if T < 1e9 else 'any'}"] = q
        out[f"blend50_if_within_{int(T) if T < 1e9 else 'any'}"] = np.where(u & (diff <= T), 0.5 * est + 0.5 * p, p)
    obs = u & (m["ev_move_gap"].fillna(1e9).to_numpy() <= 30)
    out["replace_if_start_observed"] = np.where(obs, est, p)
    out["replace_if_start_observed_within_300"] = np.where(obs & (diff <= 300), est, p)
    fs = (m["ev_first_seen_moving"] == True).to_numpy() & ~np.isnan(m["est_fs"].to_numpy(float))
    out["lower_bound_first_seen"] = np.where(fs, np.maximum(p, m["est_fs"].to_numpy(float)), p)
    out["replace_within_300_plus_lower_bound"] = np.where(fs, np.maximum(out["replace_if_within_300"], m["est_fs"].to_numpy(float)), out["replace_if_within_300"])
    d = {"departures": int(len(m)), "usable": int(u.sum()), "offsets_from_other_day": {k: round(v, 1) for k, v in off.items()}}
    for k, q in out.items():
        dsse = float(((q - y) ** 2 - base).sum())
        rules.setdefault(k, []).append(dsse)
        d[k] = {"dSSE": round(dsse), "rows_changed": int((q != p).sum())}
    # 공항별: 덧씌운 행에서 모델과 규칙의 RMSE
    per = {}
    for ap, g in m.assign(q=out["replace_if_within_300"]).groupby("ADEP_mvt"):
        ch = g["q"] != g["pred"]
        if ch.sum():
            per[ap] = {"changed": int(ch.sum()), "model_rmse": round(float(np.sqrt(((g.loc[ch, "pred"] - g.loc[ch, "TAXITIME_SEC_mvt"]) ** 2).mean())), 1),
                       "overlay_rmse": round(float(np.sqrt(((g.loc[ch, "q"] - g.loc[ch, "TAXITIME_SEC_mvt"]) ** 2).mean())), 1)}
    d["per_airport_replace_within_300"] = per
    res["days"][day] = d

print("\nrule: total dSSE on the two days -> ruler RMSE if every ruler day gained like these two (x31)")
summ = {}
for k, v in rules.items():
    tot = sum(v)
    new = np.sqrt(max(SSE_RULER + tot * 31, 0) / N_RULER)
    summ[k] = {"dSSE_two_days": round(tot), "ruler_rmse_extrapolated": round(float(new), 3), "per_day": [round(x) for x in v]}
    print(f"  {k:45s} dSSE {tot:14,.0f}  -> {new:8.3f}   per day {[round(x) for x in v]}")
res["summary"] = summ
for day in DAYS:
    print(day, "per airport (replace within 300):", res["days"][day]["per_airport_replace_within_300"])
(OUT / "overlay_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
