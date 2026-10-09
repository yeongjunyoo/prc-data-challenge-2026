# SPDX-License-Identifier: GPL-3.0-only
"""R17-A copy of research/r16_b/s30_gate_r16b.py (sha16 3d1436f851ff9d0b, vault research/2026-10-08-r17-a-adsb-fill.md section 1): output
research/r17_a/gate_<spec>.json. Nothing else changes (same rulers, day resampling, gate 7.8 and probing-guard file list). Original text follows.
R16-B copy of research/r13_l/s30_gate_mac.py (sha16 c7d8a562d6d00bbb, vault note research/2026-10-08-r16-b-tabm.md sections 1 and 2): output
research/r16_b/gate_<spec>.json, and the probing guard also reads realqueue/hold/ and every research/*/queue_ready*/ folder (the VAR-1 folder
research/r14_brief/queue_ready_var1/ is not matched by research/*/queue_ready/). Only the guard's file list grows (stricter); every number is computed as before.
Original text follows.
R13-L Mac copy of research/r13_l/s30_gate_r13.py (ROOT from the file location). R13-L copy of research/r10_b/s30_gate.py: reference = 500 (sv114c3), output research/r13_l/gate_<spec>.json, guard over every
research/*/queue_ready/ folder. Original text follows.
R10-B S2 gate (pre-registered vault note 7.7, 2026-10-07 07:20 KST; gate 7.8 added by note 7.19, 14:51 KST, for builds measured after it):
a candidate manifest against 497's chain.
val and fwd all-rows dMSE < 0 with day-resampled P(dMSE < 0) >= 0.9 (UTC MVT day from cache/ctx_v3.parquet, 2,000 draws, seed 0, as
research/ctx_corr/day_boot.py); body_nonG (y <= 3600 and not G = LIRF without NM flight match) reported; test movement against 497; probing guard
(rounded submission values differ from every ledger submission and every queue/ and queue_ready/ file on more than 1% of rows).
    python s30_gate.py <candidate manifest> [<reference manifest>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]  # R13-L Mac copy: the sandbox root that holds this file
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if sys.platform == "win32":
    import ctypes
    ctypes.WinDLL("kernel32").SetPriorityClass(ctypes.WinDLL("kernel32").GetCurrentProcess(), 0x00004000)
REF = "preds/manifest_m6r27wincbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgjaavsk.json"
cand = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else REF
mc, mr = (json.loads((ROOT / p).read_text(encoding="utf-8")) for p in (cand, ref))
t = pq.read_table(ROOT / "cache" / "all.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "y", "src"], filters=[("PHASE_mvt", "==", "DEP")],
                  read_dictionary=["ADEP_mvt"])
A = pd.DataFrame({"ap": t.column("ADEP_mvt").to_pandas().astype(str).to_numpy(), "nm": t.column("FLIGHT_ID_mvt").is_valid().to_numpy(zero_copy_only=False),
                  "y": t.column("y").to_numpy().astype(float)}, index=t.column("MVT_ID_mvt").to_numpy())
del t
day = pd.read_parquet(ROOT / "cache" / "ctx_v3.parquet", columns=["MVT_ID_mvt", "day"]).set_index("MVT_ID_mvt")["day"].astype(str)
out = {"candidate": mc["spec"], "reference": mr["spec"]}
for st, key in (("val", "holdout"), ("fwd", "forward")):
    pc = pd.read_parquet(ROOT / mc[key]["path"]).set_index("MVT_ID_mvt")["pred"].astype(float)
    pr = pd.read_parquet(ROOT / mr[key]["path"]).set_index("MVT_ID_mvt")["pred"].astype(float).reindex(pc.index)
    M = A.reindex(pc.index)
    y = M["y"].to_numpy()
    dd = (pc.to_numpy() - y) ** 2 - (pr.to_numpy() - y) ** 2
    G = (M["ap"].to_numpy() == "LIRF") & ~M["nm"].to_numpy()
    bng = np.where((y <= 3600) & ~G, dd, 0.0)
    g = pd.DataFrame({"d": dd, "b": bng, "day": day.reindex(pc.index).to_numpy()}).groupby("day")[["d", "b"]].sum()
    n = pd.Series(1.0, index=pc.index).groupby(day.reindex(pc.index).to_numpy()).sum().reindex(g.index).to_numpy()
    rng = np.random.default_rng(0)
    dr = rng.integers(0, len(g), size=(2000, len(g)))
    bt = g["d"].to_numpy()[dr].sum(axis=1) / n[dr].sum(axis=1)
    bb = g["b"].to_numpy()[dr].sum(axis=1) / n[dr].sum(axis=1)
    mon = pd.Series(dd).groupby(pd.Series(day.reindex(pc.index).to_numpy()).str[5:7].to_numpy()).sum() / len(dd)
    out[st] = {"rmse_ref": round(float(np.sqrt(np.mean((pr.to_numpy() - y) ** 2))), 3), "rmse": round(float(np.sqrt(np.mean((pc.to_numpy() - y) ** 2))), 3),
               "dmse": round(float(dd.mean()), 2), "boot_sd": round(float(bt.std(ddof=1)), 2), "p_below0": round(float((bt < 0).mean()), 4),
               "body_nonG": round(float(bng.sum() / len(dd)), 2), "body_p_below0": round(float((bb < 0).mean()), 4),
               "months": {k: round(float(v), 2) for k, v in mon.items()}}
    print(f"[{st}] {out[st]}", flush=True)
pc = pd.read_parquet(ROOT / mc["test"]["path"]).set_index("MVT_ID_mvt")["pred"].astype(float)
pr = pd.read_parquet(ROOT / mr["test"]["path"]).set_index("MVT_ID_mvt")["pred"].astype(float).reindex(pc.index)
mv = pc.to_numpy() - pr.to_numpy()
out["test_move_vs_ref"] = {"move_mse": round(float(np.mean(mv ** 2)), 2), "rows_moved": int((np.abs(mv) > 0.5).sum()), "mean_move": round(float(mv.mean()), 3)}
import src.shot as S  # noqa: E402
tpl = S.template_ids()
vals = S.submission_values(pc, tpl)
gd = S.guard(vals, tpl)
qd = {}
# Mac copy: the real queue/ and queue/done/ are copied to realqueue/ (the sandbox queue/ holds this lane's private entries)
for f in (sorted((ROOT / "realqueue").glob("*.parquet")) + sorted((ROOT / "realqueue" / "done").glob("*.parquet")) + sorted((ROOT / "realqueue" / "hold").glob("*.parquet"))
          + sorted((ROOT / "research").glob("*/queue_ready*/*.parquet"))):
    q = pd.read_parquet(f)
    col = "pred" if "pred" in q.columns else "TAXITIME_SEC_mvt"
    qd[f.stem] = float(np.mean(S.submission_values(q.set_index("MVT_ID_mvt")[col], tpl) != vals))
near = min(qd, key=qd.get) if qd else None
out["guard"] = {"ledger_pass": bool(gd["pass"]), "ledger_nearest": gd["nearest"], "ledger_min_diff": gd["min_diff_frac"],
                "queue_nearest": near, "queue_min_diff": round(qd[near], 5) if near else None}
out["gate"] = {"val": out["val"]["dmse"] < 0 and out["val"]["p_below0"] >= 0.9, "fwd": out["fwd"]["dmse"] < 0 and out["fwd"]["p_below0"] >= 0.9,
               "guard": bool(gd["pass"]) and (qd[near] if near else 1) >= 0.01}
out["pass"] = all(out["gate"].values())
# gate 7.8 (vault note 7.19, coordinator decision 14:5x, registered 14:51 for builds not yet measured): body_nonG < 0 with P >= 0.9 on val and
# fwd, all-rows val < 0 with P >= 0.9, all-rows fwd point estimate < 0, guard; all-rows fwd P is reported
out["gate78"] = {"a_body_val": out["val"]["body_nonG"] < 0 and out["val"]["body_p_below0"] >= 0.9,
                 "a_body_fwd": out["fwd"]["body_nonG"] < 0 and out["fwd"]["body_p_below0"] >= 0.9,
                 "b_val_all": out["val"]["dmse"] < 0 and out["val"]["p_below0"] >= 0.9,
                 "c_fwd_point": out["fwd"]["dmse"] < 0, "d_guard": out["gate"]["guard"]}
out["pass78"] = all(out["gate78"].values())
print(f"[test] {out['test_move_vs_ref']} | guard {out['guard']}", flush=True)
print(f"[gate] 7.7 {out['gate']} -> {'PASS' if out['pass'] else 'FAIL'}", flush=True)
print(f"[gate7.8] {out['gate78']} (fwd all-rows P {out['fwd']['p_below0']}) -> {'PASS' if out['pass78'] else 'FAIL'}", flush=True)
(ROOT / "research" / "r17_a" / f"gate_{mc['spec']}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
