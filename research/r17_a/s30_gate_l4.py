# SPDX-License-Identifier: GPL-3.0-only
"""R17-A L4 gate on the real settings only: research/r17_a/s30_gate_r17a.py (sha16 01cc337baa60db03, of the R10-B s30_gate.py family) with the
same rulers, the same day resampling (UTC MVT day from cache/ctx_v3.parquet, 2,000 draws, seed 0) and the same body_nonG definition
(y <= 3600 and not G = LIRF without NM flight match), restricted to the settings that are real L4 in this build (private notes
research/2026-10-08-r17-a-adsb-fill.md section 8; 10-10 10:2x). The test move and the probing guard (d) are not computed here:
full is a placeholder in the gate-only clone, and F3 measures (d) and the test move on the final root after full.
Gate 7.8 parts by setting: val gives (b) all-rows dMSE < 0 with P(dMSE < 0) >= 0.9 and the val half of (a) body_nonG < 0 with P >= 0.9;
fwd gives the fwd half of (a) and (c) all-rows fwd point estimate < 0 (the all-rows fwd P is reported).
    python research/r17_a/s30_gate_l4.py <candidate manifest> <reference manifest> <settings val or val,fwd> <placeholder map json> <out json>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
cand, ref, settings, pmap, outp = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4], sys.argv[5]
assert set(settings) <= {"val", "fwd"} and settings[0] == "val"
mc, mr = (json.loads((ROOT / p).read_text(encoding="utf-8")) for p in (cand, ref))


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


t = pq.read_table(ROOT / "cache" / "all.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "y", "src"], filters=[("PHASE_mvt", "==", "DEP")],
                  read_dictionary=["ADEP_mvt"])
A = pd.DataFrame({"ap": t.column("ADEP_mvt").to_pandas().astype(str).to_numpy(), "nm": t.column("FLIGHT_ID_mvt").is_valid().to_numpy(zero_copy_only=False),
                  "y": t.column("y").to_numpy().astype(float)}, index=t.column("MVT_ID_mvt").to_numpy())
del t
day = pd.read_parquet(ROOT / "cache" / "ctx_v3.parquet", columns=["MVT_ID_mvt", "day"]).set_index("MVT_ID_mvt")["day"].astype(str)
out = {"candidate": mc["spec"], "reference": mr["spec"], "settings_real": settings,
       "placeholder_map": json.loads((ROOT / pmap).read_text(encoding="utf-8")),
       "files": {}, "not_computed": "test move and probing guard (d): full is a placeholder here; F3 measures them on the final root"}
for st, key in (("val", "holdout"), ("fwd", "forward")):
    if st not in settings:
        continue
    out["files"][st] = {"candidate": mc[key]["path"], "candidate_sha16": sha16(ROOT / mc[key]["path"]),
                        "reference": mr[key]["path"], "reference_sha16": sha16(ROOT / mr[key]["path"])}
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
    out[st] = {"rows": int(len(dd)), "rmse_ref": round(float(np.sqrt(np.mean((pr.to_numpy() - y) ** 2))), 3),
               "rmse": round(float(np.sqrt(np.mean((pc.to_numpy() - y) ** 2))), 3), "dmse": round(float(dd.mean()), 2),
               "boot_sd": round(float(bt.std(ddof=1)), 2), "p_below0": round(float((bt < 0).mean()), 4),
               "body_nonG": round(float(bng.sum() / len(dd)), 2), "body_p_below0": round(float((bb < 0).mean()), 4),
               "months": {k: round(float(v), 2) for k, v in mon.items()}}
    print(f"[{st}] {out[st]}", flush=True)
v = out["val"]
parts = {"b_val_all": v["dmse"] < 0 and v["p_below0"] >= 0.9, "a_body_val": v["body_nonG"] < 0 and v["body_p_below0"] >= 0.9}
if "fwd" in settings:
    f = out["fwd"]
    parts["a_body_fwd"] = f["body_nonG"] < 0 and f["body_p_below0"] >= 0.9
    parts["c_fwd_point"] = f["dmse"] < 0
out["gate78_parts"] = parts
out["parts_pass"] = all(parts.values())
print(f"[gate7.8 parts on {settings}] {parts} -> {'PASS' if out['parts_pass'] else 'FAIL'}", flush=True)
(ROOT / outp).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
