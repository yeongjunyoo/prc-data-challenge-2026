"""F2 Package S acceptance, rules 2 to 4 of the approval (coordinator 10-11 01:17 KST; registered in the PRC master at 01:17, before any
number of the package, and after C3's gate 7.8 failure was known, which the registration states). Rule 1 (construction) is s_check.py.
  2. three-chain val and fwd body ΔMSE point estimates <= 0 (body = body_nonG: y <= 3600 and not G = LIRF without an NM flight match,
     summed over body rows and divided by all rows of the ruler, as research/r17_a/s30_gate_l4.py); no significance requirement.
  3. tripwires: all-rows ΔMSE <= +2 x the day-bootstrap sd on both rulers (UTC MVT day from cache/ctx_v3.parquet, 2,000 draws, seed 0, the
     s30 resampling), and the final move MSE <= 150 (mean over the final test rows of (S final - the family's final)^2).
  4. (coordinator addition) one board shot of the package's Jan/Jul twin 604 on UTC 10-12: reject if it is more than +100 MSE worse than the
     base twin (597's v43 228.5043 for the 598c family; 601's score for the L4 family): S^2 - base^2 > 100 rejects. A veto only: it never
     selects.
ΔMSE = candidate minus reference; negative is better.
    python -X utf8 research/f2/s_gate.py sbx <candidate manifest> <reference manifest> <out json>        (rules 2 and 3a; cwd = the root)
    python -X utf8 research/f2/s_gate.py move <S final test parquet> <family final test parquet> <out json>   (rule 3b)
    python -X utf8 research/f2/s_gate.py board <604 score> <base twin score> <out json>                    (rule 4)
    python -X utf8 research/f2/s_gate.py verdict <out json> <json> [<json> ...]   every input's "pass" true -> adopt (rule 1's s_check jsons too)
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path.cwd()
MOVE_MAX, BOARD_MAX, K_SD = 150.0, 100.0, 2.0


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def ruler_frame() -> tuple[pd.DataFrame, pd.Series]:
    t = pq.read_table(ROOT / "cache" / "all.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "y", "src"],
                      filters=[("PHASE_mvt", "==", "DEP")], read_dictionary=["ADEP_mvt"])
    A = pd.DataFrame({"ap": t.column("ADEP_mvt").to_pandas().astype(str).to_numpy(),
                      "nm": t.column("FLIGHT_ID_mvt").is_valid().to_numpy(zero_copy_only=False),
                      "y": t.column("y").to_numpy().astype(float)}, index=t.column("MVT_ID_mvt").to_numpy())
    day = pd.read_parquet(ROOT / "cache" / "ctx_v3.parquet", columns=["MVT_ID_mvt", "day"]).set_index("MVT_ID_mvt")["day"].astype(str)
    return A, day


def ruler_stats(pc: pd.Series, pr: pd.Series, A: pd.DataFrame, day: pd.Series) -> dict:
    if not pc.index.equals(pr.index):
        if set(pc.index) != set(pr.index) or len(pc) != len(pr):
            raise SystemExit("candidate and reference rows differ")
        pr = pr.reindex(pc.index)
    M = A.reindex(pc.index)
    if M["y"].isna().any():
        raise SystemExit("rows without a label in the ruler")
    y = M["y"].to_numpy()
    dd = (pc.to_numpy() - y) ** 2 - (pr.to_numpy() - y) ** 2
    G = (M["ap"].to_numpy() == "LIRF") & ~M["nm"].to_numpy()
    bng = np.where((y <= 3600) & ~G, dd, 0.0)
    dv = day.reindex(pc.index).to_numpy()
    g = pd.DataFrame({"d": dd, "b": bng, "day": dv}).groupby("day")[["d", "b"]].sum()
    n = pd.Series(1.0, index=pc.index).groupby(dv).sum().reindex(g.index).to_numpy()
    rng = np.random.default_rng(0)
    dr = rng.integers(0, len(g), size=(2000, len(g)))
    bt = g["d"].to_numpy()[dr].sum(axis=1) / n[dr].sum(axis=1)
    bb = g["b"].to_numpy()[dr].sum(axis=1) / n[dr].sum(axis=1)
    raw = {"dmse": float(dd.mean()), "boot_sd": float(bt.std(ddof=1)), "body_nonG": float(bng.sum() / len(dd))}  # the rules read these
    return {"rows": int(len(dd)), "days": int(len(g)), "rmse_ref": round(float(np.sqrt(np.mean((pr.to_numpy() - y) ** 2))), 3),
            "rmse": round(float(np.sqrt(np.mean((pc.to_numpy() - y) ** 2))), 3), "dmse": round(raw["dmse"], 2),
            "boot_sd": round(raw["boot_sd"], 2), "p_below0": round(float((bt < 0).mean()), 4),
            "body_nonG": round(raw["body_nonG"], 2), "body_p_below0": round(float((bb < 0).mean()), 4),
            "body_rows": int(((y <= 3600) & ~G).sum()), "g_rows": int(G.sum()), "raw": raw}


def sbx(cand: str, ref: str) -> dict:
    mc, mr = (json.loads((ROOT / p).read_text(encoding="utf-8")) for p in (cand, ref))
    A, day = ruler_frame()
    out = {"candidate": mc["spec"], "reference": mr["spec"], "files": {}}
    for st, key in (("val", "holdout"), ("fwd", "forward")):
        pcp, prp = ROOT / mc[key]["path"], ROOT / mr[key]["path"]
        out["files"][st] = {"candidate": mc[key]["path"], "candidate_sha16": sha16(pcp), "reference": mr[key]["path"], "reference_sha16": sha16(prp)}
        pc = pd.read_parquet(pcp).set_index("MVT_ID_mvt")["pred"].astype(float)
        pr = pd.read_parquet(prp).set_index("MVT_ID_mvt")["pred"].astype(float)
        out[st] = ruler_stats(pc, pr, A, day)
        print(f"[{st}] {out[st]}", flush=True)
    v, f = out["val"]["raw"], out["fwd"]["raw"]  # unrounded: a body of +0.004 must not pass as 0.0 (rounding is for display only)
    out["rules"] = {"2_body_val_le0": v["body_nonG"] <= 0, "2_body_fwd_le0": f["body_nonG"] <= 0,
                    "3a_all_val_le_2sd": v["dmse"] <= K_SD * v["boot_sd"], "3a_all_fwd_le_2sd": f["dmse"] <= K_SD * f["boot_sd"]}
    out["thresholds"] = {"3a_val": round(K_SD * v["boot_sd"], 2), "3a_fwd": round(K_SD * f["boot_sd"], 2)}
    out["pass"] = all(out["rules"].values())
    return out


def move(sp: str, bp: str) -> dict:
    a = pd.read_parquet(sp).set_index("MVT_ID_mvt")["pred"].astype(float)
    b = pd.read_parquet(bp).set_index("MVT_ID_mvt")["pred"].astype(float)
    if len(a) != len(b) or set(a.index) != set(b.index) or a.index.has_duplicates:
        raise SystemExit("the two finals do not cover the same rows")
    mv = a.to_numpy() - b.reindex(a.index).to_numpy()
    if not np.isfinite(mv).all():
        raise SystemExit("non-finite predictions")
    mse = float(np.mean(mv ** 2))
    out = {"s_final": sp, "s_sha16": sha16(Path(sp)), "family_final": bp, "family_sha16": sha16(Path(bp)), "rows": int(len(mv)),
           "move_mse": round(mse, 2), "mean_move": round(float(mv.mean()), 3), "rows_moved_over_0.5s": int((np.abs(mv) > 0.5).sum()),
           "max_abs_move": round(float(np.abs(mv).max()), 2), "limit": MOVE_MAX, "raw": {"move_mse": mse}}
    out["rules"] = {"3b_move_le_150": mse <= MOVE_MAX}  # unrounded
    out["pass"] = out["rules"]["3b_move_le_150"]
    return out


def board(s: float, b: float) -> dict:
    d = s * s - b * b
    out = {"s_twin_score": s, "base_twin_score": b, "mse_diff": round(d, 2), "limit": BOARD_MAX,
           "reject_above_score": round(math.sqrt(b * b + BOARD_MAX), 4), "note": "veto only; never selects"}
    out["rules"] = {"4_board_le_plus100": d <= BOARD_MAX}
    out["pass"] = out["rules"]["4_board_le_plus100"]
    return out


def main() -> int:
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if a[0] == "sbx" and len(a) == 4:
        r, outp = sbx(a[1], a[2]), a[3]
    elif a[0] == "move" and len(a) == 4:
        r, outp = move(a[1], a[2]), a[3]
    elif a[0] == "board" and len(a) == 4:
        r, outp = board(float(a[1]), float(a[2])), a[3]
    elif a[0] == "verdict" and len(a) >= 3:
        parts = {}
        for p in a[2:]:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            parts[p] = bool(d.get("pass"))
        r, outp = {"inputs": parts, "pass": all(parts.values()), "adopt": all(parts.values())}, a[1]
    else:
        raise SystemExit(__doc__)
    r["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    Path(outp).write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"s_gate {a[0]}: {r.get('rules', r.get('inputs'))} -> {'PASS' if r['pass'] else 'FAIL'} ({outp})", flush=True)
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
