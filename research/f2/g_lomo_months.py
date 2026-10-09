# SPDX-License-Identifier: GPL-3.0-only
"""F2 fork: per-month (2025 lomo-12) G-row SSE after each G stage, from existing caches only (no leg retraining).
Stages: one-key leg G model (model3.g_probs, recomputed here) -> gkey avg (two keys) -> gkeyx (12 cols, guard 43,200) -> gkeyxn avg_xncr
(linear pool of the four components) -> gpool S1 (log pool, month-out a) -> glong G1 (r from the month-out rows).
Normal and shift taxi = constant mu of the other 11 months (median normal y), as in research/r9_lirf/c7_gpool.py; no band rule (no leg p_art),
so gjoint a is not measured here (see research/r11_g/m2_bandA.txt). Sources: research/r6_g/s5_P.parquet (gkeyx P, Pav, mu),
research/r11_g/e1_probs.parquet + e1_info.parquet (current-tool components, chain P after S1, a and r per lomo month)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
import importlib.util
from src import sub_g
from src.model3 import g_probs
def tool(n):
    s = importlib.util.spec_from_file_location(n, ROOT / "tools" / f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
GK = tool("gkey")
G = GK.g_frame("feats_v27")
s = G["src"].to_numpy(); fin = np.isfinite(G["y"].to_numpy(float))
P5 = pd.read_parquet(ROOT / "research/r6_g/s5_P.parquet")
E1 = pd.read_parquet(ROOT / "research/r11_g/e1_probs.parquet")
E1 = E1[E1["role"] == "eval"]
INFO = pd.read_parquet(ROOT / "research/r11_g/e1_info.parquet").set_index("setting")
CUT, D0 = 43200.0, 30000.0
rows = []
maxdiff_av = 0.0
for m in range(1, 13):
    st = f"lomo{m}"
    Ge = G[s == m]; Gf = G[(s != 0) & (s != m) & fin]
    P_leg = g_probs(Gf, Ge)
    ids = Ge["MVT_ID_mvt"].to_numpy()
    S5 = P5[P5["setting"] == st].set_index("MVT_ID_mvt").reindex(ids)
    e1 = E1[E1["setting"] == st].set_index("MVT_ID_mvt").reindex(ids)
    d, dd, y, mu = (S5[c].to_numpy(float) for c in ("d", "dd", "y", "mu"))
    av_e1 = e1[["avn", "ava", "avs"]].to_numpy(float)
    av_s5 = S5[["Pav0", "Pav1", "Pav2"]].to_numpy(float)
    maxdiff_av = max(maxdiff_av, float(np.nanmax(np.abs(av_e1 - av_s5))))
    Px = S5[["P0", "P1", "P2"]].to_numpy(float)  # gkeyx with guard (s5_cur_q: P = guard applied)
    C = [e1[[f"c{i}n", f"c{i}a", f"c{i}s"]].to_numpy(float) for i in range(4)]
    lin = 0.5 * (0.5 * C[0] + 0.5 * C[1]) + 0.5 * (0.5 * C[2] + 0.5 * C[3])
    g = d >= CUT
    lin[g] = av_e1[g]
    S1 = e1[["pn", "pa", "ps"]].to_numpy(float)
    r = float(INFO.loc[st, "r"])
    pn, pa, ps = S1[:, 0], S1[:, 1], S1[:, 2]
    band = d >= D0; tot = pa + ps; wa = np.where(tot > 0, pa / np.where(tot > 0, tot, 1), 0.5); free = (1 - r) * pn
    G1 = np.column_stack([np.where(band, r * pn, pn), np.where(band, pa + free * wa, pa), np.where(band, ps + free * (1 - wa), ps)])
    ok = np.isfinite(y)
    def sse(P):
        return float(((sub_g.expected(P, d, dd, mu, mu) - y)[ok] ** 2).sum())
    rec = {"m": m, "rows": int(ok.sum()), "leg1key": sse(P_leg), "gkey_avg": sse(av_e1), "gkeyx": sse(Px), "gkeyxn": sse(lin), "gpool": sse(S1), "glong": sse(G1)}
    rows.append(rec)
R = pd.DataFrame(rows).set_index("m")
st_ = ["leg1key", "gkey_avg", "gkeyx", "gkeyxn", "gpool", "glong"]
print(f"check: gkey avg probabilities s5_P vs e1 max abs diff {maxdiff_av:.3g}")
out = pd.DataFrame({"rows": R["rows"], "SSE_leg_e8": (R["leg1key"] / 1e8).round(2)})
for a, b in zip(st_[:-1], st_[1:]):
    out[f"{b}_%"] = (100 * (R[b] / R[a] - 1)).round(2)
out["all_vs_leg_%"] = (100 * (R["glong"] / R["leg1key"] - 1)).round(2)
print(out.to_string())
tot = R.sum()
print("12-month total, % vs previous stage:", {b: round(100 * (tot[b] / tot[a] - 1), 2) for a, b in zip(st_[:-1], st_[1:])}, "all vs leg", round(100 * (tot["glong"] / tot["leg1key"] - 1), 2))
for lab, ms in (("Feb+Jun", [2, 6]), ("Jan+Jul", [1, 7])):
    t = R.loc[ms].sum()
    print(lab, {b: round(100 * (t[b] / t[a] - 1), 2) for a, b in zip(st_[:-1], st_[1:])}, "all vs leg", round(100 * (t["glong"] / t["leg1key"] - 1), 2))
