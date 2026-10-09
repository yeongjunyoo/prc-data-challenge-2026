# SPDX-License-Identifier: GPL-3.0-only
"""F3: the build manifest of the first final file (beside the per-spec manifests src/manifest.py writes), research/f3/out/final_build_manifest.json:
what this rebuild of 514 did differently from 514's own builders, with the hashes that pin each difference.
  - inputs: final_ranking and final_submitting, the tables of the final root (sha16), R17-A pack 2, the 2025 rows kept from today's tables;
  - fit cache: research/f3/fitwrap.py sha16 (frozen), the interim feats_v27 sha16, the wrapper test and T1b, the twin, and for every leg run of
    the final root the loaded / fitted / uncached counts from research/f3/fc_legs/fitwrap_log.jsonl;
  - legs: CatBoost at 8 threads (R16-D's thread-invariance measurement), XGBoost not refit (weight 0, placeholder check per chain), TabM device;
  - recorded one-off changes: ctx_v2 st_k cell of MVT_ID 182512875 kept at today's 0, geo regression tolerance;
  - the file: sha16, rows, checks (research/f3/out/final_file_<spec>.json) and the daily-format candidate's name.
    python -X utf8 research/f3/final_manifest.py <spec>
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

O = Path("research/f3/out")


def sha16(p) -> str | None:
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def jl(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def leg_runs() -> dict:
    """Runs of the final root through fitwrap load: the start lines (target, arguments, time) and the summed counts of their end lines
    (the log interleaves concurrent runs and an end line does not name its run, so counts are summed, not paired)."""
    ev = [json.loads(x) for x in open("research/f3/fc_legs/fitwrap_log.jsonl", encoding="utf-8") if x.strip()]
    starts = [e for e in ev if e.get("event") == "start" and e.get("mode") == "load" and str(e.get("cwd", "")).endswith("final_run")]
    t0 = min((e["t"] for e in starts), default=None)
    ends = [e for e in ev if e.get("event") == "end" and t0 and e["t"] >= t0 and "stats" in e]
    loads = [e for e in ev if e.get("lib") and e.get("event") in ("loaded", "MISS_fit", "MISS_STOP", "uncached") and t0 and e["t"] >= t0]
    return {"runs": [{"target": e["target"], "args": " ".join(e["args"]), "start": e["t"]} for e in starts],
            "events_by_kind": {k: sum(1 for e in loads if e["event"] == k) for k in ("loaded", "MISS_fit", "MISS_STOP", "uncached")},
            "end_lines": len(ends), "end_rc_not_ok": [e for e in ends if e.get("rc") != "ok"]}


def main():
    spec = sys.argv[1]
    tables = {t: sha16(f"cache/{t}.parquet") for t in ("all", "feats_v25", "feats_v27", "ctx_v1", "ctx_v2", "ctx_v3", "adsb_v2", "adsb_v1nd", "geo_v1nd",
                                                        "antiwin_link_v1", "gkey_v1", "gkeyx_v1", "gkeyxn_v2", "artmm_v1", "artmm_cells_v1", "arthist_v1")}
    fin = leg_runs()
    m = {
        "written": time.strftime("%Y-%m-%d %H:%M:%S KST"),
        "spec": spec,
        "rebuild_of": "514 (v35, daily LB 228.384), code sha16 of its 47 files equal in this root",
        "data": {"ranking": sha16("data/ranking.parquet"), "submitting": sha16("data/submitting.parquet"),
                 "note": "final_ranking 4b0692a792846079, final_submitting b3445d80e29bf599 (organizer, 10-08 11:57 UTC)"},
        "tables_sha16": tables,
        "r17a_pack2": jl(O / "adsb_pack2.json"),
        "feats_v27_check": jl(O / "check_v27.json"),
        "interim_vs_real_2025_rows": jl(O / "v27_interim_vs_real.json"),
        "fit_cache": {"fitwrap_sha16": sha16("research/f3/fitwrap.py"), "fitwrap_sha16_frozen": "54114020d2aa3d3c",
                      "interim_feats_v27_sha16": "60028382131600d3", "store": "research/f3/fc_legs",
                      "wrapper_test": jl(O / "wrapper_test.json"),
                      "twin_sk_s0": jl(O / "twin_sk_s0.json"),
                      "final_root_load_runs": fin},
        "legs": {"catboost_threads": "8 for every CatBoost leg (514: cbwin 10, cb 5, cbap 8); R16-D refit cbwin at 8 threads and reproduced the "
                                     "deployed mu_cat exactly (max diff 0.0 s), research/2026-10-08-r16-d-nd-legs.md",
                 "xgboost": "not refit: weight exactly 0 in the three full blends; placeholder mu_xgb = 0, blend output identical at 0 and 1 "
                            "(research/f3/out/final_chain_<A|B|C>.json checks.xgb_weight)",
                 "tabm": {"device": "mps (514's command); the CUDA run is kept aside, see tabm_move",
                          "cuda_run_sha16": "1fd9f495c0bfb04a", "cuda_run": "research/f3/out/tabm_cuda/full_m6r27tmcwin_s0.parquet",
                          "cache_sha16": sha16("cache/full_m6r27tmcwin_s0.parquet"), "move_cuda_vs_mps": jl(O / "tabm_move.json")},
                 "realmlp_device": "mps (514's command)"},
        "chains": {k: jl(O / f"final_chain_{k}.json") for k in ("A", "B", "C")},
        "one_off_changes": ["ctx_v2: st_k of MVT_ID 182512875 (2025-01, EDDM) kept at today's value 0 (patch_ctx2025.py, ctx_v2.json patched_2025; "
                            "coordinator 10-08 22:5x)",
                            "ctx_v1: stand_p codes frozen from today's table, new prefixes appended (LEMD_PE 159, LSZH_WE 160)",
                            "geo_final regression check: discrete columns bit-equal, distances within 1e-6 m (first exact run: 1.8e-12 m)",
                            "2026-06-11 rows absent from adsb_v2, adsb_v1nd, geo_v1nd (R17-A pack 2: short source day read as a day without ADS-B)",
                            "src/manifest.py:126 asserted the public template's 344,841 test rows; in this root it checks against the template's own "
                            "row count (87c960a26a0d7def -> 030ea98285708daa); the recording tool, in no manifest's code_sha",
                            "tail tools keep N_TEST = 344841 (hash-pinned code): their report field move_mse_test_n is a sum over 670,790 rows divided "
                            "by 344,841 here, a reporting-only number (predictions unaffected)",
                            "research/f3/final_chain.py check_xgb_weight links data/ into its sandbox (blend_oof.ruler_days reads the 2025-01 and -07 "
                            "training files); the rerun blend of chain A is byte-identical to the 10:48 output (84d4f8b748a911d9)",
                            "TabM full leg: MPS with 514's command (cache sha16 a4c0f7e9085c644c, 10-09 10:48-12:37); the CUDA run (1fd9f495c0bfb04a) "
                            "is kept in research/f3/out/tabm_cuda/ and compared in tabm_move.json"],
        "file": {"path": "research/f3/out/spirited-tiger_final.parquet", "sha16": sha16(O / "spirited-tiger_final.parquet"),
                 "checks": jl(O / f"final_file_{spec}.json")},
        "daily_candidate": jl(Path("research/f3/queue_ready") / f"f3f1_{spec}.json"),
        "pc_candidates": {
            "daily_580": {"name": f"580_{spec}ff1", "where": "prc-2026/research/f3/queue_ready/", "test_sha16": "6e433c26a3cdd59b",
                          "rows": "Jan/Jul 344,841, public-phase template order", "feats_live": "cache/feats_v27_f3final.parquet 8fc36fd4314708a8",
                          "checks": "candidate(), pred_problems, sane, guard (vs v35 95.98% rows differ), empty-on-test, composition (514's tail-share warning only)",
                          "expected_daily": "228.6 to 229.1, about 228.85; alarm above 229.4 (reference v35 514 228.384)"},
            "final_590": {"name": f"590_{spec}ff1", "where": "prc-2026/research/f3/queue_ready_final/ (not in queue/)",
                          "parquet_sha256": "6845ed61803d305ea35327dea9f8bfd003bea033831d6222663d1c607ee05120",
                          "checks": "final_data_problems none, template 670,790 rows same IDs and order, pred_problems none, sane, candidate(), "
                                    "empty-on-final none; the int file built with build_final's formula is byte-identical to this root's file"}},
        "build_steps": {"phase3b": "research/f3/phase3b.sh (resume after the 10:48 stop: xgb-check sandbox got data/)",
                        "phase3c": "research/f3/phase3c.sh (resume after the 14:10 stop: src/manifest.py public-template assert)",
                        "final_chain_sha16": sha16("research/f3/final_chain.py"), "mac_step_sha16": sha16("research/f3/mac_step.sh"),
                        "src_manifest_sha16": sha16("src/manifest.py"),
                        "threads": "F3_MAXTH 3, F3CAP 3, TOTCAP 15 beside F2's fj CatBoosts (coordinator 10-09 13:5x); total-cap override after 300 s "
                                   "logged in logs/cap_override.log (none happened if absent)",
                        "cap_override_log": (Path("logs/cap_override.log").read_text(encoding="utf-8") if Path("logs/cap_override.log").exists() else None)},
    }
    (O / "final_build_manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in m.items() if k in ("spec", "tables_sha16", "file")}, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(m["fit_cache"]["final_root_load_runs"]["events_by_kind"]))


if __name__ == "__main__":
    main()
