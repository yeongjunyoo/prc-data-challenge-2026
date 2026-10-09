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
                            "10-08 22:5x)",
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
                        "threads": "F3_MAXTH 3, F3CAP 3, TOTCAP 15 beside F2's fj CatBoosts (10-09 13:5x); total-cap override after 300 s "
                                   "logged in logs/cap_override.log (none happened if absent)",
                        "cap_override_log": (Path("logs/cap_override.log").read_text(encoding="utf-8") if Path("logs/cap_override.log").exists() else None)},
    }
    (O / "final_build_manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in m.items() if k in ("spec", "tables_sha16", "file")}, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(m["fit_cache"]["final_root_load_runs"]["events_by_kind"]))


# The build record of a merged final (10-10 05:0x, after the adversarial review: final_build_manifest.json describes 590, not the file
# that will be submitted). One record per name, beside the 590 one: research/f3/out/final_build_manifest_<name>.json. The legs, fit cache and
# one-off changes of the s50 build stay in final_build_manifest.json, which this record names; here: the default chains it merges into, the
# four-month corrector, the merge and its checks, every stage manifest the file comes from (sha16 of the manifest and of the files it names),
# the final file and its checks, and the PC candidate (names and sha256 from the PC's SHA256SUMS, passed as a json file).
#   python -X utf8 research/f3/final_manifest.py <spec> --name 594b|598b [--pc <json>]
MID = "adsbgeondctxv3x50ws75"
TAILSUF = ["", "am", "amfh", "amfhgkc", "amfhgkcx", "amfhgkcxncr", "amfhgkcxncrdb", "amfhgkcxncrdbaw", "amfhgkcxncrdbawgp", "amfhgkcxncrdbawgpgl",
           "amfhgkcxncrdbawgpglgja"]
PROFILES = {
    "594b": dict(
        variant="Bb", ff="ff4", default="592", default_ff="ff2", stem="m6r27", feats="feats_v27",
        bases={"A": "m6r27wincbrmfw4sv434tmc", "B": "m6r27skcbrmfw4sv434", "C": "m6r27apcbrmfw4sv114"},
        rebuild_of=("531 (v36, daily LB 228.1704) rebuilt on final_ranking as the Default 592 (research/f3/final_chain.py --cfg s75: 514's legs and "
                    "blends from the s50 build, correctors at scale2 0.75 loaded from the fit cache, 531's tails), then option B variant Bb with F2's "
                    "kit (research/f2/send_final/kit_final.sh): Jan/Jul rows T 0.75 (592's corrector), Feb/Jun rows the four-month stage 2 L of "
                    "research/f2/ctx_corr4.py, unmatched 0.75, matched 0.5, merged after cavg by research/f2/merge_corr.py, then 531's tails and "
                    "chainavg3 on the merged tag"),
        chains=["final_chain_s75_A.json", "final_chain_s75_B.json", "final_chain_s75_C.json"],
        kit="research/f2/send_final/kit_final.sh", kit_pins="research/f2/send_final/sha16.txt", logs="f2k_", c4m_cache="research/f2/fitcache_4m",
        drivers=["research/f3/phase4a.sh", "research/f3/phase4b.sh", "research/f3/phase4c.sh"],
        checks={"split_check (F2's check finals: Ba and Bb against 592)": "research/f2/out/split_check.json",
                "corrector-stage check log": "logs/f2k_check_corr.log", "check finals log": "logs/f2k_check_finals.log",
                "twins A": "research/f2/out/twins_A.json", "twins B": "research/f2/out/twins_B.json", "twins C": "research/f2/out/twins_C.json",
                "iii_boundary (F2, f2_run)": str(Path(__import__("os").environ.get("PRC_WORK", ".")) / "f2_run/research/f2/out/iii_boundary.json")},
        extra={"s75_correctors_clone": "research/f3/out/collect_s75.json"}),
    "598b": dict(
        variant="Bb", ff="ff7", default="596", default_ff="ff5", stem="m6r28", feats="feats_v28",
        bases={"A": "m6r28wincbrmfw4sv434tmc", "B": "m6r28skcbrmfw4sv434", "C": "m6r28apcbrmfw4sv114"},
        rebuild_of=("L3S (570's configuration) rebuilt on final_ranking as 596 (research/f3/final_chain.py --cfg l3s75: the nine L3 LightGBM legs "
                    "refit in the full setting on this root's feats_v28 with R17-A's arguments, their G bands and seed averages, 514's other "
                    "learners, correctors at scale2 0.75 loaded from L3's fit cache, 570's tails), then option B variant Bb with research/f3/kit_l3.sh "
                    "(F2's kit with L3's names): Jan/Jul rows T 0.75 (596's corrector), Feb/Jun rows the four-month stage 2 L on L3's base with the "
                    "L3 leave-Feb/Jun-out extras, unmatched 0.75, matched 0.5, merged after cavg, then 570's tails and chainavg3 on the merged tag"),
        chains=["final_chain_l3_A.json", "final_chain_l3_B.json", "final_chain_l3_C.json"],
        kit="research/f3/kit_l3.sh", kit_pins="research/f3/out/l3_fj_sha16.txt", logs="l3k_", c4m_cache="research/f2/fitcache_4m_l3",
        drivers=["research/f3/phase5.sh", "research/f3/phase5b.sh"],
        checks={"split_check after Bb (check finals: Bb against 596)": "research/f3/out/split_check_l3_afterBb.json",
                "split_check (check finals: Bb and Ba against 596)": "research/f3/out/split_check_l3.json",
                "corrector-stage check": "research/f3/out/split_check_l3_corr.json",
                "twins A": "research/f3/out/twins_l3_A.json", "twins B": "research/f3/out/twins_l3_B.json", "twins C": "research/f3/out/twins_l3_C.json",
                "iii_boundary (Bb against 596)": "research/f3/out/iii_boundary_l3_Bb.json"},
        extra={"feats_v28_check": "research/f3/out/feats_v28_final.json", "feats_v28_check_first_run": "research/f3/out/feats_v28_final.try1.json",
               "clones_from_r17a_sbx_and_f2_run": "research/f3/out/collect_l3.json", "legs": "research/f3/out/orch_l3legs.json",
               "fj_blends": "research/f3/out/fj_blends_l3.json"}),
}


def man_entry(spec: str) -> dict:
    p = Path("preds") / f"manifest_{spec}.json"
    m = jl(p)
    if m is None:
        return {"manifest": str(p), "exists": False}
    return {"manifest": str(p), "manifest_sha16": sha16(p), **{k: (m.get(k) or {}).get("sha") for k in ("test", "holdout", "forward", "es_valid_ids")},
            "feats": m.get("feats"), "feats_sha": m.get("feats_sha")}


def fit_line(log: Path):
    if not log.exists():
        return None
    xs = [x.strip() for x in log.read_text(encoding="utf-8", errors="replace").splitlines() if "fit cache" in x]
    return xs[-1] if xs else None


def variant_main(argv: list[str]) -> None:
    spec, name = argv[0], argv[argv.index("--name") + 1]
    pc = jl(argv[argv.index("--pc") + 1]) if "--pc" in argv else None
    P = PROFILES[name]
    V = P["variant"]
    chains = {}
    for X, T in P["bases"].items():
        chains[X] = {
            "base": T,
            "default_c3 (the merge's T, and its val/fwd source)": man_entry(f"{T}{MID}c3"),
            "four_month_L": {sc: {"cavg": {"path": f"cache/full_{T}{tg}c3_s0.parquet", "sha16": sha16(f"cache/full_{T}{tg}c3_s0.parquet")},
                                  "seeds_fit_cache": {k: fit_line(Path("logs") / f"{P['logs']}c4m_{X}_{sc}_s{k}.log") for k in (0, 1, 2)}}
                             for sc, tg in (("0.75", "adsbgeondctxv3x50ws75m4"), ("0.5", "adsbgeondctxv3x50wm4"))},
            "merge_notes": {s: {"path": f"notes/merge_{T}{MID}c3r{V}_{s}.json", "sha16": sha16(f"notes/merge_{T}{MID}c3r{V}_{s}.json")}
                            for s in ("val", "fwd", "full")},
            "jobs_file": {"path": f"notes/jobs_{T}{MID}c3r{V}.txt", "sha16": sha16(f"notes/jobs_{T}{MID}c3r{V}.txt")},
            "stage_manifests": {f"c3r{V}{s}": man_entry(f"{T}{MID}c3r{V}{s}") for s in TAILSUF},
        }
    final_m = man_entry(spec)
    ffp = O / P["ff"] / "spirited-tiger_final.parquet"
    rec = {
        "written": time.strftime("%Y-%m-%d %H:%M:%S KST"),
        "name": name, "spec": spec,
        "what": f"the build record of {name} (variant {V}), written beside final_build_manifest.json (590) by research/f3/final_manifest.py --name {name}",
        "rebuild_of": P["rebuild_of"],
        "default_final": {"name": P["default"], "file": str(O / P["default_ff"] / "spirited-tiger_final.parquet"),
                          "sha16": sha16(O / P["default_ff"] / "spirited-tiger_final.parquet")},
        "s50_build_record": {"path": str(O / "final_build_manifest.json"), "sha16": sha16(O / "final_build_manifest.json"),
                             "note": "legs, fit cache, CatBoost threads, XGBoost placeholder, TabM device and the one-off changes of the s50 build"},
        "data": {"ranking": sha16("data/ranking.parquet"), "submitting": sha16("data/submitting.parquet"),
                 "note": "final_ranking 4b0692a792846079, final_submitting b3445d80e29bf599 (organizer, 10-08 11:57 UTC)"},
        "tables_sha16": {t: sha16(f"cache/{t}.parquet") for t in ("all", "feats_v25", P["feats"]) + (("adsb_v3",) if P["feats"] == "feats_v28" else ())
                         + ("ctx_v3", "adsb_v1nd", "geo_v1nd", "antiwin_link_v1", "gkey_v1", "gkeyx_v1", "gkeyxn_v2", "artmm_v1", "artmm_cells_v1",
                            "arthist_v1")},
        "code_sha16": {f: sha16(f) for f in [P["kit"], P["kit_pins"], "research/f2/merge_corr.py", "research/f2/ctx_corr4.py", "research/f2/ctx4_table.py",
                                             "research/f2/check_split.py", "research/f2/iii_boundary.py", "research/f3/final_chain.py",
                                             "research/f3/mac_step.sh", "research/f3/final_file.py", "src/manifest.py", "src/aiu_shift.py",
                                             "tools/chainavg3.py"] + P["drivers"]},
        "default_chain_states": {c: jl(O / c) for c in P["chains"]},
        "four_month_fit_cache": P["c4m_cache"],
        "chains": chains,
        "final_manifest": final_m,
        "stage_manifests_count": sum(1 for X in chains.values() for e in X["stage_manifests"].values() if e.get("exists", True))
                                 + sum(1 for X in chains.values() if X["default_c3 (the merge's T, and its val/fwd source)"].get("exists", True))
                                 + (1 if final_m.get("exists", True) else 0),
        "checks": {k: {"path": v, "sha16": sha16(v), **({"pass": (jl(v) or {}).get("pass"), "fails": (jl(v) or {}).get("fails")}
                                                        if v.endswith(".json") and jl(v) is not None else {})}
                   for k, v in P["checks"].items()},
        "records": {k: {"path": v, "sha16": sha16(v), "content": jl(v)} for k, v in P["extra"].items()},
        "file": {"path": str(ffp), "sha16": sha16(ffp), "checks": jl(O / f"final_file_{spec}.json")},
        "pc_candidate": pc,
        "cap_override_log": (Path("logs/cap_override.log").read_text(encoding="utf-8") if Path("logs/cap_override.log").exists() else None),
    }
    out = O / f"final_build_manifest_{name}.json"
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"out": str(out), "sha16": sha16(out), "spec": spec, "file": rec["file"]["sha16"], "stage_manifests": rec["stage_manifests_count"],
                      "missing": [e["manifest"] for X in chains.values() for e in X["stage_manifests"].values() if not e.get("exists", True)]
                      + [X["default_c3 (the merge's T, and its val/fwd source)"]["manifest"] for X in chains.values()
                         if not X["default_c3 (the merge's T, and its val/fwd source)"].get("exists", True)]}, indent=1))


if __name__ == "__main__":
    if "--name" in sys.argv:
        variant_main(sys.argv[1:])
    else:
        main()
