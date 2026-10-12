"""F3 Package S: the build record of 605c (and of its default 603), research/f3/final_manifest.py's helpers imported unchanged (its sha16 is
asserted), written as research/f3/out/final_build_manifest_605c.json in the final root. Chains A and B are the family's c-form chains (their
records are the family's own build record, research/f3/out/final_build_manifest_598c.json or _602c.json, referenced with its sha16); chain C is
package S's: the T form (603's chain C), the four-month corrector with the S fj extras, the BbF merge and its tails. Also recorded: F2's S tools as
cloned (research/f3/out/collect_s_<family>.json), the learner runs (research/f2/out/s_learn_<family>_<setting>.json), the averages
(research/f2/out/avg_s_*.json), the construction checks (research/f2/out/s_check_*.json), the final move (s_gate.py move), the checks of the c
form, the final file ff13 and the PC candidate.
    python -X utf8 research/f3/final_manifest_s.py <598c|l4> [--pc <json>]        (cwd = the final root)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN = "5524ea3f81b54858"  # research/f3/final_manifest.py as it wrote the 594c and 598c records
if hashlib.sha256((HERE / "final_manifest.py").read_bytes()).hexdigest()[:16] != PIN:
    raise SystemExit(f"research/f3/final_manifest.py is not {PIN}: stop")
sys.path.insert(0, str(HERE))
import final_manifest as fm  # noqa: E402

MID = "adsbgeondctxv3x50ws75"
FAMS = {"598c": dict(XP="", default="596", default_ff="ff5", final="598c", final_ff="ff9", fam_record="final_build_manifest_598c.json",
                     fam_logs="598c_", fam_kit="research/f3/kit_c.sh", T_S="m6r28sapcbrmfw4sv114", FJ="l3sapcbrmfw4"),
        "l4": dict(XP="x", default="600", default_ff="ff10", final="602c", final_ff="ff11", fam_record="final_build_manifest_602c.json",
                   fam_logs="l4c_", fam_kit="research/f3/kit_l4c.sh", T_S="m6r28xsapcbrmfw4sv114", FJ="l4sapcbrmfw4")}


def main():
    a = sys.argv[1:]
    if not a or a[0] not in FAMS:
        raise SystemExit(__doc__)
    fam, F = a[0], FAMS[a[0]]
    pc = fm.jl(a[a.index("--pc") + 1]) if "--pc" in a else None
    O = fm.O
    XP, T = F["XP"], F["T_S"]
    S603 = f"m6r28{XP}wincbrmfw4sv434tmc{MID}c3amfhgkcxncrdbawgpglgjaavskaps"
    S605 = f"m6r28{XP}wincbrmfw4sv434tmc{MID}c3rBbFamfhgkcxncrdbawgpglgjaavskaps"
    SDEF = f"m6r28{XP}wincbrmfw4sv434tmc{MID}c3amfhgkcxncrdbawgpglgjaavskap"
    SFIN = f"m6r28{XP}wincbrmfw4sv434tmc{MID}c3rBbFamfhgkcxncrdbawgpglgjaavskap"
    FS = f"s{fam}"
    files = lambda pat: sorted(str(p) for p in Path(".").glob(pat))  # noqa: E731
    jrec = lambda p: {"path": p, "sha16": fm.sha16(p), "content": fm.jl(p)}  # noqa: E731
    chainC = {
        "T": T,
        "T_form": {s or "c3": fm.man_entry(f"{T}{MID}c3{s}") for s in fm.TAILSUF},
        "four_month_L": {sc: {"cavg": {"path": f"cache/full_{T}{tg}c3_s0.parquet", "sha16": fm.sha16(f"cache/full_{T}{tg}c3_s0.parquet")},
                              "seeds_fit_cache": {k: fm.fit_line(Path("logs") / f"{FS}_c4m_C_{sc}_s{k}.log") for k in (0, 1, 2)}}
                         for sc, tg in (("0.75", "adsbgeondctxv3x50ws75m4F"), ("0.5", "adsbgeondctxv3x50wm4F"))},
        "fj_extra": {"tag": F["FJ"], "file": {"path": f"cache/fj_{F['FJ']}_s0.parquet", "sha16": fm.sha16(f"cache/fj_{F['FJ']}_s0.parquet")},
                     "pin": {"path": f"research/f2/out/s_fj_sha16_{fam}.txt", "sha16": fm.sha16(f"research/f2/out/s_fj_sha16_{fam}.txt"),
                             "content": (Path(f"research/f2/out/s_fj_sha16_{fam}.txt").read_text().strip()
                                         if Path(f"research/f2/out/s_fj_sha16_{fam}.txt").exists() else None)},
                     "blend": jrec(f"research/f2/out/s_fj_blend_{fam}.json")},
        "merge_notes": {s: {"path": f"notes/merge_{T}{MID}c3rBbF_{s}.json", "sha16": fm.sha16(f"notes/merge_{T}{MID}c3rBbF_{s}.json")} for s in ("val", "fwd", "full")},
        "jobs_file": {"path": f"notes/jobs_{T}{MID}c3rBbF.txt", "sha16": fm.sha16(f"notes/jobs_{T}{MID}c3rBbF.txt")},
        "c_form": {f"c3rBbF{s}": fm.man_entry(f"{T}{MID}c3rBbF{s}") for s in fm.TAILSUF},
    }
    rec = {
        "written": time.strftime("%Y-%m-%d %H:%M:%S KST"), "name": "605c", "spec": S605, "family": fam,
        "what": "the build record of package S's final 605c and its default 603, written by research/f3/final_manifest_s.py",
        "rebuild_of": (f"{F['final']}'s form ({F['default']} as its default) with chain C at package S's seed counts (coordinator 10-11 01:17; F2's "
                       "research/f2/s_spec.py): LightGBM ap and CatBoost ap seeds 1, 2 and chain C's own RealMLP seeds fitted in the full setting here "
                       "with F2's research/f2/s_learn.py, averaged with research/f2/avg_s.py; chain C's T form rebuilt with research/f3/final_chain_s.py "
                       "(val and fwd of every stage F2's S sandbox's, research/f3/collect_s.py; correctors at 0.75 from its fit cache) -> 603 = chainavg3 "
                       f"of {F['default']}'s chains A and B and the S chain C; the fj learners at the same seed counts, the S fj blend "
                       "(research/f2/fj_blend_s.py), chain C's four-month corrector on F2's fjF tables with those extras, the BbF merge and tails "
                       f"(research/f3/kit_s.sh) -> 605c = chainavg3 of {F['final']}'s c-form chains A and B and the S chain C"),
        "default_final": {"name": "603", "spec": S603, "manifest": fm.man_entry(S603), "file": str(O / "ff12" / "spirited-tiger_final.parquet"),
                          "sha16": fm.sha16(O / "ff12" / "spirited-tiger_final.parquet"), "checks": fm.jl(O / f"final_file_{S603}.json")},
        "family": {"default": {"name": F["default"], "manifest": fm.man_entry(SDEF), "file_sha16": fm.sha16(O / F["default_ff"] / "spirited-tiger_final.parquet")},
                   "final": {"name": F["final"], "manifest": fm.man_entry(SFIN), "file_sha16": fm.sha16(O / F["final_ff"] / "spirited-tiger_final.parquet")},
                   "build_record": {"path": str(O / F["fam_record"]), "sha16": fm.sha16(O / F["fam_record"]),
                                    "note": "chains A and B of 603 and 605c are this record's, untouched"}},
        "data": {"ranking": fm.sha16("data/ranking.parquet"), "submitting": fm.sha16("data/submitting.parquet")},
        "tables_sha16": {t: fm.sha16(f"cache/{t}.parquet") for t in ("all", "feats_v25", "feats_v27", "feats_v28", "ctx_v3", "adsb_v1nd", "geo_v1nd",
                                                                    "adsb_v1ndfjF", "geo_v1ndfjF", "antiwin_link_v1", "gkey_v1", "gkeyx_v1", "gkeyxn_v2",
                                                                    "artmm_v1", "artmm_cells_v1", "arthist_v1")},
        "code_sha16": {f: fm.sha16(f) for f in ["research/f3/phase8.sh", "research/f3/collect_s.py", "research/f3/final_chain_s.py", "research/f3/kit_s.sh",
                                                "research/f3/daily_candidate_s.py", "research/f3/final_manifest_s.py", "research/f3/final_chain.py",
                                                "research/f3/final_chain_l4.py", F["fam_kit"], "research/f3/mac_step.sh", "research/f3/final_file.py",
                                                "research/f3/test_move.py", "research/f2/merge_corr.py", "research/f2/ctx_corr4.py", "research/f2/ctx4_table.py",
                                                "research/f2/check_split.py", "research/f2/iii_boundary.py", "src/manifest.py", "src/aiu_shift.py",
                                                "tools/chainavg3.py", "research/f2/avg_s.py", "research/f2/s_spec.py", "research/f2/s_check.py",
                                                "research/f2/s_gate.py", "research/f2/fj_blend_s.py", "research/f2/s_learn.py", "research/f2/rm_smoke.py"]},
        "s_tools_and_clones": jrec(str(O / f"collect_s_{fam}.json")),
        "learner_runs": {p: jrec(p) for p in files(f"research/f2/out/s_learn_{fam}_*.json")},
        "fj_seeds_from_f2_run": jrec(str(O / "collect_s_598c_fj.json")) if fam == "598c" else None,
        "averages": {p: jrec(p) for p in files("research/f2/out/avg_s_*.json")},
        "construction_checks": {p: {"path": p, "sha16": fm.sha16(p), "pass": (fm.jl(p) or {}).get("pass")} for p in files("research/f2/out/s_check_*.json")},
        "chain_C": chainC,
        "fit_caches": {"T corrector": f"research/f3/fitcache_s{fam}", "four-month stage 1": f"research/f2/fitcache_4m_s{fam}"},
        "final_manifest": fm.man_entry(S605),
        "checks": {k: {"path": v, "sha16": fm.sha16(v), **({"pass": (fm.jl(v) or {}).get("pass"), "fails": (fm.jl(v) or {}).get("fails")} if fm.jl(v) is not None else {})}
                   for k, v in {"split_check (check finals: BbF against 603)": f"research/f3/out/split_check_{FS}.json",
                                "corrector-stage check": f"research/f3/out/split_check_{FS}_corr.json",
                                "twins C": f"research/f3/out/twins_{FS}_C.json",
                                "iii_boundary (BbF against 603)": f"research/f3/out/iii_boundary_{FS}.json",
                                "s_gate move (605c against the family final, MSE <= 150)": f"research/f3/out/s_gate_move_{fam}.json",
                                "test move 603 against the family default": "research/f3/out/test_move_603_vs_default.json",
                                "test move 605c against the family final": "research/f3/out/test_move_605c_vs_final.json"}.items()},
        "file": {"path": str(O / "ff13" / "spirited-tiger_final.parquet"), "sha16": fm.sha16(O / "ff13" / "spirited-tiger_final.parquet"),
                 "checks": fm.jl(O / f"final_file_{S605}.json")},
        "pc_candidate": pc,
        "cap_override_log": (Path("logs/cap_override.log").read_text(encoding="utf-8") if Path("logs/cap_override.log").exists() else None),
    }
    out = O / "final_build_manifest_605c.json"
    if out.exists():
        raise SystemExit(f"exists: {out}")
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    miss = [k for k, v in chainC["T_form"].items() if not v.get("exists", True)] + [k for k, v in chainC["c_form"].items() if not v.get("exists", True)]
    print(json.dumps({"out": str(out), "sha16": fm.sha16(out), "spec": S605, "file": rec["file"]["sha16"], "missing_stage_manifests": miss}, indent=1))


if __name__ == "__main__":
    main()
