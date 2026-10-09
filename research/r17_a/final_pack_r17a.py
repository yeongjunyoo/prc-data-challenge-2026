# SPDX-License-Identifier: GPL-3.0-only
"""R17-A: the final-phase ADS-B package for F3 (private notes research/2026-10-08-r17-a-adsb-fill.md section 4, 10-08 22:2x handoff), built in the
Mac sandbox <work>/r17a_sbx into final/ (never into F3's root).

Rule for all four test months (2026-01, 02, 06, 07): every departure of data/final_phase/final_ranking.parquet is matched again on the final
file's own (jittered) MVT time with research/adsb/adsb_match.match_day unchanged (no reuse of the public-phase matches).
Steps (each refuses to overwrite a finished output):
  1. check that the 58 days 2026-02-01..28 and 2026-06-01..30 are extracted in <adsb>/r17a/out/<day>/ (meta.json complete, gz/archive >= 0.62)
  2. final/events/<day>/{events.parquet, meta.json} for all 120 test days (APFS clones: Jan/Jul from <adsb>/rb/events, Feb/Jun from
     <adsb>/r17a/out), final/points/<day>/points.parquet for the 58 new days (Jan/Jul points stay on the PC, <pc adsb out>/<day>/)
  3. final/prc_final.parquet (research/r17_a/prep_final_r17a.py output, the matcher's columns) and final/prc_final_stand.parquet (MVT_ID_mvt,
     STAND_mvt of the same departures, for research/obs2/obs_rows.py)
  4. research/adsb/adsb_match.py --prc final/prc_final.parquet --out final/events <120 days> -> final/events/<day>/matched.parquet and
     final/events/match_summary.json (the per-day match outputs)
  5. research/adsb/build_feats2.py --name adsb_final --prc final/prc_final.parquet --out-dir final/events (adsb_v2-style table, the legs' ab_*
     input) and research/adsb/build_feats.py --name adsb_v1final --prc final/prc_final.parquet --out-dir final/events (adsb_v1-style table, the
     corrector's stage-1 input); both copied from cache/ to final/
  5b. research/r17_a/merge_final_r17a.py: final/adsb_v2_merged (cache/adsb_v2's 2025 rows + adsb_final) and final/adsb_v1_merged (cache/adsb_v1's
     2025 rows + adsb_v1final), and final/adsb_v1nd_merged (build_nd.py's four columns emptied in adsb_v1_merged)
  6. final/manifest.json: sha16 of every file, per-month and per-airport departures and matches, the rule, code sha16s and F3's build commands
    python -X utf8 research/r17_a/final_pack_r17a.py [--partial] [--out <dir>]   (sandbox root; --partial skips step 1's count, --out writes elsewhere for a test)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HOME = Path(os.environ.get("PRC_ADSB_HOME", "."))  # ADS-B working folder (rb/events, r17a/out)
NEW, OLD = HOME / "r17a" / "out", HOME / "rb" / "events"
F = ROOT / (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "final")
PY = str(HOME / "venv311" / "bin" / "python")


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def days_of(month: str) -> list[str]:
    d = dt.date.fromisoformat(month + "-01")
    out = []
    while d.strftime("%Y-%m") == month:
        out.append(d.isoformat())
        d += dt.timedelta(1)
    return out


def clone(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        assert sha16(dst) == sha16(src), f"differs: {dst}"
        return
    subprocess.run(["cp", "-c", str(src), str(dst)], check=True)


def main() -> int:
    partial = "--partial" in sys.argv[1:]
    new_days = days_of("2026-02") + days_of("2026-06")
    old_days = days_of("2026-01") + days_of("2026-07")
    bad, short = [], []
    for d in new_days:
        mf = NEW / d / "meta.json"
        if not mf.exists():
            bad.append(d)
            continue
        m = json.loads(mf.read_text())
        if not m.get("complete", m.get("mode") == "disk") or m["gz_bytes"] / m["bytes"] < 0.62:
            bad.append(d)
        if m.get("n_events", 0) < 1000:
            short.append((d, m.get("n_events")))
    if bad and not partial:
        raise SystemExit(f"Feb/Jun days not ready: {bad}")
    use_new = [d for d in new_days if d not in bad]
    print(f"new days ready {len(use_new)} of 58, not ready {bad}, days with < 1000 events {short}", flush=True)
    for d in old_days:
        for fn in ("events.parquet", "meta.json"):
            clone(OLD / d / fn, F / "events" / d / fn)
    for d in use_new:
        for fn in ("events.parquet", "meta.json"):
            clone(NEW / d / fn, F / "events" / d / fn)
        clone(NEW / d / "points.parquet", F / "points" / d / "points.parquet")
    for fn in ("README.txt", "LICENSE-ODbL.txt", "LICENSE-cc0.txt"):
        clone(NEW / use_new[0] / fn, F / fn)
    clone(ROOT / "research" / "r17_a" / "prc_final.parquet", F / "prc_final.parquet")
    st = pd.read_parquet(ROOT / "data" / "final_phase" / "final_ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "STAND_mvt"])
    st = st[st.PHASE_mvt == "DEP"][["MVT_ID_mvt", "STAND_mvt"]].sort_values("MVT_ID_mvt").reset_index(drop=True)
    st.to_parquet(F / "prc_final_stand.parquet", index=False)
    days = sorted(old_days + use_new)
    subprocess.run([PY, "-X", "utf8", "research/adsb/adsb_match.py", "--prc", str(F / "prc_final.parquet"), "--out", str(F / "events"), *days],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    for script, name in (("research/adsb/build_feats2.py", "adsb_final"), ("research/adsb/build_feats.py", "adsb_v1final")):
        subprocess.run([PY, "-X", "utf8", script, "--name", name, "--prc", str(F / "prc_final.parquet"), "--out-dir", str(F / "events")],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        for ext in ("parquet", "json"):
            src, dst = ROOT / "cache" / f"{name}.{ext}", F / f"{name}.{ext}"
            dst.write_bytes(src.read_bytes())
    # 5b. merged tables (the public-phase 2025 rows + all final-test rows), and the corrector's nd form of the v1 one (build_nd.py's four columns)
    for old, want, fin, out in (("adsb_v2", "8ee80c46165c4ef9", "adsb_final", "adsb_v2_merged"), ("adsb_v1", "2bb0952d93192201", "adsb_v1final", "adsb_v1_merged")):
        assert sha16(ROOT / "cache" / f"{old}.parquet") == want, old
        for ext in ("parquet", "json"):
            (F / f"{out}.{ext}").unlink(missing_ok=True)
        subprocess.run([PY, "-X", "utf8", "research/r17_a/merge_final_r17a.py", str(ROOT / "cache" / f"{old}.parquet"), str(F / f"{fin}.parquet"),
                        str(F / f"{out}.parquet")], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    import pyarrow as pa
    import pyarrow.parquet as pq
    tv = pq.read_table(F / "adsb_v1_merged.parquet")
    for c in ("ev_n_pts", "ev_to_gap", "ev_max_gap_after_fm", "ev_share_mlat"):
        i = tv.schema.get_field_index(c)
        assert i >= 0, c
        tv = tv.set_column(i, pa.field(c, pa.float64()), pa.nulls(tv.num_rows, pa.float64()))
    pq.write_table(tv, F / "adsb_v1nd_merged.parquet")
    t = pd.read_parquet(F / "adsb_final.parquet")
    t["ym"] = t.day.str[:7]
    g = t.groupby(["ym", "ADEP_mvt"])
    cnt = pd.DataFrame({"departures": g.size(), "matched": g.matched.sum(), "fm_seen": g.fm_seen.sum(), "fm_ok": g.fm_ok.sum()})
    cnt["match_rate"] = (cnt.matched / cnt.departures).round(4)
    prc = pd.read_parquet(F / "prc_final.parquet", columns=["MVT_ID_mvt", "day"])
    prc_by_month = prc.day.str[:7].value_counts().sort_index().to_dict()
    files = {str(p.relative_to(F)): sha16(p) for p in sorted(F.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    code = {s: sha16(ROOT / s) for s in ("research/adsb/adsb_match.py", "research/adsb/build_feats.py", "research/adsb/build_feats2.py",
                                         "research/r17_a/adsb_day_r17a.py", "research/r17_a/prep_final_r17a.py", "research/r17_a/final_pack_r17a.py")}
    man = {"what": "R17-A final-phase ADS-B package (adsb.lol globe_history, ODbL 1.0): per-day extracts and matches of the 120 test days of "
                   "data/final_phase/final_ranking.parquet, and the two per-departure tables",
           "rule_jan_jul": "re-matched on final_ranking's own jittered MVT time for all four months (one code path); measured 10-08 22:09: "
                           "Jan/Jul on final vs ranking.parquet times: matched 250,564 vs 250,562, 10 rows differ, 1 of 250,558 both-matched rows "
                           "took another event, est_mvt/est_first/est_mvt_last/dt_to move by exactly the per-row MVT jitter",
           "final_ranking_sha16": sha16(ROOT / "data" / "final_phase" / "final_ranking.parquet"),
           "departures_by_month": prc_by_month, "days": len(days), "days_by_month": pd.Series([d[:7] for d in days]).value_counts().sort_index().to_dict(),
           "new_days_not_ready": bad, "days_under_1000_events": short,
           "by_month_airport": {f"{k[0]} {k[1]}": {kk: (float(v) if kk == "match_rate" else int(v)) for kk, v in r.items()} for k, r in cnt.iterrows()},
           "by_month": {k: {"departures": int(v.departures.sum()), "matched": int(v.matched.sum()), "match_rate": round(float(v.matched.sum() / v.departures.sum()), 4)}
                        for k, v in cnt.groupby(level=0)},
           "files_sha16": files, "code_sha16": code,
           "layout": {"events/<day>/events.parquet, meta.json": "adsb_day extracts (Jan/Jul = the public-phase days, Feb/Jun new); the --out-dir of "
                                                               "build_feats.py, build_feats2.py and research/obs2/obs_rows.py",
                      "events/<day>/matched.parquet, events/match_summary.json": "adsb_match.py per-day matches on prc_final.parquet",
                      "points/<day>/points.parquet": "Feb/Jun ground points (Jan/Jul points: PC <pc adsb out>/<day>/points.parquet)",
                      "prc_final.parquet": "final_ranking DEP rows with the matcher's columns (research/r17_a/prep_final_r17a.py)",
                      "prc_final_stand.parquet": "MVT_ID_mvt, STAND_mvt of the same rows (obs_rows.py needs STAND_mvt)",
                      "adsb_final.parquet": "adsb_v2-style table (build_feats2.py): one row per final-test departure of the 120 days",
                      "adsb_v1final.parquet": "adsb_v1-style table (build_feats.py): same rows and values; the corrector's stage-1 format",
                      "adsb_v2_merged.parquet": "cache/adsb_v2 (8ee80c46165c4ef9) 2025 rows (178 days, feats_v27's training coverage) + adsb_final "
                                                "(research/r17_a/merge_final_r17a.py): src.feats_adsb --adsb input for a feats_v27-final",
                      "adsb_v1_merged.parquet": "cache/adsb_v1 (2bb0952d93192201) 2025 ruler rows + adsb_v1final: the corrector's adsb_v1 for the final",
                      "adsb_v1nd_merged.parquet": "adsb_v1_merged with build_nd.py's four columns emptied (ev_n_pts, ev_to_gap, ev_max_gap_after_fm, "
                                                  "ev_share_mlat): the corrector's --adsb adsb_v1nd for the final; geo_v1nd is F3's (obs_rows.py on events/)"},
           "written_kst": dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")}
    (F / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(cnt.to_string(), flush=True)
    print(json.dumps({k: man[k] for k in ("departures_by_month", "days_by_month", "by_month", "new_days_not_ready", "days_under_1000_events")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
