# SPDX-License-Identifier: GPL-3.0-only
"""R17-A second pack for F3 (coordinator 10-09 02:0x, private notes research/2026-10-08-r17-a-adsb-fill.md section 4.5): 2026-06-11 becomes a no-data
day. The source day is short (both releases about 1.6 GB, 89 takeoff events, all at EDDM), so its final-test departures must look exactly as a
day without ADS-B looks in training: absent from the per-departure table (src/feats_adsb.py then gives every ab_* column NaN, ab_matched
included; src/adsb_corr.py / src/ctx_corr.py treat an absent row as not matched). All airports, EDDM included.
Writes, beside the first pack's files (left in place):
  final/adsb_v2_merged_v2.parquet   (+ .json)   adsb_v2_merged without the 2026-06-11 rows (the legs' --adsb input)
  final/adsb_v1nd_merged_v2.parquet (+ .json)   adsb_v1nd_merged without the 2026-06-11 rows (the corrector's --adsb input)
then adds a "pack2" block to final/manifest.json (rule, affected rows by airport, new sha16s) and refreshes its files_sha16, writes
final/README_R17A.txt, and touches research/r17_a/final_pack2.DONE.
    python -X utf8 research/r17_a/final_pack2_r17a.py        (sandbox root)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "final"
DAY = "2026-06-11"


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main() -> int:
    man_p = F / "manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8"))
    out = {}
    counts = None
    for src, dst in (("adsb_v2_merged", "adsb_v2_merged_v2"), ("adsb_v1nd_merged", "adsb_v1nd_merged_v2")):
        sp, dp = F / f"{src}.parquet", F / f"{dst}.parquet"
        if dp.exists():
            raise SystemExit(f"exists, refusing: {dp}")
        assert sha16(sp) == man["files_sha16"][f"{src}.parquet"], f"{src} changed since the first pack"
        t = pd.read_parquet(sp)
        drop = (t["day"] == DAY).to_numpy()
        g = t[drop].groupby("ADEP_mvt").agg(rows=("matched", "size"), matched_before=("matched", "sum"))
        if counts is None:
            counts = {k: {"rows": int(r.rows), "matched_before": int(r.matched_before)} for k, r in g.iterrows()}
        t[~drop].reset_index(drop=True).to_parquet(dp, index=False)
        meta = {"script": "research/r17_a/final_pack2_r17a.py", "source": f"final/{src}.parquet", "source_sha16": sha16(sp),
                "rule": f"rows of UTC day {DAY} removed (no-data day, as a training day without ADS-B)", "rows_removed": int(drop.sum()),
                "rows": int((~drop).sum()), "sha16": sha16(dp)}
        dp.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        out[f"{dst}.parquet"] = meta
        print(dst, meta, flush=True)
    assert out["adsb_v2_merged_v2.parquet"]["rows_removed"] == out["adsb_v1nd_merged_v2.parquet"]["rows_removed"]
    man["pack2"] = {"why": f"{DAY} is short at the source (both releases about 1.6 GB, 89 takeoff events, all at EDDM; median day 3,846); its final-test "
                           "departures are set exactly as a day without ADS-B is in training: absent from the per-departure table, so every ab_* "
                           "column is NaN (not ab_matched 0) and the corrector leaves them as not matched; all airports, EDDM included",
                    "no_data_days": [DAY], "rows_removed_by_airport": counts, "rows_removed": out["adsb_v2_merged_v2.parquet"]["rows_removed"],
                    "files": {k: v["sha16"] for k, v in out.items()},
                    "use": {"legs": "final/adsb_v2_merged_v2.parquet as <root>/cache/adsb_v2.parquet, then python -m src.feats_adsb feats_v27 --adsb adsb_v2 --mode raw",
                            "corrector": f"final/adsb_v1nd_merged_v2.parquet as <root>/cache/adsb_v1nd.parquet, --adsb_sha {out['adsb_v1nd_merged_v2.parquet']['sha16']}",
                            "rebuild_from_events": f"build_feats2.py / build_feats.py on final/events include {DAY}; drop that day's rows afterwards"},
                    "written_kst": dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")}
    readme = F / "README_R17A.txt"
    lines = ["R17-A final-phase ADS-B package (adsb.lol globe_history, ODbL 1.0; README.txt and LICENSE-*.txt are the source's own files).",
             "Private note: research/2026-10-08-r17-a-adsb-fill.md, sections 4.2 to 4.5.",
             "",
             "Use these (pack 2, current):",
             f"  adsb_v2_merged_v2.parquet   sha16 {out['adsb_v2_merged_v2.parquet']['sha16']}   legs: put as <root>/cache/adsb_v2.parquet, then",
             "                              python -m src.feats_adsb feats_v27 --adsb adsb_v2 --mode raw",
             f"  adsb_v1nd_merged_v2.parquet sha16 {out['adsb_v1nd_merged_v2.parquet']['sha16']}   corrector: put as <root>/cache/adsb_v1nd.parquet,",
             f"                              ctx_corr3 ... --adsb adsb_v1nd --adsb_sha {out['adsb_v1nd_merged_v2.parquet']['sha16']}",
             f"  Pack 2 removes the {out['adsb_v2_merged_v2.parquet']['rows_removed']} final-test departures of {DAY} (no-data day: 89 takeoff events, all EDDM),",
             "  so they read as a training day without ADS-B (all ab_* NaN), all airports.",
             "",
             "Also here (pack 1, unchanged): adsb_final.parquet = adsb_v1final.parquet (final-test rows only, 120 days), adsb_v2_merged, adsb_v1_merged,",
             "adsb_v1nd_merged (these three still hold 2026-06-11), events/<day>/ (extracts, matched.parquet, match_summary.json), points/<day>/ (Feb/Jun),",
             "prc_final.parquet, prc_final_stand.parquet (STAND_mvt for research/obs2/obs_rows.py), manifest.json (all sha16s, counts by month and airport).",
             "All four test months are matched on final_ranking's own jittered MVT time. Days switched from staging-0 to prod-0: 2026-06-06, 06-07, 06-23, 06-25."]
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    man["files_sha16"] = {str(p.relative_to(F)): sha16(p) for p in sorted(F.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    man_p.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "research" / "r17_a" / "final_pack2.DONE").write_text(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    print(json.dumps({"rows_removed_by_airport": counts, "manifest_sha16": sha16(man_p)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
