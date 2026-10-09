# SPDX-License-Identifier: GPL-3.0-only
"""Per-departure observation rows with positions (first point, stand stop) for the 124 route A days, and per airport-day
event counts. Read-only on D:/prc-adsb/out/<day>/events.parquet; matching is research/adsb/adsb_match.match_day unchanged.

Writes research/obs2/obs_rows.parquet (one row per PRC departure) and research/obs2/ev_counts.parquet.
Usage: .venv/Scripts/python.exe research/obs2/obs_rows.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "adsb"))
from adsb_match import match_day  # noqa: E402

OUT = Path("D:/prc-adsb/out")
KEEP = ["MVT_ID_mvt", "day", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "CALLSIGN_flt", "mvt", "how", "matched",
        "fm_seen", "fm_ok", "first_moving", "ev_seg_reason", "ev_t_first", "ev_first_lat", "ev_first_lon", "ev_first_gs",
        "ev_first_dist_m", "ev_first_src", "ev_first_stationary", "ev_stand_seen", "ev_stand_dur", "ev_stand_lat", "ev_stand_lon",
        "ev_t_fm", "ev_fm_gap", "ev_n_mid_stops", "ev_mid_stop_max", "ev_n_long_stops", "ev_t_fm_last", "ev_t_to", "ev_to_gap",
        "ev_n_pts", "ev_share_mlat", "ev_icao", "ev_typ", "ev_cs"]


def main():
    days = [d.strip() for d in open(ROOT / "research/adsb/days_routeA_all.txt", encoding="utf-8").read().split()]
    prc = pd.read_parquet(ROOT / "research/adsb/prc_days.parquet",
                          columns=["MVT_ID_mvt", "day", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "AIRCRAFT_TYPE_mvt",
                                   "RUNWAY_mvt", "STAND_mvt"])
    parts, cnt = [], []
    for d in sorted(set(days)):
        ev = pd.read_parquet(OUT / d / "events.parquet")
        m = match_day(prc[prc.day == d], ev)
        parts.append(m[[c for c in KEEP if c in m.columns]])
        used = set(m.loc[m.matched, "ev"].astype(int))
        ev["used"] = ev.index.isin(used)
        g = ev.groupby("apt").agg(events=("t_to", "size"), used=("used", "sum"), with_cs=("cs", lambda s: int((s.fillna("") != "").sum())))
        g["day"] = d
        cnt.append(g.reset_index())
    t = pd.concat(parts, ignore_index=True)
    assert t.MVT_ID_mvt.is_unique
    t.to_parquet(HERE / "obs_rows.parquet", index=False)
    pd.concat(cnt, ignore_index=True).to_parquet(HERE / "ev_counts.parquet", index=False)
    print(len(t), t.matched.mean(), t.fm_ok.mean())


if __name__ == "__main__":
    main()
