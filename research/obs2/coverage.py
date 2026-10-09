# SPDX-License-Identifier: GPL-3.0-only
"""Coverage and failure modes of the route A extractor, by airport and month, from research/obs2/obs_rows.parquet.
Also builds the empirical stand map (median ADS-B stand-stop position per airport and PRC STAND_mvt, from rows whose
motion start was seen; no labels used) and the distance from the departure's stand to the first ADS-B point.
Writes research/obs2/coverage.json, stand_map.parquet, obs_geo.parquet.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def dist_m(la1, lo1, la2, lo2):
    k = np.cos(np.radians((la1 + la2) / 2))
    return np.hypot((lo1 - lo2) * 111200.0 * k, (la1 - la2) * 111200.0)


def main():
    t = pd.read_parquet(HERE / "obs_rows.parquet")
    ev = pd.read_parquet(HERE / "ev_counts.parquet")
    t["ym"] = t.day.str[:7]
    t["stand_id"] = t.STAND_mvt.fillna("").astype(str).str.strip()
    t["m0"] = t.matched & ~t.fm_ok
    t["f_first_moving"] = t.matched & ~t.fm_seen & (t.ev_seg_reason != "landing")
    t["f_gap_at_stand"] = t.fm_seen & ~t.fm_ok
    t["f_landing_nostand"] = t.matched & ~t.fm_seen & (t.ev_seg_reason == "landing")
    t["tow_like"] = t.fm_ok & ((t.ev_n_long_stops >= 2) | ((t.ev_t_fm_last - t.ev_t_fm) > 600))
    t["long_mid_stop"] = t.fm_ok & (t.ev_mid_stop_max >= 300)
    # empirical stand map from rows with a seen stand stop (fm_seen), per airport and PRC stand id
    s = t[t.fm_seen & t.ev_stand_lat.notna() & (t.stand_id != "")]
    g = s.groupby(["ADEP_mvt", "stand_id"])
    sm = pd.DataFrame({"n": g.size(), "lat": g.ev_stand_lat.median(), "lon": g.ev_stand_lon.median()})
    s = s.join(sm[["lat", "lon"]], on=["ADEP_mvt", "stand_id"])
    s["dev"] = dist_m(s.ev_stand_lat, s.ev_stand_lon, s.lat, s.lon)
    sm["dev_med"] = s.groupby(["ADEP_mvt", "stand_id"]).dev.median()
    sm["ok"] = (sm.n >= 3) & (sm.dev_med <= 60)
    sm.reset_index().to_parquet(HERE / "stand_map.parquet", index=False)
    t = t.join(sm.loc[sm.ok, ["lat", "lon"]].rename(columns={"lat": "st_lat", "lon": "st_lon"}), on=["ADEP_mvt", "stand_id"])
    t["d_stand_first"] = dist_m(t.st_lat, t.st_lon, t.ev_first_lat, t.ev_first_lon)
    t.to_parquet(HERE / "obs_geo.parquet", index=False)
    rows = []
    for (ap, ym), x in t.groupby(["ADEP_mvt", "ym"]):
        m = x[x.matched]
        e = ev[(ev.apt == ap) & (ev.day.str[:7] == ym)]
        m0 = x[x.m0]
        rows.append(dict(
            ap=ap, ym=ym, n=len(x), stand_fill=round(100 * (x.stand_id != "").mean(), 1),
            matched=round(100 * x.matched.mean(), 1), fm_ok=round(100 * x.fm_ok.mean(), 1), fm_seen=round(100 * x.fm_seen.mean(), 1),
            first_moving=round(100 * x.f_first_moving.mean(), 1), gap_at_stand=round(100 * x.f_gap_at_stand.mean(), 1),
            landing_nostand=round(100 * x.f_landing_nostand.mean(), 1),
            ev_per_dep=round(len(e) and e.events.sum() / len(x), 3), ev_unused=int(e.events.sum() - e.used.sum()) if len(e) else 0,
            tow_like_fm=round(100 * x.tow_like.sum() / max(x.fm_ok.sum(), 1), 1),
            longmid_fm=round(100 * x.long_mid_stop.sum() / max(x.fm_ok.sum(), 1), 1),
            mlat_med=round(float(m.ev_share_mlat.median()), 3) if len(m) else None,
            first_dist_m0_med=round(float(m0.ev_first_dist_m.median()), 0) if len(m0) else None,
            first_gs_m0_med=round(float(m0.ev_first_gs.median()), 1) if len(m0) else None,
            m0_stand_known=round(100 * m0.st_lat.notna().mean(), 1) if len(m0) else None,
            m0_dsf_med=round(float(m0.d_stand_first.median()), 0) if m0.st_lat.notna().any() else None,
            m0_dsf_le150=round(100 * (m0.d_stand_first <= 150).sum() / max(m0.st_lat.notna().sum(), 1), 1) if len(m0) else None,
        ))
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(R.to_string(index=False))
    tot = t.groupby(t.day.str[:4]).agg(n=("matched", "size"), matched=("matched", "mean"), fm_ok=("fm_ok", "mean"),
                                         first_moving=("f_first_moving", "mean"), gap_at_stand=("f_gap_at_stand", "mean"),
                                         landing_nostand=("f_landing_nostand", "mean"))
    print(tot)
    print("stand map:", int(sm.ok.sum()), "stands ok of", len(sm), "| per airport ok:", sm[sm.ok].groupby(level=0).size().to_dict())
    (HERE / "coverage.json").write_text(json.dumps({"by_airport_month": rows, "by_year": tot.reset_index().to_dict("records")},
                                                   indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
