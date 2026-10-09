# SPDX-License-Identifier: GPL-3.0-only
"""공개 METAR(Iowa Environmental Mesonet ASOS 내려받기, 가입 없음)를 10개 공항, 2025-01-01 ~ 2026-08-01 UTC 로 받는다.

결과: data/metar/<ICAO>.csv (station, valid(UTC), tmpf, dwpf, relh, sknt, gust, vsby, wxcodes, skyc1, skyl1, p01i)
"""
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "metar"
STATIONS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
FIELDS = ["tmpf", "dwpf", "relh", "sknt", "gust", "vsby", "wxcodes", "skyc1", "skyl1", "p01i"]


def url(st: str) -> str:
    q = [("station", st)] + [("data", f) for f in FIELDS] + [
        ("year1", "2025"), ("month1", "1"), ("day1", "1"), ("year2", "2026"), ("month2", "8"), ("day2", "1"),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"), ("missing", "empty"),
        ("trace", "empty"), ("direct", "no"), ("report_type", "3"), ("report_type", "4")]
    return "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + urllib.parse.urlencode(q)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    for st in STATIONS:
        p = OUT / f"{st}.csv"
        if p.exists() and p.stat().st_size > 100000:
            print(st, "exists", p.stat().st_size)
            continue
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url(st), timeout=300) as r:
                    data = r.read()
                p.write_bytes(data)
                print(st, "rows", data.count(b"\n") - 1, "bytes", len(data))
                break
            except Exception as e:  # noqa: BLE001
                print(st, "attempt", attempt, "failed", repr(e))
                time.sleep(10)
        time.sleep(3)


if __name__ == "__main__":
    main()
