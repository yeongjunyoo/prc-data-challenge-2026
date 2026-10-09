# SPDX-License-Identifier: GPL-3.0-only
# 리더보드 필드 판독(G7 대시보드). 공개 API 기본 응답은 팀 순위가 아니라 점수순 제출 50행이다(10-02 16:53 실측: 50행이 6팀).
# 순위는 원문대로 팀별 최고 RMSE 로 매긴다. 전 페이지를 받아 notes/lb/ 에 원본을 두고 notes/field.csv 에 한 줄을 더한다.
import csv
import datetime as dt
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

sys.stdout.reconfigure(encoding="utf-8")
BASE = "https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c"
TEAM = "spirited-tiger"
ROOT = Path(__file__).resolve().parent.parent
MARKS = (1, 3, 10, 20, 50)


def fetch() -> list[dict]:
    items, cursor = [], None
    while True:
        q = {"limit": 200} | ({"cursor": cursor} if cursor else {})
        req = urllib.request.Request(f"{BASE}/leaderboard?{urllib.parse.urlencode(q)}", headers={"User-Agent": "Mozilla/5.0"})
        d = json.loads(urllib.request.urlopen(req, timeout=60).read())
        items += d.get("items", [])
        cursor = d.get("nextCursor")
        if not d.get("items") or not cursor:
            return items
        time.sleep(0.3)


def team_best(items: list[dict]) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for r in items:
        if r.get("score") is not None:
            best[r["teamName"]] = min(best.get(r["teamName"], float("inf")), r["score"])
    return sorted(best.items(), key=lambda x: x[1])


def main() -> None:
    now = dt.datetime.now(ZoneInfo("Asia/Seoul"))
    items = fetch()
    out = ROOT / "notes" / "lb" / f"{now:%Y-%m-%d_%H%M}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    # 기본 응답(인자 없음) 원본도 둔다. 표시가 팀 순위가 아니라 점수순 제출이라는 근거다
    req = urllib.request.Request(f"{BASE}/leaderboard", headers={"User-Agent": "Mozilla/5.0"})
    default = json.loads(urllib.request.urlopen(req, timeout=60).read())
    out.with_name(out.stem + "_default.json").write_text(json.dumps(default, ensure_ascii=False), encoding="utf-8")
    print(f"default page: rows {len(default.get('items', []))}, teams {len({r['teamName'] for r in default.get('items', [])})}")
    rk = team_best(items)
    ours = next(((i + 1, s) for i, (t, s) in enumerate(rk) if t == TEAM), (None, None))
    row = {"fetched_kst": f"{now:%Y-%m-%d %H:%M}", "submissions": len(items), "teams": len(rk)}
    row |= {f"rank{m}": (rk[m - 1][1] if m <= len(rk) else "") for m in MARKS}
    row |= {"ours_rank": ours[0] or "", "ours_best": ours[1] or "",
            "ours_top_pct": round(100 * ours[0] / len(rk), 1) if ours[0] else ""}
    f = ROOT / "notes" / "field.csv"
    new = not f.exists()
    with f.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, ensure_ascii=False))
    print("top10:", [(i + 1, t, s) for i, (t, s) in enumerate(rk[:10])])


if __name__ == "__main__":
    main()
