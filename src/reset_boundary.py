# SPDX-License-Identifier: GPL-3.0-only
# 하루 쿼터 리셋 경계를 다른 팀 제출 기록(리더보드 API 전 페이지, PRC 조사 최종 스냅샷 10-02 14:52:54~14:53:21 KST)으로 판정한다.
# 날짜 경계를 바꿔 묶었을 때 팀별 하루 제출이 5 를 넘지 않는 경계가 리셋이다. 결과는 notes/reset_boundary.txt.
import collections
import datetime as dt
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
SRC = "notes/lb_all_2026-10-02.json"
d = json.load(open(SRC, encoding="utf-8"))
rows = d if isinstance(d, list) else next(v for v in d.values() if isinstance(v, list))
lines = [f"source {SRC}, submissions {len(rows)}"]
for name, shift in (("UTC", 0), ("KST", 9), ("CET", 1), ("CEST", 2)):
    c = collections.Counter((r["teamId"], (dt.datetime.fromisoformat(r["processedAt"].replace("Z", "+00:00"))
                                           + dt.timedelta(hours=shift)).date()) for r in rows)
    lines.append(f"{name}: max per team-day {max(c.values())}, team-days over 5: {sum(v > 5 for v in c.values())}")
open("notes/reset_boundary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
