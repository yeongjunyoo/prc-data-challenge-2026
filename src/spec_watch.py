# SPDX-License-Identifier: GPL-3.0-only
# 요강 쪽 변경 감시. 순위 쪽 원문: “We will post urgent and important news here”. 다섯 쪽의 본문 글자를 해시해
# notes/spec_hash.json 과 비교하고, 바뀐 쪽은 notes/spec/ 에 새 글자를 두고 「원문 변경」 줄과 차이 몇 줄을 찍는다.
# 판단은 하지 않는다. 다음 작업에서 우리가 그 쪽을 읽고 MISSION 을 고친다.
import datetime as dt
import difflib
import hashlib
import json
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
SITE = "https://prc-data-challenge-2026.netlify.app/"
PAGES = ["index.html", "ranking.html", "data.html", "eligibility.html", "rationale.html"]
STATE = ROOT / "notes" / "spec_hash.json"
SNAP = ROOT / "notes" / "spec"


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip, self.depth_main, self.out = 0, 0, []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1
        if tag == "main":
            self.depth_main += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self.skip:
            self.skip -= 1
        if tag == "main" and self.depth_main:
            self.depth_main -= 1

    def handle_data(self, data):
        if not self.skip and self.depth_main and data.strip():
            self.out.append(" ".join(data.split()))


def page_text(name: str) -> str:
    req = urllib.request.Request(SITE + name, headers={"User-Agent": "Mozilla/5.0"})
    p = _Text()
    p.feed(urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace"))
    return "\n".join(p.out)


def main() -> None:
    now = dt.datetime.now(ZoneInfo("Asia/Seoul"))
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    SNAP.mkdir(parents=True, exist_ok=True)
    changed = []
    for name in PAGES:
        try:
            text = page_text(name)
        except Exception as e:  # 한 쪽이 안 받아져도 나머지는 본다
            print(f"원문 받기 실패: {name} {type(e).__name__}")
            continue
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        old = state.get(name)
        if old and old["hash"] != h:
            prev = (SNAP / old["file"]).read_text(encoding="utf-8").splitlines() if old.get("file") else []
            diff = [ln for ln in difflib.unified_diff(prev, text.splitlines(), lineterm="", n=0) if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
            changed.append(name)
            print(f"원문 변경: {name} ({len(diff)}줄) " + " | ".join(d[:160] for d in diff[:6]))
        if not old or old["hash"] != h:
            f = f"{name.split('.')[0]}_{now:%Y%m%d_%H%M}.txt"
            (SNAP / f).write_text(text, encoding="utf-8")
            state[name] = {"hash": h, "file": f, "seen_kst": f"{now:%Y-%m-%d %H:%M}"}
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"spec_watch {now:%H:%M} 쪽 {len(PAGES)}, 변경 {len(changed)}" + (f" {changed}" if changed else ""))


if __name__ == "__main__":
    main()
