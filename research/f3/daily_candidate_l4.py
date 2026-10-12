# SPDX-License-Identifier: GPL-3.0-only
"""F3 L4 final (10-10 10:07 and 10:5x): the veto twin 601's Mac part, research/f3/daily_candidate.py imported unchanged (its sha16 is
asserted) with one more name prefix, f3f4: 600's (L4S on final_ranking) January and July rows in the public-phase template order, NOT fired here.
The two reference slots take 597's float file (research/f3/queue_ready/f3f3_<L3S spec>.parquet, the same rows from 596) and v43 (the int file 597
fired as, 228.5043, copied from prc-2026/subs to research/f3/in/spirited-tiger_v43.parquet).
    python -X utf8 research/f3/daily_candidate_l4.py <spec> f3f4_<spec> <old ranking> <old submitting> <597 float parquet> <v43 parquet> <out dir>
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN = "6277203e1ad12d2f"  # research/f3/daily_candidate.py as it ran for 597 (PC and Mac final_run, 10-10 10:50)
if hashlib.sha256((HERE / "daily_candidate.py").read_bytes()).hexdigest()[:16] != PIN:
    raise SystemExit(f"research/f3/daily_candidate.py is not {PIN}: stop")
sys.path.insert(0, str(HERE))
import daily_candidate as dc  # noqa: E402

dc.TEXTS["f3f4"] = dict(
    refs=("vs_597_float", "vs_v43_int"), same="R17-A 의 공개 뿌리 L4S",
    method=("F3 L4 판 결선 뿌리 기준 600(L4S = L3S 꼴에 L4 의 CatBoost, RealMLP 학습기를 feats_v28 로 다시 맞춤, 결선 뿌리에서 full 만 맞추고 "
            "val, fwd 는 R17-A 의 공개 뿌리 판, research/f3, private notes research/2026-10-08-f3-jitter.md)의 1, 7월 344,841행을 공개 단계 틀의 "
            "순서로 옮긴 일일 형식 후보(601, 597 꼴 = T 0.75). L4 의 보드 거부권(10-10 10:07 등록): 한 발로 v43 597 의 228.5043 "
            "보다 낮아야 한다. 쏘는 것은 예약 작업이다."))

if __name__ == "__main__":
    if len(sys.argv) < 3 or not sys.argv[2].startswith("f3f4_"):
        raise SystemExit(__doc__)
    dc.main()
