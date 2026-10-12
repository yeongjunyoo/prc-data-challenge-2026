"""F3 Package S (coordinator 10-11 01:17 and 01:2x): 604's Mac part, research/f3/daily_candidate.py imported unchanged (its sha16 is asserted)
with one more name prefix, f3f5: 603's (the S default) January and July rows in the public-phase template order, NOT fired here. The two
reference slots take the base twin's float file and its fired int file: 597 (research/f3/queue_ready/f3f3_<596 spec>.parquet, v43) for the 598c
family, 601 (research/f3/queue_ready/f3f4_<600 spec>.parquet and its fired file) for the L4 family. The family is read from the spec.
    python -X utf8 research/f3/daily_candidate_s.py <603 spec> f3f5_<603 spec> <old ranking> <old submitting> <base float> <base int> <out dir>
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN = "6277203e1ad12d2f"  # research/f3/daily_candidate.py as it ran for 597 (PC and Mac final_run)
if hashlib.sha256((HERE / "daily_candidate.py").read_bytes()).hexdigest()[:16] != PIN:
    raise SystemExit(f"research/f3/daily_candidate.py is not {PIN}: stop")
sys.path.insert(0, str(HERE))
import daily_candidate as dc  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 8 or not sys.argv[2].startswith("f3f5_") or not sys.argv[1].endswith("avskaps"):
        raise SystemExit(__doc__)
    l4 = sys.argv[1].startswith("m6r28xwin")
    fam, base, dflt = ("L4", "601", "600") if l4 else ("598c", "597", "596")
    dc.TEXTS["f3f5"] = dict(
        refs=(f"vs_{base}_float", f"vs_{base}_fired_int"), same="F2 모래상자의 S 세 사슬 끝",
        method=(f"F3 꾸러미 S 결선 뿌리 기준 603({fam} 가족의 기본안 {dflt} 에서 사슬 C 만 씨앗을 늘려 평균한 판: LightGBM ap 3 씨앗, CatBoost ap 3 씨앗, "
                f"사슬 C 의 RealMLP 씨앗 4 ~ 7, research/f3, vault research/2026-10-08-f3-jitter.md)의 1, 7월 344,841행을 공개 단계 틀의 순서로 옮긴 "
                f"일일 형식 후보(604, 597 꼴 = T 0.75). 꾸러미 S 의 보드 거부권(코디네이터 10-11 01:17 등록): 이 한 발이 기준 쌍둥이 {base} 보다 "
                f"MSE 로 +100 넘게 나쁘면 버린다. 쏘는 것은 사이클 세션이다."))
    dc.main()
