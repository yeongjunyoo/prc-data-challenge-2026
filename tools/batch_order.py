# SPDX-License-Identifier: GPL-3.0-only
# 측정 묶음의 쏘는 순서를 어느 점수도 나오기 전에 정한다(private notes G2 ③). 규칙: 차례 Δ(이웃한 두 후보의 로컬 RMSE 차)의 최소 절댓값이 가장 큰
# 순열, 같으면 차례 Δ 절댓값의 합이 큰 순열, 그래도 같으면 로컬 RMSE 가 낮은 후보로 시작하는 순열. 분해능 안의 작은 Δ 끼리만 차례를 채우지 않으려는
# 것이고, 닫힘 ② 의 문턱(80%)이나 판정 수는 바꾸지 않는다. 결과는 등록 때 --note 로 묶음 파일에 남긴다.
#     .venv/Scripts/python.exe -X utf8 tools/batch_order.py <큐 이름> x5
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.batches import candidate  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
names = sys.argv[1:]
loc = {}
for s in names:
    info, why = candidate(s)
    if why:
        raise SystemExit(why)
    loc[s] = info["local_rmse"]


def steps(p):
    return [round(abs(loc[b] - loc[a]), 3) for a, b in zip(p, p[1:])]


def key(p):
    d = steps(p)
    return (min(d), round(sum(d), 3), -loc[p[0]])


ranked = sorted(itertools.permutations(names), key=key, reverse=True)
best = ranked[0]
print("로컬:", ", ".join(f"{s} {loc[s]}" for s in names))
print("순서:", " → ".join(best))
print("차례 |Δ|:", steps(best), "최소", min(steps(best)), "합", round(sum(steps(best)), 3))
for p in ranked[1:6]:
    print("  다음 후보 순서:", " → ".join(p), steps(p))
