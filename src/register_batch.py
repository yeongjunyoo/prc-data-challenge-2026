# SPDX-License-Identifier: GPL-3.0-only
"""측정 묶음을 등록한다(private notes §0 G2 측정 단계). 진짜 후보 셋 이상을 그중 어느 것의 점수도 나오기 전에 묶고, 등록부는 다시 쓰지 않는다.

    python -m src.register_batch <큐 이름> x5 [...] [--exception notes/exceptions/<번호>.json] [--note "<순서를 정한 이유>"] [--dry]

검사: 장부의 변조 흔적, 이미 쏘였거나 다른 묶음에 든 후보, 후보마다 manifest 와 해시와 다시 잰 로컬 값과 조기 종료 행, 모든 쌍의 비교(시드 수 포함)를
지금 다시 내고, 준비 점검을 지금 다시 돌린다. 적는 순서가 쏘는 순서다. --dry 는 검사만 하고 적지 않는다. 결과: notes/batches/<번호>.json
준비 점검이 걸리면 등록하지 않는다. 예외는 --exception 으로 준 독립 승인 기록(src/exception.py)이 이 후보와 순서 그대로를 묶고 같은 검사로 범위 안일 때만
받고, 그 예외를 묶음 파일에 고정한다. 준비 점검 출력은 notes/readiness_records/<번호>_register.json 에 그대로 남는다(실패도).
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.batches import BATCHES, INDEX, MIN_BATCH, candidate, entries, ledger_state, now, readiness, registered_stems, sha, write_seal  # noqa: E402
from src import exception as X  # noqa: E402

EXC_RULE = (" 이 묶음은 준비 점검의 한정 예외(exception)를 안고 등록됐다. 예외는 이 묶음에만 쓰이고, 쏘기 직전마다 같은 예외 검사가 범위를 다시 재며 "
            "범위 밖이면 멈춘다.")
RULE = ("이 묶음의 후보는 어느 것의 점수도 나오기 전에 이 파일로 묶였다. 이 파일은 고치거나 지우지 않는다(장부의 sha_batch 가 지킨다). 발사기는 적힌 "
        "순서대로 쏘고, 이 묶음이 끝나기 전에는 다른 묶음이나 후보를 쏘지 않는다. 점수를 보고 남은 후보를 바꾸거나 빼지 않는다. 묶음은 모든 후보가 쏘이면 끝난다.")


def stem_of(p: str) -> str:
    return p.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".parquet")


def main(stems: list[str], dry: bool, exc_rel: str | None = None, note: str | None = None) -> int:
    if len(stems) < MIN_BATCH:
        print(f"거부: 묶음은 후보 {MIN_BATCH}개 이상이다(받은 것 {len(stems)})")
        return 1
    if len(set(stems)) != len(stems):
        print("거부: 같은 후보가 두 번 있다")
        return 1
    st = ledger_state()
    if st["problems"]:
        print("거부: 장부에 변조 흔적이 있다:", "; ".join(st["problems"]))
        return 1
    taken = (registered_stems() | st["stems"]) & set(stems)
    if taken:
        print("거부: 이미 등록됐거나 쏘인 후보:", ", ".join(sorted(taken)))
        return 1
    infos = []
    for s in stems:
        i, why = candidate(s)
        if why:
            print("거부:", why)
            return 1
        if i["pred_sha"] in st["pred_shas"]:
            print(f"거부: {s} 의 예측(sha {i['pred_sha']})은 이미 쏘였다")
            return 1
        infos.append(i)
    compares = []
    for a, b in itertools.combinations(infos, 2):
        r = subprocess.run([sys.executable, "-X", "utf8", "-m", "src.compare", a["holdout"], b["holdout"], "--ka", str(a["seeds"]), "--kb", str(b["seeds"])],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            print("거부: 비교 실패", r.stderr[-300:])
            return 1
        c = json.loads(r.stdout)
        f = f"notes/compare/{stem_of(a['holdout'])}__{stem_of(b['holdout'])}.json"
        compares.append({"a": a["stem"], "b": b["stem"], "file": f, "sha": sha(f), "delta_b_minus_a": c["delta_b_minus_a"],
                         "sd_delta": c["sd_delta"], "sd_boot": c["sd_boot"], "sd_seed": c["sd_seed"], "verdict": c["verdict"]})
    ok, comp = readiness(stems)
    failed = [x["check"] for x in comp.get("checks", []) if not x["pass"]]
    exc = None
    if ok and exc_rel:
        print("거부: 준비 점검이 통과해 예외가 필요 없다")
        return 1
    if not ok:
        if not exc_rel or comp.get("error"):
            print("거부: 준비 점검 미통과", comp.get("error") or failed)
            return 1
        probs, got = X.check(X.load(exc_rel), exc_rel, stems, comp, "register")
        if probs:
            print("거부: 준비 점검 미통과이고 예외 범위 밖이다:", "; ".join(probs))
            return 1
        exc = X.binding(exc_rel) | {"checked_at_registration": got}
        print("준비 점검 미통과 칸이 예외", exc_rel, "범위 안이다:", got["nan_counts"])
    n = len(entries()) + 1
    path = BATCHES / f"{n:03d}.json"
    rec = ROOT / "notes" / "readiness_records" / f"{n:03d}_register.json"
    entry = {"number": n, "registered_kst": now(), "candidates": infos, "compares": compares,
             "readiness": {"file": rec.relative_to(ROOT).as_posix(), "all_pass": bool(ok), "failed": failed, "checks": len(comp.get("checks", []))},
             "rule": RULE + (EXC_RULE if exc else "")}
    if exc:
        entry["exception"] = exc
    if note:
        entry["note"] = note
    for c in compares:
        print(f"  {c['a']} → {c['b']}: Δ {c['delta_b_minus_a']} ± {c['sd_delta']} | {c['verdict']}")
    if dry:
        print("(dry) 등록 가능, 적지 않았다:", path.relative_to(ROOT).as_posix())
        return 0
    if path.exists() or rec.exists():
        print("거부: 같은 번호의 등록부나 준비 점검 기록이 있다", path)
        return 1
    rec.parent.mkdir(parents=True, exist_ok=True)
    with open(rec, "x", encoding="utf-8") as f:
        f.write(json.dumps(comp, ensure_ascii=False, indent=1))
    entry["readiness"]["sha"] = sha(entry["readiness"]["file"])
    text = json.dumps(entry, ensure_ascii=False, indent=1)
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)
    rel = path.relative_to(ROOT).as_posix()
    new = not INDEX.exists()
    with INDEX.open("a", encoding="utf-8", newline="") as f:
        if new:
            f.write("number,path,sha,registered_kst\n")
        f.write(f"{n},{rel},{sha(rel)},{entry['registered_kst']}\n")
    print("등록", path.relative_to(ROOT).as_posix(), "sha", sha(path.relative_to(ROOT).as_posix()), "순서", " → ".join(stems))
    sp = write_seal(entry)
    r = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "tools" / "seal_batch.py"), f"{n}"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    print("봉인", sp.relative_to(ROOT).as_posix(), (r.stdout or r.stderr).strip()[-300:])
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    opt = {}
    for k in ("--exception", "--note"):
        if k in a:
            i = a.index(k)
            opt[k] = a[i + 1]
            del a[i:i + 2]
    sys.exit(main([x for x in a if x != "--dry"], "--dry" in sys.argv, opt.get("--exception"), opt.get("--note")))
