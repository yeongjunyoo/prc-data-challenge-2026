# SPDX-License-Identifier: GPL-3.0-only
"""발사 큐(private notes §0 G6: 우리는 큐를 채우고, 발사는 이 도구가 한다). 게이트는 여기서 강제한다.

큐 항목: `queue/<순번>_<이름>.parquet`(칸 MVT_ID_mvt, pred 초)와 같은 이름의 `.json`
    {"kind": "CANDIDATE" | "DIAG" | "ANCHOR", "method": "...", "local_rmse": 123.4, "note": "...", "manifest": "preds/manifest_<spec>.json"}
게이트(`GATE.json`): candidate_allowed 가 false 면 CANDIDATE 를 쏘지 않는다(감사가 측정 단계를 열 때 true). scope 의 ranking 이 team_best,
    split 이 none 이 아니면 쏘지 않는다. DIAG 는 diag_used < diag_limit 일 때만(지금 한도 0), ANCHOR 는 한 번뿐.
측정 묶음(늘, private notes §0 G2): CANDIDATE 는 `src/register_batch.py` 로 등록한 묶음으로만 쏜다(`src/batches.py`). 번호가 가장 낮은 끝나지 않은
    묶음 하나만, 적힌 순서대로 쏘고, 앞 후보를 못 쏘면 멈춘다. 쏘기 직전 그 후보의 등록 해시, 다시 잰 로컬 값, 비교 파일, 준비 점검을 다시 본다.
    장부에 변조 흔적(묶음 파일 없음, 해시 다름)이 있거나 팀 버킷에 장부에 없는 제출이 있으면 후보를 쏘지 않는다. 올리기는 이 도구가 쓰는 일회용
    허가(notes/push_permit.json, 파일 이름과 sha256)가 있어야만 된다(src/s3io.py push).
장부: 올리기가 끝난 뒤에만 한 줄을 더한다(올리지 못한 파일은 지우고 장부에 없다). note 에 uploaded=1 과 묶음 연결이 남는다. 결과가 늦으면 그 줄은
    대기로 남고 다음 실행 첫머리에서 같은 파일의 결과를 다시 읽어 적는다(같은 예측을 다시 올리지 않는다).
쿼터: 오늘(UTC) 채점 수를 팀 버킷의 _result.json 으로 센다.
    python -m src.fire_queue [--dry]
단일 발(10-05 체제, GATE.json firing single): 묶음 길(batch_step)은 남겨 두되 쓰지 않는다. 이름 없이 부르면 대기 결과만 읽고 쏘지 않는다.
    한 발씩, 기대 이득으로 고른 후보를 이유(--why)와 참고한 서버 점수(--refs)를 결정 장부(subs/fire_events.jsonl)에 먼저 적고 쏜다.
    검사는 src/shot.py 머리 주석: 게이트와 scope, manifest 와 코드 사본(candidate), 이미 쏜 후보, 장부와 제출 파일, 장부 밖 제출, 쿼터(결과와 올림 중 큰 쪽),
    결선 몫(2026-10-13, 14 UTC 에는 결정 장부에 결선 올림이 생기기 전까지 그날 마지막 칸을 쓰지 않는다, final_reserve),
    상식 검사, NaN 과 빠진 행, 시험에서 통째로 빈 칸, 엿보기 막이(모든 앞 제출과 1% 이상 행에서 달라야 한다). 준비 점검은 경고로만 돌려 intent 에 남긴다.
    python -m src.fire_queue fire-one <큐 이름> --why "<이유>" --refs "<참고한 서버 점수>" [--dry]
    --dry 는 검사를 끝까지 돌려 보고하고 올리기 전에 멈춘다(파일을 만들지 않고 결정 장부에도 적지 않는다). queue/done 의 쏜 후보도 dry 로 볼 수 있다.
결선 발(10-08 공지, 마감 2026-10-14T10:00Z): 넉 달(1, 2, 6, 7월) 예측 큐 후보 하나를 spirited-tiger_final.parquet 로 만들어 한 번 올린다.
    단일 발의 검사(엿보기 막이만 빼고, 이유는 결정 장부에)에 결선 자료 고정값(만들기 전과 올리기 직전), 결선 틀 검사, 버킷의 결선 파일 검사를 더한다.
    python -m src.fire_queue fire-final <큐 이름> --why "<이유>" --refs "<참고한 서버 점수>" [--dry] [--replace-final]
    올린 뒤 확인이 끊기면(upload_unknown, upload_unverified) 결선 파일을 남긴다. 그때는 PUT 없이 확인해 적는다(무장 없음):
    python -m src.fire_queue verify-final
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.batches import entries, ledger_rows, ledger_state, note_tokens, registered_stems, sha, verify_before_shot  # noqa: E402
from src.make_sub import LEDGER, build, ledger_append, next_version  # noqa: E402
from src.s3io import DAILY_QUOTA, PERMIT, TEAM, TEAM_BUCKET, ls, push, quota_used_today, result  # noqa: E402

QUEUE, DONE, GATE = ROOT / "queue", ROOT / "queue" / "done", ROOT / "GATE.json"
KST = ZoneInfo("Asia/Seoul")
SCOPE = {"ranking": "team_best", "split": "none"}
FINAL_RESERVE_DAYS = (dt.date(2026, 10, 13), dt.date(2026, 10, 14))  # 결선 몫을 남기는 UTC 날(마감 2026-10-14T10:00Z, 10-10 감사)


def now() -> str:
    return dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M")


def utc_today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def final_reserve(left: int) -> str | None:
    """결선 몫: 2026-10-13, 14(UTC)에 결정 장부에 결선 올림(uploaded, verify-final 이 적은 것 포함)이 아직 없으면 fire-one 은 그날 마지막 칸을
    쓰지 않는다(남은 칸이 1 이하면 막는다). 결선 파일이 하루 5발에 드는지 원문에 없어 fire-final 은 남은 칸이 있어야 쏘므로, 단일 발이 그 칸을 먼저
    쓰면 결선이 막힌다. fire-final 과 묶음 길은 이 검사를 쓰지 않는다."""
    from src import shot as S
    today = utc_today()
    if today not in FINAL_RESERVE_DAYS or left > 1 or any(e.get("event") == "uploaded" for e in S.final_events()):
        return None
    return (f"오늘(UTC {today})의 마지막 칸은 결선 몫이다(남은 칸 {left}): 결정 장부에 결선 올림이 아직 없어 2026-10-13, 14(UTC)에는 "
            "fire-one 이 그날 마지막 칸을 쓰지 않는다")


def record(version: int, score: float) -> None:
    rows = list(csv.DictReader(open(LEDGER, encoding="utf-8")))
    for r in rows:
        if r["version"] == str(version):
            r["server_rmse"], r["server_read_kst"] = str(score), now()
    w = csv.DictWriter(open(LEDGER, "w", newline="", encoding="utf-8"), fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)


def sane(p: Path) -> str | None:
    """후보 값의 상식 검사. 기준은 2025 학습 출발편(평균 991, 표준편차 546)뿐이다. 상수 진단 발로 푼 시험 통계는 쓰지 않는다(private notes §5)."""
    return sane_values(pd.read_parquet(p)["pred"])


def sane_values(v: pd.Series) -> str | None:
    """sane 의 값 판(결선 파일의 TAXITIME_SEC_mvt 에도 같은 문턱을 쓴다)."""
    if not v.map(math.isfinite).all():
        return "유한하지 않은 예측이 있다"
    if not 500 <= v.mean() <= 2000 or v.std() < 100:
        return f"예측 분포가 학습 분포와 너무 다르다(평균 {v.mean():.0f}, 표준편차 {v.std():.0f})"
    return None


def allowed(meta: dict, gate: dict) -> str | None:
    kind = meta.get("kind")
    if kind == "CANDIDATE" and not gate.get("candidate_allowed"):
        return "게이트 감사 통과 전이라 CANDIDATE 금지(GATE.json)"
    scope = gate.get("scope") or {}
    if kind == "CANDIDATE" and any(scope.get(k) != v for k, v in SCOPE.items()):
        return f"GATE.json scope 의 ranking, split 이 {SCOPE} 가 아니다(측정 단계는 팀별 최고 점수, 분할 없는 대회만)"
    if kind == "DIAG" and gate.get("diag_used", 0) >= gate.get("diag_limit", 0):
        return "진단 발 한도를 다 썼다"
    if kind == "ANCHOR" and gate.get("anchor_fired"):
        return f"ANCHOR 는 한 번뿐이다(이미 {gate['anchor_fired']})"
    if kind not in ("CANDIDATE", "DIAG", "ANCHOR"):
        return f"모르는 kind: {kind}"
    return None


def save_result(name: str, res: dict, version: int) -> None:
    (ROOT / "subs" / "results" / f"{name}_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    record(version, res["score"])


def recover_pending(dry: bool) -> int:
    """올렸지만 결과를 못 읽은 줄(uploaded=1, server_rmse 없음)의 결과를 같은 파일 이름으로 다시 읽는다."""
    n = 0
    for r in ledger_rows():
        if r.get("server_rmse") or "uploaded=1" not in (r.get("note") or "").split():
            continue
        try:
            res = result(r["file"])
        except Exception:  # noqa: BLE001  아직 결과가 없다
            print(now(), "대기 중", r["file"], flush=True)
            continue
        if dry:
            print(now(), "(dry) 대기 결과를 적을 차례", r["file"], res.get("score"), flush=True)
            continue
        save_result(r["file"], res, int(r["version"]))
        if "event" in note_tokens(r.get("note", "")):  # 단일 발: 결정 장부에도 결과를 더한다
            from src import shot as S
            S.event_append({"event": "scored", "file": r["file"], "score": res.get("score"), "via": "recover_pending"})
        print(now(), "대기 결과 적음", r["file"], res.get("score"), flush=True)
        n += 1
    return n


def fire_one(p: Path, meta: dict, note: str) -> str | None:
    """올리기가 끝나야 장부에 적는다. 올리기 예외가 나면 버킷을 보고 정한다. 결과가 늦으면 대기로 두고 이름을 돌려준다."""
    pred = pd.read_parquet(p).set_index("MVT_ID_mvt")["pred"]
    method = f'{meta.get("kind")} {meta.get("method", "")}'
    note = note + " uploaded=1"
    sub = build(pred, method, meta.get("local_rmse"), note, write_ledger=False)
    tok = note_tokens(note)  # 묶음과 후보: push 가 지금 차례인 등록 후보인지, 올릴 값이 그 예측에서 나왔는지 다시 맞춘다(src/batches.py upload_problem)
    PERMIT.write_text(json.dumps({"file": sub.name, "sha256": hashlib.sha256(sub.read_bytes()).hexdigest(), "issued_kst": now(),
                                  "issued_utc_ts": dt.datetime.now(dt.timezone.utc).timestamp(), "kind": meta.get("kind"),
                                  "batch": tok.get("batch"), "sha_batch": tok.get("sha_batch"), "stem": tok.get("stem"),
                                  "by": "src/fire_queue.py"}), encoding="utf-8")
    try:
        push(sub)
    except BaseException as e:  # noqa: BLE001  push 는 SystemExit 도 낸다
        if isinstance(e, KeyboardInterrupt):
            raise
        if sub.name not in {o["key"] for o in ls(TEAM_BUCKET)}:
            sub.unlink(missing_ok=True)
            print(now(), "올리기 실패, 장부에 적지 않았다", p.name, repr(e)[:200], flush=True)
            return None
        print(now(), "올리기 예외가 났지만 버킷에 있어 올린 것으로 본다", sub.name, flush=True)
    ledger_append(sub, method, meta.get("local_rmse"), note)
    DONE.mkdir(parents=True, exist_ok=True)
    for q in (p, p.with_suffix(".json")):
        shutil.move(str(q), str(DONE / q.name))
    for _ in range(24):
        time.sleep(10)
        try:
            res = result(sub.name)
        except Exception:  # noqa: BLE001  결과 파일이 아직 없다(약 20초 걸린다)
            continue
        save_result(sub.name, res, int(sub.stem.split("_v")[1]))
        print(now(), "fired", sub.name, res.get("status"), res.get("score"), flush=True)
        return sub.name
    print(now(), "결과 없음(4분), 대기로 둔다. 다음 실행이 같은 파일의 결과를 읽는다", sub.name, flush=True)
    return sub.name


def link(path: Path, cand: dict, entry: dict | None = None) -> str:
    rel = path.relative_to(ROOT).as_posix()
    x = (entry or {}).get("exception")
    return (f" | batch={rel} sha_batch={sha(rel)} stem={cand['stem']} manifest={cand['manifest']} sha_manifest={cand['manifest_sha']}"
            f" spec={cand['spec']} feats={cand['feats']} holdout={cand['holdout']} sha_pred={cand['pred_sha']}"
            + (f" exception={x['file']} sha_exception={x['sha']}" if x else ""))


def batch_step(gate: dict, left: int, dry: bool) -> tuple[int, int]:
    """번호가 가장 낮은 끝나지 않은 묶음 하나를 적힌 순서대로 쏜다. 돌려주는 값: (쏜 수, 남은 쿼터, 막힘). 올리기 실패는 (-1, left, True)."""
    st = ledger_state()
    if st["problems"]:
        print(now(), "장부 변조 흔적으로 후보 발사를 멈춘다:", "; ".join(st["problems"]), flush=True)
        return 0, left, True
    for path, e in entries():
        rel = path.relative_to(ROOT).as_posix()
        remaining = [c for c in e["candidates"] if (rel, c["stem"]) not in st["fired"]]
        if remaining:
            break
    else:
        return 0, left, False
    print(now(), f"묶음 {path.name}: 남은 후보 {[c['stem'] for c in remaining]}", flush=True)
    fired = 0
    for cand in remaining:
        p = QUEUE / f"{cand['stem']}.parquet"
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8")) if p.with_suffix(".json").exists() else {"kind": "CANDIDATE"}
        why = (allowed(meta, gate) or (sane(p) if p.exists() else "큐 파일이 없다") or (None if left > 0 else "오늘(UTC) 쿼터를 다 썼다")
               or verify_before_shot(e, cand["stem"]))
        if why:
            print(now(), "stop", cand["stem"], why, "(묶음 순서를 지켜 뒤 후보도 쏘지 않는다)", flush=True)
            return fired, left, True
        if dry:
            print(now(), "(dry) would fire", cand["stem"], "from", path.name, flush=True)
            left -= 1
            continue
        if fire_one(p, meta, meta.get("note", "") + link(path, cand, e)) is None:
            return -1, left, True
        fired += 1
        left -= 1
    return fired, left, False


def main(dry: bool) -> int:
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    recover_pending(dry)
    items = sorted(p for p in QUEUE.glob("*.parquet") if not p.name.startswith("."))  # 맥에서 옮긴 ._ 리소스 포크 파일은 후보가 아니다(10-05)
    left = DAILY_QUOTA - quota_used_today()
    print(now(), f"queue {len(items)}, quota left {left}, candidate_allowed {gate.get('candidate_allowed')}", flush=True)
    if gate.get("firing") == "single":  # 10-05: 묶음 길은 남겨 두되 쓰지 않는다. 깨우기(예약 작업)가 이 함수를 불러도 대기 결과만 읽는다
        print(now(), "묶음 경로는 은퇴했다(GATE.json firing single, 10-05). 대기 결과만 읽고 쏘지 않는다. 발사는 "
              "python -m src.fire_queue fire-one <큐 이름> --why ... --refs ... 로 한 발씩 한다", flush=True)
        return 0
    reg = registered_stems()
    for p in items:
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        if meta.get("kind") == "CANDIDATE" and p.stem not in reg:
            print(now(), "skip", p.name, "등록된 측정 묶음의 후보가 아니다(src/register_batch.py)", flush=True)
    in_bucket = {o["key"] for o in ls(TEAM_BUCKET) if o["key"].startswith(f"{TEAM}_v") and o["key"].endswith(".parquet")}
    off_ledger = sorted(in_bucket - {r["file"] for r in ledger_rows()})
    if off_ledger:  # 발사기 밖에서 올린 제출이 있다: 쿼터와 판정이 장부와 어긋나므로 후보를 쏘지 않는다
        print(now(), "장부에 없는 제출이 버킷에 있어 후보 발사를 멈춘다:", ", ".join(off_ledger), flush=True)
        fired, blocked = 0, True
    else:
        fired, left, blocked = batch_step(gate, left, dry)
    if fired < 0:
        return 2
    if blocked:  # 막힌 묶음이 있으면 다른 발(ANCHOR, DIAG)로 내려가지 않는다
        print(now(), f"fired {fired} candidates, quota left {left}, 묶음이 막혀 여기서 멈춘다", flush=True)
        items = []
    for p in items:  # ANCHOR, DIAG: 게이트 규칙대로(지금은 둘 다 막혀 있다)
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8")) if p.with_suffix(".json").exists() else {}
        if meta.get("kind") == "CANDIDATE" or not p.exists():
            continue
        why = allowed(meta, gate) or sane(p) or (None if left > 0 else "오늘(UTC) 쿼터를 다 썼다")
        if why:
            print(now(), "skip", p.name, why, flush=True)
            continue
        if dry:
            print(now(), "(dry) would fire", p.name, meta.get("kind"), flush=True)
            left -= 1
            continue
        if fire_one(p, meta, meta.get("note", "")) is None:
            return 2
        if meta.get("kind") == "DIAG":
            gate["diag_used"] = gate.get("diag_used", 0) + 1
            GATE.write_text(json.dumps(gate, ensure_ascii=False, indent=1), encoding="utf-8")
        left -= 1
    print(now(), f"fired {fired} candidates, quota left {left}", flush=True)
    if fired:  # 판정이 늘었으니 닫힘 증거를 다시 계산한다
        subprocess.run([sys.executable, "-X", "utf8", "-m", "src.sign_agree"], cwd=ROOT, capture_output=True)
        sa = json.loads((ROOT / "notes" / "sign_agreement.json").read_text(encoding="utf-8"))
        print(now(), f"sign_agree judged {sa['judged_candidates']}, steps {sa['steps_agree']}, blockers {len(sa['local_selection_blockers'])}", flush=True)
    return 0


def readiness_warning(name: str, qdir: Path, meta: dict) -> dict:
    """준비 점검(src/composition.py)을 경고로만 돌린다. 끝나지 않으면(피처 판 전체를 읽어 3 GB 를 넘는다) 막지 않고, 앞서 쓴 기록(큐 json 의
    readiness, notes/composition_<이름>.json, 묶음 등록 기록 notes/readiness_records 의 순서)의 걸린 칸을 대신 옮긴다. 입력(큐 예측, 홀드아웃,
    피처 판)은 candidate() 가 해시로 확인했다."""
    from src import shot as S
    out = f"notes/composition_fireone_{name}.json"
    t0, res = time.time(), {}
    try:
        r = subprocess.run([sys.executable, "-X", "utf8", "-m", "src.composition", "--queue-dir", qdir.relative_to(ROOT).as_posix(), "--only", name,
                            "--out", out], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
        res["rc"], err = r.returncode, (r.stderr or r.stdout or "")[-300:]
    except subprocess.TimeoutExpired:
        res["rc"], err = None, "1800초 안에 끝나지 않았다"
    outp = ROOT / out
    if outp.exists() and outp.stat().st_mtime >= t0 - 1:
        comp = json.loads(outp.read_text(encoding="utf-8"))
        res.update(source="fire", file=out, ran=True, all_pass=bool(comp.get("all_pass")), failed=S.readiness_failed(comp, name))
        return res
    res.update(ran=False, error=err)
    regs = [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "notes" / "readiness_records").glob("*_register.json"), reverse=True)
            if name in json.loads(p.read_text(encoding="utf-8")).get("candidates", [])]  # 묶음 등록 때의 기록(171, 201, 211 같은 옛 후보)
    rec = next((f for f in (meta.get("readiness"), f"notes/composition_{name}.json", *regs) if f and (ROOT / f).exists()), None)
    if rec:
        comp = json.loads((ROOT / rec).read_text(encoding="utf-8"))
        res.update(source="record", file=rec, all_pass=bool(comp.get("all_pass")), failed=S.readiness_failed(comp, name),
                   readiness_warn_at_enqueue=bool(meta.get("readiness_warn")))
    return res


def fire_single(name: str, why: str, refs: str, dry: bool) -> int:
    """단일 발(10-05 체제). 검사, 결정 장부 intent, 허가, push, 장부, 결과 순서. 막는 것이 하나라도 있으면 쏘지 않는다(dry 는 끝까지 보고)."""
    from src import shot as S
    from src.batches import candidate
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    hard, warn = [], []
    recover_pending(dry)
    if gate.get("firing") != "single":
        hard.append("GATE.json firing 이 single 이 아니다(묶음 체제에서는 python -m src.fire_queue 가 묶음을 쏜다)")
    hard += [g] if (g := allowed({"kind": "CANDIDATE"}, gate)) else []
    if not why.strip() or not refs.strip():
        hard.append("--why(이유)와 --refs(참고한 서버 점수)를 둘 다 적어야 한다")
    qdir = QUEUE if (QUEUE / f"{name}.parquet").exists() else DONE if (DONE / f"{name}.parquet").exists() else None
    if qdir is None:
        print(now(), "refuse", f"큐에 {name} 이 없다(queue/, queue/done/)", flush=True)
        return 1
    if qdir == DONE:
        hard.append(f"{name} 은 이미 쏜 후보다(queue/done)")
    p = qdir / f"{name}.parquet"
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    try:
        cand, w = candidate(name, qdir)  # manifest, 설정과 명령, 해시, 코드 사본, 큐 예측 해시, 다시 잰 로컬 값, 조기 종료 행
    except SystemExit as e:
        cand, w = {}, f"{name}: {e}"
    hard += [w] if w else []
    rows, probs = S.ledger_files()
    hard += probs
    toks = [note_tokens(r.get("note", "")) for r in rows]
    if any(t.get("stem") == name for t in toks):
        hard.append(f"장부에 {name} 이 이미 있다")
    if cand and any(t.get("sha_pred") == cand["pred_sha"] for t in toks):
        hard.append(f"장부에 같은 예측(sha_pred {cand['pred_sha']})이 이미 있다")
    bs = S.bucket_state(ls(TEAM_BUCKET), rows)
    if bs["off_ledger"]:
        hard.append(f"장부에 없는 제출이 버킷에 있다: {', '.join(bs['off_ledger'])}")
    if bs["left"] <= 0:
        hard.append(f"오늘(UTC) 쿼터를 다 썼다(결과 {bs['results_today_utc']}, 올림 {bs['uploads_today_utc']})")
    elif (reserve := final_reserve(bs["left"])):
        hard.append(reserve)
    tpl = S.template_ids()
    pp = S.pred_problems(p, tpl)
    hard += pp
    hard += [s] if not pp and (s := sane(p)) else []
    if cand:
        m = json.loads((ROOT / cand["manifest"]).read_text(encoding="utf-8"))
        e = S.empty_on_test(m["feats"], m["feats_sha"])
        hard += S.empty_problems(e)
        if e.get("rare_columns_empty_on_test"):
            warn.append(f"드문 칸이 시험에서 비었다(정보): {sorted(e['rare_columns_empty_on_test'])}")
    values = None if pp else S.submission_values(pd.read_parquet(p).set_index("MVT_ID_mvt")["pred"], tpl)
    gd = S.guard(values, tpl, rows) if values is not None else {"pass": False, "min_diff_frac": None, "nearest": None, "rule": "", "problems": []}
    if values is not None and not gd["pass"]:
        hard.append(f"엿보기 막이: 앞 제출 {gd['nearest']} 와 {gd['min_diff_frac']:.4%} 행에서만 다르다({gd['rule']})" if gd["nearest"] and not gd["problems"]
                    else "엿보기 막이를 잴 수 없었다: " + "; ".join(gd["problems"]))
    ready = readiness_warning(name, qdir, meta) if (dry or not hard) else {}
    if ready and not ready.get("ran"):
        last = (ready.get("error") or "").strip().splitlines()[-1:] or [""]
        warn.append(f"준비 점검이 지금 끝나지 않았다(rc {ready.get('rc')}, 경고만, {last[0][-120:]}). "
                    + (f"기록 {ready['file']} 를 옮긴다" if ready.get("file") else "옮길 기록도 없다"))
    src_ = {"fire": "지금", "record": "기록"}.get(ready.get("source"), "")
    warn += [f"준비 점검 경고({src_}): {c['check']} = {c['value']} (폭 {c['band']})" for c in ready.get("failed", [])]
    for h in hard:
        print(now(), "BLOCK", h, flush=True)
    for x in warn:
        print(now(), "WARN", x, flush=True)
    if gd.get("nearest"):
        print(now(), "guard", f"가장 가까운 앞 제출 {gd['nearest']} 와 {gd['min_diff_frac']:.4%} 행에서 다르다(문턱 {S.GUARD_MIN:.0%})", flush=True)
    print(now(), "quota", f"오늘(UTC) 결과 {bs['results_today_utc']}, 올림 {bs['uploads_today_utc']}, 남음 {bs['left']}", flush=True)
    if hard:
        print(now(), "(dry) would refuse" if dry else "refuse", f"{name}: 막는 것 {len(hard)}개", flush=True)
        return 1
    if dry:
        print(now(), "(dry) would fire", f"{name} 을 spirited-tiger_v{next_version()}.parquet 로, why 와 refs 를 결정 장부에 먼저 적고 올린다. "
              "올리기 전에 멈췄다(파일, 결정 장부, 허가 모두 쓰지 않음)", flush=True)
        return 0
    method = f"CANDIDATE {meta.get('method', '')}"
    sub = build(pd.read_parquet(p).set_index("MVT_ID_mvt")["pred"], method, meta.get("local_rmse"), "", write_ledger=False)
    if not np.array_equal(pd.read_parquet(sub)["TAXITIME_SEC_mvt"].to_numpy(), values):
        sub.unlink(missing_ok=True)
        print(now(), "refuse", "만든 제출 파일의 값이 막이에서 잰 값과 다르다(파일을 지웠다)", flush=True)
        return 1
    digest, version = S.sha256_file(sub), int(sub.stem.split("_v")[1])
    ev = S.event_append({"event": "intent", "file": sub.name, "sha256": digest, "version": version, "stem": name, "why": why, "refs": refs,
                         "manifest": cand["manifest"], "sha_manifest": cand["manifest_sha"], "sha_pred": cand["pred_sha"], "spec": cand["spec"],
                         "feats": cand["feats"], "local_rmse": cand["local_rmse"], "readiness": ready, "warnings": warn,
                         "guard": {k: gd[k] for k in ("min_diff_frac", "nearest", "rule")},
                         "quota": {k: bs[k] for k in ("results_today_utc", "uploads_today_utc", "left")}, "by": "src/fire_queue.py fire-one"})
    note = (meta.get("note", "") + f" | why: {S.note_text(why)} | refs: {S.note_text(refs)} | mode=single event={S.EVENTS.relative_to(ROOT).as_posix()}#{ev}"
            f" stem={name} manifest={cand['manifest']} sha_manifest={cand['manifest_sha']} spec={cand['spec']} feats={cand['feats']}"
            f" holdout={cand['holdout']} sha_pred={cand['pred_sha']} uploaded=1")
    PERMIT.write_text(json.dumps({"file": sub.name, "sha256": digest, "issued_kst": now(), "issued_utc_ts": dt.datetime.now(dt.timezone.utc).timestamp(),
                                  "kind": "CANDIDATE", "mode": "single", "stem": name, "sha_pred": cand["pred_sha"], "event": ev,
                                  "by": "src/fire_queue.py fire-one"}), encoding="utf-8")
    try:
        push(sub)
    except BaseException as e:  # noqa: BLE001  push 는 SystemExit 도 낸다
        if isinstance(e, KeyboardInterrupt):
            raise
        if sub.name not in {o["key"] for o in ls(TEAM_BUCKET)}:
            sub.unlink(missing_ok=True)
            S.event_append({"event": "upload_failed", "file": sub.name, "intent": ev, "error": repr(e)[:300]})
            print(now(), "올리기 실패, 장부에 적지 않았다", name, repr(e)[:300], flush=True)
            return 2
        print(now(), "올리기 예외가 났지만 버킷에 있어 올린 것으로 본다", sub.name, flush=True)
    ledger_append(sub, method, meta.get("local_rmse"), note)
    S.event_append({"event": "uploaded", "file": sub.name, "intent": ev, "version": version})
    DONE.mkdir(parents=True, exist_ok=True)
    for q in (p, p.with_suffix(".json")):
        shutil.move(str(q), str(DONE / q.name))
    for _ in range(24):
        time.sleep(10)
        try:
            res = result(sub.name)
        except Exception:  # noqa: BLE001  결과 파일이 아직 없다(약 20초 걸린다)
            continue
        save_result(sub.name, res, version)
        S.event_append({"event": "scored", "file": sub.name, "intent": ev, "score": res.get("score"), "status": res.get("status")})
        print(now(), "fired", sub.name, name, res.get("status"), res.get("score"), flush=True)
        return 0
    S.event_append({"event": "pending", "file": sub.name, "intent": ev})
    print(now(), "결과 없음(4분), 대기로 둔다. 다음 실행이 같은 파일의 결과를 읽는다", sub.name, flush=True)
    return 0


def fire_one_cli(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m src.fire_queue fire-one", description="단일 발(10-05 체제): 한 후보를 이유와 참고 점수를 먼저 적고 쏜다")
    ap.add_argument("name", help="큐 이름(queue/<이름>.parquet)")
    ap.add_argument("--why", required=True, help="이 후보를 지금 쏘는 이유(기대 이득)")
    ap.add_argument("--refs", required=True, help="고를 때 참고한 서버 점수")
    ap.add_argument("--dry", action="store_true", help="검사만 끝까지 돌리고 올리기 전에 멈춘다")
    a = ap.parse_args(argv)
    # 문서 속 명령이 셸 치환으로 돌아도 쏘지 않게, 실제 발은 PRC_FIRE_ARMED=<큐 이름> 을 붙여야 한다(문서에는 붙이지 않는다)
    if not a.dry and os.environ.get("PRC_FIRE_ARMED") != a.name:
        print(now(), "refuse", "실제 발 표시 변수(PRC_FIRE_ARMED)가 이 큐 이름과 같지 않다(발은 오케스트레이터가 명령줄에서만 붙인다)", flush=True)
        return 2
    return fire_single(a.name, a.why, a.refs, a.dry)


def build_final(pred: pd.Series) -> Path:
    """결선 제출 파일 subs/spirited-tiger_final.parquet. src/make_sub.build 와 같은 식(틀 순서, 반올림, 0 아래 자름, int32, 틀의 parquet 스키마)이고
    틀만 결선 틀(data/final_phase/final_submitting.parquet)이다. subs/ledger.csv 에는 적지 않는다(version 칸과 그 읽기가 _vN 을 전제한다)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src import shot as S
    schema = pq.read_schema(S.FINAL_TEMPLATE)
    tpl = S.final_template_ids()
    miss = pred.reindex(tpl).isna()
    if miss.any():
        raise SystemExit(f"예측 없는 행 {int(miss.sum())}개")
    out = pd.DataFrame({"MVT_ID_mvt": tpl, "TAXITIME_SEC_mvt": S.submission_values(pred, tpl)})
    table = pa.Table.from_pandas(out, schema=schema, preserve_index=False)
    assert table.schema.equals(schema, check_metadata=False) and table.num_rows == len(tpl)
    path = ROOT / "subs" / S.FINAL_NAME
    pq.write_table(table, path)
    return path


def fire_final(name: str, why: str, refs: str, dry: bool, replace: bool) -> int:
    """결선 발(10-08 공지). 단일 발과 같은 순서(검사, 결정 장부 intent, 일회용 허가, push, 결정 장부)에 결선 검사를 더한다: 결선 자료 고정값
    (내 파일 sha256 과 버킷 ETag, 만들기 전과 올리기 직전), 결선 틀(670,790행, 칸과 타입, 같은 ID 같은 순서, 빈 값과 음수, 상식 범위), 팀 버킷과
    결정 장부의 결선 파일(있으면 --replace-final 없이는 막는다). 엿보기 막이는 행 수가 달라 건너뛰고 그 이유를 intent 에 적는다.
    올린 뒤에는 버킷 객체의 ETag 가 파일 MD5 와 같은지 head 로 확인한다. 막는 것이 하나라도 있으면 쏘지 않는다(dry 는 끝까지 보고)."""
    from src import shot as S
    from src.batches import candidate
    from src.s3io import DATA_BUCKET, head
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    hard, warn = [], []
    recover_pending(dry)
    if gate.get("firing") != "single":
        hard.append("GATE.json firing 이 single 이 아니다(결선 발은 단일 발 체제에서만 쓴다)")
    hard += [g] if (g := allowed({"kind": "CANDIDATE"}, gate)) else []
    if not why.strip() or not refs.strip():
        hard.append("--why(이유)와 --refs(참고한 서버 점수)를 둘 다 적어야 한다")
    pins_bad, pins = S.final_data_problems(lambda k: head(DATA_BUCKET, k))  # 만들기 전: 내 결선 자료와 버킷이 받을 때와 같은가
    hard += pins_bad
    qdir = QUEUE if (QUEUE / f"{name}.parquet").exists() else DONE if (DONE / f"{name}.parquet").exists() else None
    if qdir is None:
        print(now(), "refuse", f"큐에 {name} 이 없다(queue/, queue/done/)", flush=True)
        return 1
    if qdir == DONE:
        hard.append(f"{name} 은 이미 쏜 후보다(queue/done)")
    p = qdir / f"{name}.parquet"
    try:
        cand, w = candidate(name, qdir)  # manifest, 설정과 명령, 해시, 코드 사본, 큐 예측 해시, 다시 잰 로컬 값, 조기 종료 행
    except SystemExit as e:
        cand, w = {}, f"{name}: {e}"
    hard += [w] if w else []
    rows, probs = S.ledger_files()
    hard += probs
    up = [e for e in S.final_events() if e.get("event") == "uploaded"]
    if up and not replace:
        hard.append(f"결정 장부에 결선 올림이 이미 있다(#{up[-1].get('n')}, {up[-1].get('at_kst')}). 다시 올리려면 --replace-final")
    objs = ls(TEAM_BUCKET)
    bs = S.bucket_state(objs, rows)
    if bs["off_ledger"]:
        hard.append(f"장부에 없는 제출이 버킷에 있다: {', '.join(bs['off_ledger'])}")
    if bs["left"] <= 0:
        hard.append(f"오늘(UTC) 쿼터를 다 썼다(결과 {bs['results_today_utc']}, 올림 {bs['uploads_today_utc']}). "
                    "결선 파일이 하루 5발에 드는지 원문에 없어 하나를 남겨 둔다")
    hard += S.final_bucket_problems(objs, replace)
    if S.FINAL_NAME in {o["key"] for o in objs} and not up:
        warn.append("팀 버킷의 결선 파일이 결정 장부에 올림으로 적혀 있지 않다(앞 시도가 upload_unknown 이나 upload_unverified 로 끝났을 수 있다). "
                    "다시 쏘기 전에 .venv/Scripts/python.exe -X utf8 -m src.fire_queue verify-final 로 확인한다(PUT 없음)")
    replaced = None
    if S.FINAL_NAME in {o["key"] for o in objs}:
        try:
            replaced = head(TEAM_BUCKET, S.FINAL_NAME)  # 덮어쓸 객체(--replace-final 일 때만 여기까지 온다)를 intent 에 남긴다
        except Exception as e:  # noqa: BLE001
            replaced = {"error": repr(e)[:200]}
    tpl = S.final_template_ids()
    if len(tpl) != S.FINAL_ROWS:
        hard.append(f"결선 틀이 {len(tpl)}행이다(고정 {S.FINAL_ROWS})")
    pp = S.pred_problems(p, tpl)
    hard += pp
    hard += [s] if not pp and (s := sane(p)) else []
    if cand:
        m = json.loads((ROOT / cand["manifest"]).read_text(encoding="utf-8"))
        e = S.empty_on_final(m["feats"], m["feats_sha"])
        hard += S.empty_problems(e)
        if e.get("test_rows") and e["test_rows"] < S.FINAL_ROWS:
            warn.append(f"피처 판 {m['feats']} 의 결선 시험 행이 {e['test_rows']}개다(결선 틀 {S.FINAL_ROWS}). 새 두 달이 다른 피처 판에서 왔는지 manifest 를 본다")
        if e.get("rare_columns_empty_on_test"):
            warn.append(f"드문 칸이 결선 시험에서 비었다(정보): {sorted(e['rare_columns_empty_on_test'])}")
    values = None if pp else S.submission_values(pd.read_parquet(p).set_index("MVT_ID_mvt")["pred"], tpl)
    gd = {"skipped": True, "why": S.GUARD_SKIP_FINAL}
    local = ROOT / "subs" / S.FINAL_NAME
    if local.exists():
        warn.append(f"subs/{S.FINAL_NAME} 이 이미 있다(앞 시도). 실제 발에서 subs/final_attempts/ 로 옮기고 결정 장부에 적은 뒤 새로 만든다")
    for h in hard:
        print(now(), "BLOCK", h, flush=True)
    for x in warn:
        print(now(), "WARN", x, flush=True)
    print(now(), "guard", gd["why"], flush=True)
    print(now(), "final data", "; ".join(f"{k} sha256 {v['sha256'][:16]} ETag {v['bucket_etag']}" for k, v in pins.items()), flush=True)
    print(now(), "quota", f"오늘(UTC) 결과 {bs['results_today_utc']}, 올림 {bs['uploads_today_utc']}, 남음 {bs['left']}", flush=True)
    if hard:
        print(now(), "(dry) would refuse" if dry else "refuse", f"{name}: 막는 것 {len(hard)}개", flush=True)
        return 1
    if dry:
        print(now(), "(dry) would fire", f"{name} 을 {S.FINAL_NAME} 로" + (" (팀 버킷의 결선 파일을 덮어씀)" if replaced else "")
              + ", why 와 refs 를 결정 장부에 먼저 적고 올린다. 만들기와 올리기 전에 멈췄다(파일, 결정 장부, 허가 모두 쓰지 않음)", flush=True)
        return 0
    if local.exists():
        arch = ROOT / "subs" / "final_attempts" / f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{S.FINAL_NAME}"
        arch.parent.mkdir(parents=True, exist_ok=True)
        old = S.sha256_file(local)
        shutil.move(str(local), str(arch))
        S.event_append({"event": "final_local_archived", "kind": "final", "file": S.FINAL_NAME, "sha256": old, "to": arch.relative_to(ROOT).as_posix()})
    sub = build_final(pd.read_parquet(p).set_index("MVT_ID_mvt")["pred"])
    fp = S.final_file_problems(sub, tpl)
    if fp or not np.array_equal(pd.read_parquet(sub)["TAXITIME_SEC_mvt"].to_numpy(), values):
        sub.unlink(missing_ok=True)
        print(now(), "refuse", "만든 결선 파일이 검사를 통과하지 못했다(파일을 지웠다): " + ("; ".join(fp) or "값이 큐 예측에서 잰 값과 다르다"), flush=True)
        return 1
    pins_bad, pins = S.final_data_problems(lambda k: head(DATA_BUCKET, k))  # 올리기 직전: 다시(push 의 final_upload_problem 도 다시 본다)
    if pins_bad:
        sub.unlink(missing_ok=True)
        print(now(), "refuse", "; ".join(pins_bad), flush=True)
        return 1
    digest = S.sha256_file(sub)
    ev = S.event_append({"event": "intent", "kind": "final", "file": S.FINAL_NAME, "sha256": digest, "stem": name, "why": why, "refs": refs,
                         "manifest": cand["manifest"], "sha_manifest": cand["manifest_sha"], "sha_pred": cand["pred_sha"], "spec": cand["spec"],
                         "feats": cand["feats"], "local_rmse": cand["local_rmse"], "rows": int(len(tpl)), "final_data": pins, "guard": gd,
                         "replace": replace, "replaced_object": replaced, "warnings": warn,
                         "quota": {k: bs[k] for k in ("results_today_utc", "uploads_today_utc", "left")}, "by": "src/fire_queue.py fire-final"})
    PERMIT.write_text(json.dumps({"file": S.FINAL_NAME, "sha256": digest, "issued_kst": now(), "issued_utc_ts": dt.datetime.now(dt.timezone.utc).timestamp(),
                                  "kind": "FINAL", "mode": "final", "stem": name, "sha_pred": cand["pred_sha"], "event": ev, "replace": replace,
                                  "by": "src/fire_queue.py fire-final"}), encoding="utf-8")
    try:
        push(sub)
    except BaseException as e:  # noqa: BLE001  push 는 SystemExit 도 낸다
        if isinstance(e, KeyboardInterrupt):
            raise
        rm = S.final_remote(sub)
        if rm["state"] == "unknown":  # head 도 실패: 올라갔는지 모른다. 파일을 남기고 사람이 확인한 뒤 verify-final 로 적는다
            S.event_append({"event": "upload_unknown", "kind": "final", "file": S.FINAL_NAME, "intent": ev, "put": "raised", "error": repr(e)[:300], "remote": rm})
            print(now(), "모름", f"올리기 예외({repr(e)[:200]}) 뒤 버킷 head 도 실패해({rm.get('error')}) 올라갔는지 모른다. 결선 파일은 남겨 둔다", flush=True)
            print(now(), "확인", S.final_manual_check(), flush=True)
            return 2
        if not rm["match"]:
            S.event_append({"event": "upload_failed", "kind": "final", "file": S.FINAL_NAME, "intent": ev, "error": repr(e)[:300], "remote": rm})
            sub.unlink(missing_ok=True)
            print(now(), "올리기 실패, 결선 파일을 지웠다", name, repr(e)[:300], flush=True)
            return 2
        print(now(), "올리기 예외가 났지만 버킷의 결선 객체가 이 파일과 같아(ETag = MD5) 올린 것으로 본다", flush=True)
    rm = S.final_remote(sub)
    if not rm["match"]:  # PUT 은 끝났는데 확인이 안 된다: 파일을 남기고 사람이 확인한 뒤 verify-final 로 적는다
        what = {"unknown": f"버킷 head 가 실패해({rm.get('error')}) 확인하지 못했다", "absent": f"버킷에 {S.FINAL_NAME} 이 없다(404)"}.get(
            rm["state"], f"버킷의 {S.FINAL_NAME} 이 이 파일과 같지 않다(ETag {rm.get('etag')}, MD5 {rm['md5']})")
        S.event_append({"event": "upload_unknown" if rm["state"] == "unknown" else "upload_unverified", "kind": "final", "file": S.FINAL_NAME,
                        "intent": ev, "put": "returned", "remote": rm})
        print(now(), "확인 실패", f"PUT 은 예외 없이 끝났지만 {what}. 결선 파일은 남겨 둔다", flush=True)
        print(now(), "확인", S.final_manual_check(), flush=True)
        return 2
    S.event_append({"event": "uploaded", "kind": "final", "file": S.FINAL_NAME, "intent": ev, "sha256": digest, "remote": rm})
    DONE.mkdir(parents=True, exist_ok=True)
    for q in (p, p.with_suffix(".json")):
        shutil.move(str(q), str(DONE / q.name))
    print(now(), "pushed final", S.FINAL_NAME, name, f"ETag {rm['etag']} = MD5", flush=True)
    for _ in range(24):  # 결선은 블라인드 채점이라 결과 파일이 없을 수 있다. 일일 채점기가 이 파일을 읽으면 그 결과(오류 포함)를 남긴다
        time.sleep(10)
        try:
            res = result(S.FINAL_NAME)
        except Exception:  # noqa: BLE001
            continue
        (ROOT / "subs" / "results" / f"{S.FINAL_NAME}_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        S.event_append({"event": "scored", "kind": "final", "file": S.FINAL_NAME, "intent": ev, "score": res.get("score"), "status": res.get("status")})
        print(now(), "final result", res.get("status"), res.get("score"), flush=True)
        return 0
    S.event_append({"event": "pending", "kind": "final", "file": S.FINAL_NAME, "intent": ev, "note": "4분 안에 결과 파일 없음(블라인드 채점이면 정상)"})
    print(now(), "결과 파일 없음(4분). 블라인드 채점이면 정상이다", flush=True)
    return 0


def fire_final_cli(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m src.fire_queue fire-final",
                                 description="결선 발(10-08 공지): 넉 달 예측 큐 후보 하나를 spirited-tiger_final.parquet 로 한 번 올린다")
    ap.add_argument("name", help="큐 이름(queue/<이름>.parquet, 결선 틀 670,790행을 덮는 예측)")
    ap.add_argument("--why", required=True, help="이 후보를 결선으로 고른 이유")
    ap.add_argument("--refs", required=True, help="고를 때 참고한 서버 점수와 로컬 값")
    ap.add_argument("--dry", action="store_true", help="검사만 끝까지 돌리고 만들기와 올리기 전에 멈춘다")
    ap.add_argument("--replace-final", action="store_true",
                    help="팀 버킷의 결선 파일을 덮어쓴다(원문에 다시 올리기 규칙이 없어 기본은 막는다)")
    a = ap.parse_args(argv)
    # 단일 발과 같은 무장: 실제 발은 명령줄에서 PRC_FIRE_ARMED 를 큐 이름으로 붙여야 한다. 그 문자열은 파일과 출력에 남기지 않는다
    if not a.dry and os.environ.get("PRC_FIRE_ARMED") != a.name:
        print(now(), "refuse", "실제 발 표시 변수(PRC_FIRE_ARMED)가 이 큐 이름과 같지 않다(발은 오케스트레이터가 명령줄에서만 붙인다)", flush=True)
        return 2
    return fire_final(a.name, a.why, a.refs, a.dry, a.replace_final)


def verify_final() -> int:
    """결선 올림 확인(PUT 없음). fire-final 이 upload_unknown 이나 upload_unverified 로 끝나 남겨 둔 subs/spirited-tiger_final.parquet 을 팀 버킷
    객체와 head 로 맞춘다. ETag 가 파일 MD5 와 같고 크기가 같고 그 sha256 의 결선 intent 가 결정 장부에 있으면 uploaded(via verify)를 적고 큐 후보를
    queue/done 으로 옮긴다. 버킷에 쓰지 않으므로 무장도 --replace-final 도 필요 없다. 하나라도 어긋나면 이유를 말하고 거부한다."""
    from src import shot as S
    sub = ROOT / "subs" / S.FINAL_NAME
    if not sub.exists():
        print(now(), "refuse", f"subs/{S.FINAL_NAME} 이 없다(맞춰 볼 로컬 결선 파일이 없다)", flush=True)
        return 1
    digest = S.sha256_file(sub)
    ev = S.final_intent_for(digest)
    if not ev:
        print(now(), "refuse", f"결정 장부에 이 파일(sha256 {digest[:16]})의 결선 intent 가 없다. fire-final 이 만든 파일이 아니다", flush=True)
        return 1
    done = [e for e in S.final_events() if e.get("event") == "uploaded" and e.get("intent") == ev["n"]]
    if done:
        print(now(), "already", f"intent #{ev['n']} 의 올림은 이미 적혀 있다(#{done[-1]['n']}, {done[-1].get('via', 'fire-final')})", flush=True)
        return 0
    rm = S.final_remote(sub)
    if rm["state"] == "unknown":
        print(now(), "refuse", f"팀 버킷 head 가 실패했다({rm.get('error')}). 연결을 확인하고 다시 부른다", flush=True)
        return 2
    if rm["state"] == "absent":
        print(now(), "refuse", f"팀 버킷에 {S.FINAL_NAME} 이 없다(404, 올라가지 않았다). 마감 전이면 fire-final 을 다시 쏜다", flush=True)
        return 1
    if not rm["match"]:
        print(now(), "refuse", f"팀 버킷의 {S.FINAL_NAME} 이 이 파일과 다르다(ETag {rm['etag']}, 크기 {rm['size']}; 파일 MD5 {rm['md5']}, "
              f"크기 {sub.stat().st_size}). 올라간 것이 intent #{ev['n']} 의 파일이 아니다", flush=True)
        return 1
    n = S.event_append({"event": "uploaded", "kind": "final", "file": S.FINAL_NAME, "intent": ev["n"], "sha256": digest, "remote": rm, "via": "verify"})
    moved = []
    for q in (QUEUE / f"{ev.get('stem')}.parquet", QUEUE / f"{ev.get('stem')}.json"):
        if q.exists():
            DONE.mkdir(parents=True, exist_ok=True)
            shutil.move(str(q), str(DONE / q.name))
            moved.append(q.name)
    print(now(), "verified", f"{S.FINAL_NAME} ETag {rm['etag']} = MD5, intent #{ev['n']}: uploaded(via verify) #{n} 를 적었다. "
          f"큐 done 으로 {moved or '옮길 것 없음'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:2] == ["fire-one"]:
        sys.exit(fire_one_cli(sys.argv[2:]))
    if sys.argv[1:2] == ["fire-final"]:
        sys.exit(fire_final_cli(sys.argv[2:]))
    if sys.argv[1:2] == ["verify-final"]:
        sys.exit(verify_final())
    sys.exit(main("--dry" in sys.argv))
