# SPDX-License-Identifier: GPL-3.0-only
# 단일 발 체제(10-05)의 자체 시험: fire-one(src/fire_queue.py fire_single), 올리기 직전 확인(src/shot.py upload_problem, src/s3io.py push),
# 엿보기 막이, 막는 것 셋(NaN, 빠진 행, 시험에서 통째로 빈 칸), tools/enqueue.py --readiness-warn, 묶음 길 스위치(GATE.json firing).
# 진짜 GATE.json, 장부, 결정 장부, 큐는 바꾸지 않는다: 게이트는 임시 파일(fire_queue.GATE, shot.GATE 를 바꿔 끼움), 결정 장부는 임시 파일(shot.EVENTS),
# 쿼터는 바꿔 끼운 shot.bucket_state, 큐 항목은 zz_selftest_*(끝나면 지움)로 부른다. push 는 이미 올린 v16 만 대상으로 해 확인이 다 지나도 버킷 중복으로
# 막힌다. 준비 점검 전체(3 GB 를 넘는다)는 대부분 readiness_warning 을 바꿔 끼우고, 271 하나만 진짜로 불러 기록 대체 경로를 본다.
# 마지막 사례는 dry 없이 끝까지 돌리되 올리기만 가짜로 바꾸고 subs, 장부, 결정 장부, 큐 사본을 임시 폴더로 돌린다(진짜 버킷에는 아무것도 가지 않는다).
#     .venv/Scripts/python.exe -X utf8 tools/fire_one_selftest.py   → notes/fire_one_selftest.txt
import contextlib
import csv
import datetime as dt
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import src.batches as B  # noqa: E402
import src.fire_queue as fq  # noqa: E402
import src.s3io as S3  # noqa: E402
import src.shot as S  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
out = [f"run {dt.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')} KST"]
TMP = Path(tempfile.mkdtemp(dir=ROOT / "notes", prefix="_selftest_fireone_"))
REAL = {"fq.GATE": fq.GATE, "S.GATE": S.GATE, "S.EVENTS": S.EVENTS, "S.bucket_state": S.bucket_state, "fq.readiness_warning": fq.readiness_warning}
gate_real = json.loads((ROOT / "GATE.json").read_text(encoding="utf-8"))
tpl = S.template_ids()
U171, U201, U288, U271, F131 = "171_m6r25sgc100w", "201_m6r25sgcbrw", "288_m6r25sgcbxadsbw", "271_m6r25sgc75adsbw", "131_m6r25sgcbxrw"


def said(fn, keep=("BLOCK", "WARN", "would", "refuse", "fired", "stop", "묶음 경로")) -> str:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = fn()
    except Exception as e:  # noqa: BLE001  다른 작업이 큐에 쓰는 중인 파일을 읽은 경우 따위: 시험을 멈추지 않고 적는다
        rc = f"exception {type(e).__name__}: {str(e)[:120]}"
    lines = [ln.split(" ", 2)[-1].strip() for ln in buf.getvalue().splitlines() if any(k in ln for k in keep)]
    return f"rc {rc}, " + " / ".join(lines)[:600]


def gate_file(drop=(), **over) -> Path:
    g = {k: v for k, v in {**gate_real, **over}.items() if k not in drop}
    p = TMP / f"gate_{len(list(TMP.glob('gate_*')))}.json"
    p.write_text(json.dumps(g, ensure_ascii=False), encoding="utf-8")
    return p


def no_ready(name, qdir, meta):
    return {"source": None, "ran": False, "rc": None, "error": "자체 시험: 준비 점검을 돌리지 않음"}


def left3(objs, rows):
    return {**REAL["S.bucket_state"](objs, rows), "left": 3}


def sub_file(values: np.ndarray, name: str) -> Path:
    f = TMP / name
    tbl = pa.Table.from_pandas(pd.DataFrame({"MVT_ID_mvt": tpl, "TAXITIME_SEC_mvt": values}), schema=pq.read_schema(S.TEMPLATE), preserve_index=False)
    pq.write_table(tbl, f)
    return f


def qvalues(stem: str) -> np.ndarray:
    return S.submission_values(pd.read_parquet(ROOT / "queue" / f"{stem}.parquet").set_index("MVT_ID_mvt")["pred"], tpl)


try:
    # 1) fire-one 의 막는 경로와 통과 경로(dry, 쿼터는 바꿔 끼워 남음 3)
    fq.readiness_warning = no_ready
    fq.GATE = gate_file(firing="batch")
    out.append(f"fire-one while GATE.json firing is batch: {said(lambda: fq.fire_single(U288, 'x', 'y', True))}")
    fq.GATE = gate_file(candidate_allowed=False)
    out.append(f"fire-one while candidate_allowed is false: {said(lambda: fq.fire_single(U288, 'x', 'y', True))}")
    fq.GATE = gate_file(scope={"ranking": "last", "split": "public_private"})
    out.append(f"fire-one with scope last/public_private: {said(lambda: fq.fire_single(U288, 'x', 'y', True))}")
    fq.GATE = REAL["fq.GATE"]
    out.append(f"fire-one with an empty --why: {said(lambda: fq.fire_single(U288, ' ', 'v16 266.307', True))}")
    out.append(f"fire-one with a name not in the queue: {said(lambda: fq.fire_single('zz_not_a_candidate', 'x', 'y', True))}")
    out.append(f"fire-one on {F131} (fired as v16, queue/done), real quota: {said(lambda: fq.fire_single(F131, 'x', 'y', True))}")
    out.append(f"fire-one on {U288}, real quota (today used): {said(lambda: fq.fire_single(U288, 'x', 'y', True))}")
    S.bucket_state = left3
    out.append(f"fire-one on {U288}, quota left 3: {said(lambda: fq.fire_single(U288, '자체 시험', 'v16 266.307', True))}")
    out.append(f"fire-one on {U171}, quota left 3: {said(lambda: fq.fire_single(U171, '자체 시험', 'v16 266.307', True))}")
    fq.readiness_warning = REAL["fq.readiness_warning"]
    out.append(f"fire-one on {U271}, quota left 3, real readiness (cannot finish under the cap, falls back to the record): "
               f"{said(lambda: fq.fire_single(U271, '자체 시험', 'v14 267.8428', True))}")
    fq.readiness_warning = no_ready
    S.bucket_state = REAL["S.bucket_state"]
    out.append(f"no file, event or permit written by the dry runs: events {S.EVENTS.exists()}, permit {S3.PERMIT.exists()}, "
               f"subs files {len(list((ROOT / 'subs').glob('spirited-tiger_v*.parquet')))}")

    # 2) 올리기 직전 확인(src/s3io.py push). 대상은 이미 올린 v16 이라 확인이 다 지나도 버킷 중복으로 막힌다
    S.EVENTS = TMP / "events.jsonl"
    v16 = ROOT / "subs" / "spirited-tiger_v16.parquet"
    d16 = S.sha256_file(v16)

    def push_case(permit) -> str:
        S3.PERMIT.write_text(json.dumps(permit), encoding="utf-8")
        try:
            S3.push(v16)
            return "통과(올라갔으면 안 된다)"
        except SystemExit as e:
            return str(e)[:300]
        finally:
            S3.PERMIT.unlink(missing_ok=True)

    now_ts = dt.datetime.now(dt.timezone.utc).timestamp()
    base = {"file": v16.name, "sha256": d16, "issued_utc_ts": now_ts, "kind": "CANDIDATE"}
    out.append(f"push of v16 with a batch permit while firing is single: {push_case(base)}")
    out.append(f"push of v16 with a single permit and no intent: {push_case({**base, 'mode': 'single', 'stem': F131, 'sha_pred': 'f440baacf8ae4b85'})}")
    S.event_append({"event": "intent", "file": v16.name, "sha256": d16, "stem": F131, "why": "자체 시험", "refs": "v16", "sha_pred": "f440baacf8ae4b85"})
    out.append(f"push of v16 with a single permit and a forged intent (candidate already in queue/done): "
               f"{push_case({**base, 'issued_utc_ts': dt.datetime.now(dt.timezone.utc).timestamp(), 'mode': 'single', 'stem': F131, 'sha_pred': 'f440baacf8ae4b85'})}")

    # 3) shot.upload_problem 을 직접(올리지 않는다): 171 의 값으로 만든 파일
    f171 = sub_file(qvalues(U171), "zz_171.parquet")
    d171 = S.sha256_file(f171)
    info171, _ = B.candidate(U171)
    S.event_append({"event": "intent", "file": f171.name, "sha256": d171, "stem": U171, "why": "자체 시험", "refs": "v16 266.307", "sha_pred": info171["pred_sha"]})
    ok_permit = {"file": f171.name, "sha256": d171, "issued_utc_ts": dt.datetime.now(dt.timezone.utc).timestamp(), "kind": "CANDIDATE", "mode": "single",
                 "stem": U171, "sha_pred": info171["pred_sha"]}
    out.append(f"upload check of {U171}'s values, real quota: {S.upload_problem(f171, ok_permit) or 'pass'}")
    S.bucket_state = left3
    out.append(f"upload check of {U171}'s values, quota left 3: {S.upload_problem(f171, ok_permit) or 'pass'}")
    out.append(f"  ... with a permit naming {U201}: {S.upload_problem(f171, {**ok_permit, 'stem': U201})}")
    out.append(f"  ... with a permit issued before the intent: {S.upload_problem(f171, {**ok_permit, 'issued_utc_ts': 0})}")
    f201 = sub_file(qvalues(U201), "zz_201_as_171.parquet")
    S.event_append({"event": "intent", "file": f201.name, "sha256": S.sha256_file(f201), "stem": U171, "why": "자체 시험", "refs": "v16", "sha_pred": info171["pred_sha"]})
    later = lambda: dt.datetime.now(dt.timezone.utc).timestamp()  # noqa: E731  허가는 intent 뒤에 쓴다
    out.append(f"  ... {U201}'s values under {U171}'s intent and permit: {S.upload_problem(f201, {**ok_permit, 'file': f201.name, 'issued_utc_ts': later()})}")
    fe = sub_file(qvalues(U171), "zz_171_emptywhy.parquet")
    S.event_append({"event": "intent", "file": fe.name, "sha256": S.sha256_file(fe), "stem": U171, "why": " ", "refs": "v16", "sha_pred": info171["pred_sha"]})
    out.append(f"  ... with an intent whose why is empty: {S.upload_problem(fe, {**ok_permit, 'file': fe.name, 'issued_utc_ts': later()})}")
    S.GATE = gate_file(firing="batch")
    out.append(f"  ... while GATE.json firing is batch: {S.upload_problem(f171, ok_permit)}")
    S.GATE = REAL["S.GATE"]
    S.bucket_state = REAL["S.bucket_state"]

    # 4) 엿보기 막이
    rows = S.ledger_rows()
    v16v = pd.read_parquet(v16)["TAXITIME_SEC_mvt"].to_numpy()
    rng = np.random.default_rng(0)

    def bump(frac: float) -> np.ndarray:
        x = v16v.copy()
        idx = rng.choice(len(x), int(round(frac * len(x))), replace=False)
        x[idx] += 1
        return x

    for label, vals in (("v16 itself", v16v), ("v16 with 0.5% of rows changed by 1 s", bump(0.005)), ("v16 with 0.99% of rows changed", bump(0.0099)),
                        ("v16 with 2% of rows changed", bump(0.02)), (f"{U171}'s values", qvalues(U171))):
        g = S.guard(vals, tpl, rows)
        out.append(f"guard, {label}: {'pass' if g['pass'] else 'refused'}, nearest {g['nearest']} {g['min_diff_frac']:.4%}")
    pair = []
    for r in rows:
        if int(r["version"]) < 4:
            continue
        vals = pd.read_parquet(ROOT / "subs" / r["file"])["TAXITIME_SEC_mvt"].to_numpy()
        g = S.guard(vals, tpl, [x for x in rows if x["file"] != r["file"]])
        pair.append(f"v{r['version']} {g['min_diff_frac']:.1%}({g['nearest'].split('_v')[1].split('.')[0]})")
    out.append("guard on each past candidate against all the others (min share of differing rows, nearest version): " + ", ".join(pair))

    # 5) 막는 것: NaN, 빠진 행, 같은 ID, 시험에서 통째로 빈 칸
    p171 = pd.read_parquet(ROOT / "queue" / f"{U171}.parquet")
    for label, d in (("as queued", p171), ("one NaN", p171.assign(pred=p171["pred"].where(p171.index != 5))), ("10 rows missing", p171.iloc[10:]),
                     ("a duplicated row", pd.concat([p171, p171.iloc[:1]]))):
        f = TMP / "zz_pred.parquet"
        d.to_parquet(f, index=False)
        out.append(f"pred check, {label}: {S.pred_problems(f, tpl) or 'pass'}")
    hold_ids = B.hold().index.to_numpy()
    te, ho = tpl[:500], hold_ids[:1000]
    ids = np.concatenate([te, ho])
    n = len(ids)
    syn = pd.DataFrame({"MVT_ID_mvt": ids, "normal": np.ones(n), "empty_test": np.where(np.arange(n) < 500, np.nan, 1.0),
                        "rare": np.where((np.arange(n) >= 500) & (np.arange(n) < 503), 1.0, np.nan), "all_nan": np.full(n, np.nan)})
    syn.to_parquet(TMP / "zz_feats.parquet", index=False)
    r = S.scan_empty(TMP / "zz_feats.parquet", te, ho)
    out.append(f"empty-column scan on a synthetic file (500 test, 1000 ruler rows): empty {sorted(r['empty_on_test'])}, rare {sorted(r['rare_columns_empty_on_test'])}")
    for fv in ("feats_v25", "feats_v26"):
        e = S.empty_on_test(fv, S.sha256_file(ROOT / "cache" / f"{fv}.parquet")[:16])
        out.append(f"empty-column scan, {fv}: columns {e.get('columns')}, test rows {e.get('test_rows')}, ruler rows {e.get('ruler_rows')}, "
                   f"blocks {S.empty_problems(e) or 'none'}, rare {sorted(e.get('rare_columns_empty_on_test', {})) or 'none'}")

    # 6) tools/enqueue.py --readiness-warn(준비 점검은 바꿔 끼운 결과, 시험 항목 zz_selftest_warn 은 지운다)
    spec = importlib.util.spec_from_file_location("enq", ROOT / "tools" / "enqueue.py")
    enq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enq)
    NAME, MAN = "zz_selftest_warn", "preds/manifest_m6r25sgc75adsbw.json"

    def fake_run(fails, write=True, rc=1):
        def run(cmd, **kw):
            o = cmd[cmd.index("--out") + 1]
            if write:
                comp = {"checks": [{"check": "입력 칸(자체 시험)", "value": 0.01, "band": 0.02, "pass": True}] +
                        [{"check": f"{NAME}: {c}", "value": 1.569, "band": [0.4455, 1.5545], "pass": False} for c in fails], "candidates": [NAME], "all_pass": not fails}
                (ROOT / o).write_text(json.dumps(comp, ensure_ascii=False), encoding="utf-8")
            return types.SimpleNamespace(returncode=(rc if fails or not write else 0), stdout="", stderr="" if write else "killed")
        return run

    def enq_case(fails, warn, write=True, rc=1, empty=None) -> str:
        real_run, real_empty = enq.subprocess.run, enq.S.empty_on_test
        enq.subprocess.run = fake_run(fails, write, rc)
        if empty is not None:
            enq.S.empty_on_test = lambda f, s: {"feats": f, "feats_sha": s, "test_rows": 344841, "empty_on_test": empty}
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_ = enq.main(MAN, NAME, "자체 시험", warn=warn)
            j = ROOT / "queue" / f"{NAME}.json"
            kept = json.loads(j.read_text(encoding="utf-8")) if j.exists() else None
            return (f"rc {rc_}, kept {kept is not None}" + (f", readiness_warn {kept.get('readiness_warn')}, readiness_failed {kept.get('readiness_failed')}" if kept else "")
                    + f" | {buf.getvalue().strip()[:160]}")
        finally:
            enq.subprocess.run, enq.S.empty_on_test = real_run, real_empty
            for q in (ROOT / "queue" / f"{NAME}.parquet", ROOT / "queue" / f"{NAME}.json", ROOT / "notes" / f"composition_{NAME}.json"):
                q.unlink(missing_ok=True)

    out.append(f"enqueue --readiness-warn, failing tail check: {enq_case(['3,600초 초과 예측 비중 비(자체 시험)'], True)}")
    out.append(f"enqueue without the flag, failing tail check: {enq_case(['3,600초 초과 예측 비중 비(자체 시험)'], False)}")
    out.append(f"enqueue --readiness-warn, readiness did not finish: {enq_case(['x'], True, write=False, rc=-9)}")
    out.append(f"enqueue --readiness-warn, failing check and a column empty on test: "
               f"{enq_case(['x'], True, empty={'st_fake': {'ruler_filled': 5000, 'test_filled': 0, 'expected_test_filled': 5006.0}})}")
    out.append(f"enqueue --readiness-warn, passing readiness: {enq_case([], True)}")

    # 7) 묶음 길 스위치: 깨우기가 부르는 python -m src.fire_queue(dry)
    out.append(f"batch fire path (dry) with the real GATE.json (firing single): {said(lambda: fq.main(True))}")
    fq.GATE = gate_file(drop=("firing",))
    out.append(f"batch fire path (dry) with firing removed from the gate (what 09:12 would do without the switch): {said(lambda: fq.main(True))}")
    fq.GATE = REAL["fq.GATE"]

    # 8) dry 없이 끝까지, 올리기만 가짜: build, intent, 허가, push 자리에서 진짜 shot.upload_problem, 장부 줄, 큐 옮김, 결과, 결정 장부.
    #    큐 사본, subs, 장부, 결정 장부, 결과는 모두 임시 폴더(make_sub.ROOT, make_sub.LEDGER, fire_queue.QUEUE, DONE 을 바꿔 끼움)다
    import src.make_sub as MS
    E2E = TMP / "e2e"
    (E2E / "queue").mkdir(parents=True)
    (E2E / "subs").mkdir()
    for suf in (".parquet", ".json"):
        shutil.copy2(ROOT / "queue" / f"{U288}{suf}", E2E / "queue" / f"{U288}{suf}")
    saved = (fq.QUEUE, fq.DONE, MS.ROOT, MS.LEDGER, fq.push, fq.result, fq.save_result, fq.time)
    calls = {}

    def fake_push(path):
        permit = json.loads(S3.PERMIT.read_text(encoding="utf-8"))
        S3.PERMIT.unlink()
        assert permit["file"] == path.name and permit["sha256"] == S.sha256_file(path) and permit["mode"] == "single"
        calls["upload_check"] = S.upload_problem(path, permit) or "pass"
        if calls["upload_check"] != "pass":
            raise SystemExit(calls["upload_check"])
        calls["uploaded"] = path.name  # 진짜 올리기는 하지 않는다

    try:
        fq.QUEUE, fq.DONE = E2E / "queue", E2E / "queue" / "done"
        MS.ROOT, MS.LEDGER = E2E, E2E / "subs" / "ledger.csv"
        fq.push, fq.result = fake_push, (lambda name: {"status": "SUCCESS(가짜)", "score": 123.456})
        fq.save_result = lambda name, res, v: calls.setdefault("saved", (name, res["score"], v))
        fq.time = types.SimpleNamespace(sleep=lambda s: None, time=__import__("time").time)
        fq.readiness_warning, S.bucket_state, S.EVENTS = no_ready, left3, E2E / "events.jsonl"
        r = said(lambda: fq.fire_single(U288, "자체 시험: 끝까지(올리기만 가짜)", "v16 266.307", False))
        evs = S.events()
        led = list(csv.DictReader(open(MS.LEDGER, encoding="utf-8"))) if MS.LEDGER.exists() else []
        tok = B.note_tokens(led[0]["note"]) if led else {}
        out.append(f"full path with a fake upload: {r}; check at the push point {calls.get('upload_check')}; events in order {[e['event'] for e in evs]} "
                   f"(intent why/refs written {bool(evs and evs[0].get('why') and evs[0].get('refs'))}, intent before uploaded "
                   f"{len(evs) > 1 and evs[0]['at_utc_ts'] <= evs[1]['at_utc_ts']}); ledger rows {len(led)}, note tokens mode {tok.get('mode')}, stem {tok.get('stem')}, "
                   f"uploaded {tok.get('uploaded')}, event {tok.get('event', '').split('#')[-1]}; queue copy moved to done "
                   f"{(E2E / 'queue' / 'done' / f'{U288}.json').exists()}; real queue entry untouched {(ROOT / 'queue' / f'{U288}.json').exists()}; result saved {calls.get('saved')}")
    finally:
        fq.QUEUE, fq.DONE, MS.ROOT, MS.LEDGER, fq.push, fq.result, fq.save_result, fq.time = saved
finally:
    fq.GATE, S.GATE, S.EVENTS, S.bucket_state, fq.readiness_warning = (REAL["fq.GATE"], REAL["S.GATE"], REAL["S.EVENTS"], REAL["S.bucket_state"],
                                                                     REAL["fq.readiness_warning"])
    S3.PERMIT.unlink(missing_ok=True)
    shutil.rmtree(TMP, ignore_errors=True)
text = "\n".join(out) + "\n"
(ROOT / "notes" / "fire_one_selftest.txt").write_text(text, encoding="utf-8")
print(text, end="")
