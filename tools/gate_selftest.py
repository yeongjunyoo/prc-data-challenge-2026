# SPDX-License-Identifier: GPL-3.0-only
# 발사기와 측정 묶음 등록부의 자체 시험(플레이북 §0 G2 측정 단계, G6). 진짜 GATE.json, 장부, 등록부는 건드리지 않는다.
# 묶음 예외(src/exception.py)가 있으면 그 기록의 통과와 거부(해시, 후보, 다른 실패, 다른 칸, 상한, 행, 원인, 문턱, 피처, 순서, 재사용) 사례도 돈다.
# 게이트는 메모리의 열린 문으로, 장부 변조와 이미 쏜 후보는 임시 장부 사본(src.batches.LEDGER 를 바꿔 끼움)으로, 등록 거부는 임시 큐 항목
# (zz_selftest_*, 끝나면 지움)으로 부른다. 등록된 묶음이 있으면 그 묶음의 통과, 변조(예측 해시, 로컬 값, 비교 파일), 쿼터 0 사례도 돈다.
#     .venv/Scripts/python.exe -X utf8 tools/gate_selftest.py   → notes/gate_selftest.txt
import contextlib
import copy
import csv
import datetime as dt
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import src.batches as B  # noqa: E402
import src.fire_queue as fq  # noqa: E402
from src.register_batch import main as register  # noqa: E402

Q = ROOT / "queue"
gate = {"candidate_allowed": True, "diag_used": 2, "diag_limit": 0, "anchor_fired": "spirited-tiger_v1",
        "scope": {"ranking": "team_best", "split": "none", "source": "MISSION G1"}}
out = [f"run {dt.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')} KST, gate in memory: candidate_allowed true, diag_limit 0, "
       "anchor_fired spirited-tiger_v1, scope team_best/none(+source)"]


def said(fn, *a) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a)
    return f"rc {rc}, " + " / ".join(ln.strip() for ln in buf.getvalue().splitlines() if ln.strip())[:300]


out += [f"CANDIDATE, scope with an extra key: {fq.allowed({'kind': 'CANDIDATE'}, gate) or 'allowed'}",
        f"CANDIDATE, gate closed: {fq.allowed({'kind': 'CANDIDATE'}, {**gate, 'candidate_allowed': False})}",
        f"CANDIDATE, scope last/public_private: {fq.allowed({'kind': 'CANDIDATE'}, {**gate, 'scope': {'ranking': 'last', 'split': 'public_private'}})}",
        f"DIAG: {fq.allowed({'kind': 'DIAG'}, gate)}", f"second ANCHOR: {fq.allowed({'kind': 'ANCHOR'}, gate)}",
        f"unknown kind: {fq.allowed({'kind': 'PROBE'}, gate)}"]

src = ROOT / "queue" / "hold" / "010_m4b_avg2"
tmp_items = [Q / f"zz_selftest_{i}" for i in (1, 2, 3, 4, 5)]  # 최소 묶음(5)을 넘겨야 뒤의 거부 사례가 제 검사에 닿는다
real_ledger = B.LEDGER
try:
    for t in tmp_items:
        shutil.copy2(src.with_suffix(".parquet"), t.with_suffix(".parquet"))
        meta = json.loads(src.with_suffix(".json").read_text(encoding="utf-8"))
        meta.pop("manifest", None)
        t.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    names = [t.name for t in tmp_items]
    out.append(f"candidate without manifest: {B.candidate(names[0])[1]}")
    out.append(f"register two candidates: {said(register, names[:2], False)}")
    out.append(f"register a duplicate: {said(register, [names[0], names[0], names[1], names[2], names[3]], False)}")
    out.append(f"register five without manifests: {said(register, names, False)}")
    with tempfile.TemporaryDirectory() as d:
        fake = Path(d) / "ledger.csv"
        rows = B.ledger_rows()
        fields = list(rows[0].keys()) if rows else ["version", "made_kst", "file", "sha256", "method", "local_rmse", "server_rmse", "server_read_kst", "note"]
        tamper = {k: "" for k in fields} | {"version": "999", "file": "spirited-tiger_v999.parquet", "method": "CANDIDATE selftest",
                                            "note": f"x | batch=notes/batches/999.json sha_batch=0000000000000000 stem={names[2]} sha_pred=deadbeefdeadbeef uploaded=1"}
        with fake.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows + [tamper])
        B.LEDGER = fake
        st = B.ledger_state()
        out.append(f"ledger pointing at a missing batch file: problems {st['problems']}")
        out.append(f"register while the ledger shows tampering: {said(register, names, False)}")
        out.append(f"fire while the ledger shows tampering: {said(lambda: fq.batch_step(gate, 5, True))}")
        out.append(f"fired stems read from the ledger: {sorted(st['stems'])}, fired prediction hashes: {sorted(st['pred_shas'])}")
        fired_only = {**tamper, "note": f"x | stem={names[0]} sha_pred=deadbeefdeadbeef uploaded=1"}
        with fake.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows + [fired_only])
        out.append(f"ledger without tampering: problems {B.ledger_state()['problems']}")
        out.append(f"register a candidate the ledger shows as fired: {said(register, names, False)}")
finally:
    B.LEDGER = real_ledger
    for t in tmp_items:
        t.with_suffix(".parquet").unlink(missing_ok=True)
        t.with_suffix(".json").unlink(missing_ok=True)
out.append(f"dry fire through the registered batches (none registered means nothing fires): {said(lambda: fq.batch_step(gate, 5, True))}")

import datetime as _dt  # noqa: E402
import hashlib as _hl  # noqa: E402
import src.s3io as S  # noqa: E402
v1 = ROOT / "subs" / "spirited-tiger_v1.parquet"  # 이미 올린 파일: 허가가 다 맞아도 버킷 중복으로 거부되는 안전한 대상
real_gate = json.loads((ROOT / "GATE.json").read_text(encoding="utf-8"))
digest = _hl.sha256(v1.read_bytes()).hexdigest()
now_ts = _dt.datetime.now(_dt.timezone.utc).timestamp()


def push_case(permit):
    if permit is None:
        S.PERMIT.unlink(missing_ok=True)
    else:
        S.PERMIT.write_text(json.dumps(permit), encoding="utf-8")
    try:
        S.push(v1)
        return "통과(올라갔으면 안 된다)"
    except SystemExit as e:
        return str(e)
    finally:
        S.PERMIT.unlink(missing_ok=True)


out.append(f"push without a permit: {push_case(None)}")
out.append(f"push with a permit for other bytes: {push_case({'file': v1.name, 'sha256': '0' * 64, 'issued_utc_ts': now_ts, 'kind': 'CANDIDATE'})}")
out.append(f"push with an expired permit: {push_case({'file': v1.name, 'sha256': digest, 'issued_utc_ts': now_ts - 3600, 'kind': 'CANDIDATE'})}")
if not real_gate.get("candidate_allowed"):
    out.append(f"push of a CANDIDATE while GATE.json is closed: {push_case({'file': v1.name, 'sha256': digest, 'issued_utc_ts': now_ts, 'kind': 'CANDIDATE'})}")
out.append(f"push of a second ANCHOR: {push_case({'file': v1.name, 'sha256': digest, 'issued_utc_ts': now_ts, 'kind': 'ANCHOR'})}")
with tempfile.TemporaryDirectory(dir=ROOT / "notes", prefix="_selftest_batches_") as d:
    real_b, real_i = B.BATCHES, B.INDEX
    try:
        B.BATCHES = Path(d)
        B.INDEX = Path(d) / "index.csv"
        (Path(d) / "001.json").write_text("{}", encoding="utf-8")
        out.append(f"batch file missing from the index: {B.ledger_state()['problems']}")
        rel_d = Path(d).relative_to(ROOT).as_posix()
        B.INDEX.write_text("\n".join(["number,path,sha,registered_kst", f"1,{rel_d}/001.json,0000000000000000,x",
                                      f"2,{rel_d}/002.json,0000000000000000,x", ""]), encoding="utf-8")
        out.append(f"index sha differs, and an indexed file is missing: {B.ledger_state()['problems'][:2]}")
        out.append(f"batch step while the index shows a problem: {said(lambda: fq.batch_step(gate, 5, True))}")
    finally:
        B.BATCHES, B.INDEX = real_b, real_i
reg = B.entries()
if not reg:
    out.append("registered batches: none yet, so the pass, tamper and quota cases run after the first registration")
for path, e in reg:
    rel = path.relative_to(ROOT).as_posix()
    st = B.ledger_state()
    remaining = [c["stem"] for c in e["candidates"] if (rel, c["stem"]) not in st["fired"]]
    if not remaining:
        continue
    s0 = remaining[0]
    out.append(f"{path.name} {s0} as registered: {B.verify_before_shot(e, s0) or 'would fire'}")
    out.append(f"{path.name} dry batch step with quota 5: {said(lambda: fq.batch_step(gate, 5, True))}")
    out.append(f"{path.name} dry batch step with quota 0: {said(lambda: fq.batch_step(gate, 0, True))}")
    for field, val in (("pred_sha", "0" * 16), ("local_rmse", next(c for c in e["candidates"] if c["stem"] == s0)["local_rmse"] + 1.0)):
        t = copy.deepcopy(e)
        next(c for c in t["candidates"] if c["stem"] == s0)[field] = val
        out.append(f"{path.name} {s0} with {field} changed after registration: {B.verify_before_shot(t, s0)}")
    t = copy.deepcopy(e)
    for c in t["compares"]:
        c["sha"] = "0" * 16
    out.append(f"{path.name} {s0} with its compare files changed: {B.verify_before_shot(t, s0)}")
    out.append(f"register {path.name}'s five again: {said(register, [c['stem'] for c in e['candidates']], True)}")
XDIR = ROOT / "notes" / "exceptions"
xfiles = sorted(XDIR.glob("[0-9][0-9][0-9].json")) if XDIR.exists() else []
if not xfiles:
    out.append("exception records: none yet, so the exception cases run after the first record")
for xp in xfiles:
    import src.exception as X  # noqa: E402

    def mod(E: dict, keys: tuple, val) -> dict:
        t = copy.deepcopy(E)
        d = t
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = val
        return t

    xrel = xp.relative_to(ROOT).as_posix()
    E = X.load(xrel)
    names = [c["stem"] for c in E["candidates"]]
    users = [(bp, be) for bp, be in B.entries() if (be.get("exception") or {}).get("file") == xrel]
    ok, comp = B.readiness(names)
    out.append(f"{xp.name}: readiness of its candidates without the exception: all_pass {ok}, failing "
               f"{[(c.get('kind'), sorted(c.get('ref') or {})) for c in comp['checks'] if not c['pass']]}")
    if users:
        brel, xsha, mode, probe = users[0][0].relative_to(ROOT).as_posix(), users[0][1]["exception"]["sha"], "shot", [names[0]]
    else:
        brel, xsha, mode, probe = None, None, "register", names

    def xc(E2=None, stems=None, comp2=None, mode2=None, batch="default", xs="default") -> str:
        P, _ = X.check(E2 or E, xrel, stems or probe, comp2 or comp, mode2 or mode, brel if batch == "default" else batch, xsha if xs == "default" else xs)
        return "pass" if not P else "refused: " + "; ".join(P)[:220]

    extra_col = copy.deepcopy(comp)
    for c in extra_col["checks"]:
        if not c["pass"] and c.get("kind") == "feat_fill":
            c["ref"] = {**c["ref"], "ac_type": {"diff_pp": 1.0}}
            break
    out.append(f"{xp.name} {mode} with {probe} as recorded: {xc()}")
    cases = [("its file hash differs from the one bound in the batch", dict(xs="0" * 16)),
             ("a stem outside the exception", dict(stems=["zz_not_in_exception"], mode2="shot")),
             ("an extra failing readiness check", dict(comp2={**comp, "checks": comp["checks"] + [{"check": "가짜 실패(자체 시험)", "pass": False}]})),
             ("a failing fill check with a column outside the exception", dict(comp2=extra_col)),
             ("test missing-row cap lowered to 57", dict(E2=mod(E, ("caps", "test", "nan_max"), 57))),
             ("recorded test row IDs changed", dict(E2=mod(E, ("rows", "prefix", "test", "sha"), "0" * 16))),
             ("cause airport changed", dict(E2=mod(E, ("cause", "airport"), "LIRA"))),
             ("RMSE change limit tightened to 0.05", dict(E2=mod(E, ("thresholds", "rmse_change_max"), 0.05))),
             ("feature file hash changed", dict(E2=mod(E, ("feats_sha",), "0" * 16))),
             ("registration with the order reversed", dict(stems=list(reversed(names)), mode2="register"))]
    if users:
        cases += [("registration while a batch already uses it", dict(stems=names, mode2="register")),
                  ("a shot from a different batch number", dict(batch="notes/batches/999.json"))]
    for label, kw in cases:
        out.append(f"{xp.name} {label}: {xc(**kw)}")
    out.append(f"register its candidates without --exception (dry): {said(register, names, True)}")
    out.append(f"register its candidates with the exception (dry): {said(lambda: register(names, True, xrel))}")
    for bp, be in users:
        with tempfile.TemporaryDirectory(dir=ROOT / "notes", prefix="_selftest_batches_") as d:
            real_b, real_i = B.BATCHES, B.INDEX
            try:
                B.BATCHES, B.INDEX = Path(d), Path(d) / "index.csv"
                t = copy.deepcopy(be)
                t["exception"]["sha"] = "0" * 16
                (Path(d) / bp.name).write_text(json.dumps(t, ensure_ascii=False), encoding="utf-8")
                rel_t = (Path(d) / bp.name).relative_to(ROOT).as_posix()
                B.INDEX.write_text(f"number,path,sha,registered_kst\n1,{rel_t},{B.sha(rel_t)},x\n", encoding="utf-8")
                out.append(f"ledger state with {bp.name}'s bound exception hash changed: {B.ledger_state()['problems']}")
            finally:
                B.BATCHES, B.INDEX = real_b, real_i

# 재검토 5판(10-03) 사례: 예외 검사는 준비 점검 통과와 무관하게 돈다(마), 값이 빈 채움 통계는 면제하지 않는다(라), 문턱은 반올림 전 값(가),
# config 밖 명령 플래그(지적 3), 올리기 직전 확인(지적 6), 봉인(다, 지적 7), 등록 경로가 예외 분기에 닿는다(새 구멍)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import src.exception as X2  # noqa: E402
import src.register_batch as RB  # noqa: E402
from src.make_sub import TEMPLATE  # noqa: E402

for bp0, be0 in B.entries():
    xb, nxt0, rel0 = be0.get("exception"), be0["candidates"][0]["stem"], bp0.relative_to(ROOT).as_posix()
    real_ready = B.readiness
    try:
        B.readiness = lambda stems: (True, {"checks": [], "candidates": stems})
        t = copy.deepcopy(be0)
        if xb:
            t["exception"]["sha"] = "0" * 16
        out.append(f"r5 마: {bp0.name} {nxt0} when readiness passes but the bound exception hash changed: {B.verify_before_shot(t, nxt0)}")
        out.append(f"r5 마: {bp0.name} {nxt0} when readiness passes and the exception is intact: {B.verify_before_shot(be0, nxt0) or 'would fire'}")
    finally:
        B.readiness = real_ready
    if xb:
        E0 = X2.load(xb["file"])
        _, comp0 = B.readiness([nxt0])
        nan_comp = copy.deepcopy(comp0)
        for c in nan_comp["checks"]:
            if not c["pass"] and c.get("kind") == "feat_fill":
                c["ref"] = {k: {**v, "diff_pp": None} for k, v in c["ref"].items()}
        P, got0 = X2.check(E0, xb["file"], [nxt0], nan_comp, "shot", rel0, xb["sha"])
        out.append(f"r5 라: a NaN fill statistic in an allowed column: {'refused: ' + '; '.join(P)[:180] if P else 'pass (wrong)'}")
        sens = got0["sensitivity"]
        mx = max(abs(v["change_exact"]) for v in sens["candidates"].values())
        out.append(f"r5 가: largest exact RMSE change {mx:.9f}; with the limit 1e-9 under it: "
                   f"{X2.over(sens, {'rmse_change_max': mx - 1e-9, 'pair_delta_change_max': 1}) or 'pass (wrong)'}")
    m0 = json.loads((ROOT / be0["candidates"][0]["manifest"]).read_text(encoding="utf-8"))
    m1 = copy.deepcopy(m0)
    tc = m1["commands"]["test"]
    m1["commands"]["test"] = [c + " --lr 0.5" for c in (tc if isinstance(tc, list) else [tc])]
    out.append(f"r5 지적 3: a command flag outside the config: {B.config_in_commands(m1)}")
    out.append(f"r5 지적 3: {be0['candidates'][0]['manifest']} as registered: {B.config_in_commands(m0) or 'pass'}")
    with tempfile.TemporaryDirectory(dir=ROOT / "notes", prefix="_selftest_upload_") as d:
        tpl = pd.read_parquet(TEMPLATE)["MVT_ID_mvt"]

        def subfile(stem: str, name: str) -> Path:
            pred = pd.read_parquet(Q / f"{stem}.parquet").set_index("MVT_ID_mvt")["pred"].reindex(tpl.to_numpy())
            f = Path(d) / name
            pd.DataFrame({"MVT_ID_mvt": tpl.to_numpy(), "TAXITIME_SEC_mvt": np.clip(np.rint(pred.to_numpy()), 0, None).astype("int32")}).to_parquet(f, index=False)
            return f

        good = {"batch": rel0, "sha_batch": B.sha(rel0), "stem": nxt0}
        f_ok, f_other = subfile(nxt0, "ok.parquet"), subfile(be0["candidates"][1]["stem"], "other.parquet")
        out.append(f"r5 지적 6: upload check, the next candidate's own values with the shooter's permit fields: {B.upload_problem(f_ok, good) or 'pass'}")
        out.append(f"r5 지적 6: a hand-made permit naming a later candidate: {B.upload_problem(f_ok, {**good, 'stem': be0['candidates'][1]['stem']})}")
        out.append(f"r5 지적 6: the right permit fields but another candidate's values: {B.upload_problem(f_other, good)}")
        out.append(f"r5 지적 6: a hand-made permit without batch fields: {B.upload_problem(f_ok, {'file': 'x'})}")
    sp0 = B.SEALS / f"{bp0.stem}.json"
    out.append(f"r5 다: seal {sp0.relative_to(ROOT).as_posix()} as written: {B.seal_problems(sp0) or 'no problems'}")
    with tempfile.TemporaryDirectory(dir=ROOT / "notes", prefix="_selftest_seal_") as d:
        s1 = json.loads(sp0.read_text(encoding="utf-8"))
        s1["files"][next(iter(s1["files"]))] = "0" * 64
        t1 = Path(d) / sp0.name
        t1.write_text(json.dumps(s1, ensure_ascii=False, indent=1), encoding="utf-8")
        shutil.copy2(sp0.parent / (sp0.name + ".ots"), Path(d) / (sp0.name + ".ots"))
        out.append(f"r5 다: the seal with one file hash changed, next to its old proof: {B.seal_problems(t1)}")
        t2 = Path(d) / "002.json"
        shutil.copy2(sp0, t2)
        out.append(f"r5 다: a seal without its OpenTimestamps proof: {B.seal_problems(t2)}")
    with tempfile.TemporaryDirectory(dir=ROOT / "notes", prefix="_selftest_regx_") as d:
        cmp_files = [c["file"] for c in be0["compares"]]
        saved = {f: (ROOT / f).read_bytes() for f in cmp_files}
        real = (B.BATCHES, B.INDEX, RB.BATCHES, RB.INDEX)
        try:
            B.BATCHES = RB.BATCHES = Path(d)
            B.INDEX = RB.INDEX = Path(d) / "index.csv"
            names0 = [c["stem"] for c in be0["candidates"]]
            out.append(f"r5: dry registration of {bp0.name}'s five in an empty registry, with its exception: {said(lambda: register(names0, True, xb['file'] if xb else None))}")
            out.append(f"r5: the same without --exception: {said(register, names0, True)}")
        finally:
            B.BATCHES, B.INDEX, RB.BATCHES, RB.INDEX = real
            changed = [f for f, b in saved.items() if (ROOT / f).read_bytes() != b]
            for f, b in saved.items():
                (ROOT / f).write_bytes(b)
            out.append(f"r5: compare files rewritten by that dry registration and changed: {changed or 'none (deterministic)'}; originals restored")
    # 재검토 6판: 올리기 경로도 쏘기 직전 검사 전체를 돈다, 봉인은 RFC 3161 시각 증명(고정 인증서)이 있어야 하고 그 시각이 첫 발보다 앞서야 한다
    real_ready = B.readiness
    try:
        B.readiness = lambda stems: (False, {"error": "가짜 준비 점검 오류(자체 시험)"})
        with tempfile.TemporaryDirectory(dir=ROOT / "notes", prefix="_selftest_upload6_") as d:
            tpl6 = pd.read_parquet(TEMPLATE)["MVT_ID_mvt"]
            pred6 = pd.read_parquet(Q / f"{nxt0}.parquet").set_index("MVT_ID_mvt")["pred"].reindex(tpl6.to_numpy())
            f6 = Path(d) / "ok.parquet"
            pd.DataFrame({"MVT_ID_mvt": tpl6.to_numpy(), "TAXITIME_SEC_mvt": np.clip(np.rint(pred6.to_numpy()), 0, None).astype("int32")}).to_parquet(f6, index=False)
            out.append(f"r6: upload with the right permit fields and values while the pre-shot check fails: "
                       f"{B.upload_problem(f6, {'batch': rel0, 'sha_batch': B.sha(rel0), 'stem': nxt0})}")
    finally:
        B.readiness = real_ready
    out.append(f"r6: RFC 3161 token of {sp0.name}: {B.tsa_check(sp0, 'freetsa')}")
    with tempfile.TemporaryDirectory(dir=ROOT / "notes", prefix="_selftest_tsa_") as d:
        s3 = json.loads(sp0.read_text(encoding="utf-8"))
        s3["sealed_kst"] = s3["sealed_kst"] + " "
        t3 = Path(d) / sp0.name
        t3.write_text(json.dumps(s3, ensure_ascii=False, indent=1), encoding="utf-8")
        for suf in (".ots", ".freetsa.tsr"):
            shutil.copy2(sp0.parent / (sp0.name + suf), Path(d) / (sp0.name + suf))
        out.append(f"r6: a rewritten seal next to the original proofs: {B.seal_problems(t3)}")
    real_rows = B.ledger_rows
    try:
        B.ledger_rows = lambda: real_rows() + [{"version": "998", "made_kst": "2026-10-03 02:00", "note": f"x | batch={rel0} stem={nxt0}", "server_rmse": ""}]
        out.append(f"r6: a first shot of {bp0.name} at 02:00 KST, before the token's signed time: {B.seal_problems(sp0)}")
    finally:
        B.ledger_rows = real_rows
    break  # 번호가 가장 낮은 묶음 하나로 충분하다
text = "\n".join(out) + "\n"
(ROOT / "notes" / "gate_selftest.txt").write_text(text, encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
print(text, end="")
