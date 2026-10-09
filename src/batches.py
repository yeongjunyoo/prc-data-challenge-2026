# SPDX-License-Identifier: GPL-3.0-only
"""측정 묶음 등록부(플레이북 §0 G2 측정 단계). 진짜 후보 셋 이상을, 그중 어느 것의 점수도 나오기 전에 해시로 묶는다.

등록부: notes/batches/<번호>.json. 한 번 쓰면 고치지 않고 지우지 않는다. 쏠 때마다 장부 note 에 batch, sha_batch, stem, manifest, sha_manifest,
    spec, feats, holdout, sha_pred, uploaded=1 이 남는다.
장부가 불변 기록이다(ledger_state): 장부가 가리키는 묶음 파일이 없거나 해시가 다르면 등록과 후보 발사를 모두 멈춘다. 장부에 나온 이름이나
    예측(sha_pred)은 다시 등록하지 않는다. 묶음은 모든 후보가 쏘이면 끝나고, 끝나기 전에는 다음 묶음을 쏘지 않는다(fire_queue).
후보 json(queue/<이름>.json): kind CANDIDATE, method, local_rmse, note, manifest(preds/manifest_<spec>.json).
manifest: spec, config(dict, 켜기 플래그는 true), commands{holdout, test, forward}(시드마다면 목록), seeds(목록), feats, feats_sha,
    code_sha{파일 경로: sha}, holdout, test, forward, es_valid_ids(각 {path, sha}), written_kst.
    config 의 플래그와 값이 세 명령에 다 있어야 하고, forward 명령에는 --setting fwd 가 있어야 한다. code_sha 의 파일마다 preds/code/<sha>_<파일 이름>
    사본이 있고 해시가 맞아야 한다. 조기 종료 행은 자의 행과 하나도 겹치지 않아야 한다. 옛 후보(잴 행으로 조기 종료)는 등록되지 않는다.
묶음 예외(src/exception.py): 준비 점검에서 걸린 칸은 원칙적으로 데이터나 피처에서 고친다. 감사가 승인한 한정 예외는 그 묶음 파일의 exception 에
    묶이고, 쏘기 직전 준비 점검이 걸리면 같은 예외 검사가 그 범위 안인지 다시 잰다. 예외 파일이 등록 뒤 바뀌면 장부 변조와 같이 멈춘다.
sha 는 파일 바이트 sha256 의 앞 16자다.
"""
from __future__ import annotations

import csv
import datetime as dt
import functools
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.ruler import RULER_MONTHS, ruler_frame, rmse_of  # noqa: E402

QUEUE, BATCHES, LEDGER = ROOT / "queue", ROOT / "notes" / "batches", ROOT / "subs" / "ledger.csv"
KST = ZoneInfo("Asia/Seoul")
RMSE_TOL, MIN_BATCH = 0.01, 5  # 묶음 하나가 G2 닫힘의 최소 판정 수(5)를 다 채운다(재검토 4판: 첫 묶음을 보고 다음 묶음을 설계하지 못하게)
INDEX = BATCHES / "index.csv"  # 등록 때 한 줄씩 더한다: 번호, 경로, sha, 시각. 첫 발 전의 등록부 변경을 잡는다
SEALS = ROOT / "notes" / "seals"  # 묶음마다 봉인(딸린 파일 전부의 sha256 과 색인 줄). tools/seal_batch.py 가 OpenTimestamps 로 외부 시각을 받는다
RUN_FLAGS = {"--setting", "--seed"}  # config 밖에서 명령에 올 수 있는 실행 플래그. 그 밖의 플래그는 config 에 있어야 한다(재검토 5판)
OPENSSL = r"C:\Program Files\Git\mingw64\bin\openssl.exe" if Path(r"C:\Program Files\Git\mingw64\bin\openssl.exe").exists() else (shutil.which("openssl") or "openssl")
# RFC 3161 시각 증명 기관과 고정한 인증서(sha256). 인증서 파일을 바꾸면 검증을 멈춘다. 서명 시각은 기관이 서명해 우리가 앞당길 수 없다(재검토 6판)
TSA = {"freetsa": {"url": "https://freetsa.org/tsr",
                   "cafile": "notes/seals/tsa/freetsa_cacert.pem", "cafile_sha256": "2151b61137ffa86bf664691ba67e7da0b19f98c758e3d228d5d8ebf27e044438",
                   "untrusted": "notes/seals/tsa/freetsa_tsa.crt", "untrusted_sha256": "8bfb0305bb64e2571ca507552ef3245cb1c2fee8728e0ff8689225081ea13467"}}
OTS_HEADER = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
NEED = ("spec", "config", "commands", "seeds", "feats", "feats_sha", "code_sha", "holdout", "test", "forward", "es_valid_ids", "written_kst")


def sha(p) -> str:
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()[:16]


def now() -> str:
    return dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


@functools.lru_cache(maxsize=1)
def hold() -> pd.DataFrame:
    R = ruler_frame()
    return R[R["month"].isin(RULER_MONTHS)]


def has_flag(tokens: list[str], k: str, v) -> bool:
    if v is True:
        return k in tokens
    return f"{k}={v}" in tokens or any(tokens[i] == k and tokens[i + 1] == str(v) for i in range(len(tokens) - 1))


def config_in_commands(m: dict) -> str | None:
    """config 의 플래그와 값이 holdout, test, forward 명령마다 다 있고, 명령에 config 와 RUN_FLAGS 밖의 플래그가 없고, forward 명령이 --setting fwd 인가."""
    cfg, cmds = m.get("config"), m.get("commands")
    if not isinstance(cfg, dict) or not cfg or not isinstance(cmds, dict):
        return "manifest 의 config(dict)와 commands(dict)가 없다"
    for which in ("holdout", "test", "forward"):
        if not cmds.get(which):
            return f"manifest 의 commands 에 {which} 가 없다"
        for c in cmds[which] if isinstance(cmds[which], list) else [cmds[which]]:
            t = c.split()
            for k, v in cfg.items():
                if not has_flag(t, k, v):
                    return f"manifest 의 commands.{which} 에 config {k} {'' if v is True else v} 가 없다"
            extra = sorted({x.split("=", 1)[0] for x in t if x.startswith("--")} - set(cfg) - RUN_FLAGS)
            if extra:
                return f"manifest 의 commands.{which} 에 config 에 없는 플래그 {', '.join(extra)} 가 있다"
            if which == "forward" and not has_flag(t, "--setting", "fwd"):
                return "manifest 의 commands.forward 가 --setting fwd 가 아니다"
    return None


def manifest_of(meta: dict) -> tuple[dict | None, str | None]:
    """manifest 를 읽고 칸, 설정과 명령, 파일 해시, 피처 판 해시, 코드 사본을 모두 다시 확인한다."""
    mp = meta.get("manifest")
    if not mp or not (ROOT / mp).exists():
        return None, "manifest 가 없다"
    m = json.loads((ROOT / mp).read_text(encoding="utf-8"))
    need = [k for k in NEED if not m.get(k)]
    if need:
        return m, f"manifest 에 없는 칸: {', '.join(need)}"
    why = config_in_commands(m)
    if why:
        return m, why
    for k in ("holdout", "test", "forward", "es_valid_ids"):
        f = m[k]["path"]
        if not (ROOT / f).exists():
            return m, f"manifest 의 {k} 파일이 없다({f})"
        if sha(f) != m[k]["sha"]:
            return m, f"manifest 의 {k} 해시가 파일과 다르다({f})"
    if not (ROOT / "cache" / f"{m['feats']}.parquet").exists() or sha(f"cache/{m['feats']}.parquet") != m["feats_sha"]:
        return m, f"피처 판 {m['feats']} 의 파일이 없거나 해시가 manifest 와 다르다"
    for f, h in m["code_sha"].items():
        snap = f"preds/code/{h}_{Path(f).name}"
        if not (ROOT / snap).exists() or sha(snap) != h:
            return m, f"코드 사본 {snap} 이 없거나 해시가 다르다"
    return m, None


def es_overlap(m: dict) -> int:
    return int(pd.read_parquet(ROOT / m["es_valid_ids"]["path"])["MVT_ID_mvt"].isin(hold().index).sum())


def candidate(stem: str, qdir: Path | None = None) -> tuple[dict, str | None]:
    """큐 후보 하나를 검사한다. 돌려주는 info 는 등록부에 적는 값이다. qdir 은 fire-one 의 dry 가 쏜 후보(queue/done)를 볼 때만 준다."""
    p = (qdir or QUEUE) / f"{stem}.parquet"
    if not p.exists() or not p.with_suffix(".json").exists():
        return {}, f"큐에 {stem} 이 없다"
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("kind") != "CANDIDATE":
        return {}, f"{stem} 은 CANDIDATE 가 아니다"
    m, why = manifest_of(meta)
    if why:
        return {}, f"{stem}: {why}"
    if sha(p.relative_to(ROOT).as_posix()) != m["test"]["sha"]:
        return {}, f"{stem}: 큐 예측이 manifest 의 시험 예측과 다르다(해시)"
    rm = rmse_of(m["holdout"]["path"], hold())
    if abs(rm - float(meta["local_rmse"])) > RMSE_TOL:
        return {}, f"{stem}: json 의 local_rmse {meta['local_rmse']} 와 홀드아웃을 다시 잰 값 {rm:.3f} 이 다르다"
    ov = es_overlap(m)
    if ov:
        return {}, f"{stem}: 조기 종료 행 {ov}개가 자의 행과 겹친다"
    return {"stem": stem, "manifest": meta["manifest"], "manifest_sha": sha(meta["manifest"]), "pred_sha": m["test"]["sha"],
            "holdout": m["holdout"]["path"], "holdout_sha": m["holdout"]["sha"], "local_rmse": round(rm, 3), "seeds": len(m["seeds"]),
            "feats": m["feats"], "spec": m["spec"]}, None


def note_tokens(note: str) -> dict:
    return {t.split("=", 1)[0]: t.split("=", 1)[1] for t in (note or "").split() if "=" in t}


def ledger_rows() -> list[dict]:
    return list(csv.DictReader(open(LEDGER, encoding="utf-8"))) if LEDGER.exists() else []


def ledger_state() -> dict:
    """장부에서 쏜 후보와 변조 흔적을 읽는다. fired: (묶음 경로, 이름) → scored | pending."""
    problems, fired, stems, pred_shas = [], {}, set(), set()
    for r in ledger_rows():
        t = note_tokens(r.get("note", ""))
        if "batch" in t:
            if not (ROOT / t["batch"]).exists():
                problems.append(f"v{r['version']}: 장부가 가리키는 묶음 파일 {t['batch']} 이 없다")
            elif sha(t["batch"]) != t.get("sha_batch"):
                problems.append(f"v{r['version']}: 묶음 파일 {t['batch']} 의 해시가 쏠 때와 다르다")
        if t.get("stem"):
            stems.add(t["stem"])
            fired[(t.get("batch"), t["stem"])] = "scored" if r.get("server_rmse") else "pending"
        if t.get("sha_pred"):
            pred_shas.add(t["sha_pred"])
    idx = {r["path"]: r["sha"] for r in csv.DictReader(open(INDEX, encoding="utf-8"))} if INDEX.exists() else {}
    for p in sorted(BATCHES.glob("*.json")) if BATCHES.exists() else []:
        rel = p.relative_to(ROOT).as_posix()
        if rel not in idx:
            problems.append(f"등록 색인에 없는 묶음 파일 {rel}")
        elif sha(rel) != idx[rel]:
            problems.append(f"묶음 파일 {rel} 의 해시가 등록 색인과 다르다")
        try:
            x = json.loads(p.read_text(encoding="utf-8")).get("exception")
        except (ValueError, AttributeError):
            x = None
        if x and (not (ROOT / x["file"]).exists() or sha(x["file"]) != x["sha"]):
            problems.append(f"묶음 {rel} 의 예외 기록 {x['file']} 이 없거나 등록 때와 해시가 다르다")
        sp = SEALS / f"{p.stem}.json"
        problems += seal_problems(sp) if sp.exists() else [f"묶음 {rel} 의 봉인 {sp.relative_to(ROOT).as_posix()} 이 없다"]
    for rel in idx:
        if not (ROOT / rel).exists():
            problems.append(f"등록 색인의 묶음 파일 {rel} 이 없다")
    return {"problems": problems, "fired": fired, "stems": stems, "pred_shas": pred_shas}


def entries() -> list[tuple[Path, dict]]:
    BATCHES.mkdir(parents=True, exist_ok=True)
    return [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(BATCHES.glob("*.json"))]


def registered_stems() -> set[str]:
    return {c["stem"] for _, e in entries() for c in e["candidates"]}


def readiness(stems: list[str]) -> tuple[bool, dict]:
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "src.composition", "--only", *stems], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if r.returncode not in (0, 1):
        return False, {"error": (r.stderr or r.stdout)[-400:]}
    out = json.loads((ROOT / "notes" / "composition.json").read_text(encoding="utf-8"))
    return bool(out.get("all_pass")) and set(stems) <= set(out.get("candidates", [])), out


def verify_before_shot(entry: dict, stem: str) -> str | None:
    """후보 하나를 쏘기 직전에 다시 검사한다: 등록 때 묶은 해시와 로컬 값이 그대로이고, 그 후보가 든 비교 파일이 그대로이며, 준비 점검을 지금 통과한다."""
    reg = next(c for c in entry["candidates"] if c["stem"] == stem)
    info, why = candidate(stem)
    if why:
        return why
    for f in ("manifest_sha", "pred_sha", "holdout_sha"):
        if info[f] != reg[f]:
            return f"{stem}: 등록 뒤 {f} 가 바뀌었다"
    if abs(info["local_rmse"] - reg["local_rmse"]) > RMSE_TOL:
        return f"{stem}: 등록 뒤 로컬 값이 바뀌었다"
    for c in entry["compares"]:
        if stem in (c["a"], c["b"]) and sha(c["file"]) != c["sha"]:
            return f"{stem}: 등록 뒤 비교 파일 {c['file']} 이 바뀌었다"
    ok, out = readiness([stem])
    x = entry.get("exception")
    if out.get("error") or (not ok and not x):
        return f"{stem}: 준비 점검을 지금 다시 돌려 통과하지 못했다({out.get('error') or [c['check'] for c in out.get('checks', []) if not c['pass']][:3]})"
    if not x:
        return None
    from src import exception as X  # 순환 import 를 피해 여기서 부른다. 예외가 묶인 묶음은 준비 점검이 통과해도 매 발 같은 검사를 돈다(재검토 5판)
    probs, got = X.check(X.load(x["file"]), x["file"], [stem], out, "shot", f"notes/batches/{entry['number']:03d}.json", x["sha"])
    if probs:
        return f"{stem}: 묶음 예외 검사를 통과하지 못했다({'; '.join(probs)[:400]})"
    print(now(), f"{stem}: 묶음 예외 {x['file']} 검사 통과(준비 점검 {'통과' if ok else '미통과, 걸린 칸은 예외 범위 안'}, 결측 {got['nan_counts']})", flush=True)
    return None


def seal_files(entry: dict) -> list[str]:
    """봉인에 넣는 파일: 묶음 파일, 등록 때 준비 점검 기록, 비교 파일, 후보마다 manifest 와 홀드아웃, 시험, 전방 예측, 조기 종료 행, 예외 기록과 그 근거."""
    fs = [f"notes/batches/{entry['number']:03d}.json", entry["readiness"]["file"]] + [c["file"] for c in entry["compares"]]
    for c in entry["candidates"]:
        m = json.loads((ROOT / c["manifest"]).read_text(encoding="utf-8"))
        fs += [c["manifest"]] + [m[k]["path"] for k in ("holdout", "test", "forward", "es_valid_ids")]
    x = entry.get("exception")
    if x:
        E = json.loads((ROOT / x["file"]).read_text(encoding="utf-8"))
        fs += [x["file"]] + [f["path"] for f in E["files"].values()]
    return sorted(set(fs))


def write_seal(entry: dict) -> Path:
    """등록한 묶음의 봉인을 한 번만 쓴다: 딸린 파일 전부의 sha256 과 색인 줄. 외부 시각(OpenTimestamps)은 tools/seal_batch.py 가 받는다."""
    SEALS.mkdir(parents=True, exist_ok=True)
    rel = f"notes/batches/{entry['number']:03d}.json"
    row = next(ln for ln in INDEX.read_text(encoding="utf-8").splitlines() if ln.split(",")[1:2] == [rel])
    seal = {"batch": rel, "sealed_kst": now(), "index_row": row,
            "files": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in seal_files(entry)},
            "rule": "이 파일은 고치지 않는다. 등록과 발사는 이 해시와 지금 파일을 맞추고, 이 파일 자체는 같은 이름의 .ots(OpenTimestamps 증명) 다이제스트와 맞춘다. "
                    "외부 시각은 `ots verify` 로 본다"}
    sp = SEALS / f"{entry['number']:03d}.json"
    with open(sp, "x", encoding="utf-8") as f:
        f.write(json.dumps(seal, ensure_ascii=False, indent=1))
    return sp


def tsa_check(sp: Path, name: str) -> tuple[dt.datetime | None, str | None]:
    """봉인 파일의 RFC 3161 시각 증명 하나를 고정한 인증서로 openssl 이 검증하고, 기관이 서명한 시각(UTC)을 돌려준다."""
    t, tok = TSA[name], sp.parent / f"{sp.name}.{name}.tsr"
    if not tok.exists():
        return None, f"{name} 시각 증명이 없다"
    for k in ("cafile", "untrusted"):
        if not (ROOT / t[k]).exists() or hashlib.sha256((ROOT / t[k]).read_bytes()).hexdigest() != t[k + "_sha256"]:
            return None, f"{name} 인증서 {t[k]} 가 고정한 해시와 다르다"
    v = subprocess.run([OPENSSL, "ts", "-verify", "-data", str(sp), "-in", str(tok), "-CAfile", str(ROOT / t["cafile"]), "-untrusted", str(ROOT / t["untrusted"])],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if "Verification: OK" not in v.stdout:
        return None, f"{name} 시각 증명이 봉인 파일과 맞지 않거나 서명이 검증되지 않는다"
    r = subprocess.run([OPENSSL, "ts", "-reply", "-in", str(tok), "-text"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("Time stamp:")), "")
    try:
        return dt.datetime.strptime(" ".join(line.split()[2:]), "%b %d %H:%M:%S %Y %Z").replace(tzinfo=dt.timezone.utc), None
    except ValueError:
        return None, f"{name} 시각 증명의 시각을 읽지 못했다({line[:60]})"


def first_shot_utc(batch_rel: str) -> dt.datetime | None:
    """장부에서 그 묶음의 가장 이른 발의 시각(made_kst, 분 단위)을 UTC 로."""
    ts = [dt.datetime.strptime(r["made_kst"], "%Y-%m-%d %H:%M").replace(tzinfo=KST) for r in ledger_rows()
          if note_tokens(r.get("note", "")).get("batch") == batch_rel and r.get("made_kst")]
    return min(ts).astimezone(dt.timezone.utc) if ts else None


def seal_problems(sp: Path) -> list[str]:
    """봉인의 sha256 목록과 지금 파일, 색인 줄, 봉인 파일과 그 .ots 증명의 다이제스트, 그리고 RFC 3161 시각 증명(고정 인증서로 검증, 서명 시각이 그 묶음의 첫 발보다
    앞서야 한다)을 맞춘다. 대기 중인 .ots 만으로는 독립 기록이 아니다(재검토 6판). 우리가 봉인과 증명을 함께 바꾸면 새 시각 증명은 첫 발보다 늦게 찍힌다."""
    rel = sp.relative_to(ROOT).as_posix()
    s = json.loads(sp.read_text(encoding="utf-8"))
    P = [f"봉인 {rel}: {f} 이 없거나 봉인 때와 다르다" for f, h in s["files"].items()
         if not (ROOT / f).exists() or hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
    if not INDEX.exists() or s["index_row"] not in INDEX.read_text(encoding="utf-8").splitlines():
        P.append(f"봉인 {rel} 의 색인 줄이 등록 색인에 없다")
    ots = sp.parent / (sp.name + ".ots")
    if not ots.exists():
        return P + [f"봉인 {rel} 의 OpenTimestamps 증명({ots.name})이 없다"]
    b, i = ots.read_bytes(), len(OTS_HEADER)
    if not (b.startswith(OTS_HEADER) and b[i] == 1 and b[i + 1] == 0x08 and b[i + 2:i + 34] == hashlib.sha256(sp.read_bytes()).digest()):
        P.append(f"봉인 {rel} 이 OpenTimestamps 증명의 다이제스트와 다르다")
    good, why = [], []
    for name in TSA:
        g, w = tsa_check(sp, name)
        (good.append(g) if g else why.append(w))
    if not good:
        return P + [f"봉인 {rel} 에 검증되는 RFC 3161 시각 증명이 없다({'; '.join(why)})"]
    first = first_shot_utc(s["batch"])
    if first and min(good) > first:
        P.append(f"봉인 {rel} 의 시각 증명({min(good):%Y-%m-%d %H:%M:%S} UTC)이 이 묶음의 첫 발({first:%Y-%m-%d %H:%M} UTC)보다 늦다")
    return P


def upload_problem(sub: Path, permit: dict) -> str | None:
    """올리기 직전(src/s3io.py push, CANDIDATE) 다시 본다. 허가를 손으로 만들어도 지금 차례인 등록 후보의 예측 말고는 올리지 못하게 한다(재검토 5판):
    GATE.json scope, 장부와 색인과 봉인의 변조 흔적, 허가의 묶음과 후보가 지금 차례(번호가 가장 낮은 끝나지 않은 묶음의 쏘지 않은 첫 후보)인지,
    그 후보의 큐 예측이 등록 해시와 같은지, 올릴 파일의 값이 그 예측을 제출 틀 순서로 반올림하고 0 아래를 자른 값과 같은지."""
    from src.make_sub import TEMPLATE
    gate = json.loads((ROOT / "GATE.json").read_text(encoding="utf-8"))
    sc = gate.get("scope") or {}
    if (sc.get("ranking"), sc.get("split")) != ("team_best", "none"):
        return "GATE.json scope 의 ranking, split 이 team_best, none 이 아니다"
    st = ledger_state()
    if st["problems"]:
        return "변조 흔적: " + "; ".join(st["problems"])[:300]
    for path, e in entries():
        rel = path.relative_to(ROOT).as_posix()
        left = [c for c in e["candidates"] if (rel, c["stem"]) not in st["fired"]]
        if left:
            break
    else:
        return "쏠 차례인 등록 묶음이 없다"
    nxt = left[0]
    if (permit.get("batch"), permit.get("stem"), permit.get("sha_batch")) != (rel, nxt["stem"], sha(rel)):
        return f"허가의 묶음과 후보({permit.get('batch')} {permit.get('stem')})가 지금 차례({rel} {nxt['stem']})와 다르다"
    q = QUEUE / f"{nxt['stem']}.parquet"
    if not q.exists() or sha(q.relative_to(ROOT).as_posix()) != nxt["pred_sha"]:
        return f"{nxt['stem']} 의 큐 예측이 없거나 등록 해시와 다르다"
    tpl = pd.read_parquet(TEMPLATE)["MVT_ID_mvt"].reset_index(drop=True)
    pred = pd.read_parquet(q).set_index("MVT_ID_mvt")["pred"].reindex(tpl.to_numpy())
    if pred.isna().any():
        return f"{nxt['stem']} 의 예측이 제출 틀의 {int(pred.isna().sum())}행에 없다"
    want = np.clip(np.rint(pred.to_numpy()), 0, None).astype("int32")
    got = pd.read_parquet(sub)
    if not got["MVT_ID_mvt"].reset_index(drop=True).equals(tpl) or not np.array_equal(got["TAXITIME_SEC_mvt"].to_numpy(), want):
        return "올릴 파일의 값이 등록된 예측에서 나온 값과 다르다"
    why = verify_before_shot(e, nxt["stem"])  # 허가가 손으로 만든 것이어도 쏘기 직전 검사 전체(해시, 비교 파일, 준비 점검, 예외 검사)를 여기서 다시 돈다(재검토 6판)
    return f"쏘기 직전 검사 실패: {why}" if why else None
