# SPDX-License-Identifier: GPL-3.0-only
"""단일 발 체제(10-05, GATE.json firing single)의 검사와 결정 장부. 측정 묶음 길(src/batches.py, fire_queue batch_step)은 남겨 두되 쓰지 않는다.

남기는 것(대회 원문이 요구하는 것과 재현): 학습 데이터로만 만든 진짜 예측(manifest 와 코드 사본, src/batches.candidate), 상식 검사(fire_queue.sane,
    2025 학습 분포만), 상수와 부분 교란 파일 금지(아래 엿보기 막이), 진단 발로 푼 시험 통계(notes/diag_server_solve.json) 미사용, 단일 올리기 경로
    (src/s3io.py push 의 일회용 허가)와 장부 밖 제출 검사, 공개 라이선스 데이터만(README 「외부 데이터」).
준비 점검(src/composition.py)은 경고다. 막는 것은 셋이다: 예측의 NaN(유한하지 않은 값), 제출 틀에서 빠진 행, 시험 행에서 통째로 빈 피처 칸.
    마지막 것은 자(2025-01, 07 출발편)의 채움 비율로 셈하면 시험에서 10행 이상 차 있어야 할 칸이 시험에서 0행인 경우다(empty_on_test, 칸마다 읽어
    메모리가 작다, 결과는 피처 판 해시로 notes/test_empty_<판>.json 에 둔다). 피처 판에 시험 행이 하나도 없어도 막는다.
엿보기 막이(guard): 올릴 정수 값이 앞서 낸 제출(장부의 파일) 가운데 하나와라도 1% 미만 행에서만 다르면 거부한다. 고른 행만 바꾼 파일은 그 점수 변화가
    그 행의 정답을 드러내므로 순위 이용이다(원문/render/ranking_html.txt 7줄). 같은 파일을 다시 내는 것도 여기서 막힌다.
결정 장부(subs/fire_events.jsonl, 한 줄에 사건 하나, 덧붙이기만): fire-one 은 올리기 전에 이유(why)와 참고한 서버 점수(refs), 준비 점검 경고, 막이 값을
    적고(intent), 올린 뒤(uploaded 또는 upload_failed)와 결과 뒤(scored, pending) 한 줄씩 더한다. push 는 그 파일과 해시의 intent 가 없으면 올리지 않는다.
쿼터: 오늘(UTC) 팀 버킷의 채점 결과 수와 올린 제출 수 중 큰 쪽을 쓴 것으로 센다(결과가 늦은 발도 센다).
결선(10-08 공지): fire-final 이 넉 달 결선 파일 spirited-tiger_final.parquet 하나를 올린다. 검사와 결선 자료 고정값은 아래 「결선」 절(final_upload_problem).
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KST = ZoneInfo("Asia/Seoul")
GATE = ROOT / "GATE.json"
EVENTS = ROOT / "subs" / "fire_events.jsonl"
LEDGER = ROOT / "subs" / "ledger.csv"
TEMPLATE = ROOT / "data" / "submitting.parquet"
TEAM = "spirited-tiger"
GUARD_MIN = 0.01  # 앞서 낸 모든 제출과 이 비율 이상의 행에서 달라야 한다
EMPTY_MIN_EXPECTED = 10  # 자의 채움 비율로 시험에서 이만큼 이상 차 있어야 할 칸이 0행이면 막는다(그보다 드문 칸은 정보로만 적는다)
DAILY_QUOTA = 5  # ranking.html: “There is currently a limit of 5 submissions per day”
SKIP_COLS = ("MVT_ID_mvt", "y", "src")  # src/composition.py feat_fill 과 같은 칸을 뺀다


def now_kst() -> str:
    return dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_gate() -> dict:
    return json.loads(GATE.read_text(encoding="utf-8"))


# ---- 막는 것 셋: NaN, 빠진 행, 시험에서 통째로 빈 칸 ----

def template_ids() -> np.ndarray:
    return pd.read_parquet(TEMPLATE, columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()


def pred_problems(path: Path, tpl: np.ndarray | None = None) -> list[str]:
    """예측 파일(MVT_ID_mvt, pred)의 NaN, 같은 ID 중복, 제출 틀에서 빠진 행."""
    d = pd.read_parquet(path)
    if not {"MVT_ID_mvt", "pred"} <= set(d.columns):
        return [f"{Path(path).name}: MVT_ID_mvt, pred 칸이 없다"]
    P = []
    bad = ~np.isfinite(d["pred"].to_numpy(dtype=float))
    if bad.any():
        P.append(f"{Path(path).name}: 유한하지 않은 예측(NaN 등) {int(bad.sum())}행")
    dup = d["MVT_ID_mvt"].duplicated()
    if dup.any():
        P.append(f"{Path(path).name}: 같은 MVT_ID 가 두 번 이상 나온 행 {int(dup.sum())}")
    tpl = template_ids() if tpl is None else tpl
    miss = ~np.isin(tpl, d["MVT_ID_mvt"].to_numpy())
    if miss.any():
        P.append(f"{Path(path).name}: 제출 틀 {len(tpl)}행 중 예측이 없는 행 {int(miss.sum())}")
    return P


def scan_empty(path: Path, test_ids: np.ndarray, ruler_ids: np.ndarray) -> dict:
    """피처 parquet 하나를 칸마다 읽어 시험 행과 자 행의 채움(값이 있고 NaN 이 아님) 수를 센다. 한 칸씩만 메모리에 올린다."""
    pf = pq.ParquetFile(path)
    ids = pf.read(columns=["MVT_ID_mvt"]).column(0).to_numpy()
    m_te, m_ho = np.isin(ids, test_ids), np.isin(ids, ruler_ids)
    n_te, n_ho = int(m_te.sum()), int(m_ho.sum())
    empty, rare = {}, {}
    cols = [c for c in pf.schema_arrow.names if c not in SKIP_COLS]
    for c in cols:
        v = pf.read(columns=[c]).column(0).to_pandas().notna().to_numpy()
        te, ho = int(v[m_te].sum()), int(v[m_ho].sum())
        if te == 0 and ho > 0:
            expect = ho / n_ho * n_te if n_ho else 0.0
            (empty if expect >= EMPTY_MIN_EXPECTED else rare)[c] = {"ruler_filled": ho, "test_filled": 0, "expected_test_filled": round(expect, 1)}
    return {"columns": len(cols), "test_rows": n_te, "ruler_rows": n_ho, "empty_on_test": empty, "rare_columns_empty_on_test": rare}


def empty_on_test(feats: str, feats_sha: str) -> dict:
    """피처 판 하나의 「시험에서 통째로 빈 칸」. 결과는 feats_sha 로 notes/test_empty_<판>.json 에 두고 같은 해시면 다시 쓴다."""
    out = ROOT / "notes" / f"test_empty_{feats}.json"
    if out.exists():
        c = json.loads(out.read_text(encoding="utf-8"))
        if c.get("feats_sha") == feats_sha:
            return c
    f = ROOT / "cache" / f"{feats}.parquet"
    if not f.exists() or sha256_file(f)[:16] != feats_sha:
        return {"feats": feats, "feats_sha": feats_sha, "error": f"피처 판 {feats} 의 파일이 없거나 해시가 manifest 와 다르다"}
    from src.batches import hold  # 자(src/ruler.py, 2025-01, 07 출발편)
    r = scan_empty(f, template_ids(), hold().index.to_numpy())
    r = {"feats": feats, "feats_sha": feats_sha, **r, "written_kst": now_kst(),
         "rule": f"시험 행 채움 0 이고 자의 채움 비율로 낸 시험 기대 채움이 {EMPTY_MIN_EXPECTED}행 이상이면 막는다(src/shot.py). 그보다 드문 칸은 정보"}
    out.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def empty_problems(e: dict) -> list[str]:
    if e.get("error"):
        return [e["error"]]
    if not e.get("test_rows"):
        return [f"피처 판 {e.get('feats')} 에 시험 행이 없다"]
    return [f"피처 판 {e['feats']} 의 칸 {c} 이 시험 {e['test_rows']}행에서 통째로 비었다(자 {v['ruler_filled']}행 채움, 시험 기대 {v['expected_test_filled']}행)"
            for c, v in e.get("empty_on_test", {}).items()]


def readiness_failed(comp: dict, stem: str | None = None) -> list[dict]:
    """준비 점검 출력에서 걸린 칸(경고로 적는 값). stem 을 주면 다른 후보의 칸은 뺀다(입력 칸은 남긴다)."""
    others = set(comp.get("candidates", [])) - {stem} if stem else set()
    return [{"check": c.get("check"), "value": c.get("value"), "band": c.get("band")} for c in comp.get("checks", [])
            if not c.get("pass", True) and str(c.get("check", "")).split(":")[0] not in others]


# ---- 장부, 쿼터, 엿보기 막이 ----

def ledger_rows() -> list[dict]:
    return list(csv.DictReader(open(LEDGER, encoding="utf-8"))) if LEDGER.exists() else []


def ledger_files() -> tuple[list[dict], list[str]]:
    """장부의 줄과, 그 줄이 가리키는 제출 파일이 subs/ 에 없거나 장부의 sha256 앞 16자와 다른 문제."""
    rows, probs = ledger_rows(), []
    for r in rows:
        f = ROOT / "subs" / r["file"]
        if not f.exists():
            probs.append(f"장부의 {r['file']} 이 subs/ 에 없다")
        elif sha256_file(f)[:16] != r["sha256"]:
            probs.append(f"장부의 {r['file']} 의 해시가 장부와 다르다")
    return rows, probs


def submission_values(pred: pd.Series, tpl: np.ndarray) -> np.ndarray:
    """src/make_sub.build 가 올릴 파일에 쓰는 값과 같은 식: 제출 틀 순서, 반올림, 0 아래 자름, int32."""
    return np.clip(np.rint(pred.reindex(tpl).to_numpy(dtype=float)), 0, None).astype("int32")


def guard(values: np.ndarray, tpl: np.ndarray, rows: list[dict] | None = None) -> dict:
    """올릴 값(제출 틀 순서)이 장부의 제출마다 몇 % 행에서 다른가. 가장 가까운 제출과의 비율이 GUARD_MIN 아래면 통과하지 못한다."""
    rows = ledger_rows() if rows is None else rows
    per, probs = {}, []
    for r in rows:
        f = ROOT / "subs" / r["file"]
        if not f.exists():
            probs.append(f"장부의 {r['file']} 이 subs/ 에 없어 비교하지 못했다")
            continue
        d = pd.read_parquet(f)
        prev = d["TAXITIME_SEC_mvt"].to_numpy() if np.array_equal(d["MVT_ID_mvt"].to_numpy(), tpl) else \
            d.set_index("MVT_ID_mvt")["TAXITIME_SEC_mvt"].reindex(tpl).to_numpy()
        per[r["file"]] = float(np.mean(prev != values))
    near = min(per, key=per.get) if per else None
    ok = not probs and (near is None or per[near] >= GUARD_MIN)
    return {"pass": ok, "min_diff_frac": None if near is None else round(per[near], 6), "nearest": near, "rule": f"모든 앞 제출과 {GUARD_MIN:.0%} 이상의 행에서 달라야 한다",
            "per_file": {k: round(v, 6) for k, v in per.items()}, "problems": probs}


def bucket_state(objs: list[dict], rows: list[dict]) -> dict:
    """팀 버킷 목록(src/s3io.ls)에서 오늘(UTC) 쓴 쿼터와 장부 밖 제출."""
    today = dt.datetime.now(dt.timezone.utc).date()

    def today_(o) -> bool:
        return dt.datetime.fromisoformat(o["modified"]).astimezone(dt.timezone.utc).date() == today

    subs = {o["key"] for o in objs if o["key"].startswith(f"{TEAM}_v") and o["key"].endswith(".parquet")}
    results = sum(1 for o in objs if o["key"].endswith("_result.json") and today_(o))
    uploads = sum(1 for o in objs if o["key"] in subs and today_(o))
    used = max(results, uploads)
    return {"results_today_utc": results, "uploads_today_utc": uploads, "used": used, "left": DAILY_QUOTA - used,
            "off_ledger": sorted(subs - {r["file"] for r in rows})}


# ---- 결정 장부 ----

def events() -> list[dict]:
    if not EVENTS.exists():
        return []
    return [json.loads(ln) for ln in EVENTS.read_text(encoding="utf-8").splitlines() if ln.strip()]


def event_append(ev: dict) -> int:
    """한 줄을 더하고 디스크에 내린 뒤 번호를 돌려준다. 고치거나 지우지 않는다."""
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    n = len(events()) + 1
    row = {"n": n, "at_kst": now_kst(), "at_utc_ts": dt.datetime.now(dt.timezone.utc).timestamp(), **ev}
    with open(EVENTS, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return n


def intent_for(file: str, sha256: str) -> dict | None:
    hit = [e for e in events() if e.get("event") == "intent" and e.get("file") == file and e.get("sha256") == sha256]
    return hit[-1] if hit else None


def note_text(s: str) -> str:
    """장부 note 에 넣는 자유 글. note 의 key=value 토큰(stem, sha_pred, uploaded 등)을 흉내 내지 못하게 = 를 띄우고 줄바꿈을 없앤다."""
    return " ".join(s.replace("=", " = ").split())


# ---- 올리기 직전(src/s3io.py push 가 부른다) ----

def upload_problem(sub: Path, permit: dict) -> str | None:
    """단일 발 허가(mode single)의 올리기 직전 확인. 허가를 손으로 만들어도 쏘기 직전 검사 전체를 여기서 다시 돈다(준비 점검 경고는 빼고):
    GATE.json(firing single, candidate_allowed, scope), 결정 장부의 intent(이 파일과 해시, why 와 refs, 허가보다 먼저), 큐 후보의 manifest 와 해시와
    로컬 값과 조기 종료 행, 올릴 값이 그 예측에서 나왔는지, 상식 검사, NaN 과 빠진 행, 시험에서 통째로 빈 칸, 엿보기 막이, 쿼터, 장부 밖 제출."""
    from src.batches import QUEUE, candidate
    from src.fire_queue import sane
    from src.s3io import TEAM_BUCKET, ls
    g = load_gate()
    sc = g.get("scope") or {}
    if g.get("firing") != "single":
        return "GATE.json firing 이 single 이 아니다(단일 발 허가는 그때만 쓴다)"
    if not g.get("candidate_allowed"):
        return "GATE.json candidate_allowed 가 false 다"
    if (sc.get("ranking"), sc.get("split")) != ("team_best", "none"):
        return "GATE.json scope 의 ranking, split 이 team_best, none 이 아니다"
    digest = sha256_file(sub)
    ev = intent_for(sub.name, digest)
    if not ev:
        return f"결정 장부({EVENTS.relative_to(ROOT).as_posix()})에 이 파일과 해시의 intent(이유와 참고 점수)가 없다"
    if not str(ev.get("why", "")).strip() or not str(ev.get("refs", "")).strip():
        return "intent 의 why 나 refs 가 비었다"
    if float(ev.get("at_utc_ts", 0)) > float(permit.get("issued_utc_ts", 0)):
        return "intent 가 허가보다 늦게 적혔다"
    stem = permit.get("stem")
    if not stem or ev.get("stem") != stem:
        return f"허가의 후보({stem})와 intent 의 후보({ev.get('stem')})가 다르다"
    try:
        info, why = candidate(stem)
    except SystemExit as e:
        info, why = {}, f"{stem}: {e}"
    if why:
        return why
    if info["pred_sha"] != permit.get("sha_pred") or info["pred_sha"] != ev.get("sha_pred"):
        return f"{stem}: 허가나 intent 의 sha_pred 가 큐 예측과 다르다"
    q = QUEUE / f"{stem}.parquet"
    tpl = template_ids()
    hard = pred_problems(q, tpl) + ([w] if (w := sane(q)) else [])
    m = json.loads((ROOT / info["manifest"]).read_text(encoding="utf-8"))
    hard += empty_problems(empty_on_test(m["feats"], m["feats_sha"]))
    if hard:
        return "; ".join(hard)[:400]
    want = submission_values(pd.read_parquet(q).set_index("MVT_ID_mvt")["pred"], tpl)
    got = pd.read_parquet(sub)
    if not np.array_equal(got["MVT_ID_mvt"].to_numpy(), tpl) or not np.array_equal(got["TAXITIME_SEC_mvt"].to_numpy(), want):
        return "올릴 파일의 값이 그 후보의 큐 예측에서 나온 값과 다르다"
    rows, probs = ledger_files()
    if probs:
        return "장부와 제출 파일이 어긋난다: " + "; ".join(probs)[:300]
    gd = guard(want, tpl, rows)
    if not gd["pass"]:
        return f"엿보기 막이: 앞 제출 {gd['nearest']} 와 {gd['min_diff_frac']:.4%} 행에서만 다르다({gd['rule']})"
    bs = bucket_state(ls(TEAM_BUCKET), rows)
    if bs["off_ledger"]:
        return f"장부에 없는 제출이 버킷에 있다: {', '.join(bs['off_ledger'])}"
    if bs["left"] <= 0:
        return f"오늘(UTC) 쿼터를 다 썼다(결과 {bs['results_today_utc']}, 올림 {bs['uploads_today_utc']})"
    return None


# ---- 결선(주최 10-08 공지: 넉 달 예측 한 번, 마감 2026-10-14T10:00Z) ----
# 원문(대회 사이트 10-08 판, private notes 원문/render/*_2026-10-08.txt): ranking 575~579줄 “Final Submission / Please submit you unique final submission
# using the following filename: <team-name>_final.parquet”, data 609~614줄 “The Final Datasets … final_ranking.parquet, final_submitting.parquet”,
# eligibility 556줄 “Final single submission is uploaded by 2026-10-14T12:00 CET (2026-10-14T12:00+02:00)”.
# 올리는 곳: 같은 팀 버킷(Discord 10-08 14:03Z espinielli “(4) same team bucket … i.e. <team name>_final.parquet”).
# 두 번 올리기: 원문은 “single”, “unique” 뿐이고 다시 올리면 어느 파일을 쓰는지 적지 않았다(arnavhm13 12:57Z 물음 (4)의 이 부분은 14:03Z 답에 없다).
# 그래서 팀 버킷이나 결정 장부에 결선 파일이 있으면 막고, fire-final --replace-final 을 따로 줄 때만 덮어쓴다.
# 올린 뒤 확인이 끊기면(upload_unknown, upload_unverified) 결선 파일을 남기고, fire_queue verify-final 이 PUT 없이 head 로 확인해 적는다.
FINAL_DIR = ROOT / "data" / "final_phase"
FINAL_TEMPLATE = FINAL_DIR / "final_submitting.parquet"
FINAL_NAME = f"{TEAM}_final.parquet"
FINAL_ROWS = 670_790
FINAL_COLUMNS = [("MVT_ID_mvt", "double"), ("TAXITIME_SEC_mvt", "int32")]
# 10-08 13:44Z 에 버킷 head 와 받은 파일로 잰 값(private notes research/2026-10-08-final-phase-facts.md 4절). 만들기 전과 올리기 직전에 다시 맞춘다.
# 주최가 다시 올리면 결선은 막힌다: 새 파일을 받아 예측을 다시 만든 뒤에만 이 값을 손으로 고친다.
FINAL_PINS = {
    "final_ranking.parquet": {"sha256": "4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f",
                              "etag": "8970d7057c0c72b5138020578f6bc861-6", "size": 85_853_399},
    "final_submitting.parquet": {"sha256": "b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924",
                                 "etag": "2c1f407296e217fd7d242c9d51ba27ad", "size": 3_003_722},
}
GUARD_SKIP_FINAL = ("엿보기 막이(guard)는 건너뛴다: 장부의 앞 제출은 1월과 7월 344,841행이고 결선 파일은 넉 달 670,790행이라 행 수와 행 집합이 다르다. "
                    "결선은 한 번뿐인 블라인드 채점(Discord 10-08 07:34Z “single blind ranking”)이라 점수 차로 행의 정답을 캐는 길도 없다")


def final_template_ids() -> np.ndarray:
    return pd.read_parquet(FINAL_TEMPLATE, columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()


def final_data_problems(head) -> tuple[list[str], dict]:
    """결선 자료 둘이 받을 때와 같은가: 내 파일의 sha256 과 크기, 버킷 객체의 ETag 와 크기(head 는 읽기만 하는 HeadObject, key -> dict)."""
    P, seen = [], {}
    for key, pin in FINAL_PINS.items():
        f = FINAL_DIR / key
        if not f.exists():
            P.append(f"{f.relative_to(ROOT).as_posix()} 이 없다")
            continue
        local, size = sha256_file(f), f.stat().st_size
        try:
            h = head(key)
        except Exception as e:  # noqa: BLE001  없는 객체(404)와 연결 오류 모두 막는다
            h = {"error": repr(e)[:200]}
        seen[key] = {"sha256": local, "size": size, "bucket_etag": h.get("etag"), "bucket_size": h.get("size"), "bucket_modified": h.get("modified")}
        if local != pin["sha256"] or size != pin["size"]:
            P.append(f"내 {key} 가 받을 때와 다르다(sha256 {local[:16]}, 고정값 {pin['sha256'][:16]})")
        if h.get("error"):
            P.append(f"버킷의 {key} 를 head 로 읽지 못했다: {h['error']}")
        elif h.get("etag") != pin["etag"] or h.get("size") != pin["size"]:
            P.append(f"버킷의 {key} 가 바뀌었다(ETag {h.get('etag')}, 크기 {h.get('size')}, 수정 {h.get('modified')}; 고정값 ETag {pin['etag']})")
    return P, seen


def final_file_problems(sub: Path, tpl: np.ndarray | None = None) -> list[str]:
    """올릴 결선 파일 검사: 이름, 칸 이름과 타입(결선 틀의 parquet 스키마와 같고 MVT_ID_mvt double, TAXITIME_SEC_mvt int32), 670,790행,
    결선 틀과 같은 ID 를 같은 순서로, 빈 값과 음수 없음, 상식 범위(fire_queue.sane_values, 2025 학습 출발편 분포만)."""
    from src.fire_queue import sane_values
    P = [] if sub.name == FINAL_NAME else [f"결선 파일 이름은 {FINAL_NAME} 이어야 한다: {sub.name}"]
    got = [(f.name, str(f.type)) for f in pq.read_schema(sub)]
    want = [(f.name, str(f.type)) for f in pq.read_schema(FINAL_TEMPLATE)]
    if got != want or got != FINAL_COLUMNS:
        return P + [f"칸과 타입이 결선 틀과 다르다: {got} (틀 {want})"]
    tpl = final_template_ids() if tpl is None else tpl
    d = pd.read_parquet(sub)
    if len(d) != FINAL_ROWS or len(tpl) != FINAL_ROWS:
        P.append(f"행 수가 다르다: 파일 {len(d)}, 결선 틀 {len(tpl)}, 고정 {FINAL_ROWS}")
    ids = d["MVT_ID_mvt"].to_numpy()
    if not np.array_equal(ids, tpl):
        same_set = len(ids) == len(tpl) and np.array_equal(np.sort(ids), np.sort(tpl))
        P.append("ID 집합은 같지만 행 순서가 결선 틀과 다르다" if same_set else "ID 가 결선 틀과 다르다(빠지거나 남는 행)")
    v = d["TAXITIME_SEC_mvt"]
    if v.isna().any():
        P.append(f"TAXITIME_SEC_mvt 의 빈 값(NaN, null) {int(v.isna().sum())}행")
    else:
        if (v < 0).any():
            P.append(f"음수 예측 {int((v < 0).sum())}행")
        if w := sane_values(v.astype(float)):
            P.append(w)
    return P


def final_bucket_problems(objs: list[dict], replace: bool) -> list[str]:
    """팀 버킷 목록에서 결선 파일. 결선 이름 꼴(FINAL_NAME 과 그 결과 파일)이 아닌 final 객체는 늘 막고, FINAL_NAME 이 있으면 replace 일 때만 지나간다."""
    keys = {o["key"] for o in objs}
    odd = sorted(k for k in keys if "final" in k.lower() and not k.startswith(FINAL_NAME))
    P = [f"팀 버킷에 결선 이름 꼴이 아닌 final 객체가 있다: {', '.join(odd)}"] if odd else []
    if FINAL_NAME in keys and not replace:
        P.append(f"팀 버킷에 {FINAL_NAME} 이 이미 있다. 원문은 한 번(“single”, “unique”)만 말하고 다시 올리기를 적지 않아 막는다"
                 "(덮어쓰려면 fire-final --replace-final)")
    return P


def final_events() -> list[dict]:
    return [e for e in events() if e.get("kind") == "final"]


def final_intent_for(sha256: str) -> dict | None:
    hit = [e for e in final_events() if e.get("event") == "intent" and e.get("file") == FINAL_NAME and e.get("sha256") == sha256]
    return hit[-1] if hit else None


def final_remote(sub: Path) -> dict:
    """팀 버킷의 결선 객체가 이 파일과 같은가(head 만). 한 번에 올린(PUT 하나) 객체의 ETag 는 파일 MD5 다(v33 ~ v35 에서 확인, 10-08).
    state: present(있음, match 는 ETag 와 크기 비교), absent(404, NoSuchKey 로 없음이 확인됨), unknown(그 밖의 오류라 있는지 모른다)."""
    from botocore.exceptions import ClientError
    from src.s3io import TEAM_BUCKET, head
    md5 = hashlib.md5(sub.read_bytes()).hexdigest()
    try:
        h = head(TEAM_BUCKET, FINAL_NAME)
    except ClientError as e:
        code, status = str((e.response.get("Error") or {}).get("Code", "")), (e.response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        if code in ("404", "NoSuchKey", "NotFound") or status == 404:
            return {"state": "absent", "present": False, "match": False, "md5": md5}
        return {"state": "unknown", "present": None, "match": False, "md5": md5, "error": repr(e)[:200]}
    except Exception as e:  # noqa: BLE001  연결 끊김 등: 없다고 보지 않는다
        return {"state": "unknown", "present": None, "match": False, "md5": md5, "error": repr(e)[:200]}
    return {"state": "present", "present": True, "match": h["etag"] == md5 and h["size"] == sub.stat().st_size, "md5": md5, **h}


def final_manual_check() -> str:
    """올렸는지 모를 때 사람이 하는 확인. 둘이 같으면 올라간 것이고 verify-final 이 PUT 없이 장부에 적는다."""
    return (f".venv/Scripts/python.exe -X utf8 -m src.s3io head prc-2026-{TEAM} {FINAL_NAME} 의 etag 와 md5sum subs/{FINAL_NAME} 이 같으면 "
            "올라간 것이다. 그러면 .venv/Scripts/python.exe -X utf8 -m src.fire_queue verify-final 로 적는다(PUT 없음, 무장 없음)")


def empty_on_final(feats: str, feats_sha: str) -> dict:
    """empty_on_test 의 결선판: 시험 행을 결선 틀(넉 달 출발편)로 센다. 결과는 notes/test_empty_final_<판>.json(피처 판 해시와 결선 틀 해시가 같으면 다시 쓴다)."""
    out = ROOT / "notes" / f"test_empty_final_{feats}.json"
    tsha = FINAL_PINS["final_submitting.parquet"]["sha256"]
    if out.exists():
        c = json.loads(out.read_text(encoding="utf-8"))
        if c.get("feats_sha") == feats_sha and c.get("final_template_sha256") == tsha:
            return c
    f = ROOT / "cache" / f"{feats}.parquet"
    if not f.exists() or sha256_file(f)[:16] != feats_sha:
        return {"feats": feats, "feats_sha": feats_sha, "error": f"피처 판 {feats} 의 파일이 없거나 해시가 manifest 와 다르다"}
    from src.batches import hold
    r = scan_empty(f, final_template_ids(), hold().index.to_numpy())
    r = {"feats": feats, "feats_sha": feats_sha, "final_template_sha256": tsha, **r, "written_kst": now_kst(),
         "rule": f"결선 시험 행 채움 0 이고 자의 채움 비율로 낸 기대 채움이 {EMPTY_MIN_EXPECTED}행 이상이면 막는다(src/shot.py). 그보다 드문 칸은 정보"}
    out.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def final_upload_problem(sub: Path, permit: dict) -> str | None:
    """결선 허가(kind FINAL)의 올리기 직전 확인(src/s3io.py push 가 부른다). 허가를 손으로 만들어도 fire-final 의 검사를 여기서 다시 돈다:
    GATE.json, 결선 자료 고정값(내 파일 sha256, 버킷 ETag), 결선 파일(이름, 칸과 타입, 행 수, 같은 ID 같은 순서, 빈 값, 음수, 상식 범위),
    결정 장부의 intent(kind final, 이 파일과 해시, why 와 refs, 허가보다 먼저, replace 일치), 큐 후보의 manifest 와 해시와 로컬 값과 조기 종료 행,
    올릴 값이 그 예측에서 나왔는지, 예측의 NaN 과 빠진 행, 시험에서 통째로 빈 칸, 장부와 제출 파일, 장부 밖 제출, 쿼터, 버킷의 결선 파일.
    엿보기 막이는 건너뛴다(GUARD_SKIP_FINAL)."""
    from src.batches import QUEUE, candidate
    from src.fire_queue import sane
    from src.s3io import DATA_BUCKET, TEAM_BUCKET, head, ls
    g = load_gate()
    sc = g.get("scope") or {}
    if g.get("firing") != "single":
        return "GATE.json firing 이 single 이 아니다(결선 허가는 단일 발 체제에서만 쓴다)"
    if not g.get("candidate_allowed"):
        return "GATE.json candidate_allowed 가 false 다"
    if (sc.get("ranking"), sc.get("split")) != ("team_best", "none"):
        return "GATE.json scope 의 ranking, split 이 team_best, none 이 아니다"
    if sub.name != FINAL_NAME or permit.get("file") != FINAL_NAME or permit.get("mode") != "final":
        return f"결선 허가는 {FINAL_NAME} 하나에만, mode final 로만 쓴다"
    P, _ = final_data_problems(lambda k: head(DATA_BUCKET, k))
    tpl = final_template_ids()
    P += final_file_problems(sub, tpl)
    if P:
        return "; ".join(P)[:500]
    digest = sha256_file(sub)
    ev = final_intent_for(digest)
    if not ev:
        return f"결정 장부({EVENTS.relative_to(ROOT).as_posix()})에 이 결선 파일과 해시의 intent(kind final, 이유와 참고 점수)가 없다"
    if not str(ev.get("why", "")).strip() or not str(ev.get("refs", "")).strip():
        return "intent 의 why 나 refs 가 비었다"
    if float(ev.get("at_utc_ts", 0)) > float(permit.get("issued_utc_ts", 0)):
        return "intent 가 허가보다 늦게 적혔다"
    if bool(ev.get("replace")) != bool(permit.get("replace")):
        return "허가와 intent 의 replace 가 다르다"
    stem = permit.get("stem")
    if not stem or ev.get("stem") != stem:
        return f"허가의 후보({stem})와 intent 의 후보({ev.get('stem')})가 다르다"
    try:
        info, why = candidate(stem)
    except SystemExit as e:
        info, why = {}, f"{stem}: {e}"
    if why:
        return why
    if info["pred_sha"] != permit.get("sha_pred") or info["pred_sha"] != ev.get("sha_pred"):
        return f"{stem}: 허가나 intent 의 sha_pred 가 큐 예측과 다르다"
    q = QUEUE / f"{stem}.parquet"
    hard = pred_problems(q, tpl) + ([w] if (w := sane(q)) else [])
    m = json.loads((ROOT / info["manifest"]).read_text(encoding="utf-8"))
    hard += empty_problems(empty_on_final(m["feats"], m["feats_sha"]))
    if hard:
        return "; ".join(hard)[:400]
    want = submission_values(pd.read_parquet(q).set_index("MVT_ID_mvt")["pred"], tpl)
    if not np.array_equal(pd.read_parquet(sub)["TAXITIME_SEC_mvt"].to_numpy(), want):
        return "올릴 결선 파일의 값이 그 후보의 큐 예측에서 나온 값과 다르다"
    rows, probs = ledger_files()
    if probs:
        return "장부와 제출 파일이 어긋난다: " + "; ".join(probs)[:300]
    objs = ls(TEAM_BUCKET)
    bs = bucket_state(objs, rows)
    if bs["off_ledger"]:
        return f"장부에 없는 제출이 버킷에 있다: {', '.join(bs['off_ledger'])}"
    if bs["left"] <= 0:
        return (f"오늘(UTC) 쿼터를 다 썼다(결과 {bs['results_today_utc']}, 올림 {bs['uploads_today_utc']}). "
                "결선 파일이 하루 5발에 드는지 원문에 없어 하나를 남겨 둔다")
    P = final_bucket_problems(objs, bool(permit.get("replace")))
    return "; ".join(P) if P else None
