# SPDX-License-Identifier: GPL-3.0-only
"""측정 묶음 하나에만 쓰는 준비 점검 예외(플레이북 §0 G2: 걸린 칸은 원칙적으로 데이터나 피처에서 고친다. 감사가 승인하고 도구가 강제하는
특정 묶음의 한정 예외만 허용한다. 로컬 선택과 G2 닫힘의 승인은 별도다).

예외 기록 notes/exceptions/<번호>.json 은 한 번 쓰고 고치지 않는다. 묶는 것: 후보(이름과 manifest, 예측, 홀드아웃 해시), 피처 판과 해시, 원입력 해시,
칸 이름, 자와 시험의 결측 행 수 상한과 그 행 ID, 원인(공항과 편명 공백), 영향 문턱(재가중 RMSE 변화, 모든 쌍 Δ 변화, 부호 유지), 승인 근거(판정
전문, 검토 요청, 차이의 정의, 영향 검증의 해시). 면제되는 것은 그 피처 판의 그 칸들의 채움 비율 판정뿐이고, 원래의 준비 점검 실패는 등록 때
notes/readiness_records/ 에 그대로 남는다. 등록(src/register_batch.py --exception)과 매 발 직전(src/batches.py verify_before_shot)에 같은 check() 가
돌고, 한 예외는 한 묶음에만 쓰인다.
    python -m src.exception write <조건 json>   → notes/exceptions/<번호>.json, _ruling.txt, _brief.txt, _deviation.json, _evidence.json(모두 한 번만 쓴다)
조건 json: number, feats, columns, caps{ruler, test: {nan_max, rows}}, cause{airport, blank_column}, thresholds{rmse_change_max, pair_delta_change_max},
    candidates(이름 목록, 적힌 순서), ruling, brief, deviation(파일 경로), approval(누가 언제 무엇을 판정했나)
"""
from __future__ import annotations

import glob
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.batches import MIN_BATCH, candidate, entries, hold, now, sha  # noqa: E402
from src.ruler import ruler_hash  # noqa: E402

EXC = ROOT / "notes" / "exceptions"
RAW = ["data/training_*.parquet", "data/ranking.parquet", "data/submitting.parquet", "data/metar/*.csv"]  # 피처 판과 준비 점검이 읽는 원입력
CAND_KEYS = ("stem", "manifest", "manifest_sha", "pred_sha", "holdout", "holdout_sha", "feats", "local_rmse")
SCOPE = ("이 예외는 이 기록을 처음 쓰는 묶음 하나에만 쓰인다. 면제는 피처 판 {feats} 의 칸 {cols} 채움 비율 판정뿐이고, 다른 준비 점검, 다른 칸, 다른 피처 "
         "판, 다른 후보, 다른 묶음에는 쓰이지 않는다. 로컬 선택과 G2 닫힘의 승인은 별도다.")


def rel(p) -> str:
    return Path(p).resolve().relative_to(ROOT).as_posix()


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(sorted(int(i) for i in ids)).encode()).hexdigest()[:16]


def raw_inputs() -> dict:
    return {rel(f): sha(rel(f)) for pat in RAW for f in sorted(glob.glob(str(ROOT / pat)))}


def test_ids() -> set:
    sub = pd.read_parquet(ROOT / "data" / "submitting.parquet")
    return set(sub[[c for c in sub.columns if c.startswith("MVT_ID")][0]])


def nan_rows(feats: str, cols: list[str]) -> dict:
    """피처 판에서 칸마다 값이 빈 자의 행과 시험의 행(MVT_ID). 자와 시험의 정의는 src/composition.py feat_fill 과 같다."""
    F = pd.read_parquet(ROOT / "cache" / f"{feats}.parquet", columns=["MVT_ID_mvt", *cols])
    ru, te = F["MVT_ID_mvt"].isin(hold().index), F["MVT_ID_mvt"].isin(test_ids())
    out = {"ruler_rows": int(ru.sum()), "test_rows": int(te.sum())}
    for c in cols:
        na = F[c].isna()
        out[c] = {"ruler": sorted(int(i) for i in F.loc[na & ru, "MVT_ID_mvt"]), "test": sorted(int(i) for i in F.loc[na & te, "MVT_ID_mvt"])}
    return out


def cause(ids: list[int], airport: str, col: str) -> str | None:
    """그 행이 원자료에서 모두 그 공항이고 그 칸(편명)이 비었나."""
    s, got = set(ids), []
    for f in sorted(glob.glob(str(ROOT / "data" / "training_*.parquet"))) + [str(ROOT / "data" / "ranking.parquet")]:
        d = pd.read_parquet(f, columns=["MVT_ID_mvt", "ADEP_mvt", col])
        got.append(d[d["MVT_ID_mvt"].isin(s)])
    d = pd.concat(got)
    if len(d) != len(s) or d["MVT_ID_mvt"].nunique() != len(s):
        return f"원자료에서 찾은 행이 {len(d)}개다({len(s)}개여야 한다)"
    if not (d["ADEP_mvt"] == airport).all():
        return f"{airport} 가 아닌 행 {int((d['ADEP_mvt'] != airport).sum())}개"
    blank = d[col].isna() | (d[col].astype(str).str.strip() == "")
    if not blank.all():
        return f"{col} 이 비지 않은 행 {int((~blank).sum())}개"
    return None


def sensitivity(cands: list[dict], ruler_nan: list[int], test_nan: int, test_rows: int) -> dict:
    """빈 행을 시험의 비중으로 맞추는 가중(빈 행 p_te/p_ru, 나머지 (1-p_te)/(1-p_ru))으로 후보 RMSE 와 모든 쌍의 Δ 를 다시 잰다."""
    y = hold()["TAXITIME_SEC_mvt"].astype(float)
    m = y.index.isin(ruler_nan)
    p_ru, p_te = float(m.mean()), test_nan / test_rows
    w = np.where(m, p_te / p_ru, (1 - p_te) / (1 - p_ru))
    raw, res = {}, {}
    for c in cands:
        p = pd.read_parquet(ROOT / c["holdout"]).set_index("MVT_ID_mvt")["pred"].reindex(y.index)
        if p.isna().any():
            raise SystemExit(f"{c['holdout']}: 자의 행과 안 맞는다(빠진 행 {int(p.isna().sum())})")
        e2 = ((p - y) ** 2).to_numpy()
        r, rw = float(np.sqrt(e2.mean())), float(np.sqrt((w * e2).sum() / w.sum()))
        raw[c["stem"]] = (r, rw)
        res[c["stem"]] = {"rmse": round(r, 4), "rmse_reweighted": round(rw, 4), "change": round(rw - r, 4), "change_exact": rw - r, "nan_rows": int(m.sum()),
                          "rmse_on_nan_rows": round(float(np.sqrt(e2[m].mean())), 3), "max_abs_err_on_nan_rows": round(float(np.sqrt(e2[m].max())), 3)}
    pairs = []
    for a, b in itertools.combinations([c["stem"] for c in cands], 2):
        d, dw = raw[b][0] - raw[a][0], raw[b][1] - raw[a][1]
        pairs.append({"pair": f"{b}-{a}", "delta": round(d, 4), "delta_reweighted": round(dw, 4), "change": round(abs(dw - d), 4), "change_exact": abs(dw - d),
                      "same_sign": bool((d > 0) == (dw > 0))})
    return {"weight_nan_rows": round(p_te / p_ru, 4), "weight_other_rows": round((1 - p_te) / (1 - p_ru), 6), "candidates": res, "pairs": pairs}


def over(sens: dict, th: dict) -> list[str]:
    """문턱은 반올림 전 값(change_exact)에 건다(재검토 5판). 표시만 반올림한다."""
    bad = [f"{s}: 재가중 RMSE 변화 {v['change_exact']:.6f} 가 ±{th['rmse_change_max']} 밖" for s, v in sens["candidates"].items()
           if abs(v["change_exact"]) > th["rmse_change_max"]]
    bad += [f"{p['pair']}: Δ 변화 {p['change_exact']:.6f} > {th['pair_delta_change_max']}" for p in sens["pairs"] if p["change_exact"] > th["pair_delta_change_max"]]
    return bad + [f"{p['pair']}: 재가중으로 Δ 부호가 바뀐다" for p in sens["pairs"] if not p["same_sign"]]


def rows_problems(E: dict, rows: dict) -> list[str]:
    P = []
    if rows["ruler_rows"] != E["caps"]["ruler"]["rows"] or rows["test_rows"] != E["caps"]["test"]["rows"]:
        P.append(f"자 {rows['ruler_rows']}행, 시험 {rows['test_rows']}행이 예외의 {E['caps']['ruler']['rows']}, {E['caps']['test']['rows']}행과 다르다")
    for col in E["columns"]:
        for side in ("ruler", "test"):
            ids, cap = rows[col][side], E["caps"][side]["nan_max"]
            if len(ids) > cap:
                P.append(f"{col} 의 {side} 결측 {len(ids)}행이 상한 {cap} 을 넘는다")
            if "rows" in E and ids_sha(ids) != E["rows"][col][side]["sha"]:
                P.append(f"{col} 의 {side} 결측 행이 예외에 적힌 행과 다르다")
    return P


def check(E: dict, E_rel: str, stems: list[str], comp: dict, mode: str, batch_rel: str | None = None, E_sha: str | None = None) -> tuple[list[str], dict]:
    """예외의 범위를 다시 잰다. mode register: 묶음 후보가 예외의 후보와 같고 아직 어느 묶음도 이 예외를 쓰지 않는다. mode shot: 그 후보가 예외 안에 있고
    이 예외를 쓰는 묶음이 batch_rel 하나다. 돌려주는 값: (문제 목록, 다시 잰 값)."""
    P = []
    if E_sha is not None and (not (ROOT / E_rel).exists() or sha(E_rel) != E_sha):
        P.append("예외 기록 파일이 없거나 묶음에 적힌 해시와 다르다")
    for k, f in E["files"].items():
        if not (ROOT / f["path"]).exists() or sha(f["path"]) != f["sha"]:
            P.append(f"승인 근거 {k} 파일 {f['path']} 이 없거나 해시가 다르다")
    names = [c["stem"] for c in E["candidates"]]
    if mode == "register" and stems != names:
        P.append(f"묶음 후보와 순서가 예외의 후보 {names} 와 다르다")
    if mode == "shot" and not set(stems) <= set(names):
        P.append(f"{sorted(set(stems) - set(names))} 는 예외의 후보가 아니다")
    for s in stems:
        info, why = candidate(s)
        reg = next((c for c in E["candidates"] if c["stem"] == s), None)
        if why or reg is None:
            P.append(why or f"{s} 는 예외의 후보가 아니다")
            continue
        P += [f"{s}: {k} 가 예외에 적힌 값과 다르다" for k in ("manifest_sha", "pred_sha", "holdout_sha", "feats") if info[k] != reg[k]]
    P += [f"{c['stem']}: 홀드아웃 {c['holdout']} 의 해시가 예외와 다르다" for c in E["candidates"] if not (ROOT / c["holdout"]).exists() or sha(c["holdout"]) != c["holdout_sha"]]
    if sha(f"cache/{E['feats']}.parquet") != E["feats_sha"]:
        P.append(f"피처 판 {E['feats']} 의 해시가 예외와 다르다")
    if raw_inputs() != E["raw_inputs"]:
        P.append("원입력 파일 목록이나 해시가 예외와 다르다")
    for c in comp.get("checks", []):
        finite = all(isinstance(v, dict) and isinstance(v.get("diff_pp"), (int, float)) and math.isfinite(v["diff_pp"]) for v in (c.get("ref") or {}).values())
        if not c["pass"] and not (c.get("kind") == "feat_fill" and c.get("feats") == E["feats"] and set(c.get("ref") or {}) <= set(E["columns"]) and finite):
            P.append(f"예외 밖의 준비 점검 실패: {c['check'][:90]}")
    rows = nan_rows(E["feats"], E["columns"])
    P += rows_problems(E, rows)
    ids = sorted({i for col in E["columns"] for side in ("ruler", "test") for i in rows[col][side]})
    why = cause(ids, E["cause"]["airport"], E["cause"]["blank_column"])
    if why:
        P.append(f"결측 원인이 예외와 다르다: {why}")
    ruler_nan = sorted({i for col in E["columns"] for i in rows[col]["ruler"]})
    test_nan = len({i for col in E["columns"] for i in rows[col]["test"]})
    sens = sensitivity(E["candidates"], ruler_nan, test_nan, rows["test_rows"])
    P += over(sens, E["thresholds"])
    users = [p.relative_to(ROOT).as_posix() for p, e in entries() if (e.get("exception") or {}).get("file") == E_rel]
    if mode == "register" and users:
        P.append(f"이 예외는 이미 묶음 {users} 에 쓰였다")
    if mode == "shot" and users != [batch_rel]:
        P.append(f"이 예외를 쓰는 묶음이 {users} 다(이 묶음 {batch_rel} 하나여야 한다)")
    counts = {col: {side: len(rows[col][side]) for side in ("ruler", "test")} for col in E["columns"]}
    return P, {"nan_counts": counts, "sensitivity": sens}


def binding(E_rel: str) -> dict:
    """묶음 파일에 적는 예외(묶음 쪽 고정본). 쏠 때 이 값과 예외 파일을 다시 맞춘다."""
    E = json.loads((ROOT / E_rel).read_text(encoding="utf-8"))
    return {"file": E_rel, "sha": sha(E_rel), "number": E["number"], "feats": E["feats"], "feats_sha": E["feats_sha"], "columns": E["columns"],
            "caps": E["caps"], "rows_sha": {c: {s: E["rows"][c][s]["sha"] for s in ("ruler", "test")} for c in E["columns"]},
            "cause": E["cause"], "thresholds": E["thresholds"], "candidates": [c["stem"] for c in E["candidates"]],
            "ruling_sha": E["files"]["ruling"]["sha"], "scope": E["scope"]}


def load(E_rel: str) -> dict:
    return json.loads((ROOT / E_rel).read_text(encoding="utf-8"))


def write(terms_path: str) -> int:
    T = json.loads(Path(terms_path).read_text(encoding="utf-8"))
    n = int(T["number"])
    EXC.mkdir(parents=True, exist_ok=True)
    out = EXC / f"{n:03d}.json"
    if out.exists():
        print("거부: 같은 번호의 예외 기록이 있다", rel(out))
        return 1
    if len(T["candidates"]) < MIN_BATCH or len(set(T["candidates"])) != len(T["candidates"]):
        print(f"거부: 후보는 서로 다른 {MIN_BATCH}개 이상이다")
        return 1
    cands = []
    for s in T["candidates"]:
        info, why = candidate(s)
        if why:
            print("거부:", why)
            return 1
        if info["feats"] != T["feats"]:
            print(f"거부: {s} 의 피처 판 {info['feats']} 이 {T['feats']} 가 아니다")
            return 1
        cands.append({k: info[k] for k in CAND_KEYS})
    E = {k: T[k] for k in ("number", "feats", "columns", "caps", "cause", "thresholds", "approval")}
    E["feats_sha"] = sha(f"cache/{T['feats']}.parquet")
    rows = nan_rows(T["feats"], T["columns"])
    P = rows_problems(E, rows)
    ids = sorted({i for col in T["columns"] for side in ("ruler", "test") for i in rows[col][side]})
    why = cause(ids, T["cause"]["airport"], T["cause"]["blank_column"])
    P += [f"결측 원인: {why}"] if why else []
    ruler_nan = sorted({i for col in T["columns"] for i in rows[col]["ruler"]})
    sens = sensitivity(cands, ruler_nan, len({i for col in T["columns"] for i in rows[col]["test"]}), rows["test_rows"])
    P += over(sens, T["thresholds"])
    if P:
        print("거부:", "; ".join(P))
        return 1
    raw = raw_inputs()
    evidence = {"written_kst": now(), "what": "예외의 영향 검증과 행 확인(src/exception.py write). 후보 홀드아웃에서 결측 행을 시험 비중으로 가중한 RMSE 와 모든 쌍의 Δ",
                "feats": T["feats"], "feats_sha": E["feats_sha"], "raw_inputs": raw, "ruler_hash": ruler_hash(hold().index),
                "rows": {"ruler_rows": rows["ruler_rows"], "test_rows": rows["test_rows"],
                         **{c: {s: {"n": len(rows[c][s]), "sha": ids_sha(rows[c][s])} for s in ("ruler", "test")} for c in T["columns"]}},
                "cause": f"모든 결측 행이 원자료에서 ADEP {T['cause']['airport']} 이고 {T['cause']['blank_column']} 이 비었다", "candidates": cands,
                "sensitivity": sens, "thresholds": T["thresholds"], "code": {"src/exception.py": sha("src/exception.py")}}
    files = {}
    for k, src, suffix in (("ruling", T["ruling"], "ruling.txt"), ("brief", T["brief"], "brief.txt"), ("deviation", T["deviation"], "deviation.json")):
        dst = EXC / f"{n:03d}_{suffix}"
        with open(dst, "x", encoding="utf-8", newline="") as f:
            f.write(Path(src).read_text(encoding="utf-8"))
        files[k] = {"path": rel(dst), "sha": sha(rel(dst)), "source": str(src)}
    ev = EXC / f"{n:03d}_evidence.json"
    with open(ev, "x", encoding="utf-8") as f:
        f.write(json.dumps(evidence, ensure_ascii=False, indent=1))
    files["evidence"] = {"path": rel(ev), "sha": sha(rel(ev))}
    E.update({"written_kst": now(), "scope": SCOPE.format(feats=T["feats"], cols=", ".join(T["columns"])), "raw_inputs": raw,
              "rows": {c: {s: {"n": len(rows[c][s]), "sha": ids_sha(rows[c][s]), "ids": rows[c][s]} for s in ("ruler", "test")} for c in T["columns"]},
              "candidates": cands, "files": files})
    with open(out, "x", encoding="utf-8") as f:
        f.write(json.dumps(E, ensure_ascii=False, indent=1))
    print("예외 기록", rel(out), "sha", sha(rel(out)))
    for s, v in sens["candidates"].items():
        print(f"  {s}: RMSE {v['rmse']} → 재가중 {v['rmse_reweighted']} (변화 {v['change']})")
    print("  쌍 Δ 변화 최대", max(p["change"] for p in sens["pairs"]), "부호 유지", all(p["same_sign"] for p in sens["pairs"]))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 3 or sys.argv[1] != "write":
        print(__doc__)
        sys.exit(2)
    sys.exit(write(sys.argv[2]))
