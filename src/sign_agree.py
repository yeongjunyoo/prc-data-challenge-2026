# SPDX-License-Identifier: GPL-3.0-only
# G2 닫힘 증거(플레이북 §0 G2). 장부에서 서버 값이 있는 CANDIDATE 발을 모아 차례 Δ 부호 일치를 세고, 닫힘에 필요한 증거를 산출물에서 다시 계산한다.
# 진단 발(DIAG)은 순위 쪽 감시 조항 때문에 세지 않는다(MISSION §5). ANCHOR 는 자의 수준 확인(서버 대 로컬)으로만 따로 낸다.
# 판정 수에 드는 후보: 장부 note 의 묶음 파일이 있고 sha_batch 가 맞으며, manifest 를 지금 다시 검사해 통과하고(src/batches.py manifest_of: 해시,
#   설정과 명령, forward 는 --setting fwd, 코드 사본) sha_manifest 가 맞고, 조기 종료 행이 자와 겹치지 않는 것. 나머지 발은 이유와 함께 뺀다.
# 닫힘 증거: ⓪ 장부 변조 흔적 없음 ① 판정 5건 이상 ② 차례 Δ 부호 일치 80% 이상(실행 사슬을 나눠 쓰는 차례는 빼고 센다, 닫힘 감사 2회차) ③ 판정 후보끼리의 전방 홀드아웃(1~6월 학습, 7월 측정) Δ 부호가
#   자의 Δ 부호와 80% 이상 같다 ④ 판정 후보와 등록 후보가 쓴 피처 판마다 가림 재구축(notes/mask_test_<판>.json)의 세 목록이 비고 feats_sha 가
#   지금 cache 파일과 같고 masked_sha 가 가린 판 파일과 같다 ⑤ 자의 행 해시가 GATE.json ruler_hash 와 같다.
# 증거가 다 차도 로컬 선택은 닫힘 감사가 GATE.json local_selection_audited 를 true 로 바꾼 뒤에만 열린다(local_selection_allowed).
#     python -m src.sign_agree   → notes/sign_agreement.json
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.batches import RMSE_TOL, entries, es_overlap, ledger_rows, ledger_state, manifest_of, note_tokens, sha  # noqa: E402
from src.ruler import ruler_frame, ruler_hash, rmse_of  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
gate = json.loads((ROOT / "GATE.json").read_text(encoding="utf-8"))
rows = ledger_rows()
kind = lambda r: r["method"].split(" ", 1)[0]  # noqa: E731  장부 method 는 「<kind> <방법>」 꼴이다
judged = [r for r in rows if r["local_rmse"] and r["server_rmse"]]
anchor = [{"version": int(r["version"]), "local": float(r["local_rmse"]), "server": float(r["server_rmse"]),
           "server_over_local": round(float(r["server_rmse"]) / float(r["local_rmse"]), 4)} for r in judged if kind(r) == "ANCHOR"]
R = ruler_frame()
hold, july = R[R["month"].isin([1, 7])], R[R["month"] == 7]
st = ledger_state()

clean, excluded = [], [f"v{r['version']} {kind(r)}" for r in judged if kind(r) not in ("ANCHOR", "CANDIDATE")]
for r in judged:
    if kind(r) != "CANDIDATE":
        continue
    t = note_tokens(r.get("note", ""))
    why = None
    if not t.get("batch") or not (ROOT / t["batch"]).exists() or sha(t["batch"]) != t.get("sha_batch"):
        why = "등록된 묶음의 발이 아니거나 묶음 파일이 바뀌었다"
    else:
        reg = next((c for c in json.loads((ROOT / t["batch"]).read_text(encoding="utf-8"))["candidates"] if c["stem"] == t.get("stem")), None)
        if reg is None or reg["pred_sha"] != t.get("sha_pred") or reg["manifest_sha"] != t.get("sha_manifest"):
            why = "장부의 이름, 예측 해시, manifest 해시가 등록된 묶음의 후보와 맞지 않는다"
    if why is None:
        m, why = manifest_of({"manifest": t.get("manifest")})
        if why is None and sha(t["manifest"]) != t.get("sha_manifest"):
            why = "쏠 때와 manifest 해시가 다르다"
        if why is None and es_overlap(m):
            why = "조기 종료 행이 자와 겹친다"
        if why is None:
            rm = rmse_of(m["holdout"]["path"], hold)
            if abs(rm - float(reg["local_rmse"])) > RMSE_TOL:
                why = f"홀드아웃을 다시 잰 값 {rm:.3f} 이 등록값 {reg['local_rmse']} 과 다르다"
    if why:
        excluded.append(f"v{r['version']} CANDIDATE: {why}")
        continue
    clean.append({"v": f"v{r['version']}", "stem": t["stem"], "local": rm, "server": float(r["server_rmse"]), "m": m})  # 로컬 값은 다시 잰 값(등록값과 RMSE_TOL 안)


def agree(a, b, la, lb, oa, ob) -> dict:
    dl, do = lb - la, ob - oa
    return {"pair": f"{b}-{a}", "local_delta": round(dl, 3), "other_delta": round(do, 3), "same_sign": (dl > 0) == (do > 0)}


def run_parents() -> dict[str, set[str]]:
    """잡 파일(notes/jobs_*.txt)의 모든 명령에서 실행 태그(--tag)마다 그 실행이 읽은 실행 태그(--base, --cat)를 모은다."""
    par: dict[str, set[str]] = {}
    for jf in sorted((ROOT / "notes").glob("jobs_*.txt")):
        for ln in jf.read_text(encoding="utf-8").splitlines():
            if not ln.strip() or ln.startswith("#") or "	" not in ln:
                continue
            t = ln.split("	", 1)[1].split()
            if "--tag" in t:
                par.setdefault(t[t.index("--tag") + 1], set()).update(t[i + 1] for i, x in enumerate(t[:-1]) if x in ("--base", "--cat"))
    return par


PAR = run_parents()


def chain_runs(m: dict) -> set[str]:
    """후보의 실행 사슬: manifest 의 run_tag 에서 --base, --cat 를 잡 파일로 거슬러 올라간 모든 실행 태그(그 후보를 만든 학습 실행 전부)."""
    seen, todo = set(), [m.get("run_tag") or m["spec"]]
    while todo:
        x = todo.pop()
        if x not in seen:
            seen.add(x)
            todo.extend(PAR.get(x, ()))
    return seen


rate = lambda xs: (sum(x["same_sign"] for x in xs), len(xs))  # noqa: E731
chains = {c["v"]: chain_runs(c["m"]) for c in clean}
steps = [agree(a["v"], b["v"], a["local"], b["local"], a["server"], b["server"]) | {"shared_runs": sorted(chains[a["v"]] & chains[b["v"]])}
         for a, b in zip(clean, clean[1:])]
pairs = [agree(a["v"], b["v"], a["local"], b["local"], a["server"], b["server"]) for a, b in itertools.combinations(clean, 2)]
ok_raw, n_raw = rate(steps)
ok_s, n_s = rate([x for x in steps if not x["shared_runs"]])  # 같은 학습 실행을 나눠 쓰는 차례는 독립 증거가 아니다
fsc = {c["v"]: (rmse_of(c["m"]["holdout"]["path"], hold), rmse_of(c["m"]["forward"]["path"], july)) for c in clean}
fpairs = [agree(a["v"], b["v"], fsc[a["v"]][0], fsc[b["v"]][0], fsc[a["v"]][1], fsc[b["v"]][1]) for a, b in itertools.combinations(clean, 2)]
ok_f, n_f = rate(fpairs)

feats_used = {c["m"]["feats"] for c in clean} | {c["feats"] for _, e in entries() for c in e["candidates"]}
mask = {}
for fv in sorted(feats_used):
    f = ROOT / "notes" / f"mask_test_{fv}.json"
    if not f.exists():
        mask[fv] = "파일 없음"
        continue
    j = json.loads(f.read_text(encoding="utf-8"))
    bad = {k: j.get(k) for k in ("differing_columns", "missing_in_masked_build", "differing_te_columns_val_setting") if j.get(k) != []}
    if j.get("feats_sha") != sha(f"cache/{fv}.parquet"):
        bad["feats_sha"] = f"{j.get('feats_sha')} 대 지금 {sha(f'cache/{fv}.parquet')}"
    mpath = (j.get("masked_build") or "").split(" ", 1)[0]
    if not mpath or not (ROOT / mpath).exists() or sha(mpath) != j.get("masked_sha"):
        bad["masked_sha"] = f"가린 판 {mpath or '(경로 없음)'} 이 없거나 해시가 {j.get('masked_sha')} 와 다르다"
    mask[fv] = bad or "통과"

rh = ruler_hash(hold.index)
blockers = []
if st["problems"]:
    blockers.append("⓪ 장부 변조 흔적: " + "; ".join(st["problems"]))
if len(clean) < 5:
    blockers.append(f"① 판정 수에 드는 후보 {len(clean)}/5")
if not n_s or ok_s < 0.8 * n_s:
    blockers.append(f"② 차례 Δ 부호 일치 {ok_s}/{n_s}(실행 사슬을 나눠 쓰는 차례를 뺀 수, 뺀 것 포함 {ok_raw}/{n_raw}, 80% 필요)")
if not n_f or ok_f < 0.8 * n_f:
    blockers.append(f"③ 전방 홀드아웃 쌍 부호 일치 {ok_f}/{n_f}(판정 후보끼리, 80% 필요)")
if not mask:
    blockers.append("④ 가림 재구축: 판정 후보도 등록 후보도 없어 검사한 피처 판이 없다")
elif any(v != "통과" for v in mask.values()):
    blockers.append("④ 가림 재구축: " + ", ".join(f"{k} {v}" for k, v in mask.items() if v != "통과"))
if gate.get("ruler_hash") != rh:
    blockers.append(f"⑤ 자의 행 해시 {rh} 가 GATE.json ruler_hash({gate.get('ruler_hash')})와 다르다")

by_stem = {c["stem"]: c for c in clean}
prefixed, reg_chain = [], {}
for bp, e in entries():
    for c in e["compares"]:
        row = {"batch": bp.name, "pair": f"{c['b']}-{c['a']}", "local_delta": c["delta_b_minus_a"], "sd": c["sd_delta"], "verdict": c["verdict"]}
        a, b = by_stem.get(c["a"]), by_stem.get(c["b"])
        if a and b:
            row["server_delta"] = round(b["server"] - a["server"], 3)
            row["same_sign"] = (c["delta_b_minus_a"] > 0) == (b["server"] - a["server"] > 0)
        prefixed.append(row)
    for c in e["candidates"]:
        reg_chain[c["stem"]] = chain_runs(json.loads((ROOT / c["manifest"]).read_text(encoding="utf-8")))
groups: list[set[str]] = []  # 실행 사슬이 겹치는 등록 후보를 이어 붙인 묶음(연결 성분)
for stem, ch in reg_chain.items():
    hit = [g for g in groups if any(reg_chain[o] & ch for o in g)]
    merged = {stem}.union(*hit) if hit else {stem}
    groups = [g for g in groups if g not in hit] + [merged]
lineage_groups = [sorted(g) for g in groups if len(g) > 1]
out = {"anchor_level_check": anchor, "excluded": excluded, "judged_candidates": len(clean),
       "judged": [{k: v for k, v in c.items() if k != "m"} | {"spec": c["m"]["spec"]} for c in clean],
       "steps_agree": (ok_s, n_s), "steps_agree_including_shared_runs": (ok_raw, n_raw), "steps": steps, "pairs_agree": rate(pairs), "pairs": pairs,
       "forward": {"agree": (ok_f, n_f), "scores": fsc, "pairs": fpairs}, "mask_tests": mask, "ruler_hash": rh,
       "batch_exceptions": [{"batch": path.relative_to(ROOT).as_posix(), "exception": x["file"], "sha": x["sha"], "feats": x["feats"], "columns": x["columns"],
                             "note": "준비 점검의 한정 예외를 안고 쏜 묶음이다(src/exception.py). 닫힘 감사가 따로 본다"} for path, e in entries() if (x := e.get("exception"))],
       "batch_prefixed_pairs": prefixed, "lineage_groups": lineage_groups,
       "local_selection_blockers": blockers, "local_selection_audited": bool(gate.get("local_selection_audited")),
       "local_selection_allowed": not blockers and bool(gate.get("local_selection_audited")),
       "rule": "닫힘 증거 ⓪~⑤ 가 다 차고 닫힘 감사가 GATE.json local_selection_audited 를 true 로 바꿔야 로컬 선택이 열린다. 그 전에는 등록한 측정 묶음만 쏜다",
       "limits": "도구가 증명하지 못하는 것: 실행 때 쓴 코드가 사본과 같다는 것(사본은 실행 뒤 뜬다), 전방 학습 기간(명령의 --setting fwd 만 본다), 가림 비교의 내용(저장된 "
                 "목록과 해시만 본다). 닫힘 감사가 이 셋을 따로 본다. 닫힘 감사는 사전 고정한 전체 쌍(batch_prefixed_pairs)의 부호와 분해능 판정, "
                 "같은 학습 실행에서 나온 후보(lineage_groups, 독립 증거가 아니다)를 함께 본다(재검토 5판)"}
(ROOT / "notes" / "sign_agreement.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
