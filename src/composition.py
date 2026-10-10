# SPDX-License-Identifier: GPL-3.0-only
# G2 준비 점검(private notes §0 G2 측정 단계 ②, 리더보드를 쓰지 않는다). 등록할 때와 쏘기 직전에 다시 돈다(src/batches.py readiness).
# 로컬 자(2025-01, 07 출발편)와 시험 입력(data/ranking.parquet 중 submitting 의 행), 후보의 피처와 예측을 나란히 놓고 폭 안인지 판정한다.
# 폭의 값은 학습 기간에서만 나온다: 2025 의 다른 달 쌍(2+8, 3+9, 4+10, 5+11, 6+12)이 자(1+7)와 보이는 가장 큰 차이. 이 설계 자체는 시험 입력 통계를 본 뒤
# 정했다(아래 줄).
# 피처 칸별 채움 비율의 폭은 그 달 쌍 차이와 표본 잡음 바닥(자의 비율로 낸 두 표본 비율 차의 이항 표준오차 4배) 중 큰 쪽이다. 값이 NaN 이면 실패다.
# 이 설계(달 쌍 폭)는 자문이 10-02 18:11~18:12 에 잰 시험 입력 통계(공항 비중, NM 매칭, 꼬리 비중)를 본 뒤 18:39 에 정했다. 표본 잡음 바닥과 NaN 실패는
# 다른 모델 검토(10-02 19:07:46~19:10:07)의 지적으로 19:18 에 넣었다. 그 뒤로 설계를 바꾸면 이 머리와 private notes G2 에 적는다.
# 같은 기간 출발편 값을 간접으로 쓰는 누수는 가림 재구축(src/mask_test.py, sign_agree 의 닫힘 증거)이 본다.
#     python -m src.composition [--queue-dir queue] [--out notes/composition.json] [--only <큐 이름> ...] [--feats <피처 판> ...]   (all_pass 가 false 면 종료 코드 1)
#     --feats 는 후보 없이 피처 판의 채움 비율만 본다(학습 전에 준비 점검을 통과할 판인지 본다)
import glob
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
RULER, REFS = (1, 7), [(2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
Z = 4.0  # 채움 비율 표본 잡음 바닥: 이항 표준오차의 배수
FILL_CHUNK = 8  # 채움 비율을 한 번에 읽는 피처 칸 수
QDIR, OUT = ROOT / "queue", ROOT / "notes" / "composition.json"


def train_dep() -> pd.DataFrame:
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "FLIGHT_ID_mvt", "AOBT_3_flt", "TAXITIME_SEC_mvt"]
    tr = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(ROOT / "data" / "training_*.parquet")))], ignore_index=True)
    tr = tr[tr["PHASE_mvt"] == "DEP"]
    return tr.assign(month=tr["MVT_TIME_UTC_mvt"].dt.month)


def test_dep() -> pd.DataFrame:
    sub = pd.read_parquet(ROOT / "data" / "submitting.parquet")
    idc = [c for c in sub.columns if c.startswith("MVT_ID")][0]
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "FLIGHT_ID_mvt", "AOBT_3_flt"]
    rk = pd.read_parquet(ROOT / "data" / "ranking.parquet", columns=cols)
    return rk[rk["MVT_ID_mvt"].isin(set(sub[idc]))]


def tvd(a: pd.Series, b: pd.Series) -> float:
    return float(0.5 * a.value_counts(normalize=True).subtract(b.value_counts(normalize=True), fill_value=0).abs().sum())


def sym(ref: dict) -> list:
    d = max(abs(v - 1) for v in ref.values())
    return [round(1 - d, 4), round(1 + d, 4)]


def cand_info(stem: str) -> dict | None:
    """큐 후보의 홀드아웃, 피처 판: manifest 가 있으면 거기서, 없으면 json 의 holdout, feats 에서."""
    j = QDIR / f"{stem}.json"
    if not j.exists():
        return None
    meta = json.loads(j.read_text(encoding="utf-8"))
    if meta.get("kind") != "CANDIDATE":
        return None
    if meta.get("manifest") and (ROOT / meta["manifest"]).exists():
        m = json.loads((ROOT / meta["manifest"]).read_text(encoding="utf-8"))
        return {"holdout": m["holdout"]["path"], "feats": m["feats"]}
    return {"holdout": meta.get("holdout"), "feats": meta.get("feats")}


def feat_fill(fv: str, test_ids: set, month_of: pd.Series) -> dict:
    """피처 판 하나의 칸별 채움 비율을 자의 행과 시험 행에서 비교한다. 폭은 2025 달 쌍 차이와 자의 비율로 낸 이항 표준오차 Z 배 중 큰 쪽, NaN 은 실패.
    판을 통째로 읽지 않고 칸 FILL_CHUNK 개씩 읽는다(칸마다 값은 같고, 준비 점검의 최대 메모리가 판 두 벌에서 칸 몇 개로 준다)."""
    path = ROOT / "cache" / f"{fv}.parquet"
    ids = pd.read_parquet(path, columns=["MVT_ID_mvt"])["MVT_ID_mvt"]
    m = ids.map(month_of)
    m_ho, m_te = m.isin(RULER).to_numpy(), ids.isin(test_ids).to_numpy()
    masks = [m_ho, m_te] + [m.isin(q).to_numpy() for q in REFS]
    del ids, m
    schema = pq.ParquetFile(path).schema_arrow
    skip = {"y", "src", "MVT_ID_mvt"} | {c for c in (schema.pandas_metadata or {}).get("index_columns", []) if isinstance(c, str)}
    cols = [c for c in schema.names if c not in skip]
    parts = [[] for _ in masks]
    for i in range(0, len(cols), FILL_CHUNK):
        X = pd.read_parquet(path, columns=cols[i:i + FILL_CHUNK])
        for part, mask in zip(parts, masks):
            part.append(X.loc[mask].notna().mean())
        del X
    n_ho, n_te, *n_refs = [pd.concat(p) for p in parts]
    N_ho, N_te = int(m_ho.sum()), int(m_te.sum())
    floor = 100 * Z * (n_ho * (1 - n_ho) * (1 / N_ho + 1 / N_te)) ** 0.5  # 자의 비율만 쓴다(시험 비율로 폭을 정하지 않는다)
    band = pd.concat([100 * (n_ref - n_ho).abs() for n_ref in n_refs] + [floor], axis=1).max(axis=1)
    diff = 100 * (n_te - n_ho).abs()
    bad = diff.index[~(diff <= band)]  # NaN 도 여기서 걸린다
    return {"columns": int(len(diff)), "test_rows": N_te,
            "outside": {c: {"diff_pp": None if pd.isna(diff[c]) else round(float(diff[c]), 3), "band_pp": round(float(band[c]), 3),
                            "ruler_nonnull": round(float(n_ho[c]), 4), "test_nonnull": round(float(n_te[c]), 4)} for c in bad}}


def main(only: list[str] | None, feats_only: list[str] | None = None) -> int:
    tr, te = train_dep(), test_dep()
    ho = tr[tr["month"].isin(RULER)]
    refs = {f"{a}+{b}": tr[tr["month"].isin((a, b))] for a, b in REFS}
    stems = only if only is not None else ([] if feats_only else sorted(p.stem for p in QDIR.glob("*.json")))
    out = {"rule": "폭 = 2025 다른 달 쌍(2+8, 3+9, 4+10, 5+11, 6+12)이 자(1+7)와 보이는 가장 큰 차이(값은 학습 기간에서만 나오고, 설계는 시험 입력 통계를 본 뒤 "
                   "정했다). 피처 채움 비율은 "
                   f"그 폭과 이항 표준오차 {Z:g}배 중 큰 쪽. NaN 은 실패", "holdout_rows": len(ho), "test_rows": len(te), "checks": [], "candidates": []}

    def add(name: str, value: float, band, ref: dict, ok: bool | None = None, **kind) -> None:
        v = float(value)
        if ok is None:
            ok = (not math.isnan(v)) and (band[0] <= v <= band[1] if isinstance(band, list) else v <= band)
        out["checks"].append({"check": name, "value": None if math.isnan(v) else round(v, 5), "band": band, "ref": ref, "pass": bool(ok), **kind})

    seg = {"공항": lambda d: d["ADEP_mvt"], "시각(UTC 시)": lambda d: d["MVT_TIME_UTC_mvt"].dt.hour,
           "요일": lambda d: d["MVT_TIME_UTC_mvt"].dt.weekday}
    for k, f in seg.items():
        ref = {r: round(tvd(f(d), f(ho)), 5) for r, d in refs.items()}
        add(f"{k} 분포 TVD(시험 대 자)", tvd(f(te), f(ho)), max(ref.values()), ref)
    rates = {"NM 편 매칭(FLIGHT_ID 있음)": lambda d: float(d["FLIGHT_ID_mvt"].notna().mean()),
             "LIRF 의 AOBT_3 없음": lambda d: float(((d["ADEP_mvt"] == "LIRF") & d["AOBT_3_flt"].isna()).mean())}
    for k, f in rates.items():
        ref = {r: round(100 * abs(f(d) - f(ho)), 4) for r, d in refs.items()}
        add(f"{k} 비율 차(%p, 시험 {100 * f(te):.3f} 대 자 {100 * f(ho):.3f})", 100 * abs(f(te) - f(ho)), max(ref.values()), ref)

    y = lambda d: d["TAXITIME_SEC_mvt"].astype(float)  # noqa: E731
    tail_ref = {r: round(float((y(d) > 3600).mean() / (y(ho) > 3600).mean()), 4) for r, d in refs.items()}
    mean_ref = {r: round(float(y(d).mean() / y(ho).mean()), 4) for r, d in refs.items()}

    test_ids, month_of, feat = set(te["MVT_ID_mvt"]), tr.set_index("MVT_ID_mvt")["month"], {}
    for st in stems:
        ci = cand_info(st)
        if ci is None:
            if only:
                add(f"{st}: 큐의 CANDIDATE 다", 0, 0, {}, ok=False)
            continue
        out["candidates"].append(st)
        if not ci.get("holdout") or not ci.get("feats"):
            add(f"{st}: 홀드아웃과 피처 판 기록이 있다", 0, 0, {}, ok=False)
            continue
        hp, tp = pd.read_parquet(ROOT / ci["holdout"])["pred"], pd.read_parquet(QDIR / f"{st}.parquet")["pred"]
        add(f"{st}: 3,600초 초과 예측 비중 비(시험 {100 * (tp > 3600).mean():.3f}% / 자 {100 * (hp > 3600).mean():.3f}%, 폭은 정답 비중의 달 쌍 비)",
            (tp > 3600).mean() / (hp > 3600).mean() if (hp > 3600).mean() else float("nan"), sym(tail_ref), tail_ref)
        add(f"{st}: 예측 평균 비(시험 {tp.mean():.1f} / 자 {hp.mean():.1f}, 폭은 정답 평균의 달 쌍 비)", tp.mean() / hp.mean(), sym(mean_ref), mean_ref)
        fv = ci["feats"]
        if fv not in feat:
            feat[fv] = feat_fill(fv, test_ids, month_of)
        add(f"{st}: 피처 {fv} 채움 비율(칸 {feat[fv]['columns']}개, 시험 행 {feat[fv]['test_rows']}, 폭 밖이나 NaN {len(feat[fv]['outside'])}개)",
            len(feat[fv]["outside"]), 0, feat[fv]["outside"], kind="feat_fill", feats=fv, stem=st)  # kind, feats 는 묶음 예외(src/exception.py)가 읽는다
    for fv in feats_only or []:
        f = feat_fill(fv, test_ids, month_of)
        out["candidates"].append(f"(피처 판 {fv})")
        add(f"피처 판 {fv} 만: 채움 비율(칸 {f['columns']}개, 시험 행 {f['test_rows']}, 폭 밖이나 NaN {len(f['outside'])}개)", len(f["outside"]), 0, f["outside"],
            kind="feat_fill", feats=fv)
    for fv in sorted({ci["feats"] for st in stems if (ci := cand_info(st)) and ci.get("feats")} | set(feats_only or [])):
        tc = ROOT / "notes" / f"te_coverage_{fv}.json"
        if tc.exists():  # 정보만: 모델 안 그룹 통계의 키 등장 비율. 2026 시험의 새 키(해가 바뀐 신규 스탠드, 운항사, 목적지)는 2025 자로 재현할 수 없어 판정에 넣지 않는다
            j = json.loads(tc.read_text(encoding="utf-8"))
            below = {c: v for c, v in j.get("key_seen_summary", {}).items() if v.get("test", 1) < v.get("pseudo_min", 0)}
            out.setdefault("info", {})[f"te_key_seen_{fv}"] = {"file": tc.relative_to(ROOT).as_posix(), "columns": len(j.get("key_seen_summary", {})),
                                                               "test_below_pseudo_min": below, "note": "판정 밖(구조적 한계). private notes G2 에 적었다"}
    out["all_pass"] = all(c["pass"] for c in out["checks"])
    out["queue_dir"] = QDIR.relative_to(ROOT).as_posix()
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in out["checks"]:
        print("PASS" if c["pass"] else "FAIL", c["check"], c["value"], c["band"])
    print("all_pass", out["all_pass"])
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--queue-dir" in a:
        QDIR = ROOT / a[a.index("--queue-dir") + 1]
    if "--out" in a:
        OUT = ROOT / a[a.index("--out") + 1]
    def rest(flag: str) -> list[str] | None:
        if flag not in a:
            return None
        vals = []
        for x in a[a.index(flag) + 1:]:
            if x.startswith("--"):
                break
            vals.append(x)
        return vals
    sys.exit(main(rest("--only"), rest("--feats")))
