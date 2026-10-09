# SPDX-License-Identifier: GPL-3.0-only
"""LOBT 창: 학습 자료의 성질 |BLOCK − LOBT| ≤ 3,606초(2025 출발편 중 LOBT 가 있는 행 전부, 위반 0)에서 정답은 [d_lobt − 3,606, d_lobt + 3,606] 안에 있다
(d_lobt = 이동 시각 − LOBT, 피처 판의 칸). 출처: 공개 팀 outgoing-boat 의 연구 노트(EnioAguiar/prc-taxiout-2026, docs/research/2026-09-27-janela-lobt.md),
우리 자료로 다시 재어 확인했다(코드는 가져오지 않음). 판을 바꾸는 일은 둘이다.
  1) 일정 복사(art) 값 d_sch 가 창 밖이면(±60초 여유, art 는 |y − d_sch| ≤ 60) 그 행의 art 확률을 0 으로: pred = max(mu, 0). mu 는 바탕 pred 의 혼합식에 맞는 것(mu_of). G 행(바탕 G 규칙)은 건드리지 않는다.
  2) 모든 예측을 창 안으로 옮긴다(창이 있는 행만).
    .venv/Scripts/python.exe -X utf8 src/lobt_window.py check [--tag m6r25sgcb]       2025 위반 수, 그 판 시험 예측의 창 밖 행과 보장 하한
    .venv/Scripts/python.exe -X utf8 src/lobt_window.py apply --setting val --seed 0 --base <tag> --tag <tag>w --feats feats_v25
출력: cache/<setting>_<tag>_s<seed>.parquet (바탕과 같은 칸, pred 와 p_art 만 바뀜)
"""
from __future__ import annotations

import argparse
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.common import breakdown, rmse  # noqa: E402
from src.load import CACHE  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
HALF = 3606.0  # 2025 의 최대 |BLOCK − LOBT|
ART_TOL = 60.0


def window(d_lobt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return d_lobt - HALF, d_lobt + HALF


def project(pred, mu, p_art, g, dsch, d_lobt):
    """(새 pred, 새 p_art, 바뀐 행 표시)."""
    lo, hi = window(d_lobt)
    has = np.isfinite(d_lobt)
    art_out = has & ~g & (p_art > 0) & ((dsch < lo - ART_TOL) | (dsch > hi + ART_TOL))
    p2 = np.where(art_out, 0.0, p_art)
    out = np.where(art_out, np.maximum(mu, 0.0), pred)
    out = np.where(has, np.clip(out, lo, hi), out)
    return out, p2, art_out | (out != pred)


def feats_d_lobt(name: str, ids: np.ndarray) -> np.ndarray:
    F = pd.read_parquet(CACHE / f"{name}.parquet", columns=["MVT_ID_mvt", "d_lobt"]).set_index("MVT_ID_mvt")
    return F["d_lobt"].reindex(ids).to_numpy(dtype=float)


def check(tag: str, feats: str) -> None:
    F = pd.read_parquet(CACHE / f"{feats}.parquet", columns=["MVT_ID_mvt", "src", "y", "d_lobt"])
    tr = (F["src"] != 0).to_numpy() & np.isfinite(F["d_lobt"].to_numpy()) & np.isfinite(F["y"].to_numpy())
    gap = np.abs(F["d_lobt"].to_numpy()[tr] - F["y"].to_numpy()[tr])
    print(f"2025 LOBT 있는 행 {int(tr.sum()):,}: |y − d_lobt| > {HALF:.0f} 은 {int((gap > HALF).sum())} 행, 최대 {gap.max():.1f}")
    b = pd.read_parquet(CACHE / f"full_{tag}_s0.parquet")
    d = feats_d_lobt(feats, b["MVT_ID_mvt"].to_numpy())
    lo, hi = window(d)
    pr = b["pred"].to_numpy()
    below, above = np.isfinite(d) & (pr < lo), np.isfinite(d) & (pr > hi)
    gain = (np.sum((lo[below] - pr[below]) ** 2) + np.sum((pr[above] - hi[above]) ** 2)) / len(pr)
    print(f"{tag} 시험 예측 {len(pr):,} 행 중 LOBT 있는 행 {int(np.isfinite(d).sum()):,}, 창 밖 {int(below.sum() + above.sum())} 행(아래 {int(below.sum())}, 위 {int(above.sum())}). "
          f"옮기면 보장되는 MSE 감소 하한 {gain:,.0f}")


def mu_of(b: pd.DataFrame) -> tuple[str, np.ndarray]:
    """바탕 pred 를 만든 혼합의 mu. G 가 아닌 행에서 pred = clip(p_art·d_sch + (1 − p_art)·mu, 0) 가 맞는 것을 고르고, 맞는 것이 없으면 멈춘다."""
    g = b["g"].to_numpy().astype(bool)
    p, d, pr = (b[c].to_numpy(dtype=float) for c in ("p_art", "dsch", "pred"))
    cands = {"mu": b["mu"].to_numpy(dtype=float)}
    if "mu_cat" in b.columns:  # cat_mu.py 의 출력: mu_b = (1 − W)·mu + W·mu_cat, W = 0.5
        cands["0.5 mu + 0.5 mu_cat"] = 0.5 * b["mu"].to_numpy(dtype=float) + 0.5 * b["mu_cat"].to_numpy(dtype=float)
    for k, m in cands.items():
        if np.abs(pr - np.clip(p * d + (1 - p) * m, 0, None))[~g].max() < 1e-6:
            return k, m
    raise SystemExit("바탕 pred 가 어느 mu 의 혼합식과도 맞지 않는다: " + ", ".join(cands))


def apply(setting: str, seed: int, base: str, tag: str, feats: str) -> None:
    b = pd.read_parquet(CACHE / f"{setting}_{base}_s{seed}.parquet")
    F = pd.read_parquet(CACHE / f"{feats}.parquet", columns=["MVT_ID_mvt", "d_lobt", "ap"]).set_index("MVT_ID_mvt").reindex(b["MVT_ID_mvt"].to_numpy())
    which, mu = mu_of(b)
    pred, p2, changed = project(b["pred"].to_numpy(dtype=float), mu, b["p_art"].to_numpy(dtype=float), b["g"].to_numpy().astype(bool),
                                b["dsch"].to_numpy(dtype=float), F["d_lobt"].to_numpy(dtype=float))
    assert np.isfinite(pred).all() and (pred >= 0).all()
    out = b.copy()
    out["pred"], out["p_art"] = pred, p2
    print(f"[{tag} {setting} seed{seed}] 바탕 {base}(mu = {which}), art 확률을 0 으로 둔 행 {int((p2 != b['p_art'].to_numpy()).sum())}, 예측이 바뀐 행 {int(changed.sum())}")
    if "y" in b.columns:
        y = b["y"].to_numpy(dtype=float)
        print(f"  바탕 {rmse(b['pred'].to_numpy(), y):.3f} → 창 {rmse(pred, y):.3f}")
        print(breakdown(pred, y, F["ap"].astype(str).to_numpy(), f"{tag} {setting}"))
    dst = CACHE / f"{setting}_{tag}_s{seed}.parquet"
    out.to_parquet(dst, index=False)
    es = CACHE.parent / "preds" / f"es_valid_ids_{base}_{setting}_s{seed}.parquet"
    if es.exists():  # 창은 학습을 건드리지 않으니 조기 종료 행은 바탕 그대로
        shutil.copyfile(es, es.with_name(f"es_valid_ids_{tag}_{setting}_s{seed}.parquet"))
    print(f"saved {dst.name}")


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("cmd", choices=["check", "apply"])
    pa.add_argument("--setting", default="val", choices=["val", "fwd", "full"])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--base", default="")
    pa.add_argument("--tag", default="m6r25sgcb")
    pa.add_argument("--feats", default="feats_v25")
    a = pa.parse_args()
    if a.cmd == "check":
        check(a.tag, a.feats)
    else:
        apply(a.setting, a.seed, a.base, a.tag, a.feats)
