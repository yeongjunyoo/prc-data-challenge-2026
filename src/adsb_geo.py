# SPDX-License-Identifier: GPL-3.0-only
"""The ADS-B corrector of src/adsb_corr.py (imported unchanged) with six stand-geometry columns added to its ADS-B columns.

Geometry table cache/<geo>.parquet (research/obs2/build_geo.py; its SHA-256 prefix must equal --geo_sha), per departure:
    d_stand_first   metres from the position of the stand STAND_mvt (median of our ADS-B stand stops, else the OpenStreetMap
                    aeroway=parking_position) to the first ADS-B ground point
    st_known        whether that stand position is known
    d_stand_stop    metres from the ADS-B stand stop to the stand of the record (towing, remote stands)
    d_any_stand     metres from the first point to the nearest known stand (apron or taxiway)
    seg_code        why the ground segment starts (0 start of the track, 1 after a gap, 2 landing)
    to_gap          seconds between the last ground point and the first airborne point
No label is used. Data: adsb.lol globe_history and OpenStreetMap (both ODbL 1.0). Training, folds, outputs as in src/adsb_corr.py.

    python src/adsb_geo.py --setting val --seed 0 --base <tag> --tag <tag>adsbgeow --feats feats_v25 --adsb adsb_v1 --adsb_sha <16>
        --geo adsb_geo_v1 --geo_sha <16> --folds 10 --rounds 300 --threads 3 --scale 1.0

Original note (Korean):
ADS-B 보정기에 스탠드 기하 칸 여섯을 더한 판. src/adsb_corr.py 를 고치지 않고 불러 ADS-B 칸 목록만 늘린다.

기하 칸(cache/<geo>.parquet, research/obs2/build_geo.py 가 만듦, --geo_sha 와 sha256 앞 16자가 같아야 한다):
  d_stand_first  PRC STAND_mvt 의 위치(우리 ADS-B 스탠드 정지 위치의 중앙값, 없으면 OSM aeroway=parking_position)에서 첫 ADS-B 점까지(m)
  st_known       그 스탠드 위치를 아는가
  d_stand_stop   ADS-B 스탠드 정지 위치에서 PRC 스탠드까지(견인, 원격 스탠드 판별)
  d_any_stand    첫 점에서 가장 가까운 아는 스탠드까지(주기장인지 유도로인지)
  seg_code       지상 구간 시작 이유(0 궤적 시작, 1 공백으로 끊김, 2 착륙)
  to_gap         마지막 지상 점과 첫 공중 점 사이(초)
정답은 쓰지 않는다. 자료: adsb.lol globe_history(ODbL 1.0), OpenStreetMap(ODbL 1.0, © OpenStreetMap contributors), README 「외부 데이터」.
보정기 학습, 겹, 시드, 반복, 매개변수, 창, 출력 꼴은 src/adsb_corr.py 와 같다(val: 날 묶은 --folds 겹, fwd: 1월 날만, full: --folds 개 평균).
    .venv/Scripts/python.exe -u src/adsb_geo.py --setting val --seed 0 --base m6r25sgcbxr --tag m6r25sgcbxradsbgeow --feats feats_v25 --adsb adsb_v1 --adsb_sha <16> --geo adsb_geo_v1 --geo_sha <16> --folds 10 --rounds 300 --threads 3 --scale 1.0
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import src.adsb_corr as ac  # noqa: E402
from src.common import breakdown, rmse  # noqa: E402
from src.load import CACHE  # noqa: E402
from src.lobt_window import feats_d_lobt, mu_of, project  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
GEO = ["d_stand_first", "st_known", "d_stand_stop", "d_any_stand", "seg_code", "to_gap"]


def load_geo(name: str, want_sha: str) -> pd.DataFrame:
    """Stand-geometry columns of cache/<name>.parquet (after checking its SHA-256 prefix), indexed by MVT_ID_mvt."""
    p = CACHE / f"{name}.parquet"
    got = ac.sha16(p)
    if got != want_sha:
        raise SystemExit(f"{p.name} sha {got} != --geo_sha {want_sha}")
    return pd.read_parquet(p, columns=["MVT_ID_mvt"] + GEO).set_index("MVT_ID_mvt")


def main():
    """Command line entry: the ADS-B corrector with the geometry columns, for one setting."""
    pa = argparse.ArgumentParser()
    pa.add_argument("--setting", default="val", choices=["val", "fwd", "full"])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--base", required=True)
    pa.add_argument("--tag", required=True)
    pa.add_argument("--feats", default="feats_v25")
    pa.add_argument("--adsb", required=True)
    pa.add_argument("--adsb_sha", required=True)
    pa.add_argument("--geo", required=True)
    pa.add_argument("--geo_sha", required=True)
    pa.add_argument("--folds", type=int, default=10)
    pa.add_argument("--rounds", type=int, default=300)
    pa.add_argument("--threads", type=int, default=3)
    pa.add_argument("--scale", type=float, default=1.0)
    a = pa.parse_args()
    ac.below_normal()
    ac.ADSB_COLS = list(ac.ADSB_COLS) + GEO
    A = ac.load_adsb(a.adsb, a.adsb_sha).join(load_geo(a.geo, a.geo_sha), how="left")
    val = pd.read_parquet(CACHE / f"val_{a.base}_s{a.seed}.parquet")
    T = ac.design(val, A)
    T = T[np.isfinite(T["y"].to_numpy())]
    b = val if a.setting == "val" else pd.read_parquet(CACHE / f"{a.setting}_{a.base}_s{a.seed}.parquet")
    X = T if a.setting == "val" else ac.design(b, A)
    delta = np.zeros(len(b))
    if a.setting == "val":
        models, f = ac.fold_models(T, a.folds, a.seed, a.rounds, a.threads)
        oof = np.zeros(len(T))
        for j, mdl in enumerate(models):
            oof[f == j] = ac.predict(mdl, T[f == j])
        delta[T["pos"].to_numpy()] = oof
        note = f"{a.folds} day folds over {T['day'].nunique()} days, {len(T)} matched rows"
    elif a.setting == "fwd":
        jan = T[T["day"].str[5:7] == "01"]
        delta[X["pos"].to_numpy()] = ac.predict(ac.fit(jan, a.seed, a.rounds, a.threads), X)
        note = f"January only: {jan['day'].nunique()} days, {len(jan)} matched rows"
    else:
        models, _ = ac.fold_models(T, a.folds, a.seed, a.rounds, a.threads)
        delta[X["pos"].to_numpy()] = np.mean([ac.predict(mdl, X) for mdl in models], axis=0)
        note = f"mean of {a.folds} day-fold correctors over {T['day'].nunique()} days, {len(T)} matched rows"
    delta = a.scale * delta
    base_pred = b["pred"].to_numpy(dtype=float)
    which, mu = mu_of(b)
    p_art, g, dsch = b["p_art"].to_numpy(dtype=float), b["g"].to_numpy().astype(bool), b["dsch"].to_numpy(dtype=float)
    d = feats_d_lobt(a.feats, b["MVT_ID_mvt"].to_numpy())
    pred_w, p2, _ = project(np.clip(base_pred + delta, 0, None), mu, p_art, g, dsch, d)
    pred_basew, _, _ = project(base_pred, mu, p_art, g, dsch, d)
    assert np.isfinite(pred_w).all() and (pred_w >= 0).all()
    unmatched = np.ones(len(b), bool)
    unmatched[X["pos"].to_numpy()] = False
    assert np.array_equal(pred_w[unmatched], pred_basew[unmatched]), "매칭 안 된 행이 바탕 + 창과 달라졌다"
    out = b.copy()
    out["pred"], out["p_art"] = pred_w, p2
    print(f"[{a.tag} {a.setting} seed{a.seed}] 바탕 {a.base}(mu = {which}), ADS-B {a.adsb}({a.adsb_sha}), 기하 {a.geo}({a.geo_sha}), "
          f"매칭 행 {len(X):,} / {len(b):,}, 보정 평균 {delta[X['pos'].to_numpy()].mean():+.1f}초, {note}")
    if "y" in b.columns:
        y = b["y"].to_numpy(dtype=float)
        print(f"  바탕 {rmse(base_pred, y):.3f} → 바탕+창 {rmse(pred_basew, y):.3f} → ADS-B 기하 보정+창 {rmse(pred_w, y):.3f}")
        import pyarrow.parquet as pq
        ap = pq.read_table(CACHE / f"{a.feats}.parquet", columns=["MVT_ID_mvt", "ap"]).to_pandas(strings_to_categorical=True)
        ap = ap.set_index("MVT_ID_mvt")["ap"].reindex(b["MVT_ID_mvt"].to_numpy()).astype(str).to_numpy()
        print(breakdown(pred_w, y, ap, f"{a.tag} {a.setting}"))
        rep = ac.report(b, pred_w, pred_basew, X, a.setting, a.tag)
        rep.update(tag=a.tag, setting=a.setting, base=a.base, adsb=a.adsb, adsb_sha=a.adsb_sha, geo=a.geo, geo_sha=a.geo_sha, note=note,
                   rmse_base=round(rmse(base_pred, y), 3), rmse_base_window=round(rmse(pred_basew, y), 3),
                   rmse_adsb_window=round(rmse(pred_w, y), 3))
        (CACHE.parent / "notes" / f"adsb_corr_{a.tag}_{a.setting}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    dst = CACHE / f"{a.setting}_{a.tag}_s{a.seed}.parquet"
    out.to_parquet(dst, index=False)
    if a.setting != "full":
        P = CACHE.parent / "preds"
        shutil.copyfile(P / f"es_valid_ids_{a.base}_{a.setting}_s{a.seed}.parquet", P / f"es_valid_ids_{a.tag}_{a.setting}_s{a.seed}.parquet")
    print(f"saved {dst.name}")


if __name__ == "__main__":
    main()
