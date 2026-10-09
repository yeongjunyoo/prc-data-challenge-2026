# SPDX-License-Identifier: GPL-3.0-only
"""A3: A2 (src/ctx_corr2.py, imported, unchanged, which imports src/ctx_corr.py unchanged) plus EUROCONTROL AIU airport aggregates in stage 2.
Run it like src/ctx_corr2.py with --ctx naming a file built by src/feats_aiu.py (cache/ctx_v3: the ctx_v2 columns plus five AIU columns).
Added stage-2 columns (everything else is A2's: stage 1, folds, pairwise nesting, parameters, --scale2, the window last):
  aiu_m_txo, aiu_m_add, aiu_m_valid, aiu_d_pdd, aiu_d_txo   per-row airport-month and airport-day aggregates (src/feats_aiu.py)
  aiu_m_gap   aiu_m_txo minus the mean of clip(stage-1 prediction, 0, 3600) over the rows of the same airport and month of MVT
  aiu_d_gap   aiu_d_pdd minus the mean of max(0, (MVT - SCHED) - stage-1 prediction) over the rows of the same airport and UTC day of MVT,
              i.e. the published mean pre-departure delay against the one our predictions imply (positive when we predict too long)
The two gaps use only the stage-1 predictions and inputs of the rows being corrected (all rows of a setting, as in the test file).
--aiu_date block (option, 10-07; without it everything above is unchanged): aiu_d_pdd, aiu_d_txo and aiu_d_gap use the airport-day of each
  row's estimated off-block time, the UTC date of MVT minus the stage-1 prediction, instead of the UTC day of MVT. aiu_d_pdd is the AIU daily
  value of that date (the daily files cache/<ctx>.json names, sha16-checked), aiu_d_txo the mean of MVT - SCHED over the setting's rows of
  that airport and date minus aiu_d_pdd, and aiu_d_gap groups by that date. Rows whose date is not one of the setting's MVT dates (the day
  before a month or a test day) get NaN in the three. AIU counts a departure's pre-departure delay on its off-block date (vault
  research/2026-10-07-r7-u.md 6.1). Unlike --drop2 and --fit_cache it changes the output, so a candidate's command carries it.
Research-only flags as in src/ctx_corr2.py (--drop2, --fit_cache), removed before ctx_corr parses the rest.
    .venv/Scripts/python.exe -u src/ctx_corr3.py --setting val --seed 0 --base m6r25sgoof4 --base_seed 0 --tag <tag> --feats feats_v25
        --adsb adsb_v1 --adsb_sha <16> --geo geo_v1 --geo_sha <16> --ctx ctx_v3 --ctx_sha <16> --folds 10 --rounds 300 --rounds2 400
        --max_pred 3600 --threads 4 --scale2 0.5
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import src.ctx_corr as cc  # noqa: E402
import src.ctx_corr2 as c2  # noqa: E402

AIU_CTX = ["aiu_m_txo", "aiu_m_add", "aiu_m_valid", "aiu_d_pdd", "aiu_d_txo"]
AIU_GAP = ["aiu_m_gap", "aiu_d_gap"]
AIU_DATE = "mvt"  # --aiu_date: "mvt" (default) or "block"
DAILY = None  # --aiu_date block: AIU daily mean pre-departure delay per "airport|date"


def group_mean(codes: np.ndarray, x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    s = np.bincount(codes, weights=np.where(ok, x, 0.0))
    n = np.bincount(codes, weights=ok.astype(float))
    return np.where(n[codes] > 0, s[codes] / np.maximum(n[codes], 1), np.nan)


def aiu_daily(ctx: str) -> pd.Series:
    """--aiu_date block: the AIU daily mean pre-departure delay (src/feats_aiu.daily, the files cache/<ctx> was built from, sha16-checked
    against cache/<ctx>.json) per "airport|date"."""
    import json
    import src.feats_aiu as fa
    meta = json.loads((Path("cache") / f"{ctx}.json").read_text(encoding="utf-8"))
    for f, want in meta["inputs_sha16"].items():
        if f.startswith("all_pre_departure_delays_") and fa.sha16(fa.AIU / f) != want:
            raise SystemExit(f"data/ext/aiu/{f} sha16 differs from cache/{ctx}.json")
    d = fa.daily()
    return pd.Series(d["aiu_d_pdd"].to_numpy(dtype=float), index=(d["ap"] + "|" + d["day"]).to_numpy())


def block_day_cols(R, s1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """--aiu_date block: (aiu_d_pdd, aiu_d_txo, aiu_d_gap) by the UTC date of MVT - s1; NaN where that date is not one of R's MVT dates."""
    bday = pd.to_datetime(R.t - s1, unit="s").strftime("%Y-%m-%d").to_numpy()
    ok = pd.Series(bday).isin(set(R.day)).to_numpy()
    key = pd.Series(R.ap) + "|" + pd.Series(bday)
    gb = pd.factorize(key)[0]
    pdd = np.where(ok, key.map(DAILY).to_numpy(dtype=float), np.nan)
    txo = np.where(ok, group_mean(gb, R.cols["d_sch"]), np.nan) - pdd
    gap = pdd - group_mean(gb, np.maximum(0.0, R.cols["d_sch"] - s1))
    return pdd, txo, gap


class Rows3(c2.Rows2):
    def design2(self, s1, delta):
        if not hasattr(self, "_gm"):
            self._gm = pd.factorize(pd.Series(self.ap) + "|" + pd.Series(self.part))[0]
            self._gd = pd.factorize(pd.Series(self.ap) + "|" + pd.Series(self.day))[0]
        if AIU_DATE == "block":
            self.cols["aiu_d_pdd"], self.cols["aiu_d_txo"], self.cols["aiu_d_gap"] = block_day_cols(self, s1)
            self.cols["aiu_m_gap"] = self.cols["aiu_m_txo"] - group_mean(self._gm, np.clip(s1, 0, 3600))
            return super().design2(s1, delta)
        self.cols["aiu_m_gap"] = self.cols["aiu_m_txo"] - group_mean(self._gm, np.clip(s1, 0, 3600))
        self.cols["aiu_d_gap"] = self.cols["aiu_d_pdd"] - group_mean(self._gd, np.maximum(0.0, self.cols["d_sch"] - s1))
        return super().design2(s1, delta)


def main():
    global AIU_DATE, DAILY
    drop = c2.pop_flag("--drop2")
    cache = c2.pop_flag("--fit_cache")
    AIU_DATE = c2.pop_flag("--aiu_date") or "mvt"
    if AIU_DATE not in ("mvt", "block"):
        raise SystemExit(f"--aiu_date: mvt or block, not {AIU_DATE}")
    if AIU_DATE == "block":
        DAILY = aiu_daily(sys.argv[sys.argv.index("--ctx") + 1])
        print(f"  --aiu_date block: AIU day columns by the UTC date of MVT - stage-1 prediction ({len(DAILY)} airport-days)", flush=True)
    cc.CTX_COLS = cc.CTX_COLS + c2.NEW_CTX + AIU_CTX
    x2 = cc.X2 + c2.NEW_CTX + list(c2.PRED_GAP) + AIU_CTX + AIU_GAP
    if drop:
        gone = set(drop.split(","))
        unknown = gone - set(x2)
        if unknown:
            raise SystemExit(f"--drop2: not stage-2 columns: {sorted(unknown)}")
        x2 = [c for c in x2 if c not in gone]
        print(f"  stage-2 columns without {sorted(gone)}: {len(x2)}", flush=True)
    cc.X2 = x2
    cc.Rows = Rows3
    if cache:
        c2.fit_cache(Path(cache))
    cc.main()


if __name__ == "__main__":
    main()
