# How it works: architecture and data flow

This page shows how the parts of the repository connect, from the challenge files to the final file: which script reads what, writes what, and is fitted on which rows. Why each part exists, and what it bought on our rulers and on the board, is in `docs/DESIGN.md` (decision index at its top). The commands in order are in `docs/REPRODUCE.md`. Running the core model on other data is covered in `docs/USAGE.md`.

## 1. Data flow

```
 challenge files (data/)                    open data
   training_*.parquet (2025, labelled)        adsb.lol days -> research/adsb/ -> cache/adsb_v1, adsb_v2      per-departure ADS-B tables
   ranking.parquet (period to predict)        ADS-B stops, OpenStreetMap stands -> research/obs2/, research/ctx_corr/ -> cache/geo_v1
                                              research/adsb/gprobe/build_nd.py -> cache/adsb_v1nd, geo_v1nd  corrector inputs
   submitting.parquet (template)              IEM METAR -> src/metar_fetch.py -> data/metar/
            |                                 EUROCONTROL AIU -> data/ext/aiu/
            v
 src/load.py -> cache/all.parquet             all movements, timestamps in epoch seconds, src = month (0 = to predict)
            v
 src/feats.py -> cache/feats_v25              one row per departure (feature set 2.5)
 src/feats_adsb.py -> cache/feats_v27         + 28 ADS-B columns (feature set 2.7)
 src/feats_ctx.py, src/feats_ctx2.py, src/feats_aiu.py -> cache/ctx_v3      context table of the stage-2 corrector
            v
 legs, for each setting (val, fwd, full) and seed -> cache/<setting>_<tag>_s<seed>.parquet
   src/model3.py         LightGBM: copy classifier p_art, regression mu, G model (src/sub_g.py)
   src/cat_mu.py, src/rmlp_mu.py, src/xgb_mu.py, research/r13_l/tabm_leg.py    more regressions of mu on the same rows
   research/r13_f3/model3ap.py, research/r13_f3/cat_muap.py                    per-airport regressions (chain C)
            v
 research/r10_b/s20_avg_legs.py (seed averages) -> src/blend_fixed.py (fixed weights) -> research/r13_l/blend_add.py (TabM)
            v
 src/ctx_corr3.py, seeds 0 to 2   stage 1: ADS-B ground traces (src/adsb_corr.py); stage 2: context; LOBT window (src/lobt_window.py)
 research/r10_b/cavg.py           average of the three corrector seeds
            v
 python -m src.manifest ...  -> preds/holdout_<spec>, preds/fwd_<spec>, preds/test_<spec> and preds/manifest_<spec>.json
            v
 tail steps, each reading a manifest and writing the next one (tag suffix in brackets):
   tools/artmm.py [am] -> tools/arthist.py [fh] -> tools/gkey.py [gkc] -> tools/gkeyx.py [x] -> tools/gkeyxn.py [ncr]
   -> src/aiu_shift.py [db] -> tools/antiwin.py [aw] -> tools/gpool.py [gp] -> tools/glong.py [gl] -> tools/gjoint.py [gja]
            v
 chains A, B and C (section 3) -> tools/chainavg3.py [avskap]: (A + B + C) / 3
            v
 daily file: src/make_sub.py            final file: research/r21/r2a_tool.py (post-step R2a), then src/fire_queue.py (build_final)
```

Every step runs once per setting. A step's output for `val` (the ruler months held out), `fwd` (July predicted from January to June) and `full` (all training months, the period to predict) are separate files. A chain step's manifest names all three: `holdout`, `forward` and `test`.

## 2. Stages

| Stage | Script | Reads | Writes | Fitted on |
|---|---|---|---|---|
| Load | `src/load.py` | `data/training_*.parquet`, `data/ranking.parquet` | `cache/all.parquet` | nothing |
| Features | `src/feats.py`, `src/feats_adsb.py` | `cache/all.parquet`, METAR files, the ADS-B table | `cache/<feature table>.parquet` | nothing; the target encodings (`src/te.py`) are computed inside each leg run from that run's fitting rows |
| Leg | `src/model3.py` and the other legs | the feature table, the base leg's output (for the classifier p_art and the G rows) | `cache/<setting>_<tag>_s<seed>.parquet`, `preds/es_valid_ids_<tag>_<setting>_s<seed>.parquet` | the setting's training months; early stopping on the training days whose UTC day number is divisible by 10 |
| Blend | `src/blend_fixed.py` | the legs' outputs | a blended output in `cache/` | weights fitted once by `src/blend_oof.py` on out-of-fold predictions of the ruler days of an earlier blend, then applied unchanged |
| Corrector | `src/ctx_corr3.py` | the blend's output, the ADS-B, stand-geometry and context tables | a corrected output in `cache/`, a report in `notes/` | January and July 2025: stage 1 on the departures matched to ADS-B, stage 2 on the rows predicted below 3,600 s; in `val`, day folds keep every row away from the models that saw its day |
| Chain format | `src/manifest.py` | the corrector average's outputs | `preds/holdout_<spec>`, `preds/fwd_<spec>`, `preds/test_<spec>`, `preds/manifest_<spec>.json` | nothing |
| Tail steps | `tools/*.py`, `src/aiu_shift.py` | the previous step's manifest and prediction files, their own tables (`table` sub-command) | the next manifest and prediction files | small constants, each on the months the setting allows (`docs/DESIGN.md` section 8) |
| Chain average | `tools/chainavg3.py` | three chain ends | the average | nothing (weights 1/3 fixed in advance) |
| File | `src/make_sub.py`, `src/fire_queue.py` | the test prediction, the template | the submission file | nothing (template order, rounding, clip at 0, int32) |

## 3. The three chains

The final stack averages three chains that differ in the regression target and the learner family (`docs/DESIGN.md` section 8.7):

- A: legs that regress the label relative to an anchor inside the LOBT window (`--anchor win`), LightGBM, CatBoost and RealMLP, plus TabM.
- B: the same learners without an anchor.
- C: per-airport LightGBM and CatBoost regressions, with RealMLP.

Each chain has its own blend, corrector and tail steps; the final file also uses refits of the legs and seed averages described in `docs/DESIGN.md` sections 14.3 and 14.5.

## 4. Names of files and tags

A run writes `cache/<setting>_<tag>_s<seed>.parquet` with at least `MVT_ID_mvt`, `pred` and, for `val` and `fwd`, the label `y`; legs and blends also keep `p_art`, `mu`, `g` (G row flag) and `dsch`. The tag of a step is the tag of its input plus a suffix, so a tag spells its chain. For example `m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3am` reads: the blend tag `m6r27wincbrmfw4sv434` (legs, learners and seed counts of chain A), `tmc` (TabM added), `adsbgeondctxv3x50w` (the corrector: stage 1 on the ADS-B and stand tables `adsb_v1nd` and `geo_v1nd`, stage 2 on context table v3, the stage-2 correction times 0.5, the LOBT window), `c3` (three corrector seeds averaged), `am` (the tail step artmm). The full list of suffixes is in the glossary of `docs/REVIEW_GUIDE.md` (section 4).

## 5. Records that tie the outputs to the code

Every chain step writes a manifest with its command lines, seeds, the feature table and its hash, the SHA-256 prefix of every code file it ran, its ruler score and the hashes of its prediction files. The manifests of the submitted chains and of the final file are in `preds/`. `python scripts/verify_code_hashes.py` checks the code of this repository against them (`docs/REPRODUCE.md` sections 0.5 and 9). Prediction files and derived tables are not in the repository (`RELEASE_MANIFEST.md` section 5).

## 6. Code map

| Folder | What is there |
|---|---|
| `src/` | the model: loading, features, target encodings, legs, blends, correctors, LOBT window, ruler; and the challenge tooling (launcher, upload, gates, ledger, batch registration, diagnostics) |
| `tools/` | the tail steps, chain averaging and queue tools |
| `research/` | builders of the input tables (ADS-B, stand geometry), legs and steps that live outside `src/` (TabM, per-airport legs, seed averages, corrector average, post-step R2a), the final-phase rebuild and gates, and the analysis scripts that the documents cite |
| `examples/` | the synthetic sample, the input checker and the prediction writer of `docs/USAGE.md` |
| `preds/`, `notes/`, `results/` | manifests, job lines, step notes, submission history and organizer scores |
| `external_data/metar/` | the METAR files that every build read (`docs/DATA_AND_LICENSES.md` section 3); a rebuild copies them to `data/metar/` |
| `scripts/` | the hash check and the host scheduling scripts of the final rebuild |

`docs/USAGE.md` section 6 says which of these parts can be reused outside the challenge.
