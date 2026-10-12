# PRC Data Challenge 2026: taxi-out time prediction (team spirited-tiger)

Solution of team spirited-tiger (Yeongjun Yoo, Hongik University, Seoul) for the EUROCONTROL PRC Data Challenge 2026: predict the taxi-out time in seconds of departures from ten European airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM), trained on the 2025 movements of the same airports. The metric is the RMSE in seconds.

This project was developed with the assistance of AI tools.

## Final submission

The final phase asks for one file that predicts January, February, June and July 2026 from `final_ranking.parquet`; the organizers rank it on January and July, on February and June, and on the four months together, and review code and documents.

| Item | Value |
|---|---|
| File | `spirited-tiger_final.parquet`, 670,790 rows, in the order of `final_submitting.parquet` |
| SHA-256 | `68932fffa7f6b8a1ecfa23688518598ff1f6f98809bfb917a03df5fcb9f053c1` |
| Built by | the three-chain stack of `docs/DESIGN.md`, re-run on `final_ranking.parquet` (SHA-256 `4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f`) with the choices of `docs/DESIGN.md` section 14.5 |
| Training labels | the twelve 2025 training files only |
| Months without a public score (February, June) | every choice decided on February and June 2025 held out as unseen months, by rules written before their numbers, with six exceptions: the post-step R2a, whose rule came from a listing of the 2025 rows of its kind with their labels, so that only the condition for adopting it preceded its month-held-out numbers (`docs/DESIGN.md` section 8.8); the LightGBM legs of L3, which act on all four months, were chosen by two January and July board comparisons, v41 against v36 and, as the veto added at 2026-10-10 07:12 KST (private notes), v43 against v42, under rules written down before their scores (`docs/DESIGN.md` section 15.2); the CatBoost and RealMLP legs of L4, which act on all four months as well, were chosen by their local gate, with one January and July board comparison, v44 against v43, as a veto, under a rule written down before any number of L4 (private notes; `docs/DESIGN.md` sections 14.3 and 15.2); the seed averages of package S in chain C, which act on all four months as well, were chosen by the checks of its own rule, with one January and July board comparison, v45 against v44, as a veto, under a rule written down before any number of the package (private notes), after a smaller seed average had failed the gate of the L3 and L4 legs (`docs/DESIGN.md` sections 14.3 and 15.2); the row split of stage 2 of the corrector was decided after its gate numbers, and the scale2 of the matched February and June rows follows the literal reading of an ambiguous registered rule that we first read the other way after seeing its numbers (`docs/DESIGN.md` sections 14.3 and 14.7) |
| What the daily board decided in the final file | scale2 0.75 on the January and July rows: v36 scored ΔMSE -97.5 against v35 on the board, while our two local rulers disagreed (val -298.9, July-forward +273.1). The LightGBM legs retrained with the 2025 ADS-B days filled in (L3): used, because their board check v41 scored 228.0752, below v36's 228.1704, under a rule written before that score, and the final-input rebuild of their January and July rows scored below the same rows in v36's configuration (v43 scored 228.5043, below v42's 228.5347), as the veto added at 2026-10-10 07:12 KST (private notes) before both scores requires; on the final input their January and July gain is about a third of the public-phase gain (ΔMSE -13.9 for v43 against v42, -43.4 for v41 against v36; `docs/DESIGN.md` section 14.7). The CatBoost and RealMLP legs refit the same way (L4): used, because their local gate chose them and the board, which could only veto them, did not: v44, the final-input rebuild of their January and July rows, scored 227.9618, below v43's 228.5043, the same rows with the L3 legs, under a rule written before any number of L4 (private notes; `docs/DESIGN.md` section 15.2). Chain C's legs averaged over more seeds (package S): used, because the checks of its rule passed and the board, which could only veto it, did not: v45, the final-input rebuild of its January and July rows, scored 227.6728 against v44's 227.9618, the same rows without it (the rule drops the package above 228.1810), under a rule written before any number of the package (private notes; `docs/DESIGN.md` section 15.2). Two other knobs read on the board kept their earlier values (chain weights 1/3, monthly AIU columns). The best daily-board file, named below, is board-tuned history, not the final file (`docs/DESIGN.md` section 15.2) |

How the final file is rebuilt, with commands, machines, times, seeds and the limits of bit-for-bit reproduction: `docs/REPRODUCE.md` section 0.

## Method in ten lines

1. One feature table per departure from the organizer files, with open ADS-B ground traces (adsb.lol), METAR, OpenStreetMap stands and EUROCONTROL airport aggregates (`docs/DESIGN.md` section 5).
2. A mixture model: a classifier for whether the recorded off-block time copies the schedule, and regressions of the measured taxi time; a next-day offset is modelled too (section 4).
3. Legs: LightGBM, CatBoost, RealMLP and TabM (XGBoost with weight 0), several seeds each, blended with weights fitted once on out-of-fold predictions (section 6).
4. Two correctors: one reads the departure's ADS-B ground trace, the other context of the airport and day, including published EUROCONTROL daily delay totals (section 7).
5. Tail stages for the record-type probabilities and for the LIRF departures without a Network Manager flight record (section 8).
6. Three chains that differ in learner family or regression anchor, averaged with weights 1/3 fixed before any score (section 8.7).
7. Every model is fitted from scratch on the 2025 labels only (section 1).
8. Choices were measured on local rulers built from 2025 (January and July day folds, and a ruler fitted on January and applied to July) before each upload (section 3); the board choices are listed in section 15.
9. The final file re-runs the stack on `final_ranking.parquet`; for its February and June rows stage 2 of the corrector is refitted on four 2025 months (section 14.7).
10. February and June were decided on February and June 2025 held out as unseen months, under rules written before their numbers, with the exceptions stated above (section 14.3).

## Results

| What | Rows | Score | Source |
|---|---|---|---|
| Daily board (January and July 2026), first file v1: per-airport median | 344,841 | RMSE 668.335 | `results/submissions.csv` |
| Daily board, v35: the stack that the final file re-runs, with the changes of `docs/DESIGN.md` section 14.5 | 344,841 | RMSE 228.384 | `results/submissions.csv` |
| Daily board, v36: v35 with scale2 0.75, the board read of scale2 (the final file uses 0.75 with the L4 legs and package S) | 344,841 | RMSE 228.1704 | `results/submissions.csv` |
| Daily board, v40: the January and July rows of the first final file (v35's configuration on the final input, a check for rebuild errors) | 344,841 | RMSE 228.7879 | `results/submissions.csv` |
| Daily board, v41: v36's configuration with the L3 legs on the public-phase input (the board check of the L3 rule) | 344,841 | RMSE 228.0752 | `results/submissions.csv` |
| Daily board, v42: the January and July rows of the four-month file in v36's configuration on the final input (a check for rebuild errors and the reference of the L3 rule's veto) | 344,841 | RMSE 228.5347 | `results/submissions.csv` |
| Daily board, v43: the January and July rows of the four-month file in v36's configuration with the L3 legs on the final input (the L3 rule's veto check and the reference of the veto of the L4 legs) | 344,841 | RMSE 228.5043 | `results/submissions.csv` |
| Daily board, v44: the January and July rows of the four-month file in v36's configuration with the L4 legs on the final input (the veto check of the L4 legs and the reference of the veto of package S) | 344,841 | RMSE 227.9618 | `results/submissions.csv` |
| Daily board, v45: in effect the final file's January and July rows (v36's configuration with the L4 legs and package S on the final input; 568 rows and 17 integer values differ from the final file's, 11 of these rows moved by the post-step R2a) | 344,841 | RMSE 227.6728 | `results/submissions.csv`; `docs/DESIGN.md` section 14.7 |
| Local ruler (2025 January and July, day folds) of the final file's configuration (v36's with the L4 legs and package S) | 344,419 | RMSE 281.08 | `preds/manifest_m6r28xwincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3rBbFamfhgkcxncrdbawgpglgjaavskapsff13.json` (field `holdout`; the chain end before the post-step R2a, `docs/DESIGN.md` section 8.8) |
| February and June 2025 held out (ruler M3): one LightGBM leg of chain A after the LOBT window; with the corrector fitted on January and July | 326,873 | RMSE 287.45; 285.01 | `research/f2/out/m3_corr_f2m3.json` |
| Ruler M3, three-chain world: stage 2 fitted with other 2025 months added (form L; in this gate January, February, June and July minus the held-out month) against stage 2 fitted on January and July (form T), on these rows | 326,873 | ΔMSE -194.0 (P(Δ < 0) 1.0) | `research/f2/out_mac/gate14.json` |

Error by record form on the clean local ruler: the January and July 2025 ruler without the 3 rows that are not G rows (last row of the table) and have labels of 50,000 s or more (LFPG 84,240 and 58,206 s, LIRF 87,002 s; the 7 LIRF G rows above 50,000 s stay, by our count on the ruler rows). Candidate 468, an intermediate chain end of 2026-10-07, has an MSE of 52,743 on it against 82,450 on the full ruler (`research/r8_anatomy/a1_anatomy.txt`, `research/r8_anatomy/a1_tables_val.csv`). The squared error is concentrated in the rare record forms (`docs/DESIGN.md` sections 2.1 and 4).

| Record form | Share of rows | Share of squared error | RMSE |
|---|---|---|---|
| Measured times (the rest) | 81.0% | 35.9% | 152.9 |
| Block time within 6.5 s of the schedule or of an NM time (copies) | 13.8% | 14.5% | 235.7 |
| Label more than 600 s above the take-off time minus the NM actual off-block time, not a copy | 3.7% | 21.1% | 550.5 |
| No NM flight record, outside LIRF's G rows | 1.4% | 7.4% | 520.2 |
| LIRF departures without an NM flight record (G rows) | 0.1% | 21.1% | 3,105.9 |

## Daily-board history

Before the final phase the challenge ranked teams on January and July 2026 by their best daily submission. All 45 daily submissions, with server score, local ruler score, file hash and the purpose of each, are in `results/submissions.csv`; the organizers' scoring files are in `results/server_results/`.

<!--AUTO:DAILYBEST-->
Best daily-board submission (January and July 2026 only; board-tuned history, not the final file): v45 (spirited-tiger_v45.parquet), server RMSE 227.6728, ruler RMSE 281.08, written 2026-10-12 09:03 KST. SHA-256 of the submitted file: `154b90716630bc5c8eddc4773e3bad4bc88acbb9f9730614d358c1209bcf8409`. Manifest: `preds/manifest_m6r28xwincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskapsff12.json`.
<!--/AUTO:DAILYBEST-->

## Where to start

- Reviewers: `docs/REVIEW_GUIDE.md` maps each eligibility line and the likely questions to files, lines and organizer messages.
- `docs/DESIGN.md`: section 0 (the final file, what predicts which rows), sections 1 to 13 (the model), section 14 (months without a board, the re-drawn timestamps of the final input), section 15 (every choice that used the public board).
- `docs/REPRODUCE.md`: the order in which the scripts run.
- `docs/DATA_AND_LICENSES.md`, `docs/EXTERNAL_IDEAS.md`: data sources with their licenses; ideas from public sources and the originality audit.

## What is in this repository

| Path | Content |
|---|---|
| `src/` | Feature construction, models, correctors, evaluation (the ruler), launcher and upload checks |
| `tools/` | Stages applied after the blend (opposite-window rules, record-type and G-model steps, chain averaging) and queue tools |
| `research/` | Scripts that build inputs and legs (ADS-B extraction and matching, stand geometry, seed averaging, TabM, per-airport legs), the final rebuild (`research/f3/`), the ADS-B days of February and June 2026 (`research/r17_a/`), the month-held-out rulers and gates (`research/f2/`), the comparison of the final input with the daily input (`research/f1_final/`), and the analysis scripts and outputs cited in the documents |
| `preds/` | Manifests of the submitted chains: command lines, seeds, feature set and hash, code hashes, ruler results, output hashes. `preds/final/` holds the chain of the final file, `preds/first_final/` that of the first final file (v35's configuration on the final input). No prediction files |
| `notes/` | Job files (command lines) of the legs, the step notes of the two final chains (`notes/final/`, `notes/first_final/`) and small files cited in the documents |
| `results/` | Daily submissions table, organizers' scores, metadata and hashes of derived tables |
| `scripts/verify_code_hashes.py` | Checks the code against the hashes recorded in the manifests |
| `scripts/host/` | Host scheduling scripts of the final rebuild and the download loops of the February and June 2026 ADS-B days, path-cleaned, with a README |
| `RELEASE_MANIFEST.md` | How this tree was made from the working copies, what was left out and why |
| `LICENSE` | GNU General Public License version 3 |

Not included: the organizer data (no license is published for it), prediction files, derived data tables, raw ADS-B data and the EUROCONTROL AIU files (download commands and hashes are in `docs/REPRODUCE.md`).

Code comments are mostly in Korean; the documentation is in English.

## License

The code is released under the GNU General Public License version 3 (`LICENSE`); each source file carries the line `SPDX-License-Identifier: GPL-3.0-only`. Data sources keep their own licenses, listed in `docs/DATA_AND_LICENSES.md`.

## Attribution

- Contains information from adsb.lol (https://www.adsb.lol), made available under the Open Database License (ODbL) 1.0.
- (c) OpenStreetMap contributors, ODbL 1.0 (stand positions; the extracts are in `research/obs2/osm/`).
- Source: EUROCONTROL (Aviation Intelligence Unit, ansperformance.eu) for the airport aggregates used as model inputs.
- Weather: Iowa Environmental Mesonet of Iowa State University (ASOS/METAR, public domain).
- Ideas taken from public sources are listed with their sources in `docs/EXTERNAL_IDEAS.md`.
