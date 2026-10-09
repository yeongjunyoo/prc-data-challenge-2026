# PRC Data Challenge 2026: taxi-out time prediction (team spirited-tiger)

Solution of team spirited-tiger (Yeongjun Yoo, Hongik University, Seoul) for the EUROCONTROL PRC Data Challenge 2026: predict the taxi-out time in seconds of departures from ten European airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM), trained on the 2025 movements of the same airports. The metric is the RMSE in seconds.

## Final submission

The final phase asks for one file that predicts January, February, June and July 2026 from `final_ranking.parquet`; the organizers rank it on January and July, on February and June, and on the four months together, and review code and documents.

| Item | Value |
|---|---|
| File | `spirited-tiger_final.parquet`, 670,790 rows, in the order of `final_submitting.parquet` |
| SHA-256 | <!--FILL:final_file_sha256--> |
| Built by | the three-chain stack of `docs/DESIGN.md`, re-run on `final_ranking.parquet` (SHA-256 `4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f`) with the choices of `docs/DESIGN.md` section 14.5 |
| Training labels | the twelve 2025 training files only |
| Months without a public score (February, June) | every choice decided on February and June 2025 held out as unseen months, by rules written before their numbers, with two exceptions: the row split of stage 2 of the corrector was decided after its gate numbers, and the scale2 of the matched February and June rows follows the literal reading of an ambiguous registered rule that we first read the other way after seeing its numbers (`docs/DESIGN.md` sections 14.3 and 14.7) |

How the final file is rebuilt, with commands, machines, times, seeds and the limits of bit-for-bit reproduction: `docs/REPRODUCE.md` section 0.

## Daily-board history

Before the final phase the challenge ranked teams on January and July 2026 by their best daily submission. All 40 daily submissions, with server score, local ruler score, file hash and the purpose of each, are in `results/submissions.csv`; the organizers' scoring files are in `results/server_results/`.

<!--AUTO:DAILYBEST-->
Best daily-board submission (January and July 2026 only; history): v39 (spirited-tiger_v39.parquet), server RMSE 228.1667, ruler RMSE 281.35, written 2026-10-09 09:26 KST. SHA-256 of the submitted file: `9a36a77a409b1056badbef40b87b9b772122a91068708159ef872380008284bb`. Manifest: `preds/manifest_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskapwac.json`.
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
| `preds/` | Manifests of the submitted chains: command lines, seeds, feature set and hash, code hashes, ruler results, output hashes. No prediction files |
| `notes/` | Job files (command lines) of the legs and small files cited in the documents |
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
