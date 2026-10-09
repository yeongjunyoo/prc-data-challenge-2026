# Reproducing the final file and the daily submissions

This file lists, in execution order, what has to run to rebuild the final file `spirited-tiger_final.parquet` (section 0) and the daily submissions it builds on (sections 1 to 9), what each step needs, how long it took on our machines, and where results can differ. The commands are taken from the manifests in `preds/` (field `commands`), from the job files we used, from the drivers in `research/f3/`, and from the docstrings of the scripts. Nothing here was re-run for this document except the checks marked “verified”. Markers `FILL:<name>` stand for values that were not final when this text was written.

<!--AUTO:DAILYBEST-->
Best daily-board submission (January and July 2026 only; history): v39 (spirited-tiger_v39.parquet), server RMSE 228.1667, ruler RMSE 281.35, written 2026-10-09 09:26 KST. SHA-256 of the submitted file: `9a36a77a409b1056badbef40b87b9b772122a91068708159ef872380008284bb`. Manifest: `preds/manifest_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskapwac.json`.
<!--/AUTO:DAILYBEST-->

## 0. The final file

### 0.1 What is rebuilt

The final file is the stack of daily submission v35 (sections 3 to 5; chain commands in section 5) re-run on the final input, with the choices of `DESIGN.md` section 14.5: stage 2 of the corrector runs in v36's configuration (`--scale2 0.75`) in all three chains, and for the February and June rows a second form of stage 2, fitted on four 2025 months, is merged into each chain at the corrector stage, before the tail stages (`DESIGN.md` section 14.7). Only the setting `full` (train on all of 2025, predict the ranking rows) is re-run on the final input. The settings `val` and `fwd`, on whose 2025 rows the correctors, the blend weights and the tail coefficients are fitted, are the outputs of the daily build and are reused unchanged. A rebuild therefore has three roots:

1. Daily root: sections 2 to 5 below, run with the daily-phase `ranking.parquet` and `submitting.parquet`, all three settings. It provides `cache/val_*` and `cache/fwd_*`, the iteration counts `cache/iters_<tag>.json` that the setting `full` of each leg reads, and the corrector model stores `research/*/fitcache/` (written by `src/ctx_corr3.py --fit_cache <dir>`).
2. Final root: the same code and the same 2025 data, with `data/ranking.parquet` = `final_ranking.parquet` and `data/submitting.parquet` = `final_submitting.parquet`. `research/f3/mk_final_root_mac.sh` makes it from the daily root (code, 2025 data, METAR, AIU files and the union of the corrector model stores; the cache starts empty). The `val`, `fwd` and `iters` files of the daily root are copied into its `cache/` as read-only inputs.
3. Month-held-out root: a copy of the daily root in which one seed of each leg family of v35 except TabM is fitted on the 2025 months other than February and June and predicts February and June 2025 (setting `fj`), and the legs are blended with v35's weights. These blends are the extra training rows of the four-month form of stage 2. The context table with the 2025 months 1, 2, 6 and 7 is built there too. Step 8 copies the files it needs into the final root and checks their hashes.

### 0.2 Order of the steps (working directory = the final root, except in step 7)

```
# 1. inputs (own access key, see section 2); SHA-256 4b0692a7...c69f and b3445d80...3924 (DESIGN.md section 0)
python research/f1_final/scripts/get_final.py              # -> data/final_phase/final_ranking.parquet, final_submitting.parquet
#    copy them to data/ranking.parquet and data/submitting.parquet of the final root

# 2. ADS-B days of February and June 2026 (58 days) and the final-input matching of all four months
python research/r17_a/adsb_day_r17a.py --day <day> --manifest research/r17_a/manifest_r17a.tsv --stream ...   # per day, see its docstring
python research/r17_a/prep_final_r17a.py                   # departures of the final input with their own (re-drawn) take-off times
python research/r17_a/final_pack_r17a.py                   # match the 120 test days, merge with the daily tables' 2025 rows
python research/r17_a/final_pack2_r17a.py                  # 2026-06-11 as a day without ADS-B
python research/f3/chk_adsb_pack2.py <daily cache> <pack dir>   # checks the pack, then places cache/adsb_v2 and cache/adsb_v1nd

# 3. feature and corrector tables (unchanged src/ code; 2025 rows checked against the daily tables)
python research/f3/build_feats_final.py                    # cache/all.parquet, cache/feats_v25.parquet
python -m src.feats_adsb feats_v27 --adsb adsb_v2 --mode raw
python research/f3/check_v27.py <daily cache>/feats_v27.parquet
PRC_DAILY_CACHE=<daily cache> bash research/f3/run_ctx_final.sh     # ctx_v1 with frozen stand codes, ctx_v2, ctx_v3, comparisons
python research/f3/patch_ctx2025.py <daily cache>/ctx_v2.parquet ac01294f15b8a618   # then rebuild ctx_v3 with src/feats_aiu.py
python research/f3/geo_final.py <daily cache> <obs_geo parquet> <pack dir>         # cache/geo_v1nd.parquet

# 4. legs, setting full (commands inside final_orch.py are the v35 commands; research/f3/mac_step.sh only waits for free threads and memory)
python research/f3/final_orch.py lgb     # LightGBM: 4 + 4 + 1 legs, g_band x 3, seed averages
python research/f3/final_orch.py cb      # CatBoost: 3 + 3 + 1 legs at 8 threads, seed averages
python research/f3/final_orch.py rm      # RealMLP: 4 + 4 legs on MPS, one at a time, seed averages
python research/f3/final_orch.py tabm    # TabM: input arrays, then the leg on MPS

# 5. chains A, B, C as in v35 (scale2 0.5). The final file uses the blends and the TabM add of this step; the rest built the first final file
python research/f3/final_chain.py tables      # antiwin link, gkey, gkeyx, gkeyxn v2, artmm, arthist tables on the final input
python research/f3/final_chain.py chain A     # blend, TabM add, corrector (3 seeds), average, manifest, 10 tail stages
python research/f3/final_chain.py chain B
python research/f3/final_chain.py chain C
python research/f3/final_chain.py avg3        # tools/chainavg3.py, 1/3 each
python research/f3/final_manifest.py <spec>   # build record of the first final file: input and table hashes, fit-cache counts, leg devices

# 6. the same chains from the corrector on, in v36's configuration (scale2 0.75)
python research/f3/collect_s75.py             # v36's val and fwd corrector outputs from the daily build, as read-only inputs
python research/f3/final_chain.py chain B --from corr --cfg s75      # corrector (3 seeds), average, manifest, 10 tail stages; then C, A
python research/f3/final_chain.py avg3 --cfg s75                     # v36's configuration on all four months: the default of the checks in step 8
python research/f3/final_file.py <its spec> <comparison file> <v36 file> <its float test prediction> <daily ranking.parquet>

# 7. month-held-out root: context tables with the 2025 months 1, 2, 6, 7, then the legs fitted without February and June 2025 (seed 0)
python src/feats_ctx.py --name f2ctx_v1 --months 1,2,6,7
python src/feats_ctx2.py --name f2ctx_v2 --base f2ctx_v1 --base_sha <sha16 of cache/f2ctx_v1.parquet>
python src/feats_aiu.py --name f2ctx_v3 --base f2ctx_v2 --base_sha <sha16 of cache/f2ctx_v2.parquet>
python research/f2/model3_fj.py --setting fj --seed 0 --tag f2skwin --feats feats_v27 <LightGBM arguments of section 4> --threads 4 --mu_sep LIRF --anchor win
python research/f2/model3_fj.py --setting fj --seed 0 --tag f2sk --feats feats_v27 <LightGBM arguments of section 4> --threads 2 --mu_sep LIRF
python research/f2/model3ap_fj.py --setting fj --seed 0 --tag f2skap --feats feats_v27 <LightGBM arguments of section 4> --threads 2 --mu_sep all
python research/f2/g_band_fj.py --setting fj --seed 0 --tag f2skwing --base f2skwin --feats feats_v27        # also f2skg (base f2sk), f2skapg (base f2skap)
python research/f2/cat_mu_fj.py --setting fj --seed 0 --tag f2cbwin --base f2skwin --feats feats_v27 --anchor win --threads 4   # also f2cb (base f2sk, no anchor)
python research/f2/cat_muap_fj.py --setting fj --seed 0 --tag f2cbap --base f2skap --feats feats_v27 --threads 4 --per_ap
python research/f2/rmlp_mu_fj.py --setting fj --seed 0 --tag f2rmwin --base f2skwin --feats feats_v27 --device mps --anchor win --threads 1   # also f2rm (base f2sk)
python research/f2/fj_blends.py               # the three chains' blends with v35's weights: cache/fj_f2wincbrmfw4_s0, fj_f2skcbrmfw4_s0, fj_f2apcbrmfw4_s0

# 8. the row split of stage 2 (scripts/host/f2/kit_final.sh, run from a folder that holds it, its four python files, janjul_ids.parquet and sha16.txt)
bash kit_final.sh install                     # ctx_corr4.py, ctx4_table.py, merge_corr.py, check_split.py, janjul_ids.parquet into research/f2/, hashes checked
KIT_TH=2 bash kit_final.sh prep               # f2ctx_v3 and the three blends of step 7 (hashes checked), the fit-cache seed, the four-month context table f2ctx4f_v3
KIT_TH=2 bash kit_final.sh corr4m A           # four-month stage 2, seeds 0 to 2, scale2 0.75 and its 0.5 twin, each averaged over the seeds; then B, C
KIT_TH=3 bash kit_final.sh merge A            # val and fwd: byte copies of step 6's corrector output; full: by row group (merge_corr.py); then B, C
KIT_TH=3 bash kit_final.sh check              # the corrector-stage checks of research/f2/check_split.py
KIT_TH=3 bash kit_final.sh manifest A Bb      # the merged corrector stage's manifest; then KIT_TH=3 bash kit_final.sh tails A Bb (10 tail stages); then B, C
KIT_TH=3 bash kit_final.sh avg3 Bb            # tools/chainavg3.py, 1/3 each
KIT_TH=3 bash kit_final.sh check finals       # the five checks of DESIGN.md section 14.7 against step 6's file

# 9. the file
python research/f3/final_file.py <spec> spirited-tiger_final.parquet <v36 file> <step 6's float test prediction> <daily ranking.parquet>
```

LightGBM legs retrained with the 2025 ADS-B days filled in (L3, `DESIGN.md` section 14.3): <!--FILL:final_steps_delta-->.

Notes on the steps:

- `final_chain.py` runs the corrector with `--fit_cache research/f3/fitcache_union` and stops if any corrector model has to be fitted, so the corrector models applied to the final input are exactly those of the daily build (the corrector of the setting `full` is the average of models fitted on the 2025 ruler rows). With the stores of your own daily build in that folder the step runs as written. In steps 5 and 6 every corrector run loaded all its models and fitted none (`research/f3/out/final_chain_A.json`, `research/f3/out/final_chain_s75_A.json` and the files of chains B and C, field `checks`).
- `--cfg s75` (step 6) sets `--scale2 0.75` in the corrector, gives the stages the prefix `s75_` and the sandbox queue numbers +1000 (`research/f3/final_chain.py:72`); `--from corr` starts at the corrector, which reads the blends and the TabM add of step 5 in the same root. `collect_s75.py` copies only `val` and `fwd` outputs (`research/f3/out/collect_s75.json`); every `full` output is rebuilt on the final input.
- The model stores of the correctors (`--fit_cache <dir>` of `src/ctx_corr3.py` and `research/f2/ctx_corr4.py`) are keyed by the training rows and columns, the seed, the number of rounds, the thread count and the hashes of `src/adsb_corr.py` and `src/ctx_corr.py` (`src/ctx_corr2.py:86`): a model is loaded only when all of these match, and is otherwise fitted and stored. `kit_final.sh prep` seeds `research/f2/fitcache_4m` with the 704 stage-1 models fitted at 2 threads in the month-held-out root, which is why the four-month corrector runs at `KIT_TH=2`; each run at scale2 0.75 then fitted the 10 four-month stage-2 models and each run at 0.5 loaded all 65, because it is the same fit with another scale (field `fit_caches` of the drivers record in section 0.5). With empty stores the same commands fit every model.
- `research/f2/janjul_ids.parquet` (step 8) holds the `MVT_ID_mvt` values of the departure rows of the daily-phase `ranking.parquet`, in file order (344,841 ids); the merge uses only their set (`research/f2/merge_corr.py:79`). It lists organizer ids and is not in this repository. `kit_final.sh install` checks each file against `sha16.txt` (the list that ran is in `research/f3/out/final_drivers/`), so a rebuilt ids file needs its own line there.
- `kit_final.sh` runs each python step through `research/f3/mac_step.sh`, refuses to overwrite an output, and stops unless each scale2-0.75 run of the four-month corrector fitted 10 models and each 0.5 run none (`scripts/host/f2/kit_final.sh:97`). Our run (`scripts/host/f3/phase4b.sh`) also built the variants Ba (scale2 0.75 on the matched February and June rows: the build we chose first and dropped, `DESIGN.md` section 14.7) and D (the null merge of check 4), and the final check stopped as expected at the bit-equal form of check 3; `scripts/host/f3/phase4c.sh` then wrote the file.
- The scripts that ran steps 4 to 9 on the Mac and that no manifest hashes as a whole are listed, with the SHA-256 of the copy that ran and what it ran, in the drivers record of section 0.5; byte-for-byte copies are in `research/f3/out/final_drivers/`. `research/f3/final_chain.py` in this repository is a later version than the two that ran steps 5 and 6: each version added a configuration (`--cfg s75`, then `--cfg l3s75`) and left the earlier ones as they were.
- The legs were fitted once in an interim root before the February and June ADS-B days were ready and loaded in the final root through `research/f3/fitwrap.py`, a fit cache keyed by the exact training data, parameters, thread count and library version. Training uses 2025 rows only, and the 2025 rows of the interim and final feature tables were checked byte-identical (`research/f3/cmp_v27_interim.py`). A leg refit directly in the final root gave the same predictions within 1e-9 s and the same rounded file values (`research/f3/out/twin_sk_s0.json`: 510 cells of `pred` differ, by at most 9.1e-13 s; no int32 value differs; the wrapper's own tests are in `research/f3/out/wt/`). A cold rebuild does not need the wrapper: run the leg commands of `final_orch.py` directly.
- XGBoost is not refit; `final_chain.py` writes a placeholder with `mu_xgb` = 0 and checks that the blend output is the same with 0 and with 1 (its weight is exactly 0).
- `N_TEST = 344841` (the departures of the daily `ranking.parquet`) is hard-coded in the tail tools `tools/antiwin.py`, `arthist.py`, `artmm.py`, `chainavg3.py`, `gjoint.py`, `gkey.py`, `gkeyx.py`, `gkeyxn.py`, `glong.py` and `gpool.py` (for example `tools/gkey.py:57`) and divides only report numbers, the field `move_mse_test_n` (`move_mse_vs_a` to `move_mse_vs_c` in `chainavg3.py`) and printed check lines; in the final root these numbers are a sum over 670,790 rows divided by 344,841, no prediction depends on them, and the tools are shipped unchanged (except for the SPDX line) so that they keep the hashes in the manifests.
- The host scheduling scripts that ran these steps on our machines (waiting for the input package, locks shared with other jobs, the order of the phases, the copies between the PC and the Mac) and the download loops of the February and June ADS-B days are in `scripts/host/`, path-cleaned, with a README that gives each one's purpose and hashes; the commands they ran are the ones above.

### 0.3 Hardware, times, seeds

| Step | Machine | Time per unit | Source |
|---|---|---|---|
| `build_feats_final.py` | Mac | 130 s, peak 5.2 GB | build log (not shipped) |
| LightGBM leg, setting `full` | Mac, 4 to 6 threads | 3 to 6.5 minutes per leg when loaded through the fit cache (final root); 6.5 to 19 minutes per leg when fitted (interim root) | `research/f3/out/orch_lgb.json`; interim-root logs (not shipped) |
| CatBoost leg, setting `full` | Mac, 8 threads | 52 to 53.5 minutes per leg (seed 0 of `cbwin` was loaded from the fit cache in 1 minute); the per-airport leg 24.5 minutes | `research/f3/out/orch_cb.json` |
| RealMLP leg, setting `full` | Mac, MPS | 16.5 to 17 minutes per leg, run alone (memory footprint up to 33 GB on an 18 GB machine) | `research/f3/out/orch_rm.json`; memory: logs (not shipped) |
| TabM leg, setting `full` | Mac, MPS, 3 threads | 109 minutes (v35's run on the same device: 105.7 minutes); a CUDA run on the PC (RTX 4070 SUPER), kept aside: 13.3 minutes | `research/f3/out/orch_tabm.json`; `research/r13_l/runs/m6r27tmcwin_full_s0.json` (v35); `research/f3/out/tabm_cuda/m6r27tmcwin_full_s0.json` (CUDA) |
| Chains A, B, C, average, file (step 5) | Mac, 3 threads | per chain 16 to 16.5 minutes of step run time: blend and TabM step under 10 s, corrector 1.8 to 2.0 minutes per seed (every model loaded from the fit cache), the ten tail stages about 11 minutes (`gjoint` 4.5 to 4.7 of them); the average of the three chains 6 s; the tables of `final_chain.py tables` about 3 minutes; the file 0.9 s | logs (not shipped) |
| Chains from the corrector in v36's configuration (step 6) | Mac, 3 threads | 20 to 26.5 minutes per chain from start to end, waits included (2026-10-09 23:40 to 2026-10-10 00:48); the average 32 s; the comparison file 1 s | logs (not shipped); times of each driver in the drivers record of section 0.5 |
| Leave-February-and-June-out legs and blends (step 7) | Mac | LightGBM 15 minutes (`f2skwin`, 4 threads), 38.5 and 24 minutes (`f2sk`, `f2skap`, 2 threads); CatBoost 159, 160.5 and 73 minutes (`f2cb`, `f2cbwin`, `f2cbap`, 4 threads, two at a time); RealMLP about 12.5 minutes each (MPS); G bands and the three blends seconds | logs (not shipped) |
| Row split (step 8) | Mac, 2 threads (four-month corrector), 3 threads (the rest) | context table 3 s; four-month corrector 23 to 25 minutes per chain (six runs and two seed averages); merges seconds each; manifests, tail stages and average of one variant about 36 minutes; the checks about a minute | logs (not shipped) |

Machines: PC with Windows 11 Pro, AMD Ryzen 5 7500F, 31.7 GB RAM, NVIDIA GeForce RTX 4070 SUPER (12 GB); Mac with Apple M3 Pro (11 cores), 18 GB RAM. Seeds are those of v35: LightGBM legs 0 to 3 in chains A and B and 0 for the per-airport leg; CatBoost 0 to 2 and 0; RealMLP 0 to 3; TabM 0; correctors 0 to 2. Early stopping uses the training days whose UTC day number is divisible by 10.

### 0.4 Non-determinism of the final rebuild

- LightGBM: two direct runs of the same leg differ by up to 1.65e-10 s in the prediction (floating-point summation order of the threads); the int32 values of the file do not change (`research/f3/out/wt/wrapper_test.json`, field `direct_vs_direct_T1_T1b_max_abs`; `research/f3/out/wt/wrapper_test_t1b.json`, field `pred_int32_rounding_differ`).
- CatBoost: deterministic in our tests (`research/f3/cb_test.py`: a fit repeated on the same data is bit-identical); a refit at 8 threads reproduced the 10-thread CatBoost predictions of v35 exactly (docstring of `research/f3/final_orch.py`). The final legs ran at 8 threads.
- TabM: the same seed and training rows give different models on MPS and CUDA, and in two runs on MPS (early-stopping RMSE 192.279 on MPS for v35, 192.980 on MPS for the first final file, 192.966 on CUDA; the v35 and CUDA run reports are listed in 0.3, the final one is in the leg log, not shipped). The final file uses MPS, as v35 (`research/f3/out/orch_tabm.json`). Between the CUDA run and the MPS run of the final root, `mu_tabm` (the value that chain A's TabM step adds) moves by an MSE of 304.9 over the 670,790 rows and the leg's `pred` by 65.6 (`research/f3/tabm_move.py`, `research/f3/out/tabm_move.json`).
- RealMLP runs on MPS as in v35.
- The other sources of section 8 apply.

### 0.5 Checking that this code produced the final file

1. `python scripts/verify_code_hashes.py` compares every code file named in the CODE lists of the manifests in `preds/` with the SHA-256 recorded when each chain step ran (one SPDX line per file is ignored); `python scripts/verify_code_hashes.py --final` does the same for the manifests of the final file's chain only (`preds/final/` and the final file's own manifest in `preds/`).
2. The final file's SHA-256 is `454dc0e7fb436cb10545da6d2cd9e137e1a4da9585b9cc01c152b2a6d4a87f4c`. Its records: `research/f3/out/final_file_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3rBbamfhgkcxncrdbawgpglgjaavskap.json` (the writer's checks of the file) and `research/f3/out/final_drivers_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3rBbamfhgkcxncrdbawgpglgjaavskapff4.json` (the scripts that ran each step, with their SHA-256). The build record of the first final file, `research/f3/out/final_build_manifest.json`, holds the hashes of the input files and of every table of the final root, which the final file shares, and the checks of that file (670,790 rows, template order, no missing or negative value, int32, schema equal to the template).
3. The file is written from the chain's test prediction with the rule of `src/make_sub.py` (template order, rounding to the nearest integer, clip at 0, int32), implemented again in `research/f3/final_file.py` and in `src/fire_queue.py` (`build_final`).

## 1. Hardware and software used

| Machine | Used for | Specification |
|---|---|---|
| PC | feature tables, XGBoost, TabM, correctors, tail steps, chain averaging | Windows 11 Pro, AMD Ryzen 5 7500F (6 cores, 12 threads), 31.7 GB RAM, NVIDIA GeForce RTX 4070 SUPER (12 GB), Python 3.11.15 |
| Mac | CatBoost legs, RealMLP legs (MPS), LightGBM seeds, several complete chains | Apple M3 Pro (11 cores), 18 GB RAM, macOS, arm64, Python 3.11 |

Python packages: `requirements.txt` (versions installed on the PC, read from the package metadata on 2026-10-08). The Mac used the same versions for the packages that matter (catboost 1.2.10, numpy 2.4.6, pandas 3.0.6, pyarrow 25.0.1, torch 2.11.0 with MPS).

Every step ran under a memory cap (a memory-cap wrapper of the author's own tooling, which is not part of this repository; the cap values below are the observed peaks). The code is plain Python and does not need the cap.

## 2. Data

1. Organizer data. Put your own MinIO access key (obtained from the challenge registration as described on the data page) into two environment variables and run the pull. In the version of `src/s3io.py` in this repository the key is read from the variables `PRC_S3_LOGIN_ID` and `PRC_S3_PASSWORD` (the author's working copy of this file reads the key from a private credential store; the published copy differs from it only in the credential-reading code). No key is stored in this repository.

```
export PRC_S3_LOGIN_ID=<your access key id>
export PRC_S3_PASSWORD=<your secret>
python -m src.s3io pull        # prc-2026-datasets -> data/ (training_*.parquet, ranking.parquet, submitting.parquet; about 315 MB)
```

2. METAR (public domain): `python -m src.metar_fetch` writes `data/metar/<ICAO>.csv` (ten airports, 2025-01-01 to 2026-08-01 UTC).
3. EUROCONTROL AIU files (terms in `DATA_AND_LICENSES.md`): download the four files with the commands below and check the hashes (verified on 2026-10-08: URLs answer and the files are byte-identical to the ones we used).

```
mkdir -p data/ext/aiu
for f in all_pre_departure_delays_2025 all_pre_departure_delays_2026 taxi_out_additional_time_2025 taxi_out_additional_time_2026; do
  curl -o data/ext/aiu/$f.csv https://www.eurocontrol.int/performance/data/download/csv/$f.csv; done
# sha256 prefixes: d742eeb626e5f9f7  5952d20ef5f9c424  4aedf19520d3f84b  b72040e0987e6c7d
```

4. OpenStreetMap stand positions: the ten raw Overpass responses used (fetched 2026-10-05) are in `research/obs2/osm/` (ODbL 1.0, attribution notice in the same folder). The Overpass query text was not saved; re-querying today may return different objects.
5. ADS-B (adsb.lol, ODbL 1.0), 240 days, about 0.72 to 0.74 TB of downloads: January, February, July, August, December 2025 (all days), March 2025 (26 days), January and July 2026. Day-to-release mapping: `research/adsb/manifest_preferred.tsv` (built from the `PREFERRED_RELEASES.txt` file of each adsb.lol repository; four days 2025-12-14 to 2025-12-17 use the prod replica, 2026-01-29 uses `prod-0`; `research/adsb/fix_0129.sh` re-downloads that day). `research/adsb/routeA.sh` downloads days in parallel and extracts each finished day (`research/adsb/adsb_day.py`, single thread, below 2 GB, about 25 CPU-seconds per GB; one day 3 to 4 GB takes about 1.5 minutes). The raw tars (the downloads) are not needed after extraction; the extracted `out/<day>/{points,events}.parquet` plus `meta.json` are about 10 to 25 MB per day.

```
python research/adsb/prep_all.py                                  # PRC departure rows per day -> research/adsb/prc_all.parquet
bash research/adsb/routeA.sh research/adsb/manifest_preferred.tsv research/adsb/days_routeA_all.txt   # download + extract; set ROOT and WROOT (working folder, default ./prc-adsb), PY (python, default python) and optionally MEMCAP_CMD (a memory-cap prefix)
python research/adsb/build_feats.py --name adsb_v1 --out-dir <out dir>     # <out dir> = $WROOT/out; default: environment variable PRC_ADSB_OUT or ./adsb_out; 124 days (2025-01, 2025-07, 2026-01, 2026-07), used by the stand-geometry tables
python research/adsb/build_feats2.py --name adsb_v2 --out-dir <out dir>    # same default; all 240 days, used by feature set v2.7
python research/adsb/gprobe/build_nd.py                                    # V5 tables adsb_v1nd, geo_v1nd (inputs of the correctors)
```

The ADS-B steps are the only ones that need the raw ADS-B days (or their extracted per-day tables). Every later step reads only the tables `adsb_v1`, `adsb_v2`, `adsb_v1nd` and `geo_v1nd` in `cache/`. If you keep the extracted per-day tables (`out/<day>/`), the raw tars can be deleted after extraction.

One constant must be edited by hand: `research/obs2/obs_rows.py` is named in the CODE list of the manifests, so it is shipped unchanged (except for the SPDX line) and still holds the author's extraction folder. In the shipped file, line 19 assigns to `OUT` a Windows path of the author's extraction folder (line 3 is a docstring that repeats it). Set `OUT` to the folder that contains `<day>/events.parquet` as written by `research/adsb/adsb_day.py`. This changes the hash of that one file; `scripts/verify_code_hashes.py` then reports it as different, which is expected, and nothing else changes. All other scripts take such paths from environment variables or from their own location. The paths of the author's machines were replaced in the files that no manifest hashes, and the manifests and the files they hash are left byte for byte as they were at run time.

## 3. Feature tables

```
python src/load.py                                    # cache/all.parquet (training months + ranking)
python src/feats.py feats_v25                         # feature set v2.5
python src/feats_adsb.py feats_v27 --adsb adsb_v2 --mode raw     # v2.7 = v2.5 + 28 ADS-B columns (6 s)
python src/feats_ctx.py --name ctx_v1                 # context columns of the all-row corrector
python src/feats_ctx2.py --name ctx_v2 --base ctx_v1 --base_sha <sha16 printed by the previous step>
python src/feats_aiu.py --name ctx_v3 --base ctx_v2 --base_sha <sha16 of ctx_v2>      # adds EUROCONTROL columns
python research/obs2/obs_rows.py && python research/obs2/coverage.py && python research/obs2/build_geo.py --name adsb_geo_v1
python research/ctx_corr/freeze_geo.py                # cache/geo_v1.parquet (stand geometry, frozen)
```

Checks that were run on the feature tables and may be re-run: `python -m src.mask_test v25` (re-builds the features with the ruler labels masked and compares), `python src/te_coverage.py feats_v25`, `python -m src.composition --feats feats_v25`.

## 4. Legs (regressions of the normal taxi time `mu`)

Each leg is run for the three settings `val`, `fwd`, `full` and for each seed. The job lines we used are in `notes/jobs_*.txt` and `research/r10_b/jobs/`; the commands below are those lines. A job line is a log path, a TAB, and a command; lines starting with `#` are comments. We ran the lines one at a time in file order with a small shell loop (the runner is not part of this repository because it contains checks specific to the author's PC); a job file is run by executing each command with the working directory at the repository root and writing its output to the log path.

```
# LightGBM, window-anchored (chain A), seed k
python src/model3.py --setting {val,fwd,full} --seed k --tag m6r27skwin --feats feats_v27 --es internal --refit --wpow 1 --leaves 255 --mdl 100 --no_direct --clf_drop_prefix m_,ab_ --full_mult 1.0 --threads 6 --mu_sep LIRF --anchor win
python src/g_band.py --setting {val,fwd,full} --seed 0 --tag m6r27skwing --base m6r27skwin --feats feats_v27
# CatBoost, window-anchored
python src/cat_mu.py --setting {val,fwd,full} --seed k --tag m6r27cbwin --base m6r27skwin --feats feats_v27 --anchor win --threads 8
# RealMLP, window-anchored
python src/rmlp_mu.py --setting {val,fwd,full} --seed k --tag m6r27rmwin --base m6r27skwin --feats feats_v27 --device mps --anchor win --threads 1
# chain B: the same three commands without --anchor win, tags m6r27sk (model3 + g_band -> m6r27skg), m6r27cb, m6r27rm
# XGBoost leg (weight 0 in the blend; its output file is still read by blend_fixed)
python src/xgb_mu.py --setting {val,fwd,full} --seed 0 --tag m6r25xg --base m6r25 --feats feats_v25      # requires the base m6r25 (notes/jobs_m6r25.txt)
# TabM (chain A only)
python research/r13_l/prep_x.py --setting {val,fwd,full}                  # arrays on the same rows and features as rmlp_mu
python research/r13_l/tabm_leg.py --setting {val,fwd,full} --seed 0 --tag m6r27tmcwin --base m6r27skwin --anchor win --sched cos --epochs 20
# chain C (family ap): research/r13_f3/model3ap.py --mu_sep all  and  research/r13_f3/cat_muap.py --per_ap   (same arguments as model3 / cat_mu otherwise)
```

Seed counts of the final stack: LightGBM 4, CatBoost 3, RealMLP 4 (seeds 0 to 3, 0 to 2, 0 to 3) for chains A and B; chain C uses one LightGBM and one CatBoost per-airport leg and the RealMLP leg of chain B. Seeds are averaged leg by leg with `research/r10_b/s20_avg_legs.py <leg tag> <new tag> 0,1,2,3`. For `--setting full` the iteration count of the model is read from `cache/iters_<tag>.json`, which only the seed-0 `val` run writes; run seed 0 `val` first. Fixed-weight blend of the legs (and TabM):

```
python src/blend_fixed.py --setting {val,fwd,full} --seed 0 --tag m6r27wincbrmfw4sv434 --base m6r27skwingx4 --learners m6r27cbwinx3:mu_cat,m6r25xg:mu_xgb,m6r27rmwinx4:mu_rmlp --ref_base m6r25sg --ref_tag m6r25sgoof4 --ref_learners m6r25cb:mu_cat,m6r25xg:mu_xgb,m6r25rm:mu_rmlp --folds 10 --check
python research/r13_l/blend_add.py --setting {val,fwd,full} --tag m6r27wincbrmfw4sv434tmc --src m6r27wincbrmfw4sv434 --leg m6r27tmcwin:mu_tabm
```

(The blend reference `m6r25sgoof4` requires the earlier base model `m6r25sg`, `m6r25cb`, `m6r25rm`, `m6r25xg` of feature set v2.5; their job lines are `notes/jobs_m6r25*.txt`.)

## 5. Correctors and tail steps

The stage-2 corrector is run with seeds 0, 1, 2 and averaged; example for the window chain of the seed-base 324 (the 434 base differs only in the tags):

```
python src/ctx_corr3.py --setting {val,fwd,full} --seed {0,1,2} --base <blend tag> --base_seed 0 --tag <blend tag>adsbgeondctxv3x50w --feats feats_v25 --adsb adsb_v1nd --adsb_sha <sha16> --folds 10 --rounds 300 --rounds2 400 --max_pred 3600 --threads 4 --geo geo_v1nd --geo_sha <sha16> --ctx ctx_v3 --ctx_sha <sha16> --scale2 0.5 --fit_cache <dir>
python research/r10_b/cavg.py --setting {val,fwd,full} --seed 0 --tag <blend tag>adsbgeondctxv3x50wc3 --src <blend tag>adsbgeondctxv3x50w --seeds 0,1,2 --base <blend tag> --base_seed 0
```

`--fit_cache <dir>` stores every fitted corrector model in `<dir>` and loads it again when the same fit is asked for (section 0.2, notes). It does not change the output, so the job files and the manifests list the commands without it; our builders added `--fit_cache research/r10_b/fitcache` (for the first seed of the `s0x` bases `research/ctx_corr/fitcache`) when they ran these lines. The final root's corrector store is the union of such stores from our build sandboxes (section 0.1).

Then each tail tool has a `table` sub-command that builds its input table once (`tools/artmm.py table`, `tools/arthist.py table`, `tools/gkey.py table`, `tools/gkeyx.py table`, `tools/gkeyxn.py table --version gkeyxn_v2`, `tools/antiwin.py link`) and a reproduction command per setting. The reproduction commands below are copied from the manifests; `--src <manifest> --src_sha <hash>` name the previous stage's manifest and its hash. Run them leaf first, for `--setting val`, `--setting fwd`, `--setting full` (the manifests list the three variants as `commands.holdout`, `commands.forward`, `commands.test`).

<!--AUTO:CHAIN-->
#### Chain A (window-anchor legs plus TabM), spec `m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja`

Manifest of the chain end: `preds/manifest_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja.json`.

```
# -> m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3
python research/r10_b/cavg.py --setting {val,fwd,full} --seed 0 --tag m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3 --src m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50w --seeds 0,1,2 --base m6r27wincbrmfw4sv434tmc --base_seed 0
# -> m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3am
python tools/artmm.py --setting {val,fwd,full} --base m6r27wincbrmfw4sv434tmc --base_seed 0 --src preds/manifest_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3.json --src_sha d6422b55119e05af --corr m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3 --table artmm_v1 --table_sha 104cb46eb06c3085 --cells artmm_cells_v1 --cells_sha 6d61c60b4dfd7061 --th 0.10 --tag m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3am
# -> m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfh
python tools/arthist.py --setting {val,fwd,full} --base m6r27wincbrmfw4sv434tmc --base_seed 0 --src preds/manifest_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3am.json --src_sha ae7110246dbb869c --corr m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3 --table arthist_v1 --table_sha a02992652cfcdc2a --tag m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfh
# -> m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkc
python tools/gkey.py --setting {val,fwd,full} --src preds/manifest_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfh.json --src_sha a324035106f09304 --leg m6r27skwing --leg_feats feats_v27 --table gkey_v1 --table_sha 4889aa9c680806c8 --variant avg --tag m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkc
# -> m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcx
python tools/gkeyx.py --setting {val,fwd,full} --src preds/manifest_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkc.json --src_sha 9167b0d707aba8e5 --leg m6r27skwing --leg_feats feats_v27 --table gkey_v1 --table_sha 4889aa9c680806c8 --xtable gkeyx_v1 --xtable_sha edfa2b4856dcdd9d --variant avg_x --tag m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcx
# -> m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncr
python tools/gkeyxn.py --setting {val,fwd,full} --src preds/manifest_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcx.json --src_sha 081ffb3d13c7d1c2 --leg m6r27skwing --leg_feats feats_v27 --table gkey_v1 --table_sha 4889aa9c680806c8 --xtable gkeyx_v1 --xtable_sha edfa2b4856dcdd9d --ytable gkeyxn_v2 --ytable_sha 848e82a7da13b1f6 --variant avg_xncr --tag m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncr
# -> m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdb
python src/aiu_shift.py --setting {val,fwd,full} --src preds/manifest_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncr.json --src_sha 00b0b48640c173f0 --lam 0.5 --cmax 60 --ctx ctx_v3 --ctx_sha 0b139ddbd6e70573 --aiu25_sha d742eeb626e5f9f7 --aiu26_sha 5952d20ef5f9c424 --group block --tag m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdb
# -> m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbaw
python tools/antiwin.py --setting {val,fwd,full} --src preds/manifest_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdb.json --src_sha 09446ef566481106 --link antiwin_link_v1 --link_sha ccd99a4c56a25621 --tag m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbaw
# -> m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgp
python tools/gpool.py --setting {val,fwd,full} --src preds/manifest_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbaw.json --src_sha 2e03e76b5357a6dc --tag m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgp
# -> m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpgl
python tools/glong.py --setting {val,fwd,full} --src preds/manifest_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgp.json --src_sha a0a4d594d2c56f4e --tag m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpgl
# -> m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja
python tools/gjoint.py --setting {val,fwd,full} --src preds/manifest_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpgl.json --src_sha 1e7f62dc3bc6f456 --tag m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja --mode a
```

#### Chain B (legs without an anchor), spec `m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja`

Manifest of the chain end: `preds/manifest_m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja.json`.

```
# -> m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3
python research/r10_b/cavg.py --setting {val,fwd,full} --seed 0 --tag m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3 --src m6r27skcbrmfw4sv434adsbgeondctxv3x50w --seeds 0,1,2 --base m6r27skcbrmfw4sv434 --base_seed 0
# -> m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3am
python tools/artmm.py --setting {val,fwd,full} --base m6r27skcbrmfw4sv434 --base_seed 0 --src preds/manifest_m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3.json --src_sha 83f2e985187809fa --corr m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3 --table artmm_v1 --table_sha 104cb46eb06c3085 --cells artmm_cells_v1 --cells_sha 6d61c60b4dfd7061 --th 0.10 --tag m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3am
# -> m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfh
python tools/arthist.py --setting {val,fwd,full} --base m6r27skcbrmfw4sv434 --base_seed 0 --src preds/manifest_m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3am.json --src_sha 8dca2d3dbbe88cd6 --corr m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3 --table arthist_v1 --table_sha a02992652cfcdc2a --tag m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfh
# -> m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkc
python tools/gkey.py --setting {val,fwd,full} --src preds/manifest_m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfh.json --src_sha 05dfdbcdfce45558 --leg m6r27skg --leg_feats feats_v27 --table gkey_v1 --table_sha 4889aa9c680806c8 --variant avg --tag m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkc
# -> m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcx
python tools/gkeyx.py --setting {val,fwd,full} --src preds/manifest_m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkc.json --src_sha ddf7da1a5002909d --leg m6r27skg --leg_feats feats_v27 --table gkey_v1 --table_sha 4889aa9c680806c8 --xtable gkeyx_v1 --xtable_sha edfa2b4856dcdd9d --variant avg_x --tag m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcx
# -> m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncr
python tools/gkeyxn.py --setting {val,fwd,full} --src preds/manifest_m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcx.json --src_sha 1d040cb2f16dbcfe --leg m6r27skg --leg_feats feats_v27 --table gkey_v1 --table_sha 4889aa9c680806c8 --xtable gkeyx_v1 --xtable_sha edfa2b4856dcdd9d --ytable gkeyxn_v2 --ytable_sha 848e82a7da13b1f6 --variant avg_xncr --tag m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncr
# -> m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdb
python src/aiu_shift.py --setting {val,fwd,full} --src preds/manifest_m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncr.json --src_sha b8cd989c44e1721d --lam 0.5 --cmax 60 --ctx ctx_v3 --ctx_sha 0b139ddbd6e70573 --aiu25_sha d742eeb626e5f9f7 --aiu26_sha 5952d20ef5f9c424 --group block --tag m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdb
# -> m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdbaw
python tools/antiwin.py --setting {val,fwd,full} --src preds/manifest_m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdb.json --src_sha cb4dc24040de9ef2 --link antiwin_link_v1 --link_sha ccd99a4c56a25621 --tag m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdbaw
# -> m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdbawgp
python tools/gpool.py --setting {val,fwd,full} --src preds/manifest_m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdbaw.json --src_sha fa8648afd8c7faaa --tag m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdbawgp
# -> m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdbawgpgl
python tools/glong.py --setting {val,fwd,full} --src preds/manifest_m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdbawgp.json --src_sha ce1e227c4bfac731 --tag m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdbawgpgl
# -> m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja
python tools/gjoint.py --setting {val,fwd,full} --src preds/manifest_m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdbawgpgl.json --src_sha 33a71d77386f1095 --tag m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja --mode a
```

#### Chain C (per-airport regressions, family ap), spec `m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja`

Manifest of the chain end: `preds/manifest_m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja.json`.

```
# -> m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3
python research/r10_b/cavg.py --setting {val,fwd,full} --seed 0 --tag m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3 --src m6r27apcbrmfw4sv114adsbgeondctxv3x50w --seeds 0,1,2 --base m6r27apcbrmfw4sv114 --base_seed 0
# -> m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3am
python tools/artmm.py --setting {val,fwd,full} --base m6r27apcbrmfw4sv114 --base_seed 0 --src preds/manifest_m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3.json --src_sha 0ef7f25e4f72ee11 --corr m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3 --table artmm_v1 --table_sha 104cb46eb06c3085 --cells artmm_cells_v1 --cells_sha 6d61c60b4dfd7061 --th 0.10 --tag m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3am
# -> m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfh
python tools/arthist.py --setting {val,fwd,full} --base m6r27apcbrmfw4sv114 --base_seed 0 --src preds/manifest_m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3am.json --src_sha 3473cbf8acc6b5d8 --corr m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3 --table arthist_v1 --table_sha a02992652cfcdc2a --tag m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfh
# -> m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkc
python tools/gkey.py --setting {val,fwd,full} --src preds/manifest_m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfh.json --src_sha e76b69f968c8969d --leg m6r27skapg --leg_feats feats_v27 --table gkey_v1 --table_sha 4889aa9c680806c8 --variant avg --tag m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkc
# -> m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcx
python tools/gkeyx.py --setting {val,fwd,full} --src preds/manifest_m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkc.json --src_sha d8274d50c13786d5 --leg m6r27skapg --leg_feats feats_v27 --table gkey_v1 --table_sha 4889aa9c680806c8 --xtable gkeyx_v1 --xtable_sha edfa2b4856dcdd9d --variant avg_x --tag m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcx
# -> m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncr
python tools/gkeyxn.py --setting {val,fwd,full} --src preds/manifest_m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcx.json --src_sha 00a4175ac35798ca --leg m6r27skapg --leg_feats feats_v27 --table gkey_v1 --table_sha 4889aa9c680806c8 --xtable gkeyx_v1 --xtable_sha edfa2b4856dcdd9d --ytable gkeyxn_v2 --ytable_sha 848e82a7da13b1f6 --variant avg_xncr --tag m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncr
# -> m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdb
python src/aiu_shift.py --setting {val,fwd,full} --src preds/manifest_m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncr.json --src_sha d1342cdf007b2d99 --lam 0.5 --cmax 60 --ctx ctx_v3 --ctx_sha 0b139ddbd6e70573 --aiu25_sha d742eeb626e5f9f7 --aiu26_sha 5952d20ef5f9c424 --group block --tag m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdb
# -> m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbaw
python tools/antiwin.py --setting {val,fwd,full} --src preds/manifest_m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdb.json --src_sha 6e24a115ec3d91a5 --link antiwin_link_v1 --link_sha ccd99a4c56a25621 --tag m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbaw
# -> m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbawgp
python tools/gpool.py --setting {val,fwd,full} --src preds/manifest_m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbaw.json --src_sha 6f82d202ab7e8955 --tag m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbawgp
# -> m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbawgpgl
python tools/glong.py --setting {val,fwd,full} --src preds/manifest_m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbawgp.json --src_sha 9e792945869ff718 --tag m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbawgpgl
# -> m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja
python tools/gjoint.py --setting {val,fwd,full} --src preds/manifest_m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbawgpgl.json --src_sha bbae915e1dfa7ffd --tag m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja --mode a
```

#### Average of the three chains

```
python tools/chainavg3.py --setting {val,fwd,full} --src_a preds/manifest_m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja.json --src_a_sha a8486eedaf99ddc7 --src_b preds/manifest_m6r27skcbrmfw4sv434adsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja.json --src_b_sha 545d01331eb8ed24 --src_c preds/manifest_m6r27apcbrmfw4sv114adsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgja.json --src_c_sha 423cc8ce03bd24d6 --tag m6r27wincbrmfw4sv434tmcadsbgeondctxv3x50wc3amfhgkcxncrdbawgpglgjaavskap
```

<!--/AUTO:CHAIN-->

## 6. The daily submission files

The test prediction of a chain is `preds/test_<spec>.parquet` (`MVT_ID_mvt`, `pred`). The daily submission file is built by `src/make_sub.py`: predictions are put in the order of `submitting.parquet`, rounded to the nearest integer, clipped below at 0 and written as int32 with the schema of the template. Verified on 2026-10-08: for v35, the values of the submitted file equal `clip(rint(pred), 0)` of the prediction file for all 344,841 rows (0 mismatches, mean 1,034.5 s, maximum 104,900 s). Upload: `python -m src.s3io push subs/spirited-tiger_v<N>.parquet` (requires your own team bucket); the score is read back with `python -m src.s3io result spirited-tiger_v<N>.parquet`.

## 7. Time and memory per step (measured)

| Step | Where | Time | Peak memory | Source |
|---|---|---|---|---|
| ADS-B extraction of one day (3 to 4 GB tar) | PC, 1 thread | about 1.5 to 2 minutes; 18.7 s per GB (10 days, 31.6 GB: 591 CPU-s) | 0.84 GB | `research/adsb/` pilot, `DATA_AND_LICENSES.md` |
| ADS-B download | PC, 1 Gbit/s line | 28.7 MB/s with 4 connections; 736 GB take about 7 hours | - | pilot |
| `feats_adsb.py` | PC | 6 s | below 4 GB | `cache/feats_v27.json` |
| LightGBM leg (`model3.py`, window anchor), per seed | Mac, 6 threads | val 1,412 s, fwd 644 s, full 833 s (48 min for the three) | 6.7 / 3.7 / 5.5 GB resident | `logs/m6r27skwin_*_s0.log` |
| CatBoost leg, per seed | Mac, 8 to 10 threads | val 4,790 s (8 threads), fwd 2,423 s, full 2,740 s (2.6 h for the three) | 9.2 / 7.3 / 11.0 GB resident | `logs/m6r27cbwin_*_s0.log` |
| RealMLP leg, per seed | Mac MPS | val 739 s, fwd 380 s, full 949 s (35 min) | 11.6 / 8.7 / 13.0 GB resident (footprint up to 33 GB) | `logs/m6r27rmwin_*_s0.log` |
| XGBoost leg | PC GPU | val 106 s, full 64 s | - | `logs/m6r25xg_*_s0.log` |
| TabM leg | PC GPU | 10 to 12.6 minutes per setting (18 to 22 epochs) | memcap 8 GB (GPU allocation counts) | job logs of the author (not part of this repository) |
| `ctx_corr3.py`, per seed and setting | PC or Mac | about 3 minutes on an idle PC, 50 s on the Mac; 23 minutes on a saturated PC | 2.3 to 2.6 GB | `research/r10_b/` |
| One full tail (corrector average, artmm to gjoint) | Mac | 28 minutes for the 9 corrector runs, the average and 10 tail tools | below 2.1 GB per tool | `research/r10_b/` notes |
| `chainavg3.py` | PC | minutes | 1.7 GB | manifest |

Estimated total for a rebuild from raw data on the same machines, from the numbers above: legs about 32 hours (chain A: 4 LightGBM + 3 CatBoost + 4 RealMLP seeds = 13.4 hours; chain B the same; chain C about 5 hours; TabM 0.6 hours), correctors and tails about 4 hours, ADS-B extraction about 4 CPU-hours, ADS-B download about 7 hours. We ran the legs on two machines in parallel over three days.

## 8. Seeds and sources of non-determinism

- Seeds: legs use `--seed 0..3` as above; correctors use seeds 0 to 2 (averaged); the early-stopping days are the UTC day numbers divisible by 10 (not random); cross-fitting folds are day blocks drawn with seed 0; all fitted combination weights are fixed or fitted without random draws.
- LightGBM results depend on the thread count in the last decimals (`notes/jobs_m6r27skwin.txt`: window-anchor LightGBM ran with 6 threads, the no-anchor LightGBM with 4).
- CatBoost: the final legs ran on arm64 CPU; results differ in the last decimals from x86 runs of the same recipe (`notes/jobs_m6r27cb.txt`). GPU CatBoost runs (earlier candidates) are not bit-reproducible.
- RealMLP on MPS (Mac) and on CUDA (PC) give different numbers for the same seed; the final stack uses the MPS runs.
- Library version of the Mac for the final CatBoost and RealMLP legs: catboost 1.2.10, pytabkit 1.7.3, torch 2.11.0 (a list in `research/r10_b/` job notes).
- `src/feats.py` orders movements with non-stable `np.argsort` on timestamps (lines 137 and 163). Movements with equal timestamps (common at the five airports whose times lie on a one-minute grid) may be ordered differently by different NumPy builds or CPUs, which can change the “previous or next movement” columns of those tied rows. This is by code reading, not measured.
- Expected agreement: a rebuild should land within the seed noise of the local ruler (about 4 to 5 RMSE points for one seed of the whole pipeline; much less after the seed averages used here), not bit-identical. The hash list of the original files is in the manifests (`holdout.sha`, `test.sha`, `forward.sha`).
- The ADS-B and AIU inputs are fixed files; the AIU hashes are given in section 2. ADS-B release assets can be re-published by adsb.lol; the day-to-release mapping is in `research/adsb/manifest_preferred.tsv`.

## 9. Code hashes

The manifests of v35 record the SHA-256 prefix of each source file its chain ran; the final rebuild ran the same 47 files plus the fit-cache wrapper `research/f3/fitwrap.py` (the 47 hashes were checked equal in the final root and in the working repository on 2026-10-09). `scripts/verify_code_hashes.py` repeats the check on this repository.

<!--AUTO:CODELIST-->
| File | sha256 first 16 hex | Recorded in |
|---|---|---|
| `research/adsb/gprobe/build_nd.py` | d89f4aef50faba12 | manifests of v35 (514) and of the final rebuild |
| `research/ctx_corr/freeze_geo.py` | c8a1864d9541565c | manifests of v35 (514) and of the final rebuild |
| `research/obs2/corr_plus.py` | 9d1a4acb4f107595 | manifests of v35 (514) and of the final rebuild |
| `research/obs2/coverage.py` | 08c38f3844f6478a | manifests of v35 (514) and of the final rebuild |
| `research/obs2/obs_rows.py` | e923951e9a1e4496 | manifests of v35 (514) and of the final rebuild |
| `research/obs2/osm_stands.py` | 204c934b9e8fd84c | manifests of v35 (514) and of the final rebuild |
| `research/r10_b/cavg.py` | 70c353c25c737700 | manifests of v35 (514) and of the final rebuild |
| `research/r13_f3/cat_muap.py` | 0ab44b8b9e4b276a | manifests of v35 (514) and of the final rebuild |
| `research/r13_f3/model3ap.py` | 21733839ba861621 | manifests of v35 (514) and of the final rebuild |
| `research/r13_l/blend_add.py` | b69eb9d47a800ceb | manifests of v35 (514) and of the final rebuild |
| `research/r13_l/prep_x.py` | ce1c61190712b424 | manifests of v35 (514) and of the final rebuild |
| `research/r13_l/tabm_leg.py` | dc0f41eade5183d5 | manifests of v35 (514) and of the final rebuild |
| `research/r13_l/xrfm_leg.py` | 77e1c3ef839f9a80 | manifests of v35 (514) and of the final rebuild |
| `src/adsb_corr.py` | 486880c0b3273ded | manifests of v35 (514) and of the final rebuild |
| `src/adsb_geo.py` | 01a6d0667d3e5ac6 | manifests of v35 (514) and of the final rebuild |
| `src/aiu_shift.py` | a2caa0c808f8d59a | manifests of v35 (514) and of the final rebuild |
| `src/blend_fixed.py` | 89fa42fffe266892 | manifests of v35 (514) and of the final rebuild |
| `src/blend_mu.py` | e5827379dba9ec9a | manifests of v35 (514) and of the final rebuild |
| `src/blend_oof.py` | 42fa157949144637 | manifests of v35 (514) and of the final rebuild |
| `src/cat_mu.py` | 05956dbe5a32bfa9 | manifests of v35 (514) and of the final rebuild |
| `src/common.py` | 0fa50c37aeb1b42a | manifests of v35 (514) and of the final rebuild |
| `src/ctx_corr.py` | 41affa9791fd6d27 | manifests of v35 (514) and of the final rebuild |
| `src/ctx_corr2.py` | 8d5eb965268c3832 | manifests of v35 (514) and of the final rebuild |
| `src/ctx_corr3.py` | b5866706124a1292 | manifests of v35 (514) and of the final rebuild |
| `src/feats.py` | f60eeb7e58daa178 | manifests of v35 (514) and of the final rebuild |
| `src/feats_adsb.py` | c50556bf2e08b59d | manifests of v35 (514) and of the final rebuild |
| `src/feats_aiu.py` | 950ac59be3ce12b6 | manifests of v35 (514) and of the final rebuild |
| `src/feats_ctx.py` | 81c4378d6cf286a9 | manifests of v35 (514) and of the final rebuild |
| `src/feats_ctx2.py` | 01fc7ad52545c6ba | manifests of v35 (514) and of the final rebuild |
| `src/g_band.py` | 46969e68a18dd8d0 | manifests of v35 (514) and of the final rebuild |
| `src/load.py` | 576e89ceb820272d | manifests of v35 (514) and of the final rebuild |
| `src/lobt_window.py` | 3f6ef6d290ee429c | manifests of v35 (514) and of the final rebuild |
| `src/model3.py` | 4641d5b5d91abd26 | manifests of v35 (514) and of the final rebuild |
| `src/rmlp_mu.py` | 28fe9c7543664c2b | manifests of v35 (514) and of the final rebuild |
| `src/sub_g.py` | 9610cde7af6ca9d9 | manifests of v35 (514) and of the final rebuild |
| `src/te.py` | 0a78bd1cf5a5771a | manifests of v35 (514) and of the final rebuild |
| `src/xgb_mu.py` | eb1904bd9c883cdd | manifests of v35 (514) and of the final rebuild |
| `tools/antiwin.py` | 117a35c3d8bed472 | manifests of v35 (514) and of the final rebuild |
| `tools/arthist.py` | c35e98cccf682f53 | manifests of v35 (514) and of the final rebuild |
| `tools/artmm.py` | a78fda56cd3f3e15 | manifests of v35 (514) and of the final rebuild |
| `tools/chainavg3.py` | c47f91532c601250 | manifests of v35 (514) and of the final rebuild |
| `tools/gjoint.py` | 2e939bb4cc271ec1 | manifests of v35 (514) and of the final rebuild |
| `tools/gkey.py` | f11f50f785937ece | manifests of v35 (514) and of the final rebuild |
| `tools/gkeyx.py` | 8a1ebee045d8053e | manifests of v35 (514) and of the final rebuild |
| `tools/gkeyxn.py` | 3719a24fe7bcf65d | manifests of v35 (514) and of the final rebuild |
| `tools/glong.py` | 8caa88190fc3bbe1 | manifests of v35 (514) and of the final rebuild |
| `tools/gpool.py` | a3ab8033b847016a | manifests of v35 (514) and of the final rebuild |
| `research/f3/fitwrap.py` | 54114020d2aa3d3c | manifests of the final rebuild (fit cache wrapper the legs ran through) |
<!--/AUTO:CODELIST-->
