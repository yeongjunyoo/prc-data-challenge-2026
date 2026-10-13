# Using the model on your own data

This guide is for readers who want to run the model on other data, rather than check our challenge result (for that, see `docs/REVIEW_GUIDE.md` and `docs/REPRODUCE.md`). It covers the core of the model: the feature table and the LightGBM record-type mixture of `docs/DESIGN.md` section 4. That core already predicts taxi-out times on its own. Section 5 lists what the full stack adds and what each addition needs. Section 3 runs the core end to end on a small synthetic sample in under a minute. The organizer data is not part of this repository, because no license is published for it.

How the parts fit together: `docs/ARCHITECTURE.md`. Why they are built this way: `docs/DESIGN.md`.

## 1. Setup

- Python 3.11 (we used 3.11.15). The core needs four packages and what they pull in; the versions are those of `requirements.txt`:

```
python -m venv .venv               # with Python 3.11
source .venv/bin/activate          # Windows: .venv\Scripts\activate (Git Bash: source .venv/Scripts/activate)
pip install numpy==2.4.6 pandas==3.0.6 pyarrow==25.0.1 lightgbm==4.7.0
```

  If your default Python is another version, uv can make the environment: `uv venv --python 3.11 .venv`, activate it as above and install with `uv pip install` in place of `pip install`. A uv environment has no pip of its own (unless made with `uv venv --seed`), so a plain `pip install` there installs into another Python.
- The other legs and the correctors need the rest of `requirements.txt` (catboost, pytabkit with torch, xgboost, tabm, scipy). Its torch pin, `2.11.0+cu128`, comes from the CUDA 12.8 wheel index, not from PyPI: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128`. With uv, add `--index-strategy unsafe-best-match`; otherwise uv takes torchmetrics, a dependency of pytabkit, only from the CUDA index, which lacks the version pytabkit needs. On a Mac or without CUDA, change the line `torch==2.11.0+cu128` of `requirements.txt` to `torch==2.11.0` and install from PyPI alone (the Mac legs used torch 2.11.0 with MPS).
- Run every command from the repository root. The scripts read and write the folders `data/`, `cache/` and `preds/` below the root; the paths are fixed in `src/load.py`. `.gitignore` keeps `data/`, `cache/`, `.venv/` and prediction files out of commits.
- There is no saved model. Every run fits its models on the labelled files and predicts the movements of the input to predict in the same run. To predict new flights, add them to the input and run again (section 4.1).
- Code comments, the command-line help and the error messages are mostly in Korean. The main modules and functions have English docstrings (section 6). If Korean text prints garbled or as `\u` escapes (for example in Git Bash on Windows), set `PYTHONUTF8=1`.

## 2. Input format

The pipeline reads parquet files in the format of the challenge data, one row per movement:

| File | Content |
|---|---|
| `data/training_<YYYY-MM-DD>_<YYYY-MM-DD>.parquet` | labelled departures and arrivals of one calendar month (the challenge files cut months by UTC time). The month is read from characters 15 and 16 of the file name (`training_2025-03-01_2025-04-01.parquet` is month 3). It becomes the column `src` (1 to 12) of every row |
| `data/ranking.parquet` | the period to predict: its departures with `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` empty, and its arrivals with both filled (`src` 0) |
| `data/submitting.parquet` | the template, two columns: `MVT_ID_mvt` (double) of the departures to predict, in the order the output should have, and `TAXITIME_SEC_mvt` (int32, empty) |

The training files and `data/ranking.parquet` have the 30 columns below, with these types; the code reads them by name, and `examples/make_sample.py` writes them in the order of the challenge files.

| Column | Type | Meaning and use |
|---|---|---|
| `MVT_ID_mvt` | double | movement id, unique across all files |
| `FLIGHT_ID_mvt` | double | NM flight id; empty when the movement has no Network Manager (NM) flight record, which also leaves every `_flt` column empty |
| `FLIGHT_mvt` | string | flight number as the airport recorded it |
| `FLIGHT_RULE_mvt` | string | flight rule (I, V) |
| `ADEP_mvt`, `ADES_mvt` | string | departure and destination airport (ICAO); the airport of a departure is `ADEP_mvt`, that of an arrival `ADES_mvt` |
| `PHASE_mvt` | string | `DEP` or `ARR` |
| `MVT_TIME_UTC_mvt` | timestamp[us, UTC] | take-off (departure) or landing (arrival) |
| `BLOCK_TIME_UTC_mvt` | timestamp[us, UTC] | off-block (departure) or in-block (arrival); empty for the departures to predict |
| `SCHED_TIME_UTC_mvt` | timestamp[us, UTC] | scheduled time; needed, the mixture model is built on take-off minus schedule |
| `AIRCRAFT_TYPE_mvt`, `RUNWAY_mvt`, `STAND_mvt` | string | aircraft type, runway, stand |
| `TAXITIME_SEC_mvt` | int32 | the label: `MVT_TIME - BLOCK_TIME` in seconds for departures; for arrivals the taxi-in time, `BLOCK_TIME - MVT_TIME`; empty for the departures to predict |
| `LOBT_flt`, `IOBT_flt` | timestamp[us, UTC] | planned off-block times of the NM flight plan (equal for 99% of the departures with an NM record, `docs/DESIGN.md` section 2.1) |
| `EOBT_1_flt`, `ARVT_1_flt` | timestamp[us, UTC] | estimated off-block and arrival times of the filed flight plan |
| `AOBT_3_flt`, `ARVT_3_flt` | timestamp[us, UTC] | off-block and arrival times of the flight as flown, as the NM derived them (its off-block time is the take-off minus a taxi time, `docs/DESIGN.md` section 2.1) |
| `CALLSIGN_flt`, `ADEP_flt`, `ADES_flt`, `ADES_FILED_flt` | string | call sign and airports of the NM record |
| `MARKET_SEGMENT_flt`, `FLIGHT_RULE_flt`, `FLIGHT_TYPE_flt`, `AIRCRAFT_TYPE_flt`, `WK_TBL_CAT_flt`, `AIRCRAFT_OPERATOR_flt` | string | NM flight attributes (market segment, rule, type, aircraft type, wake category, operator) |

`docs/DESIGN.md` section 2.1 explains where these columns come from and why some recorded off-block times are copies of the schedule; that is the reason for the mixture model. In the challenge data a departure has an NM record only if its off-block time is within 3,606 s of LOBT (the NM matcher drops the others); the sample follows the same rule. Arrivals are not predicted, but the features use them (stand occupancy, taxi-in times, traffic). METAR weather is optional: `python -m src.metar_fetch` writes `data/metar/<ICAO>.csv` for the ten airports of the challenge, and airports without a file get empty weather columns.

Every column must be present. Most columns can hold empty values, but these must be filled: `MVT_ID_mvt`, `PHASE_mvt` and `MVT_TIME_UTC_mvt` on every row, `SCHED_TIME_UTC_mvt` and `ADEP_mvt` on every departure, `ADES_mvt` on every arrival, and `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` on every departure of the training files. An empty take-off or schedule time stops `src/feats.py` with `IntCastingNaNError`.

Timestamps must be stored in microseconds, as in the challenge files (`timestamp[us, UTC]`), because `src/load.py` reads every timestamp as microseconds. `src/load.py` and `src/feats.py` read a file in milliseconds without an error, as dates in January 1970, and so a file in seconds (parquet stores seconds as milliseconds); the model then works on those dates or stops (on the sample in milliseconds, `--setting val` stops with a `LightGBMError`, because every take-off falls on one day, an early-stopping day). A file in nanoseconds stops `src/feats.py` with a `NotImplementedError` that ends `(year ... is out of range)`. A time zone other than UTC is read correctly, since parquet stores the instant; times without a time zone are taken as UTC. In pandas, before writing:

- a datetime column of any unit or time zone: `df[c] = pd.to_datetime(df[c], utc=True).astype("datetime64[us, UTC]")`;
- epoch seconds stored as numbers: `df[c] = pd.to_datetime(df[c], unit="s", utc=True).astype("datetime64[us, UTC]")`. Without `unit="s"`, pandas reads the numbers as nanoseconds, which gives dates in 1970;
- local times without a time zone: `df[c] = df[c].dt.tz_localize("Europe/Amsterdam").dt.tz_convert("UTC").astype("datetime64[us, UTC]")` with the zone of the airport where the movement takes place (`ADEP_mvt` for departures, `ADES_mvt` for arrivals); the first recipe would read them as UTC. At the changes of daylight saving time, `tz_localize` needs its `ambiguous` and `nonexistent` arguments (for example `ambiguous="NaT", nonexistent="shift_forward"`; the times made empty this way are then missing values).

`python examples/check_input.py` checks the files in `data/` against this section (section 4).

## 3. Worked example on a synthetic sample

`examples/make_sample.py` invents movements with the 30 columns and types above, for two of the ten airports (EHAM and LIRF), for every day of 2025 (twelve training files) and for January 2026 (the input to predict and its template). The values follow a few made-up rules (runways with different taxi times, remote stands, congestion, delays, schedule copies, next-day offsets, the NM matcher's window) and describe no real airport. The draws are seeded, so every run writes the same files. The script refuses to write if `data/` already holds files of these names, so that the sample never mixes with real data.

```
python examples/make_sample.py
python src/load.py
python src/feats.py feats_demo
python src/model3.py --setting val --seed 0 --tag demo --feats feats_demo --refit --no_direct --threads 2
python src/model3.py --setting full --seed 0 --tag demo --feats feats_demo --refit --no_direct --threads 2
python examples/write_predictions.py cache/full_demo_s0.parquet predictions.parquet
```

What each step does:

1. `examples/make_sample.py` writes 12 training files, `data/ranking.parquet` (2,468 movements, 1,170 departures without labels) and `data/submitting.parquet`; together 31,531 movements, 2.7 MB.
2. `src/load.py` reads them into one table, `cache/all.parquet` (timestamps as epoch seconds, `src` = month, label `y`).
3. `src/feats.py feats_demo` builds the feature table `cache/feats_demo.parquet`: 15,817 departures, 102 columns.
4. `src/model3.py --setting val` fits the model on the training months other than January and July and scores January and July 2025 (the ruler, `docs/DESIGN.md` section 3). It writes `cache/val_demo_s0.parquet` and the iteration counts `cache/iters_demo.json`.
5. `src/model3.py --setting full` fits on all twelve months with those iteration counts times 1.1 times `--full_mult` (default 1.15; section 4.3) and predicts the departures of `data/ranking.parquet` into `cache/full_demo_s0.parquet`.
6. `examples/write_predictions.py` writes `predictions.parquet` in the order of the template: `MVT_ID_mvt` and `TAXITIME_SEC_mvt`, rounded to whole seconds, clipped at 0, int32. This is the rule of the challenge files (`docs/REPRODUCE.md` section 6).

`--threads 2` sets the threads of the main LightGBM models; the small multiclass model of the G rows (`src/sub_g.py`) always uses 4.

On our PC (Windows 11, Python 3.11.15, a new environment with only the four packages above) the six steps took about 20 seconds. Step 4 printed, among other lines:

```
[demo val mixture+G (clipped)] RMSE 2361.68  (n=2531)
  by y: -1000000000~1800: n=2457 mse_share=0.012 | 1800~3600: n=59 mse_share=0.002 | 3600~7200: n=10 mse_share=0.003 | 7200~80000: n=2 mse_share=0.001 | 80000~1000000000000: n=3 mse_share=0.981
  RMSE on y<3600 rows: 283.43 | MSE share of y>=3600: 0.986
  without 3 no-signal rows: mixture+G 326.03
```

and step 6:

```
wrote predictions.parquet: 1170 departures, mean 1014.8 s, median 958 s, max 8734 s
```

Other machines can differ in the last digits. How to read the ruler lines: three rows with labels above 80,000 s carry 98.1% of the squared error. They are next-day offsets: the airport recorded the right off-block clock time on the scheduled date, so the label is about a day too long. Such rows exist in the challenge data too (`docs/DESIGN.md` section 4): there, three rows with labels above 50,000 s carry more than a third of the squared error of a candidate on the January and July 2025 ruler (`README.md`, the paragraph before the record-form table). On the 2,516 rows below 3,600 s the RMSE is 283 s. For comparison, each airport's median label of the fitting months gives 332 s on the same rows:

```
python -c "import pandas as pd, numpy as np; F = pd.read_parquet('cache/feats_demo.parquet', columns=['src', 'y', 'ap']); fit = F[~F.src.isin([0, 1, 7])]; ev = F[F.src.isin([1, 7]) & (F.y < 3600)]; b = ev.ap.map(fit.groupby('ap', observed=True).y.median()).astype(float); print(round(float(np.sqrt(((b - ev.y) ** 2).mean())), 1))"
```

The numbers only show that the pipeline runs and learns the structure that the sample was drawn from. They say nothing about real airports.

To clean up, delete `data/`, `cache/`, `predictions.parquet` and the early-stopping lists `preds/es_valid_ids_demo*.parquet` (the pattern also covers the runs of section 3.1). The files that ship in `preds/` are manifests (`.json`), which stay.

### 3.1 The options of the submitted legs

The example runs `src/model3.py` with its defaults. The LightGBM legs of the final stack used other options (`docs/REPRODUCE.md` section 4; the job lines are in `notes/jobs_m6r27skwin.txt` and the files beside it). On the sample, the leg of chain B reads:

```
python src/model3.py --setting val --seed 0 --tag demob --feats feats_demo --es internal --refit --wpow 1 --leaves 255 --mdl 100 --no_direct --clf_drop_prefix m_,ab_ --full_mult 1.0 --threads 2 --mu_sep LIRF
```

It printed `RMSE 2409.40  (n=2531)`, `RMSE on y<3600 rows: 260.07` and `without 3 no-signal rows: mixture+G 296.97`. Run it with `--setting full` as well to predict. The leg of chain A adds `--anchor win`. On the sample that anchor equals the airport's median label on every ruler row, because the LOBT window never moves it, so the numbers change only in the second decimal (2409.39, 260.04, 296.91, with `--tag demoa`). What the options do:

| Option | Default | Meaning |
|---|---|---|
| `--leaves`, `--mdl` | 127, 200 | LightGBM `num_leaves` and `min_data_in_leaf` |
| `--wpow` | 0 (no weights) | weight the rows of the copy classifier by (clip(\|d_sch - 1000\|, 300, `--wcap`) / 1000) to this power, so that the rows where a wrong copy decision costs most count most |
| `--clf_drop_prefix` | none | column prefixes that the copy classifier does not see (`m_` METAR, `ab_` ADS-B) |
| `--full_mult` | 1.15 | `full` trains the iterations of the seed-0 `val` run times 1.1 (with `--refit`) times this factor; the submitted legs used 1.0 |
| `--mu_sep` | none | airports with a regression of their own (LIRF) |
| `--anchor` | none | regress the label minus an anchor: `aobt` (take-off minus AOBT_3) or `win` (the airport median moved into the LOBT window) |
| `--es`, `--refit` | internal, off | early stopping on the training days whose UTC day number is divisible by 10; refit on all fitting rows with the iterations times 1.1 |
| `--no_direct` | off | skip the direct regression of the label (column `direct`) |

The module docstring of `src/model3.py` lists the remaining options.

## 4. Running it on your own data

Put your files in `data/` with the names and columns of section 2 and check them first:

```
python examples/check_input.py
```

It reads the files one at a time and changes nothing. It prints a `PROBLEM` line for each error of format or value that it finds: file names whose month cannot be read or does not match the movements, missing columns or wrong types, timestamps not in microseconds, empty values where section 2 needs them, timestamps outside the years 2000 to 2100 (the sign of a wrong unit), labels that differ from take-off minus off-block, repeated movement ids, labels on the departures to predict, and a template that does not list exactly those departures. It prints a `NOTE` line for each of these points of sections 2, 4.2 and 4.3: timestamps without a time zone, number columns or template columns of another type, a file without arrivals or with more than 1% of its arrivals without block time or taxi-in time, airports without a time zone in `src/feats.py`, and each setting of `src/model3.py` that cannot run on the files as the code stands (no departures to score, or no G rows among the rows it fits on or scores). It exits with 1 if there is a problem. It checks the format and these rules, not whether the values are right: a time of the right type and year can still be wrong. On the sample it prints `checked 14 files: 0 problems, 0 notes`, and so it does on the challenge files of both phases. Then run steps 2 to 6. `src/load.py` and `src/feats.py` rebuild their output whenever they run as scripts, while `src/model3.py` reads the feature table `cache/<name>.parquet` as it finds it, so run steps 2 and 3 again after every change of the data.

### 4.1 New flights at the same airports

Put the movements to predict in `data/ranking.parquet` (departures without block time and label, arrivals with them) and list the departures in `data/submitting.parquet`. Then run steps 2 and 3 (the tables), step 5 (the `full` setting) and step 6; step 4 is needed only for new iteration counts. Every window feature is computed from the movements of the same UTC month and airport, whichever file holds them. So the arrivals and departures around a flight must be in one of the files, and each movement in only one: movement ids are unique across the files, and labelled movements must not be copied into `data/ranking.parquet` for context. The challenge always predicted whole months.

### 4.2 Other airports

The code was written for the ten airports of the challenge. The files that a manifest hashes (`docs/REPRODUCE.md` section 9) hold the code that ran; if you edit one, `scripts/verify_code_hashes.py` reports it as different, which is expected. What to change for an airport outside the ten:

- `src/feats.py`, the dictionary `TZ`: add the airport's time zone. Without it, its rows lose every column that depends on the local date or clock: the local hour, weekday and date change, the per-day columns (`c_day`, `day_daobt`, `day_dsch`, `day_daobt_rel`), and the METAR columns (no file is read for it).
- `src/model3.py` fits a separate model for the G rows, which are LIRF departures without an NM flight record. It needs such rows among both the rows it fits on and the rows it scores or predicts. Otherwise it stops in the G model (`src/sub_g.py`) with a `ValueError`: `need at least one array to concatenate`, or `Input data must be 2 dimensional and non empty` when only the fitted rows lack them. `examples/check_input.py` names the settings that would stop. Workaround if your data has no such rows: replace the line `P = g_probs(Gf, Ge)` in `main` by `P = g_probs(Gf, Ge) if len(Gf) and len(Ge) else np.tile([1.0, 0.0, 0.0], (len(Ge), 1))`. The G rows, if any, then take the plain regression value. Python then prints two `RuntimeWarning`s about the mean of an empty array and `mean pa nan ps nan`, which are harmless. We tested this on the sample without LIRF. The code that produced the submissions does not have this line.
- `--mu_sep LIRF` (a separate regression for LIRF) stops if no LIRF row exists; leave the option out.
- METAR: `src/metar_fetch.py` downloads the ten stations of 2025-01-01 to 2026-08-01; edit `STATIONS` and the dates.
- Data without NM records (for example outside the EUROCONTROL area) leaves every `_flt` column and `FLIGHT_ID_mvt` empty. The code runs, but the model loses its strongest inputs, and every LIRF departure becomes a G row. On the sample with all NM columns emptied, the RMSE of the rows below 3,600 s rose from 283 s to 1,376 s. We have not measured this on real data.
- The correctors and tail steps hold their own airport lists: `APTS` in `src/adsb_corr.py` and `src/feats_aiu.py`, `APS` in `src/aiu_shift.py`, `APTS` and `R2_APTS` in `tools/antiwin.py`. The ADS-B extraction holds the airports' positions in `research/adsb/adsb_day.py`. The stand positions are OpenStreetMap extracts of the ten airports (`research/obs2/osm/`). The G steps in `tools/` are about LIRF only. Section 5 has more.

### 4.3 Other periods

- The month of a training row comes from its file name and is all that `src` keeps, so twelve monthly files of one year is the layout the code expects. Two Januaries would both be month 1.
- The ruler is fixed to January and July: `VAL_MONTHS` in `src/common.py`, `RULER_MONTHS` in `src/ruler.py`, and the settings in `src/model3.py` (val holds out months 1 and 7, fwd fits on months 1 to 6 and scores month 7). Without January and July, `--setting val` has no rows to score and stops with the first error of section 4.2; without July alone it scores January, while `--setting fwd`, which scores July, stops. `--setting full` still runs if you write the iteration counts that it expects into `cache/iters_<tag>.json`, for example `{"clf": 100, "reg": 600, "refit_factor": 1.1}`, with `"reg2"` added for `--mu_sep`, `"eclf"` for `--eobt_comp` and `"direct"` without `--no_direct`. full multiplies them by `refit_factor` (1 if absent) and by `--full_mult` (default 1.15).
- `src/blend_oof.py` and `src/blend_fixed.py` read the ruler days from the file names of January and July 2025.

### 4.4 Size and time

On the challenge data (2.1 million labelled departures and about as many arrivals) one LightGBM leg took 24 minutes for `val`, 11 for `fwd` and 14 for `full` on 6 threads of an Apple M3 Pro, with up to 6.7 GB of memory (`docs/REPRODUCE.md` section 7). The other learners and the correctors cost more (same section). `--threads` sets the threads of the main models; the small multiclass model of the G rows (`src/sub_g.py`) always uses 4.

## 5. From the core model to the full stack

The final file averages three chains, each a blend of several legs followed by a corrector in two stages and ten tail steps (`docs/ARCHITECTURE.md`). The commands and their order are in `docs/REPRODUCE.md` sections 3 to 5. What each addition needs beyond the core:

| Addition | Scripts | Extra inputs and packages |
|---|---|---|
| Other legs, each mixed with the LightGBM leg's classifier | `src/cat_mu.py`, `src/rmlp_mu.py`, `src/xgb_mu.py`, `research/r13_l/tabm_leg.py`, per-airport legs `research/r13_f3/model3ap.py` and `research/r13_f3/cat_muap.py` | catboost, pytabkit and torch, xgboost, tabm |
| Seed averages and the fixed blend | `research/r10_b/s20_avg_legs.py`, `src/blend_fixed.py` (weights of a reference blend fitted by `src/blend_oof.py` on the ruler days) | the legs' outputs |
| ADS-B features (feature set 2.7) | `research/adsb/` (download and extraction of adsb.lol days, matching to departures), `src/feats_adsb.py` | raw ADS-B days (0.7 TB for the 240 days of the daily phase), `docs/REPRODUCE.md` section 2 |
| Correctors (stage 1 on ADS-B ground traces, stage 2 on context) | `src/ctx_corr3.py` (with `src/ctx_corr2.py`, `src/ctx_corr.py` and `src/adsb_corr.py` imported), `src/feats_ctx.py`, `src/feats_ctx2.py`, `src/feats_aiu.py`, `research/obs2/` and `research/ctx_corr/freeze_geo.py` (stand geometry) | the ADS-B tables, OpenStreetMap stands, EUROCONTROL airport aggregates (`docs/DATA_AND_LICENSES.md`) |
| Tail steps for record types, G rows and day totals | `tools/artmm.py`, `tools/arthist.py`, `tools/gkey.py`, `tools/gkeyx.py`, `tools/gkeyxn.py`, `src/aiu_shift.py`, `tools/antiwin.py`, `tools/gpool.py`, `tools/glong.py`, `tools/gjoint.py` | the chain's manifests; tuned to the record-keeping of these ten airports (`N_TEST = 344841` in the tools only scales printed and recorded report numbers, `docs/REPRODUCE.md` section 0.2) |
| Average of the three chains | `tools/chainavg3.py` | three chain ends |

## 6. Which parts to reuse

| Part | Files | Reuse |
|---|---|---|
| Reading the files, feature table, target encodings, ruler helpers | `src/load.py`, `src/feats.py`, `src/te.py`, `src/common.py`, `src/ruler.py` | as is for data in the format of section 2; `TZ` for other airports |
| Record-type mixture model and its G model | `src/model3.py`, `src/sub_g.py` | as is; for data without G rows, the line of section 4.2 |
| Other legs and blends | `src/cat_mu.py`, `src/rmlp_mu.py`, `src/xgb_mu.py`, `src/blend_mu.py`, `src/blend_oof.py`, `src/blend_fixed.py` | the legs as is. The blends read the ruler days from the 2025 file names (section 4.3), and on `val` and `fwd` they read the airport of each row from `cache/feats_v25.parquet`, a name that `src/feats.py` refuses to build: build under another name and rename the file, as `docs/REPRODUCE.md` section 3 does. `src/blend_oof.py` fits the weights; `src/blend_fixed.py` applies the weights of one reference blend to another leg, as the final stack did |
| LOBT window | `src/lobt_window.py` | for data with NM records. The window of 3,606 s holds in the 2025 data of the ten airports (`docs/DESIGN.md` section 8.1); check it on yours after a `full` run with `python src/lobt_window.py check --feats <feature table> --tag <tag>`, which prints (in Korean) the training rows outside the window and the predicted rows outside it. On the sample: `--feats feats_demo --tag demo`, 0 of 14,138 rows and 0 of 1,130 rows |
| ADS-B extraction and matching | `research/adsb/` | any adsb.lol day; airport positions in `research/adsb/adsb_day.py` |
| Correctors | `src/adsb_corr.py`, `src/adsb_geo.py`, `src/ctx_corr.py`, `src/ctx_corr2.py`, `src/ctx_corr3.py`, `src/feats_ctx.py`, `src/feats_ctx2.py`, `src/feats_aiu.py` | with the airport lists of section 4.2 |
| Tail steps | `tools/` | specific to the record-keeping of the challenge data |
| Launcher, upload, gates, ledger, batch registration, diagnostics | `src/fire_queue.py`, `src/shot.py`, `src/s3io.py`, `src/make_sub.py`, `src/batches.py`, `src/register_batch.py`, `src/manifest.py`, `src/diag_constants.py`, `src/diag_solve.py`, `src/tail_probe*.py` and other probes | specific to the challenge (submission rules, team bucket, our working records) |
| Final-phase rebuild | `research/f1_final/`, `research/f2/`, `research/f3/`, `research/r17_a/`, `scripts/host/` | specific to the challenge's final phase |

English docstrings: the modules and main functions of the files of `src/` in the rows above that are not specific to the challenge, and `src/metar_fetch.py`; the module docstrings of the per-airport legs (`research/r13_f3/model3ap.py`, `research/r13_f3/cat_muap.py`) and of the tail steps in `tools/`. The scripts of `research/adsb/`, `research/obs2/` and `research/r10_b/` that these rows name had English module docstrings already. Where one of the modules or functions above had a docstring written during the challenge, it is Korean and follows the English one (`RELEASE_MANIFEST.md` section 3, `scripts/release_docstrings.json`). Other docstrings, the command-line help and the error messages are mostly in Korean.
