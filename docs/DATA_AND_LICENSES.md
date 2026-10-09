# Data sources and licenses

Every dataset that enters a prediction, with its source URL, the license text as read on 2026-10-08 (13:07 to 13:10 KST, plain HTTP fetch of each page, read-only), and what the pipeline takes from it. “Verified” means the sentence was found in the page text on that date. Things we could not verify are marked “not verified”.

The challenge asks that all external datasets are “openly accessible/usable and documented” and that “all additional datasets used are openly available under an open source license” (eligibility page). Section 7 lists the sources whose license is not clearly open.

## 1. Summary

| Source | License as published | Redistributed in this repository | Used in |
|---|---|---|---|
| Organizer data (OpenSky Network / PRC): `training_*.parquet`, `ranking.parquet`, `submitting.parquet`, and for the final phase `final_ranking.parquet`, `final_submitting.parquet` | none stated on the data page (verified) | no | everything |
| IEM ASOS METAR (Iowa Environmental Mesonet) | public domain (verified) | no (fetch script only) | features v2.2 and later |
| adsb.lol `globe_history_2025`, `globe_history_2026` | ODbL 1.0, receiver contributions CC0 1.0 (verified on repository pages and documentation page) | no | ADS-B columns, correctors |
| OpenStreetMap via Overpass API | ODbL 1.0 (verified) | the ten raw extracts, with attribution notice (see 4) | stand positions |
| EUROCONTROL AIU CSV files (ansperformance.eu) | copy with source, non-commercial, no modification (verified); no named open license | no (hashes and fetch commands only) | AIU day shift, corrector columns |
| EUROCONTROL specifications and guides cited in `DESIGN.md` | various (section 6) | no | citations only |

## 2. Organizer data

- Source: OpenSky Network MinIO (S3) bucket `prc-2026-datasets`, endpoint https://s3.opensky-network.org, access with the team's own key (`python -m src.s3io pull`). Description: https://prc-data-challenge-2026.netlify.app/data.html.
- License: the data page contains no license or terms of use (verified: no occurrence of “licen” in the page text). On Discord (#prc-data-competition) a participant asked whether the data may be reused after the challenge; the organizer answered on 2026-09-26 14:21 UTC (message 1553411095434240017): “ask the owner for the license.” and added that the `_flt` columns come from the Network Manager. Nothing further is published.
- Final phase (2026-10-08): `final_ranking.parquet` (SHA-256 `4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f`, 1,341,197 rows) and `final_submitting.parquet` (SHA-256 `b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924`, 670,790 rows) in the same bucket, downloaded with `research/f1_final/scripts/get_final.py` (hashes computed again on 2026-10-09). Same schema as the training files; the January and July rows have the ids of `ranking.parquet` with re-drawn movement times (`DESIGN.md` section 14.6).
- Consequence: no raw rows, no aggregates of the training or ranking files and no per-row derived tables are in this repository. Prediction files are not shipped either; the submitted file is identified by its SHA-256 (`REPRODUCE.md`).

## 3. IEM ASOS METAR

- URL: https://mesonet.agron.iastate.edu/disclaimer.php. Verified text: “The materials found on this website are in the public domain and may be used freely by anyone for any lawful purpose.”
- Fetch: `python -m src.metar_fetch` (`src/metar_fetch.py`) reads the ASOS download service of the same site for the ten airports from 2025-01-01 to 2026-08-01 UTC; no registration. Output `data/metar/<ICAO>.csv`.
- Used: weather columns (temperature, dew point, wind, gust, visibility, weather codes, sky cover, precipitation) taking the last report within 3 hours before the take-off (`src/feats.py metar_feats`). The effect of weather on the ruler is inside the resolution (`notes/levers.md`).
- Attribution: Iowa Environmental Mesonet of Iowa State University.

## 4. adsb.lol and OpenStreetMap (both ODbL 1.0)

### 4.1 adsb.lol globe_history

- URLs: https://github.com/adsblol/globe_history_2025 and https://github.com/adsblol/globe_history_2026 (verified: the repository README states “This database is made available under the Open Database License: http://opendatacommons.org/licenses/odbl/1.0/.” and the GitHub license field reads `ODbL-1.0`); documentation https://www.adsb.lol/docs/open-data/historical/ (verified: “License: ODbL 1.0”, “The data is made available by GitHub releases, a release is available daily for each of the above instances.”).
- License text: https://opendatacommons.org/licenses/odbl/1-0/ (page reachable, verified). Each release also carries `LICENSE-ODbL.txt`; the feeders' contributions are waived to CC0 1.0 according to the release README (“By feeding adsb.lol, you agree to the extent possible under law, to waive all copyright and related or neighboring rights to your data, associating your work with the CC0 license.”). That release-README sentence was copied from the 2026-07-15 release on 2026-10-04 and was not re-opened today (not verified today).
- What is used: 240 days (2025-01 31 days, 2025-02 28, 2025-03 26, 2025-07 31, 2025-08 31, 2025-12 31, 2026-01 31, 2026-07 31) of ground traces at the ten airports, extracted by `research/adsb/adsb_day.py` into per-departure columns (matched take-off, first movement, first ground point, stop counts). The extracted databases (`cache/adsb_v1*.parquet`, `cache/adsb_v2.parquet`, `cache/adsb_geo_v1.parquet`) are derived databases under ODbL and are not published here; they can be rebuilt from the raw releases (`REPRODUCE.md`). The predictions are a Produced Work; this repository states the attribution below.
- Attribution: “Contains information from adsb.lol (https://www.adsb.lol), made available under the Open Database License (ODbL) 1.0.”
- Raw size: the 240 days are about 0.72 to 0.74 TB of release assets (sum of the asset sizes listed in `research/adsb/manifest_preferred.tsv` for these months: 736.7 GB for the eight full months; 26 of the 31 March days are used).
- Which release of a day: the file `PREFERRED_RELEASES.txt` in each repository names the replica (prod, staging, staging-0tmp, prod-0tmp) to use per day. Four days (2025-12-14 to 2025-12-17) list a replica that carries no ground ADS-B (“mlatonly”); we used the prod replica for those four days, and 2026-01-29 from `prod-0`.
- Final phase: the 58 days of February and June 2026 (about 200 GB of release assets) were added for the final input, extracted with `research/r17_a/adsb_day_r17a.py` (the extractor of `research/adsb/adsb_day.py` plus completeness checks). Day-to-release mapping: `research/r17_a/manifest_r17a.tsv` (the download folder's copy of 2026-10-09 11:53 KST; replica switches in `research/r17_a/logs/fixer.log`); 2026-06-06, 06-07, 06-23 and 06-25 come from the `prod-0` replica because the first download of the listed replica was incomplete. 2026-06-11 is short in both replicas and is used as a day without ADS-B. <!--FILL:adsb_2025_fill-->

### 4.2 OpenStreetMap

- URL: https://www.openstreetmap.org/copyright. Verified text: “OpenStreetMap is open data, licensed under the Open Data Commons Open Database License (ODbL) by the OpenStreetMap Foundation (OSMF).” and “You are free to copy, distribute, transmit and adapt our data, as long as you credit OpenStreetMap and its contributors.”
- What is used: aeroway objects (`parking_position`, `apron`, `holding_position`) within 3.5 to 6 km of each airport, fetched from the Overpass API on 2026-10-05 14:09 to 14:13 KST (the responses carry `timestamp_osm_base` 2026-10-05T05:06:45Z). The query text was not saved as a script; the ten raw responses are shipped in `research/obs2/osm/` with an attribution notice so that the stand map can be rebuilt without querying again. Used in `research/obs2/osm_stands.py` and `research/obs2/build_geo.py` to place stands for the ADS-B geometry columns.
- Attribution: “(c) OpenStreetMap contributors, ODbL 1.0”.

## 5. EUROCONTROL AIU airport aggregates

- Page: https://ansperformance.eu/about/disclaimer/ (verified 2026-10-08, same text as the copy saved on 2026-10-05). Verified text: “This data is published by EUROCONTROL for information purposes. It may be copied in whole or in part, provided that EUROCONTROL is mentioned as the source and it is not used for commercial purposes (i.e. for financial gain). The information in this document may not be modified without prior written permission from EUROCONTROL.”
- Files used (downloaded 2026-10-05 22:47 KST from https://www.eurocontrol.int/performance/data/download/csv/<file>; all four carry `Last-Modified: Sat, 19 Sep 2026 06:31 GMT`): `all_pre_departure_delays_2025.csv` (sha16 d742eeb626e5f9f7, 6,503,341 bytes), `all_pre_departure_delays_2026.csv` (5952d20ef5f9c424, 4,344,094), `taxi_out_additional_time_2025.csv` (4aedf19520d3f84b, 92,885), `taxi_out_additional_time_2026.csv` (b72040e0987e6c7d, 54,100). The fetch commands are `curl -o data/ext/aiu/<file> https://www.eurocontrol.int/performance/data/download/csv/<file>`. Verified on 2026-10-08 (about 13:25 KST): all four URLs answer HTTP 200 and the downloaded files have exactly the byte counts and SHA-256 prefixes listed above. The directory page https://www.eurocontrol.int/performance/data/download/csv/ answers HTTP 403 to a plain fetch. The file “Taxi times, Winter 2025-2026” is not used.
- What the pipeline takes: from the daily file, per airport and day, `FLT_DEP_IFR_2` and `DLY_ALL_PRE_2` (total pre-departure delay in units of 60 s). From the monthly file, per airport and month, `TF`, `VALID_FL`, `TOTAL_REF_NB_FL`, `TOTAL_REF_TIME_MIN`, `TOTAL_ADD_TIME_MIN`. Used in `src/feats_aiu.py` (context columns `aiu_*`, file `cache/ctx_v3.parquet`), `src/ctx_corr3.py` (second-stage corrector columns), and `src/aiu_shift.py` (day shift). The numbers are used as numeric inputs to fitted models and to a deterministic shift; the files are not altered and not redistributed here. No claim is made that this satisfies the sentence “may not be modified”.
- Organizer rulings on these files (OpenSky Discord, #prc-data-competition, organizer handle espinielli):
  - 2026-09-17 12:19 UTC (message 1550118982412009516), answering a question whether the non-commercial clause of the EUROCONTROL daily airport files disqualifies them for the prize: “can be considered open data for the challenge” and “if their license applies as open data, ok”.
  - 2026-09-27 15:14 UTC (message 1553786816501514261), answering whether the January and July 2026 monthly average taxi-out time per airport may be used as model input: “I already answered above: yes”.
  - 2026-10-01 14:41 UTC (message 1555228043075915909), answering whether data from a public website without a license should be assumed to be all rights reserved: “If no open license is present it is not open”.
  - 2026-10-08 14:03 UTC (message 1557755225346023585), answering a participant who asked whether a step built on `taxi_out_additional_time_2026.csv` is still OK in the final phase: “(1) no change. still ok”.
- Coverage of the final months (read from the files on 2026-10-09): `all_pre_departure_delays_2026.csv` has all days from 2026-01-01 to 2026-08-31 for the ten airports; `taxi_out_additional_time_2026.csv` has January to July 2026 for the ten airports. February and June 2026 are therefore covered like January and July.
- Risk statement: EUROCONTROL's text names no open license; the 2026-09-17 ruling is conditional (“if their license applies as open data”); the 2026-10-01 statement is general. Whether the AIU files count as open data for the prize is for the organizers to decide; the pipeline uses them as described above, and the two organizer answers are quoted in full.

## 6. Documents cited (not data inputs)

These documents were read to describe how the label is produced (`DESIGN.md` section 2.1). Nothing from them is a model input and none is redistributed here; `DESIGN.md` quotes at most short sentences with section numbers.

| Document | Edition | Terms found in the document |
|---|---|---|
| APDF specification (EUROCONTROL-SPEC-175) | 1.0, 2019-01-15 | Same sentence as in section 5: copy with source, no commercial use, no modification without written permission |
| A-CDM specification (EUROCONTROL-SPEC-198) | 1.0, 2025-01-30 | Same sentence; cover page “TLP:GREEN” |
| AIU additional taxi-out time indicator document | 01.00, 2023-03-16 | No terms sentence in the document; the AIU site terms of section 5 apply |
| NM DPI Implementation Guide | 2.700, 2025-06-30 | “All rights reserved. No part of this publication may be reproduced ... without the prior written permission of EUROCONTROL.” Only short quotations with section numbers are used |
| NM iNM release notes (Waves 2.1 to 2.4) | various | Cover page “Classification: White”; no terms sentence |
| EUROCONTROL Airport Corner A-CDM implementation pages | accessed 2026-10-08 | No terms or update date on the pages |
| ADR Airport CDM FCO Flight Crew Briefing | v1.2, 2012-10 | No terms sentence; at most two sentences are quoted |

The site-wide pages of www.eurocontrol.int were opened on 2026-10-08: “Privacy and website terms of use” (https://www.eurocontrol.int/info/privacy-and-website-terms-use) and “Disclaimers” (https://www.eurocontrol.int/info/disclaimers). Both answer HTTP 200 and contain no sentence about copying, reuse, licensing or intellectual property (no occurrence of “copy”, “licen”, “reproduc” or “intellectual” in the page text; they cover personal data, cookies, map designations and VAT). The reuse conditions that apply to the AIU files are therefore the sentence of the ansperformance.eu disclaimer quoted in section 5. Earlier guesses at other addresses (for example /legal-notice, /terms-and-conditions) return 404.

## 7. Sources whose license is not clearly open

1. Organizer data: no license is published (section 2). The challenge itself requires participants to use it.
2. EUROCONTROL AIU CSV files: “copy with source, non-commercial, no modification”, no named open license (section 5). The GNU GPLv3 under which this repository is published allows commercial use of the code; the data are not part of the repository.
3. The OPDI flight-event data (opdi.aero) were measured and rejected (`DESIGN.md` section 12); they are not used in any prediction.
4. adsb.lol and OpenStreetMap are ODbL: a database built from them and then published must be offered under ODbL. The derived tables are not published here.

## 8. Python packages (licenses as reported by the installed package metadata, 2026-10-08)

LightGBM 4.7.0 MIT; CatBoost 1.2.10 Apache-2.0; XGBoost 3.2.0 Apache-2.0; PyTorch 2.11.0 BSD-3-Clause; pytabkit 1.7.3 Apache-2.0 (RealMLP); tabm 0.0.3 Apache-2.0; rtdl_num_embeddings 0.0.12 Apache-2.0; xrfm 0.4.5 MIT; scikit-learn 1.9.1 BSD-3-Clause; pandas 3.0.6 BSD; NumPy 2.4.6 BSD-3-Clause and others; SciPy 1.17.1 BSD; PyArrow 25.0.1 Apache-2.0; boto3 1.43.107 Apache-2.0; joblib 1.6.0 BSD-3-Clause. None of these packages ships or downloads pre-trained prediction weights in the way the code uses them: RealMLP and TabM are trained from scratch on the 2025 training file.
