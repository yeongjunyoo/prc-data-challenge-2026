# OPDI flight events as an off-block source (PRC Data Challenge 2026)

Design research only. Nothing was submitted and nothing outside research/offblock and data_ext/opdi was written.

## Verdict

OPDI cannot replace the prediction. On the local ruler (holdout of 344,419 departures, 2025-01 and 2025-07) the current model scores 317.744. Replacing covered rows with the bias-corrected OPDI estimate makes it worse (320.251 with stand exits only, 337.801 with every proxy). The best blend found, a cross-month linear stack on covered rows, gives 317.229, a gain of 0.52. OPDI ground coverage exists at few airports: LSZH in every month, EDDF only in 2026, LEBL and EDDM partly in 2025. On the rows it covers, the model is already better (model RMSE 171.567 against 271.004 for the best proxy). The OPDI median error is small, but the error tail is heavy.

## Sources and license

Listings: https://www.opdi.aero/flight-event-data and https://www.opdi.aero/flight-list-data. Site coverage statement: January 01, 2022 - July 31, 2026; v0.0.2; refreshed July 2026. Flight events come as 10-day files, each covering up to but not including the second date. The flight list comes as monthly files.

| File | MB | Last-Modified |
|---|---:|---|
| flight_events_20241226_20250105.parquet | 221.3 | Mon, 13 Apr 2026 23:00:21 GMT |
| flight_events_20250105_20250115.parquet | 244.6 | Mon, 13 Apr 2026 23:06:52 GMT |
| flight_events_20250115_20250125.parquet | 224.9 | Mon, 13 Apr 2026 23:13:13 GMT |
| flight_events_20250125_20250204.parquet | 204.1 | Mon, 13 Apr 2026 23:19:25 GMT |
| flight_events_20250624_20250704.parquet | 416.0 | Tue, 14 Apr 2026 01:09:25 GMT |
| flight_events_20250704_20250714.parquet | 414.8 | Tue, 14 Apr 2026 01:17:27 GMT |
| flight_events_20250714_20250724.parquet | 389.0 | Tue, 14 Apr 2026 01:25:13 GMT |
| flight_events_20250724_20250803.parquet | 380.6 | Tue, 14 Apr 2026 01:32:57 GMT |
| flight_events_20251221_20251231.parquet | 292.4 | Mon, 13 Apr 2026 08:03:52 GMT |
| flight_events_20251231_20260110.parquet | 40.7 | Mon, 13 Apr 2026 08:06:38 GMT |
| flight_events_20260110_20260120.parquet | 271.8 | Mon, 13 Apr 2026 08:09:40 GMT |
| flight_events_20260120_20260130.parquet | 238.5 | Wed, 20 May 2026 06:39:08 GMT |
| flight_events_20260130_20260209.parquet | 289.2 | Wed, 20 May 2026 06:42:10 GMT |
| flight_events_20260619_20260629.parquet | 507.9 | Mon, 17 Aug 2026 07:25:48 GMT |
| flight_events_20260629_20260709.parquet | 506.1 | Mon, 17 Aug 2026 07:28:58 GMT |
| flight_events_20260709_20260719.parquet | 547.7 | Mon, 17 Aug 2026 07:32:12 GMT |
| flight_events_20260719_20260729.parquet | 553.5 | Mon, 17 Aug 2026 07:35:24 GMT |
| flight_events_20260729_20260808.parquet | 158.8 | Mon, 17 Aug 2026 07:38:20 GMT |
| flight_list_202412.parquet | 37.2 | Mon, 23 Feb 2026 16:07:24 GMT |
| flight_list_202501.parquet | 33.9 | Mon, 23 Feb 2026 16:08:02 GMT |
| flight_list_202502.parquet | 35.0 | Mon, 23 Feb 2026 16:08:38 GMT |
| flight_list_202506.parquet | 52.4 | Mon, 23 Feb 2026 16:11:36 GMT |
| flight_list_202507.parquet | 55.3 | Mon, 23 Feb 2026 16:12:21 GMT |
| flight_list_202508.parquet | 52.2 | Mon, 23 Feb 2026 16:13:09 GMT |
| flight_list_202512.parquet | 34.5 | Mon, 13 Apr 2026 12:12:12 GMT |
| flight_list_202601.parquet | 20.3 | Wed, 20 May 2026 06:34:56 GMT |
| flight_list_202602.parquet | 30.2 | Tue, 16 Jun 2026 11:42:30 GMT |
| flight_list_202606.parquet | 47.3 | Mon, 17 Aug 2026 06:59:18 GMT |
| flight_list_202607.parquet | 53.2 | Mon, 17 Aug 2026 06:59:40 GMT |

URL pattern: https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_YYYYMMDD_YYYYMMDD.parquet and .../flight_list/flight_list_YYYYMM.parquet

License. No named open license was found. https://www.opdi.aero/about says: "Open Data: The term Open Data refers to data, typically managed and available online, which is free of cost and accessible. Open data can be freely used, reused, and distributed provided that the data source is attributed." and "The focus of the OPDI is on open data, in particular open access and open use of data for non-commercial purposes." The Disclaimer link in the opdi.aero navigation goes to https://ansperformance.eu/about/disclaimer/, which says: "This data is published by EUROCONTROL for information purposes. It may be copied in whole or in part, provided that EUROCONTROL is mentioned as the source and it is not used for commercial purposes (i.e. for financial gain). The information in this document may not be modified without prior written permission from EUROCONTROL." The GitHub repo https://github.com/euctrl-pru/OPDI has no license file (GitHub API license field is null). The organizer rule asks for datasets "openly available under an open source license". OPDI terms are attribution plus non-commercial plus no modification, which is not a standard open license. The organizers would have to decide.

## Event types

Same set in all four months: entry-apron, entry-deicing_pad, entry-hangar, entry-parking_position, entry-runway, entry-taxiway, entry-threshold, exit-apron, exit-deicing_pad, exit-hangar, exit-parking_position, exit-runway, exit-taxiway, exit-threshold, first-xing-fl100, first-xing-fl245, first-xing-fl50, first-xing-fl70, first_seen, landing, last-xing-fl100, last-xing-fl245, last-xing-fl50, last-xing-fl70, last_seen, level-end, level-start, take-off, top-of-climb, top-of-descent.

Only the airport element events (parking_position, apron, taxiway, runway, threshold, deicing_pad, hangar) carry an airport, an OSM ref (stand name) and the callsign in info. take-off exists only when ground positions were received. Off-block-like signals are exit-parking_position and an on-ground first_seen.

## Matching

Key: ADEP, callsign (CALLSIGN_flt, or FLIGHT_mvt when null; both are real), and an OPDI anchor near the airport (take-off event, else last runway exit, else first_seen within 12 km, else last ground event) within 20 min of MVT_TIME. Departures with two or more candidates in that window: 2025-01 0, 2025-07 0, 2026-01 0, 2026-07 1. Anchor minus MVT_TIME medians: exit-runway 17.0 s (n 300,445), first_seen 35.0 s (n 250,259), take-off -5.0 s (n 19,243).

| Month | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LTFM | LSZH | All |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-01 matched % | 90.3 | 92.2 | 92.4 | 91.1 | 92.4 | 83.9 | 92.8 | 92.5 | 54.9 | 91.5 | 85.9 |
| 2025-07 matched % | 94.1 | 99.3 | 99.5 | 97.1 | 98.8 | 86.6 | 93.8 | 99.5 | 46.8 | 97.3 | 89.5 |
| 2026-01 matched % | 62.8 | 62.7 | 64.8 | 69.9 | 62.3 | 45.3 | 63.5 | 62.7 | 44.4 | 62.0 | 59.2 |
| 2026-07 matched % | 96.3 | 99.6 | 99.5 | 97.7 | 97.7 | 98.7 | 97.0 | 99.5 | 51.9 | 97.7 | 92.1 |

LTFM is about half matched even at flight-list level, and LEMD about 85 percent in 2025. Neither has ground events, so this does not change off-block coverage.

## Proxies

- stand_exit: last exit-parking_position event whose OSM ref equals STAND_mvt, before take-off
- first_park_exit: last exit of the first parking_position element the flight was seen in
- first_seen_ground: OPDI first_seen event within 6 km of the airport, at least 120 s before take-off, followed by taxi events
- best: stand_exit, else first_park_exit, else first_seen_ground

GPS jitter at the stand creates early parking exits while the aircraft is still parked. The first exit overestimates taxi time badly (LSZH RMSE 570.8 s), so the last exit of the stand is used.

## 2025 accuracy against labels (rows with n of at least 30)

Error = OPDI estimate (MVT_TIME minus the event) minus label, in seconds. bc = after per-airport median bias correction (in sample). Model = holdout prediction on the same rows.

| Proxy | Month | Airport | Coverage % | n | Median error | Median abs | p90 abs | RMSE | bc median abs | bc RMSE | Model RMSE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stand_exit | 2025-01 | EDDM | 1.2 | 132 | -44.5 | 50.0 | 118.5 | 163.4 | 26.5 | 165.9 | 192.252 |
| stand_exit | 2025-01 | LEBL | 8.6 | 1,058 | -90.5 | 95.0 | 316.3 | 201.3 | 89.0 | 159.1 | 124.758 |
| stand_exit | 2025-01 | LSZH | 40.3 | 3,919 | -67.0 | 77.0 | 348.0 | 275.5 | 47.0 | 255.3 | 180.214 |
| stand_exit | 2025-01 | ALL | 3.3 | 5,127 | -69.0 | 79.0 | 338.0 | 262.8 | 54.0 | 241.4 | 170.902 |
| stand_exit | 2025-07 | EDDF | 0.2 | 54 | -33.5 | 42.5 | 301.7 | 143.3 | 46.5 | 127.0 | 123.193 |
| stand_exit | 2025-07 | EDDM | 0.8 | 123 | -43.0 | 52.0 | 348.0 | 298.6 | 36.0 | 301.8 | 133.98 |
| stand_exit | 2025-07 | LEBL | 1.4 | 230 | -229.5 | 237.0 | 428.8 | 360.1 | 99.5 | 270.3 | 237.539 |
| stand_exit | 2025-07 | LSZH | 55.4 | 6,983 | -59.0 | 64.0 | 329.0 | 302.0 | 33.0 | 288.0 | 170.352 |
| stand_exit | 2025-07 | ALL | 3.9 | 7,393 | -60.0 | 66.0 | 331.0 | 303.3 | 35.0 | 288.8 | 172.026 |
| first_park_exit | 2025-01 | EDDM | 1.9 | 219 | -48.0 | 54.0 | 184.2 | 177.3 | 27.0 | 174.7 | 187.983 |
| first_park_exit | 2025-01 | LEBL | 11.6 | 1,421 | -181.0 | 190.0 | 364.0 | 264.4 | 120.0 | 198.6 | 133.266 |
| first_park_exit | 2025-01 | LFPG | 0.2 | 36 | -73.0 | 126.0 | 613.0 | 454.6 | 68.5 | 451.2 | 398.584 |
| first_park_exit | 2025-01 | LSZH | 51.2 | 4,987 | -104.0 | 136.0 | 393.4 | 369.2 | 86.0 | 344.8 | 189.633 |
| first_park_exit | 2025-01 | ALL | 4.3 | 6,672 | -106.0 | 136.5 | 381.0 | 346.5 | 92.0 | 318.8 | 180.88 |
| first_park_exit | 2025-07 | EDDF | 1.1 | 226 | -516.5 | 516.5 | 818.5 | 562.6 | 197.0 | 297.1 | 155.215 |
| first_park_exit | 2025-07 | EDDM | 1.3 | 200 | -49.5 | 60.0 | 357.1 | 307.8 | 38.0 | 307.7 | 139.819 |
| first_park_exit | 2025-07 | LEBL | 1.6 | 274 | -248.5 | 252.0 | 446.7 | 362.4 | 87.5 | 255.9 | 229.772 |
| first_park_exit | 2025-07 | LSZH | 71.2 | 8,974 | -79.0 | 90.0 | 384.0 | 368.5 | 59.0 | 347.9 | 174.621 |
| first_park_exit | 2025-07 | ALL | 5.1 | 9,681 | -83.0 | 99.0 | 410.0 | 373.0 | 66.0 | 349.9 | 175.357 |
| first_seen_ground | 2025-01 | EDDF | 1.7 | 255 | -531.0 | 538.0 | 866.8 | 613.7 | 165.0 | 343.2 | 155.833 |
| first_seen_ground | 2025-01 | EDDM | 17.9 | 2,012 | -356.0 | 367.5 | 634.9 | 531.4 | 173.5 | 454.5 | 191.456 |
| first_seen_ground | 2025-01 | EGLL | 0.4 | 86 | -571.5 | 603.0 | 1044.5 | 885.9 | 265.0 | 828.8 | 265.473 |
| first_seen_ground | 2025-01 | EHAM | 5.2 | 961 | -276.0 | 299.0 | 494.0 | 439.5 | 164.0 | 408.2 | 170.593 |
| first_seen_ground | 2025-01 | LEBL | 71.5 | 8,752 | -372.0 | 385.0 | 619.0 | 450.3 | 159.0 | 341.7 | 141.752 |
| first_seen_ground | 2025-01 | LFPG | 1.2 | 217 | -193.0 | 517.0 | 2011.2 | 1147.9 | 353.0 | 1167.1 | 350.317 |
| first_seen_ground | 2025-01 | LSZH | 79.0 | 7,691 | -25.0 | 79.0 | 1508.0 | 968.9 | 71.0 | 976.3 | 185.818 |
| first_seen_ground | 2025-01 | ALL | 13.0 | 19,997 | -135.0 | 282.0 | 672.4 | 714.3 | 200.0 | 713.4 | 170.662 |
| first_seen_ground | 2025-07 | EDDF | 4.4 | 934 | -426.0 | 433.5 | 711.7 | 487.9 | 163.0 | 308.5 | 155.961 |
| first_seen_ground | 2025-07 | EDDM | 10.9 | 1,730 | -400.0 | 408.0 | 637.0 | 560.2 | 135.0 | 470.7 | 219.146 |
| first_seen_ground | 2025-07 | EHAM | 3.4 | 772 | -282.5 | 322.0 | 564.7 | 614.9 | 208.5 | 608.5 | 238.348 |
| first_seen_ground | 2025-07 | LEBL | 19.2 | 3,223 | -352.0 | 376.0 | 762.8 | 572.7 | 244.0 | 483.5 | 270.847 |
| first_seen_ground | 2025-07 | LEMD | 0.9 | 175 | -167.0 | 178.0 | 543.6 | 434.2 | 92.0 | 391.5 | 191.271 |
| first_seen_ground | 2025-07 | LSZH | 94.5 | 11,922 | 3.0 | 67.0 | 1626.9 | 1060.0 | 69.0 | 1059.0 | 177.246 |
| first_seen_ground | 2025-07 | ALL | 9.8 | 18,774 | -35.0 | 145.5 | 916.0 | 910.3 | 146.0 | 915.5 | 202.439 |

Covered rows by label size: stand_exit 12,515 below 3600 s and 5 at or above; first_park_exit 16,345 below 3600 s and 8 at or above; first_seen_ground 38,756 below 3600 s and 15 at or above; best 38,819 below 3600 s and 15 at or above.

Rows where BLOCK_TIME equals SCHED_TIME (schedule fallback, 0.4 percent of the ruler): stand_exit n 24, OPDI median error -75.0, RMSE 189.2 against model RMSE 149.8; best n 97, OPDI median error -275.0, RMSE 484.3 against model RMSE 255.1. These rows are too few among covered rows to matter.

LSZH stand exits depend on the pier. Error IQR by pier: D 16 s, H 21 s, I 17 s, F 42 s (tight), against A 198 s, B 173 s, C 221 s, E 237 s, G 263 s (wide; median error on E -228.5 s, on C -232.0 s). On wide piers a group of exits sits 4 to 7 min after the label's off-block. Per-stand bias correction learned on the other month moves RMSE from 256.9 to 240.8 (January) and from 287.0 to 279.1 (July).

## 2026 coverage (ranking departures, no labels)

Share of departures with each proxy, in percent. Usable parking = stand_exit, or first_park_exit outside EDDF. At EDDF, first_park_exit is not usable: in 2025-07 its median error was -516.5 s, because the first parking element seen is usually not the departure stand.

| Month | Airport | Departures | Matched | stand_exit | first_park_exit | Usable parking | first_seen_ground | Any proxy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-01 | EDDF | 15,667 | 62.8 | 5.0 | 13.4 | 5.0 | 39.1 | 39.1 |
| 2026-01 | EDDM | 10,953 | 62.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2026-01 | EGLL | 19,283 | 64.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2026-01 | EHAM | 16,232 | 69.9 | 0.0 | 0.0 | 0.0 | 5.4 | 5.4 |
| 2026-01 | LEBL | 12,201 | 62.3 | 0.0 | 0.0 | 0.0 | 10.0 | 10.0 |
| 2026-01 | LEMD | 16,802 | 45.3 | 0.0 | 0.0 | 0.0 | 0.1 | 0.1 |
| 2026-01 | LFPG | 18,129 | 63.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2026-01 | LIRF | 11,061 | 62.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2026-01 | LTFM | 22,396 | 44.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2026-01 | LSZH | 9,995 | 62.0 | 26.8 | 38.4 | 39.4 | 58.2 | 59.2 |
| 2026-01 | ALL | 152,719 | 59.2 | 2.3 | 3.9 | 3.1 | 9.2 | 9.3 |
| 2026-07 | EDDF | 20,650 | 96.3 | 7.5 | 22.6 | 7.5 | 58.7 | 58.8 |
| 2026-07 | EDDM | 15,013 | 99.6 | 0.1 | 0.1 | 0.1 | 7.3 | 7.3 |
| 2026-07 | EGLL | 20,557 | 99.5 | 0.0 | 0.0 | 0.0 | 0.4 | 0.4 |
| 2026-07 | EHAM | 21,950 | 97.7 | 0.0 | 0.0 | 0.0 | 2.9 | 2.9 |
| 2026-07 | LEBL | 17,879 | 97.7 | 0.1 | 0.1 | 0.2 | 14.8 | 14.8 |
| 2026-07 | LEMD | 20,152 | 98.7 | 0.2 | 0.3 | 0.3 | 3.6 | 3.6 |
| 2026-07 | LFPG | 21,743 | 97.0 | 0.0 | 0.0 | 0.0 | 0.4 | 0.4 |
| 2026-07 | LIRF | 15,838 | 99.5 | 0.0 | 0.0 | 0.0 | 0.1 | 0.1 |
| 2026-07 | LTFM | 25,185 | 51.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2026-07 | LSZH | 13,155 | 97.7 | 56.4 | 68.8 | 71.2 | 95.9 | 96.4 |
| 2026-07 | ALL | 192,122 | 92.1 | 4.7 | 7.2 | 5.7 | 15.6 | 15.7 |

January 2026 gap: Events exist for 2026-01-01, 2026-01-11 to 2026-01-22, a partial 2026-01-24 (about 18 percent of a normal day) and 2026-01-25 to 2026-01-31. No events for 2026-01-02 to 2026-01-10 and 2026-01-23. The flight list shows the same days. The file flight_events_20251231_20260110 (40.7 MB, Last-Modified 2026-04-13) holds only 2025-12-30 to 2026-01-01; flight_events_20260110_20260120 starts at 2026-01-11. No other file covers the gap. January 2025 also has a gap: 2025-01-26 and 2025-01-27 hold only 13,559, 8,470 events against a median of 842,562 per day. July 2025 and July 2026 are complete.

Consistency check without labels: median absolute gap between the bias-corrected stand exit and the model's test prediction is EDDF 129.2 s in 2026-01, LSZH 156.8 s in 2026-01, EDDF 124.7 s in 2026-07, LSZH 139.3 s in 2026-07. For LSZH in 2025 it was 131.9 s and 123.6 s. Nothing suggests 2026 OPDI is sharper than 2025.

## Local RMSE (holdout, manifest manifest_m6r_avg2.json)

Model: 317.744 (2025-01 342.65, 2025-07 296.15). Bias corrections are per airport and learned on the other ruler month. OPDI estimates are clipped at 0.

| Proxy | Covered rows | Model RMSE on covered | OPDI bc RMSE on covered | Replace all covered | Use OPDI if abs diff under 120 s | Under 180 s | Average 0.25 OPDI | Cross-month OLS stack |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stand_exit | 12,520 (3.6%) | 171.567 | 271.004 | 320.251 | 317.702 | 317.701 | 317.365 | 317.367 |
| first_park_exit | 16,353 (4.8%) | 177.631 | 336.232 | 323.776 | 317.726 | 317.748 | 317.442 | 317.399 |
| first_seen_ground | 38,771 (11.3%) | 186.726 | 792.762 | 409.613 | 317.778 | 317.905 | 322.892 | 317.642 |
| best | 38,834 (11.3%) | 186.73 | 389.206 | 337.801 | 317.778 | 317.95 | 317.574 | 317.229 |

Other thresholds (30 to 600 s) and averages (0.5, 0.75) are in opdi_findings.json; none beats the table. A pier-level trust rule (use OPDI only on piers where it beat the model in the other month) gave 317.695. Ceiling check (OPDI exactly right on covered rows): 316.055 for stand_exit rows, 315.347 for usable parking rows, 311.496 for every covered row.

## Could not verify

- Whether OPDI terms satisfy the organizer's open source license rule. The site states no named license.
- EDDF accuracy in 2026: in 2025 EDDF has only 54 stand-matched rows (2025-07: median error -33.5 s, RMSE 143.3 s against model 123.193 s), so 2026 EDDF quality is extrapolated.
- Why LTFM matches only about half even at flight-list level (callsign or coverage differences on the OPDI side). It has no ground events, so it does not affect off-block.
- The cause of the late LSZH stand exits on wide piers (OSM stand geometry versus push and start procedure). The data only shows it is pier dependent.

Files: download.sh, extract.py, match.py, report.py, finalize.py, make_md.py, match_<month>.parquet, extract_<month>.json, opdi_findings.json. Raw OPDI files and derived extracts are in data_ext/opdi (about 5.6 GB).
