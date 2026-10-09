# Host scheduling scripts and download loops

These scripts scheduled, waited for, started, copied and checked the steps of the final rebuild, the download of the February and June 2026 ADS-B days, and the month-held-out rulers, on the author's PC (Windows, Git Bash and PowerShell) and Mac (macOS). They record how the steps were run; the steps themselves and their arguments are the ones listed in `docs/REPRODUCE.md` section 0 (the commands inside `research/f3/final_orch.py` and `research/f3/final_chain.py`). They are copies of the PC working copies, except where a row names the Mac copy that ran; inside them, paths such as `research/f3/twin.sh` refer to the original locations listed below.

Changes against the original, nothing else: one SPDX line at the top; the paths of the author's machines, the ssh host name of the Mac, references to private notes and one mention of the author's tooling are replaced by environment variables or placeholders (`PRC_WORK`: the Mac work folder that held the build sandboxes and the final root; `ADSB_WORK` and `ADSB_HOME`: the Mac ADS-B folders; `PY_MAC`, `PY_ADSB`, `PY_PC`, `PY_ADSB_PC`: the Python interpreters; `MAC_HOST`, `MAC_FINAL_ROOT`: the Mac as seen from the PC; `PC_REPO`, `PC_FINAL_ROOT`, `PC_LOCKS`, `ADSB_PC_WORK`: PC folders; `MEMCAP`: the author's memory-cap wrapper, not part of this repository; `<mac-work>` and `<pc adsb work>` inside strings). So they do not run elsewhere without setting those variables. Several of them wait for lock folders and marker files shared with other jobs on the same machines (for example `arc_quiet.request`, `FIRST_CLAIM`, `GO_FJ`, `REALMLP_ALONE`); those only order the jobs and do not change any output.

The original sha16 is the first 16 hex digits of the SHA-256 of the copy it was taken from (the PC working copy, or the Mac copy where the row says so); the cleaned sha16 is that of the file in this folder.

| File here | Original path | What it did | Original sha16 | Cleaned sha16 |
|---|---|---|---|---|
| `f3/phase2.sh` | `research/f3/phase2.sh` | First form of the final-root sequence after the ADS-B package (tables, feats_v27 check, LightGBM legs, geo table). Waited for the package and was replaced by phase2b.sh before it built anything | c4e60d65449ed388 | 37231244d02267d9 |
| `f3/phase2b.sh` | `research/f3/phase2b.sh` | Second form (fit-cache path). Stopped at its own ADS-B table step on the first package and built nothing; replaced by phase2c.sh | 1107421b72a7a3e3 | a47cad5c985c89a0 |
| `f3/phase2c.sh` | `research/f3/phase2c.sh` | Final-root sequence on the second ADS-B package: package check, feats_v27 and its check, interim-against-final comparison, geo table, then the LightGBM, CatBoost and RealMLP phases of research/f3/final_orch.py through the fit cache, with the twin check beside | 649e4f78b7fc1352 | d4521959ad93ea22 |
| `f3/phase3.sh` | `research/f3/phase3.sh` | After the leg phases: TabM on MPS, chains A, B and C (research/f3/final_chain.py), the three-chain average, the TabM device comparison, the final file and its checks, the daily-format copy of its January and July rows | 78b0fc40fa85e477 | abbfaaec6a92f4fb |
| `f3/realmlp_alone.sh` | `research/f3/realmlp_alone.sh` | Marks the RealMLP phase so that no ADS-B download starts on the Mac while the RealMLP legs run (memory) | de4e64f724912a34 | 6755500fde464211 |
| `f3/tabm_prep_wait.sh` | `research/f3/tabm_prep_wait.sh` | Starts the TabM input arrays (research/r13_l/prep_x.py) once the final feats_v27 has passed its check | 8cbcb4e304a7286b | d0175d8ba5ada730 |
| `f3/tabm_pc.sh` | `research/f3/tabm_pc.sh` | Runs the TabM leg on the PC GPU (CUDA) with the v35 arguments, after copying its inputs from the Mac and comparing their hashes; copies the output back | 9346ab1cd3a6c486 | b4acb6ea6381f4cf |
| `f3/mk_interim.sh` | `research/f3/mk_interim.sh` | Makes the interim root: the final root's code and tables plus an interim feats_v27 whose 2025 rows equal the final table's | ba0043c7fe68748b | cc1d5c67295feb3d |
| `f3/interim_run.sh` | `research/f3/interim_run.sh` | Runs the LightGBM legs (and, while the ADS-B package was pending, the CatBoost legs) in the interim root with the fit cache in save mode | f111041f3db86cf4 | 9b00d41a64c2a000 |
| `f3/twin.sh` | `research/f3/twin.sh` | Refits leg sk seed 0 directly in a scratch root and compares it with the same leg loaded through the fit cache (every column within 1e-9, equal int32 rounding) | d4d494aa90d07b10 | 9eb588a10b16e749 |
| `f3/wt_run.sh` | `research/f3/wt_run.sh` | Fit-cache tests T1 to T5 on the m6r25 leg and the CatBoost tests (direct, save, load, strict) | 8133d9e35b2cb2b2 | 48e0de653eeedc77 |
| `f3/wt_t1b.sh` | `research/f3/wt_t1b.sh` | A second direct run T1b to measure LightGBM's run-to-run difference | 28160f9cd238299c | bba7a958bd0599dc |
| `f3/conf_run.sh` | `research/f3/conf_run.sh` | Driver of the registered re-draw check: sampler, three feature builds, one table, one validation fit, the evaluation | d6a997b6981c9d80 | defbba57af97af65 |
| `f3/collect_ro.py` | `research/f3/collect_ro.py` | Copies the val and fwd leg outputs and the iteration files of the daily build into the final root as read-only inputs; reports conflicts between sandboxes | fb723481b79fcbb0 | d55a5610001376d3 |
| `r17_a/pilot_r17a.sh` | `research/r17_a/pilot_r17a.sh` | Two pilot days (2026-02-01 from disk, 2026-02-02 streamed) to measure download and parse speed | b8e51e4c078425ed | de808337225763a0 |
| `r17_a/dl_r17a.sh` | `research/r17_a/dl_r17a.sh` | Mac download and extraction runner, first form: workers claim days, run research/r17_a/adsb_day_r17a.py --stream, retry a failed day up to 3 times | 7256cada213d9e50 | d8a18399bf752cce |
| `r17_a/dl2_r17a.sh` | `research/r17_a/dl2_r17a.sh` | dl_r17a.sh with a first-worker number, for a second runner | 3659910c5ef121cd | f2e5fce429a529b4 |
| `r17_a/dl3_r17a.sh` | `research/r17_a/dl3_r17a.sh` | Third form: each stream declares 0.25 thread; at most 4 streams | f6a5c4e9c61324dc | 4fde5ff1613e0a44 |
| `r17_a/dl4_r17a.sh` | `research/r17_a/dl4_r17a.sh` | Fourth form: limit by the sum of the curl rate caps (at most 19.7 MB/s) instead of a stream count | 526715793c2ee5b2 | 85b6fc9fd5ffcfb7 |
| `r17_a/dl5_r17a.sh` | `research/r17_a/dl5_r17a.sh` | Fifth form: no new day while the RealMLP legs run alone; the checks are repeated right before curl starts | a1c65285c928cfc5 | fc6740ea8aee63fe |
| `r17_a/dl6_r17a.sh` | `research/r17_a/dl6_r17a.sh` | Sixth form: dl5 with the rate cap kept in each lock record | 6580c13e012e27f0 | 839d287edc70b656 |
| `r17_a/switch_r17a.sh` | `research/r17_a/switch_r17a.sh` | Moved the Mac runner from 4 streams to 2 so that the PC could add 2 under the same cap | 84320ab201ef7d82 | 44b3686379f234d5 |
| `r17_a/pc_dl_r17a.sh` | `research/r17_a/pc_dl_r17a.sh` | PC download and extraction runner: claims days on the Mac, extracts on the PC, copies the results to the Mac | dd101ebdf29c6617 | 5d551b3e9cac961a |
| `r17_a/pc_guard_r17a.ps1` | `research/r17_a/pc_guard_r17a.ps1` | PC watchdog: stops the PC download side when committed memory reaches 85% or free memory falls below 4 GB, and releases its days | d9b52e01339f91e5 | 15230a00e4b1996e |
| `r17_a/detach.ps1` | `research/r17_a/detach.ps1` | Starts a bash command line as a hidden background process on the PC | 26dd2de4c810c17b | 42eb3b655c18ab8d |
| `r17_a/fixer_r17a.py` | `research/r17_a/fixer_r17a.py` | Switches a day to the other full replica (prod-0 or staging-0) after its first download failed the completeness check; logs every switch | 850a44786a86e25d | f6c75c30e1d4e8d3 |
| `r17_a/step_r17a.sh` | `research/r17_a/step_r17a.sh` | Runs one short step under the Mac's shared rules for threads and memory | 0e3fa0a010a91013 | 1a4c6f8add78ebe4 |
| `r17_a/pack_when_ready.sh` | `research/r17_a/pack_when_ready.sh` | Builds the ADS-B package of the final input once all 58 February and June days are extracted | 8f606a82cfbac94e | 61cf7da079a32691 |
| `f2/f2_ctx_mac.sh` | `research/f2/mac/f2_ctx_mac.sh` | Builds the context tables f2ctx_v1 to v3 with January, February, June and July 2025 rows (input of the month-held-out rulers) | 9630444ae1d4621b | 0f1489418e0e3108 |
| `f2/f2_leg_mac.sh` | `research/f2/mac/f2_leg_mac.sh` | Leave-February-and-June-out LightGBM leg of the window chain (setting fj) and its G band | c8ce5c89c79d70fa | 795a2a89353f2af1 |
| `f2/f2_m3_mac.sh` | `research/f2/mac/f2_m3_mac.sh` | Runs research/f2/m3_corr.py (ruler M3, corrector) | 2ec4e0d00696f30a | 6a5802c9d5848c3a |
| `f2/f2_m3b_mac.sh` | `research/f2/mac/f2_m3b_mac.sh` | Runs research/f2/m3_corr.py, research/f2/m3_corr_lomo.py and research/f2/m3_tail.py | 12bd02364310e4b7 | 88720ba778ea5a7e |
| `f2/f2_m3c_mac.sh` | `research/f2/mac/f2_m3c_mac.sh` | Runs research/f2/m3_tail.py only | 43ae2b2a7fb491da | 30d5c2d8055b2a4f |
| `f2/f2_aiu_mac.sh` | `research/f2/mac/f2_aiu_mac.sh` | Runs research/f2/aiu_ablate.py (corrector without its AIU columns; information only) | c44c2962a6dfd48c | f7a46aaa314c53d9 |
| `f2/f2_regress_watch.sh` | `research/f2/mac/f2_regress_watch.sh` | Runs research/f2/regress_check.py on each final-root table once the table has stopped changing | 8feb731ad7409ccd | d65221a7a6c2ab29 |
| `f2/f2_smoke_mac.sh` | `research/f2/mac/f2_smoke_mac.sh` | Runs research/f2/gate14.py and gate15.py in the single-leg world and checks them against the numbers of m3_corr.py | c7fe87c41afb225e | ee894d9eb1522ab4 |
| `f2/kit_final.sh` | `research/f2/send_final/kit_final.sh` | Builds the corrector row split of the final file in the final root on the Mac (DESIGN.md section 14.7): installs the merge and check scripts after checking their hashes, builds the four-month context table and the four-month corrector (seeds 0 to 2, scale2 0.75 and 0.5), merges the two forms by row group in each chain (research/f2/merge_corr.py), writes the merged manifests, runs the ten tail stages and the three-chain average, and runs research/f2/check_split.py. This copy is the one sent to the Mac; it was equal to the Mac copy when that arrived (2026-10-10 00:13 KST) | 8069a08c39342cb4 | 01b562e838dd5562 |

Copies on the Mac (SHA-256 list of the Mac folders taken on 2026-10-09): 32 of the scripts above have a Mac copy in the folder they ran from, and 31 of those equal the original. Differs: `f2/f2_m3b_mac.sh` (Mac copy sha16 61cacef42972aba5), so the file here is not the copy that ran; it is replaced by the Mac copy at the next update. The other 5 ran on the PC or have no copy in that list.

Host scripts kept at their original path, because other scripts and the documents call them there:

| File | What it does |
|---|---|
| `research/f3/mac_step.sh` | Runs one step of the final rebuild on the Mac: waits for free threads and memory, takes a lock, runs the command with nice and /usr/bin/time; called by final_orch.py and final_chain.py |
| `research/f3/mk_final_root_mac.sh` | Makes the final root on the Mac from the daily build root (code, 2025 data, METAR, AIU files, corrector model stores) |
| `research/f3/run_ctx_final.sh` | Builds ctx_v1 (frozen stand codes), ctx_v2 and ctx_v3 in the final root and compares them with the daily tables |

Not included: helpers that only print status. The runners of the month-held-out gates 1.4 to 1.7 were still running when this folder was written: <!--FILL:f2_gate_runners-->.
