# SPDX-License-Identifier: GPL-3.0-only
"""F2 Package S (coordinator approval 10-11 01:17 KST, registered in the PRC master at 01:17 before any number): chain C only, at the
package seed counts. LightGBM ap 1 -> 3 seeds (m6r28skap seeds 0, 1, 2), CatBoost ap 1 -> 3 seeds, and chain C's own RealMLP seeds 4-7
instead of chain B's 0-3 (shared before). The same seed counts for chain C's leave-Feb/Jun-out (fj) learners, whose blend is the extra rows
of chain C's four-month corrector. Names, either family: 603 = the S default (T 0.75 form), 604 = its Jan/Jul twin (597's form),
605c = the S final (c form, BbF merge). This file only holds names, seed lists and exact commands; it runs nothing.

Rules every tool here follows (from the approval): a new seed's command is its leg's reference command with --seed changed; --base may
differ only by the B0 link convention of 514 and R17-A (<base>B0_s<k> is a symlink to the base's seed-0 file of the same setting), and the
averaged column (mu_cat, mu_rmlp; model3 p_art, mu, pred, pred_noG on non-G rows) is computed by the scripts before they read --base.
Averages: rows, order and every other column from the donor (the leg's seed-0 file, for a G-band leg its seed-0 G-band file), the
column = np.mean over the listed seeds (the donor's own column stands for seed 0 when 0 is listed), G rows of a G-band leg stay seed 0's.

    python -X utf8 research/f2/s_spec.py names <598c|l4>                       every name as JSON
    python -X utf8 research/f2/s_spec.py jobs <598c|l4> <val,fwd|full|fj> [legs]  learner jobs as TSV (research/f2/s_learn.py reads it)
    python -X utf8 research/f2/s_spec.py joblist <598c|l4>                     the final-root job list for F3 (text)
"""
from __future__ import annotations

import json
import sys

M3 = ("--feats feats_v28 --es internal --refit --wpow 1 --leaves 255 --mdl 100 --no_direct --clf_drop_prefix m_,ab_ --full_mult 1.0 "
      "--threads 6 --mu_sep all")
MID = "adsbgeondctxv3x50ws75"
TAILS = ["am", "fh", "gkc", "gkcx", "xncr", "db", "aw", "gp", "gl", "gja"]
SUF = {"am": "am", "fh": "amfh", "gkc": "amfhgkc", "gkcx": "amfhgkcx", "xncr": "amfhgkcxncr", "db": "amfhgkcxncrdb", "aw": "amfhgkcxncrdbaw",
       "gp": "amfhgkcxncrdbawgp", "gl": "amfhgkcxncrdbawgpgl", "gja": "amfhgkcxncrdbawgpglgja"}
REF = "--ref_base m6r25sg --ref_tag m6r25sgoof4 --ref_learners m6r25cb:mu_cat,m6r25xg:mu_xgb,m6r25rm:mu_rmlp --folds 10"
CORR_SBX = ("--feats feats_v25 --adsb adsb_v1nd --adsb_sha 0d03ff06267e0dee --folds 10 --rounds 300 --rounds2 400 --max_pred 3600 --threads 4 "
            "--geo geo_v1nd --geo_sha 43310b9bbeefe81b --ctx ctx_v3 --ctx_sha 0b139ddbd6e70573 --scale2 0.75")
CORR_FINAL = ("--feats feats_v25 --adsb adsb_v1nd --adsb_sha 07df03ca69833082 --folds 10 --rounds 300 --rounds2 400 --max_pred 3600 --threads 4 "
              "--geo geo_v1nd --geo_sha 9dfa71b64e1099de --ctx ctx_v3 --ctx_sha 01ac984c7afd6e05 --scale2 0.75")
FITCACHE_SBX = "research/f2/fitcache_s"  # new and empty in each S sandbox clone: every chain C corrector fit is new (the val base changed)
SUFFIX = "skaps"  # tools/chainavg3.py --suffix of 603 and 605c (the family's own avg3 used skap; the name is all it changes)
SBX_LEARN_598C = "~/nova-adsb/prcm/f2s_sbx598/research/f2/out/s_learn_598c_valfwd.json"  # the 598c sandbox learner record (--rm_seeds_from in f2_run)
AVG = "research/f2/avg_s.py"


def _leg(**k) -> dict:
    return k


FAM = {
    "598c": dict(
        label="598c (L3: feats_v28 LightGBM legs; 514's CatBoost and RealMLP learners on feats_v27)",
        sbx="f2s_sbx598", sbx_src="r17a_sbx", default="596", final="598c", twin="597", twin_score=228.5043,
        ref_sbx_manifest="preds/manifest_m6r28wincbrmfw4sv434tmcadsbgeondctxv3x50ws75c3amfhgkcxncrdbawgpglgjaavskap.json",
        ab_state={"A": "research/r17_a/sb_state_l3s_A.json", "B": "research/r17_a/sb_state_l3s_B.json"},
        fam_C=dict(T="m6r28apcbrmfw4sv114", base="m6r28skapg", learners="m6r27cbap:mu_cat,m6r25xg:mu_xgb,m6r27rmx4:mu_rmlp",
                   sbx_TCS="m6r28apcbrmfw4sv114adsbgeondctxv3x50ws75c3", fj="l3apcbrmfw4"),
        T_S="m6r28sapcbrmfw4sv114", T_G="m6r28sgapcbrmfw4sv114", GLEG="m6r28skapg", feats="feats_v28",
        final_A={"default": "m6r28wincbrmfw4sv434tmc" + MID + "c3" + SUF["gja"], "c": "m6r28wincbrmfw4sv434tmc" + MID + "c3rBbF" + SUF["gja"]},
        final_B={"default": "m6r28skcbrmfw4sv434" + MID + "c3" + SUF["gja"], "c": "m6r28skcbrmfw4sv434" + MID + "c3rBbF" + SUF["gja"]},
        kit="research/f3/kit_c.sh 598c", c4m_cache_fam="research/f2/fitcache_4m_l3", fj_xg="l3xg",
        legs={
            "lgbap": _leg(kind="lgb", tag="m6r28skap", script="research/r13_f3/model3ap.py", args=M3, base=None, seeds=[1, 2], thr=6, gb=7.5,
                          mins={"val": 18, "fwd": 14, "full": 12, "fj": 14},
                          ref={"val": ("research/r13_f3/model3ap.py --setting val --seed 0 --tag m6r28skap " + M3, "r17a_sbx research/r17_a/sb_build_r17a.py ba41a406b80b41aa"),
                               "fwd": ("research/r13_f3/model3ap.py --setting fwd --seed 0 --tag m6r28skap " + M3, "r17a_sbx research/r17_a/sb_build_r17a.py ba41a406b80b41aa"),
                               "full": ("research/r13_f3/model3ap.py --setting full --seed 0 --tag m6r28skap " + M3, "final_run research/f3/final_orch.py l3legs 9f4f84985fe21845"),
                               "fj": ("research/f2/model3ap_fj.py --setting fj --seed 0 --tag l3skap " + M3, "final_run research/f3/final_orch.py l3legs 9f4f84985fe21845")},
                          fj=dict(tag="l3skap", script="research/f2/model3ap_fj.py", base=None, root="final_run"),
                          avg=dict(mode="m3g", donor="m6r28skapg", new="m6r28skapgx3s", seeds=[0, 1, 2]),
                          fj_avg=dict(mode="m3g", donor="l3skapg", new="l3skapgx3s", seeds=[0, 1, 2])),
            "cbap": _leg(kind="cb", tag="m6r27cbap", script="research/r13_f3/cat_muap.py", args="--feats feats_v27 --threads 8 --per_ap",
                         base={"val": ("m6r27skap", "seedk"), "fwd": ("m6r27skap", "seedk"), "full": ("m6r27skap", "B0")}, seeds=[1, 2], thr=8, gb=10.5,
                         mins={"val": 45, "fwd": 26, "full": 25, "fj": 90},
                         existing={"val": "r13f3_run", "fwd": "r13f3_run"},
                         pins={"val_m6r27cbap_s1": "027038422a8df4d2", "val_m6r27cbap_s2": "0b74be64037ca153",
                               "fwd_m6r27cbap_s1": "11d79673e0f9aefc", "fwd_m6r27cbap_s2": "9ad6b872ff0fc0e4",
                               "val_m6r27cbap_s0": "25f7f1175155787c", "fwd_m6r27cbap_s0": "76635ffdf06a46cc"},
                         ref={"val": ("research/r13_f3/cat_muap.py --setting val --threads 8 --seed 0 --tag m6r27cbap --base m6r27skap --feats feats_v27 --per_ap", "r13f3_run research/r13_f3/mac/r13f3_seeds.sh c08e649b476f20ac (seeds 1, 2 ran as seed 0 with --seed k)"),
                              "fwd": ("research/r13_f3/cat_muap.py --setting fwd --threads 8 --seed 0 --tag m6r27cbap --base m6r27skap --feats feats_v27 --per_ap", "r13f3_run research/r13_f3/mac/r13f3_seeds.sh c08e649b476f20ac"),
                              "full": ("research/r13_f3/cat_muap.py --setting full --seed 0 --tag m6r27cbap --base m6r27skap --feats feats_v27 --threads 8 --per_ap", "final_run research/f3/final_orch.py cb 9f4f84985fe21845"),
                              "fj": ("research/f2/cat_muap_fj.py --setting fj --seed 0 --tag f2cbap --base f2skap --feats feats_v27 --threads 4 --per_ap", "f2_run research/f2/mac/f2_fjlegs_mac.sh 9c151c725c811f65 (logged: 4 threads)")},
                         fj=dict(tag="f2cbap", script="research/f2/cat_muap_fj.py", args="--feats feats_v27 --threads 4 --per_ap", base=("f2skap", "B0"), root="f2_run", thr=4,
                                 launch={"py_flags": ["-u"], "env": {}}),  # as seed 0 ran (f2_fjlegs_mac.sh): $PY -u, no thread env
                         avg=dict(mode="cb", donor="m6r27cbap", new="m6r27cbapx3s", seeds=[0, 1, 2]),
                         fj_avg=dict(mode="cb", donor="f2cbap", new="f2cbapx3s", seeds=[0, 1, 2])),
            "rm": _leg(kind="rm", tag="m6r27rm", script="src/rmlp_mu.py", args="--feats feats_v27 --device mps --threads 1",
                       base={"val": ("m6r27sk", "B0"), "fwd": ("m6r27sk", "B0"), "full": ("m6r27sk", "B0")}, seeds=[4, 5, 6, 7], thr=1, gb=33,
                       mins={"val": 14, "fwd": 8, "full": 17, "fj": 14},
                       ref={"val": ("src/rmlp_mu.py --setting val --seed 1 --tag m6r27rm --feats feats_v27 --device mps --threads 1 --base m6r27skB0", "r10b_run research/r10_b/mac_scripts/r10b_rmretry6_mac.sh c08f4bea7663e1d3 (514's seeds >= 1)"),
                            "fwd": ("src/rmlp_mu.py --setting fwd --seed 1 --tag m6r27rm --feats feats_v27 --device mps --threads 1 --base m6r27skB0", "r10b_run research/r10_b/mac_scripts/r10b_rmretry6_mac.sh c08f4bea7663e1d3"),
                            "full": ("src/rmlp_mu.py --setting full --seed 1 --tag m6r27rm --base m6r27sk --feats feats_v27 --device mps --threads 1", "final_run research/f3/final_orch.py rm 9f4f84985fe21845 (base at seed k there; no m6r27sk seeds 4-7 exist, so B0)"),
                            "fj": ("research/f2/rmlp_mu_fj.py --setting fj --seed 0 --tag f2rm --base f2sk --feats feats_v27 --device mps --threads 1", "f2_run research/f2/mac/f2_fjlegs_mac.sh 9c151c725c811f65")},
                       fj=dict(tag="f2rm", script="research/f2/rmlp_mu_fj.py", args="--feats feats_v27 --device mps --threads 1", base=("f2sk", "B0"), root="f2_run", thr=1,
                               launch={"py_flags": ["-u"], "env": {"OMP_NUM_THREADS": "1"}}),  # as seed 0: env OMP_NUM_THREADS=1 $PY -u
                       avg=dict(mode="rm", donor="m6r27rm", new="m6r27rmx4s", seeds=[4, 5, 6, 7]),
                       fj_avg=dict(mode="rm", donor="f2rm", new="f2rmx4s", seeds=[4, 5, 6, 7])),
        },
        placeholders={"m6r28skapgx3s": "m6r28skapg", "m6r27cbapx3s": "m6r27cbap", "m6r27rmx4s": "m6r27rmx4"},
    ),
    "l4": dict(
        label="602c (L4: CatBoost and RealMLP learners refit on feats_v28 over L3's LightGBM legs)",
        sbx="f2s_sbxl4", sbx_src="r17a_l4g_sbx", default="600", final="602c", twin="601", twin_score=None,
        ref_sbx_spec_file="research/r17_a/l4g_spec_fs75.txt",
        ab_state={"A": "research/r17_a/l4g_state_f_As75.json", "B": "research/r17_a/l4g_state_f_Bs75.json"},
        fam_C=dict(T="m6r28xapcbrmfw4sv114", base="m6r28skapg", learners="m6r28cbap:mu_cat,m6r25xg:mu_xgb,m6r28rmx4:mu_rmlp",
                   sbx_TCS="m6r28l4gfxapcbrmfw4sv114adsbgeondctxv3x50ws75c3", fj="l4apcbrmfw4"),
        T_S="m6r28xsapcbrmfw4sv114", T_G="m6r28xsgapcbrmfw4sv114", GLEG="m6r28skapg", feats="feats_v28",
        final_A={"default": "m6r28xwincbrmfw4sv434tmc" + MID + "c3" + SUF["gja"], "c": "m6r28xwincbrmfw4sv434tmc" + MID + "c3rBbF" + SUF["gja"]},
        final_B={"default": "m6r28xskcbrmfw4sv434" + MID + "c3" + SUF["gja"], "c": "m6r28xskcbrmfw4sv434" + MID + "c3rBbF" + SUF["gja"]},
        kit="research/f3/kit_l4c.sh", c4m_cache_fam="research/f2/fitcache_4m_l4", fj_xg="l4xg",
        legs={
            "lgbap": _leg(kind="lgb", tag="m6r28skap", script="research/r13_f3/model3ap.py", args=M3, base=None, seeds=[1, 2], thr=6, gb=7.5,
                          mins={"val": 18, "fwd": 14, "full": 12, "fj": 14},
                          ref={"val": ("research/r13_f3/model3ap.py --setting val --seed 0 --tag m6r28skap " + M3, "r17a_sbx research/r17_a/sb_build_r17a.py ba41a406b80b41aa"),
                               "fwd": ("research/r13_f3/model3ap.py --setting fwd --seed 0 --tag m6r28skap " + M3, "r17a_sbx research/r17_a/sb_build_r17a.py ba41a406b80b41aa"),
                               "full": ("research/r13_f3/model3ap.py --setting full --seed 0 --tag m6r28skap " + M3, "final_run research/f3/final_orch.py l3legs 9f4f84985fe21845"),
                               "fj": ("research/f2/model3ap_fj.py --setting fj --seed 0 --tag l3skap " + M3, "final_run research/f3/final_orch.py l3legs 9f4f84985fe21845")},
                          fj=dict(tag="l3skap", script="research/f2/model3ap_fj.py", base=None, root="final_run"),
                          avg=dict(mode="m3g", donor="m6r28skapg", new="m6r28skapgx3s", seeds=[0, 1, 2]),
                          fj_avg=dict(mode="m3g", donor="l3skapg", new="l3skapgx3s", seeds=[0, 1, 2])),
            "cbap": _leg(kind="cb", tag="m6r28cbap", script="research/r13_f3/cat_muap.py", args="--feats feats_v28 --threads 10 --per_ap",
                         base={"val": ("m6r28skap", "seedk"), "fwd": ("m6r28skap", "seedk"), "full": ("m6r28skap", "seedk")}, seeds=[1, 2], thr=10, gb=10.5,
                         mins={"val": 45, "fwd": 26, "full": 25, "fj": 60},
                         ref={"val": ("research/r13_f3/cat_muap.py --setting val --seed 0 --tag m6r28cbap --base m6r28skap --feats feats_v28 --threads 10 --per_ap", "r17a_sbx research/r17_a/l4_learn_r17a.sh e822599569c2b963"),
                              "fwd": ("research/r13_f3/cat_muap.py --setting fwd --seed 0 --tag m6r28cbap --base m6r28skap --feats feats_v28 --threads 10 --per_ap", "r17a_sbx research/r17_a/l4_learn_r17a.sh e822599569c2b963"),
                              "full": ("research/r13_f3/cat_muap.py --setting full --seed 0 --tag m6r28cbap --base m6r28skap --feats feats_v28 --threads 10 --per_ap", "final_run research/f3/l4_full_f3.sh c24dd1a5d9fb0014"),
                              "fj": ("research/f2/cat_muap_fj.py --setting fj --seed 0 --tag l4cbap --base l3skap --feats feats_v28 --threads 10 --per_ap", "final_run research/f3/l4_full_f3.sh c24dd1a5d9fb0014")},
                         fj=dict(tag="l4cbap", script="research/f2/cat_muap_fj.py", args="--feats feats_v28 --threads 10 --per_ap", base=("l3skap", "seedk"), root="final_run", thr=10),
                         avg=dict(mode="cb", donor="m6r28cbap", new="m6r28cbapx3s", seeds=[0, 1, 2]),
                         fj_avg=dict(mode="cb", donor="l4cbap", new="l4cbapx3s", seeds=[0, 1, 2])),
            "rm": _leg(kind="rm", tag="m6r28rm", script="src/rmlp_mu.py", args="--feats feats_v28 --device mps --threads 1",
                       base={"val": ("m6r28sk", "B0"), "fwd": ("m6r28sk", "B0"), "full": ("m6r28sk", "B0")}, seeds=[4, 5, 6, 7], thr=1, gb=33,
                       mins={"val": 14, "fwd": 8, "full": 17, "fj": 14},
                       ref={"val": ("src/rmlp_mu.py --setting val --seed 1 --tag m6r28rm --base m6r28skB0 --feats feats_v28 --device mps --threads 1", "r17a_sbx research/r17_a/l4_learn_r17a.sh e822599569c2b963 (seeds >= 1)"),
                            "fwd": ("src/rmlp_mu.py --setting fwd --seed 1 --tag m6r28rm --base m6r28skB0 --feats feats_v28 --device mps --threads 1", "r17a_sbx research/r17_a/l4_learn_r17a.sh e822599569c2b963 (seeds >= 1)"),
                            "full": ("src/rmlp_mu.py --setting full --seed 1 --tag m6r28rm --base m6r28skB0 --feats feats_v28 --device mps --threads 1", "final_run research/f3/l4_full_f3.sh c24dd1a5d9fb0014 (seeds >= 1)"),
                            "fj": ("research/f2/rmlp_mu_fj.py --setting fj --seed 0 --tag l4rm --base l3sk --feats feats_v28 --device mps --threads 1", "final_run research/f3/l4_full_f3.sh c24dd1a5d9fb0014")},
                       fj=dict(tag="l4rm", script="research/f2/rmlp_mu_fj.py", args="--feats feats_v28 --device mps --threads 1", base=("l3sk", "B0"), root="final_run", thr=1),
                       avg=dict(mode="rm", donor="m6r28rm", new="m6r28rmx4s", seeds=[4, 5, 6, 7]),
                       fj_avg=dict(mode="rm", donor="l4rm", new="l4rmx4s", seeds=[4, 5, 6, 7])),
        },
        placeholders={"m6r28skapgx3s": "m6r28skapg", "m6r28cbapx3s": "m6r28cbap", "m6r28rmx4s": "m6r28rmx4"},
    ),
}
ORDER = ["lgbap", "cbap", "rm"]  # L4's CatBoost ap seed k reads LightGBM ap seed k, so LightGBM first


def fam(f: str) -> dict:
    if f not in FAM:
        raise SystemExit(f"family must be 598c or l4, not {f!r}")
    return FAM[f]


def rm_tag(f: str, seeds: list[int], fj: bool = False) -> str:
    """The own-RealMLP average's tag carries the number of seeds that completed (four planned; three is the floor of the approval)."""
    lg = fam(f)["legs"]["rm"]
    base = (lg["fj"]["tag"] if fj else lg["tag"])
    return f"{base}x{len(seeds)}s"


def fam_rm(f: str) -> str:
    """The family chain C's RealMLP learner (shared with chain B), kept when fewer than three own seeds finish."""
    return [x.split(":")[0] for x in fam(f)["fam_C"]["learners"].split(",") if x.endswith(":mu_rmlp")][0]


def chain_names(f: str, rm_seeds: list[int] | None = None) -> dict:
    """rm_seeds None = the planned own seeds 4-7; a list = the seeds that finished (fewer than three: the family RealMLP stays)."""
    F = fam(f)
    L = F["legs"]
    rs = L["rm"]["seeds"] if rm_seeds is None else rm_seeds
    own = len(rs) >= 3
    rm_l = rm_tag(f, rs) if own else fam_rm(f)
    rm_fj = rm_tag(f, rs, fj=True) if own else L["rm"]["fj"]["tag"]
    learners = f"{L['cbap']['avg']['new']}:mu_cat,m6r25xg:mu_xgb,{rm_l}:mu_rmlp"
    fj_learners = f"{L['cbap']['fj_avg']['new']}:mu_cat,{F['fj_xg']}:mu_xgb,{rm_fj}:mu_rmlp"
    out = {"family": f, "label": F["label"], "rm_seeds": rs, "rm_own": own}
    for kind, T in (("S", F["T_S"]), ("G", F["T_G"])):
        TC = T + MID
        out[kind] = {"T": T, "base": L["lgbap"]["avg"]["new"], "learners": learners, "TC": TC, "TCS": TC + "c3",
                     "tails": {t: TC + "c3" + SUF[t] for t in TAILS}}
    S = out["S"]
    S["L75"], S["L50"] = F["T_S"] + "adsbgeondctxv3x50ws75m4F", F["T_S"] + "adsbgeondctxv3x50wm4F"
    S["merge"] = S["TCS"] + "rBbF"
    S["merge_tails"] = {t: S["merge"] + SUF[t] for t in TAILS}
    S["fj_blend"] = {"tag": "l3sapcbrmfw4" if f == "598c" else "l4sapcbrmfw4", "base": L["lgbap"]["fj_avg"]["new"], "learners": fj_learners}
    out["names"] = {"603": F["final_A"]["default"] + "av" + SUFFIX, "604": "Jan/Jul rows of 603 in 597's form (research/f3/daily_candidate.py)",
                    "605c": F["final_A"]["c"] + "av" + SUFFIX}
    return out


def base_of(f: str, leg: str, setting: str, fj: bool = False) -> tuple[str | None, str | None]:
    """(--base value, link mode) of a new seed: 'B0' = <root>B0 with links <S>_<root>B0_s<k> -> <S>_<root>_s0; 'seedk' = <root> itself."""
    L = fam(f)["legs"][leg]
    b = L["fj"]["base"] if fj else (L["base"][setting] if L["base"] else None)
    if not b:
        return None, None
    root, mode = b
    return (root + "B0" if mode == "B0" else root), mode


def cmd(f: str, leg: str, setting: str, seed: int) -> str:
    L = fam(f)["legs"][leg]
    fj = setting == "fj"
    script = L["fj"]["script"] if fj else L["script"]
    tag = L["fj"]["tag"] if fj else L["tag"]
    args = L["fj"].get("args", L["args"]) if fj else L["args"]
    b, _ = base_of(f, leg, setting, fj)
    return f"{script} --setting {setting} --seed {seed} --tag {tag}" + (f" --base {b}" if b else "") + f" {args}"


def jobs(f: str, settings: list[str], legs: list[str] | None = None) -> list[dict]:
    """One job per (leg, seed, setting), in run order: LightGBM ap first, then CatBoost ap, then RealMLP (alone on the Mac)."""
    F = fam(f)
    out = []
    for leg in ORDER:
        if legs and leg not in legs:
            continue
        L = F["legs"][leg]
        for k in L["seeds"]:
            for s in settings:
                if s in L.get("existing", {}):
                    continue  # R13-F3's files (598c CatBoost ap val and fwd seeds 1, 2): cloned and checked by s_chain.py prep, not refit
                fj = s == "fj"
                tag = L["fj"]["tag"] if fj else L["tag"]
                b, mode = base_of(f, leg, s, fj)
                links = []
                if mode == "B0":
                    links.append((f"cache/{s}_{b}_s{k}.parquet", f"{s}_{b[:-2]}_s0.parquet"))
                needs = [f"cache/{s}_{b}_s{k}.parquet"] if mode == "seedk" else []
                thr = L["fj"].get("thr", L["thr"]) if fj else L["thr"]
                out.append({"name": f"{leg}_{s}_s{k}", "leg": leg, "kind": L["kind"], "setting": s, "seed": k, "tag": tag, "threads": thr,
                            "peak_gb": L["gb"], "mins": L["mins"][s], "out": f"cache/{s}_{tag}_s{k}.parquet", "links": links, "needs": needs,
                            "root": (L["fj"]["root"] if fj else ("sandbox" if s in ("val", "fwd") else "final_run")),
                            "launch": L["fj"].get("launch") if fj else None,
                            "cmd": cmd(f, leg, s, k), "ref": L["ref"][s][0], "ref_src": L["ref"][s][1]})
    return out


def avg_cmds(f: str, settings: str, fj: bool = False, rm_seeds: list[int] | None = None) -> list[str]:
    F = fam(f)
    out = []
    for leg in ORDER:
        L = F["legs"][leg]
        a = L["fj_avg"] if fj else L["avg"]
        sfile = L["fj"]["tag"] if fj else L["tag"]
        seeds = a["seeds"]
        new = a["new"]
        if leg == "rm" and rm_seeds:
            seeds, new = rm_seeds, rm_tag(f, rm_seeds, fj)
        out.append(f"{AVG} {a['mode']} {a['donor']} {sfile} {new} {','.join(map(str, seeds))} {settings}")
    return out


def joblist(f: str) -> str:
    F = fam(f)
    n = chain_names(f)
    S = n["S"]
    L = F["legs"]
    fc_t = f"research/f3/fitcache_s{f}"
    fc_4 = f"research/f2/fitcache_4m_s{f}"
    lines = [f"Package S final-root job list, family {f} = {F['label']}", f"(research/f2/s_spec.py; names: 603 = S default, 604 = its Jan/Jul twin, 605c = S final)", ""]
    lines += ["0. Clone into final_run (cp -cp, sha16 recorded, refuse to overwrite; never a full cache, never a link) from the S sandbox "
              f"~/nova-adsb/prcm/{F['sbx']} after its gate: the list research/f2/out/s_export_{f}.json (val and fwd of the new seed files and their "
              "es files, the three averaged legs, the S chain C blend, its corrector seeds 0-2 and c3 with the c3 es files), research/f2/avg_s.py, "
              f"and its corrector fit cache {FITCACHE_SBX} twice: to {fc_t} (T corrector) and to {fc_4} (four-month stage 1).", ""]
    lines.append("1. Full learners (one heavy job at a time; RealMLP alone, OMP_NUM_THREADS=1, PYTHONFAULTHANDLER=1; nice 10):")
    for j in jobs(f, ["full"]):
        ln = f"   {j['name']}: {j['cmd']}"
        if j["links"]:
            ln += "   [link " + ", ".join(f"{a} -> {b}" for a, b in j["links"]) + "]"
        if j["needs"]:
            ln += "   [reads " + ", ".join(j["needs"]) + "]"
        lines.append(ln)
    lines.append("   iteration files read by full: cache/iters_m6r28skap.json (LightGBM ap), cache/iters_" + L["cbap"]["tag"] + ".json (CatBoost ap): the val seed-0 runs', already in final_run")
    lines.append("2. Full averages (python -X utf8, cwd final_run):")
    lines += [f"   {c}" for c in avg_cmds(f, "full")]
    lines.append("3. Chain C, T form (final_chain.py's steps and arguments; val and fwd are the clones of step 0):")
    lines.append(f"   blend full: src/blend_fixed.py --setting full --seed 0 --tag {S['T']} --base {S['base']} --learners {S['learners']} {REF}")
    lines.append("     (no --check in full, as final_chain.py; the XGBoost slot is final_run's zero placeholder, weight exactly 0)")
    for k in (0, 1, 2):
        lines.append(f"   corrector full s{k}: src/ctx_corr3.py --setting full --seed {k} --base {S['T']} --base_seed 0 --tag {S['TC']} {CORR_FINAL} --fit_cache {fc_t}"
                     "   -> expect 'fit cache ...: N loaded, 0 fitted' (any fitted model stops the chain)")
    lines.append(f"   cavg full: research/r10_b/cavg.py --setting full --seed 0 --tag {S['TCS']} --src {S['TC']} --seeds 0,1,2 --base {S['T']} --base_seed 0")
    lines.append(f"   manifest: -m src.manifest {S['TCS']} {S['TCS']} pred 0 {F['feats']} notes/jobs_{S['TCS']}.txt   (MANIFEST_EXTRA_CODE = the family chain C list + {AVG})")
    lines.append(f"   tails am .. gja with final_chain.py's tool calls (gkey --leg {F['GLEG']} --leg_feats {F['feats']}), specs " + S["tails"]["gja"].replace(S["TCS"], "<TCS>") + " etc.")
    lines.append(f"   603: tools/chainavg3.py build <A default gja queue name: ...{F['final_A']['default'][-40:]}> <B default gja> <C S gja> <num> --suffix {SUFFIX} --no-enqueue")
    lines.append(f"        spec {n['names']['603']}")
    lines.append("   604: research/f3/daily_candidate.py (unchanged, its sha16 asserted) on 603, 597's form, a new name prefix; staged to research/f3/queue_ready_s/")
    lines.append("4. fj learners (after 603 and 604 are staged; chain C at the package seed counts):")
    for j in jobs(f, ["fj"]):
        ln = f"   {j['name']} [root {j['root']}]: {j['cmd']}"
        if j["links"]:
            ln += "   [link " + ", ".join(f"{a} -> {b}" for a, b in j["links"]) + "]"
        if j["needs"]:
            ln += "   [reads " + ", ".join(j["needs"]) + "]"
        lines.append(ln)
    lines.append("   fj averages:")
    lines += [f"   {c}" for c in avg_cmds(f, "fj", fj=True)]
    fb = S["fj_blend"]
    lines.append(f"   fj blend: research/f2/fj_blend_s.py {f}  -> cache/fj_{fb['tag']}_s0.parquet = research/f2/blend_fixed_fj.py --setting fj --seed 0 --tag {fb['tag']} "
                 f"--base {fb['base']} --learners {fb['learners']} {REF} (+ the 1,000 s neutrality check of the XGBoost slot; pin research/f2/out/s_fj_sha16_{f}.txt)")
    lines.append(f"5. Chain C, c form ({F['kit']}'s corr4m, merge, manifest, tails with chain C = {S['T']}, extra = {fb['tag']} and its pinned sha16, fit cache {fc_4}):")
    lines.append(f"   corr4m 0.75 seeds 0-2 tag {S['L75']}: expect 10 fitted per seed (the S_k with the extra rows; stage 1 loads); 0.5 seeds 0-2 tag {S['L50']}: expect 0 fitted")
    lines.append(f"   cavg {S['L75']}c3, {S['L50']}c3; twins check; merge {S['merge']} (jj T 0.75 | fu L 0.75 | fm L 0.5); manifest; tails -> {S['merge_tails']['gja']}")
    lines.append(f"   605c: tools/chainavg3.py build <A c gja> <B c gja> <C S c gja> <num> --suffix {SUFFIX} --no-enqueue -> {n['names']['605c']}")
    lines.append("6. Checks: research/f2/s_check.py (construction, full and fj legs), research/f2/s_gate.py move (605c vs the family's c final, MSE <= 150), "
                 "the family's final checks (iii boundary and the rest) as F3 runs them for any final.")
    lines.append("Fit counts on the final root: T corrector 0 per seed; four-month 0.75 run 10 per seed; 0.5 run 0; twin 0. Any other count stops the chain.")
    if f == "598c":
        lines.append("598c fj CatBoost ap and RealMLP seeds (coordinator 10-11 02:2x): in f2_run, where their seed 0 ran, by F2 with research/f2/s_learn.py 598c fj "
                     f"--legs cbap,rm --rm_seeds_from {SBX_LEARN_598C} (the sandbox's finished RealMLP seeds; the same command with --dry first, "
                     "07:4x), launched as seed 0 was ($PY -u; RealMLP env OMP_NUM_THREADS=1), after research/f2/s_fjpre.py recorded and asserted that "
                     "the learner scripts and their src modules, feats_v27, the bases fj_f2skap_s0 and fj_f2sk_s0 and the B0 links are what seed 0 used "
                     "(its job lines in logs/f2_fjlegs_*.log, mtimes before its START, the seed-0 outputs' base columns bit-equal to the bases). F3 then clones "
                     "cache/fj_f2cbap_s{1,2}, fj_f2rm_s{4..7} and their preds/es_valid_ids files into final_run, checking each sha16 against f2_run's "
                     "research/f2/out/s_learn_598c_fj.json; the fj averages and the fj blend run on final_run.")
    return "\n".join(lines)


def main() -> int:
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "names":
        print(json.dumps({"names": chain_names(a[1]), "legs": fam(a[1])["legs"], "placeholders": fam(a[1])["placeholders"]}, indent=1))
        return 0
    if len(a) >= 3 and a[0] == "jobs":
        for j in jobs(a[1], a[2].split(","), a[3].split(",") if len(a) > 3 else None):
            print("\t".join([j["name"], j["kind"], str(j["threads"]), str(j["peak_gb"]), str(j["mins"]), j["out"],
                             ";".join(f"{x}>{y}" for x, y in j["links"]) or "-", ";".join(j["needs"]) or "-", j["root"], j["cmd"]]))
        return 0
    if len(a) >= 2 and a[0] == "joblist":
        print(joblist(a[1]))
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
