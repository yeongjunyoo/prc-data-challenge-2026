# SPDX-License-Identifier: GPL-3.0-only
"""R17-A release fixer on the Mac (private notes research/2026-10-08-r17-a-adsb-fill.md section 4.1). When a day's first try fails the completeness check
(${ADSB_WORK:-.}/failed/<day>.1/meta_incomplete.json: a broken archive, or a release whose asset list mixes a whole .tar with split parts), it
rewrites that day's line of ${ADSB_WORK:-.}/manifest_preferred.tsv to the other full instance of the same day (prod-0 <-> staging-0, never
mlatonly), taking the asset names and sizes from research/adsb/rel_assets_globe_history_<year>.tsv, so the runner's next try reads the
alternative (the adsb.lol README allows prod or staging for a day). It only switches when the alternative is not much smaller (>= 0.9 x the
failed release, or the failed one mixed .tar and split parts) and switches a day once. Every switch is logged to logs/fixer.log.
    nohup ${PY_ADSB:-python3} -I fixer_r17a.py >> logs/fixer.log 2>&1 < /dev/null &      (from ${ADSB_WORK:-.}; stops at STOP_FIXER)
"""
import json
import os
import time

HOME = os.environ.get("ADSB_WORK", ".")  # Mac ADS-B download folder
MAN = os.path.join(HOME, "manifest_preferred.tsv")


def ts():
    return time.strftime("%m-%d %H:%M:%S")


def assets():
    out = {}
    for y in ("2025", "2026"):
        for ln in open(os.path.join(HOME, f"rel_assets_globe_history_{y}.tsv"), encoding="utf-8"):
            tag, repo, names, sizes, _ = ln.rstrip("\n").split("\t")[:5]
            day = tag[1:11].replace(".", "-")
            inst = tag.split("-planes-readsb-")[1]
            out.setdefault(day, {})[inst] = (tag, repo, names, sizes)
    return out


def main():
    A = assets()
    done = set()
    print(f"{ts()} fixer start pid {os.getpid()}", flush=True)
    while not os.path.exists(os.path.join(HOME, "STOP_FIXER")):
        for f in sorted(os.listdir(os.path.join(HOME, "failed"))):
            if not f.endswith(".1") or f in done:
                continue
            day = f[:-2]
            done.add(f)
            lines = open(MAN, encoding="utf-8").read().splitlines()
            idx = [i for i, ln in enumerate(lines) if ln.split("\t")[0] == day]
            if not idx:
                print(f"{ts()} {day}: not in the manifest", flush=True)
                continue
            p = lines[idx[0]].split("\t")
            mi0 = os.path.join(HOME, "failed", f, "meta_incomplete.json")
            if os.path.exists(mi0) and not any(f"/{p[1]}/" in u for u in json.load(open(mi0)).get("parts", [])):
                print(f"{ts()} {day}: the manifest already moved off the failed release (now {p[1]}); left", flush=True)
                continue
            cur = p[1].split("-planes-readsb-")[1]
            names = p[3].split(",")
            mixed = any(n.endswith(".tar") for n in names) and any(".tar.a" in n for n in names)
            alt_inst = "prod-0" if cur.startswith("staging") else "staging-0"
            alt = A.get(day, {}).get(alt_inst)
            if alt is None:
                print(f"{ts()} {day}: no {alt_inst} release; left as {p[1]}", flush=True)
                continue
            an = alt[2].split(",")
            if any(n.endswith(".tar") for n in an) and any(".tar.a" in n for n in an):
                print(f"{ts()} {day}: {alt_inst} also mixes .tar and split parts; left as {p[1]}", flush=True)
                continue
            alt_total = sum(int(s) for s in alt[3].split(","))
            if not mixed and alt_total < 0.9 * int(p[5]):
                print(f"{ts()} {day}: {alt_inst} {alt_total} B is much smaller than {p[1]} {p[5]} B; left", flush=True)
                continue
            mi = os.path.join(HOME, "failed", f, "meta_incomplete.json")
            why = ""
            if os.path.exists(mi):
                m = json.load(open(mi))
                why = f"curl_rc {m.get('curl_rc')}, bytes {m.get('bytes_read')}/{m.get('bytes')}, tar end {m.get('tar_end_offset')}, bad members {m.get('bad_members')}"
            lines[idx[0]] = "\t".join([day, alt[0], alt[1], alt[2], alt[3], str(alt_total)])
            tmp = MAN + ".tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("\n".join(lines) + "\n")
            os.replace(tmp, MAN)
            print(f"{ts()} {day}: switched {p[1]} -> {alt[0]} ({alt_total} B){' (mixed assets)' if mixed else ''}; failed try: {why}", flush=True)
        time.sleep(20)
    print(f"{ts()} fixer stop", flush=True)


if __name__ == "__main__":
    main()
