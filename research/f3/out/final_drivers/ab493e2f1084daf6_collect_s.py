"""F3 Package S, final root (coordinator 10-11 01:17 and 01:2x; F2's job list 02:10): clone from F2's S sandbox, after its gate, everything the
final root's chain C at the package seed counts reads besides the full and fj fits F3 makes here. The list is the sandbox's own export,
research/f2/out/s_export_<family>.json (research/f2/s_chain.py export): val and fwd of the new seed files and their es files, the three
averaged legs, the S chain C blend and its notes, the corrector seeds 0-2 and c3 with their es files, and F2's S tools (avg_s.py, s_spec.py,
s_check.py, s_gate.py, fj_blend_s.py, s_learn.py, rm_smoke.py). Every file: cp -cp, sha16 equal to the export's record (source and copy),
never a link, never a full cache, refuse to overwrite (a target that exists with the same bytes is recorded as such). The corrector fit
cache research/f2/fitcache_s goes twice: to research/f3/fitcache_s<family> (T corrector) and research/f2/fitcache_4m_s<family> (four-month
stage 1), each refused if it exists, file count equal to the export's. Writes research/f3/out/collect_s_<family>.json (the export's sha16,
rm_seeds and names, every file's sha16), which research/f3/final_chain_s.py, kit_s.sh and phase8.sh read. Copies nothing unless the sandbox's
research/f2/out/s_verdict_sbx_<family>.json is PASS on every input, s_gate_sbx_<family>.json exists, passes and is one of those inputs, and
s_chain_<family>.FAILED and s_run_<family>.FAILED are absent (the verdict and gate sha16 and the gate's numbers go into the record).
    python -X utf8 research/f3/collect_s.py <598c|l4>        (cwd = the final root)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

P = Path.home() / "nova-adsb" / "prcm"
D = P / "final_run"
SBX = {"598c": "f2s_sbx598", "l4": "f2s_sbxl4"}


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    fam = sys.argv[1] if len(sys.argv) > 1 else ""
    if fam not in SBX:
        raise SystemExit(__doc__)
    S = P / SBX[fam]
    ep = S / "research" / "f2" / "out" / f"s_export_{fam}.json"
    E = json.loads(ep.read_text(encoding="utf-8"))
    assert E.get("family") == fam, f"export family {E.get('family')} is not {fam}"
    assert not E.get("missing"), f"the export lists missing files: {E.get('missing')}"
    assert E.get("files") and "?" not in E["files"].values(), "the export has no sha16 record (a dry export?)"
    # a FAIL verdict blocks the final root (review 10-11): the sandbox's verdict must be PASS on every input, its gate json must exist and
    # pass and be one of those inputs, and the sandbox chain must have no FAILED marker; nothing is copied otherwise
    so = S / "research" / "f2" / "out"
    vj, gj, fl, rf = so / f"s_verdict_sbx_{fam}.json", so / f"s_gate_sbx_{fam}.json", so / f"s_chain_{fam}.FAILED", so / f"s_run_{fam}.FAILED"
    assert vj.exists() and gj.exists(), f"the sandbox gate records are missing: {vj.name} {vj.exists()}, {gj.name} {gj.exists()}"
    V, G = json.loads(vj.read_text(encoding="utf-8")), json.loads(gj.read_text(encoding="utf-8"))
    assert V.get("pass") is True and V.get("adopt") is True and V.get("inputs") and all(V["inputs"].values()), \
        f"the sandbox verdict is not PASS: {vj} {V.get('inputs')}"
    assert any(Path(k).name == gj.name for k in V["inputs"]), f"the gate json {gj.name} is not among the verdict's inputs: {list(V['inputs'])}"
    assert G.get("pass") is True and G.get("rules") and all(G["rules"].values()), f"the sandbox gate is not PASS: {gj} {G.get('rules')}"
    assert not fl.exists(), f"the sandbox chain has a FAILED marker: {fl}"
    assert not rf.exists(), f"the sandbox run has a FAILED marker: {rf}"
    assert E.get("verdict") == f"research/f2/out/{vj.name}", f"the export does not name this verdict: {E.get('verdict')}"
    rep = {"from": str(S), "export": str(ep), "export_sha16": sha16(ep), "family": fam, "rm_seeds": E.get("rm_seeds"), "names": E.get("names"),
           "verdict": {"file": f"research/f2/out/{vj.name}", "sha16": sha16(vj), "pass": True, "inputs": V["inputs"]},
           "gate": {"file": f"research/f2/out/{gj.name}", "sha16": sha16(gj), "rules": G["rules"],
                    "numbers": {st: {k: (G.get(st) or {}).get(k) for k in ("dmse", "boot_sd", "body_nonG", "p_below0", "body_p_below0")} for st in ("val", "fwd")}},
           "copied": {}, "same_bytes_existing": {}, "fitcache": {}}
    todo = []
    for rel, h in E["files"].items():
        s, d = S / rel, D / rel
        assert not rel.startswith("cache/full_"), f"the export lists a full cache: {rel}"
        assert s.exists() and not s.is_symlink(), f"missing or a link in the sandbox: {s}"
        assert sha16(s) == h, f"{rel}: sandbox file {sha16(s)} is not the export's {h}"
        if d.exists() or d.is_symlink():
            assert not d.is_symlink() and sha16(d) == h, f"exists with other content (or a link): {d}"
            rep["same_bytes_existing"][rel] = h
            continue
        todo.append((rel, s, d, h))
    fc_src = S / E["fitcache"]["dir"]
    fc_dst = [D / x for x in E["fitcache"]["to"]]
    assert [x.relative_to(D).as_posix() for x in fc_dst] == [f"research/f3/fitcache_s{fam}", f"research/f2/fitcache_4m_s{fam}"], E["fitcache"]["to"]
    models = sorted(x for x in fc_src.glob("*") if x.is_file())
    assert len(models) == E["fitcache"]["files"] and models, f"fit cache {fc_src}: {len(models)} files, the export says {E['fitcache']['files']}"
    for x in fc_dst:
        assert not x.exists(), f"exists: {x}"
    for rel, s, d, h in todo:
        d.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-cp", str(s), str(d)], check=True)
        assert sha16(d) == h, f"copy of {rel} differs"
        rep["copied"][rel] = h
    for x in fc_dst:
        x.mkdir(parents=True)
        for f in models:
            subprocess.run(["cp", "-cp", str(f), str(x / f.name)], check=True)
        rep["fitcache"][x.relative_to(D).as_posix()] = {"from": str(fc_src), "files": len(models)}
    rep["tool_sha16"] = {f: sha16(D / f) for f in ("research/f2/avg_s.py", "research/f2/s_spec.py", "research/f2/s_check.py", "research/f2/s_gate.py",
                                                   "research/f2/fj_blend_s.py", "research/f2/s_learn.py", "research/f2/rm_smoke.py")}
    out = D / "research" / "f3" / "out" / f"collect_s_{fam}.json"
    assert not out.exists(), f"exists: {out}"
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"{fam}: {len(rep['copied'])} files cloned, {len(rep['same_bytes_existing'])} already there with the same bytes, fit cache {len(models)} "
          f"files twice; rm_seeds {rep['rm_seeds']}; s_spec.py {rep['tool_sha16']['research/f2/s_spec.py']}")


if __name__ == "__main__":
    main()
