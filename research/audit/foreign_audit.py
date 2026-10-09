# SPDX-License-Identifier: GPL-3.0-only
"""Foreign-code audit (read-only). Compares saved copies of other teams' code with this repository's own Python files.

Methods (all stdlib, no execution of the audited files):
  1. function-level AST hashes, exact and name-anonymised, for functions with >= 4 body statements
  2. normalised token 10-gram shingles: containment of each foreign file in each of our files
  3. distinctive identifiers (>= 10 chars, contain "_") and string literals (>= 14 chars) shared with foreign files
Usage: python -I foreign_audit.py <repo root> <out json>
"""
import ast, hashlib, io, json, keyword, os, re, sys, tokenize
from collections import defaultdict

ROOT = os.path.abspath(sys.argv[1])
OUT = sys.argv[2]

FOREIGN_FILES = [
    "research/r10_a/web/gh_arnavhm_prc_26/analysis_build_pru_calibration.py",
    "research/r10_a/web/gh_arnavhm_prc_26/analysis_oa10_stage1_427.py",
    "research/r10_a/web/gh_arnavhm_prc_26/analysis_oa11_position.py",
    "research/r10_a/web/gh_niclaswue_ATOW-Prediction_download_additional_data.py",
    "research/r14_c/web/ravindra_priors.py",
]
FOREIGN_MD = [  # saved markdown of other teams: fenced code blocks are extracted
    "research/public_km/README.md", "research/public_km/RECAP.md", "research/public_km/MODEL_ANALYSIS.md",
    "research/public_km/WINNING_PLAN.md",
    "research/r10_a/web/gh_arnavhm_prc_26/README.md", "research/r10_a/web/gh_arnavhm_prc_26/artifacts_experiments_log.md",
    "research/r10_a/web/gh_arnavhm_prc_26/PRC2026_CONTEXT.md", "research/r10_a/web/gh_arnavhm_prc_26/findings.md",
    "research/r10_a/web/gh_arnavhm_prc_26/docs_code_review_2026-10-02.md",
    "research/r10_a/web/gh_elna4os_prc_2026_README.md", "research/r10_a/web/gh_joris-decombe_taxiout_README.md",
    "research/r10_a/web/gh_joris-decombe_taxiout_RESEARCH.md", "research/r10_a/web/gh_niclaswue_ATOW-Prediction_README.md",
    "research/r14_c/web/readme_joris-decombe_taxiout.md", "research/r14_c/web/readme_dhsdb010_OPDI.md",
]
SKIP_DIRS = {".venv", "__pycache__", "web", ".git", "node_modules", "gpu_run", "_work", "_release_stage", "docs_release", "_backup", "_mac_stage", "cache", "data", "logs", "preds", "subs", "queue", "notes"}


def read(p):
    with open(p, "rb") as f:
        return f.read().decode("utf-8", "replace")


def our_files():
    res = []
    for top in ("src", "tools", "research"):
        for dp, dn, fn in os.walk(os.path.join(ROOT, top)):
            dn[:] = [d for d in dn if d not in SKIP_DIRS]
            for n in fn:
                if n.endswith(".py"):
                    rel = os.path.relpath(os.path.join(dp, n), ROOT).replace("\\", "/")
                    if rel in FOREIGN_FILES:
                        continue
                    res.append(rel)
    return sorted(res)


class Anon(ast.NodeTransformer):
    def visit_Name(self, n):
        return ast.copy_location(ast.Name(id="_", ctx=n.ctx), n)

    def visit_arg(self, n):
        return ast.copy_location(ast.arg(arg="_", annotation=None), n)

    def visit_FunctionDef(self, n):
        self.generic_visit(n)
        n.name = "_"
        return n


def strip_doc(fn):
    b = fn.body
    if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) and isinstance(b[0].value.value, str):
        fn.body = b[1:] or [ast.Pass()]
    return fn


def func_hashes(src):
    out = []
    try:
        tree = ast.parse(src)
    except Exception:
        return out
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nstmt = sum(isinstance(x, ast.stmt) for x in ast.walk(node))
            if nstmt < 5:
                continue
            import copy
            n1 = strip_doc(copy.deepcopy(node))
            n1.name = "_"
            exact = hashlib.sha256(ast.dump(n1, annotate_fields=False).encode()).hexdigest()[:16]
            n2 = Anon().visit(strip_doc(copy.deepcopy(node)))
            anon = hashlib.sha256(ast.dump(n2, annotate_fields=False).encode()).hexdigest()[:16]
            out.append((node.name, nstmt, exact, anon))
    return out


def shingles(src, k=10):
    toks = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
                continue
            if t.type == tokenize.STRING:
                toks.append("S")
            elif t.type == tokenize.NUMBER:
                toks.append("N")
            elif t.type == tokenize.NAME:
                toks.append(t.string if keyword.iskeyword(t.string) else "I")
            else:
                toks.append(t.string)
    except Exception:
        pass
    return {hash(tuple(toks[i:i + k])) for i in range(max(0, len(toks) - k + 1))}


def idents(src):
    return set(re.findall(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]{5,}\b", src))


def strings(src):
    out = set()
    try:
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) >= 14 and "\n" not in node.value:
                out.add(node.value)
    except Exception:
        pass
    return out


def md_blocks(path):
    txt = read(path)
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", txt, re.S)
    return [b for b in blocks if len(b.splitlines()) >= 5]


foreign = {}  # name -> source
for rel in FOREIGN_FILES:
    p = os.path.join(ROOT, rel)
    if os.path.exists(p):
        foreign[rel] = read(p)
for rel in FOREIGN_MD:
    p = os.path.join(ROOT, rel)
    if os.path.exists(p):
        for i, b in enumerate(md_blocks(p)):
            foreign[f"{rel}#block{i}"] = b

ours = {rel: read(os.path.join(ROOT, rel)) for rel in our_files()}
print("foreign sources:", len(foreign), " our files:", len(ours))

# 1. function hashes
fh_exact, fh_anon = defaultdict(list), defaultdict(list)
for name, src in foreign.items():
    for fn, n, ex, an in func_hashes(src):
        fh_exact[ex].append((name, fn, n)); fh_anon[an].append((name, fn, n))
res = {"foreign_sources": len(foreign), "our_files": len(ours), "foreign_functions": sum(len(v) for v in fh_exact.values()), "func_matches": [], "shingle": [], "ident": {}, "strings": {}}
for rel, src in ours.items():
    for fn, n, ex, an in func_hashes(src):
        if ex in fh_exact:
            res["func_matches"].append({"ours": rel, "func": fn, "stmts": n, "kind": "exact", "foreign": fh_exact[ex]})
        elif an in fh_anon:
            res["func_matches"].append({"ours": rel, "func": fn, "stmts": n, "kind": "anonymised", "foreign": fh_anon[an]})

# 2. shingles
fs = {name: shingles(src) for name, src in foreign.items()}
for rel, src in ours.items():
    s_o = shingles(src)
    if not s_o:
        continue
    for name, s_f in fs.items():
        if len(s_f) < 40:
            continue
        inter = len(s_f & s_o)
        cont = inter / len(s_f)
        if inter >= 25 and cont >= 0.04:
            res["shingle"].append({"ours": rel, "foreign": name, "shared10grams": inter, "foreign_total": len(s_f), "containment_of_foreign": round(cont, 3), "ours_total": len(s_o), "containment_of_ours": round(inter / len(s_o), 3)})
res["shingle"].sort(key=lambda d: -d["containment_of_foreign"])

# 3. identifiers and strings
f_id = defaultdict(set)
f_st = defaultdict(set)
for name, src in foreign.items():
    for i in idents(src):
        f_id[i].add(name)
    for s in strings(src):
        f_st[s].add(name)
o_id = defaultdict(set)
o_st = defaultdict(set)
for rel, src in ours.items():
    for i in idents(src):
        if i in f_id:
            o_id[i].add(rel)
    for s in strings(src):
        if s in f_st:
            o_st[s].add(rel)
res["ident"] = {i: {"foreign": sorted(f_id[i])[:3], "ours": sorted(v)[:5], "n_ours": len(v)} for i, v in o_id.items()}
res["strings"] = {s: {"foreign": sorted(f_st[s])[:3], "ours": sorted(v)[:5], "n_ours": len(v)} for s, v in o_st.items()}
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("function matches:", len(res["func_matches"]), " shingle pairs:", len(res["shingle"]), " shared idents:", len(res["ident"]), " shared strings:", len(res["strings"]))
