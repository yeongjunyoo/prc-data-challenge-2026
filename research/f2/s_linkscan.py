# SPDX-License-Identifier: GPL-3.0-only
"""F2 Package S: the link scan of a sandbox clone (10-11 03:1x fix 5: refuse any link into a live root, or record each one
with a reason why no write can reach through it). Reads <dir> only and never follows a link while walking it. Every symbolic link, file
or directory, is resolved as it resolves in the clone: --as <clone> when <dir> is the source before cloning (a relative link is taken
from its place in the clone; an absolute link points where it points); without --as, <dir> is the clone, and the real path is checked
as well. Classes:
  inside   resolves inside the clone: fine.
  live     resolves into <mac-workdir>/prcm itself or into another root there (final_run, r17a_sbx, r17a_l4g_sbx, f2_run, r10b_run, r13f3_run,
           or any other): a write through it (a tool that opens a fixed name, CatBoost's catboost_info, a log, a new file in a linked
           directory) would land in that root. Refused unless --allow names the link with its exact link text and a reason, and its
           target exists (through a dangling link a writer that first checks exists() would create the file in that root).
  outside  resolves outside <mac-workdir>/prcm (shared extracts, the venv): recorded with its target; not a root, not refused.
Separately (fix B), a link at a directory the run writes into (WRITE_DIRS: cache, preds, notes, logs, queue, research/f2 and the rest) or
at any ancestor of one, or <dir> itself being a link, is refused whatever its class: the clone would write into the source's directory.
Writes <out json> (the links of each class, every allowed one with its reason); exit 1 if any live link is not allowed, a write directory
is a link, or a directory could not be read.
    python -X utf8 research/f2/s_linkscan.py <dir> <out json> [--as <clone dir>] [--allow <allow json>]
    allow json: {"links": {"<path in the clone>": {"link": "<link text as os.readlink gives it>", "reason": "<why no write reaches through>"}}}
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

MARK = "/prcm"  # the name of the Mac work folder (its full path is not published)
FOLD = sys.platform in ("darwin", "win32")  # APFS and NTFS compare names without case
# Directories the S tools, the chain tools, the learners and Python's import system write into (03:5x fix B). A link at one of
# them or at any ancestor, whatever it points at and whatever its class, makes the clone share that directory with the source: refused,
# and no allow entry passes it. The scanned directory itself must not be a link either.
WRITE_DIRS = ["cache", "preds", "preds/code", "notes", "logs", "queue", "src", "tools", "research/f2", "research/f2/out", "research/f2/mac",
              "research/f2/fitcache_s"]


def write_dir_links(d: Path) -> list[str]:
    bad = ["."] if d.is_symlink() else []
    seen: set[str] = set()
    for w in WRITE_DIRS:
        parts = w.split("/")
        for i in range(1, len(parts) + 1):
            rel = "/".join(parts[:i])
            if rel not in seen:
                seen.add(rel)
                if (d / rel).is_symlink():
                    bad.append(rel)
    return bad


def norm(p: str) -> str:
    q = os.path.normpath(p).replace("\\", "/")
    return q.lower() if FOLD else q


def under(p: str, root: str) -> bool:
    return p == root or p.startswith(root.rstrip("/") + "/")


def classify(target: str, clone: str) -> tuple[str, str | None]:
    t, c = norm(target), norm(clone)
    if under(t, c):
        return "inside", None
    i = t.find(MARK)
    if i >= 0 and (len(t) == i + len(MARK) or t[i + len(MARK)] == "/"):
        rest = t[i + len(MARK) + 1:]
        root = rest.split("/", 1)[0] if rest else ""
        if root and root == c.rsplit("/", 1)[-1]:  # the clone under another spelling of its path (/System/Volumes/Data/Users/...)
            return "inside", None
        return "live", root or "prcm itself"
    return "outside", None


def scan(d: Path, clone: Path, allow: dict) -> dict:
    out = {"dir": str(d), "as": str(clone), "links": 0, "inside": 0, "live": [], "outside": [], "refused": [], "unreadable": [],
           "write_dir_links": write_dir_links(d)}
    real_check = norm(str(d)) == norm(str(clone))
    for top, dirs, files in os.walk(d, followlinks=False, onerror=lambda e: out["unreadable"].append(f"{e.filename}: {e.strerror}")):
        for name in sorted(dirs) + sorted(files):
            p = Path(top) / name
            if not p.is_symlink():
                continue
            out["links"] += 1
            rel = p.relative_to(d).as_posix()
            txt = os.readlink(p)
            here = clone / Path(rel).parent
            lexical = txt if os.path.isabs(txt) else os.path.join(str(here), txt)
            kind = "dir" if p.is_dir() else ("file" if p.exists() else "dangling")
            cls, root = classify(lexical, str(clone))
            real = None
            if real_check:  # in the clone: the path as the OS resolves it (links inside links) must agree
                real = os.path.realpath(p)
                c2, r2 = classify(real, str(clone))
                if c2 == "live" or (c2 == "outside" and cls == "inside"):
                    cls, root = c2, r2
            rec = {"path": rel, "link": txt, "target": norm(lexical), "real": norm(real) if real else None, "kind": kind}
            if cls == "inside":
                out["inside"] += 1
            elif cls == "outside":
                out["outside"].append(rec)
            else:
                a = allow.get(rel)
                ok = bool(a) and a.get("link") == txt and bool(str(a.get("reason", "")).strip()) and kind != "dangling"
                why = None if ok else ("not in the allow list" if not a else "allow entry has another link text or no reason" if kind != "dangling"
                                       else "its target is missing: a writer that checks exists() would create the file in the live root")
                rec.update(root=root, allowed=ok, reason=a.get("reason") if ok else None, why_refused=why)
                out["live"].append(rec)
                if not ok:
                    out["refused"].append(rel)
    out["pass"] = not out["refused"] and not out["unreadable"] and not out["write_dir_links"]  # unread = unchecked
    return out


def main() -> int:
    a = sys.argv[1:]
    if len(a) < 2:
        raise SystemExit(__doc__)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    d, outp = Path(a[0]), Path(a[1])
    clone = Path(a[a.index("--as") + 1]) if "--as" in a else d
    allow, allow_sha = {}, None
    if "--allow" in a:
        ap = Path(a[a.index("--allow") + 1])
        allow = json.loads(ap.read_text(encoding="utf-8"))["links"]
        allow_sha = hashlib.sha256(ap.read_bytes()).hexdigest()[:16]
    if not d.is_dir():
        raise SystemExit(f"no directory {d}")
    r = scan(d, clone, allow)
    r.update(allow=str(a[a.index("--allow") + 1]) if allow_sha else None, allow_sha16=allow_sha, written=time.strftime("%Y-%m-%d %H:%M:%S"))
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"s_linkscan {d} as {clone}: {r['links']} links, {r['inside']} inside, {len(r['live'])} into live roots "
          f"({len(r['live']) - len(r['refused'])} allowed with a reason), {len(r['outside'])} outside prcm -> {'PASS' if r['pass'] else 'FAIL'}"
          + (f"; refused {r['refused'][:10]}" if r["refused"] else "") + (f"; unreadable {r['unreadable'][:5]}" if r["unreadable"] else "")
          + (f"; write directories that are links (never allowed) {r['write_dir_links']}" if r["write_dir_links"] else ""), flush=True)
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
