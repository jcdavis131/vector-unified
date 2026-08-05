#!/usr/bin/env python3
"""A path literal in CODE must not name one particular computer. READ-ONLY.

Solo personal project, no connection to employer, built with public/free-tier only

portable_paths.py has defined `ESTATE = <repo>.parent` and `resolve()` for a while, and
`migrate_hub_portable_paths.py` already applied it to the PUBLISHED citations. But nothing
applied it to python source, and nothing checked, so 17 module-level constants across 14
files kept naming `C:/Users/jcdav/...` long after the convention existed. A convention with
no checker is a preference.

WHY IT MATTERS MORE THAN "it breaks on another box". Two of the seventeen crash loudly,
which is honest. Eight were guarded and degrade SILENTLY:

    build_vor_draft_value.py:169   if DRAFT_CSV.exists():      skips the draft-pick merge
                                                               and emits VOR values built
                                                               from no draft data
    validation_sweep.py:54         PY = str(VENV) if VENV.exists() else sys.executable
                                                               runs all five checks under a
                                                               different interpreter and
                                                               the log row never said which

That is this estate's recurring defect: a real value answering a different question than
the one it appears to answer. The path is real. It resolves — on one computer.

WHAT IT CHECKS. Every tracked `.py`, every string literal the AST can see, flagged when the
string STARTS with a drive letter or UNC root.

CALIBRATION, and each rule is a false positive it would otherwise make:

    comments            invisible to ast by construction, so `# WAS Path("C:/...")` —
                        which several files carry deliberately, recording the defect they
                        fixed — cannot be flagged. This is why the check is AST-based and
                        not a grep; a grep reports 29 hits here, of which 12 are prose.
    docstrings          collected by ast.get_docstring for module, class and function, then
                        excluded by node identity. portable_paths.py's own docstring lists
                        example laptop paths as the thing it exists to remove.
    strings that MENTION a path   check_gate_inputs_tracked.py stores the sentence
                        "HUB = Path('C:/Users/jcdav/...'), an absolute path into a sibling
                        repo" in its JSON output. Anchoring the match at the START of the
                        string separates a path from prose about a path.

NON-VACUITY IS TESTED, NOT ASSUMED. A previous static scanner in this repo passed green
because 18 of its 19 patterns matched nothing. So `--selftest` re-runs this checker against
the versions of the same files at a given commit and requires it to FIND the ones already
fixed. A checker that cannot find a known defect is not evidence of its absence.

    python pipeline/check_laptop_paths.py
    python pipeline/check_laptop_paths.py --check              # exit 1 on any finding
    python pipeline/check_laptop_paths.py --selftest 6ea5f21   # must find >=1 at that commit

Writes: data/laptop_paths_audit.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "laptop_paths_audit.json"

# ANCHORED AT THE START. `C:/x` is a path; "HUB = Path('C:/x')" is a sentence about one.
# Deliberately broad on the root itself — the rule is "no machine-local path in code", not
# "no path under C:/Users/jcdav" — matching portable_paths.ABS_RE's stated reasoning.
ABS_RE = re.compile(r"^(?:[A-Za-z]:[/\\]|\\\\[A-Za-z0-9_.-]+\\)")


# DELIBERATE FAULT INJECTION. check_guards_nonvacuous.py exists to prove the guards catch a
# laptop path, so it has to WRITE one:
#
#     doc["source_files"][0] = "C:/Users/jcdav/vector-unified/data/tennis_forward_report.json"
#
# That is a correct string in a correct file, and flagging it would be a false positive that
# teaches the reader to skim. Exempted BY FILE AND SUBSTRING rather than folded into a
# heuristic, so the exemption is arguable on sight and cannot quietly widen: if that file
# grows a second, accidental laptop path, only the injected one stays exempt.
ALLOW = {
    "pipeline/check_guards_nonvacuous.py": {
        "C:/Users/jcdav/vector-unified/data/tennis_forward_report.json":
            "injected on purpose to prove the guard is non-vacuous",
    },
}


def docstring_nodes(tree: ast.AST) -> set[int]:
    """id() of every Constant that is a docstring, so prose examples are not path claims."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def scan_source(rel: str, src: str) -> tuple[list[dict], int]:
    """-> (findings, string literals examined). The second number makes a vacuous run visible."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [], 0
    skip = docstring_nodes(tree)
    hits, examined = [], 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in skip:
            continue
        examined += 1
        if ABS_RE.match(node.value):
            if node.value in ALLOW.get(rel, {}):
                continue
            hits.append({
                "file": rel,
                "line": getattr(node, "lineno", 0),
                "literal": node.value[:160],
                "why": "a path literal in code that names one computer; use "
                       "portable_paths.ESTATE / \"vector-<repo>/...\" instead",
            })
    return hits, examined


def tracked_py() -> list[str]:
    r = subprocess.run(["git", "ls-files", "*.py"], cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def at_commit(rel: str, commit: str) -> str | None:
    r = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 on any finding")
    ap.add_argument("--selftest", metavar="COMMIT",
                    help="scan the same files as of COMMIT; exit 1 if it finds nothing, "
                         "which would mean this checker cannot detect the defect it exists for")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    files = tracked_py()

    if args.selftest:
        found, seen = 0, 0
        for rel in files:
            old = at_commit(rel, args.selftest)
            if old is None:
                continue
            seen += 1
            h, _ = scan_source(rel, old)
            found += len(h)
        print(f"  SELFTEST at {args.selftest}: {seen} file(s) readable, {found} finding(s)")
        if found == 0:
            print("SELFTEST FAILED: found nothing at a commit known to contain the defect. "
                  "This checker is vacuous.", file=sys.stderr)
            return 1
        print("  -> the checker detects the defect it was written for")
        return 0

    findings, examined = [], 0
    for rel in files:
        h, n = scan_source(rel, (ROOT / rel).read_text(encoding="utf-8", errors="replace"))
        findings += h
        examined += n

    out = {
        "question": "Does any path literal in tracked python name one particular computer?",
        "why": "portable_paths.ESTATE existed and 17 constants across 14 files ignored it. "
               "Two crash loudly on another box; eight are guarded by .exists() and degrade "
               "SILENTLY — build_vor_draft_value.py emits VOR values built from no draft "
               "data, validation_sweep.py runs all five checks under a different "
               "interpreter. A convention with no checker is a preference.",
        "method": "AST string literals only. Comments are invisible to ast by construction; "
                  "docstrings are excluded by node identity; the match is ANCHORED at the "
                  "start of the string so prose ABOUT a path is not a path claim. A grep "
                  "reports 29 hits here, 12 of them prose.",
        "files_scanned": len(files),
        "string_literals_examined": examined,
        "findings": findings,
        "fix": "from portable_paths import ESTATE   ->   ESTATE / \"vector-<repo>/...\"",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"  {len(files)} tracked .py, {examined} string literal(s) examined")
    print(f"  FINDINGS: {len(findings)}")
    for f in findings:
        print(f"    {f['file']}:{f['line']}  {f['literal'][:70]}")
    print(f"\nwrote {OUT}")
    if args.check and findings:
        print(f"CHECK FAILED: {len(findings)} path literal(s) name one computer",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
