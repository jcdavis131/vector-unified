#!/usr/bin/env python3
"""A script that prints corpus text must declare its stdout encoding. READ-ONLY.

Solo personal project, no connection to employer, built with public/free-tier only

WHAT THIS COST. Regenerating data/pitch_age_axis.json died with

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u0107'

the c-acute in a footballer's name. Windows hands a redirected stdout cp1252, and that
script prints names straight from the corpus. The artifact therefore COULD NOT BE REBUILT
AT ALL — it sat 8h stale behind a producer that crashed every time it ran, and nothing
reported that, because artifact_freshness measures mtimes and says STALE without ever
asking whether a rebuild is even possible.

Rebuilt once fixed, it went from 411 rows to 1042 and pct_scorable 21.2 -> 43.9. The live
page already published 43.8%. So the crash had quietly frozen a local artifact at less than
half its real coverage, on the side of the comparison every local gate reads.

WHAT IT CHECKS. Tracked *.py under pipeline/ and scripts/ that call print() and never call
sys.stdout.reconfigure(...). 65 files in pipeline/ already declare it; this reports the
ones that do not.

WHAT IT DOES NOT CLAIM. A missing declaration is NOT a proven crash. These scripts fail
only when a non-ASCII character actually reaches a cp1252 stdout, which depends on the data
and on whether output is redirected. One of the 43 was observed crashing; the other 42 are
the same SHAPE, not the same evidence. The finding is "this script has not said what
encoding its output is", which is a real gap and a weaker claim than "this will break".

That gap between shape and evidence is why this is REPORT-ONLY. Blocking on 42 files would
force a bulk mechanical edit across the pipeline, and a bulk mechanical edit from this
session already shipped a NameError that py_compile could not see — the fix would be more
dangerous than the defect. The list is the deliverable; fixing is deliberate and per-file.

    python pipeline/check_stdout_encoding.py
    python pipeline/check_stdout_encoding.py --check   # exit 1 if any file lacks it

Writes: data/stdout_encoding_audit.json
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "stdout_encoding_audit.json"


def declares_encoding(tree: ast.AST) -> bool:
    """Any call of the form <something>.stdout.reconfigure(...).

    Matched on the ATTRIBUTE CHAIN, not on the text `sys.stdout`, because this repo imports
    sys three different ways — `import sys`, `import sys as _sys`, and a late in-function
    import — and a substring match would miss two of them.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "reconfigure" \
                and isinstance(node.func.value, ast.Attribute) \
                and node.func.value.attr == "stdout":
            return True
    return False


def print_calls(tree: ast.AST) -> int:
    return sum(1 for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "print")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if any file lacks it")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    r = subprocess.run(["git", "ls-files", "pipeline/*.py", "scripts/*.py"], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    files = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]

    undeclared, declared, silent = [], 0, 0
    for rel in files:
        try:
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        n = print_calls(tree)
        if n == 0:
            silent += 1
            continue
        if declares_encoding(tree):
            declared += 1
        else:
            undeclared.append({"file": rel, "print_calls": n})

    undeclared.sort(key=lambda d: -d["print_calls"])
    out = {
        "question": "Does every script that prints corpus text declare its stdout encoding?",
        "why": "build_pitch_age_axis.py died with UnicodeEncodeError on 'c-acute' in a "
               "footballer's name, so data/pitch_age_axis.json could not be rebuilt at all. "
               "It sat 8h STALE behind a producer that crashed on every run. Once fixed it "
               "went 411 -> 1042 rows and pct_scorable 21.2 -> 43.9, against a live page "
               "already publishing 43.8 — the crash had frozen a local artifact at under "
               "half its real coverage.",
        "claim_limit": "A missing declaration is NOT a proven crash. These fail only when a "
                       "non-ASCII character actually reaches a cp1252 stdout, which depends "
                       "on the data and on redirection. 1 of these was OBSERVED crashing; "
                       "the rest are the same shape, not the same evidence.",
        "why_report_only": "Blocking would force a bulk mechanical edit across 42 files, "
                           "and a bulk mechanical edit this session already shipped a "
                           "NameError that py_compile could not see. The fix would be more "
                           "dangerous than the defect. Fixing is deliberate and per-file.",
        "files_scanned": len(files),
        "declared": declared,
        "no_print_calls": silent,
        "undeclared": undeclared,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"  {len(files)} tracked script(s): {declared} declare stdout encoding, "
          f"{silent} never print, {len(undeclared)} print WITHOUT declaring")
    for d in undeclared[:12]:
        print(f"    {d['file']:46} {d['print_calls']} print()")
    if len(undeclared) > 12:
        print(f"    ... {len(undeclared) - 12} more, full list in {OUT.name}")
    print(f"\nwrote {OUT}")
    if args.check and undeclared:
        print(f"CHECK FAILED: {len(undeclared)} script(s) print without declaring an "
              f"encoding", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
