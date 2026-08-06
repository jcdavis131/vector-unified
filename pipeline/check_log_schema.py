#!/usr/bin/env python3
"""A row with more cells than its header loses content silently. READ-ONLY.

Solo personal project, no connection to employer, built with public/free-tier only

data/validation_sweep_log.md and data/drift_watch_log.md are the estate's evidence trail —
"a row is evidence the check ran, including when it found nothing; an ABSENT row is not
clean, it means nobody looked." That only holds if the rows say what they appear to say.

WHAT PROMPTED IT. Appending a drift row, I wrote FIVE cells against a four-column header,
splitting a read-only note into a column that does not exist. Markdown renders it without
complaint. Caught by hand before commit, which is not a control. Scanning found one already
committed:

    data/validation_sweep_log.md:21   6 cells, 5-column header

Its sixth cell holds "tennis_forward_report drifted AGAIN — reverted. Cause: ..." — the
most important thing in the row, in a column no header names.

TWO CLASSES, AND THEY ARE NOT THE SAME DEFECT.

    MORE cells than the header    content is pushed into a column that does not exist. What
                                  a reader sees is not what was written. This is the one
                                  worth failing on.
    FEWER cells                    trailing columns are empty. Usually a column added later,
                                  which is exactly what happened here: `interpreter` was
                                  added 2026-08-05 and the five earlier rows predate it. The
                                  log's own header already says an empty cell there means
                                  "not recorded", not "pinned-venv". Reported separately,
                                  not failed.

CALIBRATION. An ESCAPED pipe is content, not a delimiter. Counting raw `|` reported
docs/CULTURAL_TEXT_SCHEMA.md:26 — `| `wiki_title` | str \\| null | ... |` — as over-wide
when the table is correct. That was a defect in this checker, not in the doc, and it was
1 of 2 findings: half the report was the tool's own bug.

    python pipeline/check_log_schema.py
    python pipeline/check_log_schema.py --check   # exit 1 on a row WIDER than its header

Writes: data/log_schema_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "log_schema_audit.json"

SEP = re.compile(r"^\s*\|(\s*:?-+:?\s*\|)+\s*$")


def cells(line: str) -> int:
    """Delimiter count. `\\|` is an escaped pipe and belongs to the cell's text."""
    return line.replace("\\|", "\x00").count("|") - 1


def scan(rel: str, text: str) -> tuple[list[dict], list[dict], int]:
    over, under, tables = [], [], 0
    lines = text.split("\n")
    ncol, in_table = 0, False
    for i, ln in enumerate(lines):
        if SEP.match(ln):
            ncol, in_table, tables = cells(ln), True, tables + 1
            continue
        s = ln.strip()
        if in_table and s.startswith("|") and s.endswith("|"):
            n = cells(ln)
            row = {"file": rel, "line": i + 1, "cells": n, "header_cells": ncol,
                   "text": s[:120]}
            if n > ncol:
                over.append(row)
            elif n < ncol:
                under.append(row)
        elif in_table and not s.startswith("|"):
            in_table = False
    return over, under, tables


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 on a row WIDER than its header")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    r = subprocess.run(["git", "ls-files", "*.md"], cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    files = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]

    over, under, tables = [], [], 0
    for rel in files:
        o, u, t = scan(rel, (ROOT / rel).read_text(encoding="utf-8", errors="replace"))
        over += o
        under += u
        tables += t

    out = {
        "question": "Does every markdown table row have the cell count its header declares?",
        "why": "validation_sweep_log.md and drift_watch_log.md are the evidence trail. A row "
               "with MORE cells than the header pushes content into a column nothing names, "
               "and markdown renders it without complaint. sweep_log:21 hides "
               "'tennis_forward_report drifted AGAIN — reverted' in a sixth cell of a "
               "five-column table.",
        "two_classes": {
            "over": "content mis-assigned to a column that does not exist — the failing class",
            "under": "trailing cells empty, usually a column added later. `interpreter` was "
                     "added 2026-08-05 and the earlier rows predate it; the log's own header "
                     "already states an empty cell there means 'not recorded'. Reported, not "
                     "failed.",
        },
        "calibration": "An escaped pipe is content. Counting raw '|' reported "
                       "docs/CULTURAL_TEXT_SCHEMA.md:26 (`str \\| null`) as over-wide when "
                       "the table is fine — 1 of 2 findings was this checker's own bug.",
        "files_scanned": len(files),
        "tables_found": tables,
        "over_wide": over,
        "under_wide": under,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"  {len(files)} tracked .md, {tables} table(s)")
    print(f"  OVER-WIDE (content in a column that does not exist): {len(over)}")
    for d in over:
        print(f"    {d['file']}:{d['line']}  {d['cells']} cells vs {d['header_cells']}")
    print(f"  under-wide (trailing empty, reported not failed): {len(under)}")
    for d in under[:6]:
        print(f"    {d['file']}:{d['line']}  {d['cells']} vs {d['header_cells']}")
    print(f"\nwrote {OUT}")
    if args.check and over:
        print(f"CHECK FAILED: {len(over)} row(s) wider than their header", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
