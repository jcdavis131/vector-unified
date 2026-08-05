#!/usr/bin/env python3
"""Run the read-only data checks and append one line to a log. No agent, no tokens.

Solo personal project, no connection to employer, built with public/free-tier only

WHY THIS IS A SCRIPT AND NOT AN AGENT PROMPT. The session crons that do this are
in-memory: they die when the Claude session exits and auto-expire after 7 days regardless.
"Always validating" cannot rest on that. This file is what a durable scheduler should call
— it is deterministic, costs nothing to run, and needs no model in the loop to decide
whether `sd 0.0044` equals the sd of the values beside it.

READ-ONLY BY CONSTRUCTION, and the list of what it refuses to run is the load-bearing part:

    NOT build_*.py / probe_*.py / acquire_*.py — they write artifacts with NO flag at all.
      One run of a checker that executed documented commands rewrote ten artifacts in this
      repo AND stripped a CORRECTED marker from vector-hoops/pipeline/seed_floor.json,
      taking three green gates red.
    NOT any trainer — train_*.py and ablation.py overwrite the shipped checkpoint
      (sport_acc 0.6851, ckpt b055641c03760624).
    NOT validate.py in full — it runs train_tennis_mtnn.py --check, which RETRAINS tennis
      and moves data/tennis_mtnn_report.json off the value dumbmodel.com publishes. That
      is how the current 6-value cited_fields disagreement got there.

So this runs three named checks and nothing else. Adding a check here means checking first
whether it writes.

    python scripts/validation_sweep.py            # run, print, append to the log
    python scripts/validation_sweep.py --quiet    # log only, for a scheduler
    python scripts/validation_sweep.py --check    # exit 1 if any check fails

Appends: data/validation_sweep_log.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "validation_sweep_log.md"

# The CUDA venv is the interpreter the estate's checks are known to run under. Falls back
# to whatever is running this file if that venv is gone, rather than dying — a sweep that
# refuses to start reports nothing, which is worse than a sweep that reports a degraded run.
VENV = Path("C:/Users/jcdav/vector-hoops/pipeline/.venv/Scripts/python.exe")
PY = str(VENV) if VENV.exists() else sys.executable

# name -> argv. Every one verified read-only: it opens artifacts and writes at most its own
# report. If you add a line here, run it once and diff `git status` across the estate first.
CHECKS: dict[str, list[str]] = {
    "field_semantics": ["pipeline/check_field_semantics.py", "--estate", "--check"],
    "cited_fields": ["pipeline/check_cited_fields.py", "--check"],
    "ablation_consistency": ["pipeline/check_ablation_consistency.py", "--check"],
    "merged_careers": ["pipeline/check_merged_careers.py", "--check"],
    "superlatives": ["pipeline/check_superlatives.py", "--check"],
}

TIMEOUT_S = 300


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quiet", action="store_true", help="log only, no stdout table")
    ap.add_argument("--check", action="store_true", help="exit 1 if any check fails")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    results: dict[str, str] = {}
    detail: dict[str, str] = {}
    for name, argv in CHECKS.items():
        t0 = time.monotonic()
        try:
            p = subprocess.run([PY, str(ROOT / argv[0]), *argv[1:]], cwd=str(ROOT),
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=TIMEOUT_S)
            rc = p.returncode
            tail = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()
            detail[name] = tail[-1][:150] if tail else ""
        except subprocess.TimeoutExpired:
            rc, detail[name] = "TIMEOUT", f"exceeded {TIMEOUT_S}s (may be load, not a defect)"
        except FileNotFoundError:
            rc, detail[name] = "ABSENT", "check script not present on this box"
        results[name] = "PASS" if rc == 0 else ("FAIL" if rc == 1 else str(rc))
        detail[name] += f"  [{time.monotonic() - t0:.1f}s]"

    # The semantics audit carries the counts worth trending; read them if it wrote one.
    counts = {}
    fs = ROOT / "data" / "field_semantics_audit.json"
    if fs.exists():
        try:
            d = json.loads(fs.read_text(encoding="utf-8"))
            counts = d.get("counts_by_arm") or {}
            counts["files_scanned"] = d.get("files_scanned")
        except Exception:
            counts = {"unreadable": True}

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    npass = sum(1 for v in results.values() if v == "PASS")
    line = (f"| {stamp} | {npass}/{len(results)} pass | "
            + " ".join(f"{k}={v}" for k, v in results.items())
            + f" | semantics {counts or 'n/a'} |")

    if not LOG.exists():
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(
            "# Validation sweep log\n\n"
            "Appended by `scripts/validation_sweep.py`. Read-only checks only — see that\n"
            "file's docstring for what is deliberately NOT run and why.\n\n"
            "A row here is evidence a sweep happened, including when it found nothing.\n"
            "An absent row is not 'clean'; it means nobody looked.\n\n"
            "| when (UTC) | summary | per-check | semantics counts |\n"
            "|---|---|---|---|\n", encoding="utf-8")
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")

    if not args.quiet:
        w = max(len(k) for k in results)
        for k, v in results.items():
            print(f"  {v:<7} {k:<{w}}  {detail[k]}")
        print(f"\n  {npass}/{len(results)} pass   semantics {counts or 'n/a'}")
        print(f"  appended to {LOG}")

    failed = [k for k, v in results.items() if v != "PASS"]
    if args.check and failed:
        print(f"SWEEP FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
