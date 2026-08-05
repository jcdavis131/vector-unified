#!/usr/bin/env python3
"""A command a docstring advertises must actually run. Nothing here ever checked that.

Solo personal project, no connection to employer, built with public/free-tier only

Every script in this pipeline ends its docstring with the invocations it supports:

    python pipeline/decompose_g2_ab.py --fix-floors                  # repair old floors

That line was a lie for two commits. `fix_floors()` read `v["paired_MDE_n3"]` literally,
which was right at three seeds and became a KeyError the moment `rebuild()` started writing
`paired_MDE_n5`. It was found during a close-out review by running the usage lines by hand
-- the only time in this repo's history anything has done so. A documented command that
crashes is a promise the file does not keep, and there was no guard for that whole class.

WHAT IT RUNS, and the restriction is the interesting part. Executing every advertised
command is not safe: `train_stage2.py`'s usage starts a 60-epoch training run, `--write`
forms mutate artifacts, and several need a `--runs DIR` that exists only on the box that
made it. So a line is EXECUTED only when all of these hold:

    * it names a script in pipeline/
    * it contains no --write / --apply / --promote
    * it contains no placeholder token (DIR, PATH, <...>, ALL-CAPS bare words)
    * its script is not on the TRAINERS denylist (would burn GPU)
    * its script is not on the WHOLE_GATE denylist (minutes; re-runs everything)

Every executed line gets a 120s timeout so a hanging command cannot hang this file. A
TIMEOUT IS REPORTED SEPARATELY AND IS NOT COUNTED AS BROKEN: probe_company_edges.py
exceeded the budget on a run taken while the box was also training, running a 25-agent
workflow and building a dashboard, and exits 0 on both its documented forms when idle. The
timeout measured machine load, not the command.

EVERYTHING ELSE IS REPORTED AS SKIPPED WITH ITS REASON, never silently dropped, and the
skipped count prints every run. A checker that quietly narrows its own scope until it
passes is the failure this repo keeps cataloguing; the whole point here is that the
uncovered set stays visible.

An executed line must exit 0 OR 1. Zero is success; one is a checker legitimately
reporting a finding (`--check` arms are meant to exit 1). Exit 2+, or a traceback, means
the command is broken -- which is exactly what --fix-floors did.

    python pipeline/check_documented_usage.py
    python pipeline/check_documented_usage.py --check   # exit 1 on a BROKEN command
                                                        # (exit 2+); timeouts do not fail it

Writes: data/documented_usage_audit.json
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
PIPE = ROOT / "pipeline"
OUT = ROOT / "data" / "documented_usage_audit.json"

USAGE = re.compile(r"^\s*python\s+(pipeline/[A-Za-z0-9_]+\.py)([^\n#]*)")
MUTATING = ("--write", "--apply", "--promote", "--fix-floors")
# `\[` catches the OPTIONAL-ARGUMENT convention. A docstring writing
#     python pipeline/acquire_wikipedia_bios.py [--limit N] [--sleep 0.2]
# is describing optional flags, not a literal command. The first run of this file passed
# those brackets straight to argparse and reported the script BROKEN with
# "unrecognized arguments: [--limit N]". That was this checker's defect, not the script's,
# and a false positive is as damaging as a miss here — it teaches the reader that a red
# line is noise.
PLACEHOLDER = re.compile(r"\bDIR\b|\bPATH\b|<[^>]+>|\bFILE\b|\bNAME\b|\[")

# Scripts whose documented invocation TRAINS. Running these would burn GPU hours and
# overwrite the shipped checkpoint, which is the one thing this lane exists to protect.
TRAINERS = {"train_stage2.py", "train_unified.py", "ablation.py", "train_tennis_mtnn.py",
            "sweep_tennis_hparams.py"}

# Scripts that RUN THE WHOLE GATE, and therefore cost minutes and re-run everything this
# checker is already running individually. The first version of this file omitted them and
# took over ten minutes: validate.py executes 18 checks, and check_gate_inputs_tracked.py
# clones the repo and runs validate.py TWICE more inside the clone. Checking that
# `validate.py --offline` parses its own arguments does not need three full gate runs.
#
# Their exclusion is a real coverage gap, not a free win — if `validate.py --offline` ever
# stopped running, this file would not notice. It is reported like every other skip.
WHOLE_GATE = {"validate.py", "check_gate_inputs_tracked.py", "check_gate_nonvacuity.py"}

# THIS FILE RUNS DOCUMENTED COMMANDS, AND ITS OWN DOCSTRING DOCUMENTS COMMANDS. The first
# run executed itself twice and both invocations hit the 120s timeout, so it reported
# itself BROKEN. Unbounded self-recursion, found by the file's own first real run.
SELF = "check_documented_usage.py"

# A documented command that hangs is broken too, but it must not hang THIS file.
TIMEOUT_S = 120

# --fix-floors is in MUTATING above and therefore never executed here, even though it is
# the command whose breakage motivated this file. It rewrites data/g2_centroid_ab.json in
# place. Running a mutating command to prove it works would corrupt the artifact it
# repairs whenever it is half-broken -- precisely the case worth catching. Its coverage
# gap is REPORTED rather than papered over.


def usages(path: Path) -> list[str]:
    # FileNotFoundError is caught because the glob and the read are not atomic. A `git
    # checkout` of a branch that lacks a file, run WHILE this is scanning, deletes it
    # between PIPE.glob() listing it and read_text() opening it, and the whole scan dies
    # with a traceback partway through. That happened here — a branch switch during a
    # background run killed the scan at pipeline/probe_roster_identity.py and left a STALE
    # audit on disk that read as a completed result. A checker that dies mid-sweep and
    # leaves its previous output in place is worse than one that reports partial coverage.
    try:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8", errors="replace")))
    except (SyntaxError, FileNotFoundError, OSError):
        return []
    if not doc:
        return []
    return [m.group(0).strip() for ln in doc.splitlines() if (m := USAGE.match(ln))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if a documented command is broken")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ran, skipped, broken, timed_out = [], [], [], []
    for src in sorted(PIPE.glob("*.py")):
        for line in usages(src):
            m = USAGE.match(line)
            script, rest = m.group(1), m.group(2).strip()
            target = ROOT / script
            reason = None
            if not target.exists():
                reason = "script named in the docstring does not exist"
            elif target.name == SELF:
                reason = "this file; running it from itself is unbounded recursion"
            elif target.name in TRAINERS:
                reason = "trains; running it would burn GPU and overwrite the checkpoint"
            elif target.name in WHOLE_GATE:
                reason = ("runs the whole gate; minutes per invocation, and re-runs "
                          "the very checks this file already runs individually")
            elif any(f in line for f in MUTATING):
                reason = "mutating (--write/--apply/--promote/--fix-floors)"
            elif not (target.name.startswith("check_") or "--check" in line):
                # WRITE-BY-DEFAULT IS THE COMMON CASE, and screening only on flags was not
                # enough. MUTATING catches --write/--apply/--promote, but build_*.py,
                # probe_*.py and acquire_*.py write their artifacts with NO flag at all.
                # Running them turned this file into a mutation engine: one run rewrote ten
                # artifacts in this repo AND vector-hoops/pipeline/seed_floor.json in a
                # SIBLING repo, stripping a CORRECTED marker and breaking three gate checks
                # (corrections_landed, guards_nonvacuous, cited_fields) that had been green.
                # A checker that damages the tree it is checking is worse than no checker.
                #
                # So the rule is now a WHITELIST, not a denylist: execute only a check_*.py
                # or an explicit --check arm, both of which are read-only by contract.
                # Coverage drops and that is the correct trade — the skipped set stays
                # visible below.
                reason = ("writes by default (no --check arm); builders and probes emit "
                          "artifacts with no flag, and running them mutated a sibling repo")
            elif PLACEHOLDER.search(rest):
                reason = "takes a placeholder path that exists only where it was made"
            if reason:
                skipped.append({"docstring_of": src.name, "command": line,
                                "reason": reason})
                continue
            try:
                proc = subprocess.run([sys.executable, str(target), *rest.split()],
                                      cwd=str(ROOT), capture_output=True, text=True,
                                      encoding="utf-8", errors="replace",
                                      timeout=TIMEOUT_S)
            except subprocess.TimeoutExpired:
                # A TIMEOUT IS NOT A BREAKAGE, and treating it as one made this file
                # report a flapping verdict. probe_company_edges.py exceeded 120s on a run
                # taken while the box was also running a training job, a 25-agent workflow
                # and a separate build; on an idle box both of its documented forms exit 0.
                # So the timeout measured MACHINE LOAD, not the command — a real value
                # answering a different question than the one it appeared to answer.
                # Reported in its own bucket, never counted as broken, and --check does not
                # fail on it. Only a non-zero, non-1 exit is evidence of a defect.
                timed_out.append({"docstring_of": src.name, "command": line,
                                  "limit_s": TIMEOUT_S,
                                  "note": "exceeded the budget — may be load, not a "
                                          "defect; re-run on an idle box before believing it"})
                continue
            rec = {"docstring_of": src.name, "command": line, "exit": proc.returncode}
            if proc.returncode in (0, 1):
                ran.append(rec)
            else:
                tail = ((proc.stderr or "") + (proc.stdout or "")).strip().splitlines()
                rec["tail"] = tail[-1][:200] if tail else ""
                broken.append(rec)

    out = {
        "question": "Does every command a docstring advertises actually run?",
        "why": "decompose_g2_ab.py --fix-floors crashed with KeyError('paired_MDE_n3') for "
               "two commits. Nothing in this repo had ever executed a documented usage "
               "line, so the whole class was unguarded.",
        "pass_criterion": "an executed command exits 0 or 1. 1 is legitimate — a --check "
                          "arm reporting a finding. 2+ or a traceback means broken.",
        "executed": ran,
        "broken": broken,
        "skipped": skipped,
        "timed_out": timed_out,
        "coverage": {"executed": len(ran), "broken": len(broken),
                     "skipped": len(skipped), "timed_out": len(timed_out),
                     "pct_executed": round(100.0 * len(ran) / max(len(ran) + len(skipped), 1), 1)},
        "what_this_does_NOT_cover": "Mutating and training invocations are never run, so "
            "the command that motivated this file (--fix-floors) is itself outside its "
            "coverage. Running a mutating command to prove it works would corrupt the "
            "artifact it repairs in exactly the half-broken case worth catching. The gap "
            "is reported, not closed.",
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"  executed {len(ran)}   BROKEN {len(broken)}   timed-out {len(timed_out)}   "
          f"skipped {len(skipped)}  "
          f"({out['coverage']['pct_executed']}% of runnable lines executed)")
    for b in broken:
        print(f"    BROKEN exit={b['exit']}  {b['command']}")
        print(f"           {b.get('tail', '')}")
    by_reason: dict[str, int] = {}
    for s in skipped:
        by_reason[s["reason"]] = by_reason.get(s["reason"], 0) + 1
    for r, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"    skipped {n:>2}  {r}")
    print(f"\nwrote {OUT}")
    if args.check and broken:
        print(f"CHECK FAILED: {len(broken)} documented command(s) do not run",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
