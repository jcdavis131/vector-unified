#!/usr/bin/env python3
"""One gate that runs every checker in this repo — Phase 7 close-out.

Solo personal project, no connection to employer, built with public/free-tier only

Phase 7 produced several independent checkers and several guards embedded in builders, and
NOTHING RAN THEM TOGETHER. A checker nobody invokes is a comment with a shebang. Each was
mutation-verified when written and then left to be remembered.

    check_draft_value_invariants.py   I1-I6, cross-artifact consistency
    check_wikidata_qids.py            every hard-coded QID is still what the code thinks

REGISTRATION IS MANDATORY, mirroring check_wikidata_qids.py's own rule and for the same
reason: a check_*.py that exists but is not in CHECKS would be silently absent from the
gate, which is the failure mode this file exists to end. Discovery is by glob, and an
unregistered checker is a FAILURE, not a skip.

NETWORK IS A REAL DEPENDENCY, and it is declared per check rather than assumed. The QID
check queries live Wikidata; --offline skips it and says so in the summary instead of
reporting a pass it did not earn.

WHAT THIS DOES NOT COVER, stated so the green line is not read as more than it is. The
guards inside builders — `_verify_season_years` in resolve_names.py, the empty-sport
refusal in pull_honors_wikidata.py, the missing-model refusal in embed_eval.py — only fire
when their builder runs, and a builder re-run costs a network pull. They are listed in
`unrun_guards` in the output so they are visible without being claimed as verified.

    python pipeline/validate.py
    python pipeline/validate.py --offline   # skip network-dependent checks
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPE = ROOT / "pipeline"

# name -> (argv after the interpreter, needs_network)
CHECKS: dict[str, tuple[list[str], bool]] = {
    "draft_value_invariants": (["check_draft_value_invariants.py", "--check"], False),
    "wikidata_qids": (["check_wikidata_qids.py", "--check"], True),
    # ~4 min: 50 permutation shuffles, each recomputing G3 over 6,000-point silhouettes.
    "gate_nonvacuity": (["check_gate_nonvacuity.py", "--check"], False),
    "merged_careers": (["check_merged_careers.py", "--check"], False),
    "artifact_freshness": (["check_artifact_freshness.py", "--check"], False),
    # The PUBLISHED site, checked the same way and for the reason 7.30 found the hard
    # way: dumbmodel.com served "48-d" for weeks after mtnn_meta.json said 64. Needs
    # network for the live-vs-committed comparison.
    "hub_freshness": (["check_hub_freshness.py", "--check"], True),
    # Every fabrication this project produced was a superlative, six for six, and a
    # warning in the generator prompt demonstrably did not stop the sixth. Local only.
    "superlatives": (["check_superlatives.py", "--check"], False),
    # The G1 position arm was pinned at 1.0/0.0 by the mask bug from Phase 2 to 7.21.
    # This is the first thing that can report it regressing.
    "g1_position": (["probe_g1_position.py", "--check"], False),
    # Plants a defect in front of each guard and requires it to notice. The guards
    # were each mutation-tested once, by hand, in the turn that created them — and a
    # commit message does not run.
    "guards_nonvacuous": (["check_guards_nonvacuous.py", "--check"], False),
    # Fails if the shuffled-target arm does not collapse — i.e. if the tennis
    # evaluation is leaking and its +0.0949 gain cannot be trusted.
    "tennis_forward": (["build_tennis_forward.py", "--check"], False),
}

# Guards that live inside builders and only fire when that builder runs. Listed, not run:
# re-running them costs a live Wikidata pull, and claiming them as checked would be exactly
# the kind of unearned green this file was written to prevent.
UNRUN_GUARDS = {
    "resolve_names.py::_verify_season_years":
        "refuses to write when a derived season year is >1y from its own label",
    "pull_honors_wikidata.py::empty-sport refusal":
        "refuses to write when any sport produced zero honors or zero handles rows",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--offline", action="store_true",
                    help="skip checks that need live network")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    found = {p.name for p in PIPE.glob("check_*.py")}
    registered = {argv[0] for argv, _ in CHECKS.values()}
    unregistered = sorted(found - registered)

    results: list[tuple[str, str, float, str]] = []
    for name, (argv, needs_net) in CHECKS.items():
        if needs_net and args.offline:
            results.append((name, "SKIP", 0.0, "--offline"))
            continue
        t0 = time.monotonic()
        proc = subprocess.run([sys.executable, str(PIPE / argv[0]), *argv[1:]],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", cwd=str(ROOT))
        dt = time.monotonic() - t0
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()
        results.append((name, "PASS" if proc.returncode == 0 else "FAIL", dt,
                        tail[-1][:110] if tail else ""))

    width = max(len(n) for n in CHECKS)
    for name, status, dt, note in results:
        print(f"  {status:<4} {name:<{width}}  {dt:5.1f}s  {note}")

    if unregistered:
        print()
        for f in unregistered:
            print(f"  FAIL unregistered checker {f} — add it to CHECKS in validate.py; a "
                  f"checker outside the gate is a comment with a shebang")

    print("\nguards not exercised here (they fire only when their builder runs):")
    for g, what in UNRUN_GUARDS.items():
        print(f"  - {g}: {what}")

    failed = [n for n, s, _, _ in results if s == "FAIL"]
    skipped = [n for n, s, _, _ in results if s == "SKIP"]
    if failed or unregistered:
        print(f"\n{len(failed)} check(s) failed, {len(unregistered)} unregistered.")
        return 1
    suffix = f" ({len(skipped)} skipped: {', '.join(skipped)})" if skipped else ""
    print(f"\nall {len(results) - len(skipped)} check(s) pass{suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
