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
    # A page's `source` strings are machine-readable citations, and nothing had ever read
    # them as anything but prose. Checks that a cited field EXISTS in the file cited.
    # Covers 76 of 126 references; the other 48 use an indexed or prose shorthand this
    # deliberately refuses to guess at, and it prints that count every run rather than
    # implying full coverage.
    "cited_fields": (["check_cited_fields.py", "--check"], False),
    # The tennis MTNN must keep beating the learned LINEAR map it was justified against
    # (0.0584). If a future change drops it below that, a projection matrix is the better
    # model and this registers the regression instead of leaving it in a report nobody reads.
    "tennis_mtnn": (["train_tennis_mtnn.py", "--check"], False),
    # The fourth and last forward probe. Fails only if the RUN is broken — PPR carrying
    # forward (which would make persistence an artefact of duplication) or a null with no
    # spread. Not on a negative answer: 0 of 4 positions earning their keep IS the finding.
    "gridiron_forward": (["build_gridiron_forward.py", "--check"], False),
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
    "hoops_forward": (["build_hoops_forward.py", "--check"], False),
    "equities_forward": (["build_equities_forward.py", "--check"], False),
    # BLOCKING AGAIN, and this time the mutation is wired rather than described. It was
    # blocking, then demoted in 306f645 when a real mutation exited 0 and proved the
    # guard green over an empty class — it only matched corrections quoting their target
    # VERBATIM, and no such block was left in the estate. The guard now accepts a
    # DECLARED target (corrects_field: "REPO_STATE_WARNING.also_destroyed"), exact path,
    # no inference. corrections_landed/marker_removed in check_guards_nonvacuous.py now
    # reports clean=0 planted=1 restored=0, so the gate is known to be able to fail.
    # Coverage is 1 declared block of 20; the other 19 are counted and reported, not
    # guessed at.
    "corrections_landed": (["check_corrections_landed.py", "--check"], False),
    # Not a check_*.py, so the glob does not demand it — registered because its
    # --check arm is a real guard: it REFUSES the archetype join unless the gridiron
    # block of unified_matrix.npz and gridiron_season_emb.npz are provably the same
    # rows (player_idx in order AND cosine 1.0000 row-for-row). Equal row counts are
    # exactly the trap probe_g1_position.py exists for.
    "bridge_join": (["query_bridge.py", "--check"], False),
    # BLOCKING, and it now PASSES with the one real disagreement DECLARED rather than
    # demoted. It was registered red: `full@seed7` has three different records across the
    # three ablation artifacts, differing in 6 of 8 fields, while seeds 8 and 9 are
    # bit-identical. The cause was then measured rather than argued — ablation.py is
    # irreproducible at a fixed seed (0.6940 / 0.6926 / 0.6827 on three consecutive runs of
    # one config), so those artifacts CAN NEVER AGREE.
    #
    # That changed what the gate should do. Permanently red for an unfixable condition
    # teaches the reader to skip the line, which is the reasoning already applied to
    # internal_prose. Deleting it would be worse — it is the only thing that would notice a
    # NEW disagreement. So the known one is declared in KNOWN with its evidence, and both
    # arms are mutation-tested: planting a fresh disagreement exits 1, and a KNOWN entry
    # that no longer applies also exits 1, so a declaration cannot outlive its defect.
    "ablation_consistency": (["check_ablation_consistency.py", "--check"], False),
    # "It means what we think it means." Every past instance of this estate's core defect
    # was found by a person reading a file; this decides the mechanical part — a name that
    # asserts a range, an aggregate beside its own sample, a count beside its collection.
    # READ-ONLY by construction, which is deliberate: check_documented_usage.py executed
    # documented commands and one run mutated ten artifacts here plus a sibling repo's
    # seed_floor.json. This one opens files and writes only its own report.
    #
    # It found a real defect on its first clean run, in an artifact written earlier the
    # same day: stage2_seed_floor.json stored sd 0.0044 beside values whose own sd is
    # 0.0043, because sd was computed at full precision and rounded independently of the
    # values. Fixed at the source in build_seed_floor.py.
    "field_semantics": (["check_field_semantics.py", "--check"], False),
    # A seed set AFTER model construction controls nothing. REPORT-ONLY (no --check),
    # for two reasons. It is a SHAPE heuristic — it cannot know which constructors draw
    # random weights — so a false positive is possible by construction. And its one current
    # finding, ablation.py::train_config, is real but deliberately unfixed: correcting the
    # order changes what every historical ablation number means, and the three shipped
    # artifacts were produced under it. Blocking would leave the gate permanently red for a
    # condition nobody has decided to resolve.
    #
    # Precision on real data is 1 finding / 329 files across 6 repos, and it discriminates:
    # it flags ablation.py and NOT train_stage2.py or vector-realty's train_mtnn.py, both of
    # which seed first and both of which were measured reproducing bit-identically.
    "seed_before_init": (["check_seed_before_init.py", "--estate"], False),
    # REPORT-ONLY ON PURPOSE — note there is no --check, so it always exits 0.
    # Measured precision is 2 of 23 over the 173-artifact estate, about 9%. The two real
    # finds were worth having, but a BLOCKING gate at 9% precision trains its reader to
    # ignore it, and 21 false alarms per run is a worse failure than the one it catches.
    # Registered rather than omitted because validate.py treats an unregistered checker
    # as a FAILURE, and silently deleting it to keep the board green would be the exact
    # kind of unearned green this file exists to prevent.
    "internal_prose": (["check_internal_prose.py"], False),
}

# Large artifacts a check CANNOT RUN WITHOUT, declared per check.
#
# A fresh clone of this repo fails 9 of 19 checks. Seven of those fail only on a clone,
# and the cause is not a defect in the check — it is that .gitignore deliberately excludes
# `pipeline/data/`, `assets/*.json` and most of `data/`, exactly as the rule at the top of
# .gitignore says it should ("large GENERATED artifacts are not [tracked] ... would put
# ~43 MB of JSON blobs into history"). Tracking these six would add ~24 MB and contradict
# that rule.
#
# So the fix is not to track them and not to leave the check reporting FAIL. A check that
# CANNOT RUN must report neither FAIL, which asserts a defect was found, nor PASS, which
# asserts something was verified. It reports UNAVAILABLE, counted separately and printed
# with the missing path, exactly as --offline already yields SKIP.
#
# THIS IS NOT A WAY TO QUIET A FAILING CHECK. On this box every prerequisite exists, so
# every check runs and this map changes nothing. It only distinguishes "could not run"
# from "ran and found a problem" for a reader who does not have the artifacts.
#
# Measured with pipeline/check_gate_inputs_tracked.py, which clones the repo and runs this
# file inside the clone.
PREREQS: dict[str, list[str]] = {
    "draft_value_invariants": ["data/qb_survivorship_probe.json"],
    "gate_nonvacuity": ["pipeline/data/unified_stage2_best.pt"],
    "tennis_mtnn": ["pipeline/data/tennis_matrix.npz"],
    "tennis_forward": ["pipeline/data/tennis_matrix.npz"],
    "bridge_join": ["pipeline/data/unified_matrix.npz"],
    "g1_position": ["assets/unified.json"],
    # guards_nonvacuous plants defects in front of the other guards, so it inherits
    # whatever they need. Listed with the union rather than left to fail opaquely.
    "guards_nonvacuous": ["pipeline/data/tennis_matrix.npz", "assets/unified.json",
                          "../vector-hub/assets/data"],
    # ESTATE-RELATIVE, and these only became visible once the laptop paths were fixed.
    # Four gates read the published pages in the SIBLING repo vector-hub. While they
    # hardcoded C:/Users/jcdav/vector-hub they resolved from anywhere — including from
    # inside a temp clone, which is why the first clone audit looked better than the truth:
    # the clone was reading my real vector-hub, not simulating a machine that lacks it.
    # Now they derive from portable_paths.ESTATE, so a checkout without its siblings
    # correctly cannot run them. That is N/A, not a defect.
    "superlatives": ["../vector-hub/assets/data"],
    "cited_fields": ["../vector-hub/assets/data"],
    "hub_freshness": ["../vector-hub/assets/data"],
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

    # check_gate_inputs_tracked.py RUNS THIS FILE (it clones the repo and invokes
    # validate.py --offline inside the clone to find checks that only work on this box).
    # Registering it in CHECKS would be unbounded recursion. It is named here, explicitly,
    # rather than left to trip the unregistered-checker FAILURE — an exclusion a reader
    # can see and argue with is not the same as a checker quietly forgotten, which is the
    # failure mode that rule exists to catch.
    # check_documented_usage.py is excluded for a DIFFERENT reason than recursion: it
    # executes 108 documented commands and takes minutes, so registering it would multiply
    # the gate cost for a meta-check best run deliberately. Named here rather than left to
    # trip the unregistered-checker FAILURE, so the exclusion is arguable rather than
    # forgotten.
    RUNS_VALIDATE = {"check_gate_inputs_tracked.py", "check_documented_usage.py"}
    found = {p.name for p in PIPE.glob("check_*.py")} - RUNS_VALIDATE
    registered = {argv[0] for argv, _ in CHECKS.values()}
    unregistered = sorted(found - registered)

    results: list[tuple[str, str, float, str]] = []
    for name, (argv, needs_net) in CHECKS.items():
        if needs_net and args.offline:
            results.append((name, "SKIP", 0.0, "--offline"))
            continue
        absent = [p for p in PREREQS.get(name, []) if not (ROOT / p).exists()]
        if absent:
            results.append((name, "N/A", 0.0,
                            f"cannot run: {absent[0]} absent (deliberately untracked)"))
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
    unavail = [n for n, s, _, _ in results if s == "N/A"]
    # UNAVAILABLE is never folded into the pass count. A reader without the artifacts must
    # see that N checks did not run, not a green line implying they did.
    if unavail:
        print(f"\n{len(unavail)} check(s) COULD NOT RUN (prerequisite artifact absent, "
              f"deliberately untracked — see PREREQS in this file): {', '.join(unavail)}")
    if failed or unregistered:
        print(f"\n{len(failed)} check(s) failed, {len(unregistered)} unregistered.")
        return 1
    ran = len(results) - len(skipped) - len(unavail)
    bits = []
    if skipped:
        bits.append(f"{len(skipped)} skipped: {', '.join(skipped)}")
    if unavail:
        bits.append(f"{len(unavail)} could not run")
    suffix = f" ({'; '.join(bits)})" if bits else ""
    print(f"\nall {ran} check(s) that could run pass{suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
