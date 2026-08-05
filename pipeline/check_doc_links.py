#!/usr/bin/env python3
"""A path named in a tracked doc must exist where that doc lives. READ-ONLY.

Solo personal project, no connection to employer, built with public/free-tier only

This one class bit three times in a day, each time caught by hand:

    tools/dashboard/server.py   committed to master, launched from the work branch
                                -> "No such file or directory"
    LOCAL_GPU_G2_RESULT.md      indexed in the README, existed only on master
    CONTACTS.md                 same

Records get pushed to master, tooling stays on a branch, and a document written on one
side names a file that is not on the other. The failure surfaces later, as a dead link or
a missing file, to whoever followed the pointer. Nothing checked it.

WHAT IT CHECKS. Every tracked `.md` file, every BACKTICKED token that looks like a
repo-relative path with a known extension, and whether that path exists in the working
tree. Backticks only, and a known extension only, because prose is full of things that
look like paths and are not — the calibration lesson from check_field_semantics.py, whose
first draft reported 173 findings of which essentially all were false.

DELIBERATELY NOT CHECKED, and each exclusion is a false positive it would otherwise make:

    absolute paths (C:/..., /usr/...)   often deliberately quoted as examples of the
                                        laptop-path defect this estate is removing
    URLs                                not files
    paths with a glob or placeholder    `data/*.json`, `<name>.json`
    sibling repos (vector-*/...)        may legitimately be absent on this box

MEASURED PRECISION. Calibration took six passes: 349 broken of 504 (69%, essentially all
false) -> 46 -> 32 -> 22 of 172 -> 3 of 125, plus 7 moved out of "broken" entirely into
unbuilt_build_outputs. The scanned denominator drops from 172 to 125 because bare filenames
left scope. EVERY fix was a real defect in this file, not in the docs, which is the point
worth keeping:

    treated bare filenames as paths
    resolved references from the repo root instead of from the doc containing them
      (docs/SPEC.md naming STAGE2_PLAN.md, which sits right beside it)
    judged handoff docs whose subject is another repo
    kept an in_scope() exception for bare root-level .md that CONTRADICTED its own stated
      rule, and that could not do anything but produce false positives: a bare name that
      resolves is never reported, so the exception only ever fired on names that do not.
      9 findings, all of them prose naming a document rather than pointing at one.
    counted gitignored BUILD OUTPUTS as dead links. `pipeline/data/mtnn_best.pt` is
      something you make; a doc saying where the checkpoint lands is correct even on a
      machine that has never trained one. 7 findings, now reported separately under
      unbuilt_build_outputs rather than dropped — an unbuilt artifact is still worth
      seeing, it is just not a documentation defect.

WHAT THE REMAINING 3 ARE, in full — no sampling, the list is short enough to state:

    docs/TAXONOMY.md:187   pipeline/career_trajectories.py
    docs/TAXONOMY.md:286   docs/ARCHETYPE_ERA_RESEARCH.md
    tasks/todo.md:42       docs/MTNN_V5_PROMOTE_GATE.md

Each names a real directory, and none has ever existed anywhere in this repo's history.
They are ASPIRATIONAL — design docs naming work that was planned and not built. That is a
true finding and NOT a typo, so deleting the references would erase intent.

STILL NOT REGISTERED IN validate.py. Not because precision is poor — it is 3/3 on
inspection — but because every remaining finding is a judgement call about the operator's
design docs, and a blocking gate would force that decision by holding the board red until
someone deletes a plan. A registered check should be one a reader can FIX; these are ones
only the author can ANSWER. Run it deliberately.

    python pipeline/check_doc_links.py
    python pipeline/check_doc_links.py --check   # exit 1 on a broken reference

Writes: data/doc_links_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "doc_links_audit.json"

# A backticked token with a known extension. Anchored to the whole backtick span so
# `see foo.py for details` does not match on the word alone.
LINK = re.compile(r"`([A-Za-z0-9_./\\-]+\.(?:md|py|json|npz|pt|ps1|cmd|html|txt))`")

SKIP_PREFIX = ("http", "vector-", "C:", "/usr", "~", "..", "bundles/")
SKIP_CHARS = ("*", "?", "<", ">")

# ONLY PATHS THIS REPO OWNS. The first draft accepted any backticked token with a known
# extension and reported 349 broken of 504 checked — 69%, essentially all false. Two
# classes drowned it:
#
#   `vectors.json`, `embedding_v3.npz`   bare filenames. Prose refers to artifacts BY NAME
#                                        all through this estate; that is not a path claim.
#   `bundles/coordination/active-tasks.md`  a Hatch path, deliberately named, and
#                                        documented as unreachable from this box.
#
# So a reference must either sit under a directory this repo owns, or be a root-level .md.
# Anything else is a name, not a location, and this check has no business ruling on it.
# A checker at 31% precision teaches its reader to skip the line — the calibration lesson
# from check_field_semantics.py, which needed three passes for the same reason.
OWNED_DIRS = ("data/", "pipeline/", "tools/", "assets/", "scripts/", "docs/", ".github/")

# Docs whose subject is ANOTHER repo. A handoff describing work in vector-hoops writes
# `assets/vectors.json` meaning that repo's assets, and resolving it here is a category
# error — 10 of the remaining findings were exactly this. Excluded by name so the
# exclusion is arguable rather than silently folded into a heuristic.
CROSS_REPO_DOCS = {"LOCAL_GPU_HANDOFF.md", "COORDINATION_LOCAL_GPU.md",
                   "CLAIM_BOARD_PROMPT.md", "COORDINATION.md",
                   "COORDINATION_LOCAL_GPU_BLOCKER.md"}


def in_scope(ref: str) -> bool:
    # A REFERENCE MUST NAME A DIRECTORY THIS REPO OWNS. There used to be an exception here
    # for bare root-level markdown ("e.g. CONTACTS.md — a real location claim"), and it was
    # wrong twice over.
    #
    # It contradicted this file's own rule, stated above: a bare filename is a NAME, not a
    # location. And it could not do anything BUT produce false positives — a bare name that
    # resolves is never reported, so the exception only ever fired on names that do not.
    # All 9 findings it generated were prose:
    #
    #   tasks/todo.md:178   "...carried the superseded numbers; `SPEC.md`, `STAGE2_PLAN.md`"
    #                       naming WHICH documents were corrected. The files are real and
    #                       sit in docs/. Nobody was ever meant to follow that as a path.
    #   CONTACTS.md:28      "cannot write `active-tasks.md`" — a Hatch file the same
    #                       sentence documents as unreachable from this box.
    #
    # A rule whose only reachable effect is a false positive is worse than no rule: it is
    # the checker manufacturing the defect it reports.
    return ref.startswith(OWNED_DIRS)


def ignored(ref: str) -> bool:
    """Is this path covered by .gitignore? Then it is something you build, not a dead link."""
    r = subprocess.run(["git", "check-ignore", "-q", ref], cwd=str(ROOT),
                       capture_output=True)
    return r.returncode == 0


def tracked_docs() -> list[str]:
    r = subprocess.run(["git", "ls-files", "*.md"], cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 on a broken reference")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    docs = tracked_docs()
    broken, unbuilt, checked, skipped = [], [], 0, 0
    for d in docs:
        if d in CROSS_REPO_DOCS:
            continue
        p = ROOT / d
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in LINK.finditer(text):
            ref = m.group(1).replace("\\", "/")
            if (ref.startswith(SKIP_PREFIX) or any(c in ref for c in SKIP_CHARS)
                    or not in_scope(ref)):
                skipped += 1
                continue
            checked += 1
            # RESOLVE RELATIVE TO THE DOC FIRST. A markdown reference is relative to the
            # file containing it, not to the repo root. Resolving only from the root made
            # docs/SPEC.md's `STAGE2_PLAN.md` look broken while docs/STAGE2_PLAN.md sat
            # right beside it — 20-odd of the first pass's findings were that single
            # mistake, a checker reporting a defect that was entirely its own.
            here = (p.parent / ref)
            if here.exists() or (ROOT / ref).exists():
                continue
            row = {"doc": d, "reference": ref,
                   "line": text[:m.start()].count("\n") + 1}
            # A MISSING PATH THAT .gitignore COVERS IS A BUILD OUTPUT, NOT A DEAD LINK.
            # `pipeline/data/mtnn_best.pt` and `assets/pedigree.json` are things you MAKE;
            # a doc naming where the checkpoint lands is correct even on a machine that has
            # never trained one. Counting those as broken is what held precision near 90%
            # and kept this check unregistered. Separated rather than dropped — an unbuilt
            # artifact is still worth seeing, it is just not a documentation defect.
            if ignored(ref):
                row["why"] = ("covered by .gitignore — a build output that has not been "
                              "built here, NOT a broken reference")
                unbuilt.append(row)
            else:
                row["why"] = ("named in a tracked doc but absent from the working tree — "
                              "typically committed to a different branch, or planned and "
                              "never written")
                broken.append(row)

    out = {
        "question": "Does every path a tracked doc names exist where that doc lives?",
        "why": "Records go to master, tooling stays on a branch, and a document on one "
               "side names a file that is not on the other. Bit three times in one day: "
               "tools/dashboard/server.py, LOCAL_GPU_G2_RESULT.md, CONTACTS.md. The "
               "failure surfaces later, to whoever follows the pointer.",
        "scope": "tracked *.md; backticked tokens with a known extension that sit under a directory this repo owns (data/, pipeline/, tools/, assets/, scripts/, docs/, .github/). A bare filename is a NAME, not a location, and is out of scope — including a bare root-level .md, which used to be excepted and generated 9 false positives, all of them prose naming a document rather than pointing at one.",
        "deliberately_not_checked": {
            "absolute paths": "often quoted as EXAMPLES of the laptop-path defect",
            "URLs": "not files",
            "globs and placeholders": "data/*.json, <name>.json",
            "sibling repos (vector-*)": "may legitimately be absent on this box",
        },
        "docs_scanned": len(docs),
        "references_checked": checked,
        "references_skipped": skipped,
        "broken": broken,
        "unbuilt_build_outputs": unbuilt,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"  {len(docs)} tracked doc(s), {checked} reference(s) checked, "
          f"{skipped} skipped by scope")
    print(f"  BROKEN: {len(broken)}")
    for b in broken:
        print(f"    {b['doc']}:{b['line']}  ->  {b['reference']}")
    print(f"  unbuilt build outputs (gitignored, NOT a doc defect): {len(unbuilt)}")
    for b in unbuilt:
        print(f"    {b['doc']}:{b['line']}  ->  {b['reference']}")
    print(f"\nwrote {OUT}")
    if args.check and broken:
        print(f"CHECK FAILED: {len(broken)} doc reference(s) point at files that are not "
              f"here", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
