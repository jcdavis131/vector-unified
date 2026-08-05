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

MEASURED PRECISION, AND WHY IT IS NOT REGISTERED IN validate.py. Calibration took four
passes: 349 broken of 504 (69%, essentially all false) -> 46 -> 32 -> 22 of 172. The three
fixes were real defects in this file, not in the docs: it treated bare filenames as paths,
it resolved references from the repo root instead of from the doc containing them
(docs/SPEC.md naming STAGE2_PLAN.md, which sits beside it), and it judged handoff docs whose
subject is another repo.

The 22 that survive are mostly genuine — spot-checked five, and none has ever existed in
this repo's history: pipeline/career_trajectories.py, assets/pedigree.json,
docs/ARCHETYPE_ERA_RESEARCH.md, docs/domain-migration-plan.md, pipeline/data/mtnn_best.pt.
Some of the rest are gitignored artifacts that are absent on purpose, so precision is good
but not 100%, and this is NOT registered as a gate. A check that is right most of the time
still teaches its reader to skim, and internal_prose is already report-only here for the
same reason. Run it deliberately.

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
    if ref.startswith(OWNED_DIRS):
        return True
    # root-level markdown, e.g. CONTACTS.md — a real location claim
    return "/" not in ref and ref.endswith(".md")


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
    broken, checked, skipped = [], 0, 0
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
            if not (here.exists() or (ROOT / ref).exists()):
                broken.append({"doc": d, "reference": ref,
                               "line": text[:m.start()].count("\n") + 1,
                               "why": "named in a tracked doc but absent from the working "
                                      "tree — typically committed to a different branch"})

    out = {
        "question": "Does every path a tracked doc names exist where that doc lives?",
        "why": "Records go to master, tooling stays on a branch, and a document on one "
               "side names a file that is not on the other. Bit three times in one day: "
               "tools/dashboard/server.py, LOCAL_GPU_G2_RESULT.md, CONTACTS.md. The "
               "failure surfaces later, to whoever follows the pointer.",
        "scope": "tracked *.md; backticked tokens with a known extension that are either under a directory this repo owns (data/, pipeline/, tools/, assets/, scripts/, docs/, .github/) or a root-level .md. A bare filename is a NAME, not a location, and is out of scope.",
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
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"  {len(docs)} tracked doc(s), {checked} reference(s) checked, "
          f"{skipped} skipped by scope")
    print(f"  BROKEN: {len(broken)}")
    for b in broken:
        print(f"    {b['doc']}:{b['line']}  ->  {b['reference']}")
    print(f"\nwrote {OUT}")
    if args.check and broken:
        print(f"CHECK FAILED: {len(broken)} doc reference(s) point at files that are not "
              f"here", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
