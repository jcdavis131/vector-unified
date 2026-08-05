#!/usr/bin/env python3
"""Does the gate still work on a fresh clone? Clone it and run it. Do not parse for it.

Solo personal project, no connection to employer, built with public/free-tier only

.gitignore line 27 is `data/*` with explicit `!data/<file>` negations. So a checker that
reads a data/ file which is not negated passes on this box and dies on a clone, and
nothing detected that.

I SHIPPED EXACTLY THAT BUG: check_ablation_consistency.py was registered BLOCKING while
all three of its inputs were untracked. Then I wrote a static scanner to find others, and
IT WAS VACUOUS -- 18 of 19 checks came back "tracked=0 ignored=0" because the regexes
looked for "data/x.json" and DATA / "x.json" while the actual idiom is
ROOT / "data" / "x.json". Its one hit was a checker's own OUTPUT, not an input. A green
scanner over an empty match set is the same defect it was written to find.

SO THIS DOES THE REAL THING INSTEAD: `git clone --local` into a temp dir, check out the
same commit, run `validate.py --offline` there, and diff the per-check verdicts against
the same run in the working tree. No parsing, no idiom assumptions, no false confidence.

WHAT IT FOUND, and it was much larger than the bug that prompted it:

    working tree   15 PASS   2 SKIP   2 FAIL   0 N/A
    fresh clone     8 PASS   2 SKIP   9 FAIL   0 N/A     <- before the fix

Seven checks beyond mine failed only on a clone, nearly all FileNotFoundError, because
.gitignore deliberately excludes the ~24 MB of matrices, checkpoints and assets they need.
validate.py now declares those prerequisites and reports N/A instead of FAIL:

    fresh clone     8 PASS   2 SKIP   2 FAIL   7 N/A     <- after
    pass-here-fail-on-clone: 0. The 2 remaining are the genuine pre-existing failures,
    the same two the working tree reports.

check_cited_fields.py verified 64 published values here and 0 in the clone and printed PASS
both times. It now refuses to pass on zero coverage, so the current state is:

    fresh clone     7 PASS   2 SKIP   3 FAIL   7 N/A
    pass-here-fail-on-clone: 1, and it is a TRUE positive.

THAT ONE WAS NOT SILENCED THE WAY THE SEVEN WERE -- it was fixed at the source. It was
check_cited_fields.py:95 hardcoding HUB = Path("C:/Users/jcdav/vector-hub/assets/data"),
an absolute path into a sibling repo: machine-specific rather than clone-specific, so it
had verified zero published values for every reader who is not the author while printing a
green line. It now resolves through portable_paths.ESTATE, and the current state is:

    fresh clone     9 PASS   2 SKIP   1 FAIL   9 N/A
    pass-here-fail-on-clone: 0.  fail-in-both: artifact_freshness, a genuine open finding.

RESOLVING IT WAS THE PRECONDITION FOR N/A BEING HONEST HERE. While the absolute path
remained, "the sibling repo is missing" was a lie about the cause -- the check was broken
on every box, not just in a clone. Once the path is derived from the repo root, a sibling
that is genuinely absent from a temp clone directory IS the same missing-prerequisite
condition the seven had, and reports N/A truthfully. Order matters: N/A first would have
buried the defect.

The same laptop-path defect turned out to be estate-wide rather than confined to this one
gate -- 37 literals across 20 files, 8 of them guarded by .exists() and therefore silent.
pipeline/check_laptop_paths.py now blocks on it.

NOT REGISTERED IN validate.py, DELIBERATELY. This script runs validate.py; registering it
inside validate.py is unbounded recursion. It is a standalone diagnostic, run by hand, and
that is why validate.py's "unregistered checker is a FAILURE" rule does not apply -- it is
excluded by name below rather than by being quietly forgotten.

    python pipeline/check_gate_inputs_tracked.py
    python pipeline/check_gate_inputs_tracked.py --check   # exit 1 if a check is clone-broken

Writes: data/gate_inputs_tracked_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "gate_inputs_tracked_audit.json"
# N/A MUST BE IN THIS PATTERN. It was not, and the omission made this audit understate
# its own subject: after validate.py gained the N/A state, a clone reported 8 PASS / 2
# FAIL / 2 SKIP = 12 of 19 checks, and the 7 that could not run simply vanished from the
# summary rather than being counted. An audit whose totals silently shrink is the same
# disappearing-coverage defect it was written to detect.
LINE = re.compile(r"^\s{2}(PASS|FAIL|SKIP|N/A)\s+(\S+)\s")


def verdicts(cwd: Path) -> dict[str, str]:
    p = subprocess.run([sys.executable, str(cwd / "pipeline" / "validate.py"), "--offline"],
                       cwd=str(cwd), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = {}
    for ln in (p.stdout or "").splitlines():
        m = LINE.match(ln)
        if m:
            out[m.group(2)] = m.group(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any check passes here but fails on a clone")
    ap.add_argument("--keep", action="store_true", help="do not delete the temp clone")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    tmp = Path(tempfile.mkdtemp(prefix="gateclone_"))
    clone = tmp / "repo"
    try:
        r = subprocess.run(["git", "clone", "-q", "--local", str(ROOT), str(clone)],
                           capture_output=True, text=True)
        if r.returncode:
            print(f"FAIL: clone failed: {r.stderr[:300]}", file=sys.stderr)
            return 2
        subprocess.run(["git", "checkout", "-q", head], cwd=clone, capture_output=True)

        # RESTORE WHAT THE WORKING-TREE RUN MUTATES. verdicts(ROOT) runs validate.py HERE,
        # and validate.py registers tennis_mtnn as `train_tennis_mtnn.py --check`, an arm
        # that RETRAINS. So every audit moved data/tennis_forward_report.json off the value
        # dumbmodel.com cites — ridge16_all_features_r 0.8427 -> 0.8332, observed three
        # times, twice after the sweep prompt was "fixed" by removing its direct validate.py
        # call. The rule was evaded transitively: the thing the sweep still ran called it.
        #
        # The comparison genuinely needs the working tree (a clone lacks the untracked
        # artifacts, which is the whole point), so the fix is not to stop running it but to
        # put back what it moves. Only TRACKED files are restored, and only to their
        # committed state — nothing untracked is touched, so a legitimately regenerated
        # audit still shows up in git status.
        # SNAPSHOT CONTENT, RESTORE CONTENT — not `git checkout`, and not keyed on which
        # files were dirty. The first version compared `git diff --name-only` before and
        # after and restored only files that BECAME dirty. Tested with a file that was
        # ALREADY dirty: validate.py destroyed the local edit, the guard skipped the file
        # because it was in the before-set, and the run's drift was left in place —
        # ridge16_all_features_r sitting at 0.8332 instead of 0.8427. Worst of both, and
        # the comment claiming it "leaves pre-existing local edits alone" was simply wrong;
        # the edit was already gone by then.
        #
        # Restoring to HEAD would clobber a legitimate local edit. Restoring to the PRE-RUN
        # bytes undoes exactly what this audit did and nothing else, whatever state the
        # tree was in when it started.
        #
        # THE data/ + assets/ SCOPE IS MEASURED, not assumed. Hashing all 163 tracked files
        # in the repo before and after a full run leaves exactly ONE changed:
        # data/gate_inputs_tracked_audit.json, this file's own report, written after the
        # restore below. Nothing outside the snapshotted directories moves. That was an
        # open question — "not known to touch anything else" is an absence of evidence —
        # and it is now a measurement. Re-run it if the check list in validate.py grows.
        def snapshot() -> dict[str, bytes]:
            r = subprocess.run(["git", "ls-files", "data", "assets"], cwd=str(ROOT),
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            snap = {}
            for rel in (ln.strip() for ln in (r.stdout or "").splitlines()):
                f = ROOT / rel
                if rel and f.is_file():
                    try:
                        snap[rel] = f.read_bytes()
                    except OSError:
                        pass
            return snap

        before = snapshot()
        here = verdicts(ROOT)
        moved = []
        for rel, blob in before.items():
            f = ROOT / rel
            try:
                if f.is_file() and f.read_bytes() != blob:
                    f.write_bytes(blob)
                    moved.append(rel)
            except OSError:
                pass
        if moved:
            print(f"  restored {len(moved)} tracked artifact(s) this audit mutated, to "
                  f"their PRE-RUN bytes: {', '.join(sorted(moved)[:4])}")
        there = verdicts(clone)
        n_here = len(list((ROOT / "data").glob("*")))
        n_there = len(list((clone / "data").glob("*")))

        total_here, total_there = len(here), len(there)
        if total_here != total_there:
            print(f"  WARNING: {total_here} checks parsed here vs "
                  f"{total_there} in clone — a verdict state is unparsed",
                  file=sys.stderr)
        broke = sorted(k for k in here
                       if here[k] == "PASS" and there.get(k) == "FAIL")
        both = sorted(k for k in here if here[k] == "FAIL" and there.get(k) == "FAIL")

        out = {
            "question": "Does every registered check still work on a fresh clone?",
            "method": "git clone --local, checkout the same commit, run validate.py "
                      "--offline there, diff per-check verdicts against the working tree. "
                      "A previous static-parsing attempt was VACUOUS (18/19 checks matched "
                      "no path literal) and is recorded here so it is not retried.",
            "commit": head,
            "data_files": {"working_tree": n_here, "clone": n_there,
                           "exist_only_locally": n_here - n_there},
            "summary": {
                "working_tree": {v: sum(1 for x in here.values() if x == v)
                                 for v in ("PASS", "FAIL", "SKIP", "N/A")},
                "clone": {v: sum(1 for x in there.values() if x == v)
                          for v in ("PASS", "FAIL", "SKIP", "N/A")}},
            "pass_here_fail_on_clone": broke,
            "fail_in_both": both,
            "per_check": {k: {"working_tree": here[k], "clone": there.get(k, "?")}
                          for k in here},
            "FIXED_was_a_vacuous_pass": "check_cited_fields.py verified 64 published "
                "values in the working tree and 0 in the clone, and printed PASS both "
                "times. It now refuses to pass on zero coverage, so it appears in "
                "pass_here_fail_on_clone as a TRUE positive rather than being invisible.",
            "RESOLVED_the_last_pass_here_fail_on_clone": "cited_fields was the one true "
                "positive here and it was FIXED AT THE SOURCE rather than silenced. "
                "check_cited_fields.py:95 hardcoded an absolute path into vector-hub, so "
                "it was machine-specific, not clone-specific, and verified zero published "
                "values on any box that is not the author's while printing green. It now "
                "resolves through portable_paths.ESTATE. Resolving it was the PRECONDITION "
                "for N/A being honest: while the absolute path remained, 'the sibling repo "
                "is missing' misstated the cause. The same defect proved estate-wide — 37 "
                "path literals across 20 files, 8 guarded by .exists() and therefore "
                "silent — and pipeline/check_laptop_paths.py now blocks on it.",
            "why_not_registered": "This script runs validate.py. Registering it inside "
                "validate.py is unbounded recursion. Excluded by name in validate.py's "
                "glob rather than silently forgotten.",
        }
        OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

        print(f"  commit {head[:12]}   data/ files: {n_here} here, {n_there} in clone")
        print(f"  working tree {out['summary']['working_tree']}")
        print(f"  fresh clone  {out['summary']['clone']}")
        print(f"\n  PASS here, FAIL on clone ({len(broke)}):")
        for k in broke:
            print(f"    {k}")
        print(f"  FAIL in both ({len(both)}): {', '.join(both) or 'none'}")
        print(f"\nwrote {OUT}")
        if args.check and broke:
            print(f"CHECK FAILED: {len(broke)} check(s) pass here and fail on a fresh "
                  f"clone", file=sys.stderr)
            return 1
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
