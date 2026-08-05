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

WHAT IT FOUND, and it is much larger than the bug that prompted it:

    working tree   15 PASS   2 SKIP   2 FAIL
    fresh clone     8 PASS   2 SKIP   9 FAIL

Seven checks beyond mine fail only on a clone, nearly all FileNotFoundError. And one is
worse than a failure: check_cited_fields.py verifies 64 published values here and 0 in the
clone, and prints PASS both times. A gate that goes green over an empty set on every clone
is not a weaker gate, it is a misleading one.

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
LINE = re.compile(r"^\s{2}(PASS|FAIL|SKIP)\s+(\S+)\s")


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

        here = verdicts(ROOT)
        there = verdicts(clone)
        n_here = len(list((ROOT / "data").glob("*")))
        n_there = len(list((clone / "data").glob("*")))

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
                                 for v in ("PASS", "FAIL", "SKIP")},
                "clone": {v: sum(1 for x in there.values() if x == v)
                          for v in ("PASS", "FAIL", "SKIP")}},
            "pass_here_fail_on_clone": broke,
            "fail_in_both": both,
            "per_check": {k: {"working_tree": here[k], "clone": there.get(k, "?")}
                          for k in here},
            "KNOWN_VACUOUS_PASS": "check_cited_fields.py verifies 64 published values in "
                "the working tree and 0 in the clone, and prints PASS both times. It is "
                "NOT in pass_here_fail_on_clone because it does not fail — that is the "
                "problem. A gate that goes green over an empty set on every clone is "
                "misleading rather than merely weak. Counting it as a pass is how this "
                "audit would have missed it.",
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
