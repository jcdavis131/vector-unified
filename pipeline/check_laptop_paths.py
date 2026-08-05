#!/usr/bin/env python3
"""A path literal in CODE must not name one particular computer. READ-ONLY.

Solo personal project, no connection to employer, built with public/free-tier only

portable_paths.py has defined `ESTATE = <repo>.parent` and `resolve()` for a while, and
`migrate_hub_portable_paths.py` already applied it to the PUBLISHED citations. But nothing
applied it to python source, and nothing checked, so 17 module-level constants across 14
files kept naming `C:/Users/jcdav/...` long after the convention existed. A convention with
no checker is a preference.

WHY IT MATTERS MORE THAN "it breaks on another box". Two of the seventeen crash loudly,
which is honest. Eight were guarded and degrade SILENTLY:

    build_vor_draft_value.py:169   if DRAFT_CSV.exists():      skips the draft-pick merge
                                                               and emits VOR values built
                                                               from no draft data
    validation_sweep.py:54         PY = str(VENV) if VENV.exists() else sys.executable
                                                               runs all five checks under a
                                                               different interpreter and
                                                               the log row never said which

That is this estate's recurring defect: a real value answering a different question than
the one it appears to answer. The path is real. It resolves — on one computer.

WHAT IT CHECKS. Every tracked `.py`, every string literal the AST can see, flagged when the
string STARTS with a drive letter or UNC root.

CALIBRATION, and each rule is a false positive it would otherwise make:

    comments            invisible to ast by construction, so `# WAS Path("C:/...")` —
                        which several files carry deliberately, recording the defect they
                        fixed — cannot be flagged. This is why the check is AST-based and
                        not a grep; a grep reports 29 hits here, of which 12 are prose.
    docstrings          collected by ast.get_docstring for module, class and function, then
                        excluded by node identity. portable_paths.py's own docstring lists
                        example laptop paths as the thing it exists to remove.
    strings that MENTION a path   check_gate_inputs_tracked.py stores the sentence
                        "HUB = Path('C:/Users/jcdav/...'), an absolute path into a sibling
                        repo" in its JSON output. Anchoring the match at the START of the
                        string separates a path from prose about a path.

NON-VACUITY IS TESTED, NOT ASSUMED. A previous static scanner in this repo passed green
because 18 of its 19 patterns matched nothing. So `--selftest` re-runs this checker against
the versions of the same files at a given commit and requires it to FIND the ones already
fixed. A checker that cannot find a known defect is not evidence of its absence.

THE ESTATE IS NOT CLEAN, AND FIXING THIS REPO IS WHAT REVEALED IT. vector-unified is 0.
`--estate` scans 358 tracked .py across 8 repos and finds 19 more:

    vector-hoops      16
    vector-equities    3
    everything else    0

SIX OF THOSE HARDCODE A SESSION SCRATCHPAD and are on a clock:

    SC = Path(r"C:\\Users\\jcdav\\AppData\\Local\\Temp\\claude\\C--Users-jcdav"
              r"\\be69d382-ce38-4d23-b6d1-d92c62546c02\\scratchpad\\hoops_ab")

That id is the CURRENT session. The directory exists right now, which is exactly why
nothing has failed and why nobody noticed — these scripts work today and break the moment
the session ends. Same defect as sweep_tennis_hparams.py, which was fixed here with
tempfile.mkdtemp() per run; it turns out the pattern had already propagated to two sibling
repos, one of which is the live-site repo.

NOT FIXED FROM HERE. Those are other repos' lanes, and vector-hoops master is a deploy.
The fix is known and proven, and it is one commit in each repo — the operator's call.

    python pipeline/check_laptop_paths.py
    python pipeline/check_laptop_paths.py --check              # exit 1 on any finding
    python pipeline/check_laptop_paths.py --estate             # all 8 repos
    python pipeline/check_laptop_paths.py --selftest 4f29a8a~1 # must find >=1 at that commit

Writes: data/laptop_paths_audit.json (local) or data/laptop_paths_estate_audit.json
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
# TWO FILES, ONE PER SCOPE, and that is not tidiness. validate.py registers the LOCAL form
# (111 files, 0 findings); --estate scans 358 files across 8 repos and finds 19. Writing
# both to one path means the committed artifact says 111 or 358 depending on who ran last,
# and a reader cannot tell which. data/seed_order_audit.json did exactly that earlier —
# recorded 108 files when the gate scans 330, silently claiming a third of its own coverage.
OUT = ROOT / "data" / "laptop_paths_audit.json"
OUT_ESTATE = ROOT / "data" / "laptop_paths_estate_audit.json"

# ANCHORED AT THE START. `C:/x` is a path; "HUB = Path('C:/x')" is a sentence about one.
# Deliberately broad on the root itself — the rule is "no machine-local path in code", not
# "no path under C:/Users/jcdav" — matching portable_paths.ABS_RE's stated reasoning.
ABS_RE = re.compile(r"^(?:[A-Za-z]:[/\\]|\\\\[A-Za-z0-9_.-]+\\)")


# DELIBERATE FAULT INJECTION. check_guards_nonvacuous.py exists to prove the guards catch a
# laptop path, so it has to WRITE one:
#
#     doc["source_files"][0] = "C:/Users/jcdav/vector-unified/data/tennis_forward_report.json"
#
# That is a correct string in a correct file, and flagging it would be a false positive that
# teaches the reader to skim. Exempted BY FILE AND SUBSTRING rather than folded into a
# heuristic, so the exemption is arguable on sight and cannot quietly widen: if that file
# grows a second, accidental laptop path, only the injected one stays exempt.
ALLOW = {
    "pipeline/check_guards_nonvacuous.py": {
        "C:/Users/jcdav/vector-unified/data/tennis_forward_report.json":
            "injected on purpose to prove the guard is non-vacuous",
    },
}


def docstring_nodes(tree: ast.AST) -> set[int]:
    """id() of every Constant that is a docstring, so prose examples are not path claims."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def allow_table_nodes(tree: ast.AST) -> set[int]:
    """id() of every Constant inside the ALLOW assignment — the exemption table itself.

    THE ALLOWLIST IS MADE OF THE THING IT EXEMPTS. Its keys are laptop-path strings written
    in code, so this checker flags its own table. It did not show up until the file was
    COMMITTED: scanning is driven by `git ls-files`, so while check_laptop_paths.py was
    untracked it scanned 110 files and never once looked at itself. The count going 110 ->
    111 is what exposed it.

    Scoped to the ALLOW node rather than skipping this whole file, so an accidental laptop
    path anywhere else in here still fails. A checker that exempts itself wholesale is the
    one place a laptop path would never be found.
    """
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "ALLOW" for t in node.targets):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant):
                    out.add(id(sub))
    return out


def scan_source(rel: str, src: str) -> tuple[list[dict], int]:
    """-> (findings, string literals examined). The second number makes a vacuous run visible."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [], 0
    skip = docstring_nodes(tree) | allow_table_nodes(tree)
    hits, examined = [], 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in skip:
            continue
        examined += 1
        if ABS_RE.match(node.value):
            if node.value in ALLOW.get(rel, {}):
                continue
            hits.append({
                "file": rel,
                "line": getattr(node, "lineno", 0),
                "literal": node.value[:160],
                "why": "a path literal in code that names one computer; use "
                       "portable_paths.ESTATE / \"vector-<repo>/...\" instead",
            })
    return hits, examined


ESTATE_REPOS = ("vector-hoops", "vector-gridiron", "vector-pitch", "vector-equities",
                "vector-hub", "vector-realty", "vector-tennis", "vector-unified")


def tracked_py(repo: Path | None = None) -> list[str]:
    r = subprocess.run(["git", "ls-files", "*.py"], cwd=str(repo or ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def estate_scan() -> tuple[list[dict], int, int, dict]:
    """Scan every sibling repo. The defect this file fixed here also lives next door.

    vector-unified is 0 after the 37-literal cleanup. The estate is NOT: 16 in vector-hoops
    and 3 in vector-equities, and six of those are the SESSION-SCRATCHPAD form — the same
    defect fixed in sweep_tennis_hparams.py, propagated across repos rather than confined
    to one file.
    """
    findings, files_n, lits = [], 0, 0
    per_repo: dict[str, int] = {}
    for name in ESTATE_REPOS:
        d = ROOT.parent / name
        if not (d / ".git").exists():
            continue
        n = 0
        for rel in tracked_py(d):
            f = d / rel
            if not f.is_file():
                continue
            files_n += 1
            # SCAN WITH THE REPO-RELATIVE KEY, then prefix for reporting. ALLOW is keyed by
            # repo-relative path, so passing "vector-unified/pipeline/..." would miss the
            # deliberate fault-injection exemption and report it as a finding — a checker
            # breaking its own calibration the moment its scope widened.
            try:
                h, ex = scan_source(rel, f.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            lits += ex
            for x in h:
                x["file"] = f"{name}/{rel}"
            findings += h
            n += len(h)
        per_repo[name] = n
    return findings, files_n, lits, per_repo


def at_commit(rel: str, commit: str) -> str | None:
    r = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 on any finding")
    ap.add_argument("--estate", action="store_true",
                    help="scan every sibling repo, not just this one")
    ap.add_argument("--selftest", metavar="COMMIT",
                    help="scan the same files as of COMMIT; exit 1 if it finds nothing, "
                         "which would mean this checker cannot detect the defect it exists for")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    files = tracked_py()

    if args.selftest:
        found, seen = 0, 0
        for rel in files:
            old = at_commit(rel, args.selftest)
            if old is None:
                continue
            seen += 1
            h, _ = scan_source(rel, old)
            found += len(h)
        print(f"  SELFTEST at {args.selftest}: {seen} file(s) readable, {found} finding(s)")
        if found == 0:
            # SAY WHAT WAS MEASURED, NOT WHAT IT IMPLIES. This used to print "This checker
            # is vacuous", which is only true if COMMIT actually predates the fix — and the
            # first thing I did was point it at 4f29a8a, the commit AFTER, where 0 is the
            # correct answer. A tool that reports a real measurement as the wrong conclusion
            # is the defect this repo keeps finding, in the checker written to find it.
            print(f"SELFTEST INCONCLUSIVE: 0 findings at {args.selftest}. Either this "
                  f"checker is vacuous, or {args.selftest} is already clean — pass a commit "
                  f"from BEFORE the laptop paths were removed (4f29a8a~1 has 36).",
                  file=sys.stderr)
            return 1
        print("  -> the checker detects the defect it was written for")
        return 0

    per_repo: dict = {}
    if args.estate:
        findings, n_files, examined, per_repo = estate_scan()
        files = [""] * n_files
    else:
        findings, examined = [], 0
        for rel in files:
            h, n = scan_source(rel, (ROOT / rel).read_text(encoding="utf-8",
                                                           errors="replace"))
            findings += h
            examined += n

    out = {
        "question": "Does any path literal in tracked python name one particular computer?",
        "why": "portable_paths.ESTATE existed and 17 constants across 14 files ignored it. "
               "Two crash loudly on another box; eight are guarded by .exists() and degrade "
               "SILENTLY — build_vor_draft_value.py emits VOR values built from no draft "
               "data, validation_sweep.py runs all five checks under a different "
               "interpreter. A convention with no checker is a preference.",
        "method": "AST string literals only. Comments are invisible to ast by construction; "
                  "docstrings are excluded by node identity; the match is ANCHORED at the "
                  "start of the string so prose ABOUT a path is not a path claim. A grep "
                  "reports 29 hits here, 12 of them prose.",
        "scope": "estate" if args.estate else "vector-unified only",
        "files_scanned": len(files),
        "string_literals_examined": examined,
        "per_repo": per_repo,
        "findings": findings,
        "fix": "from portable_paths import ESTATE   ->   ESTATE / \"vector-<repo>/...\"",
    }
    dest = OUT_ESTATE if args.estate else OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"  {len(files)} tracked .py, {examined} string literal(s) examined"
          f"{'  [ESTATE]' if args.estate else ''}")
    if per_repo:
        for k, v in per_repo.items():
            print(f"    {k:18} {v}")
    print(f"  FINDINGS: {len(findings)}")
    for f in findings:
        print(f"    {f['file']}:{f['line']}  {f['literal'][:70]}")
    print(f"\nwrote {OUT}")
    if args.check and findings:
        print(f"CHECK FAILED: {len(findings)} path literal(s) name one computer",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
