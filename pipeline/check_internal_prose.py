#!/usr/bin/env python3
"""A number written in an artifact's prose must match that artifact's own fields.

Solo personal project, no connection to employer, built with public/free-tier only

check_cited_fields.py verifies that dumbmodel.com's pages agree with the artifacts they
cite. Nothing verifies that an artifact agrees with ITSELF, and it turns out they do not
always. Found by hand while auditing promotion gates:

    data/gate_nonvacuity.json
      note_rank prose : "both score 12.4, exactly the real value"
      its own fields  : real 12.1, global_shuffle 12.1, within_sport_shuffle 12.1

The three fields agree with each other, so the substantive claim survives -- the nulls
score what the real model scores, so rank_nondeg_pass detects collapse and nothing more.
Only the number quoted in the summary is stale. But a reader who took 12.4 and compared
it against unified_report.json's effective_rank of 12.0 would be comparing two numbers
neither of which is in that file, and would conclude the rank had dropped. That is this
repo's whole subject: a real value answering a different question than the one it appears
to answer.

THE HARD PART IS NOT FINDING NUMBERS, IT IS NOT DROWNING IN THEM. A naive "every number
in every string must appear as a field" produces hundreds of hits, because prose
legitimately cites other files, years, counts of things that are not fields, and figures
from other repos. check_cited_fields.py learned this the expensive way -- three parser
versions and 43 false-positive MISSING fields before it was inverted to parse only what
is unambiguous and REPORT the uncovered count rather than guess at it. Same discipline
here.

So this reports one narrow, high-signal class and counts everything it skipped:

    STALE     a prose number that matches NO field exactly, but IS within rounding of a
              field at one fewer decimal place, or within 5% of one. 12.4 against a field
              of 12.1 qualifies. This is the shape a number takes when the artifact was
              regenerated and the summary was not.

Everything else is counted, not reported:

    EXACT     the number appears as a field somewhere in the file. Fine.
    UNMATCHED nothing close. Almost always a legitimate external reference -- a year, a
              line number, another repo's figure, a count of something not stored. NOT
              flagged, because flagging it is what produced 43 false positives last time.

A STALE hit is a CANDIDATE, not a verdict. Two unrelated quantities can sit within 5% of
each other by chance, and the script says so rather than asserting.

MEASURED PRECISION IS 2 OF 23 ACROSS THE 173-ARTIFACT ESTATE, about 9%. That is reported
rather than implied, and it is why this is REPORT-ONLY and not registered in validate.py:
a gate at 9% precision trains its reader to ignore it. The 21 others are benign in ways
worth knowing, because each one is a shape the check cannot distinguish from a defect:

  contrasted quantities  "val (n=761) and test (n=790) are near-identical in size" --
                         the sentence exists to say they DIFFER, and "near-identical"
                         reads as an identity claim.
  deliberate comparison  "0.8038 ... matches seed 31 uniquely (seed 7 gives 0.8354)"
  prose rounding         "a CQS seed sd near 0.73" against a field of 0.7315
  ranges                 "per-position 0.58-0.74"

    python pipeline/check_internal_prose.py            # report
    python pipeline/check_internal_prose.py --check    # exit 1 if any STALE candidate

Writes: data/internal_prose_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "internal_prose_audit.json"
ESTATE = ROOT.parent

# The whole estate, not just this repo. The first version scanned vector-unified's 73
# artifacts and said so as a caveat; the other four repos hold 105 more, including the
# seed_floor.json files this session wrote. Sibling roots are derived from ROOT.parent
# rather than hardcoded, matching portable_paths.py.
SCAN_DIRS = [ROOT / "data", ROOT / "data" / "market_cultural"]

def _sibling_repos():
    """DISCOVERED, NOT HARDCODED. This list used to be a literal four-tuple, and
    vector-realty was built as a sixth domain without anything noticing it existed — the
    subagent that built it had to report the omission by hand. A hardcoded estate list is
    a thing that silently goes stale exactly when the estate grows, which is the only
    time it matters.

    A sibling qualifies if it is a vector-* directory with a pipeline/ inside. The
    discovered list is PRINTED every run so a missing repo is visible rather than
    inferred from an unchanged total."""
    return sorted(p.name for p in ESTATE.iterdir()
                  if p.is_dir() and p.name.startswith("vector-")
                  and p.name != ROOT.name and (p / "pipeline").is_dir())


SIBLINGS = _sibling_repos()
for _sib in SIBLINGS:
    SCAN_DIRS += [ESTATE / _sib / "pipeline" / "data", ESTATE / _sib / "pipeline"]

# A number with a decimal point, or a 3+ digit integer. Bare small integers (0, 1, 5, 11)
# are far too common in prose to be worth testing and produce nothing but noise.
NUM = re.compile(r"(?<![\w.])(\d+\.\d+|\d{3,})(?![\w.])")

# Prose fields only. A `source` string is a machine-readable citation and belongs to
# check_cited_fields.py; re-checking it here would double-report the same defect.
SKIP_KEYS = {"source", "source_files", "source_hashes", "_verification",
             "_portable_path_rewrites", "built", "verify_commands"}


def numeric_fields(obj, path="", out=None):
    """Every numeric leaf in the document, with its path."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            numeric_fields(v, f"{path}.{k}", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            numeric_fields(v, f"{path}[{i}]", out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[path] = float(obj)
    return out


def prose_strings(obj, path="", out=None):
    """Every string leaf whose key is not a machine-readable citation."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP_KEYS:
                continue
            prose_strings(v, f"{path}.{k}", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            prose_strings(v, f"{path}[{i}]", out)
    elif isinstance(obj, str) and len(obj) > 20:
        out.append((path, obj))
    return out


# AN ASSERTION CUE IS WHAT MAKES A NEAR-MISS A DEFECT.
#
# The first version of this file flagged any prose number within 5% of a field, or equal
# to one at a coarser rounding. It produced 134 hits over 75 artifacts and almost all
# were noise of three kinds:
#
#     "Persistence is 0.85-0.92 here"          a RANGE; neither end is a field
#     "below the 0.01 bar this file set"       a THRESHOLD, not a measurement
#     "a 200-draw simulation"                  a COUNT of something not stored
#
# Which is precisely the failure this file's own docstring cites check_cited_fields.py
# for having made three times. Proximity alone is not evidence: in a document full of
# correlated quantities, some prose number will sit near some field by chance.
#
# What made gate_nonvacuity.json's 12.4 a real defect was not that 12.1 was nearby. It
# was the sentence CLAIMING they were the same number: "both score 12.4, exactly the real
# value". So the near-miss must be accompanied by an identity claim within the same
# clause. That is narrow on purpose -- it will miss stale numbers written without an
# assertion, and the report says so rather than implying completeness.
ASSERTION = re.compile(
    r"\b(exactly|identical|the same|equals?|matches|precisely|namely|i\.e\.|"
    r"unchanged|is the|was the|which is)\b", re.I)
ASSERTION_WINDOW = 55


def classify(n: float, fields: dict[str, float], text: str, span):
    """EXACT / STALE(candidates) / NEAR_NO_CLAIM / UNMATCHED for one prose number."""
    if any(abs(n - v) < 1e-9 for v in fields.values()):
        return "EXACT", []
    # SIGN: prose writes magnitudes ("CQS dropped 0.0054") where the field is signed
    # (-0.0054). The estate scan flagged exactly that as STALE against an unrelated
    # 0.0047, because the correct field never compared equal. Match on magnitude, and
    # treat an exact magnitude match as EXACT rather than as a near-miss.
    if any(abs(abs(n) - abs(v)) < 1e-9 for v in fields.values()):
        return "EXACT", []
    cands = []
    for p, v in fields.items():
        if v == 0:
            continue
        av = abs(v)
        if round(av, max(0, _dp(n) - 1)) == round(abs(n), max(0, _dp(n) - 1)):
            cands.append({"field": p, "value": v, "why": "equal at one fewer decimal"})
        elif abs(abs(n) - av) / av < 0.05:
            cands.append({"field": p, "value": v, "why": "within 5%"})
    if not cands:
        return "UNMATCHED", []
    lo = max(0, span[0] - ASSERTION_WINDOW)
    hi = min(len(text), span[1] + ASSERTION_WINDOW)
    if not ASSERTION.search(text[lo:hi]):
        return "NEAR_NO_CLAIM", []
    return "STALE", cands[:4]


def _repo_of(p: Path) -> str:
    """Which repo a scanned artifact belongs to. File names collide across repos --
    every one of the five has some form of report json -- so a finding that says only
    'seed_floor.json' is ambiguous."""
    try:
        return p.resolve().relative_to(ESTATE).parts[0]
    except ValueError:
        return "?"


def _dp(x: float) -> int:
    s = repr(x)
    return len(s.split(".")[1]) if "." in s else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any STALE candidate is found")
    args = ap.parse_args()

    # EXCLUDE THIS SCRIPT'S OWN OUTPUT. The first run wrote internal_prose_audit.json
    # into data/, which the next run then scanned -- and every finding carries a
    # `context` string quoting the prose it flagged, so each real hit reappeared as a
    # fresh hit against the quoted copy. 13 of the first 33 candidates were the script
    # eating its own tail. A checker that reports its own output as a defect is
    # measuring itself, which is the failure mode this whole file is about.
    #
    # EXCLUDE UPSTREAM TEXT. wikipedia_bios.json holds Wikipedia extracts, not prose
    # anyone here wrote. "141 international victories" and "116 touchdown receptions"
    # are facts about athletes that happen to sit near an unrelated numeric field. There
    # is nothing to keep in sync, so flagging them is pure noise.
    # EXCLUDE DATA TABLES. full_history_universe.json is 7,370 rows of ticker/company
    # records; its "prose" is company NAMES. The estate scan flagged a company whose name
    # contains 1000 as being near an unrelated field of 957. There is nothing in a data
    # table to keep in sync with anything, so every hit is noise by construction.
    EXCLUDE = {OUT.name, "wikipedia_bios.json", "cultural_text.json",
               "full_history_universe.json", "universe.json", "officers.json",
               "merged_careers.json", "tennis_entities.json"}
    files = []
    for d in SCAN_DIRS:
        if d.exists():
            files += sorted(p for p in d.glob("*.json")
                            if p.is_file() and p.name not in EXCLUDE)

    tally = {"EXACT": 0, "STALE": 0, "NEAR_NO_CLAIM": 0, "UNMATCHED": 0}
    findings, skipped_files = [], []
    for f in files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            skipped_files.append({"file": f.name, "repo": _repo_of(f),
                                  "error": f"{type(e).__name__}: {e}"})
            continue
        fields = numeric_fields(doc)
        if not fields:
            continue
        for path, text in prose_strings(doc):
            for m in NUM.finditer(text):
                n = float(m.group(1))
                verdict, cands = classify(n, fields, text, m.span())
                tally[verdict] += 1
                if verdict == "STALE":
                    i = max(0, m.start() - 60)
                    findings.append({
                        "file": f.name,
                        "repo": _repo_of(f),
                        "prose_path": path, "number_in_prose": n,
                        "context": text[i:m.end() + 60].replace("\n", " "),
                        "near_fields": cands,
                    })

    out = {
        "question": "Does any artifact's prose quote a number that disagrees with that "
                    "same artifact's own fields?",
        "found_by": "hand, while auditing promotion gates — gate_nonvacuity.json's "
                    "note_rank says the shuffles 'both score 12.4, exactly the real "
                    "value' while its own three fields say 12.1.",
        "files_scanned": len(files), "files_unreadable": skipped_files,
        "prose_numbers_classified": tally,
        "what_STALE_means": "matches NO field exactly, but is within rounding at one "
                            "fewer decimal place of a field, or within 5% of one. That "
                            "is the shape a number takes when the artifact was "
                            "regenerated and its summary was not.",
        "what_NEAR_NO_CLAIM_means": "near a field, but the surrounding clause makes no "
            "identity claim. Counted, NOT reported. The first version of this script had "
            "no such requirement and produced 134 hits over 75 artifacts, almost all "
            "ranges ('persistence is 0.85-0.92'), thresholds ('below the 0.01 bar') and "
            "counts ('a 200-draw simulation') — reproducing the exact false-positive "
            "failure this script's docstring cites check_cited_fields.py for.",
        "what_UNMATCHED_means": "nothing close in this file. Counted, NOT reported — "
                                "almost always a legitimate external reference (a year, "
                                "another repo's figure, a count of something not stored). "
                                "Reporting these is what produced 43 false-positive "
                                "MISSING fields in check_cited_fields.py's first three "
                                "versions.",
        "this_is_narrow_on_purpose": "Requiring an identity claim within 55 characters "
            "means a stale number written without an assertion is MISSED. Recall is not "
            "measured and is certainly below 1.0. The alternative was 134 candidates of "
            "which almost none were defects, which is worse than a known gap.",
        "a_STALE_hit_is_a_candidate_not_a_verdict": "Two unrelated quantities can sit "
            "within 5% of each other by chance. Each hit carries its context and the "
            "fields it is near so it can be judged rather than believed.",
        "findings": findings,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"estate: {len(SIBLINGS)} sibling repos discovered — "
          f"{', '.join(SIBLINGS)}")
    print(f"scanned {len(files)} artifacts")
    print(f"  prose numbers: EXACT {tally['EXACT']}  STALE {tally['STALE']}  "
          f"NEAR_NO_CLAIM {tally['NEAR_NO_CLAIM']}  UNMATCHED {tally['UNMATCHED']}"
          f"   (last two counted, not reported)")
    for x in findings[:25]:
        near = ", ".join("{}={}".format(c["field"], c["value"])
                         for c in x["near_fields"][:2])
        print(f"\n  {x['file']} {x['prose_path']}")
        print(f"    prose says {x['number_in_prose']}  near: {near}")
        print(f"    ...{x['context'][:110]}...")
    print(f"\nwrote {OUT}")
    if args.check and findings:
        print(f"CHECK FAILED: {len(findings)} STALE candidate(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
