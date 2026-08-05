#!/usr/bin/env python3
"""A CORRECTION key must reach the text it corrects, not just sit beside it.

Solo personal project, no connection to employer, built with public/free-tier only

WHY THIS EXISTS: I made the same mistake three times in one session, and only noticed
the third because I happened to be reading around an unrelated false positive.

When a claim in an artifact turns out to be wrong, the instinct "annotate rather than
silently edit" is right -- a silent edit destroys the record of what was believed and
why. But adding a CORRECTION_* key and stopping there leaves the WRONG SENTENCE fully
readable as a standalone assertion, and nothing in the shape of the document tells a
reader to keep going:

  vector-hoops/pipeline/seed_floor.json
     REPO_STATE_WARNING.also_destroyed      "...and there is no backup"
     CORRECTION_embedding_v3_backups_exist  "Eight backups sit beside it"

  vector-unified/data/gate_nonvacuity.json
     note_rank                              "both score 12.4, exactly the real value"
     CORRECTION_note_rank_prose_disagrees   "its own fields say 12.1"

  vector-unified/data/gridiron_forward_report.json
     why_per_position_is_the_finding        "Pooled persistence is 0.7642 against 0.58-0.74"
     CORRECTION_why_per_position_..._stale  "0.7642 appears in NO field of this file"

A reader who stops at REPO_STATE_WARNING -- the key whose NAME says to read it -- acts on
the wrong fact. The correction is worse than useless there: it makes the file look
audited while the audited claim still stands.

THE RULE THIS ENFORCES. If a document contains a CORRECTION_*/CORRECTED_* key, then the
text that key corrects must carry an inline marker (CORRECTED / SUPERSEDED / see
CORRECTION_) at the point of the wrong claim. The correction key stays as the record of
why; the inline marker is what a linear reader actually hits.

HOW IT DECIDES WHAT WAS CORRECTED. A correction block usually quotes the wrong claim
verbatim in a field like prose_says / corrects / verdict_as_shipped. Those quotes are the
handle. For each quoted fragment long enough to be distinctive, the document is searched
for that fragment OUTSIDE the correction block. If it is found and the surrounding text
carries no marker, the correction did not land.

WHAT IT DELIBERATELY DOES NOT DO. It does not judge whether a correction is right, and it
does not flag a correction block that quotes nothing -- some corrections describe a
process ("the first version of this audit was incomplete") rather than replacing a
sentence, and there is no wrong text to reach. Those are counted, not reported.

THE BLIND SPOT, FOUND BY TRIPPING OVER IT AN HOUR AFTER WRITING THIS FILE. This check can
only see an UNLANDED correction. It cannot see a DELETED one, and deletion is the routine
case rather than the exotic one:

    gate_nonvacuity.json and gridiron_forward_report.json are GENERATED. Running
    validate.py regenerates both. The inline [CORRECTED: ...] markers and the
    CORRECTION_* keys written into them were wiped on the next validate run -- and this
    check went GREEN, because with the correction key gone there was nothing left to
    check. Zero findings, for the worst possible reason.

So a green here means "no correction failed to land", NOT "every correction is intact".
The lesson is upstream of the check: A GENERATED FILE CANNOT HOLD ITS OWN CORRECTION.
Annotating the artifact is futile; the fix belongs in the generator, and the durable form
is to INTERPOLATE the value rather than retype it, so the prose cannot drift from the
numbers it describes. Both files above were fixed that way — see the comments at
check_gate_nonvacuity.py's note_rank and build_gridiron_forward.py's
why_per_position_is_the_finding.

    python pipeline/check_corrections_landed.py            # report
    python pipeline/check_corrections_landed.py --check     # exit 1 if any did not land

Writes: data/corrections_landed_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ESTATE = ROOT.parent
OUT = ROOT / "data" / "corrections_landed_audit.json"

SCAN_DIRS = [ROOT / "data", ROOT / "data" / "market_cultural"]
for _sib in ("vector-hoops", "vector-gridiron", "vector-pitch", "vector-equities"):
    SCAN_DIRS += [ESTATE / _sib / "pipeline" / "data", ESTATE / _sib / "pipeline"]

CORRECTION_KEY = re.compile(r"CORRECTION|CORRECTED|SUPERSEDED", re.I)

# Fields inside a correction block that quote the wrong claim verbatim.
QUOTE_FIELDS = ("prose_says", "corrects", "verdict_as_shipped", "was", "old",
                "previously", "said", "claimed", "original")

# An inline marker at the point of the wrong claim. This is what a linear reader hits.
MARKER = re.compile(r"CORRECTED|SUPERSEDED|see CORRECTION_|\[STALE", re.I)

# Short fragments match everywhere and prove nothing.
MIN_FRAGMENT = 25


def walk(obj, path=""):
    """(path, key, value) for every dict entry in the document."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")


def all_strings(obj, path="", skip_prefix=None):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}"
            if skip_prefix and p.startswith(skip_prefix):
                continue
            out += all_strings(v, p, skip_prefix)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += all_strings(v, f"{path}[{i}]", skip_prefix)
    elif isinstance(obj, str):
        out.append((path, obj))
    return out


def quoted_claims(block) -> list[str]:
    """Verbatim fragments of the wrong claim, from a correction block."""
    out = []
    if isinstance(block, str):
        # a bare-string correction: pull anything in double quotes
        out += [m.group(1) for m in re.finditer(r'"([^"]{25,})"', block)]
    elif isinstance(block, dict):
        for k, v in block.items():
            if isinstance(v, str) and any(q in k.lower() for q in QUOTE_FIELDS):
                out.append(v)
                out += [m.group(1) for m in re.finditer(r'"([^"]{25,})"', v)]
    return [s.strip() for s in out if len(s.strip()) >= MIN_FRAGMENT]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any correction did not reach its target text")
    args = ap.parse_args()

    files = []
    for d in SCAN_DIRS:
        if d.exists():
            files += sorted(p for p in d.glob("*.json")
                            if p.is_file() and p.name != OUT.name)

    findings, n_blocks, n_no_quote, n_landed = [], 0, 0, 0
    for f in files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            repo = f.resolve().relative_to(ESTATE).parts[0]
        except ValueError:
            repo = "?"
        for parent, key, val in walk(doc):
            if not CORRECTION_KEY.search(key):
                continue
            n_blocks += 1
            block_path = f"{parent}.{key}"
            claims = quoted_claims(val)
            if not claims:
                n_no_quote += 1
                continue
            outside = all_strings(doc, "", skip_prefix=block_path)
            unlanded = []
            for c in claims:
                # match on a distinctive middle slice, so trivial reformatting
                # (a comma, a wrapped line) does not defeat it
                frag = c[:80]
                for path, text in outside:
                    if frag in text and not MARKER.search(text):
                        unlanded.append({"claim_fragment": frag,
                                         "still_asserted_at": path,
                                         "context": text[max(0, text.find(frag) - 40):
                                                         text.find(frag) + 140]})
                        break
            if unlanded:
                findings.append({"repo": repo, "file": f.name,
                                 "correction_key": block_path, "unlanded": unlanded})
            else:
                n_landed += 1

    out = {
        "question": "Does every CORRECTION key reach the text it corrects, or does the "
                    "wrong claim still stand un-annotated in the same file?",
        "why": "Adding a correction key and stopping there leaves the wrong sentence "
               "fully readable as a standalone assertion. A reader who stops at the key "
               "whose NAME says to read it acts on the wrong fact, and the file looks "
               "audited while the audited claim still stands. I made this mistake three "
               "times in one session and caught the third by accident.",
        "files_scanned": len(files),
        "correction_blocks_found": n_blocks,
        "blocks_that_quote_no_claim": n_no_quote,
        "blocks_whose_correction_landed": n_landed,
        "blocks_that_did_NOT_land": len(findings),
        "what_counts_as_landed": "an inline marker (CORRECTED / SUPERSEDED / see "
                                 "CORRECTION_) at the point of the wrong claim. The "
                                 "correction key stays as the record of why; the marker "
                                 "is what a linear reader hits.",
        "not_reported": "correction blocks that quote nothing. Some describe a process "
                        "('the first version of this audit was incomplete') rather than "
                        "replacing a sentence, so there is no wrong text to reach. "
                        "Counted, not flagged.",
        "findings": findings,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"scanned {len(files)} artifacts")
    print(f"  correction blocks {n_blocks}  landed {n_landed}  "
          f"quote-nothing {n_no_quote}  DID NOT LAND {len(findings)}")
    for x in findings:
        print(f"\n  {x['repo']}/{x['file']}  {x['correction_key']}")
        for u in x["unlanded"]:
            print(f"    still asserted at {u['still_asserted_at']}")
            print(f"      ...{u['context'][:120]}...")
    print(f"\nwrote {OUT}")
    if args.check and findings:
        print(f"CHECK FAILED: {len(findings)} correction(s) did not land", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
