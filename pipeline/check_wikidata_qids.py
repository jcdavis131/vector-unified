#!/usr/bin/env python3
"""Every hard-coded Wikidata QID must still be the entity the code thinks it is.

Solo personal project, no connection to employer, built with public/free-tier only

`SPORT_Q["gridiron"]` was **Q9398 — Grugliasco, an Italian comune** — in TWO files. Every
gridiron query asked Wikidata for people whose sport is a town near Turin, got an empty
result set, and wrote it out. An empty result is indistinguishable from "no matches", so
the failure was silent across two raw artifacts, one derived artifact, and the athlete-side
sponsor census, until the hole was measured from the far end in 7.12.

A QID is an opaque integer. Nothing about `Q9398` looks wrong next to `Q5372` and `Q2736`,
which is exactly why it survived review — the same shape as every other defect this phase:
a real value answering a different question than the one it appears to answer.

So the QIDs are pinned to their expected English labels and checked against live Wikidata.
A label check rather than a type check because it is unambiguous and needs no ontology
reasoning: "Grugliasco" != "American football" fails instantly, and no P31/P279* traversal
would have been needed to see it.

BANNED QIDS ARE SEPARATE FROM UNKNOWN ONES. A registry that only checks identity would
pass a revived Q9398 — "Q9398 is Grugliasco, correct!" — which answers the wrong question.
Q9398 is therefore banned outright rather than registered.

UNREGISTERED QIDS ARE A FAILURE, not a skip. A new hard-coded QID that nobody declared is
precisely the case this exists to prevent, so the checker refuses to pass while one is
unaccounted for. Adding it to EXPECT is a one-line, deliberate act.

    python pipeline/check_wikidata_qids.py
    python pipeline/check_wikidata_qids.py --check   # exit 1 on any mismatch
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PIPE = ROOT / "pipeline"
API = "https://www.wikidata.org/w/api.php"
UA = "vector-unified/0.1 (personal research; contact via github)"

# QID -> the English label Wikidata must still report. Every hard-coded QID in pipeline/
# belongs here. Comments say what the code uses it FOR, which is the part a label alone
# does not tell you.
EXPECT = {
    # sports, used as P641 targets
    "Q5372": "basketball",
    "Q41323": "American football",
    "Q2736": "association football",
    # leagues
    "Q155223": "National Basketball Association",
    "Q1215884": "National Football League",
    # class terms traversed with P31/P279*
    "Q12973014": "sports team",
    "Q476028": "association football club",
    "Q135408445": "men's national association football team",
    "Q43229": "organization",
    "Q4830453": "business",
    "Q783794": "company",
    "Q5": "human",
    # occupation, used to scope the pitch name-resolution query (7.9)
    "Q937857": "association football player",
}

# QIDs that must NEVER appear as an operative literal, with the reason. A registry that
# only checks identity would happily pass a revived Q9398 — "Q9398 is Grugliasco, correct!"
# — which is the wrong question. Some values are banned regardless of whether they resolve.
BANNED = {
    "Q9398": ("Grugliasco, an Italian comune. It sat in SPORT_Q['gridiron'] in two files "
              "and silently emptied every gridiron query. The correct value is Q41323."),
}

# OPERATIVE occurrences only: a SPARQL `wd:` prefix, an entity URL, or a bare quoted
# literal. A plain word-boundary match on Q\d+ is useless here — it hits "Q1." and "Q2."
# question numbering in docstrings and reported eight false positives on the first run,
# which is how a checker teaches people to ignore it.
QID_RE = re.compile(r"""(?:wd:|entity/)(Q\d{1,9})\b|["'](Q\d{1,9})["']""")


def scan() -> dict[str, list[str]]:
    """QID -> ['file:line', ...] for every QID literal under pipeline/."""
    found: dict[str, list[str]] = {}
    for f in sorted(PIPE.glob("*.py")):
        if f.name == Path(__file__).name:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for a, b in QID_RE.findall(line):
                found.setdefault(a or b, []).append(f"{f.name}:{i}")
    return found


def labels(qids: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for i in range(0, len(qids), 50):
        chunk = qids[i:i + 50]
        r = requests.get(API, params={
            "action": "wbgetentities", "ids": "|".join(chunk),
            "props": "labels", "languages": "en", "format": "json"},
            headers={"User-Agent": UA}, timeout=120)
        r.raise_for_status()
        for qid, ent in (r.json().get("entities") or {}).items():
            out[qid] = (None if "missing" in ent
                        else (ent.get("labels") or {}).get("en", {}).get("value"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 on any problem")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    found = scan()
    unregistered = sorted(q for q in found if q not in EXPECT)
    got = labels(sorted(set(found) | set(EXPECT)))

    problems: list[str] = []
    for q in unregistered:
        problems.append(
            f"UNREGISTERED {q} ({got.get(q) or '?'}) at {', '.join(found[q][:3])} — add it "
            f"to EXPECT with its label, or the next wrong QID hides the same way Q9398 did")
    for q, why in sorted(BANNED.items()):
        if q in found:
            problems.append(f"BANNED {q} at {', '.join(found[q][:3])} — {why}")
    for q, want in sorted(EXPECT.items()):
        have = got.get(q)
        if have is None:
            problems.append(f"MISSING {q}: Wikidata has no such entity (expected {want!r})")
        elif have != want:
            problems.append(f"CHANGED {q}: expected {want!r}, Wikidata now says {have!r}")

    print(f"{len(found)} distinct QID literal(s) under pipeline/, "
          f"{len(EXPECT)} registered, {len(BANNED)} banned\n")
    print(f"{'QID':<12} {'label':<42} sites")
    for q in sorted(found, key=lambda x: int(x[1:])):
        tag = "  BANNED" if q in BANNED else ""
        print(f"{q:<12} {str(got.get(q)):<42} {len(found[q])}{tag}")

    if not problems:
        print("\nevery hard-coded QID still resolves to the entity the code expects.")
        return 0
    print(f"\n{len(problems)} problem(s):")
    for p in problems:
        print(f"  {p}")
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
