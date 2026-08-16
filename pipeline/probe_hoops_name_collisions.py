#!/usr/bin/env python3
"""Date of birth as the identity key the source does not have.

Solo personal project, no connection to employer, built with public/free-tier only

Operator, 2026-08-03: "some players have the same exact name so you need to check dates of
birth and team to get unique players consistently."

`check_merged_careers.py` implements the arithmetic half of that — a player cannot record a
season before he was drafted — and it is definitive where it fires. It is also **blind to
most of the problem**:

    caught      Jaren Jackson Sr./Jr., because Jr.'s draft year postdates Sr.'s seasons
    caught      44 names with more than one draft entry
    MISSED      two same-name players drafted in overlapping eras
    MISSED      any pair where one or both went undrafted

Nothing in `vectors.json` separates them: no suffixes, no team, and its `id` is a row index
(LeBron James has 23). So the identity key has to come from outside, and Wikidata has it
free — the same SPARQL machinery `probe_pitch_expectation_sources.py` already runs.

THE METHOD, and the mistake it is built to avoid. The pitch probe's first version took
whichever candidate SPARQL returned first and reported 90.7% coverage with an age range of
14 to 129, because "Cristiano Ronaldo" matches the player and his son. So here **every**
candidate is kept, and a name is called COLLIDING when two or more distinct QIDs are both
plausible NBA players by birth year. Nothing is merged, nothing is picked.

    PLAUSIBLE = born between (earliest season - MAX_AGE) and (latest season - MIN_AGE)

A name with two plausible people is ambiguous even if this repo cannot say which seasons
belong to whom — and that is the point. The output is a list of names whose careers cannot
be trusted as one person's, to be unioned with the arithmetic set.

CROSS-CHECK IS THE DELIVERABLE. Two independent methods should agree on the cases both can
see. If DOB does not re-find Jaren Jackson, one of the two is wrong and the disagreement is
worth more than either result alone.

AND IT ALREADY PAID, in the direction that matters — by ACQUITTING. `check_merged_careers.py`
calls a name AMBIGUOUS when it holds more than one draft row and its docstring called that
"definitive that two people share the name". It is not. A player who is drafted and does not
sign can re-enter, so one person can hold two rows:

    arvydas sabonis   draft 1985 #77 (Atlanta, voided as underage) + 1986 #24 (Portland)
                      Wikidata: ONE qid, Q297750, born 1964

AMBIGUOUS answers "does this name have more than one draft row", which is a different
question from "are there two people". The year gap is NOT the discriminator either, and that
is worth recording because it looked like one: the ambiguous year-spans are bimodal with an
empty region at 2-3y, five names sitting at exactly 1y. Three of those five are real
collisions —

    justin jackson    3 qids (1990, 1995, 1997)
    marcus williams   3 qids (1985, 1986, 2002)
    larry robinson    2 qids (1951, 1968)

— so a 1-year cut would have been wrong 3 times out of 5. The empty 2-3y region is a
five-sample artifact, not a separation. DOB is the arbiter; a span is not.

    python pipeline/probe_hoops_name_collisions.py
    python pipeline/probe_hoops_name_collisions.py --limit 300   # quick pass
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_hoops_vor_draft_value as B

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "hoops_name_collisions.json"

ENDPOINT = "https://query.wikidata.org/sparql"
UA = "vector-unified/0.1 (personal research; contact via github)"
BASKETBALL = "Q5372"
BATCH = 40
SLEEP = 1.0
MIN_AGE, MAX_AGE = 18, 42
# Tightened from 17-45 after the first run. The wide band let a Bobby Jones born 1962 pass
# as plausible for corpus seasons 2006-2007 — a 44-year-old rookie — and flagged the name
# as colliding on the strength of a player from a different era. 42 is above the oldest
# real NBA seasons and below absurdity.


def query(names: list[str]) -> list[dict]:
    values = " ".join(f'"{n}"@en' for n in names)
    q = f"""
SELECT ?item ?name ?dob WHERE {{
  VALUES ?name {{ {values} }}
  {{ ?item rdfs:label ?name }} UNION {{ ?item skos:altLabel ?name }}
  ?item wdt:P106 wd:Q3665646 .
  OPTIONAL {{ ?item wdt:P569 ?dob }}
  # No P279* traversal and no label service: both make this query heavy enough that
  # Wikidata ends the response early. The occupation is checked directly.
}}
"""
    r = requests.get(
        ENDPOINT,
        params={"query": q, "format": "json"},
        headers={"User-Agent": UA, "Accept": "application/sparql-results+json"},
        timeout=240,
    )
    r.raise_for_status()
    # strict=False: some Wikidata labels carry raw control characters, and the default
    # parser rejects the whole response over one of them. The first run lost both batches
    # to "Invalid control character at: line 1255 column 50" and reported 0 matches, which
    # would have read as a data finding rather than a transport failure.
    return json.loads(r.text, strict=False)["results"]["bindings"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    vec = json.loads(B.VECTORS.read_text(encoding="utf-8"))
    span: dict[str, list[int]] = collections.defaultdict(list)
    display: dict[str, str] = {}
    for p in vec["players"]:
        n = B.norm_name(p["name"])
        span[n].append(B.season_start(str(p["season"])))
        display.setdefault(n, str(p["name"]))
    names = sorted(span)
    if args.limit:
        names = names[: args.limit]
    print(f"{len(names)} distinct corpus names")

    cands: dict[str, list[dict]] = collections.defaultdict(list)
    failed = 0
    for i in range(0, len(names), BATCH):
        chunk = [display[n] for n in names[i : i + BATCH]]
        try:
            res = query(chunk)
        except Exception as e:
            failed += 1
            print(f"  batch {i // BATCH + 1} failed: {str(e)[:70]}")
            continue
        for b in res:
            nm = B.norm_name(b.get("name", {}).get("value", ""))
            cands[nm].append(
                {
                    "qid": b["item"]["value"].rsplit("/", 1)[-1],
                    "dob": (b.get("dob") or {}).get("value"),
                }
            )
        print(
            f"  batch {i // BATCH + 1}/{(len(names) - 1) // BATCH + 1}: " f"{len(cands)} names with a candidate",
            flush=True,
        )
        time.sleep(SLEEP)

    colliding, resolved, unmatched = {}, [], 0
    for n in names:
        cs = cands.get(n) or []
        if not cs:
            unmatched += 1
            continue
        lo, hi = min(span[n]), max(span[n])
        seen: dict[str, int] = {}
        for c in cs:
            dob = c.get("dob")
            if not (dob and dob[:4].lstrip("-").isdigit()):
                continue
            by = int(dob[:4])
            # Plausible for at least one ACTUAL season in the span, not merely for the
            # span's outer envelope: a single [lo-MAX, hi-MIN] window admits anyone who
            # could have played in any year, which over a long career is nearly everyone.
            if any(MIN_AGE <= y - by <= MAX_AGE for y in span[n]):
                seen[c["qid"]] = by
        if len(seen) >= 2:
            colliding[n] = {
                "qids": seen,
                "corpus_seasons": [lo, hi],
                "birth_years": sorted(set(seen.values())),
            }
        elif len(seen) == 1:
            # KEEP THE NAMES, not just the count. Exactly one age-plausible person for a
            # name is positive evidence of a SINGLE person, and that is the only thing that
            # can acquit a name the arithmetic detector flagged as AMBIGUOUS. Arvydas
            # Sabonis holds two draft rows — 1985 #77 voided as underage, 1986 #24 — and one
            # QID born 1964. Storing `resolved` as an int threw away the acquittal.
            resolved.append(n)

    # ---- cross-check against the arithmetic detector -------------------------
    seasons = sorted({str(p["season"]) for p in vec["players"]}, key=B.season_start)
    series, _ = B.vor_series(seasons, B.eligible_pairs(vec))
    draft = json.loads(B.DRAFT.read_text(encoding="utf-8"))["players"]
    # acquit=False: this file COMPUTES the acquittal. Taking the acquitted set here
    # would make the probe converge on its own previous output.
    arith = B.merged_names(series, draft, acquit=False)
    both = sorted(set(colliding) & arith)
    dob_only = sorted(set(colliding) - arith)
    arith_only = sorted(arith - set(colliding))

    # ---- SUFFIX SWEEP, and it exists because the pass above cannot acquit --------
    # The query matches labels EXACTLY, and the corpus display name carries no suffix, so
    # asking for "Glen Rice" never returns Glen Rice Jr. (Q4811246, born 1991) and the name
    # comes back resolved-single. Acquitting on that would clear a genuine father/son pair.
    # `resolved_single_person` above answers "how many exact-label matches are age-plausible",
    # which is a different question from "how many people have this name".
    #
    # Scoped to the arithmetic-flagged names because that is the only place the answer
    # changes a decision — extra candidates can never turn a collision back into one person,
    # so names already colliding need no sweep. 45 names is one batch; sweeping all 2,415
    # across four suffixes would be ~227 batches for no decision it could alter.
    SUFFIXES = ["", " Jr.", " Sr.", " II", " III", " IV"]
    sweep_targets = sorted(arith)
    sweep: dict[str, dict[str, int]] = collections.defaultdict(dict)
    sweep_failed = 0
    probe_names = [display.get(n, n) + s for n in sweep_targets for s in SUFFIXES]
    for i in range(0, len(probe_names), BATCH):
        try:
            res = query(probe_names[i : i + BATCH])
        except Exception as e:
            sweep_failed += 1
            print(f"  sweep batch {i // BATCH + 1} failed: {str(e)[:70]}")
            continue
        for b in res:
            nm = B.norm_name(b.get("name", {}).get("value", ""))
            dob = (b.get("dob") or {}).get("value")
            if not (dob and dob[:4].lstrip("-").isdigit()):
                continue
            by = int(dob[:4])
            if nm in span and any(MIN_AGE <= y - by <= MAX_AGE for y in span[nm]):
                sweep[nm][b["item"]["value"].rsplit("/", 1)[-1]] = by
        time.sleep(SLEEP)

    # A name is ACQUITTED only when the suffix sweep — which had every chance to find a
    # second person and did not — still returns exactly one age-plausible QID.
    acquitted = {n: sweep[n] for n in sweep_targets if len(sweep.get(n, {})) == 1}
    confirmed = {n: sweep[n] for n in sweep_targets if len(sweep.get(n, {})) >= 2}
    unknown = [n for n in sweep_targets if not sweep.get(n)]
    # Excluded = arithmetic flag NOT overturned by DOB. Unknown stays excluded: no evidence
    # is not evidence of one person.
    exclusion = sorted(set(arith) - set(acquitted))

    report = {
        "operator_report": ("Same-exact-name players need date of birth and team to separate. " "Reported 2026-08-03."),
        "source": "Wikidata SPARQL — free, no key",
        "names_probed": len(names),
        "matched_to_wikidata": len(cands),
        "unmatched": unmatched,
        "resolved_single_person": len(resolved),
        # The full list, not a sample. A downstream consumer needs to ask "is THIS name
        # resolved?", which a truncated list answers wrongly and silently — the same trap
        # `colliding` set below: it reported 119 while storing 80, and reading a `False` off
        # it for `marcus williams` nearly acquitted a genuine three-person collision.
        "resolved_names": sorted(resolved),
        "colliding_names": len(colliding),
        "WHAT_COLLIDING_MEANS": (
            "A SUPERSET OF SUSPICION, not a definitive set. It says two basketball players "
            "with this name exist and are both age-plausible for these seasons — it does "
            "NOT say both appear in this corpus. Andrew Wiggins flags because a second "
            "basketball player born 1992 exists somewhere; only one is in the NBA. Treat "
            "these as names to verify, and the ARITHMETIC set in check_merged_careers.py "
            "as the ones already proven."
        ),
        "failed_batches": failed,
        "plausibility_band": [MIN_AGE, MAX_AGE],
        "method_note": (
            "EVERY candidate is kept. The pitch probe's first version took whichever row "
            "SPARQL returned first and reported an age range of 14 to 129, because "
            "Cristiano Ronaldo matches his own son. A name is COLLIDING when two or more "
            "distinct QIDs are both plausible by birth year against the corpus season span. "
            "Nothing is merged and nothing is picked."
        ),
        "SUFFIX_SWEEP": {
            "what": (
                "Re-queries the arithmetic-flagged names across "
                f"{SUFFIXES} because label matching is EXACT and the corpus display "
                "name carries no suffix — asking for 'Glen Rice' never returns Glen "
                "Rice Jr. (Q4811246, b.1991). Without this the bare pass reports him "
                "as one person and the father/son pair is acquitted."
            ),
            "names_swept": len(sweep_targets),
            "failed_batches": sweep_failed,
            "acquitted": {k: v for k, v in sorted(acquitted.items())},
            "confirmed_two_or_more": {k: v for k, v in sorted(confirmed.items())},
            "no_wikidata_answer": unknown,
            "EXCLUSION_SET": exclusion,
            "rule": (
                "EXCLUDE = arithmetic-flagged AND NOT acquitted by the suffix sweep. "
                "A name with no Wikidata answer stays excluded — no evidence is not "
                "evidence of one person. DOB-colliding names that the arithmetic test "
                "did NOT flag are deliberately absent: `colliding` is a superset of "
                "suspicion (a second same-name player existing somewhere does not put "
                "him in this corpus), so it never excludes on its own."
            ),
        },
        "cross_check": {
            "arithmetic_detector_count": len(arith),
            "found_by_both": both,
            "found_by_DOB_only": dob_only[:40],
            "found_by_ARITHMETIC_only": arith_only[:40],
            "n_dob_only": len(dob_only),
            "n_arith_only": len(arith_only),
            "note": (
                "Two independent methods. Agreement on the cases both can see is the "
                "confirmation; DOB-only cases are what the arithmetic test is blind "
                "to — same-name players drafted in overlapping eras, or undrafted. "
                "ARITHMETIC-only cases are names Wikidata has one or zero entries "
                "for, which is a coverage limit of the DOB method, not a refutation."
            ),
        },
        # ALL of them. This was `[:80]` while `colliding_names` reported 119, so the file
        # said 119 and answered questions about 80 — a silent cap that reads as coverage.
        "colliding": dict(sorted(colliding.items())),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"\nmatched {len(cands)}/{len(names)}   unmatched {unmatched}   "
        f"single-person {len(resolved)}   COLLIDING {len(colliding)}"
    )
    print(f"\nsuffix sweep over the {len(sweep_targets)} arithmetic-flagged names:")
    print(f"  ACQUITTED (one person, re-draft) : {len(acquitted)}  " f"{sorted(acquitted)[:5]}")
    print(f"  CONFIRMED (>=2 people)           : {len(confirmed)}  " f"{sorted(confirmed)[:5]}")
    print(f"  no Wikidata answer               : {len(unknown)}  {unknown[:5]}")
    print(f"  EXCLUSION SET                    : {len(exclusion)}")
    print(f"\ncross-check vs the arithmetic detector ({len(arith)} names):")
    print(f"  found by BOTH        : {len(both)}  {both[:6]}")
    print(f"  found by DOB only    : {len(dob_only)}  {dob_only[:6]}")
    print(f"  found by ARITH only  : {len(arith_only)}  {arith_only[:6]}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
