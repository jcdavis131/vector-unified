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

import build_hoops_vor_draft_value as B  # noqa: E402

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
    r = requests.get(ENDPOINT, params={"query": q, "format": "json"},
                     headers={"User-Agent": UA,
                              "Accept": "application/sparql-results+json"}, timeout=240)
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
        names = names[:args.limit]
    print(f"{len(names)} distinct corpus names")

    cands: dict[str, list[dict]] = collections.defaultdict(list)
    failed = 0
    for i in range(0, len(names), BATCH):
        chunk = [display[n] for n in names[i:i + BATCH]]
        try:
            res = query(chunk)
        except Exception as e:                                    # noqa: BLE001
            failed += 1
            print(f"  batch {i // BATCH + 1} failed: {str(e)[:70]}")
            continue
        for b in res:
            nm = B.norm_name(b.get("name", {}).get("value", ""))
            cands[nm].append({"qid": b["item"]["value"].rsplit("/", 1)[-1],
                              "dob": (b.get("dob") or {}).get("value")})
        print(f"  batch {i // BATCH + 1}/{(len(names) - 1) // BATCH + 1}: "
              f"{len(cands)} names with a candidate", flush=True)
        time.sleep(SLEEP)

    colliding, resolved, unmatched = {}, 0, 0
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
            colliding[n] = {"qids": seen, "corpus_seasons": [lo, hi],
                            "birth_years": sorted(set(seen.values()))}
        elif len(seen) == 1:
            resolved += 1

    # ---- cross-check against the arithmetic detector -------------------------
    seasons = sorted({str(p["season"]) for p in vec["players"]}, key=B.season_start)
    series, _ = B.vor_series(seasons, B.eligible_pairs(vec))
    draft = json.loads(B.DRAFT.read_text(encoding="utf-8"))["players"]
    arith = B.merged_names(series, draft)
    both = sorted(set(colliding) & arith)
    dob_only = sorted(set(colliding) - arith)
    arith_only = sorted(arith - set(colliding))

    report = {
        "operator_report": (
            "Same-exact-name players need date of birth and team to separate. "
            "Reported 2026-08-03."),
        "source": "Wikidata SPARQL — free, no key",
        "names_probed": len(names),
        "matched_to_wikidata": len(cands),
        "unmatched": unmatched,
        "resolved_single_person": resolved,
        "colliding_names": len(colliding),
        "WHAT_COLLIDING_MEANS": (
            "A SUPERSET OF SUSPICION, not a definitive set. It says two basketball players "
            "with this name exist and are both age-plausible for these seasons — it does "
            "NOT say both appear in this corpus. Andrew Wiggins flags because a second "
            "basketball player born 1992 exists somewhere; only one is in the NBA. Treat "
            "these as names to verify, and the ARITHMETIC set in check_merged_careers.py "
            "as the ones already proven."),
        "failed_batches": failed,
        "plausibility_band": [MIN_AGE, MAX_AGE],
        "method_note": (
            "EVERY candidate is kept. The pitch probe's first version took whichever row "
            "SPARQL returned first and reported an age range of 14 to 129, because "
            "Cristiano Ronaldo matches his own son. A name is COLLIDING when two or more "
            "distinct QIDs are both plausible by birth year against the corpus season span. "
            "Nothing is merged and nothing is picked."),
        "cross_check": {
            "arithmetic_detector_count": len(arith),
            "found_by_both": both,
            "found_by_DOB_only": dob_only[:40],
            "found_by_ARITHMETIC_only": arith_only[:40],
            "n_dob_only": len(dob_only), "n_arith_only": len(arith_only),
            "note": ("Two independent methods. Agreement on the cases both can see is the "
                     "confirmation; DOB-only cases are what the arithmetic test is blind "
                     "to — same-name players drafted in overlapping eras, or undrafted. "
                     "ARITHMETIC-only cases are names Wikidata has one or zero entries "
                     "for, which is a coverage limit of the DOB method, not a refutation."),
        },
        "colliding": dict(sorted(colliding.items())[:80]),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nmatched {len(cands)}/{len(names)}   unmatched {unmatched}   "
          f"single-person {resolved}   COLLIDING {len(colliding)}")
    print(f"\ncross-check vs the arithmetic detector ({len(arith)} names):")
    print(f"  found by BOTH        : {len(both)}  {both[:6]}")
    print(f"  found by DOB only    : {len(dob_only)}  {dob_only[:6]}")
    print(f"  found by ARITH only  : {len(arith_only)}  {arith_only[:6]}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
