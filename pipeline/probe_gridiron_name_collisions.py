#!/usr/bin/env python3
"""Date of birth as the identity key gridiron's source does not have.

Solo personal project, no connection to employer, built with public/free-tier only

The hoops twin of this file (probe_hoops_name_collisions.py) established that
`merged_names()`'s second test is NOT definitive: ">1 draft entry for the name" answers
"does this name have more than one draft row", not "are there two people". A drafted player
who does not sign can re-enter. 21 of 45 flagged hoops names turned out to be one person.

THE SAME TEST RUNS IN GRIDIRON AND WAS LEFT UNPATCHED. Measured before writing this:

    gridiron merged total                    101
      >1 draft year (OVERTURNABLE)            92     <- 91% of the exclusion set
      season before draft (arithmetic, safe)   9

and 60 careers were being dropped from trajectory_axis_gridiron.json on that basis.

GRIDIRON IS NOT HOOPS AND THE ANSWER SHOULD NOT BE ASSUMED SYMMETRIC. The NFL drafts 262
players a year against the NBA's 60, so genuine same-name collisions are far denser and
close draft years are weak evidence of anything. Real pairs sit at 2-year spans:

    roy williams   2002 (S, Dallas)   + 2004 (WR, Detroit)      two people
    zach miller    2007 (TE, Oakland) + 2009 (TE, Jacksonville) two people

The hoops acquittal rate (47%) is not a prediction for this file. DOB decides.

THE PLAUSIBILITY BAND IS COMPUTED, NOT ASSERTED, and that is the one design change from the
hoops probe. There I wrote `MIN_AGE, MAX_AGE = 17, 45`, watched a 44-year-old rookie sneak
through, and tightened to 18/42 — a threshold tuned against the cases it was meant to judge.
Here the band is derived from names Wikidata returns exactly ONE candidate for, which are
unambiguous by construction, and the percentile is pre-registered before the first run:

    BAND = [floor(p0.5), ceil(p99.5)] of (season - birth_year) over single-candidate names

Pre-registering matters because the band is the only free parameter and a band chosen after
seeing the answers is not evidence. A mis-assigned single candidate would show up as an
outlier age, which is what trimming to 0.5/99.5 is for rather than a claim that the
single-candidate set is clean.

    python pipeline/probe_gridiron_name_collisions.py
    python pipeline/probe_gridiron_name_collisions.py --limit 300   # quick pass
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_vor_draft_value as G  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "gridiron_name_collisions.json"

ENDPOINT = "https://query.wikidata.org/sparql"
UA = "vector-unified/0.1 (personal research; contact via github)"
FOOTBALLER = "Q19204627"  # American football player — verified live, see check_wikidata_qids
BATCH = 40
SLEEP = 1.0
# Pre-registered BEFORE the first run. The band itself is computed from single-candidate
# names; these are the percentiles used to compute it, fixed here so they cannot be nudged
# once the answers are visible.
BAND_LO_PCT, BAND_HI_PCT = 0.5, 99.5
SUFFIXES = ["", " Jr.", " Sr.", " II", " III", " IV"]


def query(names: list[str]) -> list[dict]:
    values = " ".join(f'"{n}"@en' for n in names)
    q = f"""
SELECT ?item ?name ?dob WHERE {{
  VALUES ?name {{ {values} }}
  {{ ?item rdfs:label ?name }} UNION {{ ?item skos:altLabel ?name }}
  ?item wdt:P106 wd:{FOOTBALLER} .
  OPTIONAL {{ ?item wdt:P569 ?dob }}
}}
"""
    r = requests.get(ENDPOINT, params={"query": q, "format": "json"},
                     headers={"User-Agent": UA,
                              "Accept": "application/sparql-results+json"}, timeout=240)
    r.raise_for_status()
    # strict=False: some Wikidata labels carry raw control characters and the default parser
    # rejects the whole response over one of them, which would read as "0 matches" — a data
    # finding — rather than as the transport failure it is.
    return json.loads(r.text, strict=False)["results"]["bindings"]


def sweep(names: list[str], display: dict[str, str]) -> tuple[dict, int]:
    """Query each name across SUFFIXES. Returns {norm_name: {qid: birth_year}}, failures."""
    out: dict[str, dict[str, int]] = collections.defaultdict(dict)
    failed = 0
    probe = [display.get(n, n) + s for n in names for s in SUFFIXES]
    for i in range(0, len(probe), BATCH):
        try:
            res = query(probe[i:i + BATCH])
        except Exception as e:                                        # noqa: BLE001
            failed += 1
            print(f"  sweep batch {i // BATCH + 1} failed: {str(e)[:70]}")
            continue
        for b in res:
            dob = (b.get("dob") or {}).get("value")
            if not (dob and dob[:4].lstrip("-").isdigit()):
                continue
            nm = G.norm_name(b.get("name", {}).get("value", ""))
            out[nm][b["item"]["value"].rsplit("/", 1)[-1]] = int(dob[:4])
        time.sleep(SLEEP)
    return out, failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    gvec = json.loads(G.GRID_VEC.read_text(encoding="utf-8"))["players"]
    with G.DRAFT_CSV.open(encoding="utf-8", errors="replace", newline="") as fh:
        drows = list(csv.DictReader(fh))
    G.configure_norm([p["name"] for p in gvec],
                     [(r.get("pfr_player_name") or "").strip() for r in drows
                      if (r.get("pfr_player_name") or "").strip()])

    span: dict[str, list[int]] = collections.defaultdict(list)
    display: dict[str, str] = {}
    for p in gvec:
        n = G.norm_name(p["name"])
        span[n].append(int(p["season"]))
        display.setdefault(n, str(p["name"]))
    names = sorted(span)
    if args.limit:
        names = names[:args.limit]
    print(f"{len(names)} distinct corpus names")

    # ---- stage 1: bare pass, every candidate kept ------------------------------
    cands: dict[str, dict[str, int]] = collections.defaultdict(dict)
    failed = 0
    for i in range(0, len(names), BATCH):
        chunk = [display[n] for n in names[i:i + BATCH]]
        try:
            res = query(chunk)
        except Exception as e:                                        # noqa: BLE001
            failed += 1
            print(f"  batch {i // BATCH + 1} failed: {str(e)[:70]}")
            continue
        for b in res:
            dob = (b.get("dob") or {}).get("value")
            if not (dob and dob[:4].lstrip("-").isdigit()):
                continue
            nm = G.norm_name(b.get("name", {}).get("value", ""))
            cands[nm][b["item"]["value"].rsplit("/", 1)[-1]] = int(dob[:4])
        print(f"  batch {i // BATCH + 1}/{(len(names) - 1) // BATCH + 1}: "
              f"{len(cands)} names with a candidate", flush=True)
        time.sleep(SLEEP)

    # ---- stage 2: COMPUTE the band from single-candidate names -----------------
    ages = [s - by
            for n, qs in cands.items() if len(qs) == 1 and n in span
            for by in qs.values() for s in span[n]]
    if len(ages) < 500:
        print(f"only {len(ages)} age observations from single-candidate names — refusing "
              f"to derive a band from that. Nothing written.")
        return 2
    ages.sort()
    lo = math.floor(statistics.quantiles(ages, n=1000)[int(BAND_LO_PCT * 10) - 1])
    hi = math.ceil(statistics.quantiles(ages, n=1000)[int(BAND_HI_PCT * 10) - 1])
    print(f"\nband computed from {len(ages)} ages over "
          f"{sum(1 for qs in cands.values() if len(qs) == 1)} single-candidate names: "
          f"[{lo}, {hi}]  (p{BAND_LO_PCT}/p{BAND_HI_PCT}, pre-registered)")

    def plausible(n: str, by: int) -> bool:
        return any(lo <= s - by <= hi for s in span[n])

    # ---- stage 3: collide ------------------------------------------------------
    colliding, resolved, unmatched = {}, [], 0
    for n in names:
        qs = cands.get(n) or {}
        if not qs:
            unmatched += 1
            continue
        seen = {q: by for q, by in qs.items() if plausible(n, by)}
        if len(seen) >= 2:
            colliding[n] = {"qids": seen, "corpus_seasons": [min(span[n]), max(span[n])],
                            "birth_years": sorted(set(seen.values()))}
        elif len(seen) == 1:
            resolved.append(n)

    # ---- stage 4: suffix sweep over the arithmetic-flagged names ---------------
    gs: dict[str, list] = collections.defaultdict(list)
    for p in gvec:
        ppr = (p.get("ppg") or {}).get("ppr")
        if ppr is not None:
            gs[G.norm_name(p["name"])].append((int(p["season"]), float(ppr)))
    # acquit=False: this file COMPUTES the acquittal. Taking the acquitted set here would
    # make the probe converge on its own previous output — each run acquitting from a
    # smaller flagged set than the last, a ratchet that never re-examines a past decision.
    arith = G.merged_names(gs, G.DRAFT_CSV, acquit=False)

    dyears: dict[str, set] = collections.defaultdict(set)
    for r in drows:
        nm = (r.get("pfr_player_name") or "").strip()
        if nm and r.get("season"):
            dyears[G.norm_name(nm)].add(int(r["season"]))
    # A season strictly before the earliest draft year is arithmetic and is NEVER acquitted
    # — no re-draft explains it. Only the ">1 draft year" half is overturnable.
    impossible = {n for n in arith if len(dyears.get(n, ())) <= 1}
    overturnable = sorted(arith - impossible)

    swept, sweep_failed = sweep(overturnable, display)
    acquitted, confirmed, unknown = {}, {}, []
    for n in overturnable:
        seen = {q: by for q, by in (swept.get(n) or {}).items() if plausible(n, by)}
        if len(seen) == 1:
            acquitted[n] = seen
        elif len(seen) >= 2:
            confirmed[n] = seen
        else:
            unknown.append(n)
    exclusion = sorted(arith - set(acquitted))

    report = {
        "source": "Wikidata SPARQL — free, no key",
        "occupation_qid": FOOTBALLER,
        "names_probed": len(names),
        "matched_to_wikidata": len(cands),
        "unmatched": unmatched,
        "resolved_single_person": len(resolved),
        "resolved_names": sorted(resolved),
        "colliding_names": len(colliding),
        "failed_batches": failed,
        "PLAUSIBILITY_BAND": {
            "band": [lo, hi],
            "percentiles": [BAND_LO_PCT, BAND_HI_PCT],
            "n_age_observations": len(ages),
            "derived_from": ("names Wikidata returns exactly ONE candidate for, which are "
                             "unambiguous by construction"),
            "why_computed": (
                "The hoops probe asserted 17/45, watched a 44-year-old rookie pass, and "
                "tightened to 18/42 — a threshold tuned against the cases it judges. The "
                "percentiles here were fixed in the docstring before the first run; the "
                "band is whatever the data says. A mis-assigned single candidate shows up "
                "as an outlier age, which is what trimming to 0.5/99.5 handles."),
        },
        "WHAT_COLLIDING_MEANS": (
            "A SUPERSET OF SUSPICION. It says two American football players with this name "
            "exist and are both age-plausible for these seasons — NOT that both appear in "
            "this corpus. It never excludes on its own; only the arithmetic set does."),
        "SUFFIX_SWEEP": {
            "what": (f"Re-queries the overturnable names across {SUFFIXES}. Label matching "
                     "is EXACT and the corpus display name carries no suffix, so a bare "
                     "query for 'Marvin Harrison' cannot see Marvin Harrison Jr. Without "
                     "this the bare pass reports one person and acquits a father/son pair."),
            "arithmetic_flagged": len(arith),
            "never_acquittable_season_before_draft": sorted(impossible),
            "overturnable": len(overturnable),
            "names_swept": len(overturnable),
            "failed_batches": sweep_failed,
            "acquitted": dict(sorted(acquitted.items())),
            "confirmed_two_or_more": dict(sorted(confirmed.items())),
            "no_wikidata_answer": unknown,
            "EXCLUSION_SET": exclusion,
            "rule": ("EXCLUDE = arithmetic-flagged AND NOT acquitted. A name with no "
                     "Wikidata answer stays excluded — no evidence is not evidence of one "
                     "person."),
        },
        "cross_check": {
            "found_by_both": sorted(set(colliding) & arith),
            "found_by_DOB_only": sorted(set(colliding) - arith)[:40],
            "found_by_ARITHMETIC_only": sorted(arith - set(colliding))[:40],
            "n_dob_only": len(set(colliding) - arith),
            "n_arith_only": len(arith - set(colliding)),
        },
        "colliding": dict(sorted(colliding.items())),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nmatched {len(cands)}/{len(names)}   unmatched {unmatched}   "
          f"single-person {len(resolved)}   COLLIDING {len(colliding)}")
    print(f"\narithmetic-flagged {len(arith)}  "
          f"(season-before-draft {len(impossible)}, overturnable {len(overturnable)})")
    print(f"  ACQUITTED (one person)  : {len(acquitted)}  {sorted(acquitted)[:5]}")
    print(f"  CONFIRMED (>=2 people)  : {len(confirmed)}  {sorted(confirmed)[:5]}")
    print(f"  no Wikidata answer      : {len(unknown)}  {unknown[:5]}")
    print(f"  EXCLUSION SET           : {len(exclusion)}   (was {len(arith)})")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
