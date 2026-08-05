#!/usr/bin/env python3
"""What free source can carry a pitch EXPECTATION signal, and at what coverage? (7.9)

Solo personal project, no connection to employer, built with public/free-tier only

7.9 sat blocked on "needs transfer values (API-Football, blocked on a key)" for the whole
phase. That is not a blocker, it is a re-scoping trigger: no paid API, no key, find a free
path or say plainly that there isn't one. This measures the free path before anything is
built on it, which is the order that 7.10-7.13 established the hard way.

WHAT "EXPECTATION" HAS TO MEAN HERE. T0/T1 in hoops and gridiron is standing vs a MARKET
VALUATION MADE BEFORE THE PERFORMANCE — the draft slot. Football has no draft, so the
question is not "what number do we have" but "what free number is the same KIND of thing".
Candidates, each measured rather than assumed:

  age_at_context   Wikidata P569 -> age during the tournament/season. Free, should be
                   high-coverage, and non-circular. But it is NOT a market valuation: it
                   is a developmental prior. Comparing an age-based expectation against a
                   draft-slot expectation would repeat 7.7b exactly, where the whole
                   cross-sport finding turned out to be construct mismatch.

  club_count       P54 statement count before the context — a crude "how established".
                   Confounded by Wikidata completeness, which 7.11 showed dominates.

  senior_debut     earliest P54 start qualifier. Same completeness confound.

  sitelinks        REJECTED IN ADVANCE, not measured as a candidate. 7.11 established this
                   is a notability signal, and 7.12 established the market layer is a
                   star-only sample. Using article count as "expectation" would launder
                   fame into a prior.

WHAT THIS DOES NOT DO. It builds no axis and makes no cross-sport claim. It answers one
question: which free fields exist for these 2,430 player-seasons, and for how many.

    python pipeline/probe_pitch_expectation_sources.py
    python pipeline/probe_pitch_expectation_sources.py --limit 400   # quick pass
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from portable_paths import ESTATE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PITCH = ESTATE / "vector-pitch/assets/pitch_mtnn_embeddings.json"
OUT = ROOT / "data" / "pitch_expectation_sources.json"

ENDPOINT = "https://query.wikidata.org/sparql"
UA = "vector-unified/0.1 (personal research; contact via github)"
FOOTBALLER = "Q937857"          # association football player — verified via wbgetentities
BATCH = 150                     # names per SPARQL query; 2,430 names -> ~17 queries
SLEEP = 1.0                     # polite pacing against a free public endpoint
MIN_AGE, MAX_AGE = 15, 45       # a candidate outside this at their first context is a
                                # different person with the same name, not a footballer
                                # with a surprising birthday


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'\u2019-]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def context_year(ctx: str) -> int | None:
    m = re.search(r"(\d{4})", str(ctx))
    return int(m.group(1)) if m else None


def query(names: list[str]) -> list[dict]:
    """One batch. Matches on rdfs:label AND skos:altLabel so 'Mousa Dembele' style
    variants are not silently missed — a name join that only tries one spelling reports
    absence when it means it did not look."""
    values = " ".join(f'"{n}"@en' for n in names)
    q = f"""
SELECT ?item ?itemLabel ?name ?dob (COUNT(DISTINCT ?club) AS ?clubs)
       (MIN(?start) AS ?first_club_start) WHERE {{
  VALUES ?name {{ {values} }}
  {{ ?item rdfs:label ?name }} UNION {{ ?item skos:altLabel ?name }}
  ?item wdt:P106 wd:{FOOTBALLER} .
  OPTIONAL {{ ?item wdt:P569 ?dob }}
  OPTIONAL {{ ?item p:P54 ?st . ?st ps:P54 ?club .
             OPTIONAL {{ ?st pq:P580 ?start }} }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
GROUP BY ?item ?itemLabel ?name ?dob
"""
    r = requests.get(ENDPOINT, params={"query": q, "format": "json"},
                     headers={"User-Agent": UA,
                              "Accept": "application/sparql-results+json"}, timeout=180)
    r.raise_for_status()
    return r.json()["results"]["bindings"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="only probe the first N players")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not PITCH.exists():
        print(f"missing {PITCH}")
        return 2
    rows = json.loads(PITCH.read_text(encoding="utf-8"))["players"]
    if args.limit:
        rows = rows[:args.limit]

    # de-duplicate by name: the corpus is player-CONTEXT, so a player in three tournaments
    # is three rows and one Wikidata lookup.
    by_name: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_name[r["name"]].append(r)
    names = sorted(by_name)
    print(f"{len(rows)} player-contexts -> {len(names)} distinct names")

    candidates: dict[str, list[dict]] = collections.defaultdict(list)
    for i in range(0, len(names), BATCH):
        chunk = names[i:i + BATCH]
        try:
            res = query(chunk)
        except Exception as e:                                    # noqa: BLE001
            print(f"  batch {i // BATCH + 1} failed: {e}")
            continue
        for b in res:
            nm = b.get("name", {}).get("value")
            if not nm:
                continue
            candidates[nm].append({
                "qid": b["item"]["value"].rsplit("/", 1)[-1],
                "dob": (b.get("dob") or {}).get("value"),
                "clubs": int((b.get("clubs") or {}).get("value") or 0),
                "first_club_start": (b.get("first_club_start") or {}).get("value"),
            })
        print(f"  batch {i // BATCH + 1}/{(len(names) - 1) // BATCH + 1}: "
              f"{len(candidates)} names with at least one candidate", flush=True)
        time.sleep(SLEEP)

    # ---- AMBIGUITY IS NOT RESOLVED BY TAKING THE FIRST ROW -----------------------
    # The first version of this loop did `if nm in found: continue`, which silently
    # accepted whichever candidate SPARQL happened to return first. 14 of the first 150
    # names carry more than one: "Cristiano Ronaldo" matches the player (Q11571, born
    # 1985) AND his son (Q118125023, born 2010), and "Carlos Sánchez" matches four people.
    # That is how an age range of 14-129 appeared in the first run. It is the same naive
    # cross-sport name join that produced 17/17 false matches earlier in this phase, and it
    # is mine this time.
    #
    # A candidate survives only if its date of birth puts the player at a plausible age in
    # the EARLIEST context they appear in. Exactly one survivor -> resolved. More than one
    # -> AMBIGUOUS, and reported as such rather than guessed.
    first_ctx = {nm: min((context_year(r.get("context")) or 9999) for r in grp)
                 for nm, grp in by_name.items()}
    found: dict[str, dict] = {}
    ambiguous: dict[str, int] = {}
    for nm, cands in candidates.items():
        cy = first_ctx.get(nm)
        ok = []
        for c in cands:
            dob = c.get("dob")
            if not (dob and dob[:4].isdigit() and cy and cy != 9999):
                continue
            if MIN_AGE <= cy - int(dob[:4]) <= MAX_AGE:
                ok.append(c)
        if len(ok) == 1:
            found[nm] = ok[0]
        elif len(ok) > 1:
            ambiguous[nm] = len(ok)

    have = collections.Counter()
    age_rows = 0
    ages: list[float] = []
    for nm, group in by_name.items():
        f = found.get(nm)
        if not f:
            continue
        have["qid"] += 1
        if f["dob"]:
            have["dob"] += 1
            yr = int(f["dob"][:4]) if f["dob"][:4].isdigit() else None
            for r in group:
                cy = context_year(r.get("context"))
                if yr and cy:
                    age_rows += 1
                    ages.append(cy - yr)
        if f["clubs"]:
            have["clubs"] += 1
        if f["first_club_start"]:
            have["first_club_start"] += 1

    n = len(names)
    report = {
        "question": "Which FREE field can carry a pitch expectation signal, and at what coverage?",
        "source": "Wikidata SPARQL only — no API key, no paid feed (see the 7.9 re-scope)",
        "player_contexts": len(rows),
        "distinct_names": n,
        "candidates_returned_for": len(candidates),
        "resolved_unambiguously": have["qid"],
        "resolved_pct": round(100.0 * have["qid"] / max(n, 1), 1),
        "ambiguous_names": len(ambiguous),
        "ambiguous_examples": dict(sorted(ambiguous.items(), key=lambda kv: -kv[1])[:6]),
        "ambiguity_note": (
            f"A name is resolved only when exactly ONE candidate puts the player between "
            f"{MIN_AGE} and {MAX_AGE} at their earliest context. Taking the first SPARQL "
            f"row instead matched Cristiano Ronaldo's son and produced an age range of "
            f"14-129 on the first run."),
        "field_coverage_pct_of_names": {
            k: round(100.0 * have[k] / max(n, 1), 1)
            for k in ("dob", "clubs", "first_club_start")},
        "age_at_context": {
            "player_contexts_scorable": age_rows,
            "pct_of_contexts": round(100.0 * age_rows / max(len(rows), 1), 1),
            "min": min(ages) if ages else None, "max": max(ages) if ages else None,
        },
        "construct_warning": (
            "age_at_context is NOT the same construct as a draft slot. A draft slot is a "
            "market valuation made before the performance; age is a developmental prior. "
            "7.7b showed that comparing two different constructs under one name reversed a "
            "cross-sport finding twice. Any T0/T1-style pitch axis built on age must be "
            "labelled WITHIN-PITCH and must not be compared against the hoops or gridiron "
            "draft axes."),
        # Per-name resolution, persisted so a downstream axis joins THIS table rather than
        # re-running the query and re-deriving the ambiguity rules. `ambiguous` names are
        # carried explicitly with their candidate count so a consumer cannot mistake
        # "not resolved" for "not looked up".
        "resolved": {nm: {"qid": f["qid"], "dob": f["dob"], "clubs": f["clubs"]}
                     for nm, f in sorted(found.items())},
        "ambiguous": dict(sorted(ambiguous.items())),
        "rejected_in_advance": {
            "sitelinks / article count": (
                "7.11 showed Wikidata coverage dominates this kind of signal and 7.12 "
                "showed the market layer is a star-only sample. Using fame as a prior "
                "would launder notability into 'expectation'."),
            "transfer fees / market values": (
                "No free source. Transfermarkt is the only real one and scraping it at "
                "corpus scale is neither polite nor reliable. Recorded as absent rather "
                "than approximated."),
        },
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nresolved to Wikidata: {have['qid']}/{n} ({report['resolved_pct']}%)")
    for k, v in report["field_coverage_pct_of_names"].items():
        print(f"  {k:18} {have[k]:>5} names  {v:>5.1f}%")
    a = report["age_at_context"]
    print(f"\nage_at_context scorable on {a['player_contexts_scorable']}/{len(rows)} "
          f"contexts ({a['pct_of_contexts']}%)   range {a['min']}-{a['max']}")
    print(f"\n{report['construct_warning']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
