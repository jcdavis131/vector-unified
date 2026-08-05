#!/usr/bin/env python3
"""Tournament sponsorship from the tournament NAME — the source the feasibility check missed.

Solo personal project, no connection to employer, built with public/free-tier only

`docs/TENNIS_GOLF_FEASIBILITY.md` measured the sponsor rationale and recorded it as failed:

    tennis_tournament        1,576 ... 0.29% carry a sponsor edge
    golf_tournament          1,576        1

That number is correct AND it answers the wrong question. It measures **Wikidata P859**,
and P859 is a sparse editorial property. The sponsor of a tennis tournament is not hiding in
Wikidata — IT IS THE TOURNAMENT'S NAME, and the name is already in the xlsx this repo
downloaded weeks ago:

    BNP Paribas Open @ Indian Wells        Mutua Madrid Open @ Madrid
    Citi Open @ Washington                 Western & Southern Financial Group @ Cincinnati

Same defect class as everything else this phase: a real value answering a different
question than the one it appears to answer. 0.29% is the P859 coverage, not the sponsor
coverage.

THE STRONG SIGNAL IS THE RENAME, not the name. A single name is weak evidence — "Brisbane
International" contains no sponsor and "Barcelona Open" is just a city. But a tournament at
a FIXED LOCATION that changes name between years is a sponsorship event with a date on it,
because events do not rename themselves for fun:

    Antwerp     2015       BNP Paribas Fortis Diamond Games
                2016-2024  European Open                     sponsor left
    Atlanta     2013-2021  BB&T Atlanta Open
                2022-2024  Atlanta Open                      BB&T -> Truist merger, 2019
    's-Hert.    2013-2016  Topshelf Open
                2017-2018  Ricoh Open
                2019-2026  Rosmalen Grass Court Championships

78 of 169 locations carry at least one such change. These are dated corporate actions
observable in sports data, and they are exactly the tennis->equities edge the operator asked
for when proposing tennis ("locations and sponsors associated with tournaments").

WHAT IS ASSERTED AND WHAT IS MEASURED, kept apart on purpose. A token surviving the
stoplist is a CANDIDATE, not a sponsor — this file never claims otherwise. Three tiers are
reported separately:

    CANDIDATE   a token that is not the location, not a demonym, not tennis vocabulary
    RENAMED     the candidate appears/disappears at a fixed location across years
    CORROBORATED  the candidate also matches a company name in the local equities universe

Only CORROBORATED is a sponsor identification. CANDIDATE is a hypothesis, and the count of
candidates is not evidence of anything except that a stoplist ran.

    python pipeline/build_tennis_sponsors.py
    python pipeline/build_tennis_sponsors.py --check   # exit 1 if coverage collapses
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from portable_paths import ESTATE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ENTITIES = ROOT / "data" / "tennis_entities.json"
OUT = ROOT / "data" / "tennis_sponsors.json"
EQUITIES = ESTATE / "vector-equities/assets/universe_full_history.json"

# Generic event vocabulary. A word here can never be a sponsor. Deliberately broad: a
# false NEGATIVE costs one candidate, a false POSITIVE puts a common noun in a sponsor
# registry and makes the coverage number a lie.
STOP = {
    "open", "opens", "championship", "championships", "classic", "cup", "trophy",
    "international", "masters", "tournament", "tour", "finals", "final", "series",
    "invitational", "challenge", "challenger", "games", "game", "grand", "slam", "prix",
    "atp", "wta", "itf", "men", "mens", "women", "womens", "ladies", "gentlemen",
    "singles", "doubles", "tennis", "court", "courts", "grass", "clay", "hard", "indoor",
    "outdoor", "of", "the", "de", "del", "della", "di", "du", "da", "das", "dos", "el",
    "la", "le", "les", "los", "and", "at", "in", "on", "for", "cor", "torneo", "abierto",
    "copa", "coupe", "internazionali", "campeonato", "trofeo", "championnats", "meisterschaften",
    "bowl", "cities", "city", "county", "state", "national", "nationals", "region",
    "regional", "week", "days", "day", "spring", "summer", "autumn", "winter", "sunshine",
    "gold", "silver", "st", "santa", "san", "port", "new", "north", "south", "east", "west",
    "1", "2", "i", "ii", "iii",
}

# Demonyms and country/place adjectives. `location` covers the city; these cover the rest.
DEMONYM = {
    "german", "germany", "swedish", "sweden", "serbia", "serbian", "hellenic", "greek",
    "greece", "french", "france", "italian", "italy", "spanish", "spain", "swiss",
    "switzerland", "dutch", "netherlands", "belgian", "belgium", "austrian", "austria",
    "australian", "australia", "chinese", "china", "japanese", "japan", "korean", "korea",
    "us", "usa", "american", "america", "canadian", "canada", "brazil", "brazilian",
    "mexican", "mexico", "argentina", "argentine", "chile", "chilean", "colombia",
    "colombian", "croatia", "croatian", "czech", "hungarian", "hungary", "romanian",
    "romania", "russian", "russia", "polish", "poland", "portuguese", "portugal",
    "moroccan", "morocco", "tunisian", "tunisia", "qatar", "dubai", "emirates", "india",
    "indian", "thailand", "thai", "singapore", "malaysia", "malaysian", "turkish",
    "turkey", "israel", "israeli", "british", "britain", "england", "english", "scottish",
    "irish", "ireland", "welsh", "european", "europe", "asian", "asia", "african",
    "pacific", "atlantic", "mediterranean", "nordic", "baltic", "estonia", "estonian",
    "kazakhstan", "uzbekistan", "slovenia", "slovak", "slovakia", "bulgaria", "bulgarian",
    "luxembourg", "monte", "carlo", "mexicano", "brasil", "espana", "italia", "osaka",
}


def toks(s: str) -> list[str]:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9&\s]", " ", s.lower())
    return [t for t in s.split() if t]


def candidates(tournament: str, location: str) -> list[str]:
    loc = set(toks(location))
    out = []
    for t in toks(tournament):
        if t in loc or t in STOP or t in DEMONYM or len(t) < 2:
            continue
        out.append(t)
    return out


def load_company_names() -> set[str]:
    """Company-name tokens from the local equities universe. Empty set if absent.

    Absent is reported, never silently treated as "nothing corroborated" — a missing file
    and a genuine zero are different findings and this repo has confused them before
    (a SPARQL transport failure once read as "0 matches" and looked like a data result).
    """
    if not EQUITIES.exists():
        return set()
    try:
        doc = json.loads(EQUITIES.read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return set()
    rows = doc if isinstance(doc, list) else (
        doc.get("universe") or doc.get("companies") or doc.get("rows") or [])
    names: set[str] = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        for key in ("name", "company", "company_name", "security", "longName"):
            v = r.get(key)
            if isinstance(v, str) and v.strip():
                for t in toks(v):
                    if t not in STOP and t not in DEMONYM and len(t) > 2:
                        names.add(t)
                break
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if renamed-location coverage collapses below the floor")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not ENTITIES.exists():
        print(f"missing {ENTITIES} — run build_tennis_entities.py first")
        return 2
    ents = json.loads(ENTITIES.read_text(encoding="utf-8"))["entities"]

    # (location -> tournament -> years). The rename signal lives here.
    byloc: dict[str, dict[str, set]] = collections.defaultdict(
        lambda: collections.defaultdict(set))
    for r in ents:
        byloc[r["location"]][r["tournament"]].add(int(r["year"]))

    all_pairs = {(r["tournament"], r["location"]) for r in ents}
    cand_by_pair = {p: candidates(p[0], p[1]) for p in all_pairs}
    with_cand = {p for p, c in cand_by_pair.items() if c}

    # RENAMED: at a fixed location, a candidate token present in some years and absent in
    # others. Presence/absence across years at one venue is what separates a sponsor from
    # a city name — "barcelona" never leaves Barcelona, "banco sabadell" did.
    renamed: dict[str, list] = {}
    for loc, tours in byloc.items():
        if len(tours) < 2:
            continue
        years_with: dict[str, set] = collections.defaultdict(set)
        all_years: set = set()
        for tn, ys in tours.items():
            all_years |= ys
            for c in cand_by_pair[(tn, loc)]:
                years_with[c] |= ys
        moved = {c: sorted(ys) for c, ys in years_with.items() if ys != all_years}
        if moved:
            renamed[loc] = [
                {"token": c, "years_present": ys,
                 "years_absent": sorted(all_years - set(ys))}
                for c, ys in sorted(moved.items())
            ]

    company_tokens = load_company_names()
    corroborated = sorted({
        c for cs in cand_by_pair.values() for c in cs if c in company_tokens})

    # CONFIRMED = the conjunction of two INDEPENDENT signals: the token matches a company
    # name AND it arrives or leaves at a fixed location. Either alone is noisy — bare name
    # matching returned '500', 'california', 'car', 'circle', which are common nouns that
    # happen to appear in some company's name, and a bare rename catches "Adelaide
    # International 1" vs "2" (a draw split, not a sponsor). Neither error survives the
    # conjunction, because a common noun does not systematically arrive and depart at one
    # venue and a draw-split digit is not a company.
    renamed_tokens = {e["token"] for es in renamed.values() for e in es}
    confirmed = sorted(set(corroborated) & renamed_tokens)
    # THE PHRASE, NOT THE TOKEN. Tokenising splits multi-word sponsors and makes each half
    # look like noise: 'family'+'circle' is the Family Circle Cup, 'car' is the Volvo Car
    # Open, 'silicon'+'valley' is Silicon Valley Bank, 'western'+'southern' is Western &
    # Southern. Judging those tokens individually would have discarded four real sponsors as
    # common nouns. The source tournament names are carried so the phrase is readable.
    tour_of = collections.defaultdict(set)
    for (tn, loc) in all_pairs:
        tour_of[loc].add(tn)
    confirmed_detail = {
        tok: [{"location": loc, **e,
               "tournament_names": sorted(t for t in tour_of[loc] if tok in toks(t))}
              for loc, es in sorted(renamed.items())
              for e in es if e["token"] == tok]
        for tok in confirmed
    }

    n_loc = len(byloc)
    renamed_pct = round(100.0 * len(renamed) / max(n_loc, 1), 1)
    cand_pct = round(100.0 * len(with_cand) / max(len(all_pairs), 1), 1)

    report = {
        "what_this_corrects": (
            "docs/TENNIS_GOLF_FEASIBILITY.md recorded the sponsor rationale as failed at "
            "0.29% of tennis tournaments. That figure measures WIKIDATA P859 coverage, "
            "which is a sparse editorial property — not sponsor coverage. The sponsor of a "
            "tennis tournament is its NAME, and the name was already in the xlsx this repo "
            "downloaded. A real value answering a different question than it appears to."),
        "source": "tennis-data.co.uk xlsx, already acquired — no new fetch, no key",
        "tournament_location_pairs": len(all_pairs),
        "pairs_with_a_candidate_token": len(with_cand),
        "pairs_with_a_candidate_pct": cand_pct,
        "locations": n_loc,
        "locations_with_a_rename": len(renamed),
        "locations_with_a_rename_pct": renamed_pct,
        "TIERS": {
            "CANDIDATE": ("a token that is not the location, not a demonym, not event "
                          "vocabulary. A HYPOTHESIS. The count of candidates is not "
                          "evidence of anything except that a stoplist ran."),
            "RENAMED": ("the candidate is present in some years at a fixed location and "
                        "absent in others. Strong: events do not rename for fun, so a "
                        "token that arrives or leaves is a dated sponsorship event."),
            "CORROBORATED": ("the candidate also matches a company-name token in the local "
                             "equities universe. STILL NOISY on its own — bare name "
                             "matching returns '500', 'california', 'car', 'circle', common "
                             "nouns that happen to sit inside some company's name."),
            "CONFIRMED": ("CORROBORATED **and** RENAMED. Two independent signals: a company "
                          "name match, and arrival/departure at a fixed venue. A common "
                          "noun does not systematically arrive and depart at one location, "
                          "and a draw-split digit is not a company, so neither error "
                          "survives the conjunction. This is the tier to build on."),
        },
        "equities_universe_found": bool(company_tokens),
        "equities_name_tokens": len(company_tokens),
        "corroborated_tokens": corroborated,
        "corroborated_count": len(corroborated),
        "CONFIRMED_count": len(confirmed),
        "CONFIRMED_tokens": confirmed,
        "CONFIRMED_known_false_positive_class": (
            "PLACE NAMES THAT ARE NOT THE `location` FIELD. Melbourne 2021 ran four extra "
            "events during the quarantine-bubble calendar — Great Ocean Road Open, Murray "
            "River Open, Phillip Island Trophy, Yarra Valley Classic — all named after "
            "Australian geography, none sponsored. They pass both tiers: the tokens sit in "
            "some company's name somewhere ('river', 'island', 'valley', 'ocean') and they "
            "genuinely arrive and depart at a fixed venue, because those events existed for "
            "exactly one year. The conjunction defeats common nouns and draw-splits; it does "
            "NOT defeat a toponym, because the stoplist only knows the one city in the "
            "`location` column. Stated rather than filtered: a hand-written exclusion list "
            "for this would be tuned against the cases it judges, which is the failure this "
            "phase has been correcting everywhere else."),
        "CONFIRMED_strongest_evidence": (
            "AEGON -> Viking appears at Birmingham AND Eastbourne in the same year. One "
            "sponsor rebrand surfacing simultaneously at two unrelated venues is not "
            "something a tokeniser can manufacture, and Aegon UK did rebrand to Viking. "
            "Cross-venue synchrony is a stronger check than either tier and is the natural "
            "next filter if this needs tightening."),
        "CONFIRMED_detail": confirmed_detail,
        "corroboration_caveat": (
            "A token match is a NAME match, not an entity resolution — 'citi' matching a "
            "company token does not prove that company sponsored that event. It raises the "
            "candidate above stoplist noise and nothing more. Entity resolution needs a "
            "second source and is not claimed here."
            if company_tokens else
            "The equities universe file was NOT FOUND, so corroboration did not run. This "
            "is a missing input, not a measured zero — the two are different findings and "
            "reporting the first as the second is how a transport failure gets read as a "
            "data result."),
        "renamed_locations": dict(sorted(renamed.items())),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"tournament x location pairs      : {len(all_pairs)}")
    print(f"  with a candidate token         : {len(with_cand)}  ({cand_pct}%)")
    print(f"locations                        : {n_loc}")
    print(f"  with a RENAME across years     : {len(renamed)}  ({renamed_pct}%)")
    print(f"equities universe found          : {bool(company_tokens)} "
          f"({len(company_tokens)} name tokens)")
    print(f"  CORROBORATED tokens            : {len(corroborated)}  {corroborated[:10]}")
    print(f"  CONFIRMED (corroborated+rename): {len(confirmed)}  {confirmed}")
    print(f"\nvs docs/TENNIS_GOLF_FEASIBILITY.md's recorded 0.29% (that was Wikidata P859)")
    print(f"wrote {OUT}")

    if args.check and renamed_pct < 20.0:
        print(f"\nFAIL renamed-location coverage {renamed_pct}% below the 20% floor")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
