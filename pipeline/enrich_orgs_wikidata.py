"""Vector Unified — give organizations real features, and pull LOCATIONS out as entities.

build_org_entities.py established the org entity (team x season, 73.25% of athletes
linked) but only hoops orgs carried features; gridiron and pitch were identity-only,
because the corpus stores a team code and nothing about the club. This fills that in from
Wikidata, and the same pass produces the location entity the unified model needs:

    org      -> country (P17), headquarters city (P159), founded (P571), venue (P115),
                league (P118), owner (P127)
    location -> the distinct cities/countries those orgs resolve to

    python pipeline/enrich_orgs_wikidata.py
    python pipeline/enrich_orgs_wikidata.py --offline   # re-report from cache

Output: data/orgs/org_wikidata.json  { built, orgs: {...}, locations: {...}, coverage }

TWO QUERY TRAPS, both hit while building this and both worth keeping written down.

1. `?club wdt:P118 ?league` alone returns PEOPLE. P118 (league) is stated on players as
   well as teams, so the first probe came back with 77,483 "clubs" — Kyrie Irving, Walt
   Frazier, Adam Morrison. Constrained with `P31/P279* Q12973014` (sports team) plus an
   explicit `FILTER NOT EXISTS ?club wdt:P31 wd:Q5`. This is the same shape as the sponsor
   query, where P859 needed the OPPOSITE filter to keep stadium naming-rights deals out.

2. gridiron teams are 3-letter codes (ARI, ATL) in the corpus and full names in Wikidata,
   so they cannot join without a map. The map is written out explicitly below rather than
   fuzzy-matched: 32 rows that change roughly never, and a wrong fuzzy match here silently
   attaches the wrong city to a franchise.

pitch labels include NATIONAL TEAMS ("Albania", "Argentina") alongside clubs. They are
organizations too, but a country is not a franchise — they are tagged `org_kind` so a
downstream model can separate them instead of averaging a nation with a club.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
ORGS_IN = ROOT / "data" / "orgs" / "org_entities.json"
OUT = ROOT / "data" / "orgs" / "org_wikidata.json"

ENDPOINT = "https://query.wikidata.org/sparql"
UA = "vector-unified/0.1 (personal research; contact via github)"

LEAGUE_Q = {"hoops": "Q155223", "gridiron": "Q1215884"}  # NBA, NFL

# NFL code -> Wikidata label. Explicit because a fuzzy match that puts the wrong city on a
# franchise is worse than no city at all, and these 32 rows are stable.
NFL_CODES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


# Club-type affixes. Football clubs carry them on one side of the join and not the other:
# the corpus says "AC Milan" / "Arsenal", Wikidata says "A.C. Milan" / "Arsenal F.C.".
# Stripping punctuation alone left pitch at 15.13% (23/152) with 34,484 orgs pulled — the
# data was there and the KEY was wrong. Not a coverage problem, a join problem.
#
# Only generic club-type tokens are removed. "United", "City", "Rovers", "Athletic" stay:
# they distinguish real clubs (Manchester United vs Manchester City) and stripping them
# would merge distinct organisations, which is worse than failing to match.
CLUB_AFFIX = re.compile(
    r"^(fc|afc|ac|as|sc|ssc|cf|cd|rc|us|ss|sv|bsc|vfl|vfb|fk|nk|ck|1)\s+|"
    r"\s+(fc|afc|ac|as|sc|ssc|cf|cd|rc|us|ss|sv|bsc|sco|fk|nk|ck)$",
    re.I,
)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'’-]", "", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    # National teams: the corpus says "Argentina", Wikidata says "Argentina national
    # football team". Stripped so the two sides meet; org_kind still records what it is.
    s = re.sub(r"\s+(men'?s\s+)?national\s+(association\s+)?(football|basketball)\s+team$",
               "", s)
    # Applied twice: "1. FC Koln" carries two leading tokens.
    for _ in range(2):
        s = CLUB_AFFIX.sub(" ", s).strip()
    return re.sub(r"\s+", " ", s).strip()


def run_query(q: str):
    r = requests.get(
        ENDPOINT,
        params={"query": q, "format": "json"},
        headers={"User-Agent": UA, "Accept": "application/sparql-results+json"},
        timeout=240,
    )
    r.raise_for_status()
    return r.json()["results"]["bindings"]


# P1083 (maximum capacity) is reached THROUGH the venue, not stated on the club. Probed
# over 180 enriched orgs before adding it, because the obvious business fields are not
# there and it would have been easy to assume otherwise:
#
#     capacity  171/180   (95%)
#     members    20
#     employees   2
#     revenue     1
#     assets      1
#
# So Wikidata gives SCALE but not money. Capacity is the closest public proxy for market
# size, and it is what takes gridiron/pitch orgs off identity-only.
#
# CROSS-SPORT CAVEAT, load-bearing for anyone modelling on this: capacity is confounded by
# sport. NFL stadiums run ~70k, NBA arenas ~19k, so a raw comparison ranks every gridiron
# org above every hoops org on "scale" for reasons of physical format, not business size.
# Within-sport it is meaningful; across sports it needs the sport offset the joint model
# already carries.
ORG_FIELDS = """
  OPTIONAL {{ ?club wdt:P17  ?country. }}
  OPTIONAL {{ ?club wdt:P159 ?hq. }}
  OPTIONAL {{ ?club wdt:P571 ?inception. }}
  OPTIONAL {{ ?club wdt:P115 ?venue. }}
  OPTIONAL {{ ?club wdt:P127 ?owner. }}
  OPTIONAL {{ ?club wdt:P115/wdt:P1083 ?capacity. }}
  OPTIONAL {{ ?club wdt:P2124 ?members. }}
"""

SELECT = """SELECT ?club ?clubLabel ?leagueLabel ?countryLabel ?hqLabel ?inception
       ?venueLabel ?ownerLabel ?capacity ?members WHERE {{"""
TAIL = """  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""


def league_query(league_qid: str) -> str:
    return f"""{SELECT}
  ?club wdt:P118 wd:{league_qid}.
  ?club wdt:P31/wdt:P279* wd:Q12973014.
  FILTER NOT EXISTS {{ ?club wdt:P31 wd:Q5. }}
{ORG_FIELDS}
{TAIL}""".replace("{{", "{").replace("}}", "}")


def football_query() -> str:
    """Association football clubs AND national teams, both are orgs in our pitch corpus."""
    return f"""{SELECT}
  {{ ?club wdt:P31/wdt:P279* wd:Q476028. }}
  UNION
  {{ ?club wdt:P31 wd:Q135408445. }}   # men's national association football team
  FILTER NOT EXISTS {{ ?club wdt:P31 wd:Q5. }}
{ORG_FIELDS}
{TAIL}""".replace("{{", "{").replace("}}", "}")


def val(b, k):
    v = b.get(k)
    return v.get("value") if v else None


def _as_int(v):
    """Wikidata numerics arrive as strings, sometimes '19068.0' or '+19068'."""
    if v is None:
        return None
    try:
        return int(float(str(v).lstrip("+")))
    except (TypeError, ValueError):
        return None


def collect(rows, sport: str) -> dict:
    out: dict[str, dict] = {}
    for b in rows:
        label = val(b, "clubLabel")
        if not label or re.fullmatch(r"Q\d+", label):
            continue  # unlabelled entity: nothing to join on
        key = norm(label)
        rec = out.setdefault(
            key,
            {
                "wikidata": (val(b, "club") or "").rsplit("/", 1)[-1],
                "label": label,
                "sport": sport,
                "league": val(b, "leagueLabel"),
                "country": val(b, "countryLabel"),
                "city": val(b, "hqLabel"),
                "founded": (val(b, "inception") or "")[:4] or None,
                "venue": val(b, "venueLabel"),
                "owner": val(b, "ownerLabel"),
                "capacity": _as_int(val(b, "capacity")),
                "members": _as_int(val(b, "members")),
            },
        )
        for k, src in (("country", "countryLabel"), ("city", "hqLabel"),
                       ("venue", "venueLabel"), ("owner", "ownerLabel")):
            if not rec[k]:
                v = val(b, src)
                if v and not re.fullmatch(r"Q\d+", v):
                    rec[k] = v
        # A club with several venues yields several rows; keep the LARGEST capacity rather
        # than whichever row arrived first, so the value is a stable property of the org.
        for k in ("capacity", "members"):
            v = _as_int(val(b, k))
            if v is not None and (rec[k] is None or v > rec[k]):
                rec[k] = v
    return out


def acquire() -> dict:
    wd: dict[str, dict] = {}
    for sport, qid in LEAGUE_Q.items():
        print(f"[{sport}] league query ({qid})...")
        try:
            rows = run_query(league_query(qid))
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        got = collect(rows, sport)
        print(f"  {len(rows)} rows -> {len(got)} orgs")
        wd.update(got)
        time.sleep(1.0)

    print("[pitch] football clubs + national teams...")
    try:
        rows = run_query(football_query())
        got = collect(rows, "pitch")
        print(f"  {len(rows)} rows -> {len(got)} orgs")
        wd.update(got)
    except Exception as e:
        print(f"  FAILED: {e}")
    return wd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    if args.offline:
        if not OUT.exists():
            print(f"no cache at {OUT} — run without --offline first")
            return 2
        wd = json.loads(OUT.read_text(encoding="utf-8"))["orgs"]
    else:
        wd = acquire()

    if not wd:
        print("\nZERO orgs acquired — not writing. An empty file here reads downstream as")
        print("'measured, found nothing' rather than 'the query failed'.")
        return 1

    doc = json.loads(ORGS_IN.read_text(encoding="utf-8"))
    ours = {(o["sport"], o["team"]) for o in doc["orgs"]}

    enriched, misses = {}, []
    for sport, team in sorted(ours):
        lookup = NFL_CODES.get(team, team) if sport == "gridiron" else team
        hit = wd.get(norm(lookup))
        if hit:
            enriched[f"{sport}::{team}"] = {
                **hit,
                "team": team,
                "matched_on": lookup,
                # A nation is not a franchise. Tagged so a model can separate them rather
                # than averaging a country with a club.
                "org_kind": "national_team" if hit.get("league") is None
                and sport == "pitch" and hit.get("city") is None else "club",
            }
        else:
            misses.append(f"{sport}::{team}")

    locations: dict[str, dict] = {}
    for rec in enriched.values():
        city, country = rec.get("city"), rec.get("country")
        if not (city or country):
            continue
        key = norm(f"{city or ''}|{country or ''}")
        loc = locations.setdefault(
            key, {"city": city, "country": country, "n_orgs": 0, "sports": []}
        )
        loc["n_orgs"] += 1
        if rec["sport"] not in loc["sports"]:
            loc["sports"].append(rec["sport"])

    per_sport = {}
    for sport in sorted({s for s, _ in ours}):
        pool = [t for s, t in ours if s == sport]
        hit = [t for t in pool if f"{sport}::{t}" in enriched]
        per_sport[sport] = {
            "orgs": len(pool),
            "enriched": len(hit),
            "pct": round(100.0 * len(hit) / max(len(pool), 1), 2),
        }

    coverage = {
        "distinct_orgs_in_corpus": len(ours),
        "enriched": len(enriched),
        "pct": round(100.0 * len(enriched) / max(len(ours), 1), 2),
        "locations": len(locations),
        "per_sport": per_sport,
        "unmatched_sample": misses[:20],
        "unmatched_total": len(misses),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "built": datetime.now(UTC).isoformat(),
                "source": "Wikidata SPARQL (P118/P17/P159/P571/P115/P127)",
                "orgs": wd,
                "enriched": enriched,
                "locations": locations,
                "coverage": coverage,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nwikidata orgs pulled : {len(wd)}")
    print(f"corpus orgs enriched : {coverage['enriched']}/{coverage['distinct_orgs_in_corpus']}"
          f"  ({coverage['pct']}%)")
    print(f"distinct locations   : {coverage['locations']}")
    print(f"\n{'sport':10} {'enriched':>14} {'pct':>7}")
    for sport, s in sorted(per_sport.items()):
        print(f"{sport:10} {s['enriched']:6}/{s['orgs']:<7} {s['pct']:6}%")
    if misses:
        print(f"\nunmatched ({len(misses)}): {misses[:8]}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
