"""Vector Unified — acquire the athlete <-> company (sponsor) edge from Wikidata.

WHY THIS IS THE FIRST STEP, and why it is only acquisition.

The unified model today embeds ONE entity type: 20,719 player-seasons in a 64-d space.
"Companies and sponsors in the same model" is a second entity TYPE, and the only thing that
can put a company near an athlete in one geometry is a real edge between them. Forbes
(`acquire_forbes.py`) already gives endorsement DOLLARS per athlete, but never says WHICH
brand -- so the edge does not exist anywhere in this repo yet.

Coverage is make-or-break and it is cheap to measure. If Wikidata's sponsor edge covers ~1%
of our corpus there is nothing to train on and the direction needs a different source; that
is worth learning in one script rather than after building an encoder. So this script
ACQUIRES AND MEASURES. It deliberately trains nothing and ships no asset into the model.

    python pipeline/acquire_sponsors.py            # query + report
    python pipeline/acquire_sponsors.py --offline  # re-report from the cached pull

Output: data/market_cultural/sponsors.json
  { built, source, properties, companies: {qid: {...}}, edges: [...],
    coverage: {...} }

WIKIDATA PROPERTIES USED
  P859  sponsor          the edge we want. On athletes it carries kit/endorsement deals.
  P452  industry         company feature (what sector the brand is in)
  P17   country          company feature
  P571  inception        company feature (age)
  P2139 total revenue    company feature, sparse but the strongest size signal

P859 is also used team->sponsor, so rows are filtered to items that are humans (P31 Q5)
carrying our sport (P641). A stadium naming-rights deal is not an athlete endorsement.

HONEST LIMIT, stated before any number is read: Wikidata's sponsor coverage of athletes is
known to be thin and biased toward the famous. Whatever the coverage number turns out to
be, it is a FLOOR on what a paid source would give, and the bias direction (superstars
over-represented) matters for any business question asked of the result. The report prints
the number rather than a verdict.
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
OUT = ROOT / "data" / "market_cultural" / "sponsors.json"
UNIFIED = ROOT / "assets" / "unified.json"

ENDPOINT = "https://query.wikidata.org/sparql"
UA = "vector-unified/0.1 (personal research; contact via github)"

# Same three sports the corpus covers (pull_honors_wikidata.py:38).
# Q41323 = American football. This carried Q9398 — GRUGLIASCO, AN ITALIAN COMUNE — same as
# pull_honors_wikidata.py did, because the wrong value was copied between the two files.
# The consequence is not cosmetic: the 0.22% athlete-side sponsor census that has been
# quoted all through Phase 7 as "athlete-level sponsorship does not exist in Wikidata"
# reported gridiron `with_sponsor: 0`, and that zero was the QID, not a measurement.
SPORT_Q = {"hoops": "Q5372", "gridiron": "Q41323", "pitch": "Q2736"}


def norm_name(name: str) -> str:
    """Match acquire_forbes.norm_name / hoops fetch_honors.norm_name for cross-sport joins."""
    s = unicodedata.normalize("NFD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'’-]", "", s.lower())
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s.strip())
    return re.sub(r"\s+", " ", s)


def run_query(q: str):
    r = requests.get(
        ENDPOINT,
        params={"query": q, "format": "json"},
        headers={"User-Agent": UA, "Accept": "application/sparql-results+json"},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["results"]["bindings"]


def sponsors_query(sport_qid: str) -> str:
    """Athletes of one sport and the companies that sponsor them, with company features.

    `?item wdt:P31 wd:Q5` keeps this to humans: P859 is also team->sponsor and
    venue->sponsor, and a stadium naming-rights deal is not an athlete endorsement.
    """
    return f"""
SELECT ?item ?itemLabel ?sponsor ?sponsorLabel ?industryLabel ?countryLabel
       ?inception ?revenue WHERE {{
  ?item wdt:P31 wd:Q5.
  ?item wdt:P641 wd:{sport_qid}.
  ?item wdt:P859 ?sponsor.
  OPTIONAL {{ ?sponsor wdt:P452 ?industry. }}
  OPTIONAL {{ ?sponsor wdt:P17  ?country. }}
  OPTIONAL {{ ?sponsor wdt:P571 ?inception. }}
  OPTIONAL {{ ?sponsor wdt:P2139 ?revenue. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""


def val(b: dict, k: str):
    v = b.get(k)
    return v.get("value") if v else None


def qid_of(uri: str | None) -> str | None:
    return uri.rsplit("/", 1)[-1] if uri else None


def acquire() -> dict:
    companies: dict[str, dict] = {}
    edges: list[dict] = []

    for sport, qid in SPORT_Q.items():
        print(f"[{sport}] sponsor query ({qid})...")
        try:
            rows = run_query(sponsors_query(qid))
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        print(f"  {len(rows)} raw rows")
        for b in rows:
            cq = qid_of(val(b, "sponsor"))
            if not cq:
                continue
            name = val(b, "itemLabel") or ""
            cname = val(b, "sponsorLabel") or cq
            if cname == cq or not name:
                continue  # unlabelled: nothing to join or display on
            rec = companies.setdefault(
                cq,
                {
                    "qid": cq,
                    "name": cname,
                    "industry": val(b, "industryLabel"),
                    "country": val(b, "countryLabel"),
                    "inception": (val(b, "inception") or "")[:4] or None,
                    "revenue": val(b, "revenue"),
                    "n_athletes": 0,
                },
            )
            rec["n_athletes"] += 1
            edges.append(
                {
                    "athlete_qid": qid_of(val(b, "item")),
                    "athlete": name,
                    "norm": norm_name(name),
                    "sport": sport,
                    "company_qid": cq,
                    "company": cname,
                }
            )
        time.sleep(1.0)  # be a good citizen on a free public endpoint

    # A (athlete, company) pair can repeat once per optional company attribute.
    seen, deduped = set(), []
    for e in edges:
        k = (e["norm"], e["sport"], e["company_qid"])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(e)
    for c in companies.values():
        c["n_athletes"] = sum(
            1 for e in deduped if e["company_qid"] == c["qid"]
        )
    return {"companies": companies, "edges": deduped}


def measure_coverage(edges: list[dict]) -> dict:
    """How much of the ACTUAL corpus this edge set reaches. The number that decides."""
    doc = json.loads(UNIFIED.read_text(encoding="utf-8"))
    players = doc["players"]

    corpus_keys = {(norm_name(p["name"]), p["sport"]) for p in players}
    corpus_by_sport: dict[str, set] = {}
    for n, s in corpus_keys:
        corpus_by_sport.setdefault(s, set()).add(n)

    edge_keys = {(e["norm"], e["sport"]) for e in edges}
    matched = corpus_keys & edge_keys

    per_sport = {}
    for sport, names in corpus_by_sport.items():
        hit = {n for n in names if (n, sport) in edge_keys}
        per_sport[sport] = {
            "unique_athletes": len(names),
            "with_sponsor": len(hit),
            "pct": round(100.0 * len(hit) / max(len(names), 1), 2),
        }

    return {
        "corpus_player_seasons": len(players),
        "corpus_unique_athletes": len(corpus_keys),
        "edge_unique_athletes": len(edge_keys),
        "matched_unique_athletes": len(matched),
        "matched_pct": round(100.0 * len(matched) / max(len(corpus_keys), 1), 2),
        "per_sport": per_sport,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--offline", action="store_true",
                    help="re-report from the cached pull instead of querying")
    args = ap.parse_args()

    if args.offline:
        if not OUT.exists():
            print(f"no cached pull at {OUT} — run without --offline first")
            return 2
        doc = json.loads(OUT.read_text(encoding="utf-8"))
        companies, edges = doc["companies"], doc["edges"]
    else:
        got = acquire()
        companies, edges = got["companies"], got["edges"]

    if not edges:
        print("\nZERO edges acquired. Not writing an empty asset — an empty file here")
        print("would read downstream as 'measured, found nothing' rather than 'the pull")
        print("failed'. Check the endpoint and rerun.")
        return 1

    coverage = measure_coverage(edges)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "built": datetime.now(UTC).isoformat(),
                "source": "Wikidata SPARQL (P859 sponsor)",
                "properties": ["P859", "P452", "P17", "P571", "P2139"],
                "companies": companies,
                "edges": edges,
                "coverage": coverage,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\ncompanies : {len(companies)}")
    print(f"edges     : {len(edges)}")
    print(f"\ncorpus    : {coverage['corpus_player_seasons']} player-seasons, "
          f"{coverage['corpus_unique_athletes']} unique athletes")
    print(f"matched   : {coverage['matched_unique_athletes']} "
          f"({coverage['matched_pct']}% of unique athletes)")
    for sport, s in sorted(coverage["per_sport"].items()):
        print(f"  {sport:9} {s['with_sponsor']:5}/{s['unique_athletes']:5}  {s['pct']:5}%")

    top = sorted(companies.values(), key=lambda c: -c["n_athletes"])[:10]
    print("\ntop companies by athletes:")
    for c in top:
        print(f"  {c['n_athletes']:4}  {c['name']}  ({c.get('industry') or 'industry?'})")

    print(f"\nwrote {OUT}")
    print("\nNo model asset written and nothing trained — this step exists to produce the")
    print("coverage number above, which decides whether a company entity is trainable at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
