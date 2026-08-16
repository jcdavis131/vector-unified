#!/usr/bin/env python3
"""Can P31/P1056 backfill the companies P452 has no industry for?

Solo personal project, no connection to employer, built with public/free-tier only

Q8's outcome was that in US sport the useful split is WHICH company, not WHETHER, and
that the blocker is industry coverage:

    P452 industry : 103 / 196 companies (52.6%), 83 distinct values
    no industry   : 93 companies, reaching 2,745 athletes

93 uncategorised companies is not a modelling problem, it is a missing-attribute problem,
and there are two other Wikidata properties that carry the same information for different
entities. This measures whether they close the gap, before anything is built on them:

    P31   instance of          nearly universal, but often uselessly generic for a firm
                               ("public company", "business", "enterprise"). Useful only
                               when it is SPECIFIC — "airline", "bank", "insurance company".
    P1056 product or material  what the firm makes. Present for manufacturers, absent for
                               service businesses, so it should backfill a different slice
                               than P31 rather than the same one.

The measurement that matters is not "how many have P31" — almost all will. It is how many
companies WITHOUT P452 gain a SPECIFIC category, after the generic corporate-form values
are excluded. Those exclusions are declared as a constant below, before the run, so the
coverage figure cannot be improved after the fact by quietly allowing "business" to count.

    python pipeline/probe_company_sectors.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
COMPANIES = ROOT / "data" / "orgs" / "company_entities.json"
OUT = ROOT / "data" / "orgs" / "company_sector_probe.json"
ENDPOINT = "https://query.wikidata.org/sparql"
UA = "vector-unified/sector-probe (solo personal project; contact via repo)"
BATCH = 60

# DECLARED BEFORE THE RUN. P31 on a company is usually a legal/corporate FORM, which says
# nothing about what the business does. Counting these as a sector would inflate coverage
# to near 100% while adding zero information — the exact "a real number answering a
# different question" failure this estate keeps catching. If this list is ever edited,
# the coverage number moves, so it is version-controlled rather than inlined in a query.
GENERIC_P31 = {
    "business",
    "enterprise",
    "public company",
    "private company",
    "company",
    "organization",
    "corporation",
    "brand",
    "joint-stock company",
    "public limited company",
    "limited liability company",
    "conglomerate",
    "subsidiary",
    "holding company",
    "privately held company",
    "trade name",
    "state-owned enterprise",
    "societas Europaea",
    "aktiengesellschaft",
}

QUERY = """SELECT ?c ?p31Label ?p452Label ?p1056Label WHERE {{
  VALUES ?c {{ {values} }}
  OPTIONAL {{ ?c wdt:P31   ?p31. }}
  OPTIONAL {{ ?c wdt:P452  ?p452. }}
  OPTIONAL {{ ?c wdt:P1056 ?p1056. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""


def run_query(q: str, *, attempts: int = 5):
    delay = 5.0
    for attempt in range(1, attempts + 1):
        r = requests.get(
            ENDPOINT,
            params={"query": q, "format": "json"},
            headers={"User-Agent": UA, "Accept": "application/sparql-results+json"},
            timeout=240,
        )
        if r.status_code == 429 and attempt < attempts:
            wait = float(r.headers.get("Retry-After") or delay)
            print(f"  429, waiting {wait:.0f}s ({attempt}/{attempts})", file=sys.stderr)
            time.sleep(wait)
            delay *= 2
            continue
        r.raise_for_status()
        return r.json()["results"]["bindings"]
    raise RuntimeError("WDQS kept returning 429 — try again later")


def qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not COMPANIES.exists():
        print(f"missing {COMPANIES} — run build_company_entities.py first")
        return 2
    doc = json.loads(COMPANIES.read_text(encoding="utf-8"))
    comps = {c["qid"]: c for c in doc["companies"]}
    qids = sorted(comps)
    print(f"probing {len(qids)} companies for P31 / P452 / P1056...", file=sys.stderr)

    rows = []
    for i in range(0, len(qids), BATCH):
        values = " ".join(f"wd:{q}" for q in qids[i : i + BATCH])
        rows.extend(run_query(QUERY.format(values=values)))
        time.sleep(1.0)

    got: dict[str, dict[str, set]] = collections.defaultdict(lambda: {"p31": set(), "p452": set(), "p1056": set()})
    for b in rows:
        c = qid(b["c"]["value"])
        for src, key in (
            ("p31Label", "p31"),
            ("p452Label", "p452"),
            ("p1056Label", "p1056"),
        ):
            if src in b:
                got[c][key].add(b[src]["value"])

    n = len(qids)
    have_452 = {c for c in qids if got[c]["p452"]}
    missing = [c for c in qids if c not in have_452]

    def specific_p31(c):
        return {v for v in got[c]["p31"] if v.lower() not in {g.lower() for g in GENERIC_P31}}

    backfill_p31 = {c for c in missing if specific_p31(c)}
    backfill_1056 = {c for c in missing if got[c]["p1056"]}
    backfill_any = backfill_p31 | backfill_1056

    # what athletes does the backfill unlock?
    ath_of_company = {c["qid"]: c.get("athlete_count", 0) for c in doc["companies"]}
    unlocked = sum(ath_of_company.get(c, 0) for c in backfill_any)

    report = {
        "companies": n,
        "with_P452": len(have_452),
        "without_P452": len(missing),
        "backfilled_by_specific_P31": len(backfill_p31),
        "backfilled_by_P1056": len(backfill_1056),
        "backfilled_by_either": len(backfill_any),
        "coverage_before_pct": round(100.0 * len(have_452) / n, 2),
        "coverage_after_pct": round(100.0 * (len(have_452) + len(backfill_any)) / n, 2),
        "still_uncategorised": len(missing) - len(backfill_any),
        "athlete_reach_unlocked_upper_bound": unlocked,
        "generic_P31_values_excluded": sorted(GENERIC_P31),
    }

    samples = []
    for c in sorted(backfill_any, key=lambda x: -ath_of_company.get(x, 0))[:12]:
        samples.append(
            {
                "company": comps[c]["label"],
                "athletes": ath_of_company.get(c, 0),
                "specific_P31": sorted(specific_p31(c))[:3],
                "P1056": sorted(got[c]["p1056"])[:3],
            }
        )
    report["samples"] = samples

    still = [
        comps[c]["label"] for c in sorted(set(missing) - backfill_any, key=lambda x: -ath_of_company.get(x, 0))[:12]
    ]
    report["still_uncategorised_sample"] = still

    OUT.write_text(
        json.dumps(
            {
                "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": "Wikidata SPARQL (P31 / P452 / P1056)",
                "report": report,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"\ncompanies: {n}")
    print(f"  with P452 industry              {report['with_P452']:4}  ({report['coverage_before_pct']}%)")
    print(f"  without                         {report['without_P452']:4}")
    print(
        f"    backfilled by SPECIFIC P31    {report['backfilled_by_specific_P31']:4}"
        f"   (generic corporate forms excluded)"
    )
    print(f"    backfilled by P1056 product   {report['backfilled_by_P1056']:4}")
    print(f"    backfilled by either          {report['backfilled_by_either']:4}")
    print(f"  coverage after backfill         {report['coverage_after_pct']}%")
    print(f"  still uncategorised             {report['still_uncategorised']:4}")
    print(f"  athlete reach unlocked (upper bound, non-disjoint): {unlocked}")
    print("\nbiggest companies the backfill would categorise:")
    for s in samples:
        cat = s["specific_P31"] or s["P1056"]
        print(f"  {s['company'][:32]:32} {s['athletes']:>4} athletes  {', '.join(cat)[:44]}")
    if still:
        print("\nstill uncategorised, largest first:")
        print("  " + ", ".join(still[:8]))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
