#!/usr/bin/env python3
"""Is there a COMPANY edge on the org side? Probe before building anything.

Solo personal project, no connection to employer, built with public/free-tier only

WHY. docs/VALUE_SIGNAL_CENSUS.md measured Wikidata P859 (sponsor) on ATHLETES and found
13 / 5,821 (0.22%), gridiron literally 0, and concluded that business questions needing a
value signal are blocked. That conclusion is right about the athlete side and was never
tested on the ORG side, which is a different question with a different answer:

    athlete -> org      99.98%   (5,820 / 5,821, org_entities.json, 0 unresolved)
    org -> venue        86.6%    (enriched orgs, org_wikidata.json)
    org -> owner        73.3%

A 2-hop athlete -> org -> company path can be well covered even when the 1-hop
athlete -> company path is empty. Venues already on disk are visibly corporate --
Crypto.com Arena, Scotiabank Arena, Chase Center, Little Caesars Arena -- so the naming
company is reachable through the venue rather than through the person.

This probes THREE candidate edges and reports coverage. It builds nothing and writes no
entity layer, because the census's whole lesson is that a source gets measured before it
gets modelled:

    P859 on the CLUB          club/shirt sponsor. The census only ever measured this on
                              athletes; pitch shirt sponsors in particular are well
                              documented in a way personal endorsements are not.
    P127 owner + its P31      owner is already 73.3% covered but MIXES people and
                              companies ("Jeanie Buss" next to "Kroenke Sports &
                              Entertainment"). Only the company half is a company edge,
                              so the type is the measurement, not the presence.
    P115/P138 named after     the venue's naming-rights entity. "Little Caesars Arena"
                              -> Little Caesars. Resolving through P138 rather than by
                              string-matching the label means "Madison Square Garden"
                              does not become a company by accident.

Every edge is also counted AFTER filtering to entities that are actually organisations,
because a naming-rights target can be a person or a place ("Wells Fargo Center" vs
"Oracle Park" vs a venue named after a city).

    python pipeline/probe_company_edges.py
    python pipeline/probe_company_edges.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
ORG_WD = ROOT / "data" / "orgs" / "org_wikidata.json"
OUT = ROOT / "data" / "orgs" / "company_edge_probe.json"
ENDPOINT = "https://query.wikidata.org/sparql"
UA = "vector-unified/company-probe (solo personal project; contact via repo)"
BATCH = 60  # VALUES blocks stay small enough that one slow club cannot time out the rest

# Q43229 organization, Q4830453 business, Q5 human, Q783794 company
QUERY = """SELECT ?club ?sponsor ?sponsorLabel ?sponsorIsOrg
       ?owner ?ownerLabel ?ownerIsOrg ?namedAfter ?namedAfterLabel ?namedAfterIsOrg WHERE {{
  VALUES ?club {{ {values} }}
  OPTIONAL {{ ?club wdt:P859 ?sponsor.
              BIND(EXISTS {{ ?sponsor wdt:P31/wdt:P279* wd:Q43229. }} AS ?sponsorIsOrg) }}
  OPTIONAL {{ ?club wdt:P127 ?owner.
              BIND(EXISTS {{ ?owner wdt:P31/wdt:P279* wd:Q43229. }} AS ?ownerIsOrg) }}
  OPTIONAL {{ ?club wdt:P115/wdt:P138 ?namedAfter.
              BIND(EXISTS {{ ?namedAfter wdt:P31/wdt:P279* wd:Q43229. }} AS ?namedAfterIsOrg) }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""


def run_query(q: str, *, attempts: int = 5):
    """SPARQL with backoff on 429. Re-running this probe a few times in a session is
    normal (it is a probe), and WDQS throttles for it — a bare raise_for_status turns
    that into a crash halfway through and leaves a PARTIAL report on disk, which is
    worse than slow. Honours Retry-After when the server sends one."""
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
            print(f"  429 from WDQS, waiting {wait:.0f}s (attempt {attempt}/{attempts})",
                  file=sys.stderr)
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
    ap.add_argument("--limit", type=int, default=0, help="probe only N orgs (smoke)")
    args = ap.parse_args()

    doc = json.loads(ORG_WD.read_text(encoding="utf-8"))
    enriched = doc["enriched"]
    # enriched values carry the same schema as orgs; the QID is what we query on
    pairs = [(k, v.get("wikidata")) for k, v in enriched.items() if v.get("wikidata")]
    pairs = [(k, q) for k, q in pairs if q and q.startswith("Q")]
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print("no enriched orgs carry a wikidata QID — nothing to probe")
        return 2

    by_qid = {q: k for k, q in pairs}
    print(f"probing {len(pairs)} enriched orgs for company edges...", file=sys.stderr)

    rows = []
    qids = [q for _, q in pairs]
    for i in range(0, len(qids), BATCH):
        chunk = qids[i : i + BATCH]
        values = " ".join(f"wd:{q}" for q in chunk)
        rows.extend(run_query(QUERY.format(values=values)))
        print(f"  batch {i // BATCH + 1}: {len(rows)} rows so far", file=sys.stderr)
        time.sleep(1.0)

    def truthy(b, key):
        v = b.get(key, {}).get("value")
        return v == "true"

    edges = {"sponsor": {}, "owner": {}, "named_after": {}}
    for b in rows:
        club = qid(b["club"]["value"])
        for rel, node, lab, isorg in (
            ("sponsor", "sponsor", "sponsorLabel", "sponsorIsOrg"),
            ("owner", "owner", "ownerLabel", "ownerIsOrg"),
            ("named_after", "namedAfter", "namedAfterLabel", "namedAfterIsOrg"),
        ):
            if node in b:
                edges[rel].setdefault(club, []).append({
                    "qid": qid(b[node]["value"]),
                    "label": b.get(lab, {}).get("value"),
                    "is_org": truthy(b, isorg),
                })

    n = len(pairs)
    report = {"orgs_probed": n, "relations": {}}
    org_edges = {}  # org QID -> organization-typed company targets
    for rel, m in edges.items():
        for club, vs in m.items():
            for v in vs:
                if v["is_org"]:
                    org_edges.setdefault(club, set()).add((v["qid"], v["label"], rel))
    for rel, m in edges.items():
        any_org = {c for c, vs in m.items() if any(v["is_org"] for v in vs)}
        report["relations"][rel] = {
            "orgs_with_edge": len(m),
            "pct_any": round(100.0 * len(m) / n, 2),
            "orgs_with_ORGANIZATION_edge": len(any_org),
            "pct_org": round(100.0 * len(any_org) / n, 2),
            "distinct_targets": len({v["qid"] for vs in m.values() for v in vs}),
            "sample": sorted({v["label"] for vs in m.values() for v in vs if v["is_org"] and v["label"]})[:10],
        }

    # ---- the number that decides whether any of this is usable -----------------
    # Coverage on ORGS is not the business figure. What matters is how many ATHLETES
    # reach a company through their club, so the org coverage is multiplied through
    # org_entities.json's athlete->org edges (99.98% resolved) rather than assumed.
    #
    # The join key is `sport::team`, which is the key the enriched dict uses. Matching
    # on the bare team name silently produced 0/6226 on the first attempt: gridiron orgs
    # are keyed by CODE ("gridiron::ARI") while hoops/pitch use full names, so a bare-name
    # join misses every row and looks exactly like a real negative.
    reach = None
    ents_path = ROOT / "data" / "orgs" / "org_entities.json"
    if ents_path.exists():
        ents = json.loads(ents_path.read_text(encoding="utf-8"))
        keys_with = {by_qid[q] for q in org_edges if q in by_qid}
        probed_keys = set(by_qid.values())
        orgkey = {o["org_id"]: f"{o.get('sport')}::{o.get('team')}" for o in ents["orgs"]}
        matched_keys = {k for k in orgkey.values() if k in probed_keys}
        total, reached = set(), set()
        per = {}
        for e in ents["edges"]:
            a, oid = e.get("norm"), e.get("org_id")
            if not a:
                continue
            sp = e.get("sport") or (oid.split("::")[0] if oid else "?")
            per.setdefault(sp, [set(), set()])
            total.add(a)
            per[sp][1].add(a)
            if orgkey.get(oid) in keys_with:
                reached.add(a)
                per[sp][0].add(a)
        reach = {
            "org_keys_matched": len(matched_keys),
            "athletes_total": len(total),
            "athletes_reaching_a_company": len(reached),
            "pct": round(100.0 * len(reached) / len(total), 2) if total else 0.0,
            "census_athlete_sponsor_pct": 0.22,
            "per_sport": {
                sp: {"reached": len(r), "total": len(t),
                     "pct": round(100.0 * len(r) / len(t), 2) if t else 0.0}
                for sp, (r, t) in sorted(per.items())
            },
        }
        report["athlete_reach"] = reach

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "Wikidata SPARQL (P859 sponsor, P127 owner, P115/P138 venue named-after)",
        "note": ("Coverage probe only. The census measured P859 on ATHLETES (0.22%); this "
                 "measures the ORG side, which is a different question."),
        "report": report,
        "edges": edges,
        "org_names": by_qid,
    }, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"\nenriched orgs probed: {n}\n")
    print(f"{'relation':14} {'any':>8} {'any %':>7} {'is-org':>8} {'org %':>7} {'targets':>8}")
    for rel, r in report["relations"].items():
        print(f"{rel:14} {r['orgs_with_edge']:>8} {r['pct_any']:>7} "
              f"{r['orgs_with_ORGANIZATION_edge']:>8} {r['pct_org']:>7} {r['distinct_targets']:>8}")
    print("\nCompare: athlete -> sponsor (P859 on the athlete) = 0.22% (13/5,821).")
    r = report.get("athlete_reach")
    if r:
        print(f"\norgs with >=1 ORGANIZATION-typed company edge: "
              f"{len(org_edges)} / {n} = {100.0 * len(org_edges) / n:.1f}%")
        print(f"ATHLETES reaching a company in 2 hops: "
              f"{r['athletes_reaching_a_company']} / {r['athletes_total']} = {r['pct']}%")
        for sp, v in r["per_sport"].items():
            print(f"    {sp:10} {v['reached']:5} / {v['total']:5}  {v['pct']:5.1f}%")
    for rel, r in report["relations"].items():
        if r["sample"]:
            print(f"\n{rel} sample: {r['sample']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
