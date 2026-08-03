#!/usr/bin/env python3
"""Roll 83 fragmented industry values into a small sector set, using Wikidata's own P279*.

Solo personal project, no connection to employer, built with public/free-tier only

Q8 could not test WHICH company because industry is unusable as-is:

    P452 coverage      103 / 196 companies (52.6%), 83 distinct values
    and non-disjoint   "financial services", "financial sector", "economics of banking"
                       and "insurance" are four separate values for one sector; summing
                       the top-25 rows inflates athlete counts 2.95x

Two things have to happen for "which company" to become a testable split: the values must
collapse to a small set, and the set must be DISJOINT enough that a company lands in one
place.

THE ROLLUP IS NOT HAND-AUTHORED. A hand map from 83 labels to N sectors is a map I would
be writing after having seen which groupings make the answer come out. Instead: a small
list of sector ANCHORS is declared below by NAME, each anchor is resolved to a QID by
lookup (and the resolution is printed, so a wrong anchor is visible rather than assumed),
and each observed industry is assigned by asking Wikidata whether it is a subclass* of
that anchor. The hierarchy does the grouping.

WHAT IS STILL A JUDGEMENT: the anchor list itself. It is declared here, in version
control, BEFORE the assignment runs, and it is the standard top-level sector split any
sponsorship deck uses. Changing it changes the numbers, so it must be changed in a commit
rather than in a notebook.

MULTI-SECTOR COMPANIES ARE KEPT, NOT FORCED. AT&T is telecom AND media; forcing a primary
would be inventing precision. Companies with several sectors are reported as such and the
per-sector athlete counts stay explicitly non-disjoint, with the deduplicated union
printed beside them — the same guard build_company_entities.py already carries.

RESULT — THIS METHOD DOES NOT WORK, and the file is kept so it is not retried blind.

    assigned to >=1 sector : 51 / 128 business-typed companies = 39.8%

The failures are not edge cases, they are the largest companies in the graph:

    American Airlines  257 athletes  "airline"          not P279* under "transport"
    United Airlines    254            "air transport"    "
    AT&T               283            "postal and telecommunications services"
                                      not under "telecommunications"
    Crypto.com         419            "cryptocurrency exchange"
                                      not under "financial services"
    Kaseya             234            "software company" not under "information technology"

Wikidata's subclass graph is simply not connected for industry terms. An airline is not
modelled as a kind of transport, a software company is not a kind of information
technology. The rollup is not mis-specified — the edges are absent. "real estate" did not
even resolve to a QID by exact English label.

THE STANDARD-CODE ROUTE IS ALSO CLOSED, measured rather than assumed:

    P3224 NAICS  0 / 128      P1796 ISIC  0 / 128
    P5285 SIC    0 / 128      P7364 GICS  0 / 128      ANY code: 0.0%

So there is no automatic source for sector on these 128 companies. The remaining option is
a HAND-AUTHORED map, which this file deliberately avoided — and which is now the honest
choice rather than the lazy one, because two automatic routes were measured first and both
failed. There is precedent in this estate: data/archetype_map.json is a declared
"Human-authored anchor", version-controlled, changed only in a commit. A sector map should
be built the same way, and its coverage reported against the same 128 companies so it can
be compared with the 39.8% above.

    python pipeline/build_company_sectors.py
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
ORG_ENTS = ROOT / "data" / "orgs" / "org_entities.json"
OUT = ROOT / "data" / "orgs" / "company_sectors.json"
ENDPOINT = "https://query.wikidata.org/sparql"
UA = "vector-unified/company-sectors (solo personal project; contact via repo)"
BATCH = 40

# DECLARED BEFORE THE RUN. Standard top-level commercial sectors. Resolved to QIDs by
# lookup at runtime and printed, so a mis-resolved anchor shows up instead of silently
# assigning every company to the wrong bucket.
SECTOR_ANCHORS = [
    "financial services", "insurance", "telecommunications", "transport",
    "automotive industry", "energy industry", "retail", "information technology",
    "health care", "food industry", "real estate", "mass media",
    "construction", "manufacturing", "hospitality industry",
]

# Same exclusions as probe_company_sectors.py: a corporate FORM is not a sector.
GENERIC = {
    "business", "enterprise", "public company", "private company", "company",
    "organization", "corporation", "brand", "joint-stock company",
    "public limited company", "limited liability company", "conglomerate",
    "subsidiary", "holding company", "privately held company", "trade name",
    "state-owned enterprise", "societas europaea", "aktiengesellschaft",
}


def run_query(q: str, *, attempts: int = 5):
    delay = 5.0
    for attempt in range(1, attempts + 1):
        r = requests.get(
            ENDPOINT, params={"query": q, "format": "json"},
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


def resolve_anchors() -> dict[str, str]:
    """anchor label -> QID, by exact English label. Printed so it can be checked."""
    out: dict[str, str] = {}
    for name in SECTOR_ANCHORS:
        esc = name.replace('"', '\\"')
        rows = run_query(
            f'SELECT ?s WHERE {{ ?s rdfs:label "{esc}"@en. '
            f'FILTER(STRSTARTS(STR(?s), "http://www.wikidata.org/entity/Q")) }} LIMIT 1'
        )
        if rows:
            out[name] = qid(rows[0]["s"]["value"])
        time.sleep(0.4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    doc = json.loads(COMPANIES.read_text(encoding="utf-8"))
    ents = json.loads(ORG_ENTS.read_text(encoding="utf-8"))
    biz = [c for c in doc["companies"] if c.get("is_business")]
    if not biz:
        print("no business-typed companies — re-run build_company_entities.py")
        return 2
    print(f"business-typed companies: {len(biz)}", file=sys.stderr)

    print("resolving sector anchors...", file=sys.stderr)
    anchors = resolve_anchors()
    print("\nsector anchors (verify these before trusting anything below):")
    for name in SECTOR_ANCHORS:
        print(f"  {name:24} {anchors.get(name, 'UNRESOLVED')}")
    missing = [n for n in SECTOR_ANCHORS if n not in anchors]
    if missing:
        print(f"\n  UNRESOLVED ANCHORS: {missing} — those sectors cannot be assigned.")

    # ---- each company's category QIDs (P452, else specific P31) -------------
    qids = [c["qid"] for c in biz]
    cat_of: dict[str, set[str]] = collections.defaultdict(set)
    catlabel: dict[str, str] = {}
    CQ = """SELECT ?c ?cat ?catLabel ?src WHERE {{
      VALUES ?c {{ {values} }}
      {{ ?c wdt:P452 ?cat. BIND("P452" AS ?src) }}
      UNION
      {{ ?c wdt:P31 ?cat. BIND("P31" AS ?src) }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}"""
    rows = []
    for i in range(0, len(qids), BATCH):
        values = " ".join(f"wd:{q}" for q in qids[i : i + BATCH])
        rows.extend(run_query(CQ.format(values=values)))
        time.sleep(1.0)
    for b in rows:
        lab = b.get("catLabel", {}).get("value", "")
        if lab.lower() in GENERIC:
            continue
        c, cat = qid(b["c"]["value"]), qid(b["cat"]["value"])
        cat_of[c].add(cat)
        catlabel[cat] = lab

    all_cats = sorted({x for s in cat_of.values() for x in s})
    print(f"\ndistinct category values across business-typed companies: {len(all_cats)}",
          file=sys.stderr)

    # ---- which anchors is each category a subclass* of? ---------------------
    anchor_qids = {v: k for k, v in anchors.items()}
    sector_of_cat: dict[str, set[str]] = collections.defaultdict(set)
    AQ = """SELECT ?cat ?anchor WHERE {{
      VALUES ?cat {{ {cats} }}
      VALUES ?anchor {{ {anchors} }}
      ?cat wdt:P279* ?anchor.
    }}"""
    avals = " ".join(f"wd:{q}" for q in anchors.values())
    for i in range(0, len(all_cats), BATCH):
        cvals = " ".join(f"wd:{q}" for q in all_cats[i : i + BATCH])
        for b in run_query(AQ.format(cats=cvals, anchors=avals)):
            sector_of_cat[qid(b["cat"]["value"])].add(anchor_qids[qid(b["anchor"]["value"])])
        time.sleep(1.0)

    # ---- assign, and reach --------------------------------------------------
    orgkey = {o["org_id"]: f"{o.get('sport')}::{o.get('team')}" for o in ents["orgs"]}
    ath_of_key: dict[str, set[str]] = collections.defaultdict(set)
    total: set[str] = set()
    for e in ents["edges"]:
        a, oid = e.get("norm"), e.get("org_id")
        if not a:
            continue
        total.add(a)
        if orgkey.get(oid):
            ath_of_key[orgkey[oid]].add(a)

    sector_companies: dict[str, list[str]] = collections.defaultdict(list)
    sector_ath: dict[str, set[str]] = collections.defaultdict(set)
    assigned, unassigned = [], []
    for c in biz:
        secs: set[str] = set()
        for cat in cat_of.get(c["qid"], ()):
            secs |= sector_of_cat.get(cat, set())
        c_ath: set[str] = set()
        for k in c.get("orgs", ()):
            c_ath |= ath_of_key.get(k, set())
        if secs:
            assigned.append(c["qid"])
            for s in secs:
                sector_companies[s].append(c["label"])
                sector_ath[s] |= c_ath
        else:
            unassigned.append((c["label"], len(c_ath),
                               sorted(catlabel.get(x, x) for x in cat_of.get(c["qid"], ()))[:3]))

    union_all = set()
    for s in sector_ath.values():
        union_all |= s
    row_sum = sum(len(v) for v in sector_ath.values())

    report = {
        "business_typed_companies": len(biz),
        "assigned_to_a_sector": len(assigned),
        "pct_assigned": round(100.0 * len(assigned) / len(biz), 2),
        "sectors_used": len(sector_ath),
        "anchors_declared": SECTOR_ANCHORS,
        "anchors_resolved": anchors,
        "multi_sector_companies": sum(
            1 for c in biz
            if len({s for cat in cat_of.get(c["qid"], ()) for s in sector_of_cat.get(cat, set())}) > 1
        ),
        "sector_rows_are_non_disjoint": True,
        "sum_of_sector_rows": row_sum,
        "distinct_athletes_across_sectors": len(union_all),
        "inflation_if_summed": round(row_sum / len(union_all), 2) if union_all else None,
        "athletes_total": len(total),
        "by_sector": {
            s: {"companies": len(sector_companies[s]), "athletes": len(sector_ath[s])}
            for s in sorted(sector_ath, key=lambda x: -len(sector_ath[x]))
        },
    }
    OUT.write_text(json.dumps({
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "Wikidata P279* subclass rollup onto declared anchors; no hand-authored map",
        "report": report,
        "sector_companies": {s: sorted(v) for s, v in sector_companies.items()},
        "unassigned": sorted(unassigned, key=lambda t: -t[1])[:40],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"\nassigned to >=1 sector: {len(assigned)} / {len(biz)} "
          f"({report['pct_assigned']}%)   sectors used: {report['sectors_used']}"
          f"   multi-sector: {report['multi_sector_companies']}\n")
    print(f"{'sector':26} {'cos':>5} {'athletes':>9}")
    for s, v in report["by_sector"].items():
        print(f"{s:26} {v['companies']:>5} {v['athletes']:>9}")
    print(f"\nDO NOT SUM: rows {row_sum}, distinct {len(union_all)}, "
          f"inflation {report['inflation_if_summed']}x")
    if unassigned:
        print("\nunassigned, largest first (what the anchors miss):")
        for lab, n, cats in sorted(unassigned, key=lambda t: -t[1])[:10]:
            print(f"  {lab[:30]:30} {n:>4} athletes  {', '.join(cats)[:40]}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
