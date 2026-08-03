#!/usr/bin/env python3
"""Persist the company layer: company -> org -> athlete, with industry.

Solo personal project, no connection to employer, built with public/free-tier only

`probe_company_edges.py` established that the edge exists (142/187 orgs, 84.3% athlete
reach, vs 0.22% for athlete-side P859). This turns that probe into a layer other things
can join against, and adds the one attribute a brand question actually needs: what the
company DOES.

ENTITY MODEL, mirroring the org layer's `org = team x season`:

    company   = a Wikidata entity typed (P31/P279*) as an organization
    edge      = (company, relation, org) where relation is one of
                sponsor | owner | venue_named_after
    reach     = athlete -> org (org_entities.json, 99.98% resolved) -> company

Deliberately NOT a `company x year` entity. The org layer is seasonal because team
performance is; naming rights and ownership DO change (Staples Center -> Crypto.com
Arena), but Wikidata's truthy values give the CURRENT holder with no interval attached.
Stamping a current holder onto a 1996-97 roster would be a real error, so every reach
figure here is "athletes who ever played for an org whose company edge is X TODAY", and
the file says so rather than implying a time series it does not have.

INDUSTRY (P452) is the point. "AT&T" and "Bank of America" are only useful to a brand
question as telecom and banking; the raw QIDs answer "which company", not "which kind of
company", and it is the second question a sponsorship product is built on.

    python pipeline/build_company_entities.py
    python pipeline/build_company_entities.py --json
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

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "data" / "orgs" / "company_edge_probe.json"
ORG_ENTS = ROOT / "data" / "orgs" / "org_entities.json"
OUT = ROOT / "data" / "orgs" / "company_entities.json"
ENDPOINT = "https://query.wikidata.org/sparql"
UA = "vector-unified/company-entities (solo personal project; contact via repo)"
BATCH = 60

# `isBusiness` is the correction that matters. The edge probe filtered targets to
# P31/P279* -> organization (Q43229), which excluded PEOPLE but not non-commercial
# organizations — so families (Maloof, Rooney), football federations (The FA, RFEF),
# geographic entities (San Francisco Bay Area) and even a club itself (Los Angeles
# Lakers) were all counted as "companies". 68 of 196, and they carried 797 athletes.
#
# Business/company typing (Q4830453 / Q783794) is therefore emitted PER COMPANY rather
# than used as a silent filter, because Wikidata's typing has false negatives too — GEHA
# is a real health insurer and types as neither. Dropping on this flag would trade one
# wrong number for another; tagging lets the consumer choose and lets the report state
# both figures side by side.
ATTR_QUERY = """SELECT ?c ?cLabel ?industryLabel ?countryLabel ?inception ?employees ?revenue
       ?isBusiness WHERE {{
  VALUES ?c {{ {values} }}
  OPTIONAL {{ ?c wdt:P452 ?industry. }}
  OPTIONAL {{ ?c wdt:P17  ?country. }}
  OPTIONAL {{ ?c wdt:P571 ?inception. }}
  OPTIONAL {{ ?c wdt:P1128 ?employees. }}
  OPTIONAL {{ ?c wdt:P2139 ?revenue. }}
  BIND(EXISTS {{ ?c wdt:P31/wdt:P279* wd:Q4830453. }}
       || EXISTS {{ ?c wdt:P31/wdt:P279* wd:Q783794. }} AS ?isBusiness)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""


def run_query(q: str, *, attempts: int = 5):
    """SPARQL with backoff on 429 — same reasoning as probe_company_edges.run_query:
    a crash halfway through leaves a PARTIAL layer on disk, which is worse than slow."""
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
            print(f"  429 from WDQS, waiting {wait:.0f}s ({attempt}/{attempts})", file=sys.stderr)
            time.sleep(wait)
            delay *= 2
            continue
        r.raise_for_status()
        return r.json()["results"]["bindings"]
    raise RuntimeError("WDQS kept returning 429 — try again later")


def qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip Wikidata; REUSES attributes already in the output file")
    args = ap.parse_args()

    if not PROBE.exists():
        print(f"missing {PROBE} — run pipeline/probe_company_edges.py first")
        return 2
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    ents = json.loads(ORG_ENTS.read_text(encoding="utf-8"))

    # company QID -> {label, relations to orgs}
    companies: dict[str, dict] = {}
    edges: list[dict] = []
    key_of_qid = probe["org_names"]
    for rel, m in probe["edges"].items():
        for org_qid, targets in m.items():
            for t in targets:
                if not t["is_org"]:
                    continue  # a person or a place is not a company edge
                c = companies.setdefault(t["qid"], {"qid": t["qid"], "label": t["label"],
                                                     "relations": set()})
                c["relations"].add(rel)
                edges.append({"company": t["qid"], "company_label": t["label"],
                              "rel": rel, "org_qid": org_qid,
                              "org_key": key_of_qid.get(org_qid)})

    # ---- attributes ---------------------------------------------------------
    # --no-fetch REUSES what is already on disk. The first version simply skipped the
    # fetch, which meant a --no-fetch run silently rewrote the artifact with every
    # industry blanked — the same silent-data-loss shape as a read-modify-write that
    # drops the field it did not load. A flag meant to avoid network cost must not
    # destroy the thing the network paid for.
    attrs: dict[str, dict] = {}
    if args.no_fetch and OUT.exists():
        prev = json.loads(OUT.read_text(encoding="utf-8"))
        for rec in prev.get("companies", []):
            attrs[rec["qid"]] = {
                "industries": set(rec.get("industries") or []),
                "country": rec.get("country"), "inception": rec.get("inception"),
                "employees": rec.get("employees"), "revenue": rec.get("revenue"),
                "is_business": bool(rec.get("is_business")),
            }
        kept = sum(1 for a in attrs.values() if a["industries"])
        print(f"--no-fetch: reused attributes for {len(attrs)} companies "
              f"({kept} with an industry)", file=sys.stderr)
    if not args.no_fetch and companies:
        qids = sorted(companies)
        print(f"fetching attributes for {len(qids)} companies...", file=sys.stderr)
        rows = []
        for i in range(0, len(qids), BATCH):
            values = " ".join(f"wd:{q}" for q in qids[i : i + BATCH])
            rows.extend(run_query(ATTR_QUERY.format(values=values)))
            time.sleep(1.0)
        for b in rows:
            c = qid(b["c"]["value"])
            a = attrs.setdefault(c, {"industries": set(), "country": None,
                                     "inception": None, "employees": None, "revenue": None,
                                     "is_business": False})
            if b.get("isBusiness", {}).get("value") == "true":
                a["is_business"] = True
            if "industryLabel" in b:
                a["industries"].add(b["industryLabel"]["value"])
            for src, dst in (("countryLabel", "country"), ("inception", "inception"),
                             ("employees", "employees"), ("revenue", "revenue")):
                if src in b and a[dst] is None:
                    a[dst] = b[src]["value"]

    for c, rec in companies.items():
        rec["relations"] = sorted(rec["relations"])
        a = attrs.get(c, {})
        rec["industries"] = sorted(a.get("industries", []))
        for k in ("country", "inception", "employees", "revenue"):
            rec[k] = a.get(k)
        rec["is_business"] = bool(a.get("is_business"))

    # ---- reach: athletes per company ----------------------------------------
    # Join on `sport::team`, the key the enriched org dict uses. A bare-name join
    # silently returns zero because gridiron orgs are keyed by CODE — see the census.
    orgkey = {o["org_id"]: f"{o.get('sport')}::{o.get('team')}" for o in ents["orgs"]}
    orgs_of_company: dict[str, set[str]] = collections.defaultdict(set)
    for e in edges:
        if e["org_key"]:
            orgs_of_company[e["company"]].add(e["org_key"])

    athletes_of_key: dict[str, set[str]] = collections.defaultdict(set)
    all_athletes, sport_of_athlete = set(), {}
    for e in ents["edges"]:
        a, oid = e.get("norm"), e.get("org_id")
        if not a:
            continue
        all_athletes.add(a)
        sport_of_athlete[a] = e.get("sport") or (oid.split("::")[0] if oid else "?")
        k = orgkey.get(oid)
        if k:
            athletes_of_key[k].add(a)

    for c, rec in companies.items():
        ath = set()
        for k in orgs_of_company.get(c, ()):
            ath |= athletes_of_key.get(k, set())
        rec["orgs"] = sorted(orgs_of_company.get(c, ()))
        rec["athlete_count"] = len(ath)

    reached = {a for c in companies for k in orgs_of_company.get(c, ())
               for a in athletes_of_key.get(k, set())}
    biz_qids = {c for c, r in companies.items() if r.get("is_business")}
    reached_biz = {a for c in biz_qids for k in orgs_of_company.get(c, ())
                   for a in athletes_of_key.get(k, set())}
    per_sport_biz: dict[str, list[int]] = {}
    for a in all_athletes:
        sp = sport_of_athlete.get(a, "?")
        row = per_sport_biz.setdefault(sp, [0, 0])
        row[1] += 1
        if a in reached_biz:
            row[0] += 1

    # ---- industry coverage, the figure that decides the brand question -------
    with_ind = [c for c, r in companies.items() if r["industries"]]
    ind_counter: collections.Counter = collections.Counter()
    ind_athletes: dict[str, set[str]] = collections.defaultdict(set)
    for c, rec in companies.items():
        for ind in rec["industries"]:
            ind_counter[ind] += 1
            for k in orgs_of_company.get(c, ()):
                ind_athletes[ind] |= athletes_of_key.get(k, set())

    # INDUSTRY ROWS DO NOT ADD UP, and the file has to say so in numbers rather than
    # in a footnote. P452 is a fragmented taxonomy — "financial services", "financial
    # sector", "economics of banking" and "insurance" are separate values, and a company
    # carrying two of them contributes its athletes to both rows. Measured here: summing
    # the finance-ish rows gives 4,518 against 2,274 distinct athletes, a 1.99x inflation.
    # So the report carries the deduplicated union next to the sum, and anyone who adds
    # the column can see immediately that they should not have.
    top_inds = sorted(ind_athletes, key=lambda x: -len(ind_athletes[x]))[:25]
    listed_union: set[str] = set()
    for i in top_inds:
        listed_union |= ind_athletes[i]
    row_sum = sum(len(ind_athletes[i]) for i in top_inds)
    disjointness = {
        "industry_rows_are_non_disjoint": True,
        "sum_of_listed_rows": row_sum,
        "distinct_athletes_across_listed": len(listed_union),
        "inflation_if_summed": round(row_sum / len(listed_union), 2) if listed_union else None,
        "companies_with_multiple_industries": sum(1 for r in companies.values()
                                                  if len(r["industries"]) > 1),
    }

    no_ind_athletes: set[str] = set()
    for c, rec in companies.items():
        if not rec["industries"]:
            for k in orgs_of_company.get(c, ()):
                no_ind_athletes |= athletes_of_key.get(k, set())

    coverage = {
        "companies": len(companies),
        "companies_with_industry": len(with_ind),
        "pct_with_industry": round(100.0 * len(with_ind) / len(companies), 2) if companies else 0.0,
        "edges": len(edges),
        "by_relation": dict(collections.Counter(e["rel"] for e in edges)),
        "athletes_total": len(all_athletes),
        # BOTH figures, because the first one shipped and was wrong. "any organization"
        # counts families, football federations and clubs; "business-typed" is the number
        # a sponsorship claim may use. The gap is stated so nobody quotes the larger one
        # by accident.
        "athletes_reaching_ANY_org": len(reached),
        "pct_athletes_ANY_org": round(100.0 * len(reached) / len(all_athletes), 2) if all_athletes else 0.0,
        "companies_business_typed": len(biz_qids),
        "athletes_reaching_a_BUSINESS": len(reached_biz),
        "pct_athletes_BUSINESS": round(100.0 * len(reached_biz) / len(all_athletes), 2) if all_athletes else 0.0,
        "overstatement_if_any_org_used": {
            "athletes": len(reached) - len(reached_biz),
            "points": round(100.0 * (len(reached) - len(reached_biz)) / len(all_athletes), 2)
            if all_athletes else 0.0,
        },
        "pct_business_by_sport": {
            sp: {"reached": r, "total": t, "pct": round(100.0 * r / t, 2) if t else 0.0}
            for sp, (r, t) in sorted(per_sport_biz.items())
        },
        "distinct_industries": len(ind_counter),
        "companies_without_industry": len(companies) - len(with_ind),
        "athletes_reached_only_via_uncategorised_companies": len(no_ind_athletes - listed_union),
        "athletes_via_companies_with_no_industry": len(no_ind_athletes),
        "industry_disjointness": disjointness,
    }

    out = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entity_model": "company = Wikidata organization; edge = (company, rel, org); NOT time-sliced",
        "time_caveat": ("Wikidata truthy values give the CURRENT holder with no interval. "
                        "Reach means 'athletes who ever played for an org whose company edge "
                        "is X today', not a season-accurate sponsorship history."),
        "sources": {"edges": str(PROBE.name), "athletes": str(ORG_ENTS.name),
                    "attributes": "Wikidata SPARQL (P452/P17/P571/P1128/P2139)"},
        "coverage": coverage,
        "top_industries_by_athlete_reach": [
            {"industry": i, "companies": ind_counter[i], "athletes": len(ind_athletes[i])}
            for i in sorted(ind_athletes, key=lambda x: -len(ind_athletes[x]))[:25]
        ],
        "companies": sorted(companies.values(), key=lambda r: -r["athlete_count"]),
        "edges": edges,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(coverage, indent=2))
        return 0

    print(f"\ncompanies: {coverage['companies']}   edges: {coverage['edges']}   "
          f"{coverage['by_relation']}")
    print(f"with industry (P452): {coverage['companies_with_industry']} "
          f"({coverage['pct_with_industry']}%)   distinct industries: {coverage['distinct_industries']}")
    print(f"athletes reaching ANY org   : {coverage['athletes_reaching_ANY_org']} / "
          f"{coverage['athletes_total']} = {coverage['pct_athletes_ANY_org']}%"
          "   <- includes families, federations, clubs. NOT a sponsorship figure.")
    print(f"athletes reaching a BUSINESS: {coverage['athletes_reaching_a_BUSINESS']} / "
          f"{coverage['athletes_total']} = {coverage['pct_athletes_BUSINESS']}%"
          f"   ({coverage['companies_business_typed']} of {coverage['companies']} companies)")
    ov = coverage["overstatement_if_any_org_used"]
    print(f"  quoting the first line overstates by {ov['athletes']} athletes "
          f"({ov['points']} points)")
    for sp, v in coverage["pct_business_by_sport"].items():
        print(f"    {sp:10} {v['reached']:5} / {v['total']:5} = {v['pct']:5.1f}%")
    print()
    print(f"{'industry':38} {'cos':>5} {'athletes':>9}")
    for r in out["top_industries_by_athlete_reach"][:15]:
        print(f"{r['industry'][:38]:38} {r['companies']:>5} {r['athletes']:>9}")
    dj = coverage["industry_disjointness"]
    print("\nDO NOT SUM THAT COLUMN. P452 is fragmented and companies carry several values:")
    print(f"  sum of the listed rows      {dj['sum_of_listed_rows']}")
    print(f"  distinct athletes across    {dj['distinct_athletes_across_listed']}")
    print(f"  inflation if summed         {dj['inflation_if_summed']}x")
    print(f"  companies with >1 industry  {dj['companies_with_multiple_industries']}")
    print(f"  companies with NO industry  {coverage['companies_without_industry']}"
          f" (reaching {coverage['athletes_via_companies_with_no_industry']} athletes)")
    print("\ntop companies by athlete reach:")
    for r in out["companies"][:10]:
        print(f"  {r['label'][:34]:34} {r['athlete_count']:>5} athletes  "
              f"{','.join(r['relations'])}  {', '.join(r['industries'][:2])}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
