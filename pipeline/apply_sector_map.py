#!/usr/bin/env python3
"""Apply data/sector_map.json to the company layer and report coverage honestly.

Solo personal project, no connection to employer, built with public/free-tier only

The map is hand-authored (see its own note, and build_company_sectors.py for the two
automatic routes that were measured and rejected first). This applies it and reports:

  - coverage AGAINST 128 business-typed companies, never against the 98 that happen to
    carry an industry value. A grouping of values cannot categorise a company with no
    value, and reporting 98/98 would be a real number answering a different question.
  - per-sector athlete reach, with the row sum, the deduplicated union, and the inflation
    factor beside it, because a company with several sectors lands in several rows.
  - the same figures EXCLUDING sports_holdings, because a club's own holding entity
    appearing as its own "company" is not sponsorship and should never sit inside a
    brand-exposure number.

    python pipeline/apply_sector_map.py
    python pipeline/apply_sector_map.py --json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAP = ROOT / "data" / "sector_map.json"
COMPANIES = ROOT / "data" / "orgs" / "company_entities.json"
ORG_ENTS = ROOT / "data" / "orgs" / "org_entities.json"
OUT = ROOT / "data" / "orgs" / "company_sectors_applied.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    smap = json.loads(MAP.read_text(encoding="utf-8"))
    doc = json.loads(COMPANIES.read_text(encoding="utf-8"))
    ents = json.loads(ORG_ENTS.read_text(encoding="utf-8"))

    i2s = smap["industry_to_sector"]
    overrides = {a["company"]: a["sectors"] for a in smap["company_overrides"]["assignments"]}
    sector_label = {s["id"]: s["label"] for s in smap["sectors"]}

    biz = [c for c in doc["companies"] if c.get("is_business")]

    # athlete reach machinery — same sport::team join key as everywhere else
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

    sector_ath: dict[str, set[str]] = collections.defaultdict(set)
    sector_cos: dict[str, list[str]] = collections.defaultdict(list)
    assigned, unassigned = [], []
    via = collections.Counter()

    for c in biz:
        secs: set[str] = set()
        for ind in c.get("industries", []):
            for s in i2s.get(ind, []):
                secs.add(s)
        if secs:
            via["industry_value"] += 1
        if c["label"] in overrides:
            secs |= set(overrides[c["label"]])
            via["manual_override"] += 1

        c_ath: set[str] = set()
        for k in c.get("orgs", ()):
            c_ath |= ath_of_key.get(k, set())

        if secs:
            assigned.append(c["label"])
            for s in secs:
                sector_cos[s].append(c["label"])
                sector_ath[s] |= c_ath
        else:
            unassigned.append((c["label"], len(c_ath), c.get("industries", [])))

    def stats(exclude: set[str]) -> dict:
        keep = {s: v for s, v in sector_ath.items() if s not in exclude}
        union: set[str] = set()
        for v in keep.values():
            union |= v
        rows = sum(len(v) for v in keep.values())
        return {
            "sectors_used": len(keep),
            "row_sum": rows,
            "distinct_athletes": len(union),
            "inflation_if_summed": round(rows / len(union), 2) if union else None,
            "pct_of_all_athletes": round(100.0 * len(union) / len(total), 2) if total else 0.0,
        }

    report = {
        "business_typed_companies": len(biz),
        "assigned_to_a_sector": len(assigned),
        "pct_assigned_of_128": round(100.0 * len(assigned) / len(biz), 2),
        "still_unassigned": len(unassigned),
        "assignment_source": dict(via),
        "sectors_declared": len(smap["sectors"]),
        "all_sectors": stats(set()),
        "excluding_sports_holdings": stats({"sports_holdings"}),
        "by_sector": {
            s: {
                "label": sector_label.get(s, s),
                "companies": len(sector_cos[s]),
                "athletes": len(sector_ath[s]),
            }
            for s in sorted(sector_ath, key=lambda x: -len(sector_ath[x]))
        },
        "unassigned_detail": sorted(unassigned, key=lambda t: -t[1]),
    }

    OUT.write_text(
        json.dumps(
            {
                "built_from": "data/sector_map.json (human-authored anchor)",
                "report": report,
                "sector_companies": {s: sorted(v) for s, v in sector_cos.items()},
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

    print(f"business-typed companies : {report['business_typed_companies']}")
    print(
        f"assigned to >=1 sector   : {report['assigned_to_a_sector']} "
        f"({report['pct_assigned_of_128']}% OF 128, not of the 98 with an industry value)"
    )
    print(f"still unassigned         : {report['still_unassigned']}")
    print(
        f"sectors declared         : {report['sectors_declared']}   " f"used: {report['all_sectors']['sectors_used']}\n"
    )
    print(f"{'sector':34} {'cos':>4} {'athletes':>9}")
    for s, v in report["by_sector"].items():
        print(f"{v['label'][:34]:34} {v['companies']:>4} {v['athletes']:>9}")
    a, x = report["all_sectors"], report["excluding_sports_holdings"]
    print("\nDO NOT SUM the athlete column:")
    print(
        f"  all sectors            rows {a['row_sum']}  distinct {a['distinct_athletes']}"
        f"  inflation {a['inflation_if_summed']}x  = {a['pct_of_all_athletes']}% of athletes"
    )
    print(
        f"  excl. sports_holdings  rows {x['row_sum']}  distinct {x['distinct_athletes']}"
        f"  inflation {x['inflation_if_summed']}x  = {x['pct_of_all_athletes']}% of athletes"
    )
    print("  the second line is the brand-exposure figure; a club owning itself is not a sponsor.")
    if report["unassigned_detail"]:
        print("\nstill unassigned:")
        for lab, n, inds in report["unassigned_detail"]:
            print(f"  {lab[:34]:34} {n:>4} athletes  {', '.join(inds)[:38]}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
