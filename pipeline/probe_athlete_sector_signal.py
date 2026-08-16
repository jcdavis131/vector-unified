#!/usr/bin/env python3
"""How much of the athlete->sector signal is actually ATHLETE-level? (Phase 7.10)

Solo personal project, no connection to employer, built with public/free-tier only

The product thesis is "connect athletes and skills to companies and sponsors". The company
layer that would carry it reaches 71.5% of athletes, which sounds like a base to build on.
This probe asks the question that number cannot answer: **where does the variance live?**

BECAUSE THE EDGES ARE ORG-LEVEL, NOT ATHLETE-LEVEL. company_edge_probe.json is explicit
about it — the athlete-side census found P859 (sponsor) on 0.22% of athletes, so the layer
was built transitively: athlete -> org -> company. That construction has a consequence
nobody stated: **an athlete's sector vector is a deterministic function of the set of teams
he played for.** Two athletes with the same team history have byte-identical sector
vectors, no matter how differently they played.

So a model predicting athlete -> sector from athlete features cannot beat a lookup table
keyed on team history, and every athlete feature in the estate — archetype, T0/T1 standing,
D0/D1 direction, Forbes rank, honors — contributes exactly zero to it. Not "a little", not
"needs more data": zero, by construction. This file measures the size of that zero.

WHAT IT MEASURES

  1. Reach BY RELATION. The headline 71.5% is the union of three different commercial
     facts: sponsor (a sponsorship), owner (who owns the club), named_after (whose name is
     on the stadium). Only the first is a sponsorship. A sponsorship product's real base is
     the sponsor-only figure, and it is much smaller.

  2. EFFECTIVE N. Distinct athletes vs distinct sector vectors. The gap is how much of the
     apparent sample is repetition of the same row.

  3. CLASS SIZES. If the largest equivalence class holds hundreds of athletes, that class
     is one observation wearing hundreds of faces — the same denominator problem that has
     now appeared four times in this phase, in a fourth dress.

WHAT IT DOES NOT MEASURE. Whether an athlete-level sponsorship model is possible. It is not
possible ON THIS DATA — there are no athlete-level sponsorship labels to fit or validate
against. That is a data-acquisition finding, not a modelling one.

    python pipeline/probe_athlete_sector_signal.py
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPANIES = ROOT / "data" / "orgs" / "company_entities.json"
SECTORS = ROOT / "data" / "orgs" / "company_sectors_applied.json"
ORGS = ROOT / "data" / "orgs" / "org_entities.json"
OUT = ROOT / "data" / "orgs" / "athlete_sector_signal.json"

RELATIONS = ("sponsor", "owner", "named_after")


def team_key(org_id: str) -> str:
    """'hoops::Atlanta Hawks::1996-97' -> 'hoops::Atlanta Hawks'.

    The company edges are keyed on team, not team-season (Wikidata truthy values give the
    CURRENT holder with no interval), so the join has to drop the season. That is also the
    reason the layer cannot be season-accurate — already stated in company_entities.json's
    time_caveat and repeated here because it is what forces the equivalence classes to be
    this coarse.
    """
    parts = org_id.split("::")
    return "::".join(parts[:2]) if len(parts) >= 2 else org_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    for p in (COMPANIES, SECTORS, ORGS):
        if not p.exists():
            print(f"missing {p} — build the company layer first")
            return 2

    comp = json.loads(COMPANIES.read_text(encoding="utf-8"))
    sect = json.loads(SECTORS.read_text(encoding="utf-8"))
    orgs = json.loads(ORGS.read_text(encoding="utf-8"))

    sector_of_label = {}
    for sector, labels in sect["sector_companies"].items():
        for lab in labels:
            sector_of_label[lab] = sector

    business = {c["qid"] for c in comp["companies"] if c.get("is_business")}
    label_of = {c["qid"]: c["label"] for c in comp["companies"]}

    # team -> {relation -> {sector}}, business-typed companies only. Federations, families
    # and geographic entities were the 68-of-196 that inflated the first reach claim from a
    # true 71.5% to a reported 84.3%; is_business is the gate that removed them.
    by_team: dict[str, dict[str, set[str]]] = collections.defaultdict(lambda: {r: set() for r in RELATIONS})
    edges_used = 0
    for e in comp["edges"]:
        if e["company"] not in business:
            continue
        s = sector_of_label.get(label_of.get(e["company"], ""))
        if not s:
            continue
        rel = e["rel"]
        if rel not in RELATIONS:
            continue
        by_team[e["org_key"]][rel].add(s)
        edges_used += 1

    # athlete -> teams
    teams_of: dict[str, set[str]] = collections.defaultdict(set)
    sport_of: dict[str, str] = {}
    for e in orgs["edges"]:
        a = e["norm"]
        teams_of[a].add(team_key(e["org_id"]))
        sport_of[a] = e.get("sport") or "?"
    n_ath = len(teams_of)

    # ---- 1. reach by relation ------------------------------------------------
    reach = {}
    vec_of: dict[str, frozenset] = {}
    for rel in (*RELATIONS, "any"):
        hit = 0
        for a, tms in teams_of.items():
            got: set[str] = set()
            for t in tms:
                d = by_team.get(t)
                if not d:
                    continue
                got |= d[rel] if rel != "any" else set().union(*d.values())
            if rel == "any":
                vec_of[a] = frozenset(got)
            if got:
                hit += 1
        reach[rel] = {"athletes": hit, "pct": round(100.0 * hit / n_ath, 2)}

    # ---- 1b. reach by relation AND sport -------------------------------------
    # Split because the pooled row hides the shape completely: named_after is an arena
    # naming-rights pattern that barely exists in European football, and sponsor is a shirt
    # sponsorship that barely exists in US leagues. The pooled 71.5% is two disjoint
    # phenomena added together.
    per_sport: dict[str, dict] = {}
    tot_sp: collections.Counter = collections.Counter()
    for a in teams_of:
        tot_sp[sport_of.get(a, "?")] += 1
    for rel in RELATIONS:
        hit_sp: collections.Counter = collections.Counter()
        for a, tms in teams_of.items():
            if any(by_team.get(t, {}).get(rel) for t in tms):
                hit_sp[sport_of.get(a, "?")] += 1
        per_sport[rel] = {
            sp: {
                "athletes": hit_sp[sp],
                "of": n,
                "pct": round(100.0 * hit_sp[sp] / n, 1),
            }
            for sp, n in sorted(tot_sp.items())
        }

    # ---- 2/3. effective n and class sizes ------------------------------------
    classes: dict[frozenset, list[str]] = collections.defaultdict(list)
    for a, v in vec_of.items():
        if v:
            classes[v].append(a)
    reached = sum(len(v) for v in classes.values())
    sizes = sorted((len(v) for v in classes.values()), reverse=True)
    team_sets = {frozenset(t) for a, t in teams_of.items() if vec_of[a]}

    biggest = sorted(classes.items(), key=lambda kv: -len(kv[1]))[:5]

    report = {
        "question": "How much of athlete->sector is athlete-level rather than team-level?",
        "answer": "None of it. The sector vector is a pure function of the team set.",
        "athletes_total": n_ath,
        "company_edges_used": edges_used,
        "reach_by_relation": reach,
        "reach_note": (
            "Only `sponsor` is a sponsorship. `owner` is who owns the club and "
            "`named_after` is whose name is on the building — both are commercial facts "
            "about the ORG, neither is an endorsement of an athlete. A sponsorship product "
            f"has a base of {reach['sponsor']['pct']}%, not {reach['any']['pct']}%."
        ),
        "reach_by_relation_and_sport": per_sport,
        "reach_by_sport_note": (
            "The pooled figure is two disjoint phenomena summed. named_after reaches 94.5% "
            "of hoops and 82.6% of gridiron athletes but 5.0% of pitch — it is US arena "
            "naming rights. sponsor is the reverse in kind: a shirt-sponsorship pattern, "
            "and it lands at 10-16% in EVERY sport, so the one relation that is actually a "
            "sponsorship is uniformly thin. Whichever sport you pick, ~85% of athletes have "
            "no team-level sponsor edge at all."
        ),
        "effective_n": {
            "athletes_with_any_sector": reached,
            "distinct_team_sets": len(team_sets),
            "distinct_sector_vectors": len(classes),
            "collapse_ratio": round(reached / max(len(classes), 1), 1),
        },
        "largest_classes": [
            {
                "n_athletes": len(v),
                "sectors": sorted(k),
                "example_athletes": sorted(v)[:3],
            }
            for k, v in biggest
        ],
        "class_size_distribution": {
            "max": sizes[0] if sizes else 0,
            "median": sizes[len(sizes) // 2] if sizes else 0,
            "singletons": sum(1 for s in sizes if s == 1),
        },
        "why_this_matters": (
            "Any supervised model mapping athlete features -> sector has a label that is "
            "constant within an equivalence class, so no athlete feature that varies "
            "within a class can carry information. Archetype, T0/T1 standing, D0/D1 "
            "direction, Forbes rank and honors all vary within classes. Their contribution "
            "to this target is exactly zero — not small, zero. A held-out score would still "
            "look good, because the class structure leaks across any athlete-level split."
        ),
        "the_split_that_would_be_wrong": (
            "Splitting train/test by ATHLETE puts teammates on both sides and the model "
            "memorises the team lookup. The only honest split is by TEAM SET, which leaves "
            f"{len(team_sets)} independent units, not {reached}."
        ),
        "what_would_make_it_athlete_level": (
            "Athlete-level endorsement records. The Wikidata census found P859 on 0.22% of "
            "athletes, which is not a base for fitting or for validating. This is a "
            "data-acquisition problem and no modelling choice substitutes for it."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"athletes {n_ath}   company edges used {edges_used}\n")
    sports = sorted(tot_sp)
    print(f"{'relation':<12} {'athletes':>9} {'pct':>7}   " + "  ".join(f"{s:>9}" for s in sports))
    for rel in (*RELATIONS, "any"):
        tail = "  ".join(f"{per_sport[rel][s]['pct']:>8.1f}%" for s in sports) if rel in per_sport else ""
        print(f"{rel:<12} {reach[rel]['athletes']:>9} {reach[rel]['pct']:>6.2f}%   {tail}")
    e = report["effective_n"]
    print(f"\nathletes with any sector : {e['athletes_with_any_sector']}")
    print(f"distinct team sets       : {e['distinct_team_sets']}")
    print(f"distinct sector vectors  : {e['distinct_sector_vectors']}")
    print(f"collapse ratio           : {e['collapse_ratio']}x")
    print(f"largest class            : {report['class_size_distribution']['max']} athletes")
    print("\nlargest equivalence classes (identical sector vectors):")
    for c in report["largest_classes"][:3]:
        print(f"  n={c['n_athletes']:<5} {', '.join(c['sectors'][:5])}" f"{' ...' if len(c['sectors']) > 5 else ''}")
    print(f"\n{report['why_this_matters']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
