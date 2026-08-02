"""Vector Unified — ask the entity graph a question, with the confounds handled.

First query layer over athlete -> org -> attrs. Everything before this built registries;
this is the part that returns an answer.

    python pipeline/query_graph.py                 # all questions
    python pipeline/query_graph.py --q archetype   # one

Q1 archetype x market scale
    Do cross-sport archetypes concentrate in large-market organizations?

TWO METHOD DECISIONS THAT DECIDE WHETHER THE ANSWER MEANS ANYTHING.

1. CAPACITY IS RANKED WITHIN SPORT, never compared raw. NFL stadiums run ~70k and NBA
   arenas ~19k, so a raw cross-sport capacity comparison ranks every gridiron org above
   every hoops org and then reports the sport mix of each archetype as if it were a market
   finding. Percentile-within-sport removes the format offset and leaves the thing actually
   asked: is this archetype in a BIG club FOR ITS SPORT.

2. A SHUFFLE BASELINE RUNS EVERY TIME. Archetype labels are permuted and the same statistic
   recomputed, so the spread you see has something to be compared against. Without it any
   ordering looks like a finding — 12 groups over 20,719 rows will always produce a
   highest and a lowest.

The verdict is printed as a comparison against that baseline, and when the observed spread
does not clear it the output says so instead of ranking the groups anyway.

LIMITS, stated up front because they bound every number below:
  - capacity is a VENUE property standing in for market size. It is not revenue, and
    Wikidata has revenue for 1 of 180 orgs (measured), so this is the available proxy
    rather than the right one.
  - capacity is STATIC per club, so a franchise that moved or rebuilt carries one value
    across all seasons.
  - 86.1% of enriched orgs have capacity; rows without it are dropped, not zero-filled,
    and the dropped count is reported.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNIFIED = ROOT / "assets" / "unified.json"
ORGS = ROOT / "data" / "orgs" / "org_entities.json"

SHUFFLES = 200
SEED = 7


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'’-]", "", s.lower())
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s.strip())
    return re.sub(r"\s+", " ", s)


def load_joined() -> tuple[list[dict], dict]:
    """player-season rows joined to their org, with capacity as a WITHIN-SPORT percentile."""
    players = json.loads(UNIFIED.read_text(encoding="utf-8"))["players"]
    doc = json.loads(ORGS.read_text(encoding="utf-8"))
    orgs = {o["org_id"]: o for o in doc["orgs"]}

    # (norm, sport, season) -> org_id. A traded player has several; keep them all and let
    # the row count reflect that rather than silently picking one.
    edge_index: dict[tuple, list[str]] = defaultdict(list)
    for e in doc["edges"]:
        if e.get("org_id"):
            edge_index[(e["norm"], e["sport"], str(e["season"]))].append(e["org_id"])

    # Capacity percentile is computed per sport, so the format offset cannot leak in.
    caps_by_sport: dict[str, list[int]] = defaultdict(list)
    for o in orgs.values():
        c = (o.get("attrs") or {}).get("capacity")
        if c:
            caps_by_sport[o["sport"]].append(c)
    for v in caps_by_sport.values():
        v.sort()

    def pct_within_sport(sport: str, cap: int) -> float:
        v = caps_by_sport[sport]
        below = sum(1 for x in v if x < cap)
        return 100.0 * below / max(len(v) - 1, 1)

    rows, dropped = [], {"no_edge": 0, "no_capacity": 0, "no_arch": 0}
    for p in players:
        arch = p.get("cross_arch")
        if arch is None:
            dropped["no_arch"] += 1
            continue
        oids = edge_index.get((norm_name(p["name"]), p["sport"], str(p["season"])))
        if not oids:
            dropped["no_edge"] += 1
            continue
        for oid in oids:
            o = orgs.get(oid)
            cap = (o.get("attrs") or {}).get("capacity") if o else None
            if not cap:
                dropped["no_capacity"] += 1
                continue
            rows.append(
                {
                    "sport": p["sport"],
                    "arch": str(arch),
                    "capacity": cap,
                    "cap_pct": pct_within_sport(p["sport"], cap),
                    "org": o["team"],
                    "city": (o.get("attrs") or {}).get("city"),
                }
            )
    return rows, dropped


def q_archetype(rows: list[dict], dropped: dict) -> None:
    print("Q1  Do cross-sport archetypes concentrate in large-market organizations?")
    print(f"    joined rows {len(rows)}   dropped {dropped}\n")
    if len(rows) < 100:
        print("    too few joined rows to say anything. Not ranking them.")
        return

    by_arch: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_arch[r["arch"]].append(r["cap_pct"])
    means = {a: statistics.mean(v) for a, v in by_arch.items() if len(v) >= 30}
    if len(means) < 2:
        print("    fewer than 2 archetypes with n>=30. Not ranking them.")
        return
    observed_spread = max(means.values()) - min(means.values())

    # Same statistic on permuted labels. If the real spread does not clear this, the
    # ordering below is what 12 groups do by chance and must not be read as a finding.
    rng = random.Random(SEED)
    labels = [r["arch"] for r in rows]
    vals = [r["cap_pct"] for r in rows]
    null_spreads = []
    for _ in range(SHUFFLES):
        rng.shuffle(labels)
        g: dict[str, list[float]] = defaultdict(list)
        for a, v in zip(labels, vals):
            g[a].append(v)
        m = [statistics.mean(v) for v in g.values() if len(v) >= 30]
        if len(m) >= 2:
            null_spreads.append(max(m) - min(m))
    null_p95 = sorted(null_spreads)[int(0.95 * len(null_spreads))] if null_spreads else 0.0

    print(f"    {'archetype':12} {'n':>6}  mean capacity percentile within sport")
    for a, m in sorted(means.items(), key=lambda kv: -kv[1]):
        print(f"    {a:12} {len(by_arch[a]):6}  {m:5.1f}")
    print()
    print(f"    observed spread  : {observed_spread:5.1f} percentile points")
    print(f"    shuffle p95      : {null_p95:5.1f}   ({SHUFFLES} permutations, seed {SEED})")
    if observed_spread > null_p95:
        print("    VERDICT: spread exceeds the shuffled baseline — archetypes do differ")
        print("             in the market scale of the clubs they sit in.")
    else:
        print("    VERDICT: spread does NOT clear the shuffled baseline. The ordering above")
        print("             is what this many groups produce by chance. No finding.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--q", default="all", choices=["all", "archetype"])
    args = ap.parse_args()

    rows, dropped = load_joined()
    if not rows:
        print("No joined rows. Run build_org_entities.py and enrich_orgs_wikidata.py first.")
        return 1
    q_archetype(rows, dropped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
