"""Vector Unified — organizations as a SECOND entity type, alongside athletes.

Operator direction (2026-08-02): "teams are organizations like the equities companies, and
should be treated similarly but with links to individual athletes as distinct entities."

So this mirrors the vector-equities shape. There a company is a row per TICKER x FISCAL
YEAR carrying feature towers. Here an organization is a row per TEAM x SEASON carrying the
same kind of operating features, and athletes stay their own entity type joined by a typed
edge rather than folded into the org row.

    equities   ticker  x fiscal_year -> ~122 features in 17 towers, sector label
    unified    team    x season      -> operating features,          sport label

WHY THIS AND NOT SPONSORS. The athlete->brand edge was the first attempt and it is not
obtainable from public data at usable coverage — measured in acquire_sponsors.py (2dbb4f0):
Wikidata P859 reaches 13 of 5,821 athletes (0.22%), gridiron literally zero, and of 429
Wikipedia bios exactly 2 mention any brand with 0 endorsement cues. A team, by contrast, IS
a business with revenue and a market, every athlete has one, and the edge already exists in
data this estate owns.

SOURCES, all public and already local:
  hoops     vector-hoops/pipeline/data/team_season_<season>.json  (31 seasons x 30 teams:
            PACE, OFF_RATING, DEF_RATING, NET_RATING, W, L, WIN_PCT)
            vector-hoops/pipeline/data/roster_context.json        (3,463 player->team rows)
  gridiron  team label on each player-season in assets/unified.json
  pitch     same

    python pipeline/build_org_entities.py

Output: data/orgs/org_entities.json
  { built, orgs: [{org_id, sport, team, season, name, features{}}],
    edges: [{norm, sport, season, org_id, source}], coverage: {...} }

Writes NO model asset and trains nothing. Same rule as acquire_sponsors.py: produce the
registry and the coverage number first, so the decision to embed a second entity type is
made against measured reach rather than an assumption about it.

KNOWN ASYMMETRY, recorded rather than smoothed over: only hoops has org FEATURES here.
gridiron and pitch orgs are currently identity-only (they exist, they have athletes, they
carry no operating stats yet), because the corpus stores a team code and nothing about the
team. Any model trained on this must not read a missing feature vector as a zeroed one.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOOPS = ROOT.parent / "vector-hoops" / "pipeline" / "data"
UNIFIED = ROOT / "assets" / "unified.json"
OUT = ROOT / "data" / "orgs" / "org_entities.json"

HOOPS_FEATURES = ["PACE", "OFF_RATING", "DEF_RATING", "NET_RATING", "W", "L", "WIN_PCT"]


def norm_name(name: str) -> str:
    """Match acquire_forbes.norm_name / hoops fetch_honors.norm_name for cross-sport joins."""
    s = unicodedata.normalize("NFD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'’-]", "", s.lower())
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s.strip())
    return re.sub(r"\s+", " ", s)


def org_id(sport: str, team: str, season: str) -> str:
    return f"{sport}::{team}::{season}"


def load_hoops_orgs() -> tuple[list[dict], list[str]]:
    """One org row per NBA team-season, with the operating features the league publishes."""
    orgs, seasons = [], []
    for p in sorted(HOOPS.glob("team_season_*.json")):
        season = p.stem.replace("team_season_", "")
        if season == "manifest":
            continue
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        seasons.append(season)
        for r in rows:
            name = r.get("TEAM_NAME")
            if not name:
                continue
            orgs.append(
                {
                    "org_id": org_id("hoops", name, season),
                    "sport": "hoops",
                    "team": name,
                    "team_id": r.get("TEAM_ID"),
                    "season": season,
                    "name": name,
                    "features": {k: r.get(k) for k in HOOPS_FEATURES},
                }
            )
    return orgs, seasons


def load_hoops_edges() -> list[dict]:
    """Athlete -> org for hoops. Prefers the full roster pull, falls back to rotation-only.

    acquire_hoops_rosters.py asks LeagueDashPlayerStats for the TEAM_ABBREVIATION that
    vector-hoops/build_min_gp.py was already fetching and discarding. It covers all 30
    corpus seasons and reaches 99.96% of hoops athletes.

    roster_context.json remains the fallback, and it is worth naming why it was never
    enough: it is built at an 800-minute ROTATION threshold over gamelog seasons
    2015-16..2025-26, so it is rotation players in the gamelog era — 35.53%. That is a
    property of how it is DEFINED, not a gap to be tuned out of it.
    """
    full = ROOT / "data" / "orgs" / "hoops_rosters.json"
    if full.exists():
        doc = json.loads(full.read_text(encoding="utf-8"))
        edges = doc.get("edges", [])
        if edges:
            return edges

    p = HOOPS / "roster_context.json"
    if not p.exists():
        return []
    print("  (falling back to roster_context.json — run acquire_hoops_rosters.py for full)")
    doc = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for e in doc.get("entries", []):
        name, team, season = e.get("name"), e.get("team"), e.get("season")
        if not (name and team and season):
            continue
        out.append(
            {
                "norm": norm_name(name),
                "name": name,
                "sport": "hoops",
                "season": season,
                "team_code": team,
                "team_id": e.get("teamId"),
                "source": "vector-hoops/roster_context.json",
            }
        )
    return out


def load_corpus_orgs_and_edges() -> tuple[list[dict], list[dict], dict]:
    """gridiron + pitch: the team label already carried on each player-season."""
    doc = json.loads(UNIFIED.read_text(encoding="utf-8"))
    players = doc["players"]
    seen: set[tuple] = set()
    orgs, edges = [], []
    missing = Counter()

    for p in players:
        sport, team, season = p["sport"], p.get("team"), p.get("season")
        if sport == "hoops":
            continue  # covered by the richer hoops source above
        if not team:
            missing[sport] += 1
            continue
        key = (sport, team, season)
        if key not in seen:
            seen.add(key)
            orgs.append(
                {
                    "org_id": org_id(sport, team, str(season)),
                    "sport": sport,
                    "team": team,
                    "team_id": None,
                    "season": str(season),
                    "name": team,
                    # Identity-only: the corpus stores a code and nothing about the club.
                    # Explicitly empty so a consumer cannot mistake absent for zero.
                    "features": {},
                }
            )
        edges.append(
            {
                "norm": norm_name(p["name"]),
                "name": p["name"],
                "sport": sport,
                "season": str(season),
                "team_code": team,
                "team_id": None,
                "source": "assets/unified.json",
            }
        )
    return orgs, edges, dict(missing)


def main() -> int:
    hoops_orgs, hoops_seasons = load_hoops_orgs()
    hoops_edges = load_hoops_edges()
    corpus_orgs, corpus_edges, missing_team = load_corpus_orgs_and_edges()

    orgs = hoops_orgs + corpus_orgs
    edges = hoops_edges + corpus_edges
    if not orgs or not edges:
        print("Refusing to write: orgs or edges came back empty, which would read")
        print("downstream as 'measured, found nothing' rather than 'a source moved'.")
        return 1

    doc = json.loads(UNIFIED.read_text(encoding="utf-8"))
    players = doc["players"]
    corpus_athletes = {(norm_name(p["name"]), p["sport"]) for p in players}
    edge_athletes = {(e["norm"], e["sport"]) for e in edges}
    matched = corpus_athletes & edge_athletes

    per_sport = {}
    for sport in sorted({s for _, s in corpus_athletes}):
        pool = {n for n, s in corpus_athletes if s == sport}
        hit = {n for n in pool if (n, sport) in edge_athletes}
        n_orgs = sum(1 for o in orgs if o["sport"] == sport)
        with_feats = sum(1 for o in orgs if o["sport"] == sport and o["features"])
        per_sport[sport] = {
            "unique_athletes": len(pool),
            "linked_to_an_org": len(hit),
            "pct": round(100.0 * len(hit) / max(len(pool), 1), 2),
            "orgs": n_orgs,
            "orgs_with_features": with_feats,
        }

    coverage = {
        "corpus_player_seasons": len(players),
        "corpus_unique_athletes": len(corpus_athletes),
        "orgs": len(orgs),
        "edges": len(edges),
        "matched_unique_athletes": len(matched),
        "matched_pct": round(100.0 * len(matched) / max(len(corpus_athletes), 1), 2),
        "player_seasons_missing_team": missing_team,
        "per_sport": per_sport,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "built": datetime.now(UTC).isoformat(),
                "entity_model": "org = team x season, mirroring equities ticker x fiscal_year",
                "sources": {
                    "hoops": "vector-hoops team_season_*.json + roster_context.json",
                    "gridiron": "assets/unified.json team label",
                    "pitch": "assets/unified.json team label",
                },
                "orgs": orgs,
                "edges": edges,
                "coverage": coverage,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"orgs  : {len(orgs)}   ({len(hoops_orgs)} hoops with features, "
          f"{len(corpus_orgs)} identity-only)")
    print(f"edges : {len(edges)}")
    print(f"\ncorpus: {coverage['corpus_player_seasons']} player-seasons, "
          f"{coverage['corpus_unique_athletes']} unique athletes")
    print(f"linked: {coverage['matched_unique_athletes']} "
          f"({coverage['matched_pct']}% of unique athletes)\n")
    print(f"{'sport':10} {'linked':>16}  {'pct':>6}  {'orgs':>6}  {'w/feats':>8}")
    for sport, s in sorted(per_sport.items()):
        print(f"{sport:10} {s['linked_to_an_org']:6}/{s['unique_athletes']:<9} "
              f"{s['pct']:6}  {s['orgs']:6}  {s['orgs_with_features']:8}")
    if missing_team:
        print(f"\nplayer-seasons with no team label: {missing_team}")

    by_sport = defaultdict(int)
    for o in orgs:
        by_sport[o["sport"]] += 1
    print(f"\nwrote {OUT}")
    print("\nNo model asset written and nothing trained. The number above decides whether")
    print("an org entity is worth embedding, the same way acquire_sponsors.py decided")
    print("against a brand entity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
