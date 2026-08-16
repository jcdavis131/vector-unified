"""Vector Unified — the full hoops athlete -> org edge, 1996-97 to 2025-26.

build_org_entities.py (6beef47) linked 73.25% of athletes to an organization, and hoops was
the drag at 35.53%. The cause was known rather than mysterious: the only player->team source
in the estate is vector-hoops/pipeline/data/roster_context.json, which is built at an
800-minute ROTATION threshold over gamelog seasons 2015-16..2025-26. It is rotation players
in the gamelog era, not every athlete in a 30-season corpus. Raising coverage means a fuller
source, not a lower threshold.

The fuller source was already being fetched and thrown away. vector-hoops/pipeline/
build_min_gp.py calls LeagueDashPlayerStats and keeps player_id / name / season / MPG / GP —
the same response carries TEAM_ID and TEAM_ABBREVIATION on every row. This asks the same
endpoint for the columns that were dropped.

    python pipeline/acquire_hoops_rosters.py
    python pipeline/acquire_hoops_rosters.py --offline   # cache only, no network

Output: data/orgs/hoops_rosters.json { built, seasons, edges, coverage }
Cache:  data/orgs/cache/hoops_roster_<season>.json  (one file per season, re-runs are free)

TWO THINGS THIS DOES NOT DO.

It does not resolve multi-team seasons into a single org. A traded player appears once per
team with TEAM_ABBREVIATION "TOT" also present in some seasons; "TOT" is a LEAGUE TOTAL row,
not a franchise, and is dropped. A player genuinely on two teams keeps two edges, which is
the truth about that season and is left for the model to handle rather than silently
collapsed to the last team.

It does not write a model asset or train anything, same as every step before it. The
coverage number is the deliverable.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNIFIED = ROOT / "assets" / "unified.json"
CACHE = ROOT / "data" / "orgs" / "cache"
OUT = ROOT / "data" / "orgs" / "hoops_rosters.json"

# Not a franchise: the league's own combined row for a player who changed teams.
NOT_A_TEAM = {"TOT"}


def norm_name(name: str) -> str:
    """Match acquire_forbes.norm_name / hoops fetch_honors.norm_name for cross-sport joins."""
    s = unicodedata.normalize("NFD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'’-]", "", s.lower())
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s.strip())
    return re.sub(r"\s+", " ", s)


def fetch_season(season: str, offline: bool) -> list[dict] | None:
    """One season of player -> team rows, cached. Returns None when unavailable."""
    # v2: MIN and GP are kept alongside the team. They were always on the response and
    # always discarded — the same shape as the TEAM_ABBREVIATION this script exists to stop
    # discarding. Minutes are the usage proxy Q6 stratifies on, so the finding that
    # archetype predicts pay can be tested INSIDE a usage band instead of across all of it.
    # New filename rather than deleting: the v1 pull stays reproducible.
    cache_p = CACHE / f"hoops_roster_v2_{season}.json"
    if cache_p.exists():
        try:
            return json.loads(cache_p.read_text(encoding="utf-8"))
        except Exception:
            pass
    if offline:
        return None

    from nba_api.stats.endpoints import leaguedashplayerstats

    last = None
    for attempt in range(4):
        try:
            r = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Base",
                timeout=75,
            )
            df = r.get_data_frames()[0]
            rows = []
            for _, x in df.iterrows():
                team = str(x.get("TEAM_ABBREVIATION") or "").strip()
                if not team or team in NOT_A_TEAM:
                    continue

                def _num(v):
                    try:
                        f = float(v)
                    except (TypeError, ValueError):
                        return None
                    return None if f != f else f  # NaN is not a measurement

                rows.append(
                    {
                        "player_id": int(x["PLAYER_ID"]),
                        "name": str(x.get("PLAYER_NAME") or ""),
                        "season": season,
                        "team_code": team,
                        "team_id": int(x["TEAM_ID"]) if x.get("TEAM_ID") else None,
                        # per_mode_detailed="PerGame", so MIN is minutes per game.
                        "min": _num(x.get("MIN")),
                        "gp": _num(x.get("GP")),
                    }
                )
            CACHE.mkdir(parents=True, exist_ok=True)
            cache_p.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
            return rows
        except Exception as e:
            last = e
            time.sleep(2.0 * (attempt + 1))
    print(f"  {season}: FAILED after 4 attempts ({last})")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--offline", action="store_true", help="use cache only, never network")
    args = ap.parse_args()

    doc = json.loads(UNIFIED.read_text(encoding="utf-8"))
    players = doc["players"]
    seasons = sorted({p["season"] for p in players if p["sport"] == "hoops"})
    print(f"hoops seasons in corpus: {len(seasons)}  ({seasons[0]} .. {seasons[-1]})")

    edges, got_seasons, missed = [], [], []
    for s in seasons:
        rows = fetch_season(s, args.offline)
        if rows is None:
            missed.append(s)
            continue
        got_seasons.append(s)
        for r in rows:
            edges.append(
                {
                    "norm": norm_name(r["name"]),
                    "name": r["name"],
                    "sport": "hoops",
                    "season": r["season"],
                    "team_code": r["team_code"],
                    "team_id": r.get("team_id"),
                    # Carried through to the edge, not just the cache. The first attempt
                    # added min/gp to the CACHED rows and left this projection untouched, so
                    # the pull succeeded and the edges came back 0% populated — the same
                    # discarded-column bug this script exists to fix, reintroduced one layer
                    # up. Caught by asserting the field was present rather than assuming it.
                    "min": r.get("min"),
                    "gp": r.get("gp"),
                    "source": "nba_api LeagueDashPlayerStats",
                }
            )
        print(f"  {s}: {len(rows)} player-team rows")

    if not edges:
        print("\nZERO edges. Not writing — an empty file reads downstream as")
        print("'measured, found nothing' rather than 'the pull failed'.")
        return 1

    # Coverage against the corpus, measured exactly as build_org_entities.py does it.
    hoops_athletes = {norm_name(p["name"]) for p in players if p["sport"] == "hoops"}
    edge_athletes = {e["norm"] for e in edges}
    matched = hoops_athletes & edge_athletes

    coverage = {
        "seasons_requested": len(seasons),
        "seasons_retrieved": len(got_seasons),
        "seasons_missing": missed,
        "edges": len(edges),
        "corpus_hoops_athletes": len(hoops_athletes),
        "linked": len(matched),
        "pct": round(100.0 * len(matched) / max(len(hoops_athletes), 1), 2),
        "prior_pct_roster_context_only": 35.53,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "built": datetime.now(UTC).isoformat(),
                "source": "nba_api LeagueDashPlayerStats (TEAM_ID/TEAM_ABBREVIATION)",
                "seasons": got_seasons,
                "edges": edges,
                "coverage": coverage,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nseasons retrieved : {len(got_seasons)}/{len(seasons)}")
    if missed:
        print(f"seasons MISSING   : {missed}")
    print(f"edges             : {len(edges)}")
    print(f"hoops athletes    : {coverage['linked']}/{coverage['corpus_hoops_athletes']}" f"  ({coverage['pct']}%)")
    print(f"was (rotation-only source): {coverage['prior_pct_roster_context_only']}%")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
