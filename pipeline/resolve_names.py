"""Vector Unified — market/cultural name-resolution index.

Builds data/market/name_index.json: the canonical join key (name_norm + nationality +
birthyear) for every player in the unified corpus, deduped per (sport, native_player_id).
This is the foundation that lets external market/cultural pulls (Forbes, Spotrac,
Transfermarkt, Wikidata) be matched back to native IDs.

Inputs (read-only): load_encoders.load_all() -> per-sport records
  {sport, player_id, name, season, pos, team, native_cluster}
  - pitch: team == nationality (e.g. "Belgium")
  - hoops: team == "" (empty — no nationality in meta)
  - gridiron: team == team abbr (e.g. "KC" — NOT nationality)

Outputs:
  data/market/name_index.json
    players:   [ {sport, player_id, name, name_norm, nationality, birthyear,
                   pos, team, seasons:[...], season_years:[...], bio_augment_needed:bool} ]
    by_canonical: { "name_norm|nationality|birthyear": [player_idx, ...] }  (reverse match)
    coverage:  { sport: {n_players, has_nationality, has_birthyear, bio_augment_needed,
                         ambiguous_name_only} }

Honesty: nationality/birthyear are None where unknown (hoops+gridiron). Name-only matches
are flagged ambiguous and left masked by downstream joins (schema §4/§8).
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from load_encoders import load_all, ROOT

DATA = ROOT / "data"
MARKET = DATA / "market"

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "jr.", "sr."}


def strip_diacritics(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def name_norm(name: str) -> str:
    """Canonical name key: lowercased, diacritics stripped, suffixes removed, whitespace
    collapsed. Keeps the full name (first+last) — downstream matching can also tokenize."""
    s = strip_diacritics(name).lower().strip()
    # drop suffix tokens
    toks = [t for t in re.split(r"[\s\-.]+", s) if t and t not in SUFFIXES]
    return " ".join(toks)


def season_year(season, sport: str):
    """Normalize a sport-native season label to an int year (for season-varying joins)."""
    s = str(season).strip()
    if sport == "gridiron":
        try:
            return int(s)
        except ValueError:
            return None
    if sport == "hoops":
        # "2023-24" -> 2024 (the year the season ends); "2023" -> 2023
        m = re.match(r"^(\d{4})-?(\d{2,4})?$", s)
        if not m:
            return None
        start = int(m.group(1))
        end = m.group(2)
        if end:
            end_i = int(end)
            end_i = end_i + 2000 if end_i < 100 else end_i
            return end_i
        return start
    if sport == "pitch":
        # "WC 2018", "EURO 2020", "2022/2023" -> first 4-digit year
        m = re.search(r"(\d{4})", s)
        return int(m.group(1)) if m else None
    return None


def nationality_from_team(sport: str, team: str):
    """pitch team is a country; hoops team is ''; gridiron team is a club abbr."""
    if sport == "pitch" and team:
        return team
    return None


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    MARKET.mkdir(parents=True, exist_ok=True)
    all_data = load_all(verbose=False)

    # dedup per (sport, player_id); collect seasons
    per_player: dict[tuple[str, str], dict] = {}
    for sport, bundle in all_data.items():
        for r in bundle["records"]:
            key = (sport, str(r["player_id"]))
            nat = nationality_from_team(sport, str(r.get("team", "")))
            sy = season_year(r["season"], sport)
            if key not in per_player:
                per_player[key] = {
                    "sport": sport,
                    "player_id": str(r["player_id"]),
                    "name": str(r["name"]),
                    "name_norm": name_norm(str(r["name"])),
                    "nationality": nat,
                    "birthyear": None,
                    "pos": str(r.get("pos", "")),
                    "team": str(r.get("team", "")),
                    "seasons": [],
                    "season_years": [],
                }
            rec = per_player[key]
            if str(r["season"]) not in rec["seasons"]:
                rec["seasons"].append(str(r["season"]))
            if sy is not None and sy not in rec["season_years"]:
                rec["season_years"].append(sy)

    players = list(per_player.values())
    # sort seasons
    for p in players:
        p["seasons"] = sorted(p["seasons"], key=str)
        p["season_years"] = sorted(set(p["season_years"]))
        p["bio_augment_needed"] = (p["nationality"] is None) or (p["birthyear"] is None)

    # reverse lookup by canonical key (name_norm | nationality | birthyear)
    by_canonical: dict[str, list[int]] = defaultdict(list)
    name_only: dict[str, list[int]] = defaultdict(list)  # for ambiguity detection
    for i, p in enumerate(players):
        ck = f"{p['name_norm']}|{p['nationality']}|{p['birthyear']}"
        by_canonical[ck].append(i)
        name_only[p["name_norm"]].append(i)

    # coverage + ambiguity
    coverage = {}
    for sport in ("hoops", "gridiron", "pitch"):
        idxs = [i for i, p in enumerate(players) if p["sport"] == sport]
        n = len(idxs)
        has_nat = sum(1 for i in idxs if players[i]["nationality"])
        has_by = sum(1 for i in idxs if players[i]["birthyear"] is not None)
        amb = sum(1 for i in idxs if len(name_only[players[i]["name_norm"]]) > 1)
        coverage[sport] = {
            "n_players": n,
            "has_nationality": has_nat,
            "has_birthyear": has_by,
            "bio_augment_needed": n - has_nat,  # birthyear absent for all currently
            "ambiguous_name_only": amb,
        }

    out = {
        "n_players": len(players),
        "players": players,
        "by_canonical": {k: v for k, v in by_canonical.items()},
        "coverage": coverage,
        "schema_version": 1,
        "note": ("name_norm + nationality + birthyear is the canonical join key. "
                 "birthyear is None for all sports (bio-augment pull pending for "
                 "hoops/gridiron via BBREF/PFR; pitch via Transfermarkt/Wikipedia). "
                 "nationality present only for pitch (team=country). Ambiguous "
                 "name-only matches must be left masked downstream (schema §4/§8)."),
    }
    (MARKET / "name_index.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"wrote {MARKET / 'name_index.json'}: {len(players)} players")
    for sport, c in coverage.items():
        print(f"  {sport:9s} n={c['n_players']:5d}  nat={c['has_nationality']:5d}  "
              f"birthyear={c['has_birthyear']:5d}  bio_augment_needed={c['bio_augment_needed']:5d}  "
              f"ambiguous_name={c['ambiguous_name_only']:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
