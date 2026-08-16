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

from load_encoders import ROOT, load_all

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


def _verify_season_years(players: list[dict]) -> None:
    """Every derived year must sit within a year of the label it came from.

    A guard rather than a comment, because the century bug it catches was invisible for
    the same reason every other defect this phase was: the output looked like years.
    2097 is a plausible-looking integer in a column of plausible-looking integers, and
    5.6% of them were wrong. Only labels that BEGIN with a 4-digit year are checkable —
    pitch carries "Euro 2020" and "WC 2018", which this deliberately skips rather than
    guessing at.
    """
    bad: list[str] = []
    for p in players:
        # Recomputed from the labels rather than zipped against season_years: the two
        # lists are built with independent de-duplication, so their positions do not
        # correspond and a zip would compare unrelated pairs.
        for s in p["seasons"]:
            m = re.match(r"^(\d{4})", str(s))
            if not m:
                continue
            y = season_year(s, p["sport"])
            if y is None:
                continue
            start = int(m.group(1))
            if not (start - 1 <= y <= start + 1):
                bad.append(f"{p['sport']} {p['name']}: season {s!r} -> year {y}")
        derived = {season_year(s, p["sport"]) for s in p["seasons"]} - {None}
        if derived != set(p["season_years"]):
            bad.append(
                f"{p['sport']} {p['name']}: season_years {sorted(p['season_years'])} "
                f"!= years derivable from its labels {sorted(derived)}"
            )
    if bad:
        raise SystemExit(
            f"REFUSING TO WRITE: {len(bad)} season_year value(s) do not match their own "
            f"season label.\n  " + "\n  ".join(bad[:5]) + ("\n  ..." if len(bad) > 5 else "")
        )


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
            if end_i < 100:
                # CENTURY COMES FROM `start`, NOT FROM A HARD-CODED 2000. The previous
                # `end_i + 2000` dated every 20th-century season a hundred years late:
                # "1996-97" -> 2097, and AC Green's career read [2000, 2001, 2097, 2098].
                # 1,168 of 20,689 values (5.6%) across 530 of 5,837 players. Nothing
                # consumed season_years yet, so nothing downstream was wrong — but the
                # field's own docstring says it exists "for season-varying joins", which
                # is precisely the use that would have attached 1996 data to 2097.
                end_i += start - (start % 100)
                if end_i < start:
                    end_i += 100  # "1999-00" -> 1900 -> 2000
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
    _verify_season_years(players)
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
        "note": (
            "name_norm + nationality + birthyear is the canonical join key. "
            "birthyear is None for all sports (bio-augment pull pending for "
            "hoops/gridiron via BBREF/PFR; pitch via Transfermarkt/Wikipedia). "
            "nationality present only for pitch (team=country). Ambiguous "
            "name-only matches must be left masked downstream (schema §4/§8)."
        ),
    }
    (MARKET / "name_index.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"wrote {MARKET / 'name_index.json'}: {len(players)} players")
    for sport, c in coverage.items():
        print(
            f"  {sport:9s} n={c['n_players']:5d}  nat={c['has_nationality']:5d}  "
            f"birthyear={c['has_birthyear']:5d}  bio_augment_needed={c['bio_augment_needed']:5d}  "
            f"ambiguous_name={c['ambiguous_name_only']:4d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
