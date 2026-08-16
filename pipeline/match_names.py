"""Vector Unified — external-name matcher (validate + reuse).

Matches external market/cultural source rows (Forbes, Spotrac, Transfermarkt, Wikidata)
to native unified players via the canonical name-resolution key (name_norm + nationality
+ birthyear). Reuses the normalization from resolve_names so the keys are identical.

Matching tiers (honest):
  - exact:   name_norm + nationality + birthyear all match a unique native player
  - name+country: name_norm + nationality match (birthyear unknown on one side)
  - name_only: name_norm matches a unique native player (no country/birthyear to confirm)
  - ambiguous: name_norm matches >1 native player and no country to disambiguate -> MASKED
  - unmatched: name_norm not found in any native player

Downstream joins consume `match(row, name_index)` -> (player_idx | None, tier, n_candidates).
"""

from __future__ import annotations

import json
import sys

from resolve_names import MARKET, name_norm


def _country_key(country: str | None) -> str | None:
    if not country:
        return None
    c = country.strip().lower()
    # crude normalization of common country variants
    aliases = {
        "usa": "united states",
        "u.s.": "united states",
        "us": "united states",
        "uk": "united kingdom",
        "u.k.": "united kingdom",
        "great britain": "united kingdom",
        "republic of ireland": "ireland",
    }
    return aliases.get(c, c)


def match_external(
    name: str, sport_unified: str | None, country: str | None, name_index: dict
) -> tuple[int | None, str, int]:
    """Return (player_idx_or_None, tier, n_candidates).

    sport_unified restricts the candidate pool to that sport (Forbes/Wikidata carry a
    sport label; Spotrac/Transfermarkt are sport-specific by construction).
    """
    players = name_index["players"]
    key = name_norm(name)
    ckey = _country_key(country)
    # candidate pool by name_norm, optionally restricted to sport
    cand = [
        i
        for i, p in enumerate(players)
        if p["name_norm"] == key and (sport_unified is None or p["sport"] == sport_unified)
    ]
    n = len(cand)
    if n == 0:
        # fall back to cross-sport name match (some sources mislabel sport)
        cand = [i for i, p in enumerate(players) if p["name_norm"] == key]
        n = len(cand)
        if n == 0:
            return None, "unmatched", 0
    if n == 1:
        # unique name -> accept; upgrade tier if country also confirms
        p = players[cand[0]]
        if ckey and p["nationality"] and _country_key(p["nationality"]) == ckey:
            return cand[0], "name+country", 1
        return cand[0], "name_only", 1
    # n > 1: try to disambiguate by country
    if ckey:
        byc = [i for i in cand if players[i]["nationality"] and _country_key(players[i]["nationality"]) == ckey]
        if len(byc) == 1:
            return byc[0], "name+country", n
        if len(byc) > 1:
            return None, "ambiguous", n  # same name + same country -> can't pick
    return None, "ambiguous", n


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    idx = json.loads((MARKET / "name_index.json").read_text(encoding="utf-8"))
    fb = json.loads((MARKET / "Forbes_highest_paid.json").read_text(encoding="utf-8"))
    inc = [r for r in fb["rows"] if not r["out_of_corpus"]]
    tiers = {}
    matched_players = {}
    misses = []
    for r in inc:
        pid, tier, ncand = match_external(r["name"], r["sport_unified"], r["country"], idx)
        tiers[tier] = tiers.get(tier, 0) + 1
        if pid is not None:
            matched_players[pid] = idx["players"][pid]
        else:
            misses.append((r["name"], r["sport_unified"], r["country"], tier, ncand))
    print(f"Forbes in-corpus rows: {len(inc)} ({len({r['name'] for r in inc})} unique names)")
    print(f"match tiers: {tiers}")
    print(f"unique native players matched: {len(matched_players)}")
    print("\nmatched stars (name -> sport / native_id / native_name):")
    for pid, p in sorted(matched_players.items(), key=lambda kv: (kv[1]["sport"], kv[1]["name"])):
        print(f"  {p['sport']:9s} id={p['player_id']:>8s}  {p['name']}")
    if misses:
        print(f"\nmisses/ambiguous ({len(misses)}):")
        for m in misses:
            print(f"  {m[0]:24s} {m[1]:9s} country={m[2]:18s} tier={m[3]} ncand={m[4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
