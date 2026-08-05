#!/usr/bin/env python3
"""Is a player's TEAMMATE SET the hoops/gridiron analogue of tennis's tournament calendar?

Solo personal project, no connection to employer, built with public/free-tier only

The tennis MTNN went 0.0783 -> 0.1168 when the schedule stopped being encoded as HOW MANY
tournaments and became WHICH of 287, as a binary block. The finding underneath it was that
where a player shows up identifies him better than how well he plays once there.

THE OBVIOUS TRANSLATION IS WRONG. "Which opponents" is not a choice in a team sport --
the league writes the schedule and every NBA team plays every other team. A 30-wide
opponent block is nearly constant across players and carries almost no identity.

The structural analogue is a set the player's own situation SELECTS, sparse against a
large vocabulary, partially persistent year to year. That is the TEAMMATE SET: ~15 names
for a hoops season and ~53 for gridiron, drawn from thousands, and it turns over.

PROBED BEFORE BUILDING, in the same order tennis did it. probe_tennis_calendar_identity.py
ran BEFORE the block was adopted, because a wide sparse binary block is exactly the shape
that can be memorised as a key rather than generalised from. Its four numbers were:

    jaccard own-next-year                  0.3329
    jaccard random same-tour               0.1782
    pairs with an identical set            0 of 2,926
    pct of calendars duplicated in-year    24.3

A teammate set can fail in the OPPOSITE direction from a tennis calendar. A player who
stays on his team keeps most of his teammates, so own-next-year overlap could be so high
that the block is a franchise label wearing a roster's clothes -- an identity KEY, which
is the thing the tennis probe was written to rule out. That is the specific way this can
be a bad idea, and it is stated before the numbers.

    python pipeline/probe_roster_identity.py

Writes: data/roster_identity_probe.json  --  measures only, builds nothing, trains nothing
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEMB = ROOT / "pipeline" / "data" / "gridiron_season_emb.npz"
HEMB = Path(r"C:\Users\jcdav\vector-hoops\pipeline\data\embedding_v3.npz")
HMETA = Path(r"C:\Users\jcdav\vector-hoops\assets\player_meta.json")
OUT = ROOT / "data" / "roster_identity_probe.json"

RNG = np.random.default_rng(7)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def probe(label, rows, note=""):
    """rows: list of (player_key, season_int, team_key). Returns the tennis four."""
    # teammate set = everyone else sharing (team, season)
    by_ts = defaultdict(set)
    for pk, s, t in rows:
        by_ts[(t, s)].add(pk)
    mates = {}
    for pk, s, t in rows:
        mates[(pk, s)] = by_ts[(t, s)] - {pk}

    seasons_of = defaultdict(list)
    for pk, s, t in rows:
        seasons_of[pk].append(s)

    own, rnd, identical, n_pairs = [], [], 0, 0
    by_season = defaultdict(list)
    for (pk, s) in mates:
        by_season[s].append(pk)

    for pk, ss in seasons_of.items():
        ss = sorted(set(ss))
        for a, b in zip(ss, ss[1:]):
            if b - a != 1:
                continue
            ma, mb = mates.get((pk, a), set()), mates.get((pk, b), set())
            if not ma or not mb:
                continue
            n_pairs += 1
            own.append(jaccard(ma, mb))
            if ma == mb:
                identical += 1
            # random control: another player's NEXT-year set, same season, so
            # roster-size effects are held fixed
            pool = [q for q in by_season.get(b, []) if q != pk]
            if pool:
                q = pool[int(RNG.integers(len(pool)))]
                rnd.append(jaccard(ma, mates.get((q, b), set())))

    # THE DECISIVE ARM: split own-next-year by whether the player CHANGED TEAM.
    # A tennis calendar follows the PLAYER. If a teammate set follows the FRANCHISE it is
    # a team label with extra steps, and a 30-wide one-hot carries the same information
    # more cheaply and without the false impression of richness. The headline separation
    # cannot tell these apart; this can.
    stay, move = [], []
    team_of = {(pk, s): t for pk, s, t in rows}
    for pk, ss in seasons_of.items():
        ss = sorted(set(ss))
        for a, b in zip(ss, ss[1:]):
            if b - a != 1:
                continue
            ma, mb = mates.get((pk, a), set()), mates.get((pk, b), set())
            if not ma or not mb:
                continue
            (stay if team_of[(pk, a)] == team_of[(pk, b)] else move).append(jaccard(ma, mb))

    # Two players on the SAME roster: how similar are their sets? If this is near 1 the
    # block cannot distinguish players within a team at all.
    within = []
    for (t, s), members in list(by_ts.items()):
        ml = sorted(members)
        for i in range(min(3, len(ml))):
            for j in range(i + 1, min(4, len(ml))):
                within.append(jaccard(mates[(ml[i], s)], mates[(ml[j], s)]))

    # how often is a set duplicated by someone else in the same season?
    dup = 0
    tot = 0
    for s, pks in by_season.items():
        seen = defaultdict(int)
        for pk in pks:
            seen[frozenset(mates[(pk, s)])] += 1
        for fs, c in seen.items():
            tot += c
            if c > 1:
                dup += c

    sizes = [len(v) for v in mates.values() if v]
    return {
        "label": label, "note": note,
        "n_player_seasons": len(mates),
        "median_teammates": int(np.median(sizes)) if sizes else 0,
        "max_teammates": int(max(sizes)) if sizes else 0,
        "n_consecutive_pairs": n_pairs,
        "jaccard_own_next_year": round(float(np.mean(own)), 4) if own else None,
        "jaccard_random_same_season": round(float(np.mean(rnd)), 4) if rnd else None,
        "jaccard_separation": (round(float(np.mean(own) - np.mean(rnd)), 4)
                               if own and rnd else None),
        "pairs_with_identical_set": identical,
        "pct_sets_duplicated_in_same_season": round(100.0 * dup / max(tot, 1), 1),
        "pct_duplicated_IS_A_STRUCTURAL_ZERO": "Not comparable to tennis's 24.3%. Two "
            "teammates' sets each exclude the holder, so they differ by exactly those two "
            "and can never be identical. Tennis's figure is real because two players CAN "
            "enter the same tournaments. This column measures nothing here.",
        "DECISIVE_stay_vs_move": {
            "n_stayed": len(stay), "n_moved": len(move),
            "jaccard_stayed": round(float(np.mean(stay)), 4) if stay else None,
            "jaccard_moved": round(float(np.mean(move)), 4) if move else None,
            "pct_of_stayer_overlap_retained_by_movers": (
                round(100.0 * np.mean(move) / np.mean(stay), 1)
                if stay and move and np.mean(stay) else None),
            "jaccard_between_two_players_on_the_SAME_roster":
                round(float(np.mean(within)), 4) if within else None,
        },
    }


def main() -> int:
    if not GEMB.exists():
        print(f"FAIL: missing {GEMB}", file=sys.stderr)
        return 2
    results = {}

    # ---- gridiron: team and season are columns on the artifact -----------------
    g = np.load(GEMB, allow_pickle=True)
    grows = [(str(p), int(s), str(t))
             for p, s, t in zip(g["gsis"], g["season"], g["team"])]
    results["gridiron"] = probe(
        "gridiron", grows,
        "team and season come straight off gridiron_season_emb.npz; all 5,323 rows.")

    # ---- hoops: team via the Name|Season roster map ----------------------------
    if HEMB.exists() and HMETA.exists():
        h = np.load(HEMB, allow_pickle=True)
        roster = json.loads(HMETA.read_text(encoding="utf-8"))["roster"]
        hrows = []
        for pid, nm, se in zip(h["player_id"], h["name"], h["season"]):
            key = f"{nm}|{se}"
            t = roster.get(key)
            if not t:
                continue
            yr = str(se)[:4]
            if not yr.isdigit():
                continue
            hrows.append((str(pid), int(yr), t))
        results["hoops"] = probe(
            "hoops", hrows,
            f"team via assets/player_meta.json roster map; {len(hrows)} of "
            f"{len(h['name'])} corpus rows covered (2015-16..2025-26 only). "
            "Keyed on player_id, NOT name — the Jaren Jackson hazard.")

    tennis = {
        "jaccard_own_next_year": 0.3329, "jaccard_random_same_tour": 0.1782,
        "jaccard_separation": 0.1547, "pairs_with_identical_set": 0,
        "pct_duplicated_in_same_tour_year": 24.3, "median_per_player_year": 17,
    }

    out = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": "Is a player's TEAMMATE SET the hoops/gridiron analogue of tennis's "
                    "287-wide tournament calendar — a sparse chosen set that identifies?",
        "why_not_opponents": "The league writes the schedule. Every NBA team plays every "
                             "other team, so a 30-wide opponent block is nearly constant "
                             "across players and carries almost no identity. The tennis "
                             "calendar identifies because the player CHOOSES it.",
        "how_this_can_fail_stated_first": "A teammate set can fail in the OPPOSITE "
            "direction from a tennis calendar. A player who stays put keeps most of his "
            "teammates, so own-next-year overlap could be so high that the block is a "
            "franchise label wearing a roster's clothes — an identity KEY, which is "
            "exactly what probe_tennis_calendar_identity.py was written to rule out.",
        "tennis_reference": tennis,
        "results": results,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"{'':<10}{'own':>8}{'random':>9}{'sep':>8}{'ident':>7}{'dup%':>7}"
          f"{'median':>8}{'pairs':>8}")
    print(f"  {'tennis':<8}{tennis['jaccard_own_next_year']:>8}"
          f"{tennis['jaccard_random_same_tour']:>9}{tennis['jaccard_separation']:>8}"
          f"{tennis['pairs_with_identical_set']:>7}"
          f"{tennis['pct_duplicated_in_same_tour_year']:>7}"
          f"{tennis['median_per_player_year']:>8}{'2926':>8}")
    for k, r in results.items():
        print(f"  {k:<8}{r['jaccard_own_next_year']:>8}"
              f"{r['jaccard_random_same_season']:>9}{r['jaccard_separation']:>8}"
              f"{r['pairs_with_identical_set']:>7}"
              f"{r['pct_sets_duplicated_in_same_season']:>7}"
              f"{r['median_teammates']:>8}{r['n_consecutive_pairs']:>8}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
