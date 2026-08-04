"""Turn the venue->sponsor MATCH LIST into an actual edge on player-seasons, and measure it.

acquire_venue_sponsors.py answered "does a joinable athlete<->company edge exist at all?"
-- yes, 16 venues resolve to a unique S&P 500 company, 14 correctly, against 0 for the
tennis tournament-name route. It closed by naming what it had NOT answered:

    "14 venues is the edge, not the coverage."

This script answers that. venue -> team -> player-season, then counts.

WHY ONLY GRIDIRON. The unified matrix carries sport_id, era_id, arch_id, pos_id and
player_idx -- and no team column at all. Of the three sports only gridiron keeps team on
its own artifact (gridiron_season_emb.npz, 5,323 rows, 2016-2025, 32 codes). So the
gridiron half is joinable today and the hoops half is not, and that is a fact about what
is on disk, not a judgement about which sport matters. The NBA venue->team table is parsed
and written out anyway so the hoops join is a lookup away once a team column exists;
`hoops_join_blocked_by` in the report names exactly what is missing.

WHAT AN EDGE HERE MEANS, PRECISELY. "This player-season was played by a team whose home
venue was, at some point, named for this S&P 500 company." It does NOT mean the venue
carried that name in that season -- the Wikipedia tables are CURRENT stadiums, so the
name is today's name and the corpus runs back to 2016. Delta Center was Vivint Arena
until 2023; Intuit Dome opened in 2024. Treating a 2016 Jazz season as a Delta edge is
wrong, so the report separates `named_window_edges` from the naive `all_season_edges`.

The first version of that separation filtered on the table's `Opened` column and was
itself wrong: `Opened` is when the BUILDING opened, not when the NAME went up. AT&T
Stadium opened in 2009 as Cowboys Stadium and AT&T bought the name in 2013. See the
NAMING_YEAR comment below -- the two numbers being equal in this run is a property of
which five venues matched, not evidence that the filter works.

    python pipeline/build_venue_edges.py

Reads:  data/market_cultural/_venue_wikitext_cache.json  (acquire_venue_sponsors.py --offline
        leaves it; no new fetch, no key)
        data/market_cultural/venue_sponsors.json
        pipeline/data/gridiron_season_emb.npz
Writes: data/venue_edges.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "market_cultural" / "_venue_wikitext_cache.json"
MATCHES = ROOT / "data" / "market_cultural" / "venue_sponsors.json"
GRID_EMB = ROOT / "pipeline" / "data" / "gridiron_season_emb.npz"
OUT = ROOT / "data" / "venue_edges.json"

# nflverse team codes. Hardcoded because they are a naming convention, not a derivable
# rule -- no function turns "Las Vegas Raiders" into LV and "Los Angeles Rams" into LA
# while "Los Angeles Chargers" becomes LAC. The build REFUSES if a Wikipedia team name
# fails to map, so a franchise rename breaks loudly instead of silently dropping rows.
NFL_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

# THE YEAR THE SPONSOR'S NAME WENT UP -- which is NOT the year the building opened.
#
# The first version of this script filtered seasons on the table's `Opened` column and
# called the result `season_aware`. It produced a number identical to the naive one, which
# looked like confirmation and was actually the bug hiding: every venue that survived the
# match happens to be old, so an opened-year filter never bit. AT&T Stadium is the proof
# the method was wrong -- it opened in 2009 as Cowboys Stadium and AT&T did not buy the
# name until 2013. A four-year gap that changed nothing here only because the corpus
# starts in 2016, and would silently corrupt the count the moment a Lumen Field (opened
# 2002, CenturyLink until 2020) or an Acrisure Stadium (opened 2001, Heinz Field until
# 2022) entered the match list. A real value answering a different question than the one
# it appears to answer, in my own verification step.
#
# These six are hand-verified, and hand-verification does not scale. The scalable source
# is each venue's own Wikipedia infobox `former_names`, which is one fetch per matched
# venue and is the right next step if this list grows. A venue absent from this map
# contributes 0 to the named-window count rather than an assumed year.
NAMING_YEAR = {
    "MetLife Stadium": 2011,        # opened 2010 as New Meadowlands Stadium
    "M&T Bank Stadium": 2003,       # opened 1998 as Ravens Stadium at Camden Yards
    "Raymond James Stadium": 1998,  # named from opening
    "Ford Field": 2002,             # named from opening
    "AT&T Stadium": 2013,           # opened 2009 as Cowboys Stadium
}

ROW = re.compile(r'!\s*scope="row"\s*\|\s*\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
CELL_LINKS = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def parse_table(wikitext: str) -> list[dict]:
    """Rows of (venue, teams, location, opened) from the `! scope="row"` table shape.

    Both pages put the venue in a row-scope header cell and the team(s) and location in
    the following `|` cells, so split on the row separator and read positionally within
    each row rather than trying to model the two pages' differing column counts."""
    rows = []
    for block in re.split(r"\n\|-", wikitext):
        m = ROW.search(block)
        if not m:
            continue
        venue = m.group(1).strip()
        after = block[m.end():]
        cells = [c.strip() for c in after.split("\n|") if c.strip()]
        teams, location, opened = [], "", None
        for c in cells[:4]:
            links = [a or b for a, b in CELL_LINKS.findall(c)]
            links = [x.strip() for x in links if x and "File:" not in x]
            if not teams and links and any(
                    t in NFL_ABBR or " " in t for t in links):
                teams = links
            elif not location and links:
                location = links[0]
        om = re.search(r"\b(19\d{2}|20\d{2})\b", after[:1200])
        if om:
            opened = int(om.group(1))
        rows.append({"venue": venue, "teams": teams, "location": location,
                     "opened": opened})
    return rows


def main() -> int:
    for p in (CACHE, MATCHES, GRID_EMB):
        if not p.exists():
            print(f"FAIL: missing {p}", file=sys.stderr)
            return 2

    raw = json.loads(CACHE.read_text(encoding="utf-8"))
    md = json.loads(MATCHES.read_text(encoding="utf-8"))

    # venue -> (ticker, company), unique-phrase tier only. The token tier is the loose
    # upper bound and contains State Farm Arena -> State Street, which is wrong.
    v2c = {}
    for sport, rows in md["matches"].items():
        for r in rows:
            if len(r["tier1_phrase_match"]) == 1:
                v2c[r["venue"]] = tuple(r["tier1_phrase_match"][0])

    # the two adjudicated-wrong rows are DROPPED here, by name, with the reason recorded.
    # This is a correction to a known-bad output, not a tuned filter: both were read and
    # judged in acquire_venue_sponsors.py before this script existed.
    GEOGRAPHY_FALSE_POSITIVES = {
        "Boston Garden": "named for the city; Boston Scientific is not the sponsor",
        "Cincinnati Gardens": "named for the city; Cincinnati Financial is not the sponsor",
    }
    dropped = {k: v for k, v in GEOGRAPHY_FALSE_POSITIVES.items() if k in v2c}
    for k in dropped:
        v2c.pop(k)

    tables = {s: parse_table(wt) for s, wt in raw.items()}

    # --- refuse on an unmapped NFL team rather than dropping it silently -------
    seen_teams = {t for r in tables["gridiron"] for t in r["teams"]}
    plausible = {t for t in seen_teams if any(
        t.endswith(" " + w) for w in
        ("Cardinals", "Falcons", "Ravens", "Bills", "Panthers", "Bears", "Bengals",
         "Browns", "Cowboys", "Broncos", "Lions", "Packers", "Texans", "Colts",
         "Jaguars", "Chiefs", "Raiders", "Chargers", "Rams", "Dolphins", "Vikings",
         "Patriots", "Saints", "Giants", "Jets", "Eagles", "Steelers", "49ers",
         "Seahawks", "Buccaneers", "Titans", "Commanders"))}
    unmapped = sorted(plausible - set(NFL_ABBR))
    if unmapped:
        print("FAIL: Wikipedia names an NFL team this build cannot map to a corpus "
              f"code: {unmapped}. Add it to NFL_ABBR rather than letting those "
              "player-seasons vanish from the coverage denominator.", file=sys.stderr)
        return 3

    # --- gridiron edges -------------------------------------------------------
    z = np.load(GRID_EMB, allow_pickle=True)
    team_arr = np.array([str(x) for x in z["team"]])
    season_arr = np.array([int(x) for x in z["season"]])

    edges, season_aware = [], []
    for r in tables["gridiron"]:
        if r["venue"] not in v2c:
            continue
        tk, co = v2c[r["venue"]]
        for t in r["teams"]:
            ab = NFL_ABBR.get(t)
            if not ab:
                continue
            n_all = int((team_arr == ab).sum())
            named = NAMING_YEAR.get(r["venue"])
            n_sa = int(((team_arr == ab) & (season_arr >= named)).sum()) if named else 0
            edges.append({"venue": r["venue"], "team_wiki": t, "team_code": ab,
                          "ticker": tk, "company": co, "location": r["location"],
                          "building_opened": r["opened"],
                          "sponsor_name_from": named,
                          "naming_year_source": "hand-verified" if named else "UNKNOWN",
                          "player_seasons_all": n_all,
                          "player_seasons_while_named": n_sa})
            season_aware.append(n_sa)

    all_ps = sum(e["player_seasons_all"] for e in edges)
    sa_ps = sum(season_aware)
    total = len(team_arr)

    out = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": "How many player-seasons does the venue->sponsor edge actually reach?",
        "answers_the_open_question_from": "pipeline/acquire_venue_sponsors.py — "
                                          "'14 venues is the edge, not the coverage.'",
        "gridiron": {
            "corpus_player_seasons": total,
            "corpus_seasons": [int(season_arr.min()), int(season_arr.max())],
            "n_edges": len(edges),
            "n_distinct_tickers": len({e["ticker"] for e in edges}),
            "all_season_edges_player_seasons": all_ps,
            "all_season_coverage_pct": round(100.0 * all_ps / total, 2),
            "named_window_edges_player_seasons": sa_ps,
            "named_window_coverage_pct": round(100.0 * sa_ps / total, 2),
            "which_number_to_quote": "named_window. The Wikipedia tables list "
                                     "CURRENT stadium names and the corpus starts in "
                                     "2016, so the naive count credits a season to a "
                                     "sponsor whose name may have gone up later. The two "
                                     "numbers are equal here ONLY because all five "
                                     "sponsor names predate 2016 -- AT&T Stadium was "
                                     "Cowboys Stadium until 2013, three years outside "
                                     "the corpus. Equality is a property of this edge "
                                     "list, not evidence the filter works.",
            "naming_years_are_hand_verified": True,
            "naming_year_scalable_source": "each venue's Wikipedia infobox former_names; "
                                           "one fetch per matched venue. Required before "
                                           "this edge list grows past the hand-checked 5.",
        },
        "hoops_join_blocked_by": "No team column exists for the hoops corpus. "
                                 "unified_matrix.npz carries sport_id/era_id/arch_id/"
                                 "pos_id/player_idx and gridiron_season_emb.npz carries "
                                 "team, but nothing maps a hoops player-season to a "
                                 "franchise. The NBA venue->team table is parsed and "
                                 "written below so this is a lookup away, not a rebuild.",
        "dropped_adjudicated_false_positives": dropped,
        "edges": sorted(edges, key=lambda e: -e["player_seasons_while_named"]),
        "nba_venue_teams": [r for r in tables["hoops"] if r["teams"]][:40],
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"gridiron corpus: {total} player-seasons, "
          f"{out['gridiron']['corpus_seasons'][0]}-{out['gridiron']['corpus_seasons'][1]}")
    print(f"  {len(edges)} venue->team edges over "
          f"{out['gridiron']['n_distinct_tickers']} distinct tickers")
    print(f"  naive        {all_ps:>5} player-seasons  "
          f"({out['gridiron']['all_season_coverage_pct']}%)")
    print(f"  named-window {sa_ps:>5} player-seasons  "
          f"({out['gridiron']['named_window_coverage_pct']}%)  <- quote this one")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
