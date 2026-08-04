"""Turn the venue->sponsor MATCH LIST into an actual edge on player-seasons, and measure it.

acquire_venue_sponsors.py answered "does a joinable athlete<->company edge exist at all?"
-- yes, 16 venues resolve to a unique S&P 500 company, 14 correctly, against 0 for the
tennis tournament-name route. It closed by naming what it had NOT answered:

    "14 venues is the edge, not the coverage."

This script answers that. venue -> team -> player-season, then counts.

BOTH SPORTS JOIN. The first version of this script reported hoops as blocked, on the
grounds that unified_matrix.npz carries sport_id/era_id/arch_id/pos_id/player_idx and no
team column. That was true and it was not the whole shelf: vector-hoops/assets/
player_meta.json carries `roster`, a "Name|Season" -> team map over 3,385 player-seasons
(2015-16..2025-26). It also claimed the NBA venue->team table was "parsed and written
out" -- it wrote an EMPTY list, because parse_table() matches the NFL page's
`! scope="row"` shape and the NBA page uses `| '''[[Arena]]'''` with the team and
location columns in the opposite order. Zero rows, no error, and a commit message
asserting the opposite.

A name-keyed join is the exact hazard behind the Jaren Jackson / Jaren Jackson Jr. bug,
so it is checked rather than assumed: zero "Name|Season" keys in the hoops corpus map to
more than one player_id. If that ever changes the build refuses instead of merging two
careers.

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
HOOPS_META = Path(r"C:\Users\jcdav\vector-hoops\assets\player_meta.json")
HOOPS_EMB = Path(r"C:\Users\jcdav\vector-hoops\pipeline\data\embedding_v3.npz")
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

# NBA arena naming years. Same discipline as the NFL map above and it MATTERS MORE here:
# four of these six names went up inside the corpus window (2015-16..2025-26), so an
# all-seasons count would be wrong by construction rather than wrong-in-principle-only.
NBA_NAMING_YEAR = {
    "Chase Center": 2019,       # Warriors moved from Oracle Arena
    "Fiserv Forum": 2018,       # opened as Fiserv Forum
    "Delta Center": 2023,       # was Vivint Arena / Vivint Smart Home Arena
    "Intuit Dome": 2024,        # opened as Intuit Dome
    "Ball Arena": 2020,         # was Pepsi Center until Oct 2020
    "Capital One Arena": 2017,  # was Verizon Center until Aug 2017
    "Target Center": 1990,      # named from opening
    "FedExForum": 2004,         # named from opening
}

NBA_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

ROW = re.compile(r'!\s*scope="row"\s*\|\s*\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
NBA_ARENA_CELL = re.compile(r"^\s*'''\[\[([^\]|]+)(?:\|[^\]]+)?\]\]'''")
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


def parse_nba_table(wikitext: str) -> list[dict]:
    """The NBA page uses a DIFFERENT table shape from the NFL page and parse_table()
    silently returned zero rows for it.

    NFL:  ! scope="row"|[[Venue]]  then | [[Team]] | [[Location]]
    NBA:  | '''[[Arena]]'''        then | [[Location]] | [[Team]]

    Note the column ORDER is also swapped. A parser that found the arena but read the
    cells positionally by the NFL layout would have attached every arena to its city
    instead of its team -- which is not a crash, it is a plausible-looking wrong join,
    so the team is matched against NBA_ABBR by name rather than taken by position."""
    rows = []
    for block in re.split(r"\n\|-", wikitext):
        cells = [c.strip() for c in block.split("\n|") if c.strip()]
        arena = None
        for c in cells:
            m = NBA_ARENA_CELL.match(c)
            if m:
                arena = m.group(1).strip()
                break
        if not arena:
            continue
        teams, location = [], ""
        for c in cells:
            for a, b in CELL_LINKS.findall(c):
                t = (a or b).strip()
                if t in NBA_ABBR:
                    teams.append(t)
                elif not location and "," in t and "File:" not in t:
                    location = t
        if teams:
            rows.append({"venue": arena, "teams": sorted(set(teams)),
                         "location": location})
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

    # --- hoops ---------------------------------------------------------------
    # UNBLOCKED since the first version of this script, which reported the hoops join as
    # impossible. It is not: vector-hoops/assets/player_meta.json carries a `roster` map
    # of "Name|Season" -> team code. It covers 3,385 of the 12,966 corpus player-seasons
    # (26.1%, seasons 2015-16..2025-26 only), so the hoops denominator below is the
    # ROSTER-COVERED subset, not the corpus.
    #
    # Name-keyed joins are the exact hazard behind the Jaren Jackson / Jaren Jackson Jr.
    # bug, so this is checked rather than assumed: zero "Name|Season" keys in the corpus
    # map to more than one player_id, so the key is unique here and the join is safe. If
    # that ever stops being true the build refuses.
    hoops = {"blocked": None}
    if HOOPS_META.exists() and HOOPS_EMB.exists():
        roster = json.loads(HOOPS_META.read_text(encoding="utf-8"))["roster"]
        hz = np.load(HOOPS_EMB, allow_pickle=True)
        hkeys = [f"{n}|{s}" for n, s in zip(
            (str(x) for x in hz["name"]), (str(x) for x in hz["season"]))]
        collide = {}
        for k, pid in zip(hkeys, (str(x) for x in hz["player_id"])):
            collide.setdefault(k, set()).add(pid)
        ambiguous = {k for k, v in collide.items() if len(v) > 1}
        if ambiguous:
            print(f"FAIL: {len(ambiguous)} Name|Season keys map to more than one "
                  f"player_id, e.g. {sorted(ambiguous)[:3]}. A name-keyed join would "
                  "merge two players' careers -- the Jaren Jackson bug. Refusing.",
                  file=sys.stderr)
            return 4

        nba = parse_nba_table(raw["hoops"])
        unmapped_arena = [r["venue"] for r in nba if not r["teams"]]
        hedges = []
        for r in nba:
            if r["venue"] not in v2c:
                continue
            tk, co = v2c[r["venue"]]
            named = NBA_NAMING_YEAR.get(r["venue"])
            for t in r["teams"]:
                ab = NBA_ABBR[t]
                # season strings are "2015-16"; the start year is what NAMING_YEAR
                # compares against
                rows_all = [k for k in hkeys if roster.get(k) == ab]
                rows_named = [k for k in rows_all
                              if named and int(k.split("|")[1].split("-")[0]) >= named]
                hedges.append({
                    "venue": r["venue"], "team_wiki": t, "team_code": ab,
                    "ticker": tk, "company": co, "location": r["location"],
                    "sponsor_name_from": named,
                    "naming_year_source": "hand-verified" if named else "UNKNOWN",
                    "player_seasons_all": len(rows_all),
                    "player_seasons_while_named": len(rows_named),
                })
        h_all = sum(e["player_seasons_all"] for e in hedges)
        h_named = sum(e["player_seasons_while_named"] for e in hedges)
        hoops = {
            "corpus_player_seasons": len(hkeys),
            "roster_covered_player_seasons": sum(1 for k in hkeys if k in roster),
            "roster_seasons": "2015-16..2025-26",
            "name_season_keys_colliding_on_player_id": 0,
            "nba_table_rows_parsed": len(nba),
            "nba_rows_without_a_mapped_team": unmapped_arena,
            "n_edges": len(hedges),
            "n_distinct_tickers": len({e["ticker"] for e in hedges}),
            "all_season_edges_player_seasons": h_all,
            "named_window_edges_player_seasons": h_named,
            "denominator_note": "Percentages here are over the 3,385 roster-covered "
                                "player-seasons, NOT the 12,966-row corpus. Quoting them "
                                "against the corpus would overstate coverage 3.8x.",
            "named_window_pct_of_roster_covered": None,
            "edges": sorted(hedges, key=lambda e: -e["player_seasons_while_named"]),
        }
        cov = hoops["roster_covered_player_seasons"]
        hoops["named_window_pct_of_roster_covered"] = round(100.0 * h_named / cov, 2) \
            if cov else 0.0
        hoops["all_season_pct_of_roster_covered"] = round(100.0 * h_all / cov, 2) \
            if cov else 0.0

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
        "hoops": hoops,
        "CORRECTION_hoops_was_not_actually_blocked": "8ea1d98 reported the hoops join as "
            "impossible because unified_matrix.npz has no team column, and claimed the "
            "NBA venue->team table was 'parsed and written out'. Both were wrong. "
            "vector-hoops/assets/player_meta.json carries roster: Name|Season -> team, "
            "and parse_table() had silently returned ZERO rows for the NBA page because "
            "that page uses a different table shape. The commit shipped an empty list "
            "under a claim that it was populated.",
        "dropped_adjudicated_false_positives": dropped,
        "edges": sorted(edges, key=lambda e: -e["player_seasons_while_named"]),
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
    if hoops.get("n_edges"):
        print(f"hoops corpus: {hoops['corpus_player_seasons']} player-seasons, "
              f"{hoops['roster_covered_player_seasons']} covered by the roster map "
              f"({hoops['roster_seasons']})")
        print(f"  {hoops['n_edges']} venue->team edges over "
              f"{hoops['n_distinct_tickers']} distinct tickers")
        print(f"  naive        {hoops['all_season_edges_player_seasons']:>5} "
              f"player-seasons  ({hoops['all_season_pct_of_roster_covered']}%)")
        print(f"  named-window {hoops['named_window_edges_player_seasons']:>5} "
              f"player-seasons  ({hoops['named_window_pct_of_roster_covered']}%)"
              f"  <- quote this one")
        print(f"  the naming-year filter removes "
              f"{hoops['all_season_edges_player_seasons'] - hoops['named_window_edges_player_seasons']}"
              f" player-seasons here; on gridiron it removed 0")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
