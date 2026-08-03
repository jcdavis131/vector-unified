#!/usr/bin/env python3
"""Player x TOURNAMENT entities from individual match play — the right grain for tennis.

Solo personal project, no connection to employer, built with public/free-tier only

Operator direction: for individual sports, use individual match play and tournament results.
This supersedes the player-SEASON aggregation in `probe_tennis_expectation.py`, which was
the wrong unit and threw away most of what the corpus knows.

WHY THE SEASON WAS WRONG. A season win rate says a player won 62% of matches. It does not
say against whom, on what surface, at what stage, or in which draws. 67,081 matches collapse
to ~2,500 player-seasons — a 27x loss — and the discarded columns are exactly the ones that
make tennis worth adding: opponent rank, surface, round reached, location, series tier.

Worse, a season win rate is *confounded by schedule*. A player who enters weak ATP250 draws
wins more than one who enters Masters 1000 draws, at equal ability. That is the same shape
as the team-strength confound that dominated the pitch age axis (+0.2626 against a -0.1671
signal), arriving through the opponent instead of the teammate — and unlike pitch, this
corpus records the opponent, so it can be controlled rather than merely noted.

THE ENTITY is player x tournament-edition, mirroring pitch's player x context and
vector-hoops' player x season. One row per player per event they appeared in.

DELIVERY, and every part of it is opponent-aware:

    rounds_won          matches won in this event
    round_reached       furthest round, ordered by ROUND_ORDER
    draw_progress       round_reached / max round in that event — normalised because a
                        Grand Slam draw is 7 rounds and an ATP250 is 5, so "reached the
                        quarterfinal" is not one thing
    opp_rank_median     median opponent rank faced — the schedule control
    beat_better_ranked  wins over a higher-ranked opponent; the schedule-free signal
    set_ratio, game_ratio

EXPECTATION is the player's rank ENTERING the event, taken from their first match in it.
That is genuinely prior to the result, unlike a season-median rank which absorbs the event
being scored. `probe_tennis_expectation.py` had to lag by a full season to get a clean
prior; at this grain the rank is already prior, and the lag is unnecessary.

WALKOVERS AND RETIREMENTS ARE FLAGGED, NOT DROPPED. `Comment` marks 75 retirements and 18
walkovers in 2024 alone. A walkover is a win with no play; counting it as delivery inflates
the winner and counting it as a loss punishes an injury. Both counts are carried on the row
so a consumer decides, rather than this file deciding silently.

    python pipeline/build_tennis_entities.py
    python pipeline/build_tennis_entities.py --tour wta
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acquire_tennis import YEARS, path_for, read_sheet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "tennis_entities.json"

# Ordered, because "further" has to be comparable. Round Robin sits with the first stage:
# it is a group phase, not a knockout depth, and ranking it above 1st Round would say a
# player who lost every group match went further than one who won a knockout match.
ROUND_ORDER = {
    "Round Robin": 1, "1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4,
    "Quarterfinals": 5, "Semifinals": 6, "The Final": 7,
}
RANK_CAP = 500.0
MIN_ROWS = 200


def player_key(name: str) -> str:
    """A conservative join key. Reported, never used to merge silently.

    THE NAME FORMAT IS NOT STABLE FOR THE SAME HUMAN, and the damage is concentrated:

        Zhang K-L. / Zhang K.L.          same player, two spellings
        Zhang Ze   / Zhang Ze.           same player, trailing period
        Wang Xiy.  / Wang Xiyu           same player, abbreviated vs not
        Lu J.J     / Lu Jia-Jing         same player
        Lee Y-H.   / Lee Y.H.            same player

    Almost every case is a Chinese or Korean name, where the source transliterates
    inconsistently. So `distinct players = 1964` is an OVERCOUNT and those players' records
    are fragmented across spellings — an axis built on the raw name would systematically
    split exactly one population's careers. Same class as the Brazilian mononyms that made
    115 pitch names unresolvable, arriving through transliteration instead of ambiguity.

    This normalises punctuation and case ONLY. It does NOT expand abbreviations, because
    `Zhang Ze` and `Zhang Zh.` are plausibly different people and merging them would trade
    a fragmentation error for a conflation error — the worse of the two. The collapse count
    is reported so the size of the residual problem is visible rather than assumed away.
    """
    s = name.lower().replace("-", " ").replace(".", " ")
    return " ".join(s.split())


def num(v, default=None):
    try:
        f = float(v)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


def build(women: bool) -> list[dict]:
    ent: dict[tuple, dict] = {}
    for y in YEARS:
        p = path_for(y, women)
        if not p.exists():
            continue
        hdr, body = read_sheet(p)
        i = {c: k for k, c in enumerate(hdr)}
        need = ("Tournament", "Round", "Winner", "Loser", "WRank", "LRank")
        if not all(c in i for c in need):
            continue

        # max round actually played in each event, so draw_progress is normalised against
        # the draw that existed rather than against a constant
        maxr: dict[tuple, int] = collections.defaultdict(int)
        for r in body:
            key = (str(r[i["Tournament"]]).strip(), y)
            maxr[key] = max(maxr[key], ROUND_ORDER.get(str(r[i["Round"]]).strip(), 0))

        for r in body:
            tname = str(r[i["Tournament"]]).strip()
            if not tname:
                continue
            ev = (tname, y)
            rnd = ROUND_ORDER.get(str(r[i["Round"]]).strip(), 0)
            comment = str(r[i["Comment"]]).strip() if "Comment" in i else ""
            wr, lr = num(r[i["WRank"]], RANK_CAP), num(r[i["LRank"]], RANK_CAP)
            ws, ls = num(r[i["Wsets"]], 0) or 0, num(r[i["Lsets"]], 0) or 0
            gw = sum(num(r[i[c]], 0) or 0 for c in ("W1", "W2", "W3", "W4", "W5") if c in i)
            gl = sum(num(r[i[c]], 0) or 0 for c in ("L1", "L2", "L3", "L4", "L5") if c in i)

            for name, won, own, opp, sf, sa, gf, ga in (
                (str(r[i["Winner"]]).strip(), 1, wr, lr, ws, ls, gw, gl),
                (str(r[i["Loser"]]).strip(), 0, lr, wr, ls, ws, gl, gw),
            ):
                if not name:
                    continue
                k = (name, *ev)
                d = ent.get(k)
                if d is None:
                    d = ent[k] = {
                        "player": name, "tournament": tname, "year": y,
                        "tour": "wta" if women else "atp",
                        "location": str(r[i["Location"]]).strip() if "Location" in i else "",
                        "surface": str(r[i["Surface"]]).strip() if "Surface" in i else "",
                        "court": str(r[i["Court"]]).strip() if "Court" in i else "",
                        # ATP files carry `Series`, WTA files carry `Tier`. Reading only
                        # `Series` left all 33,096 WTA rows with an empty tier — a silent
                        # half-corpus gap that a per-tour build would have hidden, since
                        # each tour's own file looks complete on its own.
                        "series": (str(r[i["Series"]]).strip() if "Series" in i
                                   else str(r[i["Tier"]]).strip() if "Tier" in i else ""),
                        "matches": 0, "wins": 0, "round_reached": 0,
                        "sets_for": 0.0, "sets_against": 0.0,
                        "games_for": 0.0, "games_against": 0.0,
                        "opp_ranks": [], "beat_better_ranked": 0,
                        "walkovers": 0, "retirements": 0,
                        "_first_rank": own, "_first_round": rnd,
                    }
                # ENTERING rank = the rank in the player's EARLIEST match of the event.
                # Rows are not guaranteed to be in draw order, so this is chosen by round
                # rather than by file position.
                if rnd and rnd < d["_first_round"]:
                    d["_first_round"], d["_first_rank"] = rnd, own
                d["matches"] += 1
                d["wins"] += won
                # Furthest ROUND PLAYED. A win in round r means the player also played
                # r+1 unless r was the final, and that later match is its own row, so
                # taking the max over played rounds is correct without adding 1 here.
                d["round_reached"] = max(d["round_reached"], rnd)
                d["sets_for"] += sf
                d["sets_against"] += sa
                d["games_for"] += gf
                d["games_against"] += ga
                if opp is not None:
                    d["opp_ranks"].append(opp)
                if won and own is not None and opp is not None and opp < own:
                    d["beat_better_ranked"] += 1
                if comment == "Walkover":
                    d["walkovers"] += 1
                elif comment == "Retired":
                    d["retirements"] += 1

        # Attach each event's real draw depth. Keyed on (tournament, year) rather than
        # on a loop variable, because `ev` at this point holds whatever the last match row
        # happened to be — a bug that would have given every entity in the season the draw
        # depth of one arbitrary tournament.
        for k, d in ent.items():
            if k[2] == y:
                d["_max_round"] = maxr[(k[1], k[2])]

    rows = []
    for d in ent.values():
        mr = d.pop("_max_round", 0) or 1
        d.pop("_first_round", None)
        entering = d.pop("_first_rank", None)
        opp = d.pop("opp_ranks")
        sf, sa = d["sets_for"], d["sets_against"]
        gf, ga = d["games_for"], d["games_against"]
        d.update({
            "player_key": player_key(d["player"]),
            "entering_rank": entering,
            "draw_progress": round(d["round_reached"] / mr, 4) if mr else None,
            "max_round_in_event": mr,
            "opp_rank_median": round(statistics.median(opp), 1) if opp else None,
            "set_ratio": round(sf / max(sf + sa, 1), 4),
            "game_ratio": round(gf / max(gf + ga, 1), 4),
            "win_rate": round(d["wins"] / max(d["matches"], 1), 4),
        })
        rows.append(d)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tour", choices=("atp", "wta", "both"), default="both")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = []
    for women, tour in ((False, "atp"), (True, "wta")):
        if args.tour in (tour, "both"):
            rows += build(women)
    if len(rows) < MIN_ROWS:
        print(f"only {len(rows)} entities — did acquire_tennis.py run?")
        return 2

    by_tour = collections.Counter(r["tour"] for r in rows)
    surf = collections.Counter(r["surface"] for r in rows if r["surface"])
    series = collections.Counter(r["series"] for r in rows if r["series"])
    with_rank = sum(1 for r in rows if r["entering_rank"] is not None)
    with_prog = sum(1 for r in rows if r["draw_progress"] is not None)
    multi = sum(1 for r in rows if r["matches"] > 1)
    upsets = sum(r["beat_better_ranked"] for r in rows)

    report = {
        "entity": "player x tournament-edition — mirrors pitch's player x context",
        "supersedes": (
            "the player-SEASON aggregation in probe_tennis_expectation.py, which collapsed "
            "67,081 matches to ~2,500 rows and discarded opponent rank, surface, round and "
            "location — the columns that make tennis worth adding."),
        "rows": len(rows),
        "by_tour": dict(by_tour),
        "distinct_players_raw": len({r["player"] for r in rows}),
        "distinct_players_normalised": len({r["player_key"] for r in rows}),
        "names_collapsed_by_normalisation": (
            len({r["player"] for r in rows}) - len({r["player_key"] for r in rows})),
        "name_format_note": (
            "The source spells the same human several ways -- Zhang K-L. / Zhang K.L., "
            "Wang Xiy. / Wang Xiyu, Lu J.J / Lu Jia-Jing -- and almost every case is a "
            "Chinese or Korean name, so the fragmentation falls on one population. "
            "player_key normalises punctuation and case only. It does NOT expand "
            "abbreviations, because Zhang Ze and Zhang Zh. may be different people and "
            "conflation is worse than fragmentation. The residual is whatever the collapse "
            "count does not close."),
        "distinct_events": len({(r["tournament"], r["year"]) for r in rows}),
        "distinct_locations": len({r["location"] for r in rows if r["location"]}),
        "with_entering_rank": with_rank,
        "pct_with_entering_rank": round(100.0 * with_rank / len(rows), 1),
        "with_draw_progress": with_prog,
        "rows_with_more_than_one_match": multi,
        "wins_over_better_ranked": upsets,
        "surface_mix": dict(surf.most_common()),
        "series_mix": dict(series.most_common(8)),
        "walkovers": sum(r["walkovers"] for r in rows),
        "retirements": sum(r["retirements"] for r in rows),
        "schedule_confound_note": (
            "opp_rank_median is on every row because a season win rate is confounded by "
            "which draws a player entered — the same shape as the team-strength confound "
            "that dominated the pitch age axis, arriving through the opponent instead of "
            "the teammate. Unlike pitch, this corpus records the opponent, so it can be "
            "controlled rather than merely reported."),
        "walkover_note": (
            "Walkovers and retirements are FLAGGED, not dropped. A walkover is a win with "
            "no play: counting it as delivery inflates the winner, dropping it punishes "
            "nobody but loses a real draw advance. Both counts ride on the row so the "
            "consumer decides."),
    }
    OUT.write_text(json.dumps({"report": report, "entities": rows}, indent=2,
                              ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"entities {len(rows)}   {dict(by_tour)}")
    print(f"players {report['distinct_players_raw']} raw -> "
          f"{report['distinct_players_normalised']} normalised "
          f"({report['names_collapsed_by_normalisation']} spellings collapsed)   "
          f"events {report['distinct_events']}   locations {report['distinct_locations']}")
    print(f"entering rank on {with_rank} ({report['pct_with_entering_rank']}%)   "
          f"draw_progress on {with_prog}")
    print(f"rows with >1 match {multi}   wins over better-ranked {upsets}")
    print(f"surfaces {dict(surf.most_common())}")
    print(f"walkovers {report['walkovers']}   retirements {report['retirements']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
