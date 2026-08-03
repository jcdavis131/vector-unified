#!/usr/bin/env python3
"""Draft value in VALUE-OVER-REPLACEMENT, so positions can finally be compared.

Solo personal project, no connection to employer, built with public/free-tier only

build_draft_value_curve.py produced EV in delivery PERCENTILE units, and percentile is
computed within (season, position). "TE R1 = 70.4" means 70th among tight ends; "QB R1 =
40.2" means 40th among quarterbacks. Those are different pools, so the table could be read
down a column and never across one — and read across, it says "draft a TE before a QB",
which the number does not support.

VOR is the standard fix and the right one here: subtract each position's REPLACEMENT LEVEL,
the points a manager could have had for free, and what remains is in one currency.

    VOR(player-season) = PPR ppg - replacement PPR ppg for that (season, position)
    replacement rank   = starters per team x teams in the league
    pick value         = SUM of VOR over the first WINDOW_YEARS seasons after the draft,
                         for EVERY drafted player, washouts included at 0

WHY NOT P(survive) x E[VOR | survive], WHICH IS WHAT THE PERCENTILE VERSION DID. That
composite is only valid when delivery cannot go negative. VOR can, and does: replacement
is the STARTER line, so 78.2% of all charted player-seasons sit below it (QB 71.2%, TE
85.1%). With a negative conditional value, multiplying by a SMALLER survival probability
produces a LARGER (less negative) EV — the first run of this file ranked QB R4-7 second
best on the board at -0.30 purely because only 9% of those picks survive. A pick that
almost never works out scored better than one that usually works out slightly below
starter grade. That is an artefact of the formula, not a fact about football.

Summing over a fixed post-draft window removes the split entirely. Washouts contribute
their real (near-zero) total rather than being conditioned away, so survival is priced in
by construction instead of multiplied in. It also fixes a second error: career MEAN VOR
penalises long careers with decline years, and a team drafting wants TOTAL value
delivered, not average season quality.

LEAGUE SETTINGS ARE A CHOICE AND ARE DECLARED, not buried. A 12-team league starting
1QB/2RB/3WR/1TE puts replacement at QB12, RB24, WR36, TE12. Change the settings and the
cross-position ordering changes with them — that is a property of fantasy football, not a
defect here, and it is why the constants sit at the top of the file rather than inside a
function. Superflex leagues in particular would move QB sharply.

MIN_GAMES GUARDS THE BASELINE. Replacement level is read off a ranked list, so a player
with two starts and a fluke week would otherwise set the bar. Only seasons with >= MIN_GAMES
enter the pool that DEFINES replacement; every season is still scored against it.

The survival half is unchanged and still comes from the denominator (draft_picks.csv), and
the same draft-year window is applied to both halves — the pool mismatch that produced "25
drafted, 28 survivors" in the percentile version is not repeated here.

    python pipeline/build_vor_draft_value.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics
import unicodedata
from math import log1p
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRID_VEC = Path("C:/Users/jcdav/vector-gridiron/assets/vectors.json")
SURV = ROOT / "data" / "qb_survivorship_probe.json"
DRAFT_CSV = Path("C:/Users/jcdav/vector-gridiron/pipeline/cache/draft_picks.csv")
OUT = ROOT / "data" / "vor_draft_value.json"

# DECLARED LEAGUE SETTINGS — 12 teams, 1QB / 2RB / 3WR / 1TE.
TEAMS = 12
STARTERS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
REPLACEMENT_RANK = {pos: TEAMS * n for pos, n in STARTERS.items()}

MIN_GAMES = 8
WINDOW_YEARS = 5   # rookie deal + fifth-year option: what the pick actually buys
MIN_CELL = 8
BUCKETS = [(1, 32, "R1"), (33, 64, "R2"), (65, 105, "R3"), (106, 262, "R4-7")]
POSITIONS = ("QB", "RB", "WR", "TE")


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'\u2019-]", "", s.lower())
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s.strip())
    return re.sub(r"\s+", " ", s)


def bucket(pick) -> str | None:
    if pick is None:
        return None
    for lo, hi, name in BUCKETS:
        if lo <= pick <= hi:
            return name
    return "R4-7"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    for p in (GRID_VEC, SURV, DRAFT_CSV):
        if not p.exists():
            print(f"missing {p}")
            return 2

    vec = json.loads(GRID_VEC.read_text(encoding="utf-8"))["players"]
    surv = json.loads(SURV.read_text(encoding="utf-8"))
    lo_year, hi_year = surv["report"]["draft_year_window"]

    # ---- replacement level per (season, position) ----------------------------
    pool: dict[tuple, list[float]] = collections.defaultdict(list)
    for p in vec:
        ppr = (p.get("ppg") or {}).get("ppr")
        if ppr is None or (p.get("games") or 0) < MIN_GAMES:
            continue
        if p.get("pos") in REPLACEMENT_RANK:
            pool[(p["season"], p["pos"])].append(float(ppr))
    replacement: dict[tuple, float] = {}
    for key, vals in pool.items():
        vals.sort(reverse=True)
        rank = REPLACEMENT_RANK[key[1]]
        # If a season has fewer qualified players than the replacement rank, the last
        # qualified player IS the baseline; reporting a deeper rank we do not have would
        # invent a number.
        replacement[key] = vals[min(rank, len(vals)) - 1]

    # ---- per player-season VOR, indexed so a post-draft window can be summed --
    seasons_of: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
    for p in vec:
        ppr = (p.get("ppg") or {}).get("ppr")
        pos = p.get("pos")
        if ppr is None or pos not in REPLACEMENT_RANK:
            continue
        base = replacement.get((p["season"], pos))
        if base is None:
            continue
        # FLOORED AT ZERO, and this is the modelling fix that makes the table coherent.
        # A manager never fields a below-replacement player — he benches him and starts a
        # replacement, so a bad season is worth 0, not a negative. Without the floor a
        # pick that produced NOTHING outranked one that produced below-starter play:
        # QB R4-7 scored -2.65 against QB R1's -7.87, which says a wasted 7th-rounder
        # beats a franchise quarterback. That is an artefact of letting VOR go negative.
        seasons_of[norm_name(p["name"])].append(
            (int(p["season"]), max(0.0, float(ppr) - base)))

    # ---- EVERY drafted player, including the ones who never appear ------------
    # This is the correction. Iterating the DRAFT (the denominator) instead of the
    # survivor table means a washout contributes its real total — which is 0.0 when he
    # never records a qualifying season — rather than being conditioned out and then
    # multiplied back in with the wrong sign.
    # gridiron_pedigree.json is NOT the denominator — it was built FROM the vector set,
    # so it contains only players who recorded a season. Iterating it reproduced the exact
    # survivor bias this rewrite exists to remove, and the symptom was `never played` at
    # 0.0% in almost every cell when a quarter of late-round picks never play at all.
    # draft_picks.csv is the denominator. Read it directly.
    totals: dict[tuple, list[float]] = collections.defaultdict(list)
    player_rows: list[dict] = []
    zero_seasons: collections.Counter = collections.Counter()
    seen: set[str] = set()
    with DRAFT_CSV.open(encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            pos = (row.get("position") or "").strip().upper()
            if pos not in REPLACEMENT_RANK:
                continue
            raw = (row.get("pfr_player_name") or "").strip()
            if not raw:
                continue
            try:
                yr = int(float(row["season"]))
                pick = int(float(row["pick"]))
            except (KeyError, TypeError, ValueError):
                continue
            if not (lo_year <= yr <= hi_year):
                continue
            n = norm_name(raw)
            if n in seen:
                continue
            seen.add(n)
            b = bucket(pick)
            if not b:
                continue
            window = [v for (s, v) in seasons_of.get(n, ())
                      if yr <= s < yr + WINDOW_YEARS]
            if not window:
                zero_seasons[(pos, b)] += 1
            totals[(pos, b)].append(sum(window))
            # per-player rows so the MATCHED cross-sport correlation (7.7b) can be
            # computed on the same construct hoops now uses, instead of comparing
            # `impact` percentile against fantasy PPR percentile
            player_rows.append({
                "name": n, "pos": pos, "overall": pick, "bucket": b, "year": yr,
                "expect_log": round(max(0.0, 1 - log1p(pick) / log1p(262.0)), 4),
                "vor_total": round(sum(window), 2), "played": bool(window)})

    per = surv["report"]["per_position"]
    rows = []
    for pos in POSITIONS:
        v = per.get(pos)
        if not v:
            continue
        for _lo, _hi, b in BUCKETS:
            cell = v["by_bucket"].get(b)
            ds = totals.get((pos, b)) or []
            if len(ds) < MIN_CELL:
                rows.append({"pos": pos, "bucket": b,
                             "drafted": len(ds),
                             "survival_pct": cell["rate"] if cell else None,
                             "n_surv": len(ds), "cond_vor": None, "ev_vor": None,
                             "thin": True})
                continue
            cond = statistics.mean(ds)
            zeros = zero_seasons.get((pos, b), 0)
            rows.append({"pos": pos, "bucket": b, "drafted": len(ds),
                         "survival_pct": cell["rate"] if cell else None,
                         "n_surv": len(ds),
                         "never_played_pct": round(100.0 * zeros / len(ds), 1),
                         "cond_vor": round(cond, 2),
                         "ev_vor": round(cond, 2),
                         "thin": False})

    scored = [r for r in rows if not r["thin"]]
    by_ev = sorted(scored, key=lambda x: -x["ev_vor"])

    report = {
        "units": "PPR points per game above replacement",
        "league": {"teams": TEAMS, "starters": STARTERS,
                   "replacement_rank": REPLACEMENT_RANK},
        "league_note": ("Cross-position ordering DEPENDS on these settings. Superflex or a "
                        "2TE lineup moves QB and TE respectively. Declared here rather than "
                        "hidden in a function so the dependency is visible."),
        "min_games_for_baseline": MIN_GAMES,
        "window_years": WINDOW_YEARS,
        "formula": ("mean over ALL drafted players in the bucket of SUM(VOR) across their "
                    "first WINDOW_YEARS seasons. Washouts contribute 0.0. No conditioning, "
                    "no multiplication by survival — survival is priced in by including "
                    "the players who never played."),
        "draft_year_window": [lo_year, hi_year],
        "corr_expectation_vor": round(
            statistics.correlation([r["expect_log"] for r in player_rows],
                                   [r["vor_total"] for r in player_rows]), 4)
        if len(player_rows) > 2 else None,
        "cells": rows,
        "ranking_ev_vor": [f"{r['pos']} {r['bucket']}" for r in by_ev],
        "now_cross_position_comparable": (
            "YES, in these league settings. VOR subtracts each position's replacement "
            "level, so a point is a point. The percentile version was NOT comparable "
            "across positions and said so."),
        "still_true": ("Fantasy PPR remains a FANTASY delivery measure — blind to blocking, "
                       "route running and all defence. VOR fixes the units, not the sport."),
    }
    OUT.write_text(json.dumps({**report, "players": player_rows}, indent=2,
                              ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"VOR = PPR ppg above replacement | {TEAMS}-team, "
          f"{'/'.join(f'{n}{p}' for p, n in STARTERS.items())} "
          f"-> replacement {REPLACEMENT_RANK}\n")
    print(f"{'pos':4} {'bucket':7} {'drafted':>8} {'surv%':>7} {'never%':>7} {'EV VOR':>8}")
    for r in rows:
        if r["thin"]:
            print(f"{r['pos']:4} {r['bucket']:7} {r['drafted']:>8} "
                  f"{(str(r['survival_pct']) + '%') if r['survival_pct'] is not None else '-':>7} "
                  f"{'thin':>9} {'-':>8}  {r['n_surv']}")
        else:
            print(f"{r['pos']:4} {r['bucket']:7} {r['drafted']:>8} "
                  f"{(r['survival_pct'] if r['survival_pct'] is not None else 0):>6.1f}% "
                  f"{r['never_played_pct']:>6.1f}% {r['ev_vor']:>8.2f}")
    print("\nranked by EV in VOR (now comparable ACROSS positions):")
    for r in by_ev[:8]:
        print(f"  {r['pos']:3} {r['bucket']:5} {r['ev_vor']:>7.2f}")
    print(f"\n{report['league_note']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
