#!/usr/bin/env python3
"""Hoops draft value in VOR, built to MATCH gridiron's construct — closing 7.7b.

Solo personal project, no connection to employer, built with public/free-tier only

7.7b has been the standing caveat on the one cross-sport claim: hoops corr +0.2598 vs
gridiron +0.4236 is a real difference (bootstrap CI on the gap excludes zero), but the two
delivery measures were different CONSTRUCTS — `impact`, a curated composite of on-court
value, against fantasy PPR, which rewards volume and touchdowns and is blind to defence.
Same scoring rule, different yardsticks.

UNIT CONVERSION WOULD NOT HAVE FIXED IT. A correlation is already scale-free; what differs
is what the two numbers MEASURE, not how they are denominated. Rescaling `impact` into
points-above-replacement would have produced a comparison that looked matched and was not.
The only real fix is to compute the SAME KIND of quantity in both sports.

So this rebuilds hoops delivery as fantasy VOR, mirroring build_vor_draft_value.py step
for step:

    composite   PTS + 1.2*REB + 1.5*AST + 3*STL + 3*BLK - TOV   (standard fantasy scoring)
                from pipeline/cache/base_<season>.json, per-100 possessions, so the rate
                is already era-honest within season
    replacement TEAMS x STARTERS = 30 x 5 = 150th best composite that season
    VOR         max(0, composite - replacement), FLOORED — a manager benches a
                below-replacement player rather than starting him
    pick value  SUM of VOR over the first WINDOW_YEARS seasons after the draft
    expectation 1 - log1p(overall) / log1p(MAX_PICK), the identical transform gridiron
                uses, with MAX_PICK set to the NBA's 60 rather than the NFL's 262

DENOMINATOR IS draft_history.json (7,383 players), NOT pedigree.json. pedigree.json was
built from charted players, so iterating it would reproduce the survivor bias for the
third time in this thread. A drafted player who never records a season contributes 0.0.

Every methodological choice here is inherited rather than re-decided, because the point is
comparability. Where the sports genuinely differ — 60 picks vs 262, 5 starters vs a
position-split lineup — the difference is in a declared constant, not in the method.

    python pipeline/build_hoops_vor_draft_value.py
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import unicodedata
from math import log1p
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOOPS = Path("C:/Users/jcdav/vector-hoops")
CACHE = HOOPS / "pipeline" / "cache"
DRAFT = CACHE / "draft_history.json"
VECTORS = HOOPS / "assets" / "vectors.json"
OUT = ROOT / "data" / "hoops_vor_draft_value.json"

TEAMS, STARTERS = 30, 5
REPLACEMENT_RANK = TEAMS * STARTERS  # 150
MAX_PICK = 60.0  # NBA draft depth (NFL is 262)
WINDOW_YEARS = 5  # matches gridiron's rookie-deal window
MIN_CELL = 8
BUCKETS = [
    (1, 14, "lottery"),
    (15, 30, "late-1st"),
    (31, 45, "early-2nd"),
    (46, 60, "late-2nd"),
]

W = {"PTS": 1.0, "REB": 1.2, "AST": 1.5, "STL": 3.0, "BLK": 3.0, "TOV": -1.0}


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'\u2019-]", "", s.lower())
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s.strip())
    return re.sub(r"\s+", " ", s)


def bucket(overall) -> str | None:
    if overall is None:
        return None
    for lo, hi, name in BUCKETS:
        if lo <= overall <= hi:
            return name
    return None  # picks beyond 60 are pre-1989 drafts; out of scope


def composite(row: dict) -> float:
    reb = float(row.get("OREB") or 0) + float(row.get("DREB") or 0)
    return (
        W["PTS"] * float(row.get("PTS") or 0)
        + W["REB"] * reb
        + W["AST"] * float(row.get("AST") or 0)
        + W["STL"] * float(row.get("STL") or 0)
        + W["BLK"] * float(row.get("BLK") or 0)
        + W["TOV"] * float(row.get("TOV") or 0)
    )


def season_start(season: str) -> int:
    return int(str(season).split("-")[0])


def eligible_pairs(vec: dict) -> set[tuple[str, str]]:
    """(season, normalised name) that clear vectors.json's minutes gate."""
    return {(str(p["season"]), norm_name(p["name"])) for p in vec["players"]}


def vor_series(seasons: list[str], eligible: set[tuple[str, str]]):
    """name -> [(year, floored VOR)]. THE ONE IMPLEMENTATION.

    build_direction_axis.py previously carried its own copy of this loop, so applying the
    eligibility filter here left that copy still reading the raw cache — the direction
    axis kept reporting Damion James rising 0.00 -> 38.16 on one eligible season after the
    bug was supposedly fixed. Two copies of a rule means fixing it once fixes it once.
    """
    out: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
    seen = 0
    for season in seasons:
        f = CACHE / f"base_{season}.json"
        if not f.exists():
            continue
        seen += 1
        rows = json.loads(f.read_text(encoding="utf-8"))
        comps = {norm_name(k): composite(v) for k, v in rows.items() if (season, norm_name(k)) in eligible}
        if not comps:
            continue
        ranked = sorted(comps.values(), reverse=True)
        base = ranked[min(REPLACEMENT_RANK, len(ranked)) - 1]
        y = season_start(season)
        for n, c in comps.items():
            out[n].append((y, max(0.0, c - base)))
    return out, seen


def merged_names(series: dict, draft: dict, acquit: bool = True) -> set[str]:
    """Names that provably belong to more than one person. See check_merged_careers.py.

    Operator report 2026-08-03: Jaren Jackson and Jaren Jackson Jr. are one key. The cause
    is upstream — vector-hoops' vectors.json carries no suffixes and its `id` is a row
    index, not a player id — so this repo cannot separate them, only refuse to treat them
    as one career.

    Two definitive tests, no thresholds:
      * a season strictly BEFORE the name's earliest draft year. Arithmetic; nobody plays
        before they are drafted. Jaren Jackson has five.
      * more than one draft entry for the name. Two drafted people share it, and which
        seasons belong to whom is unknowable from this source.

    A career GAP is deliberately NOT used: Anthony Parker played in Europe 2000-2005 and
    107 eligible careers carry one. Suggestive is not definitive.

    THE SECOND TEST IS NOT DEFINITIVE and the docstring above used to claim it was. A player
    who is drafted and does not sign can re-enter, so ONE person can hold two rows:

        arvydas sabonis   1985 #77 (Atlanta, voided as underage) + 1986 #24 (Portland)
                          Wikidata: one qid, Q297750, born 1964

    ">1 draft entry" answers "does this name have more than one draft row", which is a
    different question from "are there two people". The year gap is NOT a usable
    discriminator — three of the five 1-year-span names are genuine collisions.

    ACQUITTAL IS APPLIED HERE, NOT BY CALLERS. Every call site would otherwise have to
    remember to subtract `acquitted_names()`, the same "two copies of one rule" shape as the
    five private norm_name() copies. Pass acquit=False only for the raw arithmetic set;
    probe_hoops_name_collisions.py is the sole legitimate caller, since it computes the
    acquittal and would otherwise ratchet against its own previous output.
    """
    picks_by_norm: dict[str, list] = collections.defaultdict(list)
    for raw, picks in draft.items():
        picks_by_norm[norm_name(raw)].extend(picks)
    out: set[str] = set()
    overturnable: set[str] = set()
    for name, rows in series.items():
        picks = picks_by_norm.get(name) or []
        if len(picks) > 1:
            out.add(name)
            overturnable.add(name)
            continue
        yrs = [p["year"] for p in picks if p.get("year")]
        if yrs and any(y < min(yrs) for y, _ in rows):
            out.add(name)
    return out - (acquitted_names() & overturnable) if acquit else out


COLLISIONS = Path(__file__).resolve().parent.parent / "data" / "hoops_name_collisions.json"


def acquitted_names() -> set[str]:
    """Names `merged_names()` flags that Wikidata shows to be ONE person.

    Evidence, not assumption: the suffix sweep in probe_hoops_name_collisions.py re-queries
    each flagged name across ["", " Jr.", " Sr.", " II", " III", " IV"] — a bare-label query
    cannot see Glen Rice Jr. (Q4811246, b.1991) when the corpus display name is "Glen Rice"
    — and a name is acquitted only when that sweep, having had every chance to find a second
    person, still returns exactly one age-plausible QID.

    RETURNS EMPTY WHEN THE ARTIFACT IS ABSENT, which is the conservative direction: callers
    subtract this from an exclusion set, so an empty set excludes MORE, never less. A missing
    probe must not be able to quietly re-admit a merged career.
    """
    if not COLLISIONS.exists():
        return set()
    doc = json.loads(COLLISIONS.read_text(encoding="utf-8"))
    return set((doc.get("SUFFIX_SWEEP") or {}).get("acquitted") or {})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not DRAFT.exists():
        print(f"missing {DRAFT}")
        return 2

    vec = json.loads(VECTORS.read_text(encoding="utf-8"))
    seasons = sorted({str(p["season"]) for p in vec["players"]}, key=season_start)
    first_year, last_year = season_start(seasons[0]), season_start(seasons[-1])

    # ELIGIBILITY FILTER, and it is not optional. base_<season>.json is the RAW cache;
    # vectors.json applies a schedule-aware minutes gate ("15% of season GP (clamp 10-15)
    # + 6% of 48mpg schedule total minutes, floor 450") and 572 raw players in 2023-24
    # become 484 eligible. Because the composite is PER-100 POSSESSIONS, a player with a
    # handful of minutes posts an enormous rate off nothing — the direction axis surfaced
    # Damion James rising 0.00 -> 38.16 on a career with ONE eligible season. Reading the
    # raw cache silently bypassed a gate the rest of the estate applies everywhere.
    eligible = eligible_pairs(vec)

    # ---- composite + replacement per season ----------------------------------
    vor_of, seasons_seen = vor_series(seasons, eligible)

    # ---- denominator: every drafted player -----------------------------------
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))["players"]
    # Minus the names Wikidata clears as one person. Dropping all 45 discarded 21 real
    # careers — Patrick Ewing, Glen Rice, Mike Dunleavy among them — on the strength of
    # their SONS' draft rows appearing under the same normalised key.
    merged = merged_names(vor_of, draft)
    max_draft_year = last_year - WINDOW_YEARS + 1

    totals: dict[str, list[float]] = collections.defaultdict(list)
    never = collections.Counter()
    rows_out = []
    for name, picks in draft.items():
        if not picks:
            continue
        p = picks[0]
        yr, overall = p.get("year"), p.get("overall")
        b = bucket(overall)
        if yr is None or not b:
            continue
        if yr < first_year or yr > max_draft_year:
            continue
        n = norm_name(name)
        if n in merged:
            # Two people under one key. A draft slot cannot be attributed and a career
            # total is a sum over two careers, so the row is dropped rather than scored.
            continue
        window = [v for (s, v) in vor_of.get(n, ()) if yr <= s < yr + WINDOW_YEARS]
        total = sum(window)
        if not window:
            never[b] += 1
        totals[b].append(total)
        rows_out.append(
            {
                "name": n,
                "overall": overall,
                "bucket": b,
                "year": yr,
                "expect_log": round(max(0.0, 1 - log1p(overall) / log1p(MAX_PICK)), 4),
                "vor_total": round(total, 2),
                "played": bool(window),
                # FULL-career eligible season count, not the 5-year window — see
                # I6 in check_draft_value_invariants.py.
                "seasons_total": len(vor_of.get(n, ())),
            }
        )

    cells = []
    for _lo, _hi, b in BUCKETS:
        ds = totals.get(b) or []
        if len(ds) < MIN_CELL:
            cells.append({"bucket": b, "drafted": len(ds), "thin": True})
            continue
        cells.append(
            {
                "bucket": b,
                "drafted": len(ds),
                "never_played_pct": round(100.0 * never.get(b, 0) / len(ds), 1),
                "ev_vor": round(statistics.mean(ds), 2),
                "thin": False,
            }
        )

    xs = [r["expect_log"] for r in rows_out]
    ys = [r["vor_total"] for r in rows_out]
    corr = statistics.correlation(xs, ys) if len(set(xs)) > 1 else 0.0

    report = {
        "sport": "hoops",
        "construct": "MATCHED to gridiron: fantasy VOR, floored, summed over a 5-year window",
        "scoring": W,
        "league": {
            "teams": TEAMS,
            "starters": STARTERS,
            "replacement_rank": REPLACEMENT_RANK,
        },
        "max_pick": MAX_PICK,
        "seasons_read": seasons_seen,
        "draft_year_window": [first_year, max_draft_year],
        "drafted_scored": len(rows_out),
        "merged_names_excluded": len(merged),
        "merged_note": (
            "Names that provably cover more than one person are dropped, not "
            "scored. See check_merged_careers.py; operator-reported 2026-08-03."
        ),
        "corr_expectation_vor": round(corr, 4),
        "cells": cells,
        "note_7_7b": (
            "This is the hoops half of the matched comparison. The gridiron half "
            "is data/vor_draft_value.json. Correlations computed on MATCHED "
            "constructs are the ones that may be compared; the earlier "
            "+0.2598 vs +0.4236 pair could not be."
        ),
    }
    OUT.write_text(
        json.dumps({"report": report, "players": rows_out}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"seasons read: {seasons_seen}   draft years {first_year}-{max_draft_year}")
    print(f"drafted players scored: {len(rows_out)}   replacement = {REPLACEMENT_RANK}th best\n")
    print(f"{'bucket':10} {'drafted':>8} {'never%':>7} {'EV VOR':>8}")
    for c in cells:
        if c.get("thin"):
            print(f"{c['bucket']:10} {c['drafted']:>8} {'thin':>7} {'-':>8}")
        else:
            print(f"{c['bucket']:10} {c['drafted']:>8} {c['never_played_pct']:>6.1f}% {c['ev_vor']:>8.2f}")
    print(f"\ncorr(expectation, VOR total) = {corr:+.4f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
