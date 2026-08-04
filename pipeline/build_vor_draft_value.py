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


_SUFFIX_RE = re.compile(r"\s+(jr|sr|ii|iii|iv|v)$")

# Names whose suffix must NOT be stripped, because stripping would collide them with a
# DIFFERENT name that also exists in the sources. Populated by configure_norm().
_NO_STRIP: set[str] = set()
_CONFIGURED = False


def _base(name: str) -> str:
    s = unicodedata.normalize("NFD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'\u2019-]", "", s.lower())
    return re.sub(r"\s+", " ", s.strip())


def configure_norm(*sources) -> int:
    """Decide per name whether dropping the suffix is safe. Call before norm_name().

    Takes one iterable of names PER SOURCE, not a pooled set — see the comment below.

    STRIPPING IS RIGHT SOMETIMES AND WRONG OTHERS, which is why neither blanket policy
    works. Measured on the gridiron sources:

        strip always   vec 2,704  draft 12,527  joined 1,900   but merges 3 real pairs
        keep always    vec 2,707  draft 12,552  joined 1,879   merges nothing, loses 21

    The 3 merges are father/son: `marvin harrison` + `marvin harrison jr` \u2014 the Hall of
    Famer and the Cardinals receiver \u2014 plus `oronde gadsden` + `oronde gadsden ii` and
    `cedrick wilson` + `cedrick wilson jr`. The 21 losses are ONE person spelled two ways
    across files: `chris godwin jr` in the vector set against `chris godwin` in the draft
    CSV, where stripping is exactly the repair.

    The discriminator is whether the STRIPPED form already exists as its own name in the
    sources. If both `marvin harrison` and `marvin harrison jr` appear, the source itself
    distinguished two people and stripping destroys that. If only `chris godwin jr`
    appears, stripping collides with nothing and closes a real join.

    Operator-reported 2026-08-03. This repairs the reported cause; merged_names() handles
    the residue the sources never distinguished in the first place.
    """
    global _CONFIGURED
    _NO_STRIP.clear()
    _CONFIGURED = True
    # WITHIN each source, never across their union. This is the whole subtlety and the
    # first version got it wrong: pooling the vector set and the draft CSV made
    # `chris godwin` (draft) collide with `chris godwin jr` (vectors) and protected both,
    # which is precisely backwards — that pair is ONE person spelled two ways, and the
    # protection cost the join it was supposed to save (1,880 against 1,900).
    #
    # A collision only means "two people" when ONE source lists both forms, because that
    # source had the chance to conflate them and chose not to. `marvin harrison` and
    # `marvin harrison jr` both appear in vectors.json; `chris godwin` never appears there
    # alongside `chris godwin jr`.
    for group in sources:
        bare = {_base(n) for n in group}
        for n in group:
            b = _base(n)
            stripped = _SUFFIX_RE.sub("", b).strip()
            if stripped != b and stripped in bare:
                _NO_STRIP.add(b)
                _NO_STRIP.add(stripped)
    return len(_NO_STRIP)


def _ensure_configured() -> None:
    """Configure from the canonical sources, once, on first use.

    SELF-CONFIGURING BECAUSE CALLERS CANNOT BE TRUSTED TO AGREE, and that is not a slight —
    it is what happened. Four files import norm_name(): this builder, the survivorship
    probe, the direction axis and the merged-career checker. Each had to call
    configure_norm() with the same two sources in the same order, and they did not: the
    probe built one dict before configuring, so `michael pittman jr` was protected in one
    file and stripped in another, and I4 caught the two artifacts disagreeing by exactly
    one WR.

    A shared policy that every importer must remember to install is a policy that will
    eventually differ between importers. Reading the canonical sources here makes that
    impossible — the explicit configure_norm() remains for tests and for callers with a
    different source pair.
    """
    if _CONFIGURED:
        return
    names_vec: list[str] = []
    if GRID_VEC.exists():
        names_vec = [p["name"] for p in
                     json.loads(GRID_VEC.read_text(encoding="utf-8"))["players"]]
    names_draft: list[str] = []
    if DRAFT_CSV.exists():
        with DRAFT_CSV.open(encoding="utf-8", errors="replace", newline="") as fh:
            names_draft = [(r.get("pfr_player_name") or "").strip()
                           for r in csv.DictReader(fh)]
    configure_norm(names_vec, [n for n in names_draft if n])


def norm_name(name: str) -> str:
    _ensure_configured()
    b = _base(name)
    return b if b in _NO_STRIP else _SUFFIX_RE.sub("", b).strip()


def bucket(pick) -> str | None:
    if pick is None:
        return None
    for lo, hi, name in BUCKETS:
        if lo <= pick <= hi:
            return name
    return "R4-7"


def merged_names(seasons_of: dict, draft_csv) -> set[str]:
    """Names that provably cover more than one player. Mirrors the hoops version.

    Operator report 2026-08-03 was about hoops (Jaren Jackson Sr./Jr.), but the same defect
    is LARGER here: 318 gridiron draft names carry more than one distinct draft year, and
    ten names record a season before they were drafted.

    It hit a published finding directly. `antonio brown` was the number-one D0 example in
    the gridiron direction axis — pick 195, +6.93, quoted repeatedly as the headline riser
    — and the name holds seasons from 2003 and 2005 against a 2010 draft. The Steelers
    Antonio Brown did not play in 2003. "First half 1.86 -> second half 8.79" was partly
    one player becoming another.

    Two definitive tests, no thresholds:
      * a season strictly BEFORE the name's earliest draft year
      * more than one distinct draft year for the name

    A career gap is NOT used, here or in hoops: injury and overseas years produce them.
    """
    # REFUSE IF THE NORMALISER WAS NEVER CONFIGURED. This is module state and module state
    # is forgettable: check_merged_careers.py called merged_names() without it, so inside
    # the checker `_NO_STRIP` was empty, `cedrick wilson` and `cedrick wilson jr` collapsed
    # to one name with two draft years, and the guard reported contamination the builder
    # had correctly already handled. A disagreement between a checker and the thing it
    # checks is worse than either being wrong alone.
    _ensure_configured()

    dy: dict[str, set[int]] = collections.defaultdict(set)
    with draft_csv.open(encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("pfr_player_name") or "").strip()
            if not raw:
                continue
            try:
                dy[norm_name(raw)].add(int(float(row["season"])))
            except (KeyError, TypeError, ValueError):
                continue
    out: set[str] = set()
    for name, rows in seasons_of.items():
        yrs = dy.get(name)
        if not yrs:
            continue
        if len(yrs) > 1:
            out.add(name)
        elif any(y < min(yrs) for y, _ in rows):
            out.add(name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    for p in (GRID_VEC, SURV, DRAFT_CSV):
        if not p.exists():
            print(f"missing {p}")
            return 2

    vec = json.loads(GRID_VEC.read_text(encoding="utf-8"))["players"]

    # Configure the normaliser BEFORE any name is normalised. Per source, never pooled.
    with DRAFT_CSV.open(encoding="utf-8", errors="replace", newline="") as _fh:
        _draft_names = [(r.get("pfr_player_name") or "").strip()
                        for r in csv.DictReader(_fh)]
    n_protected = configure_norm([p["name"] for p in vec],
                                 [n for n in _draft_names if n])

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
    # From the FULL vector set, not from `seasons_of`. `seasons_of` only holds seasons that
    # had a replacement baseline for their (season, pos), so a pre-draft season outside
    # that filter is invisible here and visible to check_merged_careers.py — the detector
    # and the exclusion have to see the same data. Same mismatch already fixed in
    # build_direction_axis.py; `darrell jackson` and `cedrick wilson` sat in this one.
    _full: dict[str, list] = collections.defaultdict(list)
    for _p in vec:
        _v = (_p.get("ppg") or {}).get("ppr")
        if _v is not None:
            _full[norm_name(_p["name"])].append((int(_p["season"]), float(_v)))
    merged = merged_names(_full, DRAFT_CSV)

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
            if n in merged:
                # Two players under one key. A draft slot cannot be attributed and a
                # career total sums two careers, so the row is dropped rather than scored.
                continue
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
                "vor_total": round(sum(window), 2), "played": bool(window),
                # FULL-career qualifying season count, not the 5-year window. It exists so
                # check_draft_value_invariants.py I6 can assert that the direction axis
                # derived the same series this table did. The hoops half of that pair
                # silently used a different eligibility rule for a whole build.
                "seasons_total": len(seasons_of.get(n, ()))})

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
        "suffix_protected_names": n_protected,
        "suffix_note": ("Conflict-aware suffix stripping: a suffix is dropped unless the "
                        "stripped form already exists as its own name WITHIN THE SAME "
                        "source. Keeps Marvin Harrison and Marvin Harrison Jr. apart while "
                        "still joining Chris Godwin Jr. to Chris Godwin across files. "
                        "1,903 joins against 1,900 strip-always and 1,879 keep-always."),
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
