#!/usr/bin/env python3
"""Assign T0/T1 — performance vs expectation — for hoops, the one sport with both inputs.

Solo personal project, no connection to employer, built with public/free-tier only

T0-T3 have been DECLARED since the role/trajectory split and never assigned, because the
cross-sport archetypes were fit on role features and pedigree never entered the anchor.
Both inputs exist for hoops and join at 100%:

    expectation  vector-hoops/assets/pedigree.json    2,415 players, `expect_slot`
                 stated CBA-rookie-scale curve: #1 pick 1.0, 2nd round 0.10, undrafted 0.06
    delivery     vector-hoops/assets/skills.json      grade[11] = `impact`, percentile
                 0-99 WITHIN SEASON POOL, order-aligned with vectors.json players[]

WHY `impact` AND NOT SALARY. career_surplus.json already measures surplus against salary
(0.6*z(pred_MPG)+0.4*z(pred_GP) - SALARY_LOG z). Salary is the wrong denominator for THIS
axis: rookie-scale contracts are a direct function of draft slot, so regressing salary on
pedigree partly regresses draft slot on itself. `impact` is on-court and contract-blind,
so the residual means what it says.

WHY PERCENTILE-WITHIN-SEASON. It removes era entirely. A 1997 impact grade and a 2025 one
are both "how good, among peers that year", so a 30-season career pool can be compared
without an era term.

GRAIN IS THE CAREER, not the season — that is what makes this the trajectory axis rather
than a second role axis. Each player gets one expectation and one delivery number.

THE JOIN IS SPORT-SCOPED, and this is not a stylistic choice. Joining pedigree by
normalised name across the whole unified set matches 16 gridiron and 1 pitch athlete and
ALL 17 ARE FALSE POSITIVES — NFL Matt Ryan onto an NBA Matt Ryan, NFL James Robinson onto
an NBA James Robinson drafted 1993 by POR. Several carry draft_year=None, so the wrong
record reads as a benign "undrafted" rather than an obvious error.

ASSIGNMENT. Delivery is regressed on expectation across the career pool; the residual is
what "over/under-delivered relative to billing" means. Labels are the tails, not a median
split, because the middle of this distribution is genuinely "delivered about as billed"
and forcing it into T0/T1 would manufacture a finding:

    T0  high expectation, residual in the bottom TAIL_PCT      under-delivered
    T1  low  expectation, residual in the top    TAIL_PCT      over-delivered
    --  everyone else is UNLABELLED on this axis, by design

    python pipeline/build_trajectory_axis.py
    python pipeline/build_trajectory_axis.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOOPS = Path("C:/Users/jcdav/vector-hoops")
PEDIGREE = HOOPS / "assets" / "pedigree.json"
SKILLS = HOOPS / "assets" / "skills.json"
VECTORS = HOOPS / "assets" / "vectors.json"
UNIFIED = ROOT / "assets" / "unified.json"
GRID_PED = ROOT / "data" / "gridiron_pedigree.json"
GRID_VEC = Path("C:/Users/jcdav/vector-gridiron/assets/vectors.json")
OUT = ROOT / "data" / "trajectory_axis.json"

IMPACT = "impact"
MIN_SEASONS = 4      # same floor vector-hoops uses for its career classes
TAIL_PCT = 20.0      # top/bottom fifth of the residual distribution
HIGH_EXPECT_Q = 60.0  # "high billing" = expect_slot above this percentile
LOW_EXPECT_Q = 40.0


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'\u2019-]", "", s.lower())
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s.strip())
    return re.sub(r"\s+", " ", s)


def pct_rank(sorted_vals: list[float], v: float) -> float:
    below = sum(1 for x in sorted_vals if x < v)
    return 100.0 * below / max(len(sorted_vals) - 1, 1)


def run_gridiron(args) -> int:
    """Same scoring rule as hoops, two structural differences stated rather than hidden.

    DELIVERY IS POSITION-NORMALISED. hoops' `impact` is a percentile within season and
    needs nothing more — a guard and a centre are graded on one scale. Fantasy points are
    not: a QB's raw PPR dwarfs a TE's, so an unnormalised delivery column would rank every
    quarterback above every tight end and call it over-delivery. Percentile is taken
    within (season, position).

    SCOPE IS OFFENSIVE SKILL POSITIONS. All 18 gridiron features are pass/rush/receiving,
    so linemen and defenders are absent from the vector set entirely. A gridiron T0/T1
    label describes QB/RB/WR/TE, not a roster.
    """
    if not GRID_PED.exists():
        print(f"missing {GRID_PED} — run export_gridiron_pedigree.py first")
        return 2
    ped = json.loads(GRID_PED.read_text(encoding="utf-8"))["players"]
    vec = json.loads(GRID_VEC.read_text(encoding="utf-8"))["players"]

    pools: dict[tuple, list[float]] = {}
    for p in vec:
        ppr = (p.get("ppg") or {}).get("ppr")
        if ppr is None:
            continue
        pools.setdefault((p["season"], p.get("pos")), []).append(float(ppr))
    for v in pools.values():
        v.sort()

    per_player: dict[str, list[float]] = {}
    for p in vec:
        ppr = (p.get("ppg") or {}).get("ppr")
        if ppr is None:
            continue
        pool = pools.get((p["season"], p.get("pos")))
        if not pool or len(pool) < 10:
            continue
        per_player.setdefault(norm_name(p["name"]), []).append(pct_rank(pool, float(ppr)))

    rows, dropped = [], {"no_pedigree": 0, "too_few_seasons": 0}
    for name, dels in per_player.items():
        if len(dels) < MIN_SEASONS:
            dropped["too_few_seasons"] += 1
            continue
        pd = ped.get(name)
        if not pd:
            dropped["no_pedigree"] += 1
            continue
        rows.append({
            "name": name, "seasons": len(dels),
            "expect_log": float(pd.get("expect_log") or 0.0),
            "undrafted": bool(pd.get("undrafted")),
            "overall": pd.get("overall"),
            "position": pd.get("position") or "",
            "delivery": statistics.mean(dels),
        })

    if len(rows) < 50:
        print(f"only {len(rows)} gridiron careers with both inputs — not assigning.")
        return 2

    counts, slope, intercept = score(rows, "expect_log")
    corr = statistics.correlation([r["expect_log"] for r in rows],
                                  [r["delivery"] for r in rows])
    ranked = sorted(rows, key=lambda r: r["residual"])
    report = {
        "sport": "gridiron", "careers_scored": len(rows), "dropped": dropped,
        "min_seasons": MIN_SEASONS,
        "expectation": "gridiron_pedigree.json expect_log = 1 - log1p(pick)/log1p(262)",
        "delivery": "career-mean PPR ppg percentile WITHIN (season, position)",
        "fit": {"slope": round(slope, 4), "intercept": round(intercept, 3),
                "corr_expectation_delivery": round(corr, 4)},
        "counts": counts,
        "T0_examples": [{"name": r["name"], "pos": r["position"], "pick": r["overall"],
                         "delivery": round(r["delivery"], 1),
                         "residual": round(r["residual"], 1)}
                        for r in ranked if r["trajectory"] == "T0"][:10],
        "T1_examples": [{"name": r["name"], "pos": r["position"], "pick": r["overall"],
                         "undrafted": r["undrafted"], "delivery": round(r["delivery"], 1),
                         "residual": round(r["residual"], 1)}
                        for r in reversed(ranked) if r["trajectory"] == "T1"][:10],
    }
    out = ROOT / "data" / "trajectory_axis_gridiron.json"
    out.write_text(json.dumps({
        "caveat_scope": ("OFFENSIVE SKILL POSITIONS ONLY (QB/RB/WR/TE). All 18 gridiron "
                         "features are pass/rush/receiving; linemen and defenders are not "
                         "in the vector set."),
        "caveat_survivorship": ("A high pick who never earned 4 charted seasons is ABSENT, "
                                "not labelled T0 — the worst busts leave before "
                                "accumulating a career."),
        "caveat_delivery": ("Fantasy PPR is a FANTASY delivery measure. It rewards volume "
                            "and touchdowns and ignores blocking and route-running a coach "
                            "would value. It is the delivery signal this repo has, not the "
                            "delivery signal football has."),
        "report": report, "careers": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"gridiron careers scored: {len(rows)}   dropped: {dropped}")
    print(f"expectation -> delivery corr: {corr:+.4f}  (slope {slope:+.3f})")
    print(f"labels: T0 {counts['T0']}  T1 {counts['T1']}  unlabelled {counts['unlabelled']}\n")
    print("T0 - high pick, under-delivered:")
    for e in report["T0_examples"][:6]:
        print(f"  {e['pos']:>3} pick {str(e['pick']):>4}  pct {e['delivery']:>5}"
              f"  resid {e['residual']:>6}  {e['name']}")
    print("\nT1 - low pick / undrafted, over-delivered:")
    for e in report["T1_examples"][:6]:
        pk = "undrafted" if e["undrafted"] else f"pick {e['pick']}"
        print(f"  {e['pos']:>3} {pk:>10}  pct {e['delivery']:>5}"
              f"  resid {e['residual']:>6}  {e['name']}")
    print(f"\nwrote {out}")
    return 0


def score(rows: list[dict], expect_key: str) -> tuple[dict, float, float]:
    """Residual of delivery on expectation, then tail labels. Shared by both sports so a
    cross-sport comparison can never be an artefact of two different scoring rules."""
    xs = [r[expect_key] for r in rows]
    ys = [r["delivery"] for r in rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx) if sxx else 0.0
    intercept = my - slope * mx
    for r in rows:
        r["predicted"] = intercept + slope * r[expect_key]
        r["residual"] = r["delivery"] - r["predicted"]
    res_sorted = sorted(r["residual"] for r in rows)
    exp_sorted = sorted(xs)
    counts = {"T0": 0, "T1": 0, "unlabelled": 0}
    for r in rows:
        r["residual_pct"] = pct_rank(res_sorted, r["residual"])
        r["expect_pct"] = pct_rank(exp_sorted, r[expect_key])
        t = None
        if r["expect_pct"] >= HIGH_EXPECT_Q and r["residual_pct"] <= TAIL_PCT:
            t = "T0"
        elif r["expect_pct"] <= LOW_EXPECT_Q and r["residual_pct"] >= 100.0 - TAIL_PCT:
            t = "T1"
        r["trajectory"] = t
        counts[t or "unlabelled"] += 1
    return counts, slope, intercept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--sport", choices=("hoops", "gridiron"), default="hoops")
    args = ap.parse_args()

    if args.sport == "gridiron":
        return run_gridiron(args)

    for p in (PEDIGREE, SKILLS, VECTORS):
        if not p.exists():
            print(f"missing {p}")
            return 2

    ped = json.loads(PEDIGREE.read_text(encoding="utf-8"))["players"]
    sk = json.loads(SKILLS.read_text(encoding="utf-8"))
    vec = json.loads(VECTORS.read_text(encoding="utf-8"))
    vplayers = vec["players"] if isinstance(vec, dict) else vec
    grades = sk["grades"]

    # ORDER ALIGNMENT IS AN ASSERTION, NOT AN ASSUMPTION. build_skills.py documents
    # grades as order-aligned with vectors.json players[]; if that ever stops being true
    # every grade lands on the wrong player and nothing else here would notice.
    if len(vplayers) != len(grades):
        print(f"ALIGNMENT BROKEN: vectors.json has {len(vplayers)} players, "
              f"skills.json has {len(grades)} grade rows. Refusing to score.")
        return 2
    skill_keys = [s.get("key") if isinstance(s, dict) else s for s in sk["skills"]]
    if IMPACT not in skill_keys:
        print(f"skills.json has no {IMPACT!r} column: {skill_keys}")
        return 2
    idx = skill_keys.index(IMPACT)

    # ---- career-level delivery ------------------------------------------------
    per_player: dict[str, list[float]] = {}
    for pl, g in zip(vplayers, grades, strict=True):
        per_player.setdefault(norm_name(pl["name"]), []).append(float(g[idx]))

    pednorm = {norm_name(k): v for k, v in ped.items()}

    rows = []
    dropped = {"no_pedigree": 0, "too_few_seasons": 0}
    for name, impacts in per_player.items():
        if len(impacts) < MIN_SEASONS:
            dropped["too_few_seasons"] += 1
            continue
        pd = pednorm.get(name)
        if not pd:
            dropped["no_pedigree"] += 1
            continue
        rows.append({
            "name": name,
            "seasons": len(impacts),
            "expect_slot": float(pd.get("expect_slot") or 0.06),
            "undrafted": bool(pd.get("undrafted")),
            "overall": pd.get("overall"),
            "delivery": statistics.mean(impacts),
        })

    if len(rows) < 50:
        print(f"only {len(rows)} careers with both inputs — not assigning.")
        return 2

    # ---- residual of delivery on expectation ---------------------------------
    xs = [r["expect_slot"] for r in rows]
    ys = [r["delivery"] for r in rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx) if sxx else 0.0
    intercept = my - slope * mx
    for r in rows:
        r["predicted"] = intercept + slope * r["expect_slot"]
        r["residual"] = r["delivery"] - r["predicted"]

    res_sorted = sorted(r["residual"] for r in rows)
    exp_sorted = sorted(xs)
    for r in rows:
        r["residual_pct"] = pct_rank(res_sorted, r["residual"])
        r["expect_pct"] = pct_rank(exp_sorted, r["expect_slot"])

    for r in rows:
        t = None
        if r["expect_pct"] >= HIGH_EXPECT_Q and r["residual_pct"] <= TAIL_PCT:
            t = "T0"
        elif r["expect_pct"] <= LOW_EXPECT_Q and r["residual_pct"] >= 100.0 - TAIL_PCT:
            t = "T1"
        r["trajectory"] = t

    counts = {"T0": 0, "T1": 0, "unlabelled": 0}
    for r in rows:
        counts[r["trajectory"] or "unlabelled"] += 1

    # correlation, so "does billing predict delivery at all" is answered rather than assumed
    corr = statistics.correlation(xs, ys) if len(set(xs)) > 1 else 0.0

    report = {
        "sport": "hoops",
        "careers_scored": len(rows),
        "dropped": dropped,
        "min_seasons": MIN_SEASONS,
        "expectation": "pedigree.json expect_slot (rookie-scale curve)",
        "delivery": "mean percentile-within-season `impact` grade over the career",
        "fit": {"slope": round(slope, 4), "intercept": round(intercept, 3),
                "corr_expectation_delivery": round(corr, 4)},
        "counts": counts,
        "tails": {"TAIL_PCT": TAIL_PCT, "HIGH_EXPECT_Q": HIGH_EXPECT_Q,
                  "LOW_EXPECT_Q": LOW_EXPECT_Q},
    }

    ranked = sorted(rows, key=lambda r: r["residual"])
    report["T0_examples"] = [
        {"name": r["name"], "pick": r["overall"], "delivery": round(r["delivery"], 1),
         "residual": round(r["residual"], 1)}
        for r in ranked if r["trajectory"] == "T0"][:10]
    report["T1_examples"] = [
        {"name": r["name"], "pick": r["overall"], "undrafted": r["undrafted"],
         "delivery": round(r["delivery"], 1), "residual": round(r["residual"], 1)}
        for r in reversed(ranked) if r["trajectory"] == "T1"][:10]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "built_from": {"pedigree": str(PEDIGREE), "skills": str(SKILLS)},
        "caveat_survivorship": (
            "A high pick who never earned 4 charted seasons is ABSENT, not labelled T0. "
            "The worst busts leave the league before they accumulate a career, so T0 is "
            "biased toward players who under-delivered while still being good enough to "
            "keep playing. This understates the tail it is trying to name."),
        "caveat_scope": "hoops only. gridiron has the inputs but no exported artifact; pitch has none.",
        "report": report,
        "careers": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"careers scored: {len(rows)}   dropped: {dropped}")
    print(f"expectation -> delivery corr: {corr:+.4f}  (slope {slope:+.3f})")
    print(f"labels: T0 {counts['T0']}   T1 {counts['T1']}   unlabelled {counts['unlabelled']}\n")
    print("T0 — high billing, under-delivered:")
    for e in report["T0_examples"][:6]:
        print(f"  pick {str(e['pick']):>4}  impact {e['delivery']:>5}  resid {e['residual']:>6}  {e['name']}")
    print("\nT1 — low billing, over-delivered:")
    for e in report["T1_examples"][:6]:
        pk = "undrafted" if e["undrafted"] else f"pick {e['pick']}"
        print(f"  {pk:>10}  impact {e['delivery']:>5}  resid {e['residual']:>6}  {e['name']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
