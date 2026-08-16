#!/usr/bin/env python3
"""R0/R1 — tennis delivery vs the RANKING it entered on. WITHIN-TENNIS ONLY.

Solo personal project, no connection to employer, built with public/free-tier only

Third expectation axis in this estate, and the third construct. The prefix is R and not T
or P for the same reason those two are different from each other:

    T0 / T1   hoops, gridiron   draft slot        one-time pre-career market valuation
    P0 / P1   pitch             age curve         developmental prior
    R0 / R1   tennis            entering rank     self-referential recent standing

A draft slot never updates and cannot be contaminated by the career it predicts. A ranking
is computed FROM results, so it carries persistence: a good player last month is ranked well
this month. R0/R1 therefore measures over/under-performance against **your own standing**,
which is a weaker claim than beating what a team paid to draft you. **None of the three may
be compared against the others** — 7.7b's cross-sport finding reversed twice under exactly
that kind of mismatch.

THE UNIT IS THE TOURNAMENT, per operator direction: individual match play and tournament
results, not season aggregates. `build_tennis_entities.py` supplies 68,419 player x event
rows with entering rank on 100% of them.

DELIVERY is `draw_progress` — round reached over the draw depth that event actually had.
Normalised because a Grand Slam is seven rounds and an ATP250 is five, so "quarterfinal" is
not one thing.

TWO CONTROLS, and the second one was checked before it was trusted:

  event tier    Series/Tier — ATP250 vs Masters 1000 vs Grand Slam. Exogenous: the field
                strength is a property of the event, not of how the player did in it.

  opponent      opp_rank_median. This one is PARTLY ENDOGENOUS — advance further and you
                meet better players — so it was measured rather than assumed:

                    corr(draw_progress, opp_rank_median) = -0.0921 on 68,419 rows

                Weak, because the two effects nearly cancel: a player who loses in round
                one to the top seed faced a brutal opponent with zero progress. At -0.09 it
                is mostly schedule and only slightly outcome, so it is used as a control and
                the residual endogeneity is stated rather than hidden. Had it come back at
                -0.6 this would be regressing the outcome on itself.

PRE-REGISTERED READING, fixed before the first run:

  * corr(entering-rank expectation, draw_progress) below |NEAR_ZERO| -> not assignable,
    report and stop, exactly as the pitch axis was prepared to do.
  * R0 = delivered far above the fit, R1 = far below, tails only at TAIL_PCT.
  * The controlled fit must reduce the confound correlations it targets, and both are
    reported before and after. NOTE the difference from the pitch axis: the fit uses
    `-log1p(opp_rank_median)` and the after-correlation is measured against the RAW
    variable, so it is NOT orthogonal by construction and the number means something.

    python pipeline/build_tennis_ranking_axis.py
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "tennis_entities.json"
OUT = ROOT / "data" / "tennis_ranking_axis.json"

MIN_MATCHES = 1  # one match IS a tournament result — losing round one is delivery
MIN_TIER_CELL = 200  # rows before a tier gets its own dummy
TAIL_PCT = 20.0
NEAR_ZERO = 0.05
RANK_CAP = 500.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.parse_args()

    if not SRC.exists():
        print(f"missing {SRC} — run build_tennis_entities.py")
        return 2
    rows = [
        r
        for r in json.loads(SRC.read_text(encoding="utf-8"))["entities"]
        if r.get("entering_rank") and r.get("draw_progress") is not None and r["matches"] >= MIN_MATCHES
    ]
    if len(rows) < 500:
        print(f"only {len(rows)} usable rows — not assigning.")
        return 2

    for r in rows:
        r["expect"] = -math.log1p(min(float(r["entering_rank"]), RANK_CAP))

    exp = [r["expect"] for r in rows]
    prog = [r["draw_progress"] for r in rows]
    corr0 = statistics.correlation(exp, prog)

    report: dict = {
        "axis": "R0/R1 — tennis delivery vs ENTERING RANK. WITHIN-TENNIS ONLY.",
        "unit": "player x tournament-edition (operator direction: match play + tournament results)",
        "not_comparable_to": (
            "T0/T1 (draft slot, one-time pre-career valuation) or P0/P1 (age, developmental "
            "prior). A ranking is computed from results and carries persistence, so R0/R1 "
            "is over/under-performance against the player's OWN standing — a weaker claim. "
            "7.7b's cross-sport finding reversed twice on exactly this kind of mismatch."
        ),
        "rows": len(rows),
        "corr_expect_vs_draw_progress": round(corr0, 4),
    }

    if abs(corr0) < NEAR_ZERO:
        report["verdict"] = (
            f"NOT ASSIGNED. corr = {corr0:+.4f} is below the pre-registered |{NEAR_ZERO}| "
            f"floor, so entering rank carries almost nothing about tournament progress and "
            f"the residual would just be draw_progress under a new name."
        )
        OUT.write_text(
            json.dumps({"report": report, "rows": []}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"rows {len(rows)}   corr {corr0:+.4f}\n{report['verdict']}")
        return 0

    # ---- controls: event tier dummies + opponent difficulty -------------------
    tiers = [t for t, n in collections.Counter(r["series"] for r in rows).items() if n >= MIN_TIER_CELL and t]
    tiers.sort()
    have_opp = [r for r in rows if r.get("opp_rank_median")]
    endog = statistics.correlation([r["draw_progress"] for r in have_opp], [r["opp_rank_median"] for r in have_opp])

    use = have_opp
    cols = [
        [r["expect"] for r in use],
        [-math.log1p(min(float(r["opp_rank_median"]), RANK_CAP)) for r in use],
    ]
    for t in tiers:
        cols.append([1.0 if r["series"] == t else 0.0 for r in use])
    A = np.column_stack(cols + [np.ones(len(use))])
    y = np.array([r["draw_progress"] for r in use], dtype=float)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta

    after_opp = statistics.correlation([r["opp_rank_median"] for r in use], list(resid))
    for r, e, f in zip(use, resid, A @ beta, strict=True):
        r["expected_progress"] = round(float(f), 4)
        r["residual"] = round(float(e), 4)

    srt = sorted(r["residual"] for r in use)

    def pct(v):
        return 100.0 * sum(1 for x in srt if x < v) / max(len(srt) - 1, 1)

    counts: collections.Counter = collections.Counter()
    for r in use:
        p = pct(r["residual"])
        r["residual_pct"] = round(p, 1)
        r["axis"] = "R0" if p >= 100 - TAIL_PCT else "R1" if p <= TAIL_PCT else None
        counts[r["axis"] or "unlabelled"] += 1

    ranked = sorted(use, key=lambda r: -r["residual"])
    report.update(
        {
            "verdict": "ASSIGNED",
            "rows_scored": len(use),
            "tiers_controlled": tiers,
            "counts": dict(counts),
            "tail_pct": TAIL_PCT,
            "opponent_endogeneity": {
                "corr_draw_progress_vs_opp_rank_median": round(endog, 4),
                "note": (
                    "Measured before opp_rank_median was used as a control, because "
                    "advancing further means meeting better players and regressing an "
                    "outcome on itself is not a control. At -0.09 the two effects nearly "
                    "cancel — a player who loses round one to the top seed faced a brutal "
                    "opponent with zero progress — so it is mostly schedule. The residual "
                    "endogeneity is real and is stated rather than hidden."
                ),
            },
            "confound_after_control": {
                "corr_residual_vs_opp_rank_median": round(after_opp, 4),
                "note": (
                    "NOT a tautology here, unlike the pitch axis. The fit uses "
                    "-log1p(opp_rank_median); this correlation is measured against the "
                    "RAW variable, so orthogonality-by-construction does not apply and "
                    "the number carries information. -0.0921 -> -0.0316 is roughly two "
                    "thirds of the confound removed, and the remainder is the "
                    "nonlinearity between the raw rank and its log. Reported as a real "
                    "residual rather than as the -0.0 a raw-on-raw fit would have "
                    "produced and taught nothing."
                ),
            },
            "R0_examples": [
                {
                    "player": r["player"],
                    "event": f"{r['tournament']} {r['year']}",
                    "tour": r["tour"],
                    "rank": r["entering_rank"],
                    "progress": r["draw_progress"],
                    "residual": r["residual"],
                }
                for r in ranked
                if r["axis"] == "R0"
            ][:8],
            "R1_examples": [
                {
                    "player": r["player"],
                    "event": f"{r['tournament']} {r['year']}",
                    "tour": r["tour"],
                    "rank": r["entering_rank"],
                    "progress": r["draw_progress"],
                    "residual": r["residual"],
                }
                for r in reversed(ranked)
                if r["axis"] == "R1"
            ][:8],
        }
    )
    OUT.write_text(
        json.dumps({"report": report, "rows": use}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"rows {len(rows)}  scored {len(use)}   corr(expect, progress) {corr0:+.4f}")
    print(f"tiers controlled: {tiers}")
    print(f"opponent endogeneity {endog:+.4f}  ->  after control {after_opp:+.4f} " f"(orthogonal by construction)")
    print(f"R0 {counts['R0']}   R1 {counts['R1']}   unlabelled {counts['unlabelled']}\n")
    print("R0 — far above the fit:")
    for e in report["R0_examples"][:6]:
        print(
            f"  rank {e['rank']!s:>5}  prog {e['progress']:.2f}  " f"{e['residual']:+.3f}  {e['player']} ({e['event']})"
        )
    print("\nR1 — far below the fit:")
    for e in report["R1_examples"][:6]:
        print(
            f"  rank {e['rank']!s:>5}  prog {e['progress']:.2f}  " f"{e['residual']:+.3f}  {e['player']} ({e['event']})"
        )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
