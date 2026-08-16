#!/usr/bin/env python3
"""The matched cross-sport draft-value comparison, and the record of it flipping.

Solo personal project, no connection to employer, built with public/free-tier only

WHAT THIS EXISTS TO REPORT is not a number. It is that the number has changed sign and
significance across three specifications, every one of which was defensible when it was
run, and each of which was reported as a finding at the time.

    #   specification                     gridiron   hoops     gap (grid - hoops)  verdict
    1   unmatched constructs              +0.4236   +0.2598   +0.1638             CI excludes 0
    2   matched VOR, no eligibility gate  +0.3950   +0.4534   -0.0584             CI spans 0
    3   matched VOR + eligibility gate    +0.3950   +0.4934   -0.0984             CI excludes 0

Spec 1 compared `impact` (a curated on-court composite) against fantasy PPR — the same
correlation formula applied to two different questions. Spec 2 fixed the construct and
reversed the sign. Spec 3 applied vector-hoops' own minutes gate, which the raw per-100
cache bypasses, and pushed the reversed gap back out past zero.

The honest reading: the DIRECTION of the sport gap is not robust to specification. Spec 3
is the best of the three and its CI excludes zero, but a quantity that moved +0.16 -> -0.06
-> -0.10 under three reasonable analyst choices has a specification variance larger than
its own sampling CI, and the CI does not know that. Reporting "hoops rewards draft position
more than football does" as a finding would be reporting the last coin flip.

WHY THE HISTORY IS IN THE SCRIPT rather than a commit message. Each earlier spec was
computed inline, ad hoc, and then discarded — so when the next one produced a different
answer there was nothing to compare against except memory. A number that is recomputed
from scratch each time it is questioned cannot be seen to be unstable. Specs 1 and 2 are
hard-coded here as PRIOR results, clearly labelled as such; spec 3 is recomputed live.

    python pipeline/compare_matched_draft_value.py
    python pipeline/compare_matched_draft_value.py --reps 10000
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "data" / "vor_draft_value.json"
HOOPS = ROOT / "data" / "hoops_vor_draft_value.json"
OUT = ROOT / "data" / "matched_draft_value_comparison.json"

SEED = 20260803

# Prior specifications, recorded rather than recomputed. These are what was reported at the
# time; they are kept here so the instability is visible without re-deriving it.
PRIOR = [
    {
        "spec": 1,
        "label": "unmatched constructs (impact vs fantasy PPR)",
        "gridiron": 0.4236,
        "hoops": 0.2598,
        "gap": 0.1638,
        "ci_excludes_zero": True,
        "why_superseded": "not a comparison — the same formula over two different quantities",
    },
    {
        "spec": 2,
        "label": "matched VOR, hoops read the raw per-100 cache",
        "gridiron": 0.3950,
        "hoops": 0.4534,
        "gap": -0.0584,
        "ci_excludes_zero": False,
        "why_superseded": (
            "bypassed vector-hoops' schedule-aware minutes gate; per-100 rates "
            "explode for low-minute players (572 raw vs 484 eligible in 2023-24)"
        ),
    },
]


def corr(rows: list[dict]) -> float:
    xs = [r["expect_log"] for r in rows]
    ys = [r["vor_total"] for r in rows]
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return 0.0
    return statistics.correlation(xs, ys)


def boot(rows: list[dict], reps: int, rng: random.Random) -> list[float]:
    n = len(rows)
    out = []
    for _ in range(reps):
        s = [rows[rng.randrange(n)] for _ in range(n)]
        out.append(corr(s))
    return out


def ci(vals: list[float]) -> tuple[float, float]:
    v = sorted(vals)
    lo = v[int(0.025 * (len(v) - 1))]
    hi = v[int(0.975 * (len(v) - 1))]
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    for p in (GRID, HOOPS):
        if not p.exists():
            print(f"missing {p} — build both VOR value tables first")
            return 2

    g = json.loads(GRID.read_text(encoding="utf-8"))
    h = json.loads(HOOPS.read_text(encoding="utf-8"))
    grows = g.get("players") or g.get("player_rows") or []
    hrows = h.get("players") or []
    if not grows or not hrows:
        print("one of the value tables carries no per-player rows — rebuild it")
        return 2

    rng = random.Random(SEED)
    rg, rh = corr(grows), corr(hrows)
    bg, bh = boot(grows, args.reps, rng), boot(hrows, args.reps, rng)
    # The two samples are INDEPENDENT — different sports, disjoint players — so the gap is
    # resampled by pairing independent draws, not by a paired bootstrap.
    gaps = [a - b for a, b in zip(bg, bh, strict=True)]
    glo, ghi = ci(bg)
    hlo, hhi = ci(bh)
    dlo, dhi = ci(gaps)
    excludes = not (dlo <= 0.0 <= dhi)

    live = {
        "spec": 3,
        "label": "matched VOR + vector-hoops eligibility gate",
        "gridiron": round(rg, 4),
        "hoops": round(rh, 4),
        "gap": round(rg - rh, 4),
        "gridiron_ci95": [round(glo, 4), round(ghi, 4)],
        "hoops_ci95": [round(hlo, 4), round(hhi, 4)],
        "gap_ci95": [round(dlo, 4), round(dhi, 4)],
        "ci_excludes_zero": excludes,
        "n_gridiron": len(grows),
        "n_hoops": len(hrows),
    }

    signs = {1 if s["gap"] > 0 else -1 for s in PRIOR} | {1 if live["gap"] > 0 else -1}
    report = {
        "question": "Does draft position predict delivered value more strongly in one sport?",
        "current": live,
        "prior_specifications": PRIOR,
        "sign_changed_across_specifications": len(signs) > 1,
        "spread_of_point_estimates": round(
            max([s["gap"] for s in PRIOR] + [live["gap"]]) - min([s["gap"] for s in PRIOR] + [live["gap"]]),
            4,
        ),
        "width_of_current_ci": round(dhi - dlo, 4),
        "verdict": (
            "NOT REPORTABLE AS A SPORT DIFFERENCE. The point estimate has moved across the "
            "sign boundary under three defensible specifications. Spec 3 is the best of "
            "them and its CI excludes zero, but the spread of point estimates across "
            "specifications is of the same order as the sampling CI, and the CI cannot see "
            "specification variance. The instability IS the result."
        ),
        "what_would_settle_it": (
            "A pre-registered specification chosen before seeing any of the three answers, "
            "plus a third and fourth sport. With one binary comparison and a menu of "
            "analyst choices, any desired sign is reachable."
        ),
        "known_asymmetry_kept_deliberately": (
            "expect_log uses MAX_PICK 262 for gridiron and 60 for hoops. That is a real "
            "structural difference between the drafts, declared as a constant rather than "
            "hidden in the method — but it does mean the two expectation scales are not "
            "the same function, only the same FORM. A correlation is invariant to a "
            "monotone linear rescale, not to a different log denominator."
        ),
        "seed": SEED,
        "reps": args.reps,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"{'spec':<38} {'grid':>8} {'hoops':>8} {'gap':>8}  CI")
    for s in PRIOR:
        print(
            f"{s['label'][:38]:<38} {s['gridiron']:>+8.4f} {s['hoops']:>+8.4f} "
            f"{s['gap']:>+8.4f}  {'excl 0' if s['ci_excludes_zero'] else 'spans 0'}  PRIOR"
        )
    print(
        f"{live['label'][:38]:<38} {rg:>+8.4f} {rh:>+8.4f} {live['gap']:>+8.4f}  "
        f"[{dlo:+.4f}, {dhi:+.4f}] {'excl 0' if excludes else 'spans 0'}  LIVE"
    )
    print(f"\nn: gridiron {len(grows)}  hoops {len(hrows)}   reps {args.reps}   seed {SEED}")
    print(
        f"point estimates span {report['spread_of_point_estimates']:.4f}; "
        f"current CI is {report['width_of_current_ci']:.4f} wide"
    )
    print(f"\n{report['verdict']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
