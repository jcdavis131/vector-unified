#!/usr/bin/env python3
"""Assert the things that must be true TOGETHER across the draft-value artifacts.

Solo personal project, no connection to employer, built with public/free-tier only

The draft-value table took three wrong versions to get right, and every one of them looked
publishable: plausible magnitudes, sensible-looking orderings, no absurd values. What
exposed each was an INTERNAL CONTRADICTION — two numbers that could not both be true —
and each time it was caught by reading the output carefully rather than by anything that
would fail on its own.

Face validity passed all three. That is the argument for this file.

The three failures, each now an assertion:

  I1  25 drafted, 28 survivors scored. The survival half was windowed to draft years
      1999-2022 and the delivery half was not (1980-2025), so they described different
      pools. A survivor count exceeding its own denominator is arithmetically impossible
      and is the cheapest possible tell.

  I2  `never played` = 0.0% for late-round quarterbacks, when the truth is 71.6%.
      gridiron_pedigree.json was built FROM the vector set, so iterating it saw only
      players who played. A denominator that contains no zeros is not a denominator.

  I3  A pick that produced NOTHING outranking one that produced below-starter play —
      QB R4-7 at -2.65 against QB R1 at -7.87. Unfloored VOR let a wasted seventh-rounder
      beat a franchise quarterback.

Plus two structural ones that would have caught related mistakes:

  I4  Drafted counts must AGREE between the survivorship probe and the value table. They
      read the same CSV through different code paths; disagreement means one of them is
      filtering differently and silently.

  I5  Value must not INCREASE with later draft rounds inside a position. Not a law of
      nature, but a violation means either a real and remarkable finding or a bug, and it
      should never pass unnoticed.

    python pipeline/check_draft_value_invariants.py
    python pipeline/check_draft_value_invariants.py --check   # exit 1 on any violation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SURV = ROOT / "data" / "qb_survivorship_probe.json"
VOR = ROOT / "data" / "vor_draft_value.json"
ORDER = ["R1", "R2", "R3", "R4-7"]


def check() -> list[str]:
    problems: list[str] = []
    for p in (SURV, VOR):
        if not p.exists():
            return [f"CANNOT CHECK: {p} missing — run the pipeline first."]

    surv = json.loads(SURV.read_text(encoding="utf-8"))["report"]
    vor = json.loads(VOR.read_text(encoding="utf-8"))
    cells = {(c["pos"], c["bucket"]): c for c in vor["cells"]}

    # I1 — a survivor count can never exceed its own denominator
    for (pos, b), c in cells.items():
        n_surv, drafted = c.get("n_surv"), c.get("drafted")
        if n_surv is not None and drafted and n_surv > drafted:
            problems.append(
                f"I1 {pos} {b}: {n_surv} survivors scored against {drafted} drafted — "
                f"impossible; the two halves are drawn from different pools")

    # I2 — a denominator with no zeros is not a denominator
    for (pos, b), c in cells.items():
        if c.get("thin"):
            continue
        never = c.get("never_played_pct")
        surv_pct = c.get("survival_pct")
        if never is None:
            problems.append(f"I2 {pos} {b}: no never_played_pct — cannot tell whether the "
                            f"washouts were counted")
            continue
        # If survival is low, a large share must never have played. A 9%-survival bucket
        # reporting 0% never-played means the washouts are missing from the pool.
        if surv_pct is not None and surv_pct < 40.0 and never < 5.0:
            problems.append(
                f"I2 {pos} {b}: survival {surv_pct}% but only {never}% never played — "
                f"the denominator is missing its zeros (survivor-only pool)")

    # I3 — floored value must never be negative
    for (pos, b), c in cells.items():
        ev = c.get("ev_vor")
        if ev is not None and ev < 0:
            problems.append(
                f"I3 {pos} {b}: EV {ev} is negative. VOR is floored at zero per season, "
                f"so a bucket mean cannot be — a wasted pick would outrank a good one")

    # I4 — drafted counts must agree across artifacts
    per = surv["per_position"]
    for (pos, b), c in cells.items():
        want = ((per.get(pos) or {}).get("by_bucket") or {}).get(b, {}).get("drafted")
        got = c.get("drafted")
        if want is not None and got is not None and want != got:
            problems.append(
                f"I4 {pos} {b}: survivorship probe says {want} drafted, value table says "
                f"{got} — same CSV, different filtering somewhere")

    # I5 — value must not rise with later rounds inside a position
    for pos in {p for p, _ in cells}:
        seq = [(b, cells[(pos, b)].get("ev_vor")) for b in ORDER if (pos, b) in cells]
        seq = [(b, v) for b, v in seq if v is not None]
        for (b0, v0), (b1, v1) in zip(seq, seq[1:], strict=False):
            if v1 > v0:
                problems.append(
                    f"I5 {pos}: {b1} ({v1}) is worth MORE than {b0} ({v0}). Either a "
                    f"remarkable finding or a bug — it must not pass unnoticed")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 on any violation")
    args = ap.parse_args()

    problems = check()
    if not problems:
        print("draft-value artifacts are internally consistent.")
        print("  I1 no survivor count exceeds its denominator")
        print("  I2 low-survival buckets carry their washouts")
        print("  I3 no negative floored EV")
        print("  I4 drafted counts agree across artifacts")
        print("  I5 value does not rise with later rounds")
        return 0

    print(f"{len(problems)} invariant violation(s):\n")
    for p in problems:
        print(f"  {p}")
    print("\nEach of I1-I3 is a bug that actually shipped in an earlier version of this")
    print("table and looked entirely reasonable at the time. Do not reason past them.")
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
