#!/usr/bin/env python3
"""Every fabrication this project has produced was a superlative. Six for six.

Solo personal project, no connection to employer, built with public/free-tier only

Across three rounds of generating the dumbmodel.com model cards, adversarial verifiers
caught six fabrications. NOT ONE was a misread number. Every single one was a correctly-read
value wrapped in an invented claim about its rank:

    "the two closest careers in the entire file ... differ by 0.0057"
        0.0057 was right. 160 adjacent gaps were smaller; 53 were exactly 0.0.
    "observed delivery runs from 8.5 (Anthony Bennett, pick 1) to 97.5"
        97.5 was the true global max. 8.5 was the minimum AMONG PICK-1 PLAYERS.
        The real floor is byron mullens at 5.2, pick 24.
    "they are each other's nearest comps at 0.991"
        McCaffrey.comps[0] = Bijan @ 0.991 was right. Bijan.comps[0] was someone else.
        Mutual-ness was never checked.
    "the largest first-half average of any career in this pair set" (LeBron, 24.32)
        luka doncic sits at 24.58 ON THE SAME PAGE. Produced TWICE, the second time in a
        prompt that named this exact failure as the thing not to repeat.

The pattern is stable enough to guard: a generator reads values reliably and invents rank
claims about them reliably. A warning in a prompt demonstrably does not stop it.

WHAT THIS CAN AND CANNOT DO, stated up front because the honest scope is narrow:

  AUTO-VERIFIED   claims scoped to the page's own data — "the closest call on the board",
                  "the sharpest of any name here". The page carries every value it is
                  ranking, so the claim is checkable with arithmetic and FAILS the gate
                  when false.

  FLAGGED         every other superlative. Verifying "the largest in the corpus" needs the
                  corpus, and which array to scan is not recoverable from the sentence. It
                  is REPORTED with its location so a reviewer sees it, and counted so the
                  number cannot quietly grow.

Detection is deliberately HIGH-RECALL and therefore over-fires — "the last pick in the
draft" is caught and is not a rank claim about data. That is the correct direction to be
wrong: a false positive costs one read, a false negative ships a lie on a live page.

    python pipeline/check_superlatives.py
    python pipeline/check_superlatives.py --check   # exit 1 on a FALSE auto-verified claim
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HUB = Path("C:/Users/jcdav/vector-hub")
DATA = HUB / "assets" / "data"

SUPERLATIVE = re.compile(
    r"\b(the (highest|largest|lowest|smallest|closest|most|best|worst|only|top|biggest|"
    r"sharpest|steepest|widest|narrowest)"
    r"|highest|largest|lowest|smallest|closest|biggest|sharpest|steepest"
    r"|each other'?s|of any\b|in the entire\b|anywhere in\b|no other\b|nobody else\b"
    r"|the sole\b|unmatched\b|runs from\b|ranges from\b)", re.I)

# A claim is PAGE-SCOPED when it says so. These phrases mean "among the things shown here",
# which the page carries in full and can therefore be checked with arithmetic.
PAGE_SCOPED = re.compile(
    r"\b(on (this|the) (board|page)|in (this|the) (pair set|board|page)|"
    r"of any (name|pair|player|row|career|ticker)s? (on|in) (this|the)|"
    r"shown here|listed here|among these)\b", re.I)

# Which quantity a page-scoped claim is about, and how to compute it from the rounds.
KIND = [
    (re.compile(r"\bclosest\b|\bnarrowest\b", re.I), "min_gap"),
    (re.compile(r"\bwidest\b|\bbiggest gap\b", re.I), "max_gap"),
    (re.compile(r"\bhighest\b|\blargest\b|\btop\b|\bbiggest\b", re.I), "max_value"),
    (re.compile(r"\blowest\b|\bsmallest\b", re.I), "min_value"),
]


def numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+\.\d+|\b-?\d{1,6}\b", text)]


def check_page_scoped(rounds: list[dict], idx: int, text: str) -> tuple[str, str]:
    """Returns (verdict, explanation). verdict in TRUE / FALSE / UNDECIDABLE."""
    kind = next((k for pat, k in KIND if pat.search(text)), None)
    if kind is None:
        return "UNDECIDABLE", "page-scoped but no recognised quantity"
    gaps = [(abs(float(r["a"]["value"]) - float(r["b"]["value"])), i)
            for i, r in enumerate(rounds)]
    vals = [(float(s["value"]), i, s["name"])
            for i, r in enumerate(rounds) for s in (r["a"], r["b"])]
    if kind in ("min_gap", "max_gap"):
        best = (min if kind == "min_gap" else max)(gaps)
        ok = best[1] == idx
        return ("TRUE" if ok else "FALSE",
                f"{kind}={best[0]:.4f} at round {best[1]}; this is round {idx}")
    best = (max if kind == "max_value" else min)(vals)
    named = [n for n in (rounds[idx]["a"]["name"], rounds[idx]["b"]["name"])]
    ok = best[2] in named
    return ("TRUE" if ok else "FALSE",
            f"{kind}={best[0]} held by {best[2]!r} (round {best[1]}); "
            f"this round names {named}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not DATA.exists():
        print(f"missing {DATA}")
        return 2

    false_claims, flagged, verified = [], 0, 0
    for f in sorted(DATA.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rounds = (d.get("game") or {}).get("rounds") or []
        rows: list[tuple[str, int, str]] = []
        for i, s in enumerate(d.get("headline_stats") or []):
            rows.append((f"stat[{i}].label", -1, s.get("label", "")))
        for i, s in enumerate(d.get("insights") or []):
            rows.append((f"insight[{i}].title", -1, s.get("title", "")))
            rows.append((f"insight[{i}].body", -1, s.get("body", "")))
        for i, r in enumerate(rounds):
            rows.append((f"round[{i}].reveal", i, r.get("reveal", "")))
        if d.get("caveat"):
            rows.append(("caveat", -1, d["caveat"]))

        hits = [(loc, idx, t) for loc, idx, t in rows if SUPERLATIVE.search(t)]
        auto = [(loc, idx, t) for loc, idx, t in hits
                if idx >= 0 and PAGE_SCOPED.search(t)]
        print(f"  {d['slug']:9} {len(hits):2} superlative-shaped, {len(auto)} auto-checkable")
        for loc, idx, t in auto:
            verdict, why = check_page_scoped(rounds, idx, t)
            if verdict == "TRUE":
                verified += 1
            elif verdict == "FALSE":
                false_claims.append(f"{d['slug']} {loc}: {why}\n        claim: {t[:150]}")
            print(f"       {verdict:12} {loc}  {why}")
        flagged += len(hits) - len(auto)

    print(f"\n{verified} page-scoped claim(s) verified TRUE, {len(false_claims)} FALSE, "
          f"{flagged} flagged for artifact-level review")
    print("\nFLAGGED IS NOT CLEAN. Verifying 'the largest in the corpus' needs the corpus, "
          "and the sentence does not say which array to scan. These are reported so they "
          "are visible and counted, not because they have been checked.")

    if false_claims:
        print(f"\n{len(false_claims)} FALSE page-scoped superlative(s):")
        for c in false_claims:
            print(f"  {c}")
        return 1 if args.check else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
