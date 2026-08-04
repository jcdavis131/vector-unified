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
import hashlib
import json
import re
import sys
from pathlib import Path

HUB = Path("C:/Users/jcdav/vector-hub")
DATA = HUB / "assets" / "data"
REGISTRY = Path(__file__).resolve().parent.parent / "data" / "superlative_registry.json"


def registry() -> dict[str, dict]:
    """Cleared claims, keyed by a hash of the CLAIM TEXT.

    Keyed on the sentence, never on slug+location. A clearance belongs to the words that
    were checked; if a re-extraction rewrites round[3].reveal, the old verification must not
    transfer to the new sentence sitting at the same address. Keying by location would be
    the stale-verification defect this repo has spent the phase correcting — a green mark
    describing something that is no longer there.
    """
    if not REGISTRY.exists():
        return {}
    doc = json.loads(REGISTRY.read_text(encoding="utf-8"))
    out = {}
    for e in doc.get("entries", []):
        out[e["text_sha"]] = e
    return out


def claim_key(text: str) -> str:
    """Hash the WHOLE field, not a sentence sliced out of it.

    The first version split on "." to isolate the sentence containing the superlative. That
    breaks on every decimal, and this data is almost entirely decimals — "0.986" splits into
    "0" and "986 similarity", so 13 of 14 registry entries failed to match text they were
    written from. Hashing the whole body/reveal is unambiguous and still has the property
    that matters: ANY edit to the field voids the clearance.
    """
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()[:16]

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

    REG = registry()
    cleared_by_verdict: dict[str, int] = {}
    unchecked: list[tuple[str, str, str]] = []
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
        # Registry clearances, matched on the sentence itself.
        cleared_here = 0
        still_open: list[tuple[str, str]] = []
        for loc, idx, t in hits:
            if (loc, idx, t) in auto:
                continue
            k = claim_key(t)
            sent = " ".join(t.split())
            e = REG.get(k)
            if e:
                cleared_here += 1
                cleared_by_verdict[e["verdict"]] = cleared_by_verdict.get(e["verdict"], 0) + 1
            else:
                still_open.append((loc, sent))
        print(f"  {d['slug']:9} {len(hits):2} superlative-shaped, {len(auto)} auto-checkable, "
              f"{cleared_here} cleared in registry, {len(still_open)} UNCHECKED")
        for loc, sent in still_open:
            print(f"       UNCHECKED   {loc}  {sent[:110]}")
        unchecked.extend((d["slug"], loc, sent) for loc, sent in still_open)
        for loc, idx, t in auto:
            verdict, why = check_page_scoped(rounds, idx, t)
            if verdict == "TRUE":
                verified += 1
            elif verdict == "FALSE":
                false_claims.append(f"{d['slug']} {loc}: {why}\n        claim: {t[:150]}")
            print(f"       {verdict:12} {loc}  {why}")
        flagged += len(still_open)

    print(f"\n{verified} page-scoped verified by arithmetic, {len(false_claims)} FALSE, "
          f"{sum(cleared_by_verdict.values())} cleared in registry {cleared_by_verdict}, "
          f"{flagged} UNCHECKED")
    if flagged:
        print("\nUNCHECKED IS NOT CLEAN. Verifying 'the largest in the corpus' needs the "
              "corpus, and the sentence does not say which array to scan. Check each one "
              "against its artifact and add it to data/superlative_registry.json with the "
              "evidence. Do not clear one without reading the array.")
    else:
        print("\nEvery superlative on every page is either verified by arithmetic against "
              "the page's own values or cleared in the registry with named evidence. A "
              "clearance is keyed to the field's CONTENT, so any edit voids it.")

    if false_claims:
        print(f"\n{len(false_claims)} FALSE page-scoped superlative(s):")
        for c in false_claims:
            print(f"  {c}")

    # UNCHECKED FAILS TOO, not only FALSE. validate.py prints just the LAST line of a
    # check's output, so an UNCHECKED count that did not move the exit code would grow
    # behind a green line — the silent-cap failure this repo keeps finding. An unverified
    # superlative on a live page is a problem, and clearing one costs a single read of the
    # array it ranks.
    if (false_claims or flagged) and args.check:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
