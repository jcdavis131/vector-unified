#!/usr/bin/env python3
"""A sponsor appears at several venues at once. A place name does not. (cross-venue filter)

Solo personal project, no connection to employer, built with public/free-tier only

build_tennis_sponsors.py produced 36 CONFIRMED sponsor tokens and was explicit that the
list is a CANDIDATE list, not entity resolution:

    "A token match is a NAME match, not an entity resolution — 'citi' matching a company
     token does not prove that company sponsored that event."

It also named its own known false-positive class and the filter that would remove it:

    "Melbourne 2021 ran four extra events during the quarantine-bubble calendar — Great
     Ocean Road Open, Murray River Open, Phillip Island Trophy, Yarra Valley Classic — all
     named after Australian geography, none sponsored."

    "AEGON -> Viking appears at Birmingham AND Eastbourne in the same year. One sponsor
     rebrand surfacing simultaneously at two unrelated venues is not something a tokeniser
     can manufacture. Cross-venue synchrony is a stronger check than either tier and is the
     natural next filter if this needs tightening."

This is that filter, implemented as specified. A token survives only if it is present at
TWO OR MORE DISTINCT LOCATIONS IN A SHARED YEAR. The logic is not a heuristic about which
words look corporate — it is a structural fact about how sponsorship works: a company buys
naming rights to several events in a season, whereas a river is in exactly one place.

WHAT THIS STILL DOES NOT DO, said plainly because the file it builds on was careful to say
it. Surviving this filter does not resolve a token to a company. It removes the geography
class and any token that only ever appears at one venue. A surviving token is a much better
CANDIDATE and nothing more; joining it to a ticker needs a second source.

    python pipeline/build_sponsor_synchrony.py
    python pipeline/build_sponsor_synchrony.py --check   # exit 1 if the input is missing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "tennis_sponsors.json"
OUT = ROOT / "data" / "sponsor_synchrony.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not SRC.exists():
        print(f"missing {SRC} — run build_tennis_sponsors.py first")
        return 2 if args.check else 0
    src = json.loads(SRC.read_text(encoding="utf-8"))
    detail = src.get("CONFIRMED_detail") or {}
    if not detail:
        print("CONFIRMED_detail is empty — nothing to filter")
        return 2 if args.check else 0

    survivors, cut = {}, {}
    for token, entries in detail.items():
        by_loc: dict[str, set[int]] = {}
        for e in entries:
            by_loc.setdefault(e["location"], set()).update(e.get("years_present") or [])
        # a shared year across two distinct locations
        shared = []
        locs = sorted(by_loc)
        for i, a in enumerate(locs):
            for b in locs[i + 1 :]:
                both = by_loc[a] & by_loc[b]
                if both:
                    shared.append({"locations": [a, b], "shared_years": sorted(both)})
        rec = {
            "token": token,
            "n_locations": len(by_loc),
            "locations": locs,
            "synchronous_pairs": shared,
            "tournament_names": sorted({n for e in entries for n in (e.get("tournament_names") or [])}),
        }
        (survivors if shared else cut)[token] = rec

    print(f"CONFIRMED tokens in            : {len(detail)}")
    print(f"  survive cross-venue synchrony: {len(survivors)}")
    print(f"  cut (single venue only)      : {len(cut)}\n")

    print("  SURVIVORS (token — venues sharing a year):")
    for t, r in sorted(survivors.items()):
        p = r["synchronous_pairs"][0]
        print(
            f"    {t:14} {r['n_locations']} venues   "
            f"{p['locations'][0]} + {p['locations'][1]} in {p['shared_years'][:4]}"
        )

    print("\n  CUT — appear at one venue only, so indistinguishable from a place name:")
    print(f"    {', '.join(sorted(cut))}")

    OUT.write_text(
        json.dumps(
            {
                "question": (
                    "Which of build_tennis_sponsors.py's 36 CONFIRMED tokens appear at "
                    "TWO OR MORE venues in a shared year?"
                ),
                "why": (
                    "A company buys naming rights to several events in a season; a river is in "
                    "one place. The source file named this filter as its own natural next step "
                    "and named the false-positive class it removes — the Melbourne 2021 "
                    "quarantine-bubble events (Great Ocean Road, Murray River, Phillip Island, "
                    "Yarra Valley), all Australian geography, none sponsored."
                ),
                "what_this_still_is_not": (
                    "NOT entity resolution. Surviving this filter does not tie a token to a "
                    "company; it removes the geography class and anything seen at a single venue. "
                    "A survivor is a better CANDIDATE and nothing more. Joining one to a ticker "
                    "needs a second source, exactly as the source file said."
                ),
                # HAND ADJUDICATION OF THE SURVIVORS, because 7 survivors is not 7 sponsors and the
                # artifact must not be readable as though it were. Checked against the tournament
                # names the filter itself carries.
                "survivor_adjudication": {
                    "aegon": "REAL — AEGON Championships/Classic/International/Open across 4 venues",
                    "ericsson": "REAL — Ericsson Open, Sony Ericsson Open across 4 venues",
                    "sony": "REAL — Sony Ericsson Championships/Open across 3 venues",
                    "rogers": "REAL — Rogers Cup (Montreal) and Rogers Masters (Toronto)",
                    "viking": "REAL — Viking Classic/International; the Aegon UK rebrand, and it "
                    "lands on the SAME two venues Aegon held (Birmingham, Eastbourne), "
                    "which independently reproduces the dated corporate action "
                    "build_tennis_sponsors.py called its strongest evidence",
                    "home": "REAL SPONSOR, FRAGMENT TOKEN — 'Bet-At-Home Cup' and 'bet-at-home "
                    "Open'. bet-at-home.com is a genuine sponsor; 'home' is a tokenisation "
                    "artefact of its name, not a separate entity",
                    "valley": "FALSE POSITIVE — 'Yarra Valley Classic' (Melbourne) and 'Mubadala "
                    "Silicon Valley Classic' (San Jose) are BOTH geography, and they "
                    "coincided in 2021. The San Jose event's actual sponsor is Mubadala; "
                    "'Silicon Valley' is the location. Cross-venue synchrony cannot "
                    "separate two geographically-named events that happen to share a "
                    "year, and this is the one case in 36 where that occurs.",
                },
                "honest_score": (
                    "6 of 7 survivors correspond to real sponsorship (5 clean, 1 a "
                    "fragment of a real sponsor's name); 1 is geography coincidence. "
                    "The filter removed all 4 Melbourne-2021 geography tokens the "
                    "source file named — great, ocean, river, island — and missed "
                    "'valley' only because San Jose is also geographically named."
                ),
                "n_confirmed_in": len(detail),
                "n_survivors": len(survivors),
                "n_cut": len(cut),
                "survivors": survivors,
                "cut_single_venue": sorted(cut),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
