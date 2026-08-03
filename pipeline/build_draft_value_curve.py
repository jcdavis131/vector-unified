#!/usr/bin/env python3
"""Expected value of a draft pick = P(survive | slot) x E[delivery | survive, slot].

Solo personal project, no connection to employer, built with public/free-tier only

Two halves are measured and have never been multiplied:

    probe_qb_survivorship.py   P(survive | position, bucket)  — from the DENOMINATOR,
                               every pick in draft_picks.csv, 1999-2022
    build_trajectory_axis.py   E[delivery | survive, ...]     — survivors only, career-mean
                               PPR percentile within (season, position)

Each answers a different question, and neither is the one a draft-value model wants.
Conditional delivery says what a pick is worth GIVEN IT WORKED OUT — which is exactly the
quantity 7.7c mistook for a fact about quarterbacks before the survivorship probe showed
that at QB the slot does its work at the selection stage. Survival says how often a pick
works out and nothing about how good it is when it does.

The product is the unconditional expectation, which is what "what is this pick worth"
actually means.

    EV(position, bucket) = survival_rate x mean_delivery_of_survivors

THE MODELLING CHOICE, stated because it is a choice and not a fact: a player who does not
reach MIN_SEASONS contributes ZERO. That is right for a fantasy horizon — a QB who starts
six games and disappears returns nothing over four years — and it is wrong if you care
about trade value or a two-year window. Anyone using this for a different horizon needs a
different zero.

BUCKETS, NOT PER-PICK. Per-pick estimates over 291 quarterbacks would be noise dressed as
a curve. R1/R2/R3/R4-7 is the granularity both inputs actually support, and the cell
counts are printed so a thin cell is visible rather than smoothed over.

    python pipeline/build_draft_value_curve.py
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SURV = ROOT / "data" / "qb_survivorship_probe.json"
TRAJ = ROOT / "data" / "trajectory_axis_gridiron.json"
PEDIGREE = ROOT / "data" / "gridiron_pedigree.json"
OUT = ROOT / "data" / "draft_value_curve.json"

BUCKETS = [(1, 32, "R1"), (33, 64, "R2"), (65, 105, "R3"), (106, 262, "R4-7")]
POSITIONS = ("QB", "RB", "WR", "TE")
MIN_CELL = 8


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

    for p in (SURV, TRAJ):
        if not p.exists():
            print(f"missing {p} — run probe_qb_survivorship.py and "
                  f"build_trajectory_axis.py --sport gridiron first")
            return 2

    surv = json.loads(SURV.read_text(encoding="utf-8"))
    traj = json.loads(TRAJ.read_text(encoding="utf-8"))["careers"]

    # THE TWO HALVES MUST COME FROM THE SAME POOL, and the first version of this script
    # did not enforce it. Survival is measured over draft years 1999-2022 (everyone had
    # 4 seasons of opportunity); delivery was measured over EVERY survivor, draft years
    # 1980-2025. Multiplying them produced TE R1 with 25 drafted and 28 survivors scored
    # — more survivors than draftees, which is impossible and is what exposed it. 144 of
    # 951 delivery careers (15.1%) were out of window.
    if not PEDIGREE.exists():
        print(f"missing {PEDIGREE} — needed to window the delivery side by draft year")
        return 2
    ped = json.loads(PEDIGREE.read_text(encoding="utf-8"))["players"]
    lo_year, hi_year = surv["report"]["draft_year_window"]

    deliv: dict[tuple, list[float]] = collections.defaultdict(list)
    out_of_window = 0
    for r in traj:
        if r.get("undrafted"):
            continue  # undrafted has no slot; it is a separate row below
        yr = (ped.get(r["name"]) or {}).get("draft_year")
        if yr is None or not (lo_year <= yr <= hi_year):
            out_of_window += 1
            continue
        b = bucket(r.get("overall"))
        if b:
            deliv[(r.get("position") or "?", b)].append(r["delivery"])

    undrafted = [r["delivery"] for r in traj if r.get("undrafted")]

    per = surv["report"]["per_position"]
    rows = []
    for pos in POSITIONS:
        v = per.get(pos)
        if not v:
            continue
        for _lo, _hi, b in BUCKETS:
            cell = v["by_bucket"].get(b)
            ds = deliv.get((pos, b)) or []
            if not cell or len(ds) < MIN_CELL:
                rows.append({"pos": pos, "bucket": b,
                             "drafted": cell["drafted"] if cell else 0,
                             "survival_pct": cell["rate"] if cell else None,
                             "survivors_scored": len(ds),
                             "cond_delivery": None, "ev": None,
                             "thin": True})
                continue
            cond = statistics.mean(ds)
            rows.append({
                "pos": pos, "bucket": b,
                "drafted": cell["drafted"],
                "survival_pct": cell["rate"],
                "survivors_scored": len(ds),
                "cond_delivery": round(cond, 1),
                "ev": round(cell["rate"] / 100.0 * cond, 1),
                "thin": False,
            })

    scored = [r for r in rows if not r["thin"]]

    # Does accounting for bust risk REORDER anything? That is the question the product
    # is for — a pick that looks good conditional on working out is not the same as a
    # pick that is good.
    by_cond = [r["pos"] + " " + r["bucket"]
               for r in sorted(scored, key=lambda x: -x["cond_delivery"])]
    by_ev = [r["pos"] + " " + r["bucket"]
             for r in sorted(scored, key=lambda x: -x["ev"])]
    moved = [c for c in by_cond if by_cond.index(c) != by_ev.index(c)]

    report = {
        "definition": "EV = P(survive | pos, bucket) x E[delivery percentile | survive]",
        "zero_choice": ("A player who does not reach 4 fantasy-relevant seasons "
                        "contributes ZERO. Right for a 4-year fantasy horizon, wrong for "
                        "trade value or a 2-year window."),
        "min_cell": MIN_CELL,
        "draft_year_window": [lo_year, hi_year],
        "delivery_careers_dropped_out_of_window": out_of_window,
        "pool_alignment_note": (
            "Both halves are now restricted to the SAME draft-year window. They were not "
            "in the first version, and the symptom was TE R1 reporting 25 drafted and 28 "
            "survivors scored — more survivors than draftees. 144 of 951 delivery careers "
            "were out of window."),
        "cross_position_caveat": (
            "EV is comparable WITHIN a position across buckets. Comparing ACROSS positions "
            "requires assuming a delivery percentile at QB is worth what one is at RB, and "
            "it is not: delivery is a percentile WITHIN (season, position), so each column "
            "is scored against its own pool. Positional scarcity and the fact that one QB "
            "starts per team are outside this number entirely."),
        "cells": rows,
        "ranking_conditional": by_cond,
        "ranking_expected_value": by_ev,
        "cells_that_moved": moved,
        "undrafted_reference": {
            "survivors_scored": len(undrafted),
            "cond_delivery": round(statistics.mean(undrafted), 1) if undrafted else None,
            "note": ("Undrafted players have no slot, so they carry no EV row — their "
                     "survival denominator is every undrafted free agent ever signed, "
                     "which draft_picks.csv does not contain. Conditional delivery only."),
        },
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("EV = P(survive) x E[delivery | survive]   (delivery = PPR pct within season+pos)\n")
    print(f"{'pos':4} {'bucket':7} {'drafted':>8} {'surv%':>7} {'cond':>7} {'EV':>7}  n_surv")
    for r in rows:
        if r["thin"]:
            print(f"{r['pos']:4} {r['bucket']:7} {r['drafted']:>8} "
                  f"{(str(r['survival_pct']) + '%') if r['survival_pct'] is not None else '-':>7} "
                  f"{'thin':>7} {'-':>7}  {r['survivors_scored']}")
        else:
            print(f"{r['pos']:4} {r['bucket']:7} {r['drafted']:>8} {r['survival_pct']:>6.1f}% "
                  f"{r['cond_delivery']:>7.1f} {r['ev']:>7.1f}  {r['survivors_scored']}")
    print("\ntop 6 by CONDITIONAL delivery (what a pick is worth if it works out):")
    print("  " + " > ".join(by_cond[:6]))
    print("top 6 by EXPECTED VALUE (what a pick is worth):")
    print("  " + " > ".join(by_ev[:6]))
    print(f"\ncells whose rank changes once bust risk is priced in: {len(moved)} of {len(scored)}")
    u = report["undrafted_reference"]
    print(f"\nundrafted survivors: n={u['survivors_scored']}, conditional delivery "
          f"{u['cond_delivery']} — no EV row, see note")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
