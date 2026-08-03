#!/usr/bin/env python3
"""D0/D1 — career DIRECTION (rising / declining), split out of the trajectory axis.

Solo personal project, no connection to employer, built with public/free-tier only

T0-T3 were declared as one "trajectory" axis. Reading their own definitions shows two
different questions wearing one label:

    T0  "High draft slot / market value, low return"   standing vs EXPECTATION
    T1  "Late-round / undrafted standout producer"     (career-level, vs draft slot)

    T2  "Career arc up; form/volume jumping"           DIRECTION over TIME
    T3  "Career arc down; load/role shrinking"         (within-career, early vs late)

A player can be over-delivering against his draft slot AND declining right now. Nikola
Jokic at 34 would be T1 and T3 at once. One axis makes those mutually exclusive by
construction, which is the same category error already caught twice in this estate: role
vs trajectory (590cd0d), and role-change-pattern vs performance-vs-expectation (b75f11a).
Third time, same shape, so the split is applied rather than argued.

    STANDING   T0 / T1  — assigned, unchanged, see build_trajectory_axis.py
    DIRECTION  D0 / D1  — this file

WHY A NEW PREFIX RATHER THAN REUSING T2/T3. Same reasoning as the A->T renumbering: a
distinct prefix makes it obvious at a glance that the axes are NOT mutually exclusive, and
a stale consumer asking for "T2" gets a clean key error instead of silently reading a
direction label as a standing label.

METHOD. Each career's per-season VOR series is split into halves; direction is the change
in MEAN VOR from first half to second. Mean rather than sum, so a long career does not
read as "rising" merely for being long. Tails only — the middle of this distribution is
genuinely "about the same", and forcing it into D0/D1 would manufacture a finding.

VOR IS FLOORED AT ZERO, which matters here. A player below replacement in both halves has
delta 0 and is unlabelled — correct, because "bad then bad" is not a direction. It does
mean D0/D1 describe players who were ABOVE replacement at some point, which is a narrower
population than "all careers" and is reported as such.

    python pipeline/build_direction_axis.py --sport hoops
    python pipeline/build_direction_axis.py --sport gridiron
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = {"hoops": ROOT / "data" / "hoops_vor_draft_value.json",
       "gridiron": ROOT / "data" / "vor_draft_value.json"}
OUT = {"hoops": ROOT / "data" / "direction_axis_hoops.json",
       "gridiron": ROOT / "data" / "direction_axis_gridiron.json"}

MIN_SEASONS = 4     # >=2 per half
TAIL_PCT = 20.0


def pct_rank(sorted_vals: list[float], v: float) -> float:
    below = sum(1 for x in sorted_vals if x < v)
    return 100.0 * below / max(len(sorted_vals) - 1, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sport", choices=("hoops", "gridiron"), default="hoops")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    src = SRC[args.sport]
    if not src.exists():
        print(f"missing {src} — build the VOR value table for {args.sport} first")
        return 2
    doc = json.loads(src.read_text(encoding="utf-8"))
    players = doc.get("players") or []
    if not players:
        print(f"{src} carries no per-player rows — re-run its builder")
        return 2

    # The VOR tables store a per-player TOTAL, not the season series, so the series is
    # rebuilt here from the same source the table used. Recomputing rather than reusing
    # a total keeps this file honest about what it is measuring: within-career change,
    # which a total cannot express.
    series = _series(args.sport)
    if not series:
        print("could not rebuild the per-season VOR series")
        return 2

    rows = []
    dropped = collections.Counter()
    for p in players:
        s = sorted(series.get(p["name"], ()))
        if len(s) < MIN_SEASONS:
            dropped["too_few_seasons"] += 1
            continue
        vals = [v for _y, v in s]
        half = len(vals) // 2
        first, second = statistics.mean(vals[:half]), statistics.mean(vals[half:])
        if first == 0.0 and second == 0.0:
            dropped["never_above_replacement"] += 1
            continue
        rows.append({"name": p["name"], "seasons": len(vals),
                     "first_half": round(first, 2), "second_half": round(second, 2),
                     "delta": round(second - first, 2),
                     "overall": p.get("overall"), "pos": p.get("pos")})

    if len(rows) < 50:
        print(f"only {len(rows)} careers with a usable series — not assigning.")
        return 2

    deltas = sorted(r["delta"] for r in rows)
    counts = collections.Counter()
    for r in rows:
        pr = pct_rank(deltas, r["delta"])
        r["delta_pct"] = round(pr, 1)
        r["direction"] = ("D0" if pr >= 100.0 - TAIL_PCT else
                          "D1" if pr <= TAIL_PCT else None)
        counts[r["direction"] or "unlabelled"] += 1

    ranked = sorted(rows, key=lambda r: -r["delta"])
    report = {
        "sport": args.sport,
        "axis": "DIRECTION (D0 rising / D1 declining) — separate from STANDING (T0/T1)",
        "careers_scored": len(rows),
        "dropped": dict(dropped),
        "min_seasons": MIN_SEASONS,
        "tail_pct": TAIL_PCT,
        "counts": dict(counts),
        "population_caveat": (
            "VOR is floored at zero, so a career spent entirely below replacement has "
            "delta 0 and is dropped, not labelled — 'bad then bad' is not a direction. "
            "D0/D1 therefore describe careers that were above replacement at some point, "
            "a narrower population than 'all careers'."),
        "orthogonality_note": (
            "D is INDEPENDENT of T0/T1 by design. A player may be T1 (over-delivered "
            "against his draft slot) and D1 (declining now) at once; the previous "
            "single-axis encoding made that inexpressible."),
        "D0_examples": [{"name": r["name"], "pick": r["overall"], "pos": r.get("pos"),
                         "first": r["first_half"], "second": r["second_half"],
                         "delta": r["delta"]}
                        for r in ranked if r["direction"] == "D0"][:8],
        "D1_examples": [{"name": r["name"], "pick": r["overall"], "pos": r.get("pos"),
                         "first": r["first_half"], "second": r["second_half"],
                         "delta": r["delta"]}
                        for r in reversed(ranked) if r["direction"] == "D1"][:8],
    }
    OUT[args.sport].write_text(
        json.dumps({"report": report, "careers": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"{args.sport}: {len(rows)} careers scored   dropped {dict(dropped)}")
    print(f"D0 rising {counts['D0']}   D1 declining {counts['D1']}   "
          f"unlabelled {counts['unlabelled']}\n")
    print("D0 — rising (mean VOR, first half -> second):")
    for e in report["D0_examples"][:6]:
        print(f"  pick {str(e['pick']):>4}  {e['first']:>7.2f} -> {e['second']:>7.2f}"
              f"  ({e['delta']:+.2f})  {e['name']}")
    print("\nD1 — declining:")
    for e in report["D1_examples"][:6]:
        print(f"  pick {str(e['pick']):>4}  {e['first']:>7.2f} -> {e['second']:>7.2f}"
              f"  ({e['delta']:+.2f})  {e['name']}")
    print(f"\nwrote {OUT[args.sport]}")
    return 0


def _series(sport: str) -> dict[str, list[tuple[int, float]]]:
    """Per-season floored VOR, rebuilt with the same rules the value tables used."""
    import importlib.util
    import sys

    name = ("build_hoops_vor_draft_value" if sport == "hoops"
            else "build_vor_draft_value")
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_dir_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    if sport == "hoops":
        # ONE implementation, imported. This function used to re-derive the series here
        # and therefore missed the eligibility filter when it was added upstream — the
        # axis kept surfacing a one-eligible-season player as the biggest riser in the
        # league after the bug was "fixed".
        vec = json.loads(mod.VECTORS.read_text(encoding="utf-8"))
        seasons = sorted({str(p["season"]) for p in vec["players"]}, key=mod.season_start)
        series, _ = mod.vor_series(seasons, mod.eligible_pairs(vec))
        return series

    vec = json.loads(mod.GRID_VEC.read_text(encoding="utf-8"))["players"]
    pool: dict[tuple, list[float]] = collections.defaultdict(list)
    for p in vec:
        ppr = (p.get("ppg") or {}).get("ppr")
        if ppr is None or (p.get("games") or 0) < mod.MIN_GAMES:
            continue
        if p.get("pos") in mod.REPLACEMENT_RANK:
            pool[(p["season"], p["pos"])].append(float(ppr))
    rep = {}
    for k, v in pool.items():
        v.sort(reverse=True)
        rep[k] = v[min(mod.REPLACEMENT_RANK[k[1]], len(v)) - 1]
    out = collections.defaultdict(list)
    for p in vec:
        ppr = (p.get("ppg") or {}).get("ppr")
        pos = p.get("pos")
        if ppr is None or pos not in mod.REPLACEMENT_RANK:
            continue
        b = rep.get((p["season"], pos))
        if b is None:
            continue
        out[mod.norm_name(p["name"])].append((int(p["season"]), max(0.0, float(ppr) - b)))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
