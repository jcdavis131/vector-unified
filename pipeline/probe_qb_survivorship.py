#!/usr/bin/env python3
"""Is QB's flat draft-slot correlation caused by survivorship? Test the denominator.

Solo personal project, no connection to employer, built with public/free-tier only

7.7c found that among gridiron careers with >=4 fantasy-relevant seasons, draft slot
predicts delivery at RB (+0.57), TE (+0.48), WR (+0.47) — and NOT at QB (+0.12, 95% CI
[-0.05, +0.29], spans zero). Restricted range was tested and refuted: QB has the WIDEST
draft-expectation spread of any position.

The remaining explanation is SURVIVORSHIP, and it has a specific signature. If a high-pick
QB who busts is benched before accumulating four seasons, then draft slot buys SURVIVAL
rather than production, the surviving high-pick QBs are the ones who worked out, and the
correlation among survivors flattens. That is a truncation artefact, not a fact about
quarterbacks.

THE TEST NEEDS THE DENOMINATOR, which is the whole point. `vectors.json` contains only
players who accumulated fantasy-relevant seasons — survivors. `draft_picks.csv` contains
EVERY pick, including the ones who vanished. Joining them gives the attrition the survivor
table cannot show.

    survival = drafted player reaches >= MIN_SEASONS charted fantasy seasons

    If draft slot predicts SURVIVAL much more strongly at QB than at RB, the survivorship
    explanation is supported: slot is doing its work at the selection stage instead of
    the production stage.

    If survival is slot-dependent at ALL positions roughly equally, survivorship does not
    single out QB and the flat QB correlation needs a different explanation.

DRAFT-YEAR WINDOW IS LOAD-BEARING. A 2024 draftee has not had four seasons to accumulate,
so counting them as "did not survive" measures the calendar, not attrition. Only draft
years with at least MIN_SEASONS of elapsed opportunity are scored, and the window is
printed so the cut is visible rather than assumed.

    python pipeline/probe_qb_survivorship.py
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
GRIDIRON = Path("C:/Users/jcdav/vector-gridiron")
DRAFT_CSV = GRIDIRON / "pipeline" / "cache" / "draft_picks.csv"
VECTORS = GRIDIRON / "assets" / "vectors.json"
OUT = ROOT / "data" / "qb_survivorship_probe.json"

MIN_SEASONS = 4          # same floor the trajectory axis uses
POSITIONS = ("QB", "RB", "WR", "TE")
MAX_PICK = 262.0
BUCKETS = [(1, 32, "R1"), (33, 64, "R2"), (65, 105, "R3"), (106, 262, "R4-7")]


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'\u2019-]", "", s.lower())
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s.strip())
    return re.sub(r"\s+", " ", s)


def bucket(pick: int) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= pick <= hi:
            return name
    return "R4-7"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    for p in (DRAFT_CSV, VECTORS):
        if not p.exists():
            print(f"missing {p}")
            return 2

    vec = json.loads(VECTORS.read_text(encoding="utf-8"))["players"]
    seasons_avail = sorted({int(p["season"]) for p in vec})
    first_season, last_season = seasons_avail[0], seasons_avail[-1]

    # survivor side: name -> number of charted seasons
    charted: dict[str, int] = collections.Counter()
    for p in vec:
        charted[norm_name(p["name"])] += 1

    # denominator side: every drafted player at a fantasy position, in-window
    max_draft_year = last_season - MIN_SEASONS + 1
    rows = []
    seen: set[str] = set()
    with DRAFT_CSV.open(encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            pos = (r.get("position") or "").strip().upper()
            if pos not in POSITIONS:
                continue
            name = (r.get("pfr_player_name") or "").strip()
            if not name:
                continue
            try:
                pick = int(float(r["pick"]))
                year = int(float(r["season"]))
            except (KeyError, TypeError, ValueError):
                continue
            # A player drafted before the vector set begins cannot show 4 seasons in it
            # for reasons of coverage, not attrition.
            if year < first_season or year > max_draft_year:
                continue
            key = norm_name(name)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "name": key, "pos": pos, "pick": pick, "year": year,
                "bucket": bucket(pick),
                "expect_log": max(0.0, 1.0 - log1p(pick) / log1p(MAX_PICK)),
                "seasons": charted.get(key, 0),
                "survived": charted.get(key, 0) >= MIN_SEASONS,
            })

    if not rows:
        print("no drafted players in window — check the caches")
        return 2

    by_pos: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_pos[r["pos"]].append(r)

    per_position = {}
    for pos in POSITIONS:
        rs = by_pos.get(pos) or []
        if len(rs) < 40:
            continue
        xs = [r["expect_log"] for r in rs]
        ys = [1.0 if r["survived"] else 0.0 for r in rs]
        corr = statistics.correlation(xs, ys) if len(set(ys)) > 1 else 0.0
        buckets = {}
        for _lo, _hi, bname in BUCKETS:
            sub = [r for r in rs if r["bucket"] == bname]
            if sub:
                buckets[bname] = {
                    "drafted": len(sub),
                    "survived": sum(1 for r in sub if r["survived"]),
                    "rate": round(100.0 * sum(1 for r in sub if r["survived"]) / len(sub), 1),
                }
        per_position[pos] = {
            "drafted_in_window": len(rs),
            "survived": sum(1 for r in rs if r["survived"]),
            "survival_rate": round(100.0 * sum(1 for r in rs if r["survived"]) / len(rs), 1),
            "corr_slot_vs_survival": round(corr, 4),
            "by_bucket": buckets,
        }

    qb = per_position.get("QB", {})
    rb = per_position.get("RB", {})
    qc, rc = qb.get("corr_slot_vs_survival", 0.0), rb.get("corr_slot_vs_survival", 0.0)
    supported = qc > rc + 0.10

    report = {
        "min_seasons": MIN_SEASONS,
        "draft_year_window": [first_season, max_draft_year],
        "window_note": (f"Only draft years {first_season}-{max_draft_year} are scored. A "
                        f"player drafted later has not had {MIN_SEASONS} seasons of "
                        f"opportunity, so counting them as attrition would measure the "
                        f"calendar."),
        "per_position": per_position,
        "verdict": (
            "SURVIVORSHIP SUPPORTED at QB: draft slot predicts SURVIVAL more strongly at "
            "quarterback than at running back, so slot is doing its work at the selection "
            "stage and the flat correlation among survivors is a truncation artefact."
            if supported else
            "SURVIVORSHIP NOT SUPPORTED as a QB-specific explanation: draft slot predicts "
            "survival at QB no more strongly than at RB, so differential attrition does "
            "not single out quarterbacks and the flat QB correlation among survivors needs "
            "a different explanation."),
        "verdict_is_directional": (
            "This compares two correlations without a CI on the difference, so it is a "
            "direction, not a significance test. It answers 'does the survivorship story "
            "even point the right way', which is the question that was open."),
    }

    OUT.write_text(json.dumps({"report": report, "players": rows}, indent=2,
                              ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"draft years scored: {first_season}-{max_draft_year}  "
          f"(need {MIN_SEASONS} seasons of opportunity; vector set ends {last_season})\n")
    print(f"{'pos':4} {'drafted':>8} {'survived':>9} {'rate':>7} {'corr slot->survival':>20}")
    for pos in POSITIONS:
        v = per_position.get(pos)
        if not v:
            continue
        print(f"{pos:4} {v['drafted_in_window']:>8} {v['survived']:>9} "
              f"{v['survival_rate']:>6.1f}% {v['corr_slot_vs_survival']:>+20.4f}")
    print(f"\n{'pos':4} " + "  ".join(f"{b:>12}" for _l, _h, b in BUCKETS))
    for pos in POSITIONS:
        v = per_position.get(pos)
        if not v:
            continue
        cells = []
        for _l, _h, b in BUCKETS:
            d = v["by_bucket"].get(b)
            cells.append(f"{d['rate']:>5.1f}% ({d['drafted']:>3})" if d else f"{'-':>12}")
        print(f"{pos:4} " + "  ".join(cells))
    print(f"\nVERDICT: {report['verdict']}")
    print(f"\nNOTE: {report['verdict_is_directional']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
