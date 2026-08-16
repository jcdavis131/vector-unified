#!/usr/bin/env python3
"""Export gridiron draft pedigree as a name-joinable artifact, mirroring hoops'.

Solo personal project, no connection to employer, built with public/free-tier only

vector-gridiron HAS the expectation side — `pipeline/cache/draft_picks.csv` (12,928 rows
from nflverse) feeds a `pedigree` feature family, and build_adp.py's own docstring
describes flagging "where the model likes a player more than the room is paying for him",
which IS the T0/T1 concept. But it is consumed INSIDE the feature pipeline and never
emitted as an artifact anything else can join, so the unified model could not use it.

This writes `data/gridiron_pedigree.json` with the same shape hoops' pedigree.json uses,
so `build_trajectory_axis.py` can treat the two sports identically.

EXPECTATION IS `expect_log`, NOT AN INVENTED DOLLAR CURVE. hoops uses an anchor-
interpolated CBA rookie-scale shape (#1 = 1.00, round 2 = 0.10, undrafted = 0.06). I do
not have NFL rookie-wage-scale dollars, and inventing anchors that look plausible would
put a fabricated constant inside a result. Instead:

    expect_log = 1 - log1p(overall) / log1p(MAX_PICK)

Monotone in draft position, convex (early picks separate more than late ones, which is
the real shape), and it reuses the log transform vector-gridiron's OWN feature pipeline
already applies (`draft_pick_log = log1p(pick)`). No constant here is chosen by me beyond
MAX_PICK, which is a structural fact of the draft.

`expect_slot`-style anchors are deliberately absent. If NFL rookie-scale dollars are ever
acquired, add them as a separate field rather than replacing this one, so any result
computed on `expect_log` stays reproducible.

NAME JOIN IS SCOPED TO GRIDIRON BY THE CONSUMER, not here. Recorded because it bit once:
joining hoops pedigree across the whole unified set matched 16 gridiron and 1 pitch
athlete and ALL 17 were false positives (NFL Matt Ryan onto an NBA Matt Ryan).

    python pipeline/export_gridiron_pedigree.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from math import log1p
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# THE ONE GRIDIRON NORMALISER. This file used to carry a private copy that stripped
# `jr|sr|ii|iii|iv|v` unconditionally, which merged Marvin Harrison with Marvin Harrison Jr.
# — and, worse, disagreed with the VOR table that keys off the same names. `gridiron_
# pedigree.json`'s KEYS are norm_names, so a private copy here silently re-keys the whole
# artifact relative to every consumer of it. Import, never re-implement.
from build_vor_draft_value import norm_name

ROOT = Path(__file__).resolve().parent.parent
GRIDIRON = Path("C:/Users/jcdav/vector-gridiron")
DRAFT_CSV = GRIDIRON / "pipeline" / "cache" / "draft_picks.csv"
VECTORS = GRIDIRON / "assets" / "vectors.json"
OUT = ROOT / "data" / "gridiron_pedigree.json"

MAX_PICK = 262.0  # deepest modern 7-round draft; structural, not tuned


def expect_log(overall: int) -> float:
    return max(0.0, 1.0 - log1p(overall) / log1p(MAX_PICK))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not DRAFT_CSV.exists():
        print(f"missing {DRAFT_CSV} — run vector-gridiron's nfl_data cache first")
        return 2

    # ---- draft rows ----------------------------------------------------------
    picks: dict[str, dict] = {}
    dupes = 0
    with DRAFT_CSV.open(encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("pfr_player_name") or "").strip()
            if not name:
                continue
            try:
                overall = int(float(row["pick"]))
                rnd = int(float(row["round"]))
                season = int(float(row["season"]))
            except (KeyError, TypeError, ValueError):
                continue
            key = norm_name(name)
            if key in picks:
                # A repeated name is a real collision (two different players). Keep the
                # EARLIER pick and count it, rather than letting the later row silently
                # overwrite a Hall of Famer with a 7th-rounder of the same name.
                dupes += 1
                if picks[key]["overall"] <= overall:
                    continue
            picks[key] = {
                "name": name,
                "overall": overall,
                "round": rnd,
                "pick": overall,
                "draft_year": season,
                "team": (row.get("team") or "").strip(),
                "position": (row.get("position") or "").strip(),
                "undrafted": False,
                "expect_log": round(expect_log(overall), 4),
            }

    # ---- coverage against the athletes that actually exist -------------------
    vec = json.loads(VECTORS.read_text(encoding="utf-8"))
    athletes = {norm_name(p["name"]): p.get("pos") for p in vec["players"]}
    matched = {k: v for k, v in picks.items() if k in athletes}
    unmatched_athletes = [k for k in athletes if k not in picks]

    # Undrafted is a REAL category, not a gap: an athlete in the vector set with no draft
    # row went undrafted (or predates the cache). It is emitted so the low-expectation
    # tail is populated rather than silently dropped, which is where T1 lives.
    for k in unmatched_athletes:
        matched[k] = {
            "name": k,
            "overall": None,
            "round": None,
            "pick": None,
            "draft_year": None,
            "team": "",
            "position": athletes.get(k) or "",
            "undrafted": True,
            "expect_log": 0.0,
        }

    by_pos = collections.Counter(athletes[k] for k in matched if athletes.get(k))
    n_drafted = sum(1 for v in matched.values() if not v["undrafted"])

    report = {
        "draft_rows_read": len(picks) + dupes,
        "distinct_drafted_names": len(picks),
        "name_collisions_kept_earlier_pick": dupes,
        "athletes_in_vector_set": len(athletes),
        "athletes_with_a_draft_row": n_drafted,
        "pct_drafted": round(100.0 * n_drafted / len(athletes), 2) if athletes else 0.0,
        "athletes_marked_undrafted": len(athletes) - n_drafted,
        "by_position": dict(by_pos.most_common()),
        "expectation_field": "expect_log = 1 - log1p(overall)/log1p(262); no invented dollar anchors",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "built_from": str(DRAFT_CSV),
                "shape": "mirrors vector-hoops/assets/pedigree.json so both sports score identically",
                "caveat_scope": (
                    "vector-gridiron's vector set is OFFENSIVE SKILL POSITIONS only "
                    "(QB/RB/WR/TE) — all 18 features are pass/rush/receiving. Linemen "
                    "and defenders are absent, so any gridiron trajectory label covers "
                    "fantasy-relevant positions, not the roster."
                ),
                "caveat_join": (
                    "Scope the join to sport == 'gridiron'. Joining hoops pedigree "
                    "across the whole unified set produced 17 matches, all false."
                ),
                "report": report,
                "players": {k: v for k, v in sorted(matched.items())},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(
        f"draft rows: {report['distinct_drafted_names']} distinct names " f"({dupes} collisions, kept the earlier pick)"
    )
    print(f"athletes in the gridiron vector set : {report['athletes_in_vector_set']}")
    print(f"  with a draft row                  : {n_drafted} ({report['pct_drafted']}%)")
    print(f"  marked undrafted                  : {report['athletes_marked_undrafted']}")
    print(f"  by position                       : {report['by_position']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
