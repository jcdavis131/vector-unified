#!/usr/bin/env python3
"""Tennis feature matrix — player x season, masked, MTNN-ready.

Solo personal project, no connection to employer, built with public/free-tier only

Mirrors vector-pitch's `tm_full.npz` contract: X (n, f) features, M (n, f) mask, plus a
row-aligned meta list. The mask is not optional — a player who never played on grass has no
grass win rate, and 7.32 established the cost of treating an unobserved value as a zero.

WHAT TENNIS CAN AND CANNOT SUPPLY, measured rather than hoped. tennis-data.co.uk gives
match RESULTS: scores, ranks, surface, round, tier, odds. It does NOT give serve or shot
statistics — no aces, no first-serve percentage, no winners, no unforced errors. So the
classic "big server vs baseline grinder" style axis is **not derivable from this source**,
and this file does not fake one.

What IS derivable is arguably more interesting for a cross-sport fold, because it is
behavioural rather than mechanical:

    surface splits          hard / clay / grass win rates. Clay-courter vs grass-courter is
                            the real archetype division in tennis and it comes straight
                            out of results.
    specialisation          max surface WR minus mean surface WR — a SPECIALIST/GENERALIST
                            axis, which is the one thing here that plainly has cross-sport
                            analogues (a situational pass-rusher, a three-point specialist).
    upset rate              win rate against better-ranked opponents
    hold rate               win rate against worse-ranked opponents. Together with the
                            above these separate "beats who he should" from "steals games
                            he shouldn't", which is a genuine temperament signal.
    straight_sets_rate      dominance when winning
    decider_rate            share of matches going the full distance — the grinder axis
    games_won_pct           margin, not just outcome
    draw_progress_mean      tournament results, per operator direction
    big_event_share         Slam + Masters/WTA1000 share of schedule — ambition/access
    indoor_wr               conditions split
    retire_rate             durability; retirements are already counted per entity

EVERY FEATURE IS A RATE, so a player with three matches and a player with sixty are on the
same scale — and MIN_MATCHES exists because a rate over three matches is noise wearing a
number. The count rides on the meta row so a consumer can weight by it.

    python pipeline/build_tennis_matrix.py
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acquire_tennis import YEARS, path_for, read_sheet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTM = ROOT / "pipeline" / "data" / "tennis_matrix.npz"
OUTJ = ROOT / "pipeline" / "data" / "meta_tennis_matrix.json"
OUTR = ROOT / "data" / "tennis_matrix_report.json"

MIN_MATCHES = 8            # a rate over fewer is noise wearing a number
MIN_SURFACE_MATCHES = 3    # below this a surface WR is masked out, not zeroed
RANK_CAP = 500.0
BIG = {"Grand Slam", "Masters 1000", "WTA1000", "Masters Cup", "Tour Championships"}

FEATURES = [
    "WIN_RATE", "HARD_WR", "CLAY_WR", "GRASS_WR", "SURFACE_SPECIALISATION",
    "UPSET_RATE", "HOLD_RATE", "STRAIGHT_SETS_RATE", "DECIDER_RATE",
    "GAMES_WON_PCT", "DRAW_PROGRESS_MEAN", "BIG_EVENT_SHARE", "INDOOR_WR",
    "RETIRE_RATE", "MEAN_OPP_RANK_LOG", "ENTERING_RANK_LOG",
]


def num(v, d=None):
    try:
        f = float(v)
        return f if f == f else d
    except (TypeError, ValueError):
        return d


def collect(women: bool) -> dict[tuple, dict]:
    acc: dict[tuple, dict] = collections.defaultdict(lambda: {
        "m": 0, "w": 0, "gf": 0.0, "ga": 0.0, "straight": 0, "decider": 0,
        "vs_better": 0, "vs_better_w": 0, "vs_worse": 0, "vs_worse_w": 0,
        "indoor": 0, "indoor_w": 0, "big": 0, "retire": 0,
        "surf": collections.defaultdict(lambda: [0, 0]),
        "opp": [], "own": [],
    })
    for y in YEARS:
        p = path_for(y, women)
        if not p.exists():
            continue
        hdr, body = read_sheet(p)
        i = {c: k for k, c in enumerate(hdr)}
        if not all(c in i for c in ("Winner", "Loser", "WRank", "LRank")):
            continue
        for r in body:
            surf = str(r[i["Surface"]]).strip() if "Surface" in i else ""
            court = str(r[i["Court"]]).strip() if "Court" in i else ""
            tier = str(r[i.get("Series", i.get("Tier", 0))]).strip() if (
                "Series" in i or "Tier" in i) else ""
            comment = str(r[i["Comment"]]).strip() if "Comment" in i else ""
            wr, lr = num(r[i["WRank"]], RANK_CAP), num(r[i["LRank"]], RANK_CAP)
            ws, ls = num(r[i["Wsets"]], 0) or 0, num(r[i["Lsets"]], 0) or 0
            gw = sum(num(r[i[c]], 0) or 0 for c in ("W1", "W2", "W3", "W4", "W5") if c in i)
            gl = sum(num(r[i[c]], 0) or 0 for c in ("L1", "L2", "L3", "L4", "L5") if c in i)
            bo = num(r[i["Best of"]], 3) if "Best of" in i else 3
            went_distance = (ws + ls) >= (5 if bo == 5 else 3)
            straight = ls == 0

            for name, won, own, opp, gf, ga in (
                (str(r[i["Winner"]]).strip(), 1, wr, lr, gw, gl),
                (str(r[i["Loser"]]).strip(), 0, lr, wr, gl, gw),
            ):
                if not name:
                    continue
                d = acc[(name, y, "wta" if women else "atp")]
                d["m"] += 1
                d["w"] += won
                d["gf"] += gf
                d["ga"] += ga
                if won and straight:
                    d["straight"] += 1
                if went_distance:
                    d["decider"] += 1
                if own is not None and opp is not None:
                    if opp < own:
                        d["vs_better"] += 1
                        d["vs_better_w"] += won
                    elif opp > own:
                        d["vs_worse"] += 1
                        d["vs_worse_w"] += won
                    d["opp"].append(opp)
                    d["own"].append(own)
                if surf:
                    d["surf"][surf][0] += 1
                    d["surf"][surf][1] += won
                if court == "Indoor":
                    d["indoor"] += 1
                    d["indoor_w"] += won
                if tier in BIG:
                    d["big"] += 1
                if comment == "Retired":
                    d["retire"] += 1
    return acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    prog = {}
    ent_p = ROOT / "data" / "tennis_entities.json"
    if ent_p.exists():
        agg: dict[tuple, list] = collections.defaultdict(list)
        for e in json.loads(ent_p.read_text(encoding="utf-8"))["entities"]:
            agg[(e["player"], e["year"], e["tour"])].append(e["draw_progress"])
        prog = {k: statistics.mean(v) for k, v in agg.items() if v}

    acc = {}
    for women in (False, True):
        acc.update(collect(women))

    rows, meta = [], []
    for (name, y, tour), d in sorted(acc.items()):
        if d["m"] < MIN_MATCHES:
            continue
        f = [np.nan] * len(FEATURES)
        m = [0] * len(FEATURES)

        def put(k, v):
            if v is None:
                return
            j = FEATURES.index(k)
            f[j] = float(v)
            m[j] = 1

        put("WIN_RATE", d["w"] / d["m"])
        surf_wrs = {}
        for s, key in (("Hard", "HARD_WR"), ("Clay", "CLAY_WR"), ("Grass", "GRASS_WR")):
            n, w = d["surf"].get(s, [0, 0])
            # MASKED, not zeroed. A player who never played grass has no grass win rate,
            # and calling it 0.0 says he lost every grass match he never played.
            if n >= MIN_SURFACE_MATCHES:
                put(key, w / n)
                surf_wrs[s] = w / n
        if len(surf_wrs) >= 2:
            vals = list(surf_wrs.values())
            put("SURFACE_SPECIALISATION", max(vals) - (sum(vals) / len(vals)))
        if d["vs_better"]:
            put("UPSET_RATE", d["vs_better_w"] / d["vs_better"])
        if d["vs_worse"]:
            put("HOLD_RATE", d["vs_worse_w"] / d["vs_worse"])
        if d["w"]:
            put("STRAIGHT_SETS_RATE", d["straight"] / d["w"])
        put("DECIDER_RATE", d["decider"] / d["m"])
        if d["gf"] + d["ga"] > 0:
            put("GAMES_WON_PCT", d["gf"] / (d["gf"] + d["ga"]))
        if (name, y, tour) in prog:
            put("DRAW_PROGRESS_MEAN", prog[(name, y, tour)])
        put("BIG_EVENT_SHARE", d["big"] / d["m"])
        if d["indoor"] >= MIN_SURFACE_MATCHES:
            put("INDOOR_WR", d["indoor_w"] / d["indoor"])
        put("RETIRE_RATE", d["retire"] / d["m"])
        if d["opp"]:
            put("MEAN_OPP_RANK_LOG", -np.log1p(statistics.median(d["opp"])))
        if d["own"]:
            put("ENTERING_RANK_LOG", -np.log1p(statistics.median(d["own"])))

        rows.append(f)
        meta.append({"player": name, "year": y, "tour": tour, "matches": d["m"],
                     "wins": d["w"]})

    raw = np.array(rows, dtype=np.float64)
    # M is derived from WHERE THE VALUE IS NaN, before nan_to_num destroys the distinction.
    # Order matters: computing the mask after zero-filling would mark every zeroed hole as
    # observed, which is precisely the failure this mask exists to prevent.
    M = (~np.isnan(raw)).astype(np.int64)
    X = np.nan_to_num(raw, nan=0.0).astype(np.float32)
    OUTM.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTM, X=X, M=M, features=np.array(FEATURES))
    OUTJ.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    cov = {FEATURES[j]: int(M[:, j].sum()) for j in range(len(FEATURES))}
    report = {
        "matrix": str(OUTM.relative_to(ROOT)),
        "rows": int(X.shape[0]), "features": len(FEATURES),
        "by_tour": dict(collections.Counter(m["tour"] for m in meta)),
        "min_matches": MIN_MATCHES, "min_surface_matches": MIN_SURFACE_MATCHES,
        "feature_coverage": cov,
        "feature_coverage_pct": {k: round(100.0 * v / max(X.shape[0], 1), 1)
                                 for k, v in cov.items()},
        "mask_note": (
            "M is 1 where the feature was OBSERVED. Surface win rates are masked below "
            f"{MIN_SURFACE_MATCHES} matches on that surface rather than set to 0.0 — a "
            "player who never played grass did not lose every grass match. 7.32 is the "
            "precedent: an unobserved zero contaminated the pitch composite until the mask "
            "was applied."),
        "not_derivable": (
            "No serve or shot statistics exist in this source — no aces, first-serve "
            "percentage, winners or unforced errors. The classic big-server vs "
            "baseline-grinder style axis is therefore NOT derivable here and is not faked. "
            "What is derivable is behavioural: surface specialisation, upset vs hold rate, "
            "decider rate, margin."),
    }
    OUTR.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"rows {X.shape[0]}   features {len(FEATURES)}   {report['by_tour']}")
    for k, v in report["feature_coverage_pct"].items():
        print(f"  {k:24} {cov[k]:>6}  {v:>5.1f}%")
    print(f"\nwrote {OUTM}\nwrote {OUTJ}\nwrote {OUTR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
