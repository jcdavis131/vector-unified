#!/usr/bin/env python3
"""Can the tennis ranking carry an R0/R1 axis? The circularity has to be beaten first.

Solo personal project, no connection to employer, built with public/free-tier only

`acquire_tennis.py` measured the corpus: 67,081 matches, ranking prior on 100.0%, closing
odds on 99.9%. This asks the next question in the only order that is safe — **is the prior
actually prior?**

THE TRAP, AND IT IS SPECIFIC TO THIS SPORT. A draft slot is fixed before a player takes a
professional snap: it cannot be contaminated by the performance it is used to predict. An
ATP/WTA ranking is **computed from results**. Correlating a player's in-season rank against
that same season's win rate is close to correlating results with themselves, and it would
produce a large, meaningless number that looks exactly like a finding. Every wrong table in
Phase 7 looked publishable; this one would look better than most.

    NOT USED  rank at match time      -> contaminated by the match being scored
    USED      rank at END of season t-1 -> delivery in season t

The lag is what makes it an expectation. It is the same reason `build_trajectory_axis.py`
uses draft slot rather than career-to-date production.

CLOSING ODDS ARE DELIBERATELY NOT THE PRIOR EITHER, though they are genuinely pre-match.
A closing price is a near-optimal forecast of the very outcome being measured, so
corr(odds, result) is high by construction and says nothing about over- or
under-performance. Odds belong in this corpus as the **baseline a model must beat**, not as
the expectation an axis is built against. Recorded here so the temptation is answered once.

PRE-REGISTERED READING, fixed before the first run:

  * corr(lagged rank prior, next-season delivery) is computed per tour and pooled.
  * |corr| < NEAR_ZERO  -> the lagged ranking carries nothing and R0/R1 is not worth
    assigning; report and stop, exactly as the pitch age axis was prepared to do.
  * A SANITY CONTRAST is required in the same run: corr(SAME-season rank, delivery). If the
    lagged number is not materially smaller, the lag did not remove what it was supposed to
    and the result must not be trusted.

    python pipeline/probe_tennis_expectation.py
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acquire_tennis import YEARS, path_for, read_sheet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "tennis_expectation_probe.json"

MIN_MATCHES = 12        # per player-season, before a win rate means anything
NEAR_ZERO = 0.05
RANK_CAP = 500.0        # unranked/blank -> treated as this, and the count is reported


def season_records(women: bool) -> dict[tuple[str, int], dict]:
    """(player, year) -> {matches, wins, sets_for, sets_against, end_rank}."""
    rec: dict[tuple[str, int], dict] = collections.defaultdict(
        lambda: {"m": 0, "w": 0, "sf": 0, "sa": 0, "ranks": []})
    for y in YEARS:
        p = path_for(y, women)
        if not p.exists():
            continue
        hdr, body = read_sheet(p)
        i = {c: k for k, c in enumerate(hdr)}
        need = ("Winner", "Loser", "WRank", "LRank", "Wsets", "Lsets")
        if not all(c in i for c in need):
            continue
        for r in body:
            w, lo = str(r[i["Winner"]]).strip(), str(r[i["Loser"]]).strip()
            if not w or not lo:
                continue
            try:
                ws, ls = float(r[i["Wsets"]] or 0), float(r[i["Lsets"]] or 0)
            except ValueError:
                ws = ls = 0.0
            for name, won, sf, sa, rk in ((w, 1, ws, ls, r[i["WRank"]]),
                                          (lo, 0, ls, ws, r[i["LRank"]])):
                d = rec[(name, y)]
                d["m"] += 1
                d["w"] += won
                d["sf"] += sf
                d["sa"] += sa
                try:
                    v = float(rk)
                    if v > 0:
                        d["ranks"].append(v)
                except (TypeError, ValueError):
                    pass
    return rec


def corr(xs, ys) -> float:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    return statistics.correlation(xs, ys)


def analyse(rec: dict, tour: str) -> dict:
    # expectation = log of the player's MEDIAN rank in season t-1 (lower rank = better, so
    # negate to make "higher expectation" mean "expected to be better", matching the sign
    # convention of `1 - log1p(pick)/log1p(MAX_PICK)` in the draft axes).
    rows, unranked = [], 0
    for (name, y), d in rec.items():
        if d["m"] < MIN_MATCHES:
            continue
        prev = rec.get((name, y - 1))
        if not prev or prev["m"] < MIN_MATCHES:
            continue
        if prev["ranks"]:
            prior_rank = statistics.median(prev["ranks"])
        else:
            prior_rank = RANK_CAP
            unranked += 1
        same_rank = statistics.median(d["ranks"]) if d["ranks"] else RANK_CAP
        rows.append({
            "player": name, "year": y, "matches": d["m"],
            "win_rate": d["w"] / d["m"],
            "set_ratio": d["sf"] / max(d["sf"] + d["sa"], 1),
            "prior_expect": -math.log1p(prior_rank),
            "same_expect": -math.log1p(same_rank),
        })
    if len(rows) < 100:
        return {"tour": tour, "n": len(rows), "verdict": "TOO FEW ROWS"}

    lag_wr = corr([r["prior_expect"] for r in rows], [r["win_rate"] for r in rows])
    lag_sr = corr([r["prior_expect"] for r in rows], [r["set_ratio"] for r in rows])
    same_wr = corr([r["same_expect"] for r in rows], [r["win_rate"] for r in rows])
    shrink = (1 - abs(lag_wr) / abs(same_wr)) if same_wr and not math.isnan(same_wr) else float("nan")
    return {
        "tour": tour, "n": len(rows),
        "player_seasons_unranked_prior": unranked,
        "corr_lagged_rank_vs_win_rate": round(lag_wr, 4),
        "corr_lagged_rank_vs_set_ratio": round(lag_sr, 4),
        "corr_SAME_season_rank_vs_win_rate": round(same_wr, 4),
        "shrinkage_from_lagging": (None if math.isnan(shrink) else round(shrink, 3)),
        "verdict": ("NOT ASSIGNABLE — lagged prior carries almost nothing"
                    if abs(lag_wr) < NEAR_ZERO else "USABLE"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out = {}
    for women, tour in ((False, "atp"), (True, "wta")):
        rec = season_records(women)
        out[tour] = analyse(rec, tour)
        print(f"  {tour} player-seasons scored: {out[tour].get('n')}", flush=True)

    report = {
        "question": "Does a LAGGED tennis ranking predict next-season delivery?",
        "why_lagged": (
            "An ATP/WTA ranking is computed FROM results, so correlating in-season rank "
            "against that season's win rate is close to correlating results with "
            "themselves. A draft slot cannot be contaminated that way; a ranking can. The "
            "prior is therefore the median rank in season t-1, used against delivery in "
            "season t."),
        "why_not_odds": (
            "Closing odds are genuinely pre-match but are a near-optimal forecast of the "
            "very outcome being measured, so corr(odds, result) is high by construction. "
            "Odds belong here as the BASELINE A MODEL MUST BEAT, not as the expectation an "
            "axis is built against."),
        "min_matches_per_player_season": MIN_MATCHES,
        "unranked_prior_treated_as": RANK_CAP,
        "per_tour": out,
        "sanity_contrast": (
            "corr_SAME_season_rank_vs_win_rate is reported beside the lagged figure on "
            "purpose. If lagging does not materially shrink the correlation, the lag did "
            "not remove the contamination it exists to remove and neither number should be "
            "trusted."),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{'tour':6}{'n':>7}{'lag vs win':>12}{'lag vs sets':>13}"
          f"{'SAME vs win':>13}{'shrink':>9}  verdict")
    for tour, d in out.items():
        if "corr_lagged_rank_vs_win_rate" not in d:
            print(f"{tour:6}{d.get('n', 0):>7}   {d['verdict']}")
            continue
        print(f"{tour:6}{d['n']:>7}{d['corr_lagged_rank_vs_win_rate']:>+12.4f}"
              f"{d['corr_lagged_rank_vs_set_ratio']:>+13.4f}"
              f"{d['corr_SAME_season_rank_vs_win_rate']:>+13.4f}"
              f"{str(d['shrinkage_from_lagging']):>9}  {d['verdict']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
