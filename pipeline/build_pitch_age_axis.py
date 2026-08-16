#!/usr/bin/env python3
"""P0/P1 — pitch performance vs AGE expectation. WITHIN-PITCH ONLY. (7.32)

Solo personal project, no connection to employer, built with public/free-tier only

7.9 established what free sources give pitch: age at context on 84.5% of rows, and NO
market prior at all. This builds the only expectation axis those sources support, and the
prefix is deliberately P rather than T because it is NOT the same axis as hoops/gridiron
T0/T1:

    T0 / T1   standing vs a MARKET valuation made before the performance (draft slot)
    P0 / P1   standing vs an AGE expectation (what a player of this age typically produces)

A draft slot says what the market paid to acquire someone. Age says where they are in a
development curve. Both are priors, they are not the same prior, and 7.7b showed that
comparing two constructs under one name reversed a whole cross-sport finding twice — once
in each direction. **P0/P1 must never be compared against T0/T1.** The distinct prefix
means a consumer asking for "T0" on a pitch row gets a KeyError instead of an answer to a
different question.

DELIVERY IS SCORED WITHIN (context, position), and that is not optional. The 16 features
are per-90 rates — GOALS_P90, TACKLES_P90, PRESSURES_P90 — and a defender scored on goals
is not being measured, he is being mislabelled. Gridiron hit this exact problem in 7.7:
un-normalised PPR ranked every quarterback above every tight end. Within-cell z-scoring
makes "good for a defender at Euro 2024" the unit.

THE MASK IS APPLIED. tm_full.npz ships M alongside X because not every feature is observed
for every row; an unobserved zero is not a zero performance. Only masked-in features
contribute to a row's composite, and a row with fewer than MIN_FEATURES observed is
dropped rather than scored on a handful.

PRE-REGISTERED READING, fixed before the first run:

  * If corr(age, delivery) is near zero, an age-based expectation carries almost no
    information and P0/P1 is not worth assigning — report that and stop, rather than
    labelling tails of a residual that is just the delivery distribution again.
  * P0 = produced far ABOVE the age curve, P1 = far BELOW, tails only at TAIL_PCT.
  * The residual is from a per-position age curve, because a 33-year-old centre-back and a
    33-year-old winger are not at the same point in their curves.

    python pipeline/build_pitch_age_axis.py
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PITCH = Path("C:/Users/jcdav/vector-pitch")
MATRIX = PITCH / "pipeline" / "data" / "tm_full.npz"
META = PITCH / "pipeline" / "data" / "meta_tm_full.json"
EMB = PITCH / "assets" / "pitch_mtnn_embeddings.json"
AGES = ROOT / "data" / "pitch_expectation_sources.json"
OUT = ROOT / "data" / "pitch_age_axis.json"

MIN_FEATURES = 8  # of 16; below this a composite is a few rates, not a profile
MIN_CELL = 12  # rows needed in a (context, pos) cell before z-scoring it
MIN_AGE_CELL = 15  # rows needed in a position before fitting its age curve
TAIL_PCT = 20.0
NEAR_ZERO = 0.05  # |corr| below this = age carries almost nothing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    for p in (MATRIX, META, EMB, AGES):
        if not p.exists():
            print(f"missing {p}")
            return 2

    z = np.load(MATRIX, allow_pickle=True)
    X, Mk = z["X"], z["M"]
    meta = json.loads(META.read_text(encoding="utf-8"))
    emb = json.loads(EMB.read_text(encoding="utf-8"))["players"]

    # ALIGNMENT IS ASSERTED, NOT ASSUMED. Three files are being joined by ROW INDEX with
    # no shared key. If they ever drift the join silently attaches one player's features to
    # another's name — the same class of error as the mask-as-index bug, and just as
    # invisible in the output.
    if not (len(meta) == X.shape[0] == len(emb)):
        print(f"ROW COUNT MISMATCH: matrix {X.shape[0]}, meta {len(meta)}, emb {len(emb)}")
        return 2
    bad = [i for i in range(len(meta)) if meta[i]["name"] != emb[i]["name"] or meta[i]["context"] != emb[i]["context"]]
    if bad:
        print(
            f"ROW ALIGNMENT MISMATCH at {len(bad)} index/es, first: "
            f"meta={meta[bad[0]]['name']!r}/{meta[bad[0]]['context']!r} vs "
            f"emb={emb[bad[0]]['name']!r}/{emb[bad[0]]['context']!r}"
        )
        return 2

    resolved = json.loads(AGES.read_text(encoding="utf-8"))["resolved"]

    # ---- delivery: within-(context, pos) z-score, masked ----------------------
    cells: dict[tuple, list[int]] = collections.defaultdict(list)
    for i, m in enumerate(meta):
        cells[(m["context"], m["pos"])].append(i)

    delivery: dict[int, float] = {}
    thin_cells = 0
    for key, idx in cells.items():
        if len(idx) < MIN_CELL:
            thin_cells += 1
            continue
        sub, sm = X[idx], Mk[idx].astype(bool)
        for f in range(X.shape[1]):
            col = sub[:, f][sm[:, f]]
            if len(col) < MIN_CELL or float(np.std(col)) == 0.0:
                sm[:, f] = False
                continue
            mu, sd = float(np.mean(col)), float(np.std(col))
            sub[:, f] = (sub[:, f] - mu) / sd
        for r, i in enumerate(idx):
            obs = sm[r]
            if int(obs.sum()) < MIN_FEATURES:
                continue
            delivery[i] = float(np.mean(sub[r][obs]))

    # ---- expectation: age at context -----------------------------------------
    rows = []
    for i, m in enumerate(meta):
        if i not in delivery:
            continue
        r = resolved.get(m["name"])
        if not r or not r.get("dob"):
            continue
        dob, ctx = r["dob"], m["context"]
        yr = int(dob[:4]) if dob[:4].isdigit() else None
        cy = next((int(t) for t in ctx.split() if t.isdigit() and len(t) == 4), None)
        if cy is None:
            cy = int(ctx[-7:-3]) if ctx[-7:-3].isdigit() else None
        if not (yr and cy):
            continue
        rows.append(
            {
                "name": m["name"],
                "context": ctx,
                "pos": m["pos"],
                "age": cy - yr,
                "delivery": round(delivery[i], 4),
            }
        )

    if len(rows) < 200:
        print(f"only {len(rows)} scorable rows — not assigning.")
        return 2

    ages = [r["age"] for r in rows]
    dels = [r["delivery"] for r in rows]
    corr = statistics.correlation(ages, dels) if len(set(ages)) > 1 else 0.0

    report: dict = {
        "axis": "P0/P1 — pitch delivery vs AGE expectation. WITHIN-PITCH ONLY.",
        "not_comparable_to": (
            "T0/T1 in hoops and gridiron. Those measure standing against a MARKET "
            "valuation (draft slot); this measures standing against a DEVELOPMENTAL prior. "
            "7.9 established no free market prior exists for football. Comparing the two "
            "would repeat the construct mismatch that reversed the cross-sport draft "
            "finding twice in 7.7b."
        ),
        "rows_scorable": len(rows),
        "rows_total": len(meta),
        "pct_scorable": round(100.0 * len(rows) / len(meta), 1),
        "thin_cells_dropped": thin_cells,
        "corr_age_delivery": round(corr, 4),
        "age_range": [min(ages), max(ages)],
    }

    if abs(corr) < NEAR_ZERO:
        report["verdict"] = (
            f"NOT ASSIGNED. corr(age, delivery) = {corr:+.4f}, below the pre-registered "
            f"|{NEAR_ZERO}| floor, so an age expectation carries almost no information "
            f"here. Labelling tails of this residual would be labelling the delivery "
            f"distribution again under a new name — the residual IS delivery when the "
            f"predictor explains nothing. Reported rather than assigned."
        )
        OUT.write_text(
            json.dumps({"report": report, "rows": rows}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"scorable {len(rows)}/{len(meta)} ({report['pct_scorable']}%)   " f"corr(age, delivery) = {corr:+.4f}")
        print(f"\n{report['verdict']}")
        print(f"\nwrote {OUT}")
        return 0

    # ---- residual against a PER-POSITION age curve ---------------------------
    by_pos: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_pos[r["pos"]].append(r)
    scored = []
    for pos, grp in by_pos.items():
        if len(grp) < MIN_AGE_CELL:
            continue
        a = np.array([r["age"] for r in grp], dtype=float)
        d = np.array([r["delivery"] for r in grp], dtype=float)
        slope, intercept = np.polyfit(a, d, 1)
        for r, resid in zip(grp, d - (slope * a + intercept), strict=True):
            scored.append(
                {
                    **r,
                    "expected": round(float(slope * r["age"] + intercept), 4),
                    "residual": round(float(resid), 4),
                }
            )

    # ---- TEAM-STRENGTH CONFOUND, measured before the axis is believed ---------
    # The first run's tails were not subtle: P0 read Neymar (Brazil), Wirtz (Germany),
    # Alba (Spain), Paredes (Argentina); P1 read Azmoun and Pouraliganji (Iran), Dykes
    # (Scotland), Araujo (Peru). Delivery is z-scored within (context, position) but NOT
    # within TEAM, so a player on a weaker side posts worse per-90 rates at every age and
    # the residual inherits it. Leave-one-out teammate mean is the cheapest honest proxy
    # for team strength; if the residual tracks it, this axis is measuring the team.
    team_of = {(m["name"], m["context"]): m.get("team", "") for m in meta}
    team_groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in scored:
        r["team"] = team_of.get((r["name"], r["context"]), "")
        team_groups[(r["context"], r["team"])].append(r)
    loo_x, loo_y = [], []
    for grp in team_groups.values():
        if len(grp) < 5:
            continue
        tot = sum(g["delivery"] for g in grp)
        for g in grp:
            loo = (tot - g["delivery"]) / (len(grp) - 1)
            g["teammate_mean_delivery"] = round(loo, 4)
            loo_x.append(loo)
            loo_y.append(g["residual"])
    corr_team = statistics.correlation(loo_x, loo_y) if len(loo_x) > 2 and len(set(loo_x)) > 1 else float("nan")
    corr_age_only = corr
    report["team_strength_confound"] = {
        "n_pairs": len(loo_x),
        "corr_residual_vs_teammate_mean": (None if corr_team != corr_team else round(corr_team, 4)),
        "corr_age_vs_delivery_for_scale": round(corr_age_only, 4),
        "note": (
            "Leave-one-out teammate mean delivery within (context, team) is a proxy "
            "for team strength. If the residual tracks it more strongly than age "
            "tracks delivery, P0/P1 is largely a team-quality ranking wearing an "
            "age-adjustment label, and the tails should be read that way."
        ),
    }
    report["team_strength_confound"]["verdict"] = (
        "UNMEASURED"
        if corr_team != corr_team
        else "DOMINATES — the residual tracks team strength more strongly than age tracks "
        "delivery. Read P0/P1 as team-quality-contaminated."
        if abs(corr_team) > abs(corr_age_only)
        else "present but smaller than the age signal it is competing with"
    )

    # ---- CONTROL FOR IT, do not merely report it -----------------------------
    # corr(residual, teammate mean) = +0.263 against corr(age, delivery) = -0.167: the
    # confound is LARGER than the signal it contaminates. Reporting that and shipping the
    # uncontrolled axis anyway would be the 7.11 mistake — measure a confound, name it,
    # then quote the number it invalidates. So age and team strength are fitted TOGETHER
    # per position and the axis is the residual from both. Rows without a teammate mean
    # (squads under 5 scorable players) are dropped rather than imputed.
    ctrl = [r for r in scored if "teammate_mean_delivery" in r]
    by_pos2: dict[str, list[dict]] = collections.defaultdict(list)
    for r in ctrl:
        by_pos2[r["pos"]].append(r)
    controlled = []
    for pos, grp in by_pos2.items():
        if len(grp) < MIN_AGE_CELL:
            continue
        A = np.column_stack(
            [
                [r["age"] for r in grp],
                [r["teammate_mean_delivery"] for r in grp],
                np.ones(len(grp)),
            ]
        )
        d = np.array([r["delivery"] for r in grp], dtype=float)
        beta, *_ = np.linalg.lstsq(A, d, rcond=None)
        for r, resid in zip(grp, d - A @ beta, strict=True):
            r["residual_uncontrolled"] = r["residual"]
            r["residual"] = round(float(resid), 4)
            controlled.append(r)
    if len(controlled) >= 200:
        scored = controlled
        cx = [r["teammate_mean_delivery"] for r in scored]
        cy = [r["residual"] for r in scored]
        after = statistics.correlation(cx, cy) if len(set(cx)) > 1 else float("nan")
        report["team_strength_confound"]["corr_after_control"] = None if after != after else round(after, 4)
        report["team_strength_confound"]["control_note"] = (
            f"Age and teammate-mean delivery are now fitted together per position. "
            f"corr(residual, teammate mean) fell from "
            f"{report['team_strength_confound']['corr_residual_vs_teammate_mean']} to "
            f"{None if after != after else round(after, 4)}. "
            f"{len(controlled)} of {len(ctrl)} rows survive; squads with fewer than 5 "
            f"scorable players have no teammate mean and are dropped rather than imputed. "
            f"NOTE: -0.0 here is ORTHOGONALITY BY CONSTRUCTION, not evidence — OLS "
            f"residuals are orthogonal to their own predictors, so this check can no "
            f"longer detect a residual team effect and must not be read as proving one is "
            f"absent. What it does confirm is that the fit ran. The evidence that the "
            f"control mattered is in the TAILS: before it, P1 was Iran and Peru players "
            f"almost exclusively; after it, P1 includes Lisandro Martinez (Argentina) and "
            f"Scamacca (Italy) — players who underperformed their age curve relative to "
            f"their own strong teammates."
        )

    res = sorted(r["residual"] for r in scored)

    def pct_rank(v):
        return 100.0 * sum(1 for x in res if x < v) / max(len(res) - 1, 1)

    counts: collections.Counter = collections.Counter()
    for r in scored:
        pr = pct_rank(r["residual"])
        r["residual_pct"] = round(pr, 1)
        r["axis"] = "P0" if pr >= 100.0 - TAIL_PCT else "P1" if pr <= TAIL_PCT else None
        counts[r["axis"] or "unlabelled"] += 1

    ranked = sorted(scored, key=lambda r: -r["residual"])
    report.update(
        {
            "verdict": "ASSIGNED",
            "counts": dict(counts),
            "tail_pct": TAIL_PCT,
            "P0_examples": [
                {
                    "name": r["name"],
                    "age": r["age"],
                    "pos": r["pos"],
                    "context": r["context"],
                    "delivery": r["delivery"],
                    "expected": r["expected"],
                    "residual": r["residual"],
                }
                for r in ranked
                if r["axis"] == "P0"
            ][:8],
            "P1_examples": [
                {
                    "name": r["name"],
                    "age": r["age"],
                    "pos": r["pos"],
                    "context": r["context"],
                    "delivery": r["delivery"],
                    "expected": r["expected"],
                    "residual": r["residual"],
                }
                for r in reversed(ranked)
                if r["axis"] == "P1"
            ][:8],
        }
    )
    OUT.write_text(
        json.dumps({"report": report, "rows": scored}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(
        f"scorable {len(rows)}/{len(meta)} ({report['pct_scorable']}%)   "
        f"corr(age, delivery) = {corr:+.4f}   age {min(ages)}-{max(ages)}"
    )
    print(f"P0 {counts['P0']}   P1 {counts['P1']}   unlabelled {counts['unlabelled']}\n")
    print("P0 — above the age curve:")
    for e in report["P0_examples"][:6]:
        print(f"  {e['age']:>3}y {e['pos']:<4} {e['residual']:>+7.3f}  {e['name']} " f"({e['context']})")
    print("\nP1 — below the age curve:")
    for e in report["P1_examples"][:6]:
        print(f"  {e['age']:>3}y {e['pos']:<4} {e['residual']:>+7.3f}  {e['name']} " f"({e['context']})")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
