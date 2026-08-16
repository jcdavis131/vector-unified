#!/usr/bin/env python3
"""Does a tennis embedding have anything to beat? Baseline retrieval on the raw features.

Solo personal project, no connection to employer, built with public/free-tier only

Tennis is the one domain in this estate with NO model — 4,022 player-year-tour rows x 16
features at 91.6% observed, and nothing trained on it. The obvious next step is an MTNN
mirroring hoops and equities.

THIS FILE EXISTS TO ASK WHETHER THAT IS WORTH DOING, BEFORE DOING IT. Two reasons to check
first rather than build first:

  * 16 features into a 64-d embedding is EXPANSION, not compression. hoops compresses 142
    features and equities 118; there is no obvious representational win in inflating 16.
  * This session measured two feature improvements that produced no detectable gain, and a
    training instability that made a third unmeasurable. Building a model and then asking
    whether it helped has been the expensive order.

THE TASK, matched to how hoops and equities are scored: retrieve the SAME PLAYER's adjacent
year. Rank every other row by cosine and ask whether the player's own next-year row is in
the top 10. If the raw 16 features already do this well, a learned embedding has little to
win and the honest recommendation is not to train one.

MASK IS RESPECTED. X carries 0.0 where a feature was unobserved, so the mask column is
concatenated alongside the values rather than letting a structural zero read as a
measurement — the same rule build_tennis_forward.py uses.

WITHIN-TOUR ONLY. ATP and WTA rows are separate populations with separate ranking scales;
retrieving across them would be a different and easier task (the tour itself is a giveaway),
so candidates are restricted to the same tour.

    python pipeline/probe_tennis_retrieval.py
    python pipeline/probe_tennis_retrieval.py --check   # exit 1 if the run is broken
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "pipeline" / "data" / "tennis_matrix.npz"
META = ROOT / "pipeline" / "data" / "meta_tennis_matrix.json"
OUT = ROOT / "data" / "tennis_retrieval_probe.json"
K = 10


def recall_at_k(E, pairs, tours, k=K):
    """Share of query rows whose adjacent-year self lands in the top k, within tour."""
    if len(pairs) == 0:
        return float("nan")
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    hits = 0
    for a, b in pairs:
        same = np.where(tours == tours[a])[0]
        same = same[same != a]
        sims = En[same] @ En[a]
        order = same[np.argsort(-sims)]
        if b in order[:k]:
            hits += 1
    return hits / len(pairs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not MATRIX.exists():
        print(f"missing {MATRIX}")
        return 2
    a = np.load(MATRIX, allow_pickle=True)
    X, M = a["X"].astype(np.float64), a["M"].astype(np.float64)
    feats = [str(f) for f in a["features"]]
    meta = json.loads(META.read_text(encoding="utf-8"))
    if len(meta) != X.shape[0]:
        print(
            f"meta has {len(meta)} rows, matrix has {X.shape[0]} — refusing to join by "
            f"position across a length mismatch"
        )
        return 2

    tours = np.array([m["tour"] for m in meta])
    idx = {(m["player"], m["year"], m["tour"]): i for i, m in enumerate(meta)}
    pairs = [
        (i, idx[(m["player"], m["year"] + 1, m["tour"])])
        for i, m in enumerate(meta)
        if (m["player"], m["year"] + 1, m["tour"]) in idx
    ]

    print(f"rows {X.shape[0]}   features {len(feats)}   observed {100*M.mean():.1f}%")
    print(f"adjacent-year pairs (same tour): {len(pairs)}")
    for t in sorted(set(tours.tolist())):
        n = sum(1 for i, _ in pairs if tours[i] == t)
        print(f"  {t}: {n} pairs, {int((tours == t).sum())} rows")

    # value+mask block, the same construction build_tennis_forward.py uses
    F = np.hstack([X, M])
    # z-score per column so no single scale dominates the cosine
    mu, sd = F.mean(axis=0), F.std(axis=0)
    sd[sd == 0] = 1.0
    Fz = (F - mu) / sd

    variants = {
        "raw_values_only": X,
        "values_plus_mask": F,
        "zscored_values_plus_mask": Fz,
    }
    rows = {}
    for name, E in variants.items():
        r = recall_at_k(E, pairs, tours)
        rows[name] = round(float(r), 4)
        print(f"\n  recall@{K}  {name:26} {r:.4f}")

    # RANDOM BASELINE, computed rather than asserted. With n candidates in a tour, a random
    # ranking puts the target in the top k with probability k/n. Averaged over the actual
    # per-query candidate counts, not over a nominal corpus size.
    rnd = []
    for a_i, _ in pairs:
        n_cand = int((tours == tours[a_i]).sum()) - 1
        rnd.append(min(1.0, K / max(1, n_cand)))
    random_floor = float(np.mean(rnd)) if rnd else float("nan")
    print(f"\n  random floor (k/n per query, measured)  {random_floor:.4f}")

    best = max(rows.values())
    lift = best / random_floor if random_floor else float("nan")
    print(f"  best variant beats random by {lift:.1f}x")

    verdict = (
        "RAW FEATURES ALREADY RETRIEVE WELL — a learned embedding has little headroom here"
        if best >= 0.60
        else "RAW FEATURES RETRIEVE POORLY — there is headroom a learned embedding could take"
        if best < 0.30
        else "MIDDLING — raw features carry real signal but leave room; a model is worth trying "
        "and must be judged against this number, not against the random floor"
    )
    print(f"\n  verdict: {verdict}")

    OUT.write_text(
        json.dumps(
            {
                "question": (
                    "Before training a tennis embedding: how well do the raw 16 features "
                    f"already retrieve a player's adjacent year at recall@{K}?"
                ),
                "why_ask_first": (
                    "16 features into a 64-d embedding is EXPANSION, not compression — hoops "
                    "compresses 142 and equities 118. And this session measured two feature "
                    "improvements with no detectable gain plus an instability that made a third "
                    "unmeasurable, so building first and asking later has been the expensive order."
                ),
                "task": (
                    f"rank all same-tour rows by cosine against a query row; hit if that "
                    f"player's next-year row is in the top {K}"
                ),
                "within_tour_only": (
                    "ATP and WTA are separate populations with separate ranking "
                    "scales; retrieving across them would be an easier task "
                    "because the tour itself is a giveaway"
                ),
                "mask_note": (
                    "X carries 0.0 where unobserved, so the mask is concatenated "
                    "alongside the values rather than letting a structural zero read as "
                    "a measurement — same rule as build_tennis_forward.py"
                ),
                "rows": int(X.shape[0]),
                "features": len(feats),
                "n_pairs": len(pairs),
                "observed_pct": round(100 * float(M.mean()), 1),
                "recall_at_k": rows,
                "k": K,
                "random_floor_measured": round(random_floor, 4),
                "random_floor_note": (
                    "k/n per query using each query's actual same-tour candidate "
                    "count, averaged — not k over a nominal corpus size"
                ),
                "best_over_random": round(lift, 2),
                "verdict": verdict,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {OUT}")

    if args.check and (not pairs or np.isnan(best)):
        print("FAIL no usable pairs")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
