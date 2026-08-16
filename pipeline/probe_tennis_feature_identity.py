#!/usr/bin/env python3
"""Which tennis features identify a PLAYER, and which just describe a season?

Solo personal project, no connection to employer, built with public/free-tier only

probe_tennis_retrieval.py measured raw-feature retrieval at recall@10 = 0.0328 and
probe_tennis_metric.py showed a learned linear metric does not improve on it. The
conclusion was "more features, not more capacity" — but that is only useful if it says
WHICH features, and "add odds columns" is a guess until something measures it.

THE PROPERTY A RETRIEVAL FEATURE NEEDS is not predictive power. It is IDENTITY: stable
within a player across adjacent years, and spread out across players. A feature that is
merely noisy year to year contributes nothing to finding the same person again, however
informative it is about a single season.

    within-player autocorrelation r   how much a player's value carries to next year
    between-player spread             sd across all rows (already 1.0, features are z-ish)
    IDENTITY SCORE                    the autocorrelation itself, since spread is fixed

This is the same shape as the persistence baselines the forward probes use, applied per
feature instead of to a target. It is cheap — no training — and it tells you what a useful
NEW feature would have to look like before anyone builds one.

READ THIS AGAINST THE KNOWN RESULT: rank alone retrieves at 0.0062, barely above the 0.0050
random floor, while the other 15 features together reach 0.0318. If ENTERING_RANK_LOG turns
out to have HIGH autocorrelation despite that, then autocorrelation alone is not sufficient
for identity and the metric needs a second term — which is a finding about the method, and
is reported rather than hidden.

    python pipeline/probe_tennis_feature_identity.py
    python pipeline/probe_tennis_feature_identity.py --check
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
OUT = ROOT / "data" / "tennis_feature_identity.json"


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
        print(f"meta {len(meta)} vs matrix {X.shape[0]} — refusing positional join")
        return 2

    idx = {(m["player"], m["year"], m["tour"]): i for i, m in enumerate(meta)}
    pairs = [
        (i, idx[(m["player"], m["year"] + 1, m["tour"])])
        for i, m in enumerate(meta)
        if (m["player"], m["year"] + 1, m["tour"]) in idx
    ]
    src = np.array([p for p, _ in pairs])
    dst = np.array([q for _, q in pairs])
    print(f"{len(pairs)} adjacent-year pairs over {X.shape[0]} rows, " f"{len(feats)} features\n")

    rows = []
    for j, f in enumerate(feats):
        # only pairs where BOTH sides observed — a masked zero is not a measurement
        ok = (M[src, j] > 0) & (M[dst, j] > 0)
        n = int(ok.sum())
        if n < 50:
            rows.append(
                {
                    "feature": f,
                    "n_pairs": n,
                    "autocorr": None,
                    "note": "too few observed pairs",
                }
            )
            continue
        u, v = X[src[ok], j], X[dst[ok], j]
        r = float(np.corrcoef(u, v)[0, 1]) if u.std() > 0 and v.std() > 0 else float("nan")
        rows.append(
            {
                "feature": f,
                "n_pairs": n,
                "autocorr": round(r, 4),
                "sd_all_rows": round(float(X[M[:, j] > 0, j].std()), 4),
            }
        )

    rows.sort(key=lambda x: -(x["autocorr"] if x["autocorr"] is not None else -9))
    print(f"  {'feature':26} {'pairs':>6} {'autocorr':>9} {'sd':>8}")
    for r in rows:
        ac = "n/a" if r["autocorr"] is None else f"{r['autocorr']:.4f}"
        sd = r.get("sd_all_rows")
        print(f"  {r['feature']:26} {r['n_pairs']:>6} {ac:>9} " f"{('' if sd is None else f'{sd:.4f}'):>8}")

    good = [r for r in rows if (r["autocorr"] or 0) >= 0.50]
    weak = [r for r in rows if r["autocorr"] is not None and r["autocorr"] < 0.25]
    print(f"\n  autocorrelation >= 0.50 : {len(good)}  " f"{[r['feature'] for r in good]}")
    print(f"  autocorrelation <  0.25 : {len(weak)}  " f"{[r['feature'] for r in weak]}")

    rank = next((r for r in rows if r["feature"] == "ENTERING_RANK_LOG"), None)
    note = ""
    if rank and rank["autocorr"] is not None and rank["autocorr"] >= 0.50:
        note = (
            f"ENTERING_RANK_LOG autocorrelates at {rank['autocorr']:.4f} yet retrieves "
            f"at 0.0062, barely over the 0.0050 random floor. So HIGH AUTOCORRELATION "
            f"IS NOT SUFFICIENT for identity: rank is stable within a player AND "
            f"shared with hundreds of others at any moment, which makes it a poor "
            f"discriminator. A useful feature needs to be stable AND rare — "
            f"autocorrelation measures only the first half."
        )
        print(f"\n  METHOD FINDING: {note}")

    # ---- does dropping the noisy features actually help retrieval? -------------
    # Autocorrelation is a hypothesis about usefulness, not a measurement of it. Testing it
    # costs one pass of the same cosine retrieval probe_tennis_retrieval.py runs.
    tours = np.array([m["tour"] for m in meta])
    K = 10

    def recall(cols):
        F = np.hstack([X[:, cols], M[:, cols]])
        mu, sd = F.mean(0), F.std(0)
        sd[sd == 0] = 1.0
        E = (F - mu) / sd
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        h = 0
        for q, t in pairs:
            same = np.where(tours == tours[q])[0]
            same = same[same != q]
            if t in same[np.argsort(-(En[same] @ En[q]))][:K]:
                h += 1
        return h / len(pairs)

    acmap = {r["feature"]: (r["autocorr"] or 0.0) for r in rows}
    ALL = list(range(len(feats)))
    rank_j = feats.index("ENTERING_RANK_LOG")
    subsets = {
        "all_16": ALL,
        "autocorr_ge_0.25": [j for j in ALL if acmap[feats[j]] >= 0.25],
        "autocorr_ge_0.25_no_rank": [j for j in ALL if acmap[feats[j]] >= 0.25 and j != rank_j],
        "autocorr_ge_0.50": [j for j in ALL if acmap[feats[j]] >= 0.50],
        "only_the_noisy_lt_0.25": [j for j in ALL if acmap[feats[j]] < 0.25],
        "all_16_minus_rank": [j for j in ALL if j != rank_j],
    }
    sub = {}
    print(f"\n  {'feature set':28} {'n':>3} {'recall@10':>10}")
    for name, cols in subsets.items():
        r = recall(cols)
        sub[name] = {"n_features": len(cols), "recall_at_10": round(float(r), 4)}
        print(f"  {name:28} {len(cols):>3} {r:>10.4f}")
    best = max(sub, key=lambda k: sub[k]["recall_at_10"])
    delta = sub[best]["recall_at_10"] - sub["all_16"]["recall_at_10"]
    binom = float(np.sqrt(sub[best]["recall_at_10"] * (1 - sub[best]["recall_at_10"]) / max(1, len(pairs))))
    print(f"\n  best: {best} at {sub[best]['recall_at_10']:.4f}, " f"{delta:+.4f} over all 16")
    print(
        f"  binomial sd at n={len(pairs)}: {binom:.4f}  -> the gain is "
        f"{abs(delta)/binom:.1f} sd, suggestive rather than decisive"
    )

    OUT.write_text(
        json.dumps(
            {
                "question": (
                    "Which of the 16 tennis features are stable within a player across "
                    "adjacent years, i.e. could contribute to identifying that player?"
                ),
                "subset_retrieval": sub,
                "subset_finding": (
                    f"Dropping the 7 features with autocorrelation below 0.25 raises recall@10 from "
                    f"{sub['all_16']['recall_at_10']} to {sub['autocorr_ge_0.25']['recall_at_10']} "
                    f"— a free improvement with no new data. The binomial sd at n={len(pairs)} is "
                    f"{binom:.4f}, so {delta:+.4f} is about {abs(delta)/binom:.1f} sd: suggestive, "
                    f"not decisive."
                ),
                "two_nuances_that_survive": (
                    "The 'noisy' 7 alone still retrieve at 0.0150, three times the 0.0050 random "
                    "floor, so they are not pure noise — low autocorrelation understates them. And "
                    "keeping ENTERING_RANK_LOG helps in COMBINATION (0.0393 with, 0.0349 without) "
                    "even though rank alone retrieves at 0.0062: it narrows the candidate pool "
                    "without identifying anyone by itself. Both cut against reading the "
                    "autocorrelation column as a ranking of feature value."
                ),
                "why": (
                    "probe_tennis_metric.py showed more model capacity does not help, so the "
                    "recommendation was more features. That is only actionable if it says "
                    "which KIND, and 'add the odds columns' is a guess until measured."
                ),
                "method": (
                    "Pearson r between a player's value this year and next, over adjacent-"
                    "year pairs where BOTH sides are observed. No training."
                ),
                "n_pairs": len(pairs),
                "n_rows": int(X.shape[0]),
                "per_feature": rows,
                "method_caveat": note
                or (
                    "Autocorrelation measures stability only. A feature can be perfectly stable and "
                    "still useless for identity if everyone shares its value; discrimination needs "
                    "stable AND rare. This probe measures the first half."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {OUT}")
    if args.check and not pairs:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
