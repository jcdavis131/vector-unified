#!/usr/bin/env python3
"""Does a tennis player's PLAYING STYLE predict next year's ranking, beyond this year's?

Solo personal project, no connection to employer, built with public/free-tier only

Tennis has been a feature matrix in this repo, not a model: 4,022 player-year-tour rows x 16
masked features and nothing that predicts anything. The dumbmodel.com tennis card says so.
This is the smallest honest model over it, and it is built to be REFUSABLE.

THE QUESTION, stated so the answer can be no. Next year's entering rank is largely this
year's entering rank — measured, r = 0.7575 across 2,926 consecutive-year pairs. So the
only interesting question is whether the other fifteen features (surface splits, upset rate,
hold rate, draw progress, retirement rate...) add anything ON TOP of that. If they do not,
the honest output is "they do not", and this file says so rather than reporting a raw r that
persistence already earned.

    BASELINE      predict next rank = this rank                       r = 0.7575
    RIDGE-1       ridge on ENTERING_RANK_LOG alone
    RIDGE-16      ridge on all sixteen features
    the model earns its keep only if RIDGE-16 beats RIDGE-1 out of sample

TEMPORAL SPLIT, NOT RANDOM, and this is the whole design. A random split puts the same
player's 2019 and 2020 rows on opposite sides, and rank is so persistent that the model
would score well by having memorised the player. Train on pairs whose target year is <= the
cut, test on strictly later years. That is also the only split that matches what the model
would be USED for: predicting a season that has not happened.

MASK IS RESPECTED. X carries 0.0 where a feature was unobserved (a player who never played
grass did not lose every grass match), so the mask column is fed alongside each feature
rather than letting a structural zero read as a measurement.

NON-VACUITY IS PART OF THE RUN, not a separate script. The same pipeline is scored with the
TARGET SHUFFLED. If the shuffled arm does not collapse to ~0, the evaluation is broken and
the real number means nothing.

    python pipeline/build_tennis_forward.py
    python pipeline/build_tennis_forward.py --check   # exit 1 if the null does not collapse
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
OUT = ROOT / "data" / "tennis_forward_report.json"
SEED = 7
CUT_YEAR = 2022          # train on target years <= this, test strictly after
NULL_TOL = 0.15          # a shuffled target must land within this of zero


def ridge(Xtr, ytr, Xte, lam=1.0):
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ytr)
    return Xte @ w


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not MATRIX.exists():
        print(f"missing {MATRIX} — run build_tennis_matrix.py first")
        return 2
    a = np.load(MATRIX, allow_pickle=True)
    X, M, feats = a["X"], a["M"], [str(f) for f in a["features"]]
    meta = json.loads(META.read_text(encoding="utf-8"))
    if len(meta) != X.shape[0]:
        print(f"meta has {len(meta)} rows, matrix has {X.shape[0]} — refusing to join by "
              f"position across a length mismatch")
        return 2

    rank_j = feats.index("ENTERING_RANK_LOG")
    idx = {(m["player"], m["year"], m["tour"]): i for i, m in enumerate(meta)}
    pairs = [(i, idx[(m["player"], m["year"] + 1, m["tour"])], m["year"] + 1)
             for (k, i) in ((k, v) for k, v in idx.items())
             for m in [meta[i]]
             if (m["player"], m["year"] + 1, m["tour"]) in idx]
    pairs = [(i, k, y) for i, k, y in pairs if M[i, rank_j] == 1 and M[k, rank_j] == 1]

    # feature block = value AND its mask, so a structural zero cannot read as a measurement
    F = np.hstack([X, M.astype(np.float32)])
    src = np.array([i for i, _, _ in pairs])
    dst = np.array([k for _, k, _ in pairs])
    ty = np.array([y for _, _, y in pairs])
    y_next = X[dst, rank_j]
    y_prev = X[src, rank_j]

    tr, te = ty <= CUT_YEAR, ty > CUT_YEAR
    print(f"{len(pairs)} consecutive-year pairs   train(target<= {CUT_YEAR}) {tr.sum()}   "
          f"test {te.sum()}")
    if te.sum() < 100:
        print("test split too small to interpret")
        return 2

    persistence = r(y_prev[te], y_next[te])
    p1 = ridge(y_prev[tr, None], y_next[tr], y_prev[te, None])
    p16 = ridge(F[src][tr], y_next[tr], F[src][te])
    r1, r16 = r(p1, y_next[te]), r(p16, y_next[te])

    rng = np.random.default_rng(SEED)
    yperm = y_next[tr][rng.permutation(int(tr.sum()))]
    rnull = r(ridge(F[src][tr], yperm, F[src][te]), y_next[te])

    # ROBUSTNESS ACROSS CUTS, because one split can be lucky and a single +0.09 is not a
    # finding. Every cut year with a usable test set gets the same two models.
    sweep = []
    for cut in (2019, 2020, 2021, 2022, 2023):
        a_tr, a_te = ty <= cut, ty > cut
        if a_te.sum() < 80:
            continue
        g1 = r(ridge(y_prev[a_tr, None], y_next[a_tr], y_prev[a_te, None]), y_next[a_te])
        g16 = r(ridge(F[src][a_tr], y_next[a_tr], F[src][a_te]), y_next[a_te])
        sweep.append({"cut_year": cut, "n_test": int(a_te.sum()),
                      "rank_only_r": round(g1, 4), "all16_r": round(g16, 4),
                      "gain": round(g16 - g1, 4)})
    gains = [s_["gain"] for s_ in sweep]
    all_positive = bool(gains) and all(g > 0 for g in gains)

    gain = r16 - r1
    null_ok = abs(rnull) <= NULL_TOL
    earns = gain > 0.01 and null_ok and all_positive

    print(f"\n  persistence (this rank -> next)        r = {persistence:.4f}")
    print(f"  RIDGE-1  (rank only)                   r = {r1:.4f}")
    print(f"  RIDGE-16 (all features + masks)        r = {r16:.4f}")
    print(f"  gain from the other 15 features        {gain:+.4f}")
    print(f"  SHUFFLED-TARGET null                   r = {rnull:.4f}  "
          f"{'collapses, evaluation is sound' if null_ok else 'DOES NOT COLLAPSE'}")
    print(f"\n  verdict: {'style adds signal beyond rank' if earns else 'NO — the other features do not beat rank alone'}")

    OUT.write_text(json.dumps({
        "question": ("Does a tennis player's playing style predict next year's ranking "
                     "beyond what this year's ranking already predicts?"),
        "n_pairs": len(pairs), "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "split": f"TEMPORAL — train on target year <= {CUT_YEAR}, test strictly after",
        "why_temporal": ("A random split puts the same player's adjacent years on opposite "
                         "sides. Rank is persistent enough (r=0.7575 overall) that the model "
                         "would score well by memorising the player rather than learning "
                         "anything, and the split would not match the use — predicting a "
                         "season that has not happened."),
        "persistence_r": round(persistence, 4),
        "ridge1_rank_only_r": round(r1, 4),
        "ridge16_all_features_r": round(r16, 4),
        "gain_over_rank_alone": round(gain, 4),
        "cut_year_sweep": sweep,
        "gain_positive_at_every_cut": all_positive,
        "gain_mean_across_cuts": round(float(np.mean(gains)), 4),
        "shuffled_target_null_r": round(rnull, 4),
        "null_collapses": bool(null_ok),
        "verdict": ("style adds signal beyond rank" if earns else
                    "NO — the other 15 features do not improve on rank alone out of sample"),
        "honest_note": ("The interesting number is the GAIN, not the raw r. A raw r near "
                        "0.75 here is earned by persistence, not by the model, and quoting "
                        "it would be a real value answering a different question."),
        "mask_note": ("Each feature is fed alongside its mask column. X carries 0.0 where a "
                      "feature was unobserved — a player who never played grass did not lose "
                      "every grass match — so an unmasked zero would be read as a "
                      "measurement."),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")

    if args.check and not null_ok:
        print(f"\nFAIL shuffled-target null r={rnull:.4f} exceeds {NULL_TOL} — the "
              f"evaluation leaks and the real number cannot be trusted")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
