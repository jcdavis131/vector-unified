#!/usr/bin/env python3
"""Are there tennis ROLES in this matrix, or only quality tiers?

Solo personal project, no connection to employer, built with public/free-tier only

Before tennis can join the unified fold it needs native clusters that mean something, and
the cross-sport archetype map is built on ROLE, not on how good a player is. Hoops has
positions, gridiron has positions, pitch has positions. **Tennis has none** — every player
does the same job — so the archetype has to come out of behaviour or not at all.

THE FAILURE MODE THIS EXISTS TO CATCH. WIN_RATE, GAMES_WON_PCT, DRAW_PROGRESS_MEAN,
ENTERING_RANK_LOG and HOLD_RATE all move together with one thing: how good the player is.
k-means on the raw matrix will therefore return QUALITY TIERS — top-50, journeyman,
qualifier — and those tiers will look like clusters, produce a decent silhouette, and be
entirely useless as archetypes. Calling them roles would be the same category error that
7.8 caught between role and trajectory, and 7.32 caught between age and team strength.

So the question is asked in two parts and the first one is allowed to end it:

  1. How much of the variance is a single quality dimension? PC1 loadings are printed. If
     PC1 is quality and dominates, raw clustering is measuring the leaderboard.
  2. Is there structure LEFT once quality is projected out? Clusters are fit on the
     quality-orthogonalised matrix, and their profiles are printed on the ORIGINAL features
     so a human can judge whether they read as styles.

PRE-REGISTERED READING, fixed before the first run:

  * A cluster set whose profiles differ mainly in WIN_RATE / DRAW_PROGRESS_MEAN /
    ENTERING_RANK_LOG is a quality tiering and must be reported as such, not shipped as an
    archetype.
  * A usable role split must separate on SURFACE_SPECIALISATION, the surface win rates,
    DECIDER_RATE or STRAIGHT_SETS_RATE — the behavioural columns — with quality roughly
    flat across clusters.
  * Naming is a human judgement and is labelled as one. This file proposes nothing.

    python pipeline/probe_tennis_archetypes.py
    python pipeline/probe_tennis_archetypes.py --k 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAT = ROOT / "pipeline" / "data" / "tennis_matrix.npz"
META = ROOT / "pipeline" / "data" / "meta_tennis_matrix.json"
OUT = ROOT / "data" / "tennis_archetype_probe.json"

QUALITY = ("WIN_RATE", "GAMES_WON_PCT", "DRAW_PROGRESS_MEAN", "ENTERING_RANK_LOG",
           "HOLD_RATE")
BEHAVIOURAL = ("SURFACE_SPECIALISATION", "HARD_WR", "CLAY_WR", "GRASS_WR",
               "DECIDER_RATE", "STRAIGHT_SETS_RATE", "BIG_EVENT_SHARE", "INDOOR_WR",
               "UPSET_RATE", "RETIRE_RATE")
SEED = 7


def masked_z(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Column z-score over OBSERVED cells only, unobserved left at the column mean (0).

    Standardising over zero-filled holes would pull every masked column toward whatever
    fraction of it is missing — GRASS_WR is 55.6% observed, so its 'mean' would be mostly
    an artefact of absence.
    """
    Z = np.zeros_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        obs = M[:, j].astype(bool)
        if obs.sum() < 20:
            continue
        col = X[obs, j].astype(np.float64)
        mu, sd = col.mean(), col.std()
        if sd == 0:
            continue
        Z[obs, j] = (col - mu) / sd
    return Z


def kmeans(Z: np.ndarray, k: int, iters: int = 60) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    C = Z[rng.choice(len(Z), k, replace=False)].copy()
    lab = np.zeros(len(Z), dtype=int)
    for _ in range(iters):
        d = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == lab).all():
            break
        lab = new
        for c in range(k):
            if (lab == c).any():
                C[c] = Z[lab == c].mean(0)
    return lab


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not MAT.exists():
        print(f"missing {MAT} — run build_tennis_matrix.py")
        return 2
    z = np.load(MAT, allow_pickle=True)
    X, M, feats = z["X"], z["M"], [str(f) for f in z["features"]]
    Z = masked_z(X, M)

    # ---- 1. how much is one quality dimension? -------------------------------
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    pc1 = dict(sorted(zip(feats, Vt[0], strict=True), key=lambda kv: -abs(kv[1])))
    pc1_top = list(pc1)[:6]
    pc1_is_quality = sum(1 for f in pc1_top if f in QUALITY) >= 3

    # QUALITY AXIS, PROJECTED OUT. The axis is a direction in FEATURE space — the mean of
    # the standardised quality columns — not a per-row score, and getting that backwards is
    # what the first attempt did: it built a 4,022-long vector and tried to subtract it
    # along 16 dimensions. The direction is a unit vector of length n_features, and each
    # row loses its component along it.
    qvec = np.zeros(Z.shape[1])
    for f in QUALITY:
        if f in feats:
            qvec[feats.index(f)] = 1.0
    qvec /= np.linalg.norm(qvec)
    Zr = Z - np.outer(Z @ qvec, qvec)

    lab = kmeans(Zr, args.k)

    prof, sizes = {}, {}
    for c in range(args.k):
        sel = lab == c
        sizes[c] = int(sel.sum())
        prof[c] = {f: round(float(Z[sel, j].mean()), 3)
                   for j, f in enumerate(feats)}

    # spread across clusters, per feature: the discriminating columns
    spread = {f: round(float(max(prof[c][f] for c in prof) - min(prof[c][f] for c in prof)), 3)
              for f in feats}
    top_disc = sorted(spread, key=lambda f: -spread[f])[:6]
    quality_spread = max(spread[f] for f in QUALITY if f in spread)
    behav_spread = max(spread[f] for f in BEHAVIOURAL if f in spread)

    # PER-CLUSTER, because the global max-spread test answers "is the PARTITION a tiering"
    # and that is not the only question worth asking. A single cluster can be a clean role
    # inside an otherwise quality-driven partition, and throwing the whole partition away
    # would discard it. Flat quality + an extreme behavioural column = a role cluster.
    per_cluster = {}
    for c in range(args.k):
        qmax = max(abs(prof[c][f]) for f in QUALITY if f in prof[c])
        bmax = max(abs(prof[c][f]) for f in BEHAVIOURAL if f in prof[c])
        bfeat = max((f for f in BEHAVIOURAL if f in prof[c]),
                    key=lambda f: abs(prof[c][f]))
        per_cluster[c] = {
            "n": sizes[c],
            "max_abs_quality_z": round(qmax, 3),
            "max_abs_behavioural_z": round(bmax, 3),
            "dominant_behavioural_feature": bfeat,
            "dominant_value": prof[c][bfeat],
            "role_like": bool(qmax < 0.5 and bmax > 1.0),
        }
    role_clusters = [c for c, v in per_cluster.items() if v["role_like"]]

    verdict = (
        "ROLE-LIKE — the widest-spread columns are behavioural and quality is comparatively "
        "flat across clusters."
        if behav_spread > quality_spread else
        "QUALITY TIERING — clusters separate mainly on how good the player is. NOT usable "
        "as archetypes; naming these would repeat the role-vs-trajectory category error.")

    report = {
        "question": "Are there tennis ROLES in this matrix, or only quality tiers?",
        "rows": int(X.shape[0]), "k": args.k,
        "pc1_variance_share": round(float(var[0]), 4),
        "pc2_variance_share": round(float(var[1]), 4),
        "pc1_top_loadings": {f: round(float(pc1[f]), 3) for f in pc1_top},
        "pc1_reads_as_quality": bool(pc1_is_quality),
        "cluster_sizes": sizes,
        "cluster_profiles_in_ORIGINAL_feature_z": prof,
        "spread_across_clusters": spread,
        "top_discriminating_features": top_disc,
        "max_spread_quality_cols": round(quality_spread, 3),
        "max_spread_behavioural_cols": round(behav_spread, 3),
        "verdict": verdict,
        "per_cluster": per_cluster,
        "role_like_clusters": role_clusters,
        "per_cluster_note": (
            "A cluster is ROLE-LIKE when quality is flat (max |z| over the quality columns "
            "< 0.5) and a behavioural column is extreme (> 1.0). The global verdict asks "
            "whether the PARTITION is a tiering; this asks whether any single cluster is a "
            "clean role inside it, because discarding the whole partition would throw those "
            "away with the tiers."),
        "method_note": (
            "Clusters are fit on the matrix with the QUALITY axis projected out — the mean "
            "of the standardised WIN_RATE / GAMES_WON_PCT / DRAW_PROGRESS_MEAN / "
            "ENTERING_RANK_LOG / HOLD_RATE columns. Profiles are then printed on the "
            "ORIGINAL standardised features, so the quality spread reported above is what "
            "SURVIVED the projection rather than what was fed to k-means."),
        "naming_note": (
            "This file proposes no archetype names. Naming a cluster is a human judgement "
            "and labelling it here would launder a k-means partition into a taxonomy."),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"rows {X.shape[0]}   k {args.k}")
    print(f"PC1 {var[0]:.1%} of variance, reads as quality: {pc1_is_quality}")
    print(f"  top loadings: {', '.join(f'{f} {pc1[f]:+.2f}' for f in pc1_top)}")
    print(f"\ncluster sizes: {sizes}")
    print(f"\n{'feature':24}" + "".join(f"{c:>8}" for c in range(args.k)) + "   spread")
    for f in feats:
        print(f"{f:24}" + "".join(f"{prof[c][f]:>8.2f}" for c in range(args.k))
              + f"   {spread[f]:.2f}")
    print(f"\nwidest quality spread {quality_spread:.3f}   "
          f"widest behavioural spread {behav_spread:.3f}")
    print(f"\n{verdict}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
