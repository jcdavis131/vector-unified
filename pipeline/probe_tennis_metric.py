#!/usr/bin/env python3
"""Can a LEARNED map beat raw cosine on tennis retrieval? Decide before building an MTNN.

Solo personal project, no connection to employer, built with public/free-tier only

probe_tennis_retrieval.py measured the baseline: raw 16 features retrieve a player's
adjacent year at recall@10 = 0.0328 against a measured random floor of 0.0050. Real signal
(6.7x random, cosine separation +0.2659 own-next-year vs random peer) at a low absolute
level.

THIS IS THE CHEAP DECISIVE TEST BEFORE COMMITTING TO AN MTNN. Fit the smallest learned
thing that could help — a linear projection trained with a contrastive objective on
adjacent-year pairs — and see whether it moves recall@10 at all. If a linear metric cannot
improve on raw cosine over 16 features, a deep model over the same 16 is unlikely to, and
the honest recommendation is to not train one.

TEMPORAL SPLIT, NOT RANDOM. A random split puts a player's 2019 and 2020 rows on both sides
and the model scores by memorising the player, which is the entire task. Train on pairs
whose target year is <= the cut, evaluate strictly after — the same rule
build_tennis_forward.py uses and for the same reason.

RETRIEVAL IS SCORED WITHIN TOUR AND AGAINST THE FULL SAME-TOUR CORPUS, not against a
sampled subset. Restricting candidates to the test years would shrink the haystack and
inflate recall; the metric has to mean "found among everyone in that tour".

SEEDS. Five, because this session established that judging a change from one run is how a
lucky draw gets shipped. The spread is reported, not just the mean.

    python pipeline/probe_tennis_metric.py
    python pipeline/probe_tennis_metric.py --check   # exit 1 if the run is broken
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
OUT = ROOT / "data" / "tennis_metric_probe.json"
K = 10
CUT = 2022
DIM = 16
SEEDS = (7, 11, 13, 17, 19)
EPOCHS = 300
LR = 0.05
TEMP = 0.1


def recall_at_k(E, pairs, tours, k=K):
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    hits = 0
    for q, t in pairs:
        same = np.where(tours == tours[q])[0]
        same = same[same != q]
        order = same[np.argsort(-(En[same] @ En[q]))]
        if t in order[:k]:
            hits += 1
    return hits / len(pairs) if pairs else float("nan")


def train_linear(Fz, tr_pairs, tours, seed, dim=DIM):
    """Contrastive linear projection, in-batch negatives, plain numpy SGD."""
    rng = np.random.default_rng(seed)
    d = Fz.shape[1]
    W = rng.normal(0, 1.0 / np.sqrt(d), size=(d, dim))
    A = np.array([a for a, _ in tr_pairs])
    B = np.array([b for _, b in tr_pairs])
    for _ in range(EPOCHS):
        sel = rng.choice(len(A), size=min(256, len(A)), replace=False)
        qa, qb = Fz[A[sel]], Fz[B[sel]]
        za, zb = qa @ W, qb @ W
        za = za / (np.linalg.norm(za, axis=1, keepdims=True) + 1e-9)
        zb = zb / (np.linalg.norm(zb, axis=1, keepdims=True) + 1e-9)
        logits = (za @ zb.T) / TEMP
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        g = p - np.eye(len(sel))
        # gradient through the (unnormalised) bilinear form; the norm is treated as a
        # constant, which is the usual cheap approximation and is fine at this scale
        gW = (qa.T @ (g @ zb) + qb.T @ (g.T @ za)) / (len(sel) * TEMP)
        W -= LR * gW
    return W


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
    meta = json.loads(META.read_text(encoding="utf-8"))
    if len(meta) != X.shape[0]:
        print(f"meta {len(meta)} vs matrix {X.shape[0]} — refusing positional join")
        return 2

    tours = np.array([m["tour"] for m in meta])
    idx = {(m["player"], m["year"], m["tour"]): i for i, m in enumerate(meta)}
    pairs = [(i, idx[(m["player"], m["year"] + 1, m["tour"])], m["year"] + 1)
             for i, m in enumerate(meta)
             if (m["player"], m["year"] + 1, m["tour"]) in idx]
    tr = [(a_, b) for a_, b, y in pairs if y <= CUT]
    te = [(a_, b) for a_, b, y in pairs if y > CUT]

    F = np.hstack([X, M])
    mu, sd = F.mean(0), F.std(0)
    sd[sd == 0] = 1.0
    Fz = (F - mu) / sd

    base = recall_at_k(Fz, te, tours)
    print(f"pairs {len(pairs)}  train(target<= {CUT}) {len(tr)}  test {len(te)}")
    print(f"\n  BASELINE raw cosine, test pairs      recall@{K} = {base:.4f}")
    if len(te) < 50:
        print("test split too small to interpret")
        return 2

    got = []
    for s in SEEDS:
        W = train_linear(Fz, tr, tours, s)
        r = recall_at_k(Fz @ W, te, tours)
        got.append(r)
        print(f"  learned linear map, seed {s:<3}        recall@{K} = {r:.4f}")

    g = np.array(got)
    lift = g.mean() - base
    print(f"\n  learned mean {g.mean():.4f}  sd {g.std(ddof=1):.4f}  "
          f"range {g.max()-g.min():.4f}")
    print(f"  lift over raw cosine {lift:+.4f}")
    beats = int((g > base).sum())
    print(f"  seeds beating the baseline: {beats}/{len(SEEDS)}")

    verdict = ("A LEARNED MAP HELPS — an MTNN is worth building and must beat this, "
               "not the raw baseline" if lift > 0.01 and beats == len(SEEDS) else
               "NO — a learned linear metric does not improve on raw cosine over these 16 "
               "features, so a deeper model over the same 16 is a poor bet")
    print(f"\n  verdict: {verdict}")

    OUT.write_text(json.dumps({
        "question": ("Can a learned linear metric beat raw cosine at retrieving a tennis "
                     f"player's adjacent year (recall@{K}, within tour)?"),
        "why_before_an_mtnn": (
            "16 features into a 64-d embedding is expansion, not compression. If the "
            "smallest learned thing that could help does not, a deeper model over the same "
            "16 features is unlikely to, and this costs seconds instead of a training run."),
        "split": f"TEMPORAL — train on target year <= {CUT}, evaluate strictly after",
        "retrieval_scope": ("candidates are the FULL same-tour corpus, not just test years; "
                            "restricting to test years would shrink the haystack and "
                            "inflate recall"),
        "baseline_raw_cosine": round(float(base), 4),
        "learned_per_seed": [round(float(x), 4) for x in got],
        "learned_mean": round(float(g.mean()), 4),
        "learned_sd": round(float(g.std(ddof=1)), 4),
        "lift": round(float(lift), 4),
        "seeds_beating_baseline": f"{beats}/{len(SEEDS)}",
        "n_pairs": len(pairs), "n_train": len(tr), "n_test": len(te),
        "verdict": verdict,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    if args.check and not te:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
