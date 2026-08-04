#!/usr/bin/env python3
"""Tennis MTNN — multi-tower embedding over 28 features. Must beat 0.0584 to be worth it.

Solo personal project, no connection to employer, built with public/free-tier only

THE BAR IS NOT THE RAW BASELINE. Four probes set it:

    probe_tennis_retrieval.py          raw 16 features, cosine        0.0328
    probe_tennis_feature_identity.py   drop the 7 noisy ones          0.0393
    probe_tennis_candidate_features.py + 12 schedule/shot features    0.0584 (cosine 0.0447)
    probe_tennis_metric.py --enriched  LEARNED LINEAR over 28         0.0584, 5/5 seeds

A deep model must beat 0.0584 — the learned LINEAR mean — not the 0.0447 raw cosine and
certainly not the 0.0328 the thread started from. Landing between those would look like
progress while being worse than a projection matrix, which is the specific way this could
go wrong and be reported as success.

TOWERS BY FAMILY, mirroring how hoops and equities are built. Each tower sees its own
feature block AND that block's mask, so a structural zero cannot read as a measurement —
the rule build_tennis_matrix.py states and every probe here has followed.

    outcome     WIN_RATE, GAMES_WON_PCT
    surface     HARD_WR, CLAY_WR, GRASS_WR, INDOOR_WR, SURFACE_SPECIALISATION
    opposition  UPSET_RATE, MEAN_OPP_RANK_LOG, ENTERING_RANK_LOG
    matchshape  HOLD_RATE, STRAIGHT_SETS_RATE, DECIDER_RATE, RETIRE_RATE
    schedule    DRAW_PROGRESS_MEAN, BIG_EVENT_SHARE, TIER_SHARE_TOP, N_TOURNAMENTS,
                N_LOCATIONS, MEAN_EVENTS_PER_MONTH
    surfacemix  SURF_SHARE_HARD, SURF_SHARE_CLAY, SURF_SHARE_GRASS, INDOOR_SHARE
    shotshape   TIEBREAK_RATE, BAGEL_RATE, COMEBACK_RATE, MEAN_GAMES_PER_SET

SCHEDULE AND SURFACEMIX ARE SEPARATE TOWERS ON PURPOSE. 4821c78 measured that schedule
features alone (0.0441) out-identify all nine surviving performance features (0.0393),
while shot-shape alone reaches 0.0079 — barely over the 0.0050 random floor. Giving the
strongest family its own capacity is the point of a tower architecture; merging it into a
general "context" block would hide exactly the signal that motivated this model.

TEMPORAL SPLIT at 2022 and FIVE SEEDS, identical to the probes so the numbers are directly
comparable. Retrieval is scored within tour against the FULL same-tour corpus.

    python pipeline/train_tennis_mtnn.py
    python pipeline/train_tennis_mtnn.py --check   # exit 1 if it fails to beat the bar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

MATRIX = ROOT / "pipeline" / "data" / "tennis_matrix.npz"
META = ROOT / "pipeline" / "data" / "meta_tennis_matrix.json"
OUT = ROOT / "data" / "tennis_mtnn_report.json"
EMB = ROOT / "pipeline" / "data" / "tennis_mtnn_embedding.npz"

BAR = 0.0584          # learned-linear mean from probe_tennis_metric.py --enriched
RAW_COSINE = 0.0447   # raw cosine over the same 28 features
K = 10
CUT = 2022
SEEDS = (7, 11, 13, 17, 19)

FAMILIES = {
    "outcome": ["WIN_RATE", "GAMES_WON_PCT"],
    "surface": ["HARD_WR", "CLAY_WR", "GRASS_WR", "INDOOR_WR", "SURFACE_SPECIALISATION"],
    "opposition": ["UPSET_RATE", "MEAN_OPP_RANK_LOG", "ENTERING_RANK_LOG"],
    "matchshape": ["HOLD_RATE", "STRAIGHT_SETS_RATE", "DECIDER_RATE", "RETIRE_RATE"],
    "schedule": ["DRAW_PROGRESS_MEAN", "BIG_EVENT_SHARE", "TIER_SHARE_TOP",
                 "N_TOURNAMENTS", "N_LOCATIONS", "MEAN_EVENTS_PER_MONTH"],
    "surfacemix": ["SURF_SHARE_HARD", "SURF_SHARE_CLAY", "SURF_SHARE_GRASS",
                   "INDOOR_SHARE"],
    "shotshape": ["TIEBREAK_RATE", "BAGEL_RATE", "COMEBACK_RATE", "MEAN_GAMES_PER_SET"],
}


def recall_at_k(E, pairs, tours, k=K):
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    hits = 0
    for q, t in pairs:
        same = np.where(tours == tours[q])[0]
        same = same[same != q]
        if t in same[np.argsort(-(En[same] @ En[q]))][:k]:
            hits += 1
    return hits / len(pairs) if pairs else float("nan")


def build_inputs():
    from probe_tennis_candidate_features import CANDIDATES, build as build_cand
    a = np.load(MATRIX, allow_pickle=True)
    X, M = a["X"].astype(np.float32), a["M"].astype(np.float32)
    feats = [str(f) for f in a["features"]]
    meta = json.loads(META.read_text(encoding="utf-8"))
    if len(meta) != X.shape[0]:
        raise SystemExit(f"meta {len(meta)} vs matrix {X.shape[0]} — refusing")

    cand = build_cand()
    keys = [(m["player"], m["year"], m["tour"]) for m in meta]
    C = np.full((len(keys), len(CANDIDATES)), np.nan, dtype=np.float32)
    for r, k in enumerate(keys):
        v = cand.get(k)
        if v:
            for c, nm in enumerate(CANDIDATES):
                if v.get(nm) is not None:
                    C[r, c] = v[nm]
    CM = (~np.isnan(C)).astype(np.float32)
    X = np.hstack([X, np.nan_to_num(C)])
    M = np.hstack([M, CM])
    names = feats + list(CANDIDATES)

    # z-score values using OBSERVED rows only, so masked zeros do not drag the mean
    for j in range(X.shape[1]):
        obs = M[:, j] > 0
        if obs.sum() > 1:
            mu, sd = X[obs, j].mean(), X[obs, j].std()
            X[obs, j] = (X[obs, j] - mu) / (sd if sd > 0 else 1.0)
    return X, M, names, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--tower-width", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("torch not available")
        return 2

    X, M, names, meta = build_inputs()
    idx_of = {n: i for i, n in enumerate(names)}
    missing = [f for fam in FAMILIES.values() for f in fam if f not in idx_of]
    if missing:
        print(f"features named in FAMILIES but absent from the matrix: {missing}")
        return 2
    covered = {f for fam in FAMILIES.values() for f in fam}
    if len(covered) != len(names):
        print(f"family map covers {len(covered)} of {len(names)} features — refusing to "
              f"train on a partition that silently drops columns: "
              f"{sorted(set(names) - covered)}")
        return 2

    tours = np.array([m["tour"] for m in meta])
    idx = {(m["player"], m["year"], m["tour"]): i for i, m in enumerate(meta)}
    allp = [(i, idx[(m["player"], m["year"] + 1, m["tour"])], m["year"] + 1)
            for i, m in enumerate(meta)
            if (m["player"], m["year"] + 1, m["tour"]) in idx]
    tr = [(a, b) for a, b, y in allp if y <= CUT]
    te = [(a, b) for a, b, y in allp if y > CUT]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blocks = {k: [idx_of[f] for f in v] for k, v in FAMILIES.items()}
    print(f"rows {X.shape[0]}  features {X.shape[1]}  towers {len(blocks)}  device {dev}")
    print(f"pairs {len(allp)}  train(target<= {CUT}) {len(tr)}  test {len(te)}")
    print(f"BAR {BAR} (learned linear) / raw cosine {RAW_COSINE}\n")

    Xt = torch.tensor(X, device=dev)
    Mt = torch.tensor(M, device=dev)
    A = torch.tensor([a for a, _ in tr], device=dev)
    B = torch.tensor([b for _, b in tr], device=dev)

    class MTNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.towers = nn.ModuleDict({
                k: nn.Sequential(nn.Linear(2 * len(v), args.tower_width), nn.GELU(),
                                 nn.Dropout(args.dropout),
                                 nn.Linear(args.tower_width, args.tower_width))
                for k, v in blocks.items()})
            self.fuse = nn.Sequential(
                nn.Linear(args.tower_width * len(blocks), 64), nn.GELU(),
                nn.Dropout(args.dropout), nn.Linear(64, args.dim))

        def forward(self, xi, mi):
            zs = [self.towers[k](torch.cat([xi[:, v], mi[:, v]], 1))
                  for k, v in blocks.items()]
            z = self.fuse(torch.cat(zs, 1))
            return z / (z.norm(dim=1, keepdim=True) + 1e-9)

    got = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MTNN().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        g = torch.Generator(device="cpu").manual_seed(seed)
        for _ in range(args.epochs):
            model.train()
            sel = torch.randperm(len(A), generator=g)[:256].to(dev)
            za = model(Xt[A[sel]], Mt[A[sel]])
            zb = model(Xt[B[sel]], Mt[B[sel]])
            logits = za @ zb.T / args.temp
            lbl = torch.arange(len(sel), device=dev)
            loss = 0.5 * (nn.functional.cross_entropy(logits, lbl)
                          + nn.functional.cross_entropy(logits.T, lbl))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            E = model(Xt, Mt).cpu().numpy()
        r = recall_at_k(E, te, tours)
        got.append(r)
        print(f"  seed {seed:<3} recall@{K} = {r:.4f}  (loss {loss.item():.4f})",
              flush=True)
        if seed == SEEDS[0]:
            np.savez_compressed(EMB, E=E.astype(np.float32),
                                player=np.array([m["player"] for m in meta]),
                                year=np.array([m["year"] for m in meta]),
                                tour=tours)

    v = np.array(got)
    beats = int((v > BAR).sum())
    print(f"\n  MTNN mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  "
          f"range {v.max()-v.min():.4f}")
    print(f"  vs learned-linear bar {BAR}: {v.mean()-BAR:+.4f}   seeds over bar {beats}/5")
    print(f"  vs raw cosine {RAW_COSINE}: {v.mean()-RAW_COSINE:+.4f}")

    passes = v.mean() > BAR and beats >= 4
    verdict = (f"MTNN BEATS the learned linear map ({v.mean():.4f} > {BAR})"
               if passes else
               f"MTNN DOES NOT BEAT the learned linear map ({v.mean():.4f} vs {BAR}) — "
               f"the extra capacity is not earning its place, and a linear projection "
               f"remains the honest choice")
    print(f"\n  verdict: {verdict}")

    OUT.write_text(json.dumps({
        "question": "Does a multi-tower neural embedding beat a learned LINEAR map?",
        "bar": BAR,
        "bar_note": ("0.0584 is the learned-linear mean from probe_tennis_metric.py "
                     "--enriched, NOT the 0.0447 raw cosine and not the 0.0328 the thread "
                     "started from. A model landing between those would look like progress "
                     "while being worse than a projection matrix."),
        "raw_cosine_same_features": RAW_COSINE,
        "mtnn_per_seed": [round(float(x), 4) for x in got],
        "mtnn_mean": round(float(v.mean()), 4),
        "mtnn_sd": round(float(v.std(ddof=1)), 4),
        "seeds_over_bar": f"{beats}/5",
        "delta_vs_bar": round(float(v.mean() - BAR), 4),
        "towers": {k: len(v_) for k, v_ in FAMILIES.items()},
        "config": {"dim": args.dim, "tower_width": args.tower_width,
                   "epochs": args.epochs, "lr": args.lr, "temp": args.temp,
                   "dropout": args.dropout},
        "split": f"TEMPORAL — train target year <= {CUT}, test strictly after",
        "n_pairs": len(allp), "n_train": len(tr), "n_test": len(te),
        "verdict": verdict,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    if args.check and not passes:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
