"""Vector Unified — cross-sport analogy panel (G4 face-validity).

Loads the SHIPPED checkpoint via eval_unified.load_and_encode, encodes every
player-season into z under that checkpoint's own encoder contract, and answers
the user's actual question: "what does a power forward have in common with a
strong safety?" Concretely:

  G4  Cross-sport NN role coherence  — for every player, take their nearest
      neighbour in z AMONG THE OTHER sports; hit if that NN shares the same
      cross-sport archetype (arch_id). Overall hit-rate is the analogy accuracy.
      Target >= 60% (the design's G4 bar).

  Panel  A small named showcase: for a handful of recognizable players per sport,
      print their top-3 cross-sport neighbours (name / pos / team / archetype) so
      a human can eyeball whether the analogies are sensible (PF -> SS / defensive
      mid, etc.).

Output: data/analogy_report.json + printed panel.
Honest scope: this is face-validity, not a held-out test (no train/test split on z
because Stage 1 is unsupervised folding over all rows). G4 measures whether the
folding is self-consistent, not generalization.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
from eval_unified import g4_random_baseline, load_and_encode
from load_encoders import ROOT, SPORTS, load_all
from train_unified import load_matrix

DATA = ROOT / "data"
# index -> cross-sport archetype id, in the order build_unified_matrix assigned them
# (the 7 in-scope v0 archetypes are A0,A1,A2,A3,A4,A5,A11 -- NOT A0..A6)
_meta = json.loads((DATA / "unified_meta.json").read_text(encoding="utf-8"))
ARCH_NAMES = _meta["arch_names"]


def cross_sport_nn(z, sid):
    """For each row, index of nearest neighbour in a DIFFERENT sport (cosine on L2 z)."""
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    sim = zn @ zn.T
    same = sid[:, None] == sid[None, :]
    sim = np.where(same, -np.inf, sim)
    np.fill_diagonal(sim, -np.inf)
    nn_idx = sim.argmax(axis=1)
    nn_sim = sim[np.arange(z.shape[0]), nn_idx]
    return nn_idx, nn_sim


def named_panel(z, M, names_per_sport=4):
    """Pick recognizable players: highest within-arch centrality per archetype per sport."""
    sid = M["sport_id"].cpu().numpy()
    arch = M["arch_id"].cpu().numpy()
    recs = M["records"]
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    panel = []
    for s in range(3):
        for a in np.unique(arch):
            ia = np.where((sid == s) & (arch == a))[0]
            if len(ia) < 3:
                continue
            sub = zn[ia]
            cent = sub @ sub.T
            np.fill_diagonal(cent, -np.inf)
            top = np.argsort(-cent.sum(axis=1))[:names_per_sport]
            for t in top:
                i = ia[t]
                r = recs[i]
                panel.append(
                    {
                        "sport": SPORTS[s],
                        "name": r["name"],
                        "pos": str(r["pos"]),
                        "team": r.get("team", ""),
                        "season": r["season"],
                        "arch": ARCH_NAMES[int(arch[i])],
                    }
                )
    return panel


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # auto: GPU on personal local (CUDA avail), CPU in Hatch VM
    M = load_matrix(device)
    # Shared contract: a Stage 2 checkpoint gets its DRIFTED encoders, not the frozen
    # cached ones. See eval_unified.load_and_encode.
    model, ck, z, z_source, model_label = load_and_encode(device)
    sid = M["sport_id"].cpu().numpy()
    arch = M["arch_id"].cpu().numpy()
    pidx = M["player_idx"].cpu().numpy()
    # rebuild per-row records (name/pos/team/season) from the sport encoders via player_idx
    all_sport = load_all(verbose=False)
    recs = []
    for i in range(len(sid)):
        s = int(sid[i])
        r = all_sport[SPORTS[s]]["records"][int(pidx[i])]
        recs.append(r)
    M["records"] = recs

    nn_idx, nn_sim = cross_sport_nn(z, sid)
    hits = arch[nn_idx] == arch
    g4 = float(hits.mean())

    # BASELINE, COMPUTED, and imported rather than re-derived — this was a fourth copy
    # of the same loop. The 0.60 bar was stated without a baseline, and an archetype hit
    # rate is only interpretable against the chance that a random OTHER-SPORT row shares
    # the archetype (~0.1712 given the real mix). All three nulls in
    # check_gate_nonvacuity.py land on it, which validates both at once.
    g4_baseline = g4_random_baseline(M)

    # per-sport + per-arch breakdown
    per_sport = {}
    for s in range(3):
        m = sid == s
        per_sport[SPORTS[s]] = {
            "n": int(m.sum()),
            "hit_rate": float(hits[m].mean()),
            "mean_nn_cos": float(nn_sim[m].mean()),
        }
    per_arch = {}
    for a in np.unique(arch):
        m = arch == a
        per_arch[ARCH_NAMES[int(a)]] = {
            "n": int(m.sum()),
            "hit_rate": float(hits[m].mean()),
        }

    panel = named_panel(z, M, names_per_sport=3)
    # attach each panelist's top-3 cross-sport NN
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    sim_all = zn @ zn.T
    np.fill_diagonal(sim_all, -np.inf)
    for p in panel:
        # find this panelist's row index by matching record fields
        i = next(
            (
                k
                for k, r in enumerate(recs)
                if r["name"] == p["name"] and r["season"] == p["season"] and SPORTS[sid[k]] == p["sport"]
            ),
            None,
        )
        if i is None:
            p["nn"] = []
            continue
        sims = sim_all[i].copy()
        sims[sid == sid[i]] = -np.inf  # other sports only
        top3 = np.argsort(-sims)[:3]
        p["nn"] = [
            {
                "name": recs[j]["name"],
                "sport": SPORTS[int(sid[j])],
                "pos": str(recs[j]["pos"]),
                "team": recs[j].get("team", ""),
                "arch": ARCH_NAMES[int(arch[j])],
                "cos": float(sims[j]),
            }
            for j in top3
        ]

    report = {
        "model": model_label,
        "z_source": z_source,
        "n_rows": int(z.shape[0]),
        "d_emb": int(z.shape[1]),
        "G4_cross_sport_nn_role_coherence": {
            "hit_rate": round(g4, 4),
            "random_baseline": round(g4_baseline, 4),
            "lift_over_random": round(g4 - g4_baseline, 4),
            "target": 0.60,
            "pass": bool(g4 >= 0.60),
            "per_sport": {
                k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                for k, v in per_sport.items()
            },
            "per_arch": {
                k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                for k, v in per_arch.items()
            },
        },
        "panel": panel[:24],
    }
    (DATA / "analogy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Cross-sport analogy panel (G4) ===")
    print(
        f"G4 cross-sport NN role-coherence: hit_rate={g4:.3f} "
        f"vs random baseline {g4_baseline:.3f} (lift {g4 - g4_baseline:+.3f}), "
        f"target 0.60 {'PASS' if g4 >= 0.60 else 'FAIL'}"
    )
    for s, d in per_sport.items():
        print(f"  {s:8s} n={d['n']:>5}  hit={d['hit_rate']:.3f}  mean_nn_cos={d['mean_nn_cos']:.3f}")
    print("\nPer-archetype hit-rate:")
    for a, d in per_arch.items():
        print(f"  {a}: n={d['n']:>5}  hit={d['hit_rate']:.3f}")
    print("\nNamed panel (top-3 cross-sport neighbours):")
    for p in panel[:24]:
        nn_str = " | ".join(f"{n['name']}({n['sport'][:2]} {n['pos']}/{n['arch']}){n['cos']:.2f}" for n in p["nn"][:3])
        print(f"  [{p['sport'][:2]}] {p['name']:<22} {p['pos']}/{p['arch']}  ->  {nn_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
