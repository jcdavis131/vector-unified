"""Vector Unified — Stage 2 canonical eval.

Reconstructs the Stage 2 model (drifted per-sport encoders from enc_states +
best unified trunk) and reports the full gate panel on the LIVE model:
  G1 — per-sport encoder non-regression (live e_s role+pos) AND trunk z role+pos
  G2 — sport-invariance (sport-acc on z, effective rank)
  G3 — cross-sport archetype silhouette
  G4 — curated analogy triples (cross-sport arch agreement + retrieval)
-> data/stage2_report.json

Stage 1 v0.1 stays shipped unless G2 passes AND G1 holds (see train_stage2 verdict).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from train_unified import UnifiedTrunk, effective_rank, load_matrix, SPORTS, SEED, DATA, UCACHE
from eval_unified import knn5_acc
from load_live_encoders import load_live, DEVICE_DEF
from load_encoders import load_all

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def reconstruct_stage2(device):
    M = load_matrix(device)
    ck = torch.load(UCACHE / "unified_stage2_best.pt", map_location=device, weights_only=False)
    a = ck["args"]
    sport_dims = [int(M["E"][s].shape[1]) for s in range(3)]
    n_pos = [M["n_pos"][s] for s in SPORTS]
    model = UnifiedTrunk(sport_dims, n_seasons_era=ck["n_eras"],
                         d_adapter=a["d_adapter"], d_sport_tok=a["d_sport_tok"],
                         d_emb=a["d_emb"], n_arch=8, n_pos=n_pos, dropout=0.2).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    live = load_live(device)
    enc_states = ck["enc_states"]
    for sport in SPORTS:
        live[sport].model.load_state_dict(enc_states[sport])
        live[sport].model.eval()
    return live, model, M, ck


def live_e_s(live, device):
    return {s: live[s].encode_full_numpy(device) for s in SPORTS}


def full_z(model, live, M, device, Es):
    e_per = [torch.tensor(Es[s], dtype=torch.float32, device=device) for s in SPORTS]
    with torch.no_grad():
        z = model.encode(e_per, M["sport_id"], M["era_id"])
    return z.cpu().numpy().astype(np.float32)


def g1(live, M, Es, z_full):
    sid = M["sport_id"].cpu().numpy()
    native = M["native"].cpu().numpy()
    pos = M["pos_id"].cpu().numpy()
    posm = M["pos_mask"].cpu().numpy()
    out = {}
    for s, sport in enumerate(SPORTS):
        idx = np.where(sid == s)[0]
        e = Es[sport]
        out[sport] = {
            "n": int(len(idx)),
            "role_knn5_e_s": knn5_acc(e, native[idx]),
            "pos_knn5_e_s": knn5_acc(e, pos[idx], posm[idx]) if posm[idx].any() else None,
            "role_knn5_z": knn5_acc(z_full[idx], native[idx]),
            "pos_knn5_z": knn5_acc(z_full[idx], pos[idx], posm[idx]) if posm[idx].any() else None,
        }
    return out


def g2(z_full, M):
    sid = M["sport_id"].cpu().numpy()
    Xtr, Xte, ytr, yte = train_test_split(z_full, sid, test_size=0.2,
                                          random_state=SEED, stratify=sid)
    clf = LogisticRegression(max_iter=400, C=1.0)
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xte, yte))
    rank = float(effective_rank(torch.tensor(z_full)))
    return {"sport_acc": round(acc, 4), "chance": round(1.0 / 3.0, 4),
            "delta_vs_chance": round(acc - 1.0 / 3.0, 4),
            "effective_rank": round(rank, 1),
            "rank_nondeg_pass": bool(rank >= 12),
            "g2_pass": bool(acc <= 1.0 / 3.0 + 0.10)}


def g3(z_full, M, sample=6000):
    arch = M["arch_id"].cpu().numpy()
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(arch), min(len(arch), sample), replace=False)
    return {"silhouette": round(float(silhouette_score(z_full[sel], arch[sel], metric="cosine")), 4)}


def g4_analogy(z_full, M, records_by_sport):
    """Curated analogy triples on the Stage 2 z. Mirrors analogy_triples_eval."""
    T = json.loads((DATA / "analogy_triples.json").read_text(encoding="utf-8"))
    sid = M["sport_id"].cpu().numpy()
    arch = M["arch_id"].cpu().numpy()
    pidx = M["player_idx"].cpu().numpy()
    # build per-global-row (name, sport)
    names = []
    for i in range(len(sid)):
        sport = SPORTS[int(sid[i])]
        names.append(records_by_sport[sport][int(pidx[i])]["name"])
    sport_names = np.array([SPORTS[int(s)] for s in sid])
    E = z_full / (np.linalg.norm(z_full, axis=1, keepdims=True) + 1e-9)
    idx = defaultdict(list)
    for i, nm in enumerate(names):
        idx[(nm, sport_names[i])].append(i)
    n = len(names)
    rows = []; hits = 0; arch_agree = 0; intu_a = 0; nb = 0
    ranks = []
    for t in T["triples"]:
        a_key = (t["a"]["name"], t["a"]["sport"]); b_key = (t["b"]["name"], t["b"]["sport"])
        a_rows = idx.get(a_key, []); b_rows = idx.get(b_key, [])
        if not a_rows or not b_rows:
            rows.append({"triple": t, "status": "MISSING"}); continue
        a = a_rows[0]; sa = sport_names[a]
        sims = E @ E[a]; sims[np.arange(n) == a] = -np.inf; sims[sport_names == sa] = -np.inf
        order = np.argsort(-sims); top10 = order[:10]
        rank_of_b = None
        for br in b_rows:
            r = int(np.where(order == br)[0][0])
            if rank_of_b is None or r < rank_of_b:
                rank_of_b = r
        in_top10 = rank_of_b is not None and rank_of_b < 10
        a_arch = int(arch[a]); b_arch = int(arch[b_rows[0]])
        agree = a_arch == b_arch
        im = t["role_intuitive"] == a_arch
        if in_top10: hits += 1
        if agree: arch_agree += 1
        if im: intu_a += 1
        nb += 1
        if rank_of_b is not None: ranks.append(rank_of_b)
        rows.append({"a": t["a"], "b": t["b"], "role_intuitive": t["role_intuitive"],
                     "a_arch": a_arch, "b_arch": b_arch, "arch_agree": bool(agree),
                     "intuition_matches_a_arch": bool(im),
                     "b_best_rank": rank_of_b, "in_top10": bool(in_top10)})
    random_rank = (n - 1) / 2.0
    return {"n": nb, "arch_agreement": round(arch_agree / max(1, nb), 4),
            "retrieval_top10_hit_rate": round(hits / max(1, nb), 4),
            "mean_b_rank": round(float(np.mean(ranks)), 1) if ranks else None,
            "better_than_random_ratio": round(float(np.mean(ranks)) / random_rank, 3) if ranks else None,
            "intuition_a_match": round(intu_a / max(1, nb), 4),
            "rows": rows}


def main():
    device = DEVICE_DEF
    print(f"device={device}")
    live, model, M, ck = reconstruct_stage2(device)
    print(f"best_epoch={ck.get('best_epoch')} best_g2={ck.get('best_g2')}")
    Es = live_e_s(live, device)
    z_full = full_z(model, live, M, device, Es)
    print(f"z shape={z_full.shape} norm={np.linalg.norm(z_full,axis=1).mean():.4f}")

    print("\n=== G1 (encoder non-regression + trunk z) ===")
    g1r = g1(live, M, Es, z_full)
    for sport in SPORTS:
        g = g1r[sport]
        print(f"  {sport:9s} e_s[role={g['role_knn5_e_s']:.4f} pos={g['pos_knn5_e_s']}] "
              f"z[role={g['role_knn5_z']:.4f} pos={g['pos_knn5_z']}]")

    print("\n=== G2 (sport-invariance) ===")
    g2r = g2(z_full, M)
    print(f"  sport_acc={g2r['sport_acc']} chance={g2r['chance']} "
          f"delta={g2r['delta_vs_chance']:+.4f} rank={g2r['effective_rank']} "
          f"g2_pass={g2r['g2_pass']}")

    print("\n=== G3 (cross-sport archetype silhouette) ===")
    g3r = g3(z_full, M)
    print(f"  silhouette={g3r['silhouette']}")

    print("\n=== G4 (curated analogy triples) ===")
    frozen = load_all(verbose=False)
    records_by_sport = {s: frozen[s]["records"] for s in SPORTS}
    g4r = g4_analogy(z_full, M, records_by_sport)
    print(f"  n={g4r['n']} arch_agreement={g4r['arch_agreement']} "
          f"retrieval_top10={g4r['retrieval_top10_hit_rate']} "
          f"mean_b_rank={g4r['mean_b_rank']} better_than_random={g4r['better_than_random_ratio']}x")

    report = {
        "best_epoch": ck.get("best_epoch"), "best_g2": ck.get("best_g2"),
        "verdict_from_train": ck.get("verdict"),
        "baselines_stage0": ck.get("baselines"),
        "g1": g1r, "g2": g2r, "g3": g3r, "g4_curated": {k: v for k, v in g4r.items() if k != "rows"},
        "g4_curated_rows": g4r["rows"],
        "shippable": bool(g2r["g2_pass"] and all(
            (ck["verdict"].get(s, {}).get("role_ok") and ck["verdict"].get(s, {}).get("pos_ok"))
            for s in SPORTS) if ck.get("verdict") else g2r["g2_pass"]),
        "note": "Stage 1 v0.1 remains shipped unless shippable=True. Per-sport assets untouched (read-only).",
    }
    (DATA / "stage2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nshippable={report['shippable']}  -> wrote data/stage2_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
