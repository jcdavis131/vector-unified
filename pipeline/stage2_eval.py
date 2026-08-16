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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _torch_safe import safe_torch_load
from eval_unified import knn5_acc
from load_encoders import load_all
from load_live_encoders import DEVICE_DEF, load_live
from train_unified import (
    DATA,
    SEED,
    SPORTS,
    UCACHE,
    UnifiedTrunk,
    effective_rank,
    load_matrix,
)

# index -> cross-sport archetype LABEL ("A0", "A1", ...). Needed because
# analogy_triples.json's role_intuitive is a label and arch_id is an index; comparing them
# directly is always False. See the note at the `im =` line.
ARCH_NAMES = json.loads((DATA / "unified_meta.json").read_text(encoding="utf-8"))["arch_names"]

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def reconstruct_stage2(device):
    M = load_matrix(device)
    ck = safe_torch_load(UCACHE / "unified_stage2_best.pt", map_location=device)
    a = ck["args"]
    sport_dims = [int(M["E"][s].shape[1]) for s in range(3)]
    n_pos = [M["n_pos"][s] for s in SPORTS]
    model = UnifiedTrunk(
        sport_dims,
        n_seasons_era=ck["n_eras"],
        d_adapter=a["d_adapter"],
        d_sport_tok=a["d_sport_tok"],
        d_emb=a["d_emb"],
        n_arch=8,
        n_pos=n_pos,
        dropout=0.2,
    ).to(device)
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
    Xtr, Xte, ytr, yte = train_test_split(z_full, sid, test_size=0.2, random_state=SEED, stratify=sid)
    clf = LogisticRegression(max_iter=400, C=1.0)
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xte, yte))
    rank = float(effective_rank(torch.tensor(z_full)))
    # THE OLD TARGET WAS NOT MERELY WRONG, IT WAS UNREACHABLE. `g2_pass` was
    # `acc <= 1/3 + 0.10` = 0.4333, but the sports are 12,966 / 5,323 / 2,430 and a
    # classifier that always answers "hoops" scores 0.6258. A perfectly sport-invariant z
    # gives the classifier nothing but the class prior, so 0.6258 is the FLOOR of
    # achievable accuracy — scoring 0.4333 would require z to actively mislead. Stage 2 has
    # been reported SHIPPABLE=False since Phase 4 against a bar no embedding could clear.
    # Confirmed empirically in 7.16: a globally shuffled z scored 0.6257.
    majority = float(np.bincount(sid).max()) / len(sid)
    return {
        "sport_acc": round(acc, 4),
        "chance": round(1.0 / 3.0, 4),
        "delta_vs_chance": round(acc - 1.0 / 3.0, 4),
        "majority_class_share": round(majority, 4),
        "delta_vs_majority": round(acc - majority, 4),
        "g2_target": round(majority + 0.10, 4),
        "SUPERSEDED_g2_target_chance_plus_10": round(1.0 / 3.0 + 0.10, 4),
        "effective_rank": round(rank, 1),
        "rank_nondeg_pass": bool(rank >= 12),
        "rank_note": (
            "Detects collapse only. Effective rank is permutation-invariant, "
            "so no shuffle null can test it, and random gaussian rows score "
            "HIGHER (64.0) than the real embedding (12.4). See 7.19."
        ),
        "g2_pass": bool(acc <= majority + 0.10),
    }


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
    rows = []
    hits = 0
    arch_agree = 0
    intu_a = 0
    nb = 0
    ranks = []
    for t in T["triples"]:
        a_key = (t["a"]["name"], t["a"]["sport"])
        b_key = (t["b"]["name"], t["b"]["sport"])
        a_rows = idx.get(a_key, [])
        b_rows = idx.get(b_key, [])
        if not a_rows or not b_rows:
            rows.append({"triple": t, "status": "MISSING"})
            continue
        a = a_rows[0]
        sa = sport_names[a]
        sims = E @ E[a]
        sims[np.arange(n) == a] = -np.inf
        sims[sport_names == sa] = -np.inf
        order = np.argsort(-sims)
        top10 = order[:10]
        rank_of_b = None
        for br in b_rows:
            r = int(np.where(order == br)[0][0])
            if rank_of_b is None or r < rank_of_b:
                rank_of_b = r
        in_top10 = rank_of_b is not None and rank_of_b < 10
        a_arch = int(arch[a])
        b_arch = int(arch[b_rows[0]])
        agree = a_arch == b_arch
        # TYPE MISMATCH, and it made this metric structurally zero. role_intuitive is a
        # STRING label like "A0"; a_arch is an int index. `"A0" == 0` is always False, so
        # intuition_a_match has reported exactly 0.0000 in every Stage 2 report ever
        # written — a hard zero that reads as "the model never matches human intuition"
        # when it means "these two values were never comparable". The same quantity in
        # analogy_triples_eval.py, which compares string to string, is 0.825.
        im = str(t["role_intuitive"]) == str(ARCH_NAMES[a_arch])
        if in_top10:
            hits += 1
        if agree:
            arch_agree += 1
        if im:
            intu_a += 1
        nb += 1
        if rank_of_b is not None:
            ranks.append(rank_of_b)
        rows.append(
            {
                "a": t["a"],
                "b": t["b"],
                "role_intuitive": t["role_intuitive"],
                "a_arch": a_arch,
                "b_arch": b_arch,
                "arch_agree": bool(agree),
                "intuition_matches_a_arch": bool(im),
                "b_best_rank": rank_of_b,
                "in_top10": bool(in_top10),
                "_a_sport": sa,
                "_n_b_rows": len(b_rows),
            }
        )
    # RANDOM-RANK BASELINE, CORRECTED — two errors in one line. `(n - 1) / 2.0` used the
    # FULL pool when the ranking is over cross-sport rows only, and it treated B as a
    # single row when b_best_rank is the MINIMUM over all of B's rows. E[min of k uniform
    # draws from N] is (N-k)/(k+1). See analogy_triples_eval.py (7.18), where the same
    # defect turned "3.23x better than random" into 0.98x.
    #
    # DIRECTION ALSO FIXED. This file computed mean/random and analogy_triples_eval.py
    # computes random/mean, under the SAME field name `better_than_random_ratio`, so the
    # two reports disagreed on whether higher is better. Higher-is-better wins, matching
    # the name.
    exp_ranks = []
    for r in rows:
        if r.get("b_best_rank") is None:
            continue
        pool = int((sport_names != r["_a_sport"]).sum())
        k = r["_n_b_rows"]
        exp_ranks.append((pool - k) / (k + 1))
    random_rank = float(np.mean(exp_ranks)) if exp_ranks else None
    mean_rank = float(np.mean(ranks)) if ranks else None
    btr = (random_rank / mean_rank) if (random_rank and mean_rank) else None
    for r in rows:
        r.pop("_a_sport", None)
        r.pop("_n_b_rows", None)
    return {
        "n": nb,
        "arch_agreement": round(arch_agree / max(1, nb), 4),
        "retrieval_top10_hit_rate": round(hits / max(1, nb), 4),
        "mean_b_rank": round(mean_rank, 1) if mean_rank else None,
        "random_expected_rank": round(random_rank, 1) if random_rank else None,
        "better_than_random_ratio": round(btr, 3) if btr else None,
        "ratio_note": (
            "random/mean, so HIGHER is better and 1.0 is chance. This file "
            "previously computed mean/random under the same field name, "
            "which inverted it relative to analogy_triples_eval.py."
        ),
        "intuition_a_match": round(intu_a / max(1, nb), 4),
        "rows": rows,
    }


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
        print(
            f"  {sport:9s} e_s[role={g['role_knn5_e_s']:.4f} pos={g['pos_knn5_e_s']}] "
            f"z[role={g['role_knn5_z']:.4f} pos={g['pos_knn5_z']}]"
        )

    print("\n=== G2 (sport-invariance) ===")
    g2r = g2(z_full, M)
    print(
        f"  sport_acc={g2r['sport_acc']} chance={g2r['chance']} "
        f"delta={g2r['delta_vs_chance']:+.4f} rank={g2r['effective_rank']} "
        f"g2_pass={g2r['g2_pass']}"
    )

    print("\n=== G3 (cross-sport archetype silhouette) ===")
    g3r = g3(z_full, M)
    print(f"  silhouette={g3r['silhouette']}")

    print("\n=== G4 (curated analogy triples) ===")
    frozen = load_all(verbose=False)
    records_by_sport = {s: frozen[s]["records"] for s in SPORTS}
    g4r = g4_analogy(z_full, M, records_by_sport)
    print(
        f"  n={g4r['n']} arch_agreement={g4r['arch_agreement']} "
        f"retrieval_top10={g4r['retrieval_top10_hit_rate']} "
        f"mean_b_rank={g4r['mean_b_rank']} better_than_random={g4r['better_than_random_ratio']}x"
    )

    report = {
        "best_epoch": ck.get("best_epoch"),
        "best_g2": ck.get("best_g2"),
        "verdict_from_train": ck.get("verdict"),
        "baselines_stage0": ck.get("baselines"),
        "g1": g1r,
        "g2": g2r,
        "g3": g3r,
        "g4_curated": {k: v for k, v in g4r.items() if k != "rows"},
        "g4_curated_rows": g4r["rows"],
        "shippable": bool(
            g2r["g2_pass"]
            and all(
                (ck["verdict"].get(s, {}).get("role_ok") and ck["verdict"].get(s, {}).get("pos_ok")) for s in SPORTS
            )
            if ck.get("verdict")
            else g2r["g2_pass"]
        ),
        "note": "Stage 1 v0.1 remains shipped unless shippable=True. Per-sport assets untouched (read-only).",
    }
    (DATA / "stage2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nshippable={report['shippable']}  -> wrote data/stage2_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
