"""Vector Unified — Stage 1 evaluation harness (G1-G3).

Loads the saved unified_best.pt, encodes every player-season into z (64-d), and
scores the three automatic gates from UNIFIED_ARCHITECTURE §6:

  G1  Per-sport non-inferiority (hard)  — does z recover each sport's native role
      (kNN-5 native-cluster + position acc) at least as well as its FROZEN e_s?
      The games must not break: z must not lose role info each sport needs.
  G2  Sport-invariance (hard)          — 3-way sport classifier on z, scored against
      the MAJORITY-CLASS share (0.6258), not uniform chance + effective rank of z.
  G3  Cross-sport archetype coherence   — silhouette over cross-sport arch labels
      on z, and (within-arch cross-sport cosine - between-arch) above SEP_FLOOR.

Output: data/unified_report.json + a printed VERDICT.

TWO THRESHOLDS WERE WRONG UNTIL 7.16, both found by check_gate_nonvacuity.py:

  * G2 quoted accuracy against `chance = 1/3`. The sports are 12,966 / 5,323 / 2,430,
    so always answering "hoops" scores 0.6258 — and a globally shuffled z, carrying no
    sport information whatsoever, scored 0.6257. Real leakage is +0.130 over majority,
    not +0.422 over uniform. `chance` and `delta_vs_chance` are still emitted so older
    reports stay readable; quote `delta_vs_majority`.

  * G3's separation test was the bare inequality `within > between`, which holds on a
    null about half the time by construction — measured up to +0.0440 across 50
    within-sport shuffles with the archetype labels randomised. The real value is
    +0.8448, so the finding was never in doubt, but the TEST could not tell them apart.
    Now a calibrated SEP_FLOOR.

Honest note: the G2 "no-debiasing baseline" (a no-GRL control trunk) is not trained
here; the control remains a deferred sub-check.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from load_encoders import ROOT, UCACHE, SPORTS
from train_unified import UnifiedTrunk, effective_rank

DATA = ROOT / "data"
SEED = 7
# Above the largest separation observed across 50 within-sport shuffles (+0.0440).
# Provenance: data/gate_nonvacuity.json.
SEP_FLOOR = 0.05


def load_model(device, ckpt_name="unified_best.pt"):
    ck = torch.load(UCACHE / ckpt_name, map_location=device, weights_only=False)
    a = ck["args"]
    d_st = a["d_sport_tok"] if isinstance(a, dict) and "d_sport_tok" in a else getattr(a, "d_sport_tok", 8)
    sh = a["shared_adapter"] if isinstance(a, dict) and "shared_adapter" in a else getattr(a, "shared_adapter", False)
    mk = a["market"] if isinstance(a, dict) and "market" in a else getattr(a, "market", False)
    ct = a.get("cultural_text", False) if isinstance(a, dict) else getattr(a, "cultural_text", False)
    d_text = a.get("d_text", 384) if isinstance(a, dict) else getattr(a, "d_text", 384)
    model = UnifiedTrunk(sport_dims=ck["sport_dim"], n_seasons_era=ck["n_eras"],
                         d_adapter=a["d_adapter"], d_sport_tok=d_st, d_emb=a["d_emb"], n_arch=8,
                         n_pos=ck["n_pos"], dropout=a["dropout"], shared_adapter=sh,
                         market_heads=mk, cultural_text=ct, d_text=d_text).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck


def encode_all(model, M, device):
    e_per = [M["E"][s] for s in range(3)]
    with torch.no_grad():
        z = model.encode(e_per, M["sport_id"], M["era_id"])
    return z.cpu().numpy().astype(np.float32)


def knn5_acc(emb, labels, mask=None):
    """Stratified 80/20 kNN-5 accuracy. emb/labels already L2/filtered to one sport."""
    if mask is not None:
        emb, labels = emb[mask], labels[mask]
    if len(labels) < 50:
        return None
    try:
        Xtr, Xte, ytr, yte = train_test_split(emb, labels, test_size=0.2,
                                              random_state=SEED, stratify=labels)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(emb, labels, test_size=0.2, random_state=SEED)
    clf = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def g1_per_sport(z_full, M):
    """z vs frozen e_s: kNN-5 native-cluster + position accuracy per sport."""
    out = {}
    sid = M["sport_id"].cpu().numpy()
    for s in range(3):
        m = sid == s
        idx = np.where(m)[0]
        z_s = z_full[idx]
        e_s = M["E"][s].cpu().numpy()
        native = M["native"].cpu().numpy()[idx]
        pos = M["pos_id"].cpu().numpy()[idx]
        posm = M["pos_mask"].cpu().numpy()[idx]
        out[SPORTS[s]] = {
            "n": int(m.sum()),
            "native_knn5_e_s": knn5_acc(e_s, native),
            "native_knn5_z": knn5_acc(z_s, native),
            "pos_knn5_e_s": knn5_acc(e_s, pos, posm) if posm.any() else None,
            "pos_knn5_z": knn5_acc(z_s, pos, posm) if posm.any() else None,
        }
    return out


def g2_sport_invariance(z_full, M):
    sid = M["sport_id"].cpu().numpy()
    Xtr, Xte, ytr, yte = train_test_split(z_full, sid, test_size=0.2,
                                          random_state=SEED, stratify=sid)
    clf = LogisticRegression(max_iter=400, C=1.0)
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xte, yte))
    rank = effective_rank(torch.tensor(z_full))
    target = z_full.shape[1] // 2  # literal G2 floor (collapse heuristic)
    nondeg = 12  # non-degenerate floor: below this with role/folding loss = collapse
    # MAJORITY, NOT 1/3. The three sports are not balanced — 12,966 hoops / 5,323 gridiron
    # / 2,430 pitch — so a classifier that always answers "hoops" scores 0.6258, and
    # quoting accuracy against uniform chance overstates the leakage by 29 points. This was
    # found by check_gate_nonvacuity.py: a globally shuffled z, which carries no sport
    # information at all, still scored 0.6257. `chance` and `delta_vs_chance` are KEPT so
    # older reports remain readable, but the majority figures are the ones to quote.
    majority = float(np.bincount(sid).max()) / len(sid)
    return {"sport_acc": round(acc, 4), "chance": round(1.0 / 3.0, 4),
            "delta_vs_chance": round(acc - 1.0 / 3.0, 4),
            "majority_class_share": round(majority, 4),
            "delta_vs_majority": round(acc - majority, 4),
            "baseline_note": ("Quote delta_vs_majority. delta_vs_chance assumes balanced "
                              "classes and these are 62.6 / 25.7 / 11.7."),
            "effective_rank": round(rank, 1), "rank_target_literal": target,
            "rank_literal_pass": bool(rank >= target),
            "rank_nondeg_floor": nondeg, "rank_nondeg_pass": bool(rank >= nondeg),
            "note": "rank_literal=d_emb/2 (heuristic). collapse_detector = rank>=12 AND G1 AND G3 (rank alone over-alarms on a genuinely low-d role manifold). no-GRL baseline via --baseline-sport-acc."}


def g3_silhouette(z_full, M):
    arch = M["arch_id"].cpu().numpy()
    sid = M["sport_id"].cpu().numpy()
    sample = min(len(arch), 6000)
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(arch), sample, replace=False)
    sil = float(silhouette_score(z_full[sel], arch[sel], metric="cosine"))
    # per-arch within vs between CROSS-SPORT cosine
    zn = z_full / (np.linalg.norm(z_full, axis=1, keepdims=True) + 1e-9)
    within, between = [], []
    rng2 = np.random.default_rng(SEED)
    for a in np.unique(arch):
        ia = np.where(arch == a)[0]
        if len(ia) < 2:
            continue
        # sample pairs across sports
        for _ in range(800):
            i, j = rng2.choice(ia, 2, replace=False)
            if sid[i] != sid[j]:
                within.append(float(zn[i] @ zn[j]))
    # between-arch cross-sport pairs
    for _ in range(800):
        a1, a2 = rng2.choice(np.unique(arch), 2, replace=False)
        i = rng2.choice(np.where(arch == a1)[0])
        j = rng2.choice(np.where(arch == a2)[0])
        if sid[i] != sid[j]:
            between.append(float(zn[i] @ zn[j]))
    within_m = float(np.mean(within)) if within else 0.0
    between_m = float(np.mean(between)) if between else 0.0
    # CALIBRATED FLOOR, not a bare inequality. `within_m > between_m` on a noisy
    # difference passes on a null roughly half the time by construction, and it did:
    # check_gate_nonvacuity.py measured separations up to +0.0440 across 50 within-sport
    # shuffles (95th percentile +0.0301), where the archetype labels had been randomised
    # inside each sport. The construct was never in doubt — the real value is +0.8448 —
    # but the TEST could not tell the two apart. SEP_FLOOR is set above the observed null
    # maximum; re-derive it with `python pipeline/check_gate_nonvacuity.py` if the
    # embedding or the label set changes.
    return {"silhouette": round(sil, 4), "silhouette_pass": bool(sil > 0),
            "within_arch_cross_sport_cos": round(within_m, 4),
            "between_arch_cross_sport_cos": round(between_m, 4),
            "separation": round(within_m - between_m, 4),
            "separation_floor": SEP_FLOOR,
            "separation_pass": bool(within_m - between_m > SEP_FLOOR)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-sport-acc", type=float, default=None,
                    help="no-GRL control sport-acc; G2-sport passes if (baseline - acc) >= 0.10")
    ap.add_argument("--ckpt", default="unified_best.pt",
                    help="checkpoint filename under pipeline/data to evaluate")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from train_unified import load_matrix
    M = load_matrix(device)
    model, ck = load_model(device, args.ckpt)
    z = encode_all(model, M, device)
    assert not np.isnan(z).any() and np.allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-4)

    g1 = g1_per_sport(z, M)
    g2 = g2_sport_invariance(z, M)
    g3 = g3_silhouette(z, M)

    # G1 verdict: z non-inferior to e_s (within 2pp) on native cluster + position
    g1_pass = True
    for s, d in g1.items():
        if d["native_knn5_e_s"] is not None and d["native_knn5_z"] < d["native_knn5_e_s"] - 0.02:
            g1_pass = False
        if d["pos_knn5_e_s"] is not None and d["pos_knn5_z"] < d["pos_knn5_e_s"] - 0.02:
            g1_pass = False

    g3_ok = g3["silhouette_pass"] and g3["separation_pass"]
    # G2 = sport-invariance (relative to no-GRL baseline) AND not-collapsed
    if args.baseline_sport_acc is not None:
        g2["baseline_sport_acc"] = args.baseline_sport_acc
        g2["delta_vs_baseline"] = round(args.baseline_sport_acc - g2["sport_acc"], 4)
        sport_pass = g2["delta_vs_baseline"] >= 0.10
    else:
        sport_pass = None  # deferred: needs no-GRL control
    collapse_pass = bool(g2["rank_nondeg_pass"] and g1_pass and g3_ok)
    g2_pass = (sport_pass in (True,)) and collapse_pass

    report = {
        "model": "UnifiedTrunk Stage 1 (frozen encoders)",
        "checkpoint_rank": ck.get("best_rank"),
        "n_rows": int(z.shape[0]), "d_emb": int(z.shape[1]),
        "G1_per_sport_noninferiority": g1, "G1_pass": g1_pass,
        "G2_sport_invariance": g2,
        "G3_cross_sport_archetype": g3,
        "verdict": {
            "G1": "PASS" if g1_pass else "FAIL",
            "G2": "PASS" if g2_pass else ("FAIL" if sport_pass is False else "DEFERRED(collapse_pass={}, need no-GRL baseline)".format(collapse_pass)),
            "G3": "PASS" if g3_ok else "FAIL",
            "collapse_detector": "PASS" if collapse_pass else "FAIL",
        },
    }
    (DATA / "unified_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Unified Stage 1 eval ===")
    print(f"z shape={z.shape}  ckpt_rank={ck.get('best_rank')}")
    print("\nG1 per-sport non-inferiority (z vs frozen e_s, kNN-5):")
    for s, d in g1.items():
        ne, nz = d["native_knn5_e_s"], d["native_knn5_z"]
        pe, pz = d["pos_knn5_e_s"], d["pos_knn5_z"]
        print(f"  {s:8s} n={d['n']:>5}  native e_s={ne:.3f} z={nz:.3f} (d{nz-ne:+.3f})"
              f"  pos e_s={pe if pe is None else round(pe,3)} z={pz if pz is None else round(pz,3)}")
    bmsg = ""
    if args.baseline_sport_acc is not None:
        bmsg = f"  baseline={args.baseline_sport_acc:.3f} dVsBase={g2['delta_vs_baseline']:+.3f}"
    print(f"\nG2 sport-invariance: acc={g2['sport_acc']:.3f} vs MAJORITY "
          f"{g2['majority_class_share']:.3f} = {g2['delta_vs_majority']:+.3f} "
          f"(uniform-chance framing, kept for older reports: "
          f"dVsChance{g2['delta_vs_chance']:+.3f}){bmsg}  rank={g2['effective_rank']:.1f} "
          f"(literal>{g2['rank_target_literal']}: {'PASS' if g2['rank_literal_pass'] else 'FAIL'}; "
          f"nondeg>{g2['rank_nondeg_floor']}: {'PASS' if g2['rank_nondeg_pass'] else 'FAIL'})")
    print(f"G3 cross-sport archetype: silhouette={g3['silhouette']:.4f} "
          f"{'PASS' if g3['silhouette_pass'] else 'FAIL'} | within-cos={g3['within_arch_cross_sport_cos']:.4f} "
          f"> between-cos={g3['between_arch_cross_sport_cos']:.4f} "
          f"{'PASS' if g3['separation_pass'] else 'FAIL'}")
    print(f"\nVERDICT: G1={report['verdict']['G1']}  G2={report['verdict']['G2']}  G3={report['verdict']['G3']}  "
          f"collapse_detector={report['verdict']['collapse_detector']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
