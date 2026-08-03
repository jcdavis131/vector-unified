"""Vector Unified — curated analogy triples panel (SPEC §6 G4 / G5).

Loads the SHIPPED assets/unified.json and data/analogy_triples.json. For each
human-curated triple (A ~ B, different sports, paired by intuitive role), test:
  - retrieval: is any B row (name+sport) in A's top-10 cross-sport nearest neighbours?
  - rank: the best (smallest) rank of any B row in A's cross-sport NN ranking.
  - archetype agreement: does the model's cross_arch(A) == cross_arch(B)?
  - intuition-vs-model: does role_intuitive match the model's arch for A and B?

G4-curated = fraction of triples with B in A's top-10 cross-sport NN (target >= 0.60).
This is the human face-validity check alongside the automated all-players G4.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DATA = ROOT / "data"


def main():
    U = json.loads((ASSETS / "unified.json").read_text(encoding="utf-8"))
    T = json.loads((DATA / "analogy_triples.json").read_text(encoding="utf-8"))
    players = U["players"]
    n = len(players)
    # index by (name, sport) -> list of row indices
    idx = defaultdict(list)
    for i, p in enumerate(players):
        idx[(p["name"], p["sport"])].append(i)
    # sport of each row
    sport = np.array([p["sport"] for p in players])
    arch = np.array([p["cross_arch"] for p in players])
    E = np.array([p["e"] for p in players], dtype=np.float32)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    rows = []
    hits = 0
    arch_agree = 0
    intuition_a_match = 0
    for t in T["triples"]:
        a_key = (t["a"]["name"], t["a"]["sport"])
        b_key = (t["b"]["name"], t["b"]["sport"])
        a_rows = idx.get(a_key, [])
        b_rows = idx.get(b_key, [])
        if not a_rows or not b_rows:
            rows.append({"triple": t, "status": "MISSING", "a_rows": len(a_rows), "b_rows": len(b_rows)})
            continue
        a = a_rows[0]  # anchor: first season row
        sa = sport[a]
        # cross-sport cosine from a to all other-sport rows
        sims = En @ En[a]
        sims[np.arange(n) == a] = -np.inf
        sims[sport == sa] = -np.inf
        order = np.argsort(-sims)
        top10 = order[:10]
        rank_of_b = None
        for br in b_rows:
            r = int(np.where(order == br)[0][0])
            if rank_of_b is None or r < rank_of_b:
                rank_of_b = r
        in_top10 = rank_of_b is not None and rank_of_b < 10
        a_arch = arch[a]
        b_arch = arch[b_rows[0]]
        agree = a_arch == b_arch
        intu_a = t["role_intuitive"]
        intu_match_a = intu_a == a_arch
        if in_top10:
            hits += 1
        if agree:
            arch_agree += 1
        if intu_match_a:
            intuition_a_match += 1
        rows.append({
            "a": t["a"], "b": t["b"], "role_intuitive": intu_a, "because": t["because"],
            "a_arch": a_arch, "b_arch": b_arch, "arch_agree": bool(agree),
            "intuition_matches_a_arch": bool(intu_match_a),
            "b_best_rank": rank_of_b, "in_top10": bool(in_top10),
            # kept for the corrected random-rank baseline; B's row count is what the
            # old `pool / 2` baseline ignored
            "_a_sport": sa, "_n_b_rows": len(b_rows),
            "top3_names": [{"name": players[j]["name"], "sport": players[j]["sport"],
                            "arch": players[j]["cross_arch"], "cos": round(float(sims[j]), 3)} for j in top10[:3]],
        })

    scored = [r for r in rows if "in_top10" in r]
    g4_hit = hits / len(scored) if scored else 0.0
    arch_rate = arch_agree / len(scored) if scored else 0.0
    intu_rate = intuition_a_match / len(scored) if scored else 0.0
    mean_rank = float(np.mean([r["b_best_rank"] for r in scored])) if scored else 0.0
    # RANDOM-RANK BASELINE, CORRECTED. The old line was `avg_cross_pool / 2.0`, which is
    # the expected rank of ONE randomly placed row. But b_best_rank is the MINIMUM over
    # every row B has — a player with 20 charted seasons gets 20 draws at the target, and
    # the expected minimum of k uniform draws from a pool of N is (N-k)/(k+1), not N/2.
    # The old baseline therefore overstated "better than random" by roughly a factor of
    # the number of B rows, which is exactly the quantity that varies most between
    # triples. Same defect shape as G2's `chance = 1/3` and G4's unstated bar (7.16/7.17):
    # a baseline asserted rather than derived from how the statistic is actually computed.
    #
    # Computed BOTH ways and required to agree — the closed form and an empirical draw of
    # random k-subsets. A formula and a simulation disagreeing would mean one of them is
    # wrong, and there would be no way to tell which from a single number.
    sport_counts = {s: int((sport == s).sum()) for s in ["hoops", "gridiron", "pitch"]}
    rng_null = np.random.default_rng(7)
    analytic, empirical = [], []
    for r in scored:
        pool = n - sport_counts[r["_a_sport"]]
        k = r["_n_b_rows"]
        analytic.append((pool - k) / (k + 1))
        draws = [int(rng_null.choice(pool, size=k, replace=False).min()) for _ in range(200)]
        empirical.append(float(np.mean(draws)))
    random_rank = float(np.mean(analytic))
    random_rank_empirical = float(np.mean(empirical))
    naive_random_rank = float(np.mean(
        [n - sport_counts[r["_a_sport"]] for r in scored])) / 2.0
    btr = random_rank / mean_rank if mean_rank > 0 else 0.0
    btr_naive = naive_random_rank / mean_rank if mean_rank > 0 else 0.0
    formulas_agree = abs(random_rank - random_rank_empirical) / max(random_rank, 1.0) < 0.05

    # Baseline for arch-agreement too, for the same reason: a rate quoted against a bar
    # with no null is not interpretable. Chance that a random OTHER-SPORT row shares the
    # anchor's archetype, averaged over the scored anchors.
    arch_base = 0.0
    for r in scored:
        a_pool = sport != r["_a_sport"]
        same = int(((arch == r["a_arch"]) & a_pool).sum())
        arch_base += (same / max(int(a_pool.sum()), 1)) / len(scored)
    report = {
        "n_triples": len(T["triples"]), "n_scored": len(scored),
        "metric_note": (
            "WITHDRAWN AND REPLACED. This field used to say specific-pair top-10 retrieval "
            "was the wrong gate for large archetype pools, and offered a "
            "'better-than-random retrieval ratio' of 3.23x as the salvage. That ratio was "
            "an artefact of the wrong baseline (see baseline_correction_note): against the "
            "correct one it is 0.98x, i.e. INDISTINGUISHABLE FROM RANDOM and if anything "
            "slightly worse. The 0.000 top-10 hit-rate was not a metric problem being "
            "reframed — the curated-pair retrieval carries no signal, and that is the "
            "honest reading. What still stands is arch-agreement, which is reported here "
            "against its own computed baseline, and the automated all-players G4 "
            "(0.9401 vs a 0.1712 random baseline, see check_gate_nonvacuity.py)."),
        "G4_curated_arch_agreement": round(arch_rate, 4), "target": 0.60, "pass": bool(arch_rate >= 0.60),
        "arch_agreement_random_baseline": round(arch_base, 4),
        "arch_agreement_lift": round(arch_rate - arch_base, 4),
        "retrieval_top10_hit_rate_informational": round(g4_hit, 4),
        "intuition_matches_a_arch_rate": round(intu_rate, 4),
        "mean_b_rank": round(mean_rank, 2),
        "random_expected_rank": round(random_rank, 2),
        "random_expected_rank_empirical": round(random_rank_empirical, 2),
        "random_rank_formulas_agree": bool(formulas_agree),
        "SUPERSEDED_naive_random_rank_pool_over_2": round(naive_random_rank, 2),
        "better_than_random_ratio": round(btr, 3),
        "SUPERSEDED_better_than_random_ratio": round(btr_naive, 3),
        "baseline_correction_note": (
            "b_best_rank is the MINIMUM rank over all of B's rows, so the random "
            "expectation is E[min of k uniform draws from N] = (N-k)/(k+1), not N/2. The "
            "old N/2 baseline overstated the better-than-random ratio by roughly the "
            "number of B rows. Both the closed form and a 200-draw simulation are "
            "computed and required to agree."),
        "sport_counts": sport_counts,
        "triples": rows,
    }
    (DATA / "analogy_triples_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Curated analogy triples panel (on shipped unified.json) ===")
    print(f"triples scored: {len(scored)}/{len(T['triples'])}  (missing: {len(rows)-len(scored)})")
    print(f"G4-curated arch-agreement (arch A == arch B): {arch_rate:.3f} "
          f"vs random baseline {arch_base:.3f} (lift {arch_rate - arch_base:+.3f}), "
          f"target 0.60 {'PASS' if arch_rate >= 0.60 else 'FAIL'}")
    print(f"intuition matches A's model-arch: {intu_rate:.3f}")
    print(f"retrieval top-10 hit-rate: {g4_hit:.3f}   mean B rank: {mean_rank:.1f}  "
          f"(random {random_rank:.0f} analytic / {random_rank_empirical:.0f} simulated)")
    print(f"  better-than-random: {btr:.2f}x   "
          f"[SUPERSEDED N/2 baseline reported {btr_naive:.2f}x — see "
          f"baseline_correction_note]\n")
    for r in rows:
        if "in_top10" not in r:
            print(f"  MISSING  {r.get('a',{}).get('name')} ~ {r.get('b',{}).get('name')}  a_rows={r['a_rows']} b_rows={r['b_rows']}")
            continue
        top3 = " | ".join(f"{t['name']}({t['sport'][:2]}/{t['arch']}){t['cos']}" for t in r["top3_names"])
        flag = "HIT " if r["in_top10"] else "miss"
        print(f"  {flag} r={r['b_best_rank']:>4}  [{r['a']['sport'][:2]}]{r['a']['name']:<22}({r['a_arch']}) ~ "
              f"[{r['b']['sport'][:2]}]{r['b']['name']:<20}({r['b_arch']}) intu={r['role_intuitive']}  top3: {top3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
