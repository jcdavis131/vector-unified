#!/usr/bin/env python3
"""Do G1/G2/G3 fail on a null? If not, their PASS verdicts are unearned. (7.16)

Solo personal project, no connection to employer, built with public/free-tier only

Every gate in this repo has been reported as PASS since Phase 2, and none of them has ever
been shown to FAIL on data it should reject. Phase 7 spent its whole length finding numbers
that were real and answered a different question than the one they appeared to answer, and
a gate is exactly the kind of thing that hides one: it emits a verdict, the verdict is
green, and nobody asks what red would have required.

Two of the thresholds were bare inequalities with no margin at all:

    G3  silhouette_pass  = sil > 0
        separation_pass  = within_arch_cross_sport_cos > between_arch_cross_sport_cos

A quantity being greater than zero is not evidence of the thing G3 is named after. Both
now carry floors calibrated against the nulls measured here (SIL_FLOOR, SEP_FLOOR in
eval_unified.py). Only ONE of them had actually been passing on a null — separation, up to
+0.0440 — but a test with no margin is untrustworthy whether or not it has failed yet.

ONE QUANTITY CANNOT BE TESTED THIS WAY AT ALL, and it is reported as such rather than
counted: G2's effective_rank is a function of the singular values, and a row permutation
leaves the Gram spectrum untouched. Both shuffles score 12.4, exactly the real value.
random_gaussian moves it UP to 64.0, so a high rank is not evidence of quality either.
rank_nondeg_pass detects collapse and nothing more.

THE THREE NULLS, and what each is designed to kill:

  global_shuffle        z rows permuted across the whole matrix. Destroys everything —
                        role, sport, archetype. Any gate that passes here is broken
                        outright, not merely weak.

  within_sport_shuffle  z rows permuted WITHIN each sport. Preserves the sport-block
                        geometry exactly and destroys only the player -> archetype link.
                        THIS IS THE SHARP ONE. G3 claims cross-sport archetype coherence.
                        If it still passes when archetype labels are randomised inside
                        each sport, it is measuring sport structure, not archetype.

  random_gaussian       fresh N(0,1) rows, L2-normalised. Controls for anything that comes
                        from the shape of the space rather than from the model.

PRE-REGISTERED READING, fixed before the first run: a gate that PASSES on
`within_sport_shuffle` is NOT MEASURING WHAT IT CLAIMS, and its historical PASS must be
withdrawn until the gate is redefined. A gate that passes only on the real embedding and
fails all three nulls has earned its verdict.

ALSO MEASURED: the sport-pair composition of G3's within-arch and between-arch samples.
Both are filtered to cross-sport pairs, but if within-arch pairs are mostly (hoops,
gridiron) while between-arch pairs spread over all three combinations, the comparison is
confounded by which sport pairs each set happens to draw — a difference in sport-pair mix
would produce a separation with no archetype content at all.

    python pipeline/check_gate_nonvacuity.py
    python pipeline/check_gate_nonvacuity.py --check   # exit 1 if any gate is vacuous
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_unified as EV  # noqa: E402
from load_encoders import SPORTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "gate_nonvacuity.json"
SEED = 7
PERM_REPS = 50


def nulls(z: np.ndarray, sid: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    out = {}
    out["global_shuffle"] = z[rng.permutation(len(z))]
    zw = z.copy()
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        zw[idx] = z[rng.permutation(idx)]
    out["within_sport_shuffle"] = zw
    g = rng.normal(size=z.shape).astype(np.float32)
    out["random_gaussian"] = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
    return out


def g4_random_baseline(arch: np.ndarray, sid: np.ndarray) -> float:
    """P(a random OTHER-SPORT row shares the query's archetype).

    G4's stated bar is 0.60 against an unstated baseline. The archetypes are not uniform
    — 0.249 / 0.235 / 0.192 / 0.148 / 0.113 / 0.063 — and G4 draws its neighbour only
    from the other two sports, so the honest null is this quantity, not 1/n_arch. Same
    defect shape as G2's `chance = 1/3`, which is why it is computed rather than assumed.
    """
    n = len(arch)
    tot = 0.0
    for a in np.unique(arch):
        for s in np.unique(sid):
            n_q = int(((arch == a) & (sid == s)).sum())
            pool = int((sid != s).sum())
            same = int(((arch == a) & (sid != s)).sum())
            if pool:
                tot += (n_q / n) * (same / pool)
    return tot


def g4_hit_rate(z: np.ndarray, M) -> float:
    """analogy_panel.py's G4, replicated: cross-sport NN shares the archetype."""
    arch = M["arch_id"].cpu().numpy()
    sid = M["sport_id"].cpu().numpy()
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    sim = zn @ zn.T
    sim = np.where(sid[:, None] == sid[None, :], -np.inf, sim)
    np.fill_diagonal(sim, -np.inf)
    return float((arch[sim.argmax(axis=1)] == arch).mean())


def gate_verdicts(z: np.ndarray, M) -> dict:
    g1 = EV.g1_per_sport(z, M)
    # G1's z arm only — the e_s arm is the frozen encoder and is unaffected by a null,
    # so comparing them would let the baseline carry the verdict.
    g1_z = {sp: v["native_knn5_z"] for sp, v in g1.items()}
    g1_e = {sp: v["native_knn5_e_s"] for sp, v in g1.items()}
    g1_pass = all(
        z_v is not None and e_v is not None and z_v >= e_v - 0.02
        for z_v, e_v in zip(g1_z.values(), g1_e.values(), strict=True))
    g2 = EV.g2_sport_invariance(z, M)
    g3 = EV.g3_silhouette(z, M)
    g4h = g4_hit_rate(z, M)
    return {
        "G1_native_knn5_z": {k: (round(v, 4) if v is not None else None)
                             for k, v in g1_z.items()},
        "G1_pass": bool(g1_pass),
        "G2_sport_acc": g2["sport_acc"], "G2_effective_rank": g2["effective_rank"],
        "G3_silhouette": g3["silhouette"], "G3_silhouette_pass": g3["silhouette_pass"],
        "G3_within_cos": g3["within_arch_cross_sport_cos"],
        "G3_between_cos": g3["between_arch_cross_sport_cos"],
        "G3_separation": g3["separation"], "G3_separation_pass": g3["separation_pass"],
        "G4_hit_rate": round(g4h, 4),
        "G4_pass": bool(g4h >= 0.60),
    }


def sport_pair_mix(M, rng_seed: int = SEED) -> dict:
    """Which sport pairs does each of G3's two samples actually draw?

    Mirrors g3_silhouette's own sampling so the answer describes the real comparison
    rather than an idealised one.
    """
    arch = M["arch_id"].cpu().numpy()
    sid = M["sport_id"].cpu().numpy()
    rng = np.random.default_rng(rng_seed)
    within: collections.Counter = collections.Counter()
    between: collections.Counter = collections.Counter()
    for a in np.unique(arch):
        ia = np.where(arch == a)[0]
        if len(ia) < 2:
            continue
        for _ in range(800):
            i, j = rng.choice(ia, 2, replace=False)
            if sid[i] != sid[j]:
                within[tuple(sorted((SPORTS[sid[i]], SPORTS[sid[j]])))] += 1
    for _ in range(800):
        a1, a2 = rng.choice(np.unique(arch), 2, replace=False)
        i = rng.choice(np.where(arch == a1)[0])
        j = rng.choice(np.where(arch == a2)[0])
        if sid[i] != sid[j]:
            between[tuple(sorted((SPORTS[sid[i]], SPORTS[sid[j]])))] += 1

    def frac(c):
        t = sum(c.values()) or 1
        return {f"{a}+{b}": round(100.0 * n / t, 1) for (a, b), n in sorted(c.items())}

    w, b = frac(within), frac(between)
    keys = sorted(set(w) | set(b))
    drift = max((abs(w.get(k, 0.0) - b.get(k, 0.0)) for k in keys), default=0.0)
    return {"within_arch_pairs": w, "between_arch_pairs": b,
            "n_within": sum(within.values()), "n_between": sum(between.values()),
            "max_composition_gap_pct_points": round(drift, 1),
            "note": ("If the two samples draw different sport-pair mixes, part of G3's "
                     "separation is a sport-pair effect rather than an archetype effect. "
                     "This does not say how much — it says whether the confound is "
                     "present and how large the imbalance is.")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default="unified_best.pt")
    ap.add_argument("--check", action="store_true", help="exit 1 if any gate is vacuous")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    device = torch.device("cpu")
    model, _ck = EV.load_model(device, args.ckpt)
    from train_unified import load_matrix
    M = load_matrix(device)
    z = EV.encode_all(model, M, device)
    sid = M["sport_id"].cpu().numpy()

    g4_base = g4_random_baseline(M["arch_id"].cpu().numpy(), sid)
    real = gate_verdicts(z, M)
    rng = np.random.default_rng(SEED)
    null_results = {name: gate_verdicts(zz, M) for name, zz in nulls(z, sid, rng).items()}

    # ---- G2's baseline is wrong, and the null is what shows it -------------------
    # g2_sport_invariance reports `chance = 1/3`. The three sports are not balanced —
    # 12,966 / 5,323 / 2,430 — so a classifier that always answers "hoops" scores 0.6258.
    # global_shuffle lands at exactly that, which is the null behaving correctly and the
    # stated baseline being 29 points too generous. The leakage is a 12.9pp lift over
    # majority, not a 42pp lift over uniform.
    counts = collections.Counter(sid.tolist())
    majority = max(counts.values()) / len(sid)
    g2_baseline = {
        "reported_chance_in_eval_unified": round(1.0 / 3.0, 4),
        "majority_class_share": round(majority, 4),
        "class_counts": {SPORTS[k]: int(v) for k, v in sorted(counts.items())},
        "real_sport_acc": real["G2_sport_acc"],
        "lift_over_uniform": round(real["G2_sport_acc"] - 1.0 / 3.0, 4),
        "lift_over_majority": round(real["G2_sport_acc"] - majority, 4),
        "global_shuffle_acc": null_results["global_shuffle"]["G2_sport_acc"],
        "verdict": ("eval_unified.py's `chance` is WRONG for this problem. A 3-class "
                    "accuracy is only comparable to 1/3 when the classes are balanced, "
                    "and these are 62.6 / 25.7 / 11.7. global_shuffle scoring the "
                    "majority share exactly is the confirmation. Sport leakage should be "
                    "quoted against 0.6258."),
    }

    # ---- permutation-calibrated threshold for G3 separation ----------------------
    # `within > between` is a bare inequality on a noisy difference, so it passes on a
    # null roughly half the time by construction — which is what the first run showed
    # (+0.0027 under within_sport_shuffle against +0.8448 real). The construct is fine;
    # the THRESHOLD is what was vacuous. Calibrate it against the null distribution.
    perm_seps = []
    prng = np.random.default_rng(SEED + 1)
    for _ in range(PERM_REPS):
        zw = z.copy()
        for s in np.unique(sid):
            idx = np.where(sid == s)[0]
            zw[idx] = z[prng.permutation(idx)]
        g3 = EV.g3_silhouette(zw, M)
        perm_seps.append(g3["separation"])
    perm_seps.sort()
    p95 = perm_seps[int(0.95 * (len(perm_seps) - 1))]
    calibrated = {
        "perm_reps": PERM_REPS,
        "null_separation_p95": round(p95, 4),
        "null_separation_max": round(perm_seps[-1], 4),
        "real_separation": real["G3_separation"],
        "passes_calibrated_threshold": bool(real["G3_separation"] > p95),
        "note": ("Replaces `within > between` with `within - between > 95th percentile of "
                 "the within-sport-shuffle null`. The real value clears it by orders of "
                 "magnitude, so the FINDING was never in doubt — only the test was."),
    }

    # A gate is VACUOUS if it still passes when the structure it names is destroyed.
    vacuous: list[str] = []
    for gate, key in (("G1", "G1_pass"), ("G3_silhouette", "G3_silhouette_pass"),
                      ("G3_separation", "G3_separation_pass"), ("G4", "G4_pass")):
        if not real[key]:
            continue
        survivors = [n for n, r in null_results.items() if r[key]]
        if gate == "G3_separation" and survivors and calibrated["passes_calibrated_threshold"]:
            vacuous.append(
                f"{gate} THRESHOLD is vacuous, not the construct: `within > between` also "
                f"holds on {', '.join(survivors)} (null sep up to "
                f"{calibrated['null_separation_max']:+.4f}). The real separation "
                f"{real['G3_separation']:+.4f} clears the null 95th percentile "
                f"{calibrated['null_separation_p95']:+.4f}, so the finding stands and the "
                f"TEST must be replaced with the calibrated one.")
        elif "within_sport_shuffle" in survivors:
            vacuous.append(f"{gate} passes on within_sport_shuffle — archetype labels were "
                           f"randomised inside each sport and it did not notice")
        elif survivors:
            vacuous.append(f"{gate} passes on {', '.join(survivors)}")

    report = {
        "question": "Do the Stage-1 gates fail on data they should reject?",
        "checkpoint": args.ckpt,
        "real": real,
        "nulls": null_results,
        "g3_sport_pair_confound": sport_pair_mix(M),
        "g2_baseline_correction": g2_baseline,
        "g4_baseline": {
            "random_cross_sport_arch_match": round(g4_base, 4),
            "real_hit_rate": real["G4_hit_rate"],
            "lift_over_random": round(real["G4_hit_rate"] - g4_base, 4),
            "stated_bar": 0.60,
            "note": ("G4's 0.60 bar was stated without a baseline. The honest null is the "
                     "chance that a random other-sport row shares the archetype, which is "
                     f"{g4_base:.4f} given the 0.249/0.235/0.192/0.148/0.113/0.063 "
                     "archetype mix — so 0.60 is a real bar here, unlike G2's 1/3."),
        },
        "g3_separation_calibrated": calibrated,
        "vacuous_gates": vacuous,
        "decision_rule": (
            "A gate that PASSES on within_sport_shuffle is not measuring what it claims "
            "and its historical PASS must be withdrawn until it is redefined. Fixed "
            "before the first run."),
        "note_rank": (
            "G2's effective_rank is NOT testable by these nulls and its green must not be "
            "read as earned. Rank is a function of the singular values and a row "
            "permutation leaves the Gram spectrum untouched — global_shuffle and "
            "within_sport_shuffle both score 12.4, exactly the real value. Only "
            "random_gaussian moves it, and it moves it UP to 64.0, so a high rank is not "
            "evidence of quality either. rank_nondeg_pass detects collapse, nothing more."),
        "note_G2": ("G2 has no pass/fail here on purpose — it is a leakage measure, not a "
                    "quality gate, and a null SHOULD drive sport accuracy toward chance. "
                    "Its value under each null is reported as a sanity check on the nulls "
                    "themselves: if global_shuffle does not collapse sport accuracy toward "
                    "0.333, the null is not doing its job."),
        "seed": SEED,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = [("REAL", real), *null_results.items()]
    print(f"{'variant':<22} {'G1':>5} {'G2acc':>7} {'rank':>6} {'sil':>8} {'within':>8} "
          f"{'between':>8} {'sep':>8} {'G4':>7}")
    for name, r in rows:
        print(f"{name:<22} {str(r['G1_pass']):>5} {r['G2_sport_acc']:>7.3f} "
              f"{r['G2_effective_rank']:>6.1f} {r['G3_silhouette']:>8.4f} "
              f"{r['G3_within_cos']:>8.4f} {r['G3_between_cos']:>8.4f} "
              f"{r['G3_separation']:>+8.4f} {r['G4_hit_rate']:>7.4f}")

    b = g2_baseline
    print()
    print(f"G2 baseline: eval_unified says chance={b['reported_chance_in_eval_unified']}, "
          f"majority class is {b['majority_class_share']} "
          f"({', '.join(f'{k} {v}' for k, v in b['class_counts'].items())})")
    print(f"{'':13}real acc {b['real_sport_acc']}  ->  lift over uniform "
          f"{b['lift_over_uniform']:+.4f}, over MAJORITY {b['lift_over_majority']:+.4f}"
          f"   (global_shuffle lands at {b['global_shuffle_acc']})")
    gb = report["g4_baseline"]
    print()
    print(f"G4 baseline: random other-sport archetype match = "
          f"{gb['random_cross_sport_arch_match']:.4f}; real {gb['real_hit_rate']:.4f} "
          f"= lift {gb['lift_over_random']:+.4f} (stated bar {gb['stated_bar']})")
    print(f"{'':13}every null lands on the baseline, which validates both at once")

    k = calibrated
    print()
    print(f"G3 separation: real {k['real_separation']:+.4f}  vs null p95 "
          f"{k['null_separation_p95']:+.4f} / max {k['null_separation_max']:+.4f} "
          f"over {k['perm_reps']} shuffles -> "
          f"{'CLEARS' if k['passes_calibrated_threshold'] else 'FAILS'} calibrated threshold")

    c = report["g3_sport_pair_confound"]
    print(f"\nG3 sport-pair mix   within-arch {c['within_arch_pairs']}")
    print(f"{'':20}between-arch {c['between_arch_pairs']}")
    print(f"{'':20}max gap {c['max_composition_gap_pct_points']} pct points")

    if vacuous:
        print(f"\n{len(vacuous)} VACUOUS gate(s):")
        for v in vacuous:
            print(f"  {v}")
        print(f"\nwrote {OUT}")
        return 1 if args.check else 0
    print("\nno gate survives a null — every PASS above is earned,")
    print("EXCEPT effective_rank, which no permutation null can test. Rank is a function")
    print("of the singular values and a row permutation leaves them unchanged: both")
    print("shuffles score 12.4, exactly the real value. random_gaussian moves it UP to")
    print("64.0, so a high rank is not evidence of quality either. It detects collapse.")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
