"""Vector Unified — Stage 1 ablation (does each alignment loss earn its keep?).

SPEC §5 / Phase 5.2 house rule: drop each alignment loss and measure Δ on
G1/G2/G3/G4. A loss earns its keep if removing it worsens the gate it targets:
  SupCon  -> should drive G3 (archetype coherence) + G4 (analogy)
  CORAL   -> should help G3 (shared axis system)
  GRL     -> should drive G2 (sport-invariance)
  VICReg  -> should prevent collapse (rank) without hurting G1/G3
  task    -> anti-collapse anchor (G1); dropping it should collapse / hurt G1

Configs (each = canonical Stage 1 v0 minus one component):
  full        SupCon+CORAL+GRL+VICReg+task   (the shipped config)
  no_supcon   drop SupCon
  no_coral    drop CORAL
  no_grl      drop GRL  (grl-lambda 0)
  no_vicreg   drop var+cov
  task_only   drop SupCon+CORAL+GRL+VICReg (anti-collapse baseline)

Each config: train 30ep (warmup 5), encode z, compute G1/G2/G3/G4(sampled),
record. Output data/ablation_report.json + table. Frozen encoders (Stage 1) —
non-destructive, no per-sport regression risk.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

import numpy as np
import torch
import torch.nn.functional as F

from load_encoders import SPORTS, ROOT
from train_unified import (UnifiedTrunk, GRL, load_matrix, per_sport_pools, gather_batch,
                           supcon_loss, coral_loss, var_loss, cov_loss,
                           SEED)
from eval_unified import (g1_per_sport, encode_all, g2_sport_invariance,
                          g3_silhouette, g4_hit_rate, g4_random_baseline)

DATA = ROOT / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_meta = json.loads((DATA / "unified_meta.json").read_text(encoding="utf-8"))
ARCH_NAMES = _meta["arch_names"]


def train_config(M, cfg, epochs=30, warmup=5, seed=SEED):
    n_pos = [int(_meta["n_pos"][s]) for s in SPORTS]
    model = UnifiedTrunk(sport_dims=[int(_meta["sport_dim"][s]) for s in SPORTS],
                         n_seasons_era=M["n_eras"], d_adapter=48, d_sport_tok=0,
                         d_emb=64, n_arch=8, n_pos=n_pos, dropout=0.2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pools = per_sport_pools(M)
    q = 86
    torch.manual_seed(seed)
    # THE SEED ALONE DID NOT PIN THE RESULT, and that was measured, not suspected.
    # `ablation.py --seeds 1 --configs full` run three times back to back on one box with
    # no code change produced G2_sport_acc = 0.6940, 0.6926, 0.6827 -- three distinct
    # values from one seed. That is why full@seed7 has three different records across
    # data/ablation_report.json, ablation_grl_seeds.json and ablation_coral_vicreg_seeds.json.
    #
    # THESE LINES ARE NOT SUFFICIENT, AND THAT WAS TESTED RATHER THAN ASSUMED. They mirror
    # train_stage2.py:166-171, which IS reproducible: 34 of 37 numeric report fields come
    # back BIT-IDENTICAL across reruns, and the 3 that move are all one quantity (G2's
    # probe) by 0.0002. So the obvious hypothesis was that ablation.py differed only by
    # these lines. It does not. With them added, full@seed7 still gives 0.6827 and 0.6993
    # on two consecutive runs — a WIDER spread than the 0.0113 measured without them.
    #
    # Also ruled out: no determinism warning is emitted (so warn_only is not silently
    # letting a flagged op through), and CUBLAS_WORKSPACE_CONFIG is unset for
    # train_stage2.py too, so that is not the discriminator either.
    #
    # THE ROOT CAUSE IS UNKNOWN and is NOT claimed to be fixed here. Unlike stage 2, the
    # ablation EMBEDDING itself moves (G3_sil 0.6776 vs 0.6758, G4_hit 0.960 vs 0.963), not
    # just a downstream probe, so the nondeterminism is inside training. The lines are kept
    # because they are correct practice and match the sibling trainer, NOT because they
    # solve the problem.
    #
    # THIS DOES NOT REPAIR THE EXISTING ARTIFACTS EITHER. Runs recorded before this cannot
    # be regenerated to match — they came from a process that does not repeat. See
    # data/ablation_determinism.json and check_ablation_consistency.py.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    rng = np.random.default_rng(seed)

    def sport_clf_loss(z, sport_ids, lam):
        zr = GRL.apply(z, lam)
        return F.cross_entropy(model.sport_clf(zr), sport_ids)

    def task_loss(z, sport_ids, native, pos, posm):
        loss = z.new_zeros(())
        for s in range(3):
            m = sport_ids == s
            if m.any():
                loss = loss + F.cross_entropy(model.native_heads[s](z[m]), native[m])
                pm = m & (posm == 1)
                if pm.any():
                    pp = pos[pm].clamp(0, n_pos[s] - 1)
                    loss = loss + F.cross_entropy(model.pos_heads[s](z[pm]), pp)
        return loss / 3.0

    def one_batch():
        gi = []
        for s in range(3):
            samp = pools[s][torch.tensor(rng.choice(len(pools[s]), q, replace=True))]
            gi.append(samp)
        return torch.cat(gi)

    w_task, w_coral = 2.0, 0.5
    w_var, w_cov = cfg["w_var"], cfg["w_cov"]
    grl_lambda = cfg["grl_lambda"]
    use_sup = cfg["use_sup"]
    for epoch in range(epochs):
        lam = grl_lambda * min(1.0, max(0.0, (epoch + 1) - warmup) / max(1, 10))
        folding = (epoch + 1) > warmup
        steps = max(1, sum(len(pools[s]) for s in range(3)) // (q * 3))
        for _ in range(steps):
            gi = one_batch()
            sid, eid, arch, native, pos, posm, e_per = gather_batch(M, gi)
            opt.zero_grad()
            z, h = model.encode(e_per, sid, eid, return_raw=True)
            loss = w_task * task_loss(z, sid, native, pos, posm)
            if cfg["use_coral"]:
                loss = loss + w_coral * coral_loss(h, sid)
            if cfg["use_vicreg"]:
                loss = loss + w_var * var_loss(z) + w_cov * cov_loss(z)
            if folding and use_sup:
                loss = loss + supcon_loss(z, arch, sid, model.log_temp)
            if folding and grl_lambda > 0:
                loss = loss + 0.3 * sport_clf_loss(z, sid, lam)
            loss.backward()
            opt.step()
    return model


def gates(z, M):
    g1 = g1_per_sport(z, M)
    g1_pass = all(d["native_knn5_z"] >= d["native_knn5_e_s"] - 0.02
                  for d in g1.values() if d["native_knn5_e_s"] is not None)
    # G2/G3/G4 IMPORTED, not re-derived. This block previously carried its own
    # LogisticRegression, its own silhouette, and its own cross-sport-NN loop — the last
    # of which SAMPLED 4,000 rows for speed, so the ablation table's G4 column was never
    # comparable to the shipped G4 it sat next to. It also predated the baseline
    # corrections in 7.16-7.17, so an ablation decision could rest on a definition the
    # ship gate no longer used.
    g2 = g2_sport_invariance(z, M)
    g3 = g3_silhouette(z, M)
    g4 = g4_hit_rate(z, M)
    return {"G1_pass": g1_pass, "G1_hoops_z": round(g1["hoops"]["native_knn5_z"], 3),
            "G2_sport_acc": g2["sport_acc"],
            "G2_delta_vs_majority": g2["delta_vs_majority"],
            "G2_rank": g2["effective_rank"],
            "G3_sil": g3["silhouette"], "G4_hit": round(g4, 3),
            "G4_baseline": round(g4_random_baseline(M), 4)}


CONFIGS = {
    "full":       {"use_sup": True,  "use_coral": True,  "use_vicreg": True,  "grl_lambda": 0.05, "w_var": 1.0, "w_cov": 1.0},
    "no_supcon":  {"use_sup": False, "use_coral": True,  "use_vicreg": True,  "grl_lambda": 0.05, "w_var": 1.0, "w_cov": 1.0},
    "no_coral":   {"use_sup": True,  "use_coral": False, "use_vicreg": True,  "grl_lambda": 0.05, "w_var": 1.0, "w_cov": 1.0},
    "no_grl":     {"use_sup": True,  "use_coral": True,  "use_vicreg": True,  "grl_lambda": 0.0,  "w_var": 1.0, "w_cov": 1.0},
    "no_vicreg":  {"use_sup": True,  "use_coral": True,  "use_vicreg": False, "grl_lambda": 0.05, "w_var": 0.0, "w_cov": 0.0},
    "task_only":  {"use_sup": False, "use_coral": False, "use_vicreg": False, "grl_lambda": 0.0,  "w_var": 0.0, "w_cov": 0.0},
}


METRICS = ("G2_sport_acc", "G2_rank", "G3_sil", "G4_hit")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=1,
                    help="repeats per config; >1 puts an error bar on every delta")
    ap.add_argument("--configs", default="",
                    help="comma-separated subset (always includes `full`, the reference). "
                         "Exists so an undecided loss can be re-run at high seed count "
                         "without paying for the four already settled.")
    ap.add_argument("--out", default="ablation_report.json",
                    help="report filename under data/; use a distinct name for a targeted "
                         "subset run so it does not overwrite the full table")
    args = ap.parse_args()
    if args.configs:
        want = {c.strip() for c in args.configs.split(",") if c.strip()} | {"full"}
        unknown = want - set(CONFIGS)
        if unknown:
            print(f"unknown config(s): {', '.join(sorted(unknown))}")
            return 2
        for name in list(CONFIGS):
            if name not in want:
                del CONFIGS[name]
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    M = load_matrix(DEVICE)

    seeds = [SEED + i for i in range(args.seeds)]
    runs: dict[str, list[dict]] = {name: [] for name in CONFIGS}
    for si, sd in enumerate(seeds):
        for name, cfg in CONFIGS.items():
            print(f"--- training {name} (seed {sd}, {si + 1}/{len(seeds)}) ---", flush=True)
            g = gates(encode_all(train_config(M, cfg, seed=sd), M, DEVICE), M)
            runs[name].append(g)
            print(f"   {name:10s} G2_acc={g['G2_sport_acc']} rank={g['G2_rank']} "
                  f"G3_sil={g['G3_sil']} G4_hit={g['G4_hit']}", flush=True)

    def agg(name, key):
        v = [r[key] for r in runs[name]]
        return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

    results = {}
    for name in CONFIGS:
        r = {k: round(agg(name, k)[0], 4) for k in METRICS}
        r.update({k + "_sd": round(agg(name, k)[1], 4) for k in METRICS})
        r["G1_pass"] = all(x["G1_pass"] for x in runs[name])
        r["G2_delta_vs_majority"] = round(
            statistics.mean(x["G2_delta_vs_majority"] for x in runs[name]), 4)
        r["G4_baseline"] = runs[name][0]["G4_baseline"]
        r["n_seeds"] = len(seeds)
        results[name] = r
    (DATA / args.out).write_text(
        json.dumps({"seeds": seeds, "configs": results, "runs": runs}, indent=2),
        encoding="utf-8")
    print("\n=== Ablation summary (each loss earns its keep if dropping it worsens its target gate) ===")
    # Baselines printed BESIDE the columns, not left for the reader to remember. A G4 of
    # 0.105 is not merely "low" — it is BELOW the 0.1712 chance of a random other-sport
    # neighbour sharing the archetype, which is a categorically different statement, and
    # the bare number never said it. Same for G2: the floor is the majority share, not 0.
    g4b = results["full"]["G4_baseline"]
    majority = results["full"]["G2_sport_acc"] - results["full"]["G2_delta_vs_majority"]
    print(f"  baselines: G2 majority-class {majority:.4f} (floor, lower is better)   "
          f"G4 random {g4b:.4f} (higher is better)")
    print(f"{'config':10s} {'G1':4s} {'G2acc':>7s} {'dMaj':>7s} {'rank':>5s} {'G3sil':>6s} "
          f"{'G4hit':>6s} {'vs rand':>8s}")
    for name, g in results.items():
        below = " BELOW" if g["G4_hit"] < g4b else ""
        print(f"{name:10s} {'PASS' if g['G1_pass'] else 'FAIL':4s} {g['G2_sport_acc']:>7} "
              f"{g['G2_delta_vs_majority']:>+7.4f} {g['G2_rank']:>5} {g['G3_sil']:>6} "
              f"{g['G4_hit']:>6} {g['G4_hit'] - g4b:>+8.4f}{below}")
    # PRE-REGISTERED, and it is the whole point of --seeds. With one run per config a
    # delta of 0.003 and a delta of zero are the same observation, and the 1-seed table
    # kept CORAL and VICReg on moves of 0.001-0.003. A loss earns its keep only if
    # dropping it moves its target metric by more than 2x the pooled seed-to-seed standard
    # deviation; anything smaller is NOT DISTINGUISHABLE FROM NOISE and is reported that
    # way rather than as a small effect.
    # PAIRED, because the DESIGN is paired and the first version of this rule threw that
    # away. Every config trains on the same seed list, so seed i gives `full` and `no_grl`
    # the same initialisation and the same batch order — the right statistic is the
    # per-seed DIFFERENCE, not two independent means. The unpaired 2x-pooled-sd rule
    # returned `ns` for GRL at 3 seeds (delta +0.047 against a 0.048 threshold). The same
    # data read pairwise: 8 of 9 seeds positive, mean +0.0456, sd of the differences
    # 0.0291, so sd of the mean is 0.0097 and the effect is ~4.7 sigma. The pairing was
    # worth roughly a factor of five in power and discarding it nearly retired a loss that
    # does work.
    print("\nEarns-its-keep verdict (vs full, PAIRED by seed, |mean diff| > 2x sd-of-mean):")
    if len(seeds) < 2:
        print("  NOT DECIDABLE — one seed per config, so no noise floor was measured. "
              "Re-run with --seeds 3.")
    for name in [c for c in CONFIGS if c != "full"]:
        parts = []
        for k, label in (("G3_sil", "dG3"), ("G4_hit", "dG4"), ("G2_sport_acc", "dG2sport")):
            diffs = [x[k] - f[k] for x, f in zip(runs[name], runs["full"], strict=True)]
            mean_d = statistics.mean(diffs)
            if len(diffs) > 1:
                sem = statistics.stdev(diffs) / (len(diffs) ** 0.5)
                tag = "*" if abs(mean_d) > 2 * sem else "ns"
            else:
                sem, tag = 0.0, "?"
            n_pos = sum(1 for d in diffs if d > 0)
            parts.append(f"{label}={mean_d:+.3f}(sem {sem:.3f}){tag}[{n_pos}/{len(diffs)}+]")
        print(f"  drop {name:10s}: " + "  ".join(parts))
    print("  * exceeds 2x sd-of-mean of the PER-SEED differences   ns = not distinguishable")
    print("  [k/n+] = how many seeds moved in the same direction; a real effect should be "
          "consistent,\n  not just large on average.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
