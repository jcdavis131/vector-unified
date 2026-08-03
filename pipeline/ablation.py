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

import json
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


def train_config(M, cfg, epochs=30, warmup=5):
    n_pos = [int(_meta["n_pos"][s]) for s in SPORTS]
    model = UnifiedTrunk(sport_dims=[int(_meta["sport_dim"][s]) for s in SPORTS],
                         n_seasons_era=M["n_eras"], d_adapter=48, d_sport_tok=0,
                         d_emb=64, n_arch=8, n_pos=n_pos, dropout=0.2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pools = per_sport_pools(M)
    q = 86
    rng = np.random.default_rng(SEED)

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


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    M = load_matrix(DEVICE)
    results = {}
    for name, cfg in CONFIGS.items():
        print(f"--- training {name} ---", flush=True)
        model = train_config(M, cfg)
        z = encode_all(model, M, DEVICE)
        g = gates(z, M)
        results[name] = g
        print(f"   {name:10s} G1={'PASS' if g['G1_pass'] else 'FAIL'} "
              f"G2_acc={g['G2_sport_acc']} rank={g['G2_rank']} G3_sil={g['G3_sil']} G4_hit={g['G4_hit']}", flush=True)
    (DATA / "ablation_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== Ablation summary (each loss earns its keep if dropping it worsens its target gate) ===")
    print(f"{'config':10s} {'G1':4s} {'G2acc':>6s} {'rank':>5s} {'G3sil':>6s} {'G4hit':>6s}")
    for name, g in results.items():
        print(f"{name:10s} {'PASS' if g['G1_pass'] else 'FAIL':4s} {g['G2_sport_acc']:>6} {g['G2_rank']:>5} {g['G3_sil']:>6} {g['G4_hit']:>6}")
    full = results["full"]
    print("\nEarns-its-keep verdict (vs full):")
    for name in ("no_supcon", "no_coral", "no_grl", "no_vicreg", "task_only"):
        g = results[name]
        dG3 = g["G3_sil"] - full["G3_sil"]
        dG4 = g["G4_hit"] - full["G4_hit"]
        dG2 = g["G2_sport_acc"] - full["G2_sport_acc"]
        dR = g["G2_rank"] - full["G2_rank"]
        print(f"  drop {name:10s}: dG3={dG3:+.3f} dG4={dG4:+.3f} dG2sport={dG2:+.3f} dRank={dR:+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
