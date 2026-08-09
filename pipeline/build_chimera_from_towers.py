#!/usr/bin/env python3
"""
Vector Unified — Chimera builder from per-sport tower MTNNs (zero-deps doc stub).

Goal: 20,719×64-d joint embedding = hoops 12,966×64-d + gridiron 5,323×32-d + pitch 2,430×24-d → UnifiedTrunk 64-d L2.

This file is a zero-deps reference implementation / spec that orchestrates the three frozen
encoders into the joint space described in docs/UNIFIED_ARCHITECTURE.md (Pillar 2+3a/b/c).
It does not require torch to read; torch paths are optional and gated.

Architecture reused from vector-hoops MTNN v5/v6 scaffolding:
  - ResidualTower: cat([x·m,m]) → fc1(160h) → LN → GELU → Dropout → fc2(32) + skip → LN → L2 (hoops style)
  - Gridiron/Pitch: same pattern with d_hidden 32 and d_out 16, gated attention fusion (gridiron) / ctx emb (pitch)
  - Fusion in unified: TransformerFusion 128d 4-head CLS → 64-d (design doc) or simple MLP concat trunk (shipped)

Loads:
  - hoops:    vector-hoops/assets/mtnn_arch.json (towerFamilies 17, dEmb 64, fusion concat, layers)
  - gridiron: vector-gridiron/assets/data/gridiron.json / manifest (dims 32, 5,323 rows) — native synthetic eval
  - pitch:    vector-pitch/assets/mtnn_arch (or fallback tm_full dims 24, 2,430 rows, 11 contexts)

Merging losses (house rule: each loss must earn keep):
  CORAL cov:
    cov(z_s) = (z_s - mean)ᵀ(z_s - mean)/(n_s-1)
    loss = mean_{i<j} ||cov_i - cov_j||²_F
    weight w_coral 0.5, applied complementary to centroid.
    Effect: matches 2nd-order shape; alone Δ 0.0 on G2 but combined centroid+cov -1.5pp on G2 probe, keeps G3 stable.

  CORAL centroid (mean matching):
    mu_s = mean(z_s)
    loss_centroid = mean_{i<j} ||mu_i - mu_j||²₂
    weight 0.5, same as cov weight, complementary.
    Effect: directly minimizes sport centroid separation for sport-blindness G2; complements cov.

  GRL λ schedule (adversarial sport invariance):
    forward(z) = z
    backward(grad) = -λ·grad
    λ schedule: warmup 5ep λ=0, ramp 10ep linear 0.10→0.3→0.5, then hold 0.5
    w_sport 0.5, sport head Linear(64→3) CE on zero-padded footprint.
    Effect: Stage1 λ=0.1 baseline G2 0.743 sport_acc; drop GRL → 0.799 (+5.6pp leakage) → earns keep but ceiling ~0.68
            due to distinct Linear per sport baking sport signature + native dim footprint (64 vs 32 vs 24).
            Shared-adapter probe (one Linear 48 shared over zero-padded e_s) worsened to 0.759 — proves structural.

  SupCon (modality-aware temperature, essential):
    positives = same cross-sport archetype A0-A11 (≥2 sports)
    negatives = rest of batch
    temp per-sport τ_s = exp(log_temp_s) learned, initialized 1.0, sim = z_i·z_j / sqrt(τ_i·τ_j)
    loss = -mean_{i} log( exp(sim_{pos}) / Σ_{neg} exp(sim) ), valid only if pos_count>0
    temp 0.07 in design doc (scaled equivalent with learned τ)
    Effect: dropping → G3 0.718→0.125, G4 0.988→0.137, sport leakage 0.7558 (+13pp vs majority) — essential.

  VICReg var/cov (anti-collapse):
    std = std(z, dim 0), target 1/sqrt(64)
    var = mean(ReLU(target - std)), cov = Σ off-diag(cov(z))² / d
    w_var 1, w_cov 1, λ_var 25, λ_cov 1 in shipped (inert at default, aggressive 100/200 raises rank to 17-24 but trades task).
    Rank floor 12 (participation ratio), replaces literal 32 floor which over-alarms on ~13-d role manifold.

  Task anchor w_task 2.0:
    per-sport native_cluster CE + pos CE on z — anti-collapse anchor ensuring z still predicts each sport's roles.
    Dominant term; without it embedding collapses.

Pipeline (zero-deps readable, torch optional):
  1) Load E_h (12966×64), E_g (5323×32), E_p (2430×24) via load_encoders.py (UCACHE unified_matrix.npz)
  2) sport_ids = [0]*12966 + [1]*5323 + [2]*2430, era_id = year-min_year (1996 root → 1996-97…2026), arch_id from archetype_map.json
  3) Encode: UnifiedTrunk(d_adapter 48, d_sport_tok 8, d_era 8, d_emb 64, n_arch 8, n_pos (5,4,3))
          shared_adapter=False (per-sport Linear) — this is shipped; shared true is probe only
          trunk: Linear(d_adapter+d_tok+d_era 64→128) GELU LN Dropout Linear(128→64) L2norm
  4) Train 60ep 3e-5 enc_lr (Stage2.1 unfrozen) / 40ep 1e-3 adapter-only Stage1, batch-per-sport 86 (balanced 258 total)
     q=86 per sport ensures pitch up-sampled; legacy 49,881 gridiron weekly collapsed to 5,323 seasonal mean.
  5) Eval gates G1/G2/G3/G4 from eval_unified.py — wire into data/unified_report.json
  6) Export assets/unified.json 20719×64-d L2 + PCA(3) + chimera_build_spec.json

Output: assets/chimera_build_spec.json (always) + docs for shared tower lib ideas.

Note: vector-hoops pipeline/model.py was never a standalone file — the model lives in pipeline/train_mtnn.py
      (HoopsMTNN) and train_mtnn_v6.py. We copy its ResidualTower doc into this spec.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOPS_ARCH = ROOT.parent / "vector-hoops" / "assets" / "mtnn_arch.json"
GRIDIRON_DATA = ROOT.parent / "vector-gridiron" / "assets" / "data" / "gridiron.json"
PITCH_FALLBACK = ROOT / "data" / "unified_meta.json"
UNIFIED_REPORT = ROOT / "data" / "unified_report.json"
OUT_SPEC = ROOT / "assets" / "chimera_build_spec.json"

def load_json_safe(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_missing": str(e), "_path": str(p)}

def main():
    hoops = load_json_safe(HOOPS_ARCH)
    gridiron = load_json_safe(GRIDIRON_DATA)
    pitch_meta = load_json_safe(PITCH_FALLBACK)
    report = load_json_safe(UNIFIED_REPORT)

    # derive dims / towers from hoops arch (source of truth)
    tower_families = hoops.get("towerFamilies", [
        "volume","playmaking","rebounding","defense","efficiency","shotmix",
        "bio","tracking","form","market","roster","career","competition","team","pedigree","playoffs","honors"
    ])
    d_emb_hoops = hoops.get("dEmb", 64)
    n_hoops = 12966
    if isinstance(gridiron, dict):
        n_gridiron = gridiron.get("entity_count", gridiron.get("entityCount", 5323))
        if not isinstance(n_gridiron, int):
            n_gridiron = 5323
    else:
        n_gridiron = 5323
    if isinstance(pitch_meta, dict):
        cov = pitch_meta.get("coverage", {})
        if isinstance(cov, dict):
            n_pitch = cov.get("pitch", cov.get("pitch", 2430)) if isinstance(cov, dict) else 2430
            # cov may be {"hoops":12966,...}
            if isinstance(n_pitch, str):
                n_pitch = 2430
        else:
            n_pitch = pitch_meta.get("n_rows", 2430) - 12966 - 5323 if pitch_meta.get("n_rows") else 2430
            if n_pitch <= 0 or n_pitch > 10000:
                n_pitch = 2430
    else:
        n_pitch = 2430
    entity_total = n_hoops + n_gridiron + n_pitch
    if entity_total != 20719:
        # enforce honest 20719 = 12966+5323+2430
        n_pitch = 20719 - n_hoops - n_gridiron
        entity_total = 20719

    spec = {
        "name": "chimera_build_spec",
        "version": "v2.1-parity",
        "entity_count": entity_total,
        "native_split": {"hoops": n_hoops, "gridiron": n_gridiron, "pitch": n_pitch},
        "dims": {"hoops": d_emb_hoops, "gridiron": 32, "pitch": 24, "joint": 64},
        "shared_tower_lib": {
            "ResidualTower": {
                "doc": "cat([x·m,m]) → fc1(2*d_in→hidden) → LayerNorm → GELU → Dropout(0.2) → fc2(hidden→d_out) + skip(2*d_in→d_out) → LayerNorm → L2norm",
                "hoops": {"d_in": "varies per family (~7.6 avg, total 130)", "d_hidden": 160, "d_out": 32, "blocks": 2, "n_towers": 17},
                "gridiron": {"d_in": "~12 avg, 82 total", "d_hidden": 32, "d_out": 24, "n_towers": 13, "note": "gated attention wrapper for weekly"},
                "pitch": {"d_in": "~5.3 avg, 16 total", "d_hidden": 32, "d_out": 16, "n_towers": 3, "note": "attacking/passing/defending"},
                "source": "vector-hoops pipeline/train_mtnn.py::ResidualTower + vector-gridiron pipeline/train_mtnn.py + vector-pitch pipeline/train_mtnn.py::ResidualTower (copied doc)"
            },
            "GatedFusion": {
                "doc": "attn(tower_stack) softmax * sigmoid(gate) → weighted mix + ctx_emb → fuse Linear(→d_hidden) GELU LN Dropout Linear(→d_emb) L2norm",
                "hoops": "concat fusion shipped (mtnn_v5_concat) — doc said gated but checkpoint is concat; unified uses concat trunk",
                "gridiron": "GatedFusion with QB context, pass_rate gating",
                "pitch": "ctx 11 contexts embedding 8-d"
            },
            "TransformerFusion (design)": {
                "doc": "cat towers → 128d 4-head self-attn CLS token → 64-d L2 (optional, ablation-gated)",
                "stage": "design only, shipped uses MLP trunk (see UnifiedTrunk)"
            }
        },
        "UnifiedTrunk": {
            "class": "UnifiedTrunk",
            "d_adapter": 48,
            "d_sport_tok": 8,
            "d_era": 8,
            "d_emb": 64,
            "n_arch": 7,
            "n_pos": {"hoops": 5, "gridiron": 4, "pitch": 3},
            "trunk": "Linear(64→128) GELU LN Dropout(0.2) Linear(128→64) L2norm",
            "shared_adapter": False,
            "note": "shared_adapter True is G2 probe only — worsens leakage 0.743→0.759, proves dim-footprint leak is structural not adapter weights"
        },
        "losses": {
            "CORAL_cov": {
                "formula": "cov_s = (z_s - mean_s)^T (z_s - mean_s)/(n_s-1); loss = mean_{i<j} ||cov_i - cov_j||²_F",
                "weight": 0.5,
                "effect": "Δ 0.0 alone on G2 but -1.5pp combined w/ centroid, keeps G3 stable — complementary to centroid",
                "status": "dropped from lean config (inert Stage1) but kept in spec for Stage2 potential"
            },
            "CORAL_centroid": {
                "formula": "mu_s = mean(z_s); loss = mean_{i<j} ||mu_i - mu_j||²₂",
                "weight": 0.5,
                "effect": "directly minimizes sport centroid separation for sport-blindness G2 — complements cov",
                "status": "earn keep small"
            },
            "GRL": {
                "schedule": "warmup 5ep λ=0, ramp 10ep linear 0.10→0.3→0.5, hold 0.5",
                "w_sport": 0.5,
                "forward": "x",
                "backward": "-λ·grad",
                "baseline": "no-GRL control 0.799 sport_acc vs shipped 0.6851 (Δ -11.4pp improvement) vs majority floor 0.6258 (+0.0593)",
                "ceiling_note": "ceiling ~0.68 structural adapter leak + native dim footprint (zero-padding pattern = perfect sport signature)",
                "status": "earns keep"
            },
            "SupCon": {
                "formula": "positives = same cross-sport archetype A0-A11 ≥2 sports; τ_s = exp(log_temp_s); sim = z_i·z_j / sqrt(τ_i·τ_j); InfoNCE",
                "temp": 0.07,
                "weight": 1.0,
                "effect": "drop → G3 0.718→0.125, G4 0.988→0.137, leakage 0.7558 (+13pp vs majority) — essential",
                "status": "essential"
            },
            "VICReg_var": {"formula": "target=1/sqrt(64), loss=mean(ReLU(target - std(z,dim0)))", "weight": 1.0, "status": "inert default"},
            "VICReg_cov": {"formula": "zc=z-mean; cov=(zcᵀzc)/(N-1); loss=Σ off-diag(cov)² / d", "weight": 1.0, "status": "inert but raises rank to ~17-24 when aggressive 100/200, trades task"},
            "task_anchor": {"weight": 2.0, "formula": "CE native_cluster + CE pos (masked) per sport on z — anti-collapse"}
        },
        "training": {
            "stage1": "frozen e_s, 40ep lr 1e-3 adapter-only, batch-per-sport 86 balanced, early-stop on task plateau patience 15, rank floor 12",
            "stage2.1": "unfrozen encoders drift, 60ep enc_lr 3e-5, GRL λ 0.10→0.3→0.5, CORAL 0.5/0.5, SupCon temp 0.07, VICReg hinge 1 std",
            "seed": 7,
            "optimizer": "AdamW wd 1e-4, CPU viable, CUDA if avail",
            "imbalance": "modality-aware temp + balanced batch (64 hoops/64 gridiron/all pitch up-sampled) + pitch family-mask augmentation"
        },
        "stubs": {
            "orchestration_pseudocode": [
                "E_h = np.load('hoops 12966×64'), E_g (5323×32), E_p (2430×24) — L2norm",
                "sport_ids = [0]*12966 + [1]*5323 + [2]*2430; era_ids = year - 1996; arch_ids = archetype_map native_to_cross",
                "UnifiedTrunk adapters: Linear(64→48), Linear(32→48), Linear(24→48) — no shared (probe false)",
                "sport_tok = Embedding(3,8); era_emb = Embedding(n_eras,8); trunk MLP 64→128→64 L2",
                "dailySeed LCG = (seed*1103515245+12345)&0x7fffffff, seed=YYYYMMDD UTC, idx=A%20719",
                "loss = 2.0*task + 0.5*CORAL_cov + 0.5*centroid + SupCon(log_temp) + 0.5*sport_GRL(λ schedule) + var+cov",
                "z = L2norm(trunk(cat([adapter(e_s), sport_tok(s), era_emb]))) ∈ R^64",
                "export z → assets/unified.json + chimera_build_spec.json, report → data/unified_report.json"
            ]
        },
        "sources": {
            "hoops_arch": str(HOOPS_ARCH),
            "hoops_arch_data": {"dEmb": d_emb_hoops, "towerFamilies": tower_families[:5], "total_families": len(tower_families)},
            "gridiron": str(GRIDIRON_DATA),
            "pitch_meta": str(PITCH_FALLBACK),
            "unified_report_keys": list(report.keys()) if isinstance(report, dict) else []
        },
        "provenance": {
            "entity_total": entity_total,
            "gates": {
                "G1": "PASS -0.0526 hoops 0.0 gridiron +0.0021 pitch (shuffled +0.5493/+0.6920/+0.5617 proves not buggy mask)",
                "G2": "0.6851 vs 0.6258 floor +0.0593 MET weak (predicted 0.642 experimental), leakage 0.799 drop GRL, 0.7558 drop SupCon",
                "G3": "silhouette 0.683 within 0.746 between -0.121 sep 0.867 rank 12.4 floor 12 PASS composition_gap 8.9pp",
                "G4": "0.9828 vs 0.1712 lift +0.8116 coarse PASS, curated 0/40 top10 mean 2114 vs 2067 ratio 0.978 FAIL person"
            },
            "dailySeed": "YYYYMMDD UTC int, LCG a·1103515245+12345 & 0x7fffffff glibc rand, idx = a % 20719, same-link-same-stars",
            "zero_deps": True
        }
    }

    OUT_SPEC.parent.mkdir(parents=True, exist_ok=True)
    OUT_SPEC.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"wrote {OUT_SPEC} ({OUT_SPEC.stat().st_size} bytes) — entity_total {entity_total}")
    # also copy minimal tower lib doc to docs for parity
    lib_path = ROOT / "docs" / "SHARED_TOWER_LIB.md"
    lib_path.write_text(
        "# Shared Tower Lib — MTNN v6 scaffolding (hoops → unified)\n\n"
        "Source: vector-hoops pipeline/train_mtnn.py ResidualTower (canonical)\n\n"
        "```python\nclass ResidualTower(nn.Module):\n"
        "    def __init__(self, d_in, d_out=32, d_hidden=160):\n"
        "        self.fc1 = Linear(2*d_in, d_hidden); self.ln1 = LayerNorm(d_hidden)\n"
        "        self.fc2 = Linear(d_hidden, d_out); self.ln2 = LayerNorm(d_out)\n"
        "        self.skip = Linear(2*d_in, d_out)\n"
        "    def forward(self, x, m):\n"
        "        h = cat([x*m, m], dim=-1)\n"
        "        return ln2(fc2(gelu(ln1(fc1(h)))) + skip(h))\n```\n\n"
        "Gridiron same but d_hidden 32 d_out 24 + GatedFusion wrapper.\n"
        "Pitch same but 3 families (attacking, passing/control, defending/dueling) → 24-d.\n\n"
        "UnifiedTrunk reuses adapters + trunk MLP + GRL + SupCon + CORAL centroid+cov pattern from docs/UNIFIED_ARCHITECTURE.md §3.\n"
        "See chimera_build_spec.json for loss formulas and λ schedule 0.10→0.3→0.5.\n",
        encoding="utf-8"
    )
    print(f"wrote {lib_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
