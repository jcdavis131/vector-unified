# Stage 2.1 — Sweep plan (next attempt, once hoops artifacts stabilize)

> **Status:** PLAN. Stage 2 (v0, LR 1e-5 / 30 epochs) infrastructure is built and
> proven but its run was confounded by a concurrent hoops hillclimb overwriting
> the hoops artifacts (see `STAGE2_PLAN.md` §8). Even un-confounded, that config
> was never going to ship: G2 only moved ~0.07 (0.74→~0.70) vs the ≤0.43 target.
> This doc is the next concrete hill-climb step. **Precondition: the hoops lane
> must be stable and `load_live_encoders.py` smoke must again show cos=1.00000
> for hoops, and `unified_matrix.npz` must be rebuilt from the stabilized
> `embedding_v3.npz`.**

## 1. What Stage 2 v0 taught us (assuming the trend was real)

- Unfreezing the encode path (towers+fusion) **does** let the encoders drift
  toward a shared basis — G2 moved and G3 climbed (0.681→~0.75) — **without**
  breaking the per-sport games when G1 held (hoops role even improved +0.08 in
  the clean draw). So the Stage 2 mechanism is sound; the conservative LR/epoch
  budget was the limiter.
- Hoops is the **fragile** encoder (largest, 48-d, most heads): across draws its
  G1 role either improved (+0.08) or regressed (-0.03). Gridiron/pitch were
  rock-solid in every draw. Any sweep must watch hoops G1 first.
- 30 epochs at 1e-5 is roughly an order of magnitude short of erasing sport.

## 2. Levers (in priority order)

| Lever | v0 | v2.1 range | rationale |
|-------|----|-----------|-----------|
| encoder LR | 1e-5 | **3e-5** (primary), 1e-4 (aggressive) | biggest G2 mover; 3e-5 is 3× v0, still well under trunk LR |
| epochs | 30 | **60** | G2 was still dropping at epoch 30; give it room |
| GRL λ | 0.05 | **0.10** (primary), 0.15 | direct sport-invariance pressure |
| stagger | simultaneous | **pitch+gridiron first (15 ep), add hoops if they hold** | protects the fragile/contested hoops lane |
| GRL ramp | 10 | 5 | reach full λ sooner |
| trunk LR | 1e-3 | 1e-3 (unchanged) | trunk is stable; don't touch |

## 3. Proposed first sweep run (Stage 2.1-a)

```
encoder LR 3e-5, epochs 60, GRL λ 0.10 (ramp 5), stagger pitch+gridiron first
(add hoops at epoch 16 only if pitch+gridiron G1 held), G1 threshold 0.02,
cudnn deterministic + CUBLAS_WORKSPACE_CONFIG=:4096:8 (true reproducibility).
```

Staggering is the key safety change: it keeps the contested hoops encoder out
of the first half, so a hoops regression can't sink the run before
pitch+gridiron prove the shared basis is reachable. Hoops joins only once the
cheap sports show G2 dropping toward target without G1 loss.

## 4. Gating (unchanged contract, hardened)

- **G1 hard gate (per sport, every epoch):** live `e_s` kNN-5 role + position
  must not drop > 0.02 vs the Stage 0 (frozen) baseline. Hoops gets a 3-epoch
  grace window after it joins (it's noisy in early epochs).
- **Best checkpoint:** lowest G2 among epochs with rank ≥ 12 AND G1 within
  threshold (recorded, not blocking the save — shippability is a post-hoc
  verdict, see `train_stage2.py`).
- **Shippable** = G2 ≤ 0.43 (chance + 0.10) AND G1 holds for all 3 AND
  `load_live_encoders.py` smoke still cos ≥ 0.999 for all 3 at the saved
  encoder states. Only then does `export_unified.py` refresh `unified.json`.
- **Per-sport assets stay read-only.** Drifted encoder weights live only in
  `unified_stage2_best.pt["enc_states"]`.

## 5. Decision tree after 2.1-a

- G2 ≤ 0.43 and G1 holds → **ship Stage 2.1**, refresh `unified.json`, update
  arch §11 / SPEC §7.
- 0.50 < G2 < 0.43-miss but G1 holds and G2 still falling at epoch 60 → run
  **2.1-b** at 60 more epochs (warm-start from 2.1-a's best) before declaring.
- G2 plateau > 0.55 → the encoder LR ceiling is G1-bound (can't push harder
  without regressing hoops). Declare G2 a **soft target** (report the floor
  reached, ship the best G1/G3-improved state with an honest caveat, or keep
  v0.1). This is a user decision, not an auto one.
- Hoops G1 regresses even at 3e-5 staggered → drop hoops encoder LR to 1e-5
  while keeping pitch/gridiron at 3e-5 (asymmetric), re-run.

## 6. Reproducibility fix (carry into 2.1)

Stage 2 v0 had run-to-run variance because only `torch.manual_seed` was set —
cudnn/CuBLas matmul stayed non-deterministic (the warnings named
`CUBLAS_WORKSPACE_CONFIG`). For 2.1, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in
the env and `torch.use_deterministic_algorithms(True)` (no `warn_only`) so a
run is reproducible and a regression is real, not a draw.
