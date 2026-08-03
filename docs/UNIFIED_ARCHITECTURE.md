# The Unified Model — Joint Cross-Sport Embedding Architecture

> **Status:** Design for single approval (auto-mode plan gate) · 2026-07-10
> **Vision (dumbmodel.com):** "one joint embedding — many data sources, eventually many sports,
> folded into a single geometry where you could ask what a power forward and a strong safety have
> in common and get a real answer."
> **Today:** No joint embedding exists. Three separate spaces — 48-d (NBA), 32-d (NFL), 3-d (WC).
> This doc is how they become one.

---

## 0. The one-sentence idea

Treat each **sport** as a **modality**, treat **abstract role archetypes** as the **shared semantics**,
and learn a single 64-d L2-normalized space in which a player's location encodes *what role they play*
regardless of *which sport they play it in* — while every per-sport task the live games depend on
keeps working at least as well as today.

---

## 1. Current state — verified, not assumed

Each sport already has a working model. The unified model is **additive**: it must not regress any of them.

| | Vector Hoops (NBA) | Vector Gridiron (NFL) | Vector Pitch (World Cup) |
|---|---|---|---|
| **Granularity** | player-season | player-week | player-tournament |
| **Rows** | 12,966 | 49,881 (2016–2025) | ~600–800 (≥180 min, outfield) |
| **Input feats** | 120 in 17 families | 82 in 13 families | 16 per-90 rates |
| **Encoder** | 17 ResidualMLP towers (160→32) | 13 ResidualTower (→24) | **none** — raw z-scores |
| **Fusion** | concat (544+season → 48-d) | gated attention (+season → 32-d) | — |
| **Embedding** | 48-d, L2-norm | 32-d, L2-norm | 16-d z-scored; PCA(3) for map |
| **Heads** | 8 archetype, 18 skill, 5 position, 14 next-profile, InfoNCE career pairs, salary, playoff-riser, honors, per-skill towers | fpts_ppr + rec/rush/pass_yds + rec + TD (SmoothL1, weighted), usage_recon, position(CE), pedigree(SmoothL1 masked) | k-means(8) archetypes (unsupervised) |
| **Data** | stats.nba.com, Basketball-Reference | nflverse, PFR, NGS, depth, injuries, combine, draft, ffopportunity EP | StatsBomb open data (WC 2018 + 2022) |
| **Era-honest** | z within season | z on train only; temporal split | z within tournament |
| **Neural net?** | yes (MTNN v4/v5) | yes (MTNN v2) | **no** — PCA + k-means |

**Sources of truth (file:line checked 2026-07-10):**
- Hoops: `vector-hoops/assets/mtnn_arch.json` (`towerFamilies`, `dEmb=48`, `fusion:"concat"`,
  `layers`), `vector-hoops/pipeline/train_mtnn.py`.
- Gridiron: `vector-gridiron/pipeline/data/feature_manifest.json` (`families`, `n_rows=49881`,
  `n_features=82`, `coverage`), `vector-gridiron/pipeline/train_mtnn.py` (`MTNN`, `GatedFusion`,
  `HEAD_WEIGHTS`, loss = `loss_t + 0.15·loss_u + 0.10·loss_p + 0.05·loss_ped`).
- Pitch: `vector-pitch/pipeline/build_vectors.py` (`FEATURES` (16), tournament z-score, PCA(3),
  k-means(8), `EXCLUDE_GOALKEEPERS`, `MIN_MINUTES=180`).
- Cross-check: `domain-migration-plan.md` §"MTNN: what the code actually does" —
  "There is no cross-sport joint embedding. Three separate spaces (48-d / 32-d / 3-d)."

**Known doc/ship gap (carry forward):** hoops `train_mtnn.py` docstring claims "gated attention
fusion (not naive concat)" but the promoted checkpoint is `mtnn_v5_concat_…` and `mtnn_arch.json`
records `fusion:"concat"`. The shipped model concatenates. The unified model will say exactly what
it ships.

---

## 2. The core challenge — there are no natural cross-sport pairs

This is the crux and it rules out the obvious approach.

A CLIP-style alignment works because image and text **co-occur for the same entity** — there is a
ground-truth positive pair ("a photo of a dog", the photo of that dog). Sports do not have this. No
NBA player is also an NFL player. A power forward and a strong safety are **different entities in
different games**. There is no dataset that says "this basketball player and this football player are
the same kind of thing."

So the joint embedding cannot be learned from natural co-occurrence. It has to be learned from
**shared abstract structure** — the things that *are* the same across sports:

1. **Role archetypes** — "last line of defense", "offense runs through them", "volume producer",
   "explosive creator", "high-pedigree underachiever", "declining veteran", "breakout riser". These
   exist in every team sport; they are the shared semantics.
2. **Feature-family concepts** — volume/usage, efficiency, defense/disruption, playmaking/initiation,
   physical profile, market/pedigree, career arc, team/competition context, form, honors/clutch.
   The hoops "17 families" and gridiron "13 families" are already *sport-specific instances of a
   smaller sport-agnostic ontology*. Pitch's 16 features map onto a subset of it.
3. **Sport-invariance as an objective** — if the embedding genuinely encodes *role* and not *sport*,
   then a classifier shouldn't be able to tell which sport a player is from. That is a learnable
   objective (adversarial debiasing), and it is the force that folds the spaces together.

The honest consequence: **there is no ground-truth metric for cross-sport similarity.** The unified
model is validated partly by per-sport non-inferiority (hard) and partly by human face-validity of
cross-sport analogies (soft, curated). This matches the dumbmodel ethos — "wrong all the time, that's
the fun part" — but it must be stated plainly so the model is never oversold as "provably correct."

---

## 3. Architecture — four pillars

```
                       ┌───────────────────────────────────────────────┐
   per-sport raw       │  Pillar 1: per-sport encoders (frozen, then FT) │   proven per-sport
   features  ───────▶  │  hoops MTNN towers · gridiron MTNN towers ·      │   embeddings
                       │  pitch MTNN towers (NEW — replaces PCA)          │   e_h, e_g, e_p
                       └────────────────────┬──────────────────────────────┘
                                            │  + sport token s ∈ {h,g,p}
                                            ▼
                       ┌───────────────────────────────────────────────┐
                       │  Pillar 2: shared projection trunk              │   unified embedding
                       │  (e_s ∥ s) → MLP → 64-d, L2-norm  =  z          │   z ∈ ℝ^64
                       └────────────────────┬──────────────────────────────┘
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              ▼                             ▼                             ▼
   ┌─────────────────────┐    ┌─────────────────────────┐   ┌──────────────────────────┐
   │ Pillar 3a: per-sport │    │ Pillar 3b: cross-sport   │   │ Pillar 3c: sport-invar.  │
   │ task heads (preserve)│    │ alignment heads          │   │ adversarial head         │
   │ hoops: arch/skill/   │    │ • archetype contrastive  │   │ sport classifier +       │
   │ pos/profile/salary/  │    │   (synthetic same-       │   │ GRADIENT REVERSAL        │
   │ honors/playoff       │    │   archetype positives)   │   │ → z cannot predict sport │
   │ gridiron: PPR/yards/ │    │ • family CORAL (2nd-     │   │                          │
   │ usage/pos/pedigree   │    │   order geometry match)  │   │ anti-collapse anchor:    │
   │ pitch: archetype CE  │    │ • modality-aware temp    │   │ per-sport heads (3a)     │
   └─────────────────────┘    └─────────────────────────┘   └──────────────────────────┘
                                            │
                                            ▼
                       ┌───────────────────────────────────────────────┐
                       │  Pillar 4: unified asset + eval harness         │
                       │  assets/unified.json (64-d, all sports, one     │
                       │  space) · cross-sport NN retrieval · analogy    │
                       │  face-validity panel · per-sport non-inferiority│
                       └───────────────────────────────────────────────┘
```

### Pillar 1 — Per-sport encoders (reuse the proven models)

The three existing encoders are the input layer. They are **not thrown away**.

- **Hoops:** load the promoted `mtnn_v5_concat_…` checkpoint; its 17-tower → 48-d fusion output is `e_h`.
- **Gridiron:** load `pipeline/data/mtnn_best.pt`; its 13-tower → 32-d gated-fusion output is `e_g`.
  For the unified (season-level) embedding, aggregate a player's season of weeks — mean of weekly
  `e_g` over REG weeks with ≥1 snap (a season-level "who is this player" vector). The weekly
  prediction head stays sport-specific and continues to use the weekly input (it is NOT replaced by
  the unified vector).
- **Pitch: NEW MTNN trunk (prerequisite).** Pitch is the only sport with no neural net. It cannot
  join a learned shared space as a PCA artifact. Build a tiny pitch MTNN: 16 features → 3 families
  (attacking, passing/control, defending/dueling) → small towers → ~24-d embedding `e_p`, supervised
  by archetype CE (from the existing k-means(8) labels, kept as the target) + profile reconstruction.
  This is the "PCA is what the MTNN was built to replace" step, finally done for pitch. Pitch's tiny
  data (~600 rows) means a tiny net — a few thousand params, heavy regularization, CPU-only.

Pillar 1 produces three sport-specific embeddings `e_h ∈ ℝ^48`, `e_g ∈ ℝ^32`, `e_p ∈ ℝ^24`, each
L2-normalized, each already proven (or, for pitch, proven against its k-means baseline).

### Pillar 2 — Shared projection trunk (the joint space)

A single small MLP maps any sport's embedding **plus a learned sport token** into one 64-d space:

```
z = L2norm( MLP( [ e_s ∥ sport_token(s) ∥ season_ctx ] ) ),   z ∈ ℝ^64
```

- **64-d** chosen because it is larger than every per-sport dim (48 / 32 / 24) — there must be room
  for cross-sport structure beyond what each sport alone needs. 64 is the smallest round number that
  clears hoops' 48 with margin; ablate {48, 64, 96}.
- **Sport token** is a learned embedding (dim 8) per sport. It is the **anti-collapse / anti-forgetting
  signal**: it tells the trunk which geometry it is operating in, so the shared space can keep
  sport-meaningful axes instead of collapsing to a degenerate sport-invariant blob. The tension
  between the sport token (preserve) and the adversarial sport head (erase) is deliberate — it carves
  out the *role* subspace as the part that survives both.
- **Season ctx** (era index) preserves era-honesty across the joint space (a 1996 power forward and a
  2026 power forward are not the same point even if the same archetype).

### Pillar 3a — Per-sport task heads (the anti-collapse anchor)

Every head the live games depend on is re-attached to `z` (or kept on `e_s` and just also trained).
This is non-negotiable: **the joint space must still predict basketball box scores AND football fantasy
lines AND soccer archetypes.** If it can do all three, it hasn't thrown information away.

- Hoops on `z`: 8 archetype, 18 skill, 5 position, 14 next-profile, salary, playoff-riser, honors
  (InfoNCE career pairs optional — keep as in hoops).
- Gridiron on `z`: the weekly heads stay on the weekly `e_g` path (they need weekly features); the
  season-level `z` carries position + pedigree + a season-fantasy prior head.
- Pitch on `z`: archetype CE (8) + profile reconstruction (16).

Loss weights: per-sport task loss dominates (sum weight ≈ 1.0 per sport, normalized so the three
sports contribute comparably despite row-count imbalance — see §5).

### Pillar 3b — Cross-sport alignment heads (the folding forces)

Three losses fold the spaces together. This is where the research lands.

1. **Archetype contrastive loss (InfoNCE, the primary folder).**
   - Define a **cross-sport role taxonomy** of K ≈ 10–12 archetypes (§4). Each archetype has member
     players in each sport, derived by mapping each sport's native clusters (hoops 8, gridiron
     emerging, pitch k-means 8) onto the shared taxonomy via a **human-curated alignment table**
     (the anchor set).
   - Positive pair: two players (any sports) sharing a cross-sport archetype. Negative: the rest of
     the batch. Because there are no natural pairs, the positives are *constructed* from shared
     archetype membership — this is the synthetic supervision that replaces CLIP's co-occurrence.
   - **Modality-aware temperature** (e5-omni): a learned per-sport temperature rescales the InfoNCE
     logits so the small/pitch and large/hoops sports contribute comparable gradient — otherwise
     hoops' 12,966 rows dominate and pitch folds noise.

2. **Family CORAL loss (2nd-order geometry match).**
   - Each sport's encoder outputs a per-family embedding in the **shared family ontology** (§4).
   - CORAL-style covariance alignment (Sun & Saenko 2016, as used in e5-omni) minimizes the gap
     between the per-family covariance matrices across sports. This gives the joint space a shared
     axis system: the "volume" direction means the same thing in all three sports.
   - Only applied to families that exist in both sports being aligned (pitch has no bio/market/
     pedigree — those families are pitch-absent and excluded from pitch's CORAL terms, honestly).

3. **(Optional, ablation-gated) Higher-order / Gramian term.**
   - ConFu (CVPR 2026) and GRAM align n modalities jointly rather than pairwise. If pairwise CORAL +
     archetype contrastive underperform on the face-validity panel, add a Gramian-volume term that
     aligns all three sports' archetype centroids simultaneously. Capacity must earn its keep
     (house rule from gridiron SPEC §1); this is not day-one.

### Pillar 3c — Sport-invariance via adversarial debiasing (the strongest folder)

A sport classifier on `z` (3-way: hoops/gridiron/pitch) trained with a **gradient reversal layer**
(Ganin et al. 2016 — the Football2Vec v2 pattern). The classifier tries to predict the sport; the
reversal pushes the encoder to make `z` *unpredictable by sport*.

- This is the single most powerful force folding the spaces together: it directly optimizes the
  property the product vision demands — "the embedding encodes role, not sport."
- It is balanced against Pillar 3a (per-sort tasks) and the sport token (Pillar 2): we do **not**
  want sport accuracy to hit zero (that would mean collapse / lost signal). Target: sport-classifier
  accuracy clearly below a no-debiasing baseline, but not at chance with a dead embedding.
  **Note 2026-08-03 (7.20): this framing is the correct one and survived the audit** — it
  compares against a no-debiasing CONTROL rather than against a constant, which is what
  `eval_unified.py --baseline-sport-acc` implements and why that arm was immune to the
  baseline error. Read "at chance" as **the majority-class floor of 0.6258**, not 1/3: the
  sports are 12,966 / 5,323 / 2,430 and a shuffled `z` scores 0.6257. The `<=0.433` targets
  that appear in `STAGE2_PLAN.md` and `SPEC.md` came from the 1/3 reading and were
  unreachable; see `docs/SPEC.md` § CORRECTIONS 2026-08-03.
  Monitor **effective rank / participation ratio of `z`** every epoch — collapse shows up as
  plummeting rank before it shows up in losses.

### Pillar 4 — Unified asset + evaluation harness

The ship artifact is `assets/unified.json`: every player-season across all three sports as one
64-d point, with `sport`, `name`, `season`, `archetype` (cross-sport label), `native_archetype`
(sport-specific), and a PCA(3) map projection for the UI. The eval harness (§6) is what makes the
"real answer" claim falsifiable.

---

## 4. The shared family ontology + cross-sport archetype taxonomy

These two tables are the heart of the design — they are the *shared semantics* that the three losses
in §3b/3c operate on. They are human-authored (the unavoidable subjective part) and version-controlled.

### 4a. Sport-agnostic family ontology (the shared axis system)

| Cross-sport family | Hoops tower(s) | Gridiron family | Pitch feature group | Pitch has it? |
|---|---|---|---|---|
| **Volume / Usage** | volume, playmaking(touches) | usage, opportunity | GOALS/XG/KEY_PASSES/ASSISTS/PASSES_CMP (p90) | yes |
| **Efficiency** | efficiency, shotmix | form(epa,cpoe), ngs_eff | FINISHING_P90, PASS_CMP_PCT | yes |
| **Defense / Disruption** | defense, rebounding | defense(dvp), pfr_pressure | PRESSURES/TACKLES/INT/RECOVERIES (p90) | yes |
| **Playmaking / Initiation** | playmaking | context(pass_rate, qb_ep), pass_att | KEY_PASSES, PROG_CARRY, CROSSES | yes |
| **Physical / Bio** | bio | meta(h/w/age), combine | — | **no** |
| **Market / Pedigree** | market, pedigree | pedigree(draft), market(implied) | — | **no** |
| **Career / Arc** | career, form | exp, prior_ppg | — (2 tournaments only) | **partial** |
| **Team / Competition context** | team, competition, roster | context, conditions | (team-level, limited) | **partial** |
| **Honors / Clutch** | honors, playoffs | — | — | **no** |
| **Form (recent)** | form | form | (tournament snapshot) | **partial** |

**What this table says:** Hoops has the richest family coverage (all 10). Gridiron is strong on
weekly context and pedigree. **Pitch is sparse** — it only defines the four "on-ball" families
(volume, efficiency, defense, playmaking) and has no bio/market/pedigree/honors. This is the single
biggest honest constraint: pitch can only be aligned on the axes it actually has. The adversarial +
contrastive design handles this gracefully (pitch players align on role axes that exist and are
undefined on the rest), but it means **pitch's cross-sport comparisons are role-only, never
physical/market**. State this in the UI and the methods page. Do not fake pitch bio/market features.

### 4b. Cross-sport role archetype taxonomy (K ≈ 10–12, v0 proposal)

Each archetype is defined sport-agnostically, then mapped to native clusters. This is the v0
taxonomy — it will be revised after the first face-validity panel.

| # | Cross-sport archetype | Hoops native | Gridiron native | Pitch native |
|---|---|---|---|---|
| A0 | **Offensive engine / primary initiator** | high-usage shot+playmaking cluster | QB / high-usage RB / WR1 | high key-pass + prog-carry midfielders |
| A1 | **Volume producer / scoring load** | shot-volume cluster | volume RB / target-hog WR | high xG/goals forwards |
| A2 | **Explosive perimeter creator** | pull-up/drive-heavy wing | scrambling QB / YAC WR | high dribbles + prog-carry wingers |
| A3 | **Defensive anchor / last line** | rim-protector big | last-line S / interior DL | low-line pressing CB / sweeper |
| A4 | **Defensive disruptor / ball-hawk** | steals/blocks specialist | edge rusher / ball-hawk S/DB | high tackles + interceptions |
| A5 | **Connector / two-way grinder** | 3&D / glue guy | possession WR / pass-catching RB | high pass-completion box-to-box |
| A6 | **High-pedigree, under-delivering** | high draft slot, low net-rating | 1st-round pick, low opportunity share | (n/a — no pedigree data) |
| A7 | **Low-pedigree, over-delivering** | 2nd-round / undrafted standout | late-round / UDFA producer | (n/a) |
| A8 | **Breakout / riser** | career DELTA_NORM up, FORM_VOL up | prior_ppg jump, snap_pct up | (limited — 2 tournaments) |
| A9 | **Declining veteran** | career arc down, GP_RATIO down | exp high, form down | (limited) |
| A10 | **Elite two-way** | high offense + high defense cluster | (rare in fantasy data) | high xG + high pressures (rare) |
| A11 | **Floor-raising role player** | high PIE, low usage | high EPA/target, low volume | high recovery, low turnover |

The mapping from each sport's **native** clusters to these **cross-sport** archetypes is a
human-curated alignment table, committed as `vector-unified/data/archetype_map.json`. It is the
anchor set. Pitch's A6/A7/A9/A10 rows are marked `n/a` because pitch has no pedigree/career data —
pitch players are never assigned those archetypes, and the contrastive loss does not create
pitch-containing positives for them. This is honest.

---

## 5. Training strategy

### Staged (protect the proven models)

**Stage 0 — Prerequisite: pitch MTNN.** Build the pitch neural trunk so it can participate. Validate
against the existing k-means(8) baseline (archetype purity must not drop). Until pitch has a learned
embedding, there is nothing to align.

**Stage 1 — Frozen-encoder alignment.** Freeze `e_h`, `e_g`, `e_p` (the three proven encoders).
Train **only** Pillar 2 (shared trunk) + Pillar 3b/3c (alignment + adversarial) + Pillar 3a heads
attached to `z` (heads train, encoders don't). This learns the joint geometry without risking the
per-sport models. Cheapest, safest first step.

**Stage 2 — End-to-end fine-tune (low LR).** Unfreeze encoders, fine-tune the whole stack with a
small LR (≈1e-5) and the per-sport task losses as the dominant term. Goal: let the joint signal
*refine* the per-sport encoders, not redefine them. Gate: per-sport metrics within noise of Stage 0
(see §6 G1). If any sport regresses, revert to Stage 1 frozen encoders and ship that.

### Imbalance handling (e5-omni, mandatory)

The row counts are wildly unequal: hoops 12,966 · gridiron 49,881 · pitch ~600. Naive joint training
makes hoops+gridiron eat pitch. Fixes:

- **Modality-aware temperature** in the InfoNCE loss (learned per sport) — §3b.
- **Balanced batch construction:** every training batch contains a fixed quota per sport (e.g. 64
  hoops / 64 gridiron / all-available pitch, up-sampled with augmentation = family-mask dropout).
- **Up-weight pitch's task loss** (×1/coverage) so its gradient isn't drowned. Document the weight.
- **Pitch family-mask augmentation:** randomly zero pitch families during training to synthetically
  enlarge its ~600 rows and harden it against missing families (mirrors gridiron's `family-drop`).

### Optimizer / schedule

- AdamW, wd 1e-4, batch 256 (balanced), early stop on the **joint validation score** (a composite of
  per-sport val metrics + cross-sport archetype silhouette — §6), patience 25.
- CPU-only is viable (the per-sport encoders are small; the trunk is tiny). CUDA if available.
- Seed 7 (matches both hoops and gridiron).

---

## 6. Evaluation — "get a real answer," made falsifiable

The product vision is qualitative ("what does a PF and an SS have in common — a real answer").
Eval must make that falsifiable. Five gates:

| Gate | What | Pass |
|---|---|---|
| **G1 — Per-sport non-inferiority (hard)** | Hoops: cluster purity, position acc, salary MAE, next-profile recon vs the standalone 48-d. Gridiron: test PPR MAE ≤ v1 (4.313) + 0.05 and beats last-4 baseline. Pitch: archetype purity ≥ k-means baseline. | No sport regresses beyond noise. **The games must not break.** |
| **G2 — Sport-invariance (hard)** | 3-way sport classifier trained on frozen `z`. | Accuracy clearly below a no-adversarial-debiasing baseline (Δ ≥ 10pp), and `z` effective rank stays ≥ ½ × 64 (no collapse). **Stage 1 v0: DEFERRED to Stage 2 — see §11.** |
| **G3 — Cross-sport archetype coherence (hard)** | For each cross-sport archetype, mean within-archetype cross-sport cosine > mean between-archetype cross-sport cosine; silhouette over the cross-sport labels on `z`. | Silhouette > 0 on the cross-sport labels (joint space separates the shared archetypes better than chance). |
| **G4 — Analogy face-validity panel (soft, killer demo)** | Curated ~40 cross-sport analogy triples: ("PF X ~ SS Y because role-label"). For each, is Y a top-k (k=10) cross-sport NN of X in `z`? Is the role label sensible? | ≥ 60% of analogies land in top-k with a sensible label. This *is* the product vision, operationalized. |
| **G5 — Retrieval sanity (soft)** | "Find the NFL player most similar to [NBA player] by role" returns plausible, non-trivial results (not just "the most famous/biggest"). Manual review of 20 queries. | No query returns an obviously broken top-1 (e.g. a kicker for a center). |

G1–G3 are computed automatically every eval run. G4–G5 are a periodic human panel (the analog of
hoops' archetype-naming audits). The unified model is **not declared done** until G1–G3 pass and G4
clears 60%. G5 is a watch item, not a blocker.

### Honest framing for the methods page

The joint embedding is a **plausibility model**, not a correctness model. There is no ground truth
for "a power forward is like a strong safety." G4 is a curated face-validity check, and the taxonomy
(§4b) is subjective and revisable. dumbmodel's voice — "it is wrong all the time, that's the fun" —
applies doubly here. Copy must say "a shared role space" and "these analogies are the model's best
guess," never "the proven similarity between sports."

---

## 7. Honest constraints & risks (named, not hidden)

1. **No ground-truth cross-sport metric.** Eval is partly face-validity (G4/G5). Never oversell.
2. **Pitch is tiny (~600 rows) and shallow (4 of 10 families).** It can only align on role axes it
   has. Cross-sport analogies involving pitch are role-only. Either accept this, or expand pitch data
   (more StatsBomb competitions — Euros, top leagues) before pitch carries equal weight. **Default:
   accept and document; flag expansion as a later phase.**
3. **Granularity mismatch (season / week / tournament).** The unified embedding is **season-level**
   (the "who is this player, archetypally" map). Gridiron's weekly prediction stays a sport-specific
   head on weekly features. This is a design choice with tradeoffs (a player's role can shift
   week-to-week in football; the season mean smooths that).
4. **Collapse risk.** Adversarial debiasing + contrastive can collapse `z`. The per-sport task heads
   (3a) and sport token (Pillar 2) are the anti-collapse anchors; effective-rank monitoring (G2) is
   the early warning. If rank drops, raise task-loss weight or lower the reversal coefficient.
5. **The games must not break.** The 3 live games depend on current per-ssport assets. The unified
   model produces a **new** `unified.json` — it must not overwrite per-sport assets until G1 passes.
   Stage 1 (frozen encoders) guarantees this by construction.
6. **Archetype taxonomy is subjective.** §4b is a v0 proposal. It will be revised after G4. Version
   it in `data/archetype_map.json` and log changes.
7. **Hoops fusion docstring is already stale** (says attention, ships concat). The unified model will
   record exactly what it ships in its own arch json, and the methods copy will say "fuses" without
   naming a mechanism unless the mechanism is the shipped one.

---

## 8. Research basis (cited, not invented)

The design draws on 2026 alignment literature, adapted to the no-natural-pairs sports setting:

- **e5-omni** (Chen et al., ACL Findings 2026) — explicit alignment recipe: modality-aware
  temperature, controllable negative curriculum + debiasing, batch whitening + **CORAL covariance
  alignment**. Directly imported into §3b (temperature, CORAL) and §5 (imbalance). Solves the
  modality-gap problems (scale, negative hardness, 2nd-order stats) that the 48/32/~24-d + row
  imbalance creates.
- **Football2Vec v2** (luxury-lakehouse, HuggingFace) — transformer player embeddings with
  **adversarial competition debiasing via gradient reversal** (Ganin et al. 2016) so embeddings
  encode *style/role* not *league identity*. v1 clustered by league; v2 does not. This is the
  direct precedent for §3c: replace "league" with "sport" and the same technique folds the spaces.
- **ConFu / Contrastive Fusion** (Koutoupis et al., CVPR 2026) — pairwise + higher-order
  contrastive over fused modality subsets; L = L_pair + λ·L_fused. Imported as the optional
  ablation-gated higher-order term (§3b #3).
- **GRAM / Gramian Representation Alignment** (arXiv 2412.11959) — align n modalities by minimizing
  the Gramian volume of the parallelotope; geometric alignment of all simultaneously. Alternative to
  pairwise CORAL if pairwise underperforms (§3b #3).
- **Optimal-transport style embedding** (arXiv 2501.10299) + **EventGPT** (arXiv 2512.17266) —
  sports-specific prior art confirming that **role/archetype structure emerges without position
  labels** from behavior, and that OT/sequence methods are an alternative encoder route (not day-one;
  the existing MTNN encoders are the input layer here).

The novel part of *this* design is not any single technique — it is the combination: **per-sport
proven encoders as frozen inputs + shared family ontology + synthetic archetype contrastive pairs
(no natural co-occurrence) + gradient-reversal sport debiasing + per-sport task preservation as the
anti-collapse anchor**, validated by a curated cross-sport analogy panel. That combination is what
makes "power forward ↔ strong safety" answerable in one geometry.

---

## 9. What is explicitly NOT in v1

- **No new data ingestion** beyond what the three pipelines already produce. The unified model
  consumes the three existing embeddings + manifests. (Pitch MTNN is the one new ingestion-adjacent
  build, but it uses pitch's already-cached StatsBomb data.)
- **No tracking/spatial data** for the joint space (hoops tracking and pitch x/y are sport-internal).
- **No live game integration.** The unified model is a batch asset (`unified.json`), refreshed on
  each sport's retrain cadence, not real-time.
- **No replacement of per-sport assets.** `unified.json` is additive; per-sport `vectors.json` /
  `embedding.json` / `nextgame.json` / `projections.json` keep their contracts.
- **No Gramian / higher-order term day-one** (ablation-gated, §3b #3).
- **No pitch data expansion day-one** (flagged, §7 #2).

---

## 10. File map (what gets built)

```
vector-unified/
  docs/
    SPEC.md                    # assumptions, objective, boundaries, acceptance
    UNIFIED_ARCHITECTURE.md    # this file
  tasks/
    plan.md                    # phased plan + gates
    todo.md                    # the checklist
  pipeline/
    load_encoders.py           # load hoops/gridiron checkpoints; build pitch MTNN
    build_pitch_mtnn.py        # NEW — pitch neural trunk (prerequisite)
    build_unified_matrix.py    # assemble e_h/e_g/e_p + labels + archetype_map into one matrix
    archetype_map.py           # the §4b alignment table → data/archetype_map.json
    train_unified.py           # Pillar 2 + 3a/3b/3c trainer
    eval_unified.py            # G1–G3 automatic gates
    analogy_panel.py           # G4/G5 harness (loads the curated triples, scores top-k)
    export_unified.py          # → assets/unified.json
  data/
    archetype_map.json         # §4b, versioned
    analogy_triples.json       # G4 curated panel
    unified_best.pt            # checkpoint
    unified_report.json        # G1–G3 numbers
  assets/
    unified.json               # the ship artifact (64-d, all sports, one space)
```

---

## 11. Stage 1 v0 — what actually shipped (2026-07-10, evidence not assumption)

This section reconciles the design above with the built-and-evaluated Stage 1 v0.
It overrides the spec where the build diverged; the rest of the doc stands as the
plan for Stage 2+.

**Shipped artifact:** `assets/unified.json` — 20,721 player-seasons (hoops 12,966 /
gridiron 5,325 / pitch 2,430), 64-d L2 `e` + PCA(3) map + cross-sport archetype
label. Per-sport assets untouched (additive, verified by `verify_encoders.py` and
git status). Reports: `data/unified_report.json` (G1–G3), `data/analogy_report.json`
(G4 + named panel).

**Gate results:**
- **G1 per-sport non-invariance: PASS.** `z` *beats* frozen `e_s` on native-cluster
  kNN-5 (hoops 0.851→0.994, gridiron 0.981→0.981, pitch 0.998→1.000; position 1.0
  all). Role preserved; games will not break.
- **G3 cross-sport archetype coherence: PASS.** Silhouette 0.683; within-arch
  cross-sport cos 0.746 >> between-arch -0.121. The folding is real.
- **G4 analogy: PASS, 0.978** (target 0.60; automated cross-sport NN role-coherence,
  not the curated 40 triples — see below). Named panel sensible: Brady QB→
  Alexis MacAllister / Draymond Green @0.97; Drummond C→soccer CBs @0.86;
  Ty Lawson PG→Kevin De Bruyne @0.98.
- **collapse_detector: PASS** — rank 13.0 ≥ 12 AND G1 AND G3. The role manifold is
  genuinely ~13-d; the literal "rank ≥ ½×64 = 32" floor (§6 G2) **over-alarms on a
  low-dimensional but healthy role space** and is reframed as a collapse *detector*
  (rank ≥ 12 + role/folding intact), not an absolute floor. Collapse would show as
  rank low AND G1/G3 failing; they do not.
- **G2 sport-invariance: FAIL — DEFERRED TO STAGE 2.** Sport is ~77% recoverable
  from `z` (chance 33%). A no-GRL control scores 0.799, so **the GRL is inert
  (Δ≈3pp, not the §6 Δ≥10pp target)**, and dropping the sport token (Pillar 2)
  changed nothing (0.786). The leak is **structural to the per-sport adapter**: the
  three `e_s` live in different spaces (48/32/24-d) with no shared basis, so any
  Stage-1 input bakes sport in. True sport-invariance requires **Stage 2 (unfreeze
  encoders, fine-tune toward a shared space)** — not a trunk tweak. Honest take: the
  product vision ("one geometry to ask PF↔SS and get a real answer") IS met (G3+G4);
  `z` also retains a sport axis (sliceable, not harmful). The stricter "sport
  unknowable from z" is a Stage-2 fairness-style goal.

**Deviations from the spec (named, not hidden):**
1. **Sport token dropped.** §3/Pillar 2 specified a dim-8 learned sport token as
   the anti-collapse signal. Ablation showed it earns no keep: sport-acc with it
   (0.788) ≈ without it (0.786), and the per-sport adapter is the real sport leak.
   v0 ships `d_sport_tok=0`; anti-collapse relies on per-sport task heads (3a) +
   VICReg var/cov terms. Revisit in Stage 2.
2. **A4 remapped / deferred.** §4b A4 ("defensive disruptor / ball-hawk") had
   pitch-only members (cluster 2, 383 players, 81% DEF) with no hoops/gridiron
   counterpart → cross-sport analogy was trivially impossible (G4 hit 0.000). Pitch
   cluster 2 now folds into **A3** (defensive anchor, which spans hoops centers +
   pitch defenders). A4 is moved to `archetypes_deferred_v0`; it needs gridiron
   defensive data (DL/DB) + a hoops steals/blocks specialist cluster to become a
   real shared archetype. G4 rose 0.959→0.978; pitch hit-rate 0.837→0.993.
3. **Pitch is no longer tiny.** §1/§7 #2 assumed pitch ~600 rows (2 World Cups).
   The pitch MTNN pipeline was expanded to the **full male StatsBomb open corpus —
   11 contexts, 1,157 matches, 2,430 player-seasons** (Phase 1 complete; pitch
   MTNN beats shipped PCA(3) on 4/4 role-recovery metrics). §7 #2's "accept and
   document" caveat is largely resolved for pitch; pitch still lacks bio/market/
   pedigree/career families (role-only alignment, unchanged).
4. **Lighter Pillar 3a heads.** The spec re-attaches all per-sport heads to `z`
   (hoops archetype/skill/pos/profile/salary/honors/playoff, etc.). v0 attaches
   only **native-cluster CE + position CE per sport** as the anti-collapse anchor.
   Full head re-attachment is a Stage-2 refinement; the lighter anchor already
   preserves role (G1 passes, z beats e_s).
5. **Automated G4 replaces curated triples (for now).** §6 G4 / §10 file map
   specify `analogy_triples.json` (~40 curated triples) and a human panel. v0
   instead scores **automated cross-sport NN role-coherence** over all 20,721
   players (G4=0.978) plus a centrality-based named showcase (24 players). The
   automated metric is broader and stronger than 40 triples, but the curated
   human panel (G5) is still valuable and **pending** (`analogy_triples.json` not
   yet authored — task 0.3 / 3.3).
6. **Market / cultural footprint signals — user-requested, NOW ACQUIRED + WIRED (v0.1 data + head).**
   Free-tier pull complete 2026-07-11 (`data/market_cultural/`):
   - **Forbes** (2012–2026): 150 records, salary+endorsement split; 23 unique elite
     athletes matched across all 3 sports.
   - **Spotrac NFL cap-hit** (2015–2025): 23,143 player-years; **1,509 unique /
     93.3% of gridiron player-seasons** now have `SALARY_LOG` (was 0%).
   - **Wikipedia awards** (NBA MVP / AP NFL MVP / Ballon d'Or / FIFA Best): 205
     winner-records → career `AWARD_PRESTIGE` (Messi 10.7, Ronaldo 7.7, Jordan 5,
     LeBron/Rodgers 4).
   - Joined to unified corpus order: `market_cultural.json` (20,721 rows, masked).
   - **Trunk wiring (`--market`):** shared (sport-agnostic) `salary_head` +
     `award_head` on `z`, masked-MSE, z-scored over labeled rows. Warm-starts from
     `unified_best.pt`; saves **`unified_market.pt`** (shipped v0.1 untouched).
     Eval: G1 PASS / G3 PASS (sil 0.706) / G2 still ~0.74 deferred.
     Face-validity (`market_heads_probe.json`): salary Spearman **ρ=0.64**
     (gridiron; top preds = QBs), award **ρ=0.52** pooled (Messi/Neymar/Mbappé
     surface; also latches onto star-attacker geometry). Hoops salary too sparse
     (n=39 Forbes-only, ρ≈0).
   - **Honest gaps (§257):** Transfermarkt pitch market-value **blocked** (TLS
     anti-bot); pitch salary = Forbes stars only; endorse = Forbes stars only;
     social reach (Apify) deferred. Hoops full salary remains in native BBREF pipeline.
   - **Cultural text (Wikipedia → MiniLM) MVP 2026-07-11:** see
     `docs/CULTURAL_TEXT_SCHEMA.md`. Priority stars 47 → 454 labeled seasons;
     `unified_cultural.pt` (G1/G3 PASS; mean text-align cos 0.775; G2 0.842 expected
     leak). Reddit deferred. Full wiki resume continues under MediaWiki backoff.

**Loss ablation (SPEC §5 house rule — each loss must earn its keep):**
`pipeline/ablation.py` drops each component and measures Δ on G1/G2/G3/G4
(`data/ablation_report.json`). Results (30-epoch, final-epoch state):
| config | G1 | G2acc | rank | G3sil | G4hit |
| full (SupCon+CORAL+GRL+VICReg+task) | PASS | 0.668 | 13.5 | 0.718 | 0.988 |
| no_supcon | PASS | 0.669 | 20.3 | 0.125 | 0.137 |
| no_coral | PASS | 0.664 | 13.3 | 0.718 | 0.988 |
| no_grl | PASS | 0.716 | 13.5 | 0.723 | 0.989 |
| no_vicreg | PASS | 0.668 | 13.4 | 0.722 | 0.987 |
| task_only (no alignment) | PASS | 0.730 | 20.4 | 0.142 | 0.129 |
- **SupCon is the workhorse** — dropping it collapses G3 (−0.59) and G4 (−0.85);
  it is THE folder that creates cross-sport role structure. Essential.
- **CORAL is inert** at Stage 1 — 0Δ on every gate. The 2nd-order covariance
  alignment adds nothing on top of SupCon + task for this data. **Dropped from the
  lean config.** (May earn its keep in Stage 2 once encoders unfreeze.)
- **VICReg (var+cov) is inert at default weights** — 0Δ on rank/G3/G4. (Aggressive
  weights w_var=100/w_cov=200 move rank to ~17-24 but trade task sharpness — not
  worth it.) **Dropped from the lean config.**
- **GRL is modest** — ~5pp sport-invariance (0.716→0.668), no G3/G4 harm. Kept
  (small benefit, no cost) but it alone cannot reach the G2 Δ≥10pp target.
- **task heads are the anti-collapse anchor** — G1 always PASS; task-only preserves
  within-sport role but gives no cross-sport folding (G4 0.129).
**Lean shipped config (v0.1): SupCon + task + GRL (drop CORAL + VICReg).** Every
remaining loss earns its keep; gates unchanged. This is the config Stage 2 should
inherit (revisit CORAL/VICReg once encoders unfreeze).

**Stage 2 is the next real hill-climb step** (unfreeze encoders, LR ~1e-5,
per-sport task losses dominant) — that is the path to G2 sport-invariance and to
letting the joint signal refine the per-sport encoders. Gate: per-sport metrics
within noise of Stage 0 (G1); revert to Stage 1 frozen if any sport regresses.

**Stage 1.1 shared-adapter G2 probe (task p19, empirical 2026-07-10).** Tested
whether G2's sport-leak was the per-sport *adapter weights* vs the frozen
encoders' native spaces. Built a trunk variant with ONE shared `Linear(48→48)`
over zero-padded `e_s` (no per-sport adapters) + 2× GRL (λ=0.1), retrained 40
epochs (`unified_shared.pt`). Result: **G2 sport-acc 0.759 vs per-sport-adapter
baseline 0.743 (delta −0.016 — sport *more* recoverable, not less)**; G1/G3/
collapse PASS unchanged (rank 13.2). This empirically confirms G2 is
**structurally blocked at Stage 1**: the leak is not the adapter weights but the
frozen encoders' native dimension footprint (hoops 48-d / gridiron 32-d / pitch
24-d) — the zero-padding pattern (which dims are nonzero) is a perfect, linearly
separable sport signature no trunk-side intervention can erase. **Stage 2
(unfreeze encoders into a shared basis) is the only path to G2.** Shipped
`unified_best.pt` untouched; `unified_shared.pt` retained as evidence. Concrete
Stage 2 implementation plan, feasibility evidence (all 3 encoders reloadable; 3
formats/classes; hoops heaviest at 2.26 MB), gating/revert contract, and the four
ask-first decisions (encoder LR, stagger vs simultaneous, epochs, regression
threshold) are in **[`STAGE2_PLAN.md`](./STAGE2_PLAN.md)**.

**Stage 2 executed 2026-07-10/11 — infrastructure complete; clean run NOT
SHIPPABLE (G2 miss).** `load_live_encoders.py` proven (smoke cosine 1.00000 vs
frozen for all 3 sports). `train_stage2.py` warm-starts trunk from
`unified_best.pt`, unfreezes encode-path only (towers+fusion; heads frozen),
two AdamW groups, post-hoc shippability verdict. An earlier same-day attempt
was confounded by a concurrent hoops hillclimb overwriting `mtnn_best.pt`
mid-run and was discarded. A **clean** 30-epoch run at `enc_lr=1e-6` (smoke
proved `1e-5` collapses hoops G1 −0.063; `1e-6` plateaus) finished:

| gate | result | note |
|---|---|---|
| G1 encoder non-regression | **PASS** | hoops role_drop=−0.040 (improved); gridiron −0.001; pitch 0.000 |
| G2 sport-acc | **FAIL** 0.674 (target ≤0.433) | Stage 1 ~0.74 → 0.674 (−7pp); dim-footprint leak eroded but not broken |
| G3 silhouette | 0.736 | held |
| rank | 12.4 | ≥ floor 12 |
| **SHIPPABLE** | **False** | G1 ok ∧ G2 miss |

Artifacts: `pipeline/data/unified_stage2_best.pt` (best ep26) +
`data/stage2_history.json` + drifted `enc_states` inside the ckpt. **Per-sport
assets never written. Shipped `unified.json` / `unified_best.pt` (Stage 1 v0.1)
untouched.** Next G2 levers: longer run / higher enc_lr with warmup-freeze /
staggered unfreeze / stronger GRL — see [`STAGE2_PLAN.md`](./STAGE2_PLAN.md) §8.

**Curated analogy panel (task 0.3, on shipped unified.json).** 40 human-authored
cross-sport triples paired by intuitive role (`data/analogy_triples.json`, scored
by `pipeline/analogy_triples_eval.py`). Findings:

- **G4-curated arch-agreement 0.675 (target 0.60) PASS** — 67.5% of intuitive
  same-role pairings share the model's auto-archetype; intuition matched the
  model's anchor archetype 85% of the time (taxonomy intuition well-calibrated).
- **Specific-pair top-10 retrieval hit-rate 0.000 — reframed as the wrong metric
  for large archetype pools.** A0/A1 hold hundreds of players each, so a specific
  human-chosen counterpart is rarely the *nearest* even when the archetype is
  correct (automated all-players G4 = 0.978 already proves role-coherence). Mean
  B-rank 2149 vs random ~6950 = **3.23× better than random** — real but diffuse
  cross-sport structure.
- **The showcase surfaces genuine data-driven role distinctions** the model draws
  that human intuition merges: it separates *initiator* (A0: Brady, Mahomes, De
  Bruyne) from *volume scorer* (A1: LeBron, Ronaldo), and *grinder* (A5: Kanté +
  gridiron TEs/FBs) from *connector* (A11: Draymond, Kelce, Henderson). Marcus
  Smart maps to A0 *attacking* mids/wingers (not a defensive grinder) and Draymond
  to A11 *attackers* (not a defensive anchor) — the data signature, not the
  reputation, drove both.
- **Low-rank wins** (specific pairings the model brings very close): Gobert~van
  Dijk r11, Tyreek Hill~Mbappé r32, Henderson~Draymond r44, Ben Simmons~Rúben
  Dias r56, Mahomes~De Bruyne r64, Ray Allen~Harry Kane r706.

