# Cultural Text Signals — Schema (Wikipedia-first MVP)

> Status: live · 2026-07-11 · auto-mode board for Wikipedia → text embeddings.
> Parent: `MARKET_CULTURAL_SCHEMA.md` · honesty: UNIFIED_ARCHITECTURE §257.
> **Reddit is deferred** (ToS/API cost, noise, entity-link risk) until Wikipedia earns keep.

## Goal

Supplement the unified trunk with **cultural / narrative** signal that salary and awards
miss: how a player is described in public encyclopedia text. Same contract as market —
mask missing, never fake.

## Sources (v0)

| Source | Use | Status |
|---|---|---|
| English Wikipedia lead extracts | Free-text bio → frozen text embedding `t_p` | **THIS MVP** |
| Wikipedia page length / pageviews (optional) | Scalar fame proxies | bonus if cheap |
| Wikidata sitelinks | Resolve ambiguous names | used in acquire |
| Reddit mention heat | `SOCIAL_HEAT` scalars | **DEFERRED** |

## Features per player (career-stable; broadcast to seasons)

| Field | Type | Notes |
|---|---|---|
| `wiki_title` | str \| null | Resolved enwiki title; null → fully masked |
| `extract` | str | Lead section plain text (capped ~1200 chars at embed time) |
| `t_p` | float[384] | L2-normalized MiniLM embedding (`all-MiniLM-L6-v2`) |
| `extract_chars` | int | Fame/verbosity proxy |
| `m_text` | 0/1 | 1 iff resolve+extract+embed succeeded |

Join key: `(sport, name_norm)` → broadcast to all player-seasons in `unified.json` order.
Ambiguous Wikipedia hits → mask (`m_text=0`).

## Model wiring

- Shared `text_proj: Linear(d_emb, 384)` on trunk `z`.
- Loss: masked `(1 - cosine(text_proj(z), t_p))` (or MSE on L2 vectors).
- Warm-start from `unified_market.pt` or `unified_best.pt`; save **`unified_cultural.pt`**
  (never overwrite shipped `unified_best.pt`).
- Expect G2 sport-acc to worsen (bios leak sport names) — documented, not a ship gate.

## Coverage honesty

- Long-tail / obscure players often lack enwiki pages → masked.
- Star / award / Forbes athletes preferentially covered first; full unique pass resumes from cache.
- Pitch/hoops/gridiron disambiguation via sport keyword in search (`basketball player`,
  `American football`, `footballer`).

## v0 results (2026-07-11)

- Priority Forbes/awards ∩ corpus: **47/47** wiki leads resolved.
- Joined seasons with `m_text=1`: **454 / 20,721 (2.2%)**.
- Checkpoint: `pipeline/data/unified_cultural.pt` (warm-start `unified_market.pt`, `--d-sport-tok 0`).
- Probe: mean cosine(text_proj(z), t_p) = **0.775** on labeled rows.
- Eval: **G1 PASS**, **G3 PASS** (sil 0.682), G2 sport-acc **0.842** (worse than Stage 1 — expected;
  bios leak sport identity). Shipped `unified_best.pt` / `unified.json` untouched.
- Full-corpus wiki acquire continues via resume cache (MediaWiki 429 → backoff).

## Reddit (explicit non-action)

Deferred: ToS/API cost, entity-link noise, and weak long-tail coverage. Revisit only as
aggregate `SOCIAL_HEAT` scalars after Wikipedia coverage is broader and earns keep on G3/G4.

## File map

```
data/market_cultural/
  wikipedia_bios.json       # unique-player resolve + extracts (+ cache progress)
  cultural_text_embeds.npz  # unique-player t_p matrix
  cultural_text_matrix.npz  # 20,721×384 aligned T + m_text
  cultural_text.json        # coverage metadata
  cultural_text_probe.json  # face-validity
docs/CULTURAL_TEXT_SCHEMA.md  # this file
```

## Non-goals (v0)

- Reddit scraping / post embeddings
- Fine-tuning the text encoder
- Overwriting per-sport game assets or shipped `unified.json`
