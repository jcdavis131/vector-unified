# Market / Cultural Footprint Signals — Unified Schema & Acquisition Plan

> Status: **acquisition scoped (Phase 1 = free tier, approved 2026-07-10)**. This doc
> defines the cross-sport market/cultural feature family, the join-key strategy, the
> sources per signal per sport, and the honesty constraints. It is the spec for the
> deferred task `p22` / `m22`.
>
> Related: `UNIFIED_ARCHITECTURE.md` §4a family table (Market/Pedigree, Honors/Clutch),
> §7 #6 (deferral note), §257 ("do not fake pitch bio/market features"), `SPEC.md` §7.

## 1. Why this family exists

Role archetypes align *what a player does on the field*. Market/cultural signals align
*what a player means off the field* — and crucially, **these signals are sport-agnostic
in their units**: a dollar is a dollar, a follower is a follower, an award-prestige tier
is a tier, whether the player is in the NBA, NFL, or a World Cup side. That makes
market/cultural the most natural **cross-sport objective anchor** — the one family where
a power forward and a strong safety and a box-to-box midfielder are measured on the
*same scale*, with no sport-specific re-encoding required.

This is the family the user asked for: "bring money and sponsorships, social-media
followings, etc. — what else can you think of?" — as the cross-sport comparison points
the role-only v0 geometry lacks.

## 2. The unified feature set (v0)

Seven features, all **sport-agnostic in unit**, all **log-transformed** (heavy-tailed) or
**tier-encoded**, all **masked** where unknown (the hoops `SALARY_LOG` masked-MSE pattern,
extended honestly per §257).

| Feature | Unit | Granularity | Source family | Sport-agnostic? |
|---|---|---|---|---|
| `SALARY_LOG` | log(on-field $ that season) | season-varying | Spotrac / BBREF / Capology | yes ($是全球可比) |
| `ENDORSE_LOG` | log(off-field $ / endorsements, annual) | season-varying (star-only) | Forbes highest-paid | yes |
| `MKT_VALUE_LOG` | log(market / transfer / trade value $) | season-varying | Transfermarkt (pitch) / Spotrac trade (NFL/NBA) | yes |
| `SOCIAL_REACH_LOG` | log(total followers, primary platform) | career-broadcast (star-only in Phase 1) | Wikipedia handles → Apify (Phase 2) | yes |
| `AWARD_PRESTIGE` | tier-weighted cumulative career award score [0..1] | career-broadcast | BBREF / PFR / Wikipedia (Wikidata P166) | yes (tier mapping) |
| `AWARD_RECENT` | AWARD_PRESTIGE earned in last 2 seasons (recency) | season-varying | same | yes |
| `PED_PICK_QUALITY` | draft pedigree z (already exists hoops+gridiron) | career-broadcast | BBREF draft / PFR draft | n/a for pitch (masked) |

**Already in the model (reuse, do not rebuild):** hoops has `SALARY_LOG`,
`PED_PICK_QUALITY`, `HON_ALL_NBA_VOTE_LAG` as masked-MSE heads (`train_mtnn.py` weights
salary 0.12, pedigree 0.08, honors 0.05). The v0 plan *harmonizes* these into the unified
schema above and *fills the gaps* for gridiron + pitch rather than re-inventing them.

### Award-prestige tier mapping (cross-sport)

Sport-specific awards map to a shared 0..1 prestige scale so a Ballon d'Or and an NBA MVP
carry the same anchor weight:

| Tier | Prestige | NBA | NFL | Soccer |
|---|---|---|---|---|
| T0 (elite individual) | 1.00 | MVP, DPOY, Finals MVP | MVP, DPOY, SB MVP | Ballon d'Or, The Best, WC Golden Ball |
| T1 (elite selection) | 0.60 | All-NBA 1st | All-Pro 1st | FIFPro World XI, UEFA TOTY |
| T2 (all-star) | 0.35 | All-Star | Pro Bowl | WC TOTY, continental TOTY |
| T3 (rookie/career) | 0.25 | ROY, MIP | OROY, DROY | Golden Boy, YPOTY |
| T4 (championship) | 0.45 | ring (season) | ring (season) | continental trophy (season) |

`AWARD_PRESTIGE = 1 - prod(1 - tier_weight)` over the player's career honors (soft-or so
multiple honors accumulate but saturate). Stored as a per-player career constant;
`AWARD_RECENT` recomputes it over only the last 2 seasons.

## 3. Granularity & join model

The unified model operates at **player-season** granularity. Market/cultural signals are
either **career-level** (slowly-varying: total reach, cumulative awards) or
**season-varying** (salary that year, market value that year, recent awards):

- **Career-broadcast** (`SOCIAL_REACH_LOG`, `AWARD_PRESTIGE`, `PED_PICK_QUALITY`): one
  value per player, copied to every season-row of that player. Masked once per player.
- **Season-varying** (`SALARY_LOG`, `ENDORSE_LOG`, `MKT_VALUE_LOG`, `AWARD_RECENT`):
  joined on `(player_key, season)`. Masked per season-row where the source has no value
  for that player-season.

Season-year alignment: NBA season `2023-24` → `2024`; NFL season `2023` → `2023`; pitch
context (e.g. `WC 2018`) → tournament year. A `season_year` int column is added to the
unified matrix to drive season-varying joins.

## 4. Join-key strategy (the main integration cost)

The three sports use **disjoint native IDs** (hoops=BBREF pid, gridiron=GSIS id,
pitch=StatsBomb player_id). External sources (Forbes, Transfermarkt, Spotrac, Wikipedia)
have **no shared ID** — joins are by **name + nationality + birthyear**.

| Sport | Has in meta | Needs (bio-augment) | Source for bio |
|---|---|---|---|
| Hoops | name, pid | nationality, birthyear | BBREF roster pages (already scraped for stats; extend) |
| Gridiron | name, gsis, pos, team | nationality, birthyear | PFR roster pages |
| Pitch | name, **nationality** (team=country), pos | birthyear | Transfermarkt / Wikipedia |

**Canonical key:** `name_norm` (lowercased, diacritic-stripped, suffix-stripped) +
`nationality` + `birthyear` (birthyear optional but strongly disambiguating). The
name-resolution index (`data/market/name_index.json`) emits one canonical key per native
player row. External pulls are keyed on the same canonical key and matched back to native
IDs with a confidence flag (exact / name+country / name-only-ambiguous).

Name disambiguation honesty: ambiguous name-only matches (e.g. multiple "James Johnson")
are **left masked**, not guessed. Coverage is reported as
`{exact, name+country, name_only_ambiguous_left_masked, unmatched}` per sport.

## 5. Sources per signal per sport (Phase 1 = free tier, approved)

| Signal | Hoops | Gridiron | Pitch | Coverage |
|---|---|---|---|---|
| `SALARY_LOG` | Spotrac/BBREF (full) | Spotrac (full) | Capology (top-5 leagues, elite) | NBA/NFL full; soccer elite-only (masked long tail) |
| `ENDORSE_LOG` | Forbes top-50 (stars) | Forbes (stars) | Forbes (stars) | star-only (~50-100 athletes cross-sport) — masked for the rest |
| `MKT_VALUE_LOG` | Spotrac trade value | Spotrac trade value | Transfermarkt (full) | pitch full; NBA/NFL from trade-value proxies |
| `SOCIAL_REACH_LOG` | Wikipedia handles (stars) | Wikipedia handles (stars) | Wikipedia handles (stars) | star-only in Phase 1; full-corpus in Phase 2 (Apify, deferred) |
| `AWARD_PRESTIGE` | BBREF honors (full, **already have `honors.json`**) | PFR honors (full) | Wikipedia/Wikidata P166 (internationals) | NBA full; NFL full; soccer international-level |
| `PED_PICK_QUALITY` | BBREF draft (**already have `pedigree.json`**) | PFR draft | n/a (masked) | hoops+gridiron full; pitch masked |

### Source access notes

- **Forbes highest-paid athletes**: Wikipedia maintains the full historical table
  (`Forbes_list_of_the_world's_highest-paid_athletes`) with `Total / Salary-winnings /
  Endorsements` columns per athlete per year. Free, structured, cross-sport. Covers the
  ~50-100 elite athletes per year — the cleanest cross-sport $ anchor.
- **Spotrac**: `spotrac.com/nba/contracts` and `/nfl/contracts` — server-rendered
  sortable tables, per-player per-season cap hit / salary. Free, no auth. Full active-roster
  coverage for NBA + NFL.
- **Transfermarkt**: per-player pages with market-value history (€). Free, comprehensive
  for soccer. Cloudflare-protected → use the `transfermarkt-scraper` PyPI package or the
  mobile endpoint; cache aggressively.
- **Wikipedia / Wikidata**: Wikidata SPARQL gives `P166` (award received) + `P2002`
  (Twitter/X handle) + `P2013` (Facebook) for any athlete with a Wikidata item, in a
  single cross-sport query — the highest-leverage free pull for awards + social-handle
  seeds. No anti-bot.
- **Capology**: soccer wages, top-5 European leagues + MLS, freemium (limited queries).
  Elite / top-league coverage only.
- **BBREF / PFR**: the hoops scraping infra already pulls from BBREF (it produced
  `pedigree.json`, `honors.json`). Extend the same pattern to PFR for gridiron
  salary/honors/draft/bio.

## 6. Acquisition phasing

- **Phase 1 — free tier (APPROVED, this work):** Spotrac NBA/NFL salary + Transfermarkt
  soccer market value + BBREF/PFR/Wikipedia honors + Forbes star endorsements + Wikipedia
  star social handles + bio-augmentation (BBREF/PFR nationality+birthyear). Zero cost.
  Full `SALARY_LOG` (NBA/NFL) + `MKT_VALUE_LOG` (pitch) + `AWARD_PRESTIGE` (all) coverage;
  `ENDORSE_LOG` + `SOCIAL_REACH_LOG` star-only (masked long tail, honestly).
- **Phase 2 — paid social reach (DEFERRED):** Apify bulk follower counts (~$0.002-0.005
  per profile; ~$40-100 for the full ~20k corpus, or free $5 credits for a star subset
  first). Only after Phase 1 shows the market/cultural family earns its keep in the
  unified geometry (Δ on G3/G4).

## 7. Model integration (v0)

1. **New `market` feature family** appended to each sport's feature matrix: the 7 features
   above + 7 mask columns. Aligned to the existing family ontology (`Market/Pedigree` +
   `Honors/Clutch` rows in `UNIFIED_ARCHITECTURE.md` §4a).
2. **Per-sport market tower** in each MTNN (mirrors hoops' existing salary/pedigree/honors
   heads) OR a **shared market/cultural tower** in the unified trunk (cleaner for
   cross-sport — since the units are already sport-agnostic, a shared tower is the natural
   choice and avoids per-sport re-encoding). v0 recommendation: **shared trunk tower**.
3. **Masked-MSE reconstruction head** on `z` (the hoops pattern): predict
   `SALARY_LOG`/`MKT_VALUE_LOG`/`AWARD_PRESTIGE`/`SOCIAL_REACH_LOG` from `z`, masked where
   unknown. This makes `z` carry market/cultural information *without faking* missing
   values — the head simply doesn't train on masked rows.
4. **Cross-sport contrastive benefit**: because the market/cultural features share units
   across sports, SupCon positives can now include "same market tier" anchors (a max
   player in all three sports pulls together), strengthening the cross-sport geometry
   beyond role-only.

## 8. Honesty constraints (binding)

- **Never fake** market/cultural values (§257). Missing → masked → head does not train on
  that row. The model carries market/cultural signal only for players who actually have it.
- **Coverage is uneven and documented**: NBA/NFL get full salary+honors; pitch gets full
  market-value + international awards but sparse salary; endorsements + social-reach are
  star-only in Phase 1. The UI / methods page must state which anchors are star-only.
- **Pitch comparisons that depend on market/cultural remain star-only** until coverage
  broadens — a non-elite pitch player's `z` carries role + market-value + (maybe) awards,
  but not salary/endorsements/social. State this honestly (extend the §256 "role-only,
  never physical/market" caveat to "role + market-value + awards; salary/endorse/social
  star-only").
- **Name-only-ambiguous matches are masked**, not guessed (§4).
- **No per-sport asset is overwritten** by the market/cultural pull — new data lands in
  `vector-unified/data/market/` and is joined at unified-matrix build time.

## 9. File map (acquisition outputs)

```
vector-unified/data/market/
  name_index.json            # canonical (name_norm, nationality, birthyear) per native row
  Forbes_highest_paid.json   # cross-sport star earnings (salary + endorsements split), by year
  spotrac_nba_salary.json    # per-player per-season NBA salary
  spotrac_nfl_salary.json    # per-player per-season NFL salary
  transfermarkt_value.json   # per-player per-season soccer market value (€)
  honors_<sport>.json        # award ledger per sport (BBREF/PFR/Wikidata)
  award_prestige.json        # computed AWARD_PRESTIGE / AWARD_RECENT per player
  bio_augment_<sport>.json   # nationality + birthyear (BBREF/PFR) for name resolution
  social_handles.json        # Wikipedia/Wikidata social handles (star seed; Phase 1)
  market_cultural_features.json  # the joined 7-feature + mask table, keyed by unified player_idx
  market_cultural_report.json    # coverage stats per sport per feature (exact/masked/unmatched)
vector-unified/pipeline/
  resolve_names.py           # build name_index.json from the 3 sports' meta + bio_augment
  pull_forbes.py             # Forbes/Wikipedia highest-paid table
  pull_spotrac.py            # NBA + NFL salary
  pull_transfermarkt.py      # soccer market value
  pull_honors_wikidata.py    # SPARQL awards + social handles (cross-sport, single query)
  build_market_cultural.py   # join all pulls -> market_cultural_features.json + report
```
