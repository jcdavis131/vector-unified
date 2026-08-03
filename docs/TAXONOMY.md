# Taxonomies

**Operator decisions taken 2026-08-03**, and the draft below is updated to match:

1. **Finer sectors.** 14 → **28**, shipped as `data/sector_map.json`. Applied coverage is
   **127 / 128 business-typed companies (99.2%)**, against the 39.8% the automatic P279*
   rollup reached. The one holdout is a Wikidata QID with no resolved label, left
   unassigned rather than guessed.
2. **Insurance and Crypto stay distinct** from Financial Services. Both are separate
   sectors below.
3. **Split the role axis** — long-term correctness over continuity, games and sites to be
   reconfigured. Section 3 now specifies a two-axis model: **role** (what a player does)
   and **trajectory** (how the career is going). See §3.1.



**Status: sectors SHIPPED (`data/sector_map.json` + `pipeline/apply_sector_map.py`); the
role/trajectory split is SPECIFIED but not yet migrated into `archetype_map.json`.** Every
layer is shown with its MEASURED coverage. The sector layer is hand-authored and marked as
such, because two automatic routes were measured and rejected first
(`build_company_sectors.py`: P279* rollup 39.8%, NAICS/ISIC/SIC/GICS 0/128).

The estate already has a precedent for a declared human taxonomy —
`data/archetype_map.json` calls itself a *"Human-authored anchor for cross-sport
alignment"*. This follows the same rule: **the map is data, version-controlled, and
changed only in a commit**, so a number can never move because someone regrouped
categories in a notebook after seeing the result.

---

## 0. The five layers, and what each is for

| layer | unit | source | status |
|---|---|---|---|
| **Entity** | athlete / org / company / location | built | live |
| **Relation** | which edges exist and what they mean | built | live |
| **Role (cross-sport)** | A0–A5, A10, A11 over athletes | `archetype_map.json` | live; split from trajectory SPECIFIED, not migrated |
| **Role (native)** | K=8 per sport | per-repo model | live |
| **Trajectory** | T0–T3 over careers | was A6–A9 | specified, not migrated |
| **Sector** | company → industry group | `data/sector_map.json` | **live, 127/128** |

Roles answer *what an athlete does*. Sectors answer *what a brand sells*. The business
question dumbmodels.com is built on is the join between them. The sector layer now exists
at 127/128 coverage, so that join is TESTABLE for the first time — Q8 previously came back
UNDERPOWERED because the only available split, "has a sponsor", is 96% true in US sport.

---

## 1. Entity taxonomy

```
Athlete      person, identified by normalised name
             6,226 in the graph; 99.98% resolve to at least one org

Org          team x season   (mirrors vector-equities' ticker x fiscal_year)
             1,440 orgs, 22,322 athlete->org edges, 0 unresolved
             187 enriched with Wikidata attrs; 163 distinct locations

Company      a Wikidata entity typed P31/P279* -> business (Q4830453) or company (Q783794)
             196 organization-typed, of which 128 are BUSINESS-typed
             NOT time-sliced: Wikidata gives the CURRENT holder with no interval

Location     org country 100%, city 66.3%, venue capacity 86.1% (of enriched)
```

**The typing rule is load-bearing and was got wrong once.** Filtering company targets to
`organization` admitted families (Maloof, Rooney), football federations, geographic
entities and one club appearing as its own company — 68 of 196, carrying 797 athletes and
inflating the headline reach from 71.5% to 84.3%. `is_business` is now carried per
company rather than applied as a filter, because the typing has false negatives too
(GEHA is a real health insurer and types as neither).

---

## 2. Relation taxonomy

```
athlete --played_for--> org            22,322 edges, 99.98% resolved
org     --located_in--> location       country 100%
org     --home_venue--> venue          86.6% of enriched
company --sponsors-----> org           14.4% of enriched orgs   (P859 on the CLUB)
company --owns---------> org           40.1% typed as organizations (P127)
company --names--------> venue -> org  35.8% (P115/P138 named-after)
```

Union of the three company relations: **142 / 187 orgs (75.9%)**, 196 organizations, of
which 128 businesses → **4,451 / 6,226 athletes = 71.5%**.

By sport, business-typed: **gridiron 82.6%, hoops 95.7%, pitch 24.7%.**

The three company relations are **complementary, not redundant** — US venues carry naming
rights where European clubs carry shirt sponsors. Any one of them alone is a US-or-Europe
signal; the union is what makes it cross-sport.

**Do not model `athlete --endorsed_by--> company`.** Wikidata P859 on athletes is
13/5,821 = 0.22%, gridiron literally 0. Every company edge here is INSTITUTIONAL — who
owns the club, who named the stadium, who is on the shirt. The sellable claim is *brand
exposure through the employer*, never *this athlete has a sponsorship deal*.

---

## 3. Role taxonomy — cross-sport

Twelve archetypes as originally defined, six in v0 scope; §3.1 splits them onto two axes. Scope is not arbitrary: an archetype qualifies only
if it has native-cluster members in **at least two sports**, so the contrastive anchor
cannot be trained on a role only one sport has.

| id | label | v0 |
|---|---|---|
| A0 | Offensive engine / primary initiator | ✅ |
| A1 | Volume producer / scoring load | ✅ |
| A2 | Explosive perimeter creator | ✅ |
| A3 | Defensive anchor / last line | ✅ |
| A4 | Defensive disruptor / ball-hawk | deferred — pitch-only members |
| A5 | Connector / two-way grinder | ✅ |
| A6 | High-pedigree, under-delivering | deferred |
| A7 | Low-pedigree, over-delivering | deferred |
| A8 | Breakout / riser | deferred |
| A9 | Declining veteran | deferred |
| A10 | Elite two-way | deferred |
| A11 | Floor-raising role player | ✅ |

### 3.1 The split — DECIDED, two axes

A6–A9 were never the same kind of category as the rest. A0–A5, A10 and A11 describe
*what a player does*; A6–A9 describe *how a career is going*. One axis cannot express
"volume producer who is also a riser", and every player has both properties at once.

**Axis 1 — ROLE (what the player does), 8 values.** Unchanged IDs, so existing artifacts
and the shipped game vocabulary keep resolving:

```
A0 Offensive engine / primary initiator     A3  Defensive anchor / last line
A1 Volume producer / scoring load           A5  Connector / two-way grinder
A2 Explosive perimeter creator              A10 Elite two-way
A4 Defensive disruptor / ball-hawk          A11 Floor-raising role player
```

A4 stays deferred from the v0 contrastive anchor (pitch-only native members), and A10
likewise until it has members in two sports — deferral is about *training scope*, not
about whether the value exists.

**Axis 2 — TRAJECTORY (how the career is going), 4 values**, promoted from A6–A9:

```
T0 High-pedigree, under-delivering   (was A6)
T1 Low-pedigree, over-delivering     (was A7)
T2 Breakout / riser                  (was A8)
T3 Declining veteran                 (was A9)
```

**Why renumber rather than keep A6–A9.** Leaving them in the A-space invites exactly the
mistake the split exists to prevent: a downstream join that treats A8 as mutually
exclusive with A1. A distinct prefix makes the two axes obviously non-exclusive at a
glance, and a stale consumer that still asks for "A8" gets a clean key error instead of
silently reading a role.

**Trajectory is a career-level label; role is a season-level one.** That difference is
load-bearing: a player has one trajectory over a span and a role per season. Storing them
in one field forced a single grain onto both.

**This aligns with what vector-hoops already ships.** Its career classes — stable /
migrator / drifter / reinvention / late-bloom — are a trajectory axis under a different
name, computed per career from per-season archetypes. The unified trajectory axis should
be reconciled against those five rather than invented in parallel; that reconciliation is
open work, not done here.

**Migration cost, stated plainly:** any consumer keying on A6–A9 breaks. The operator has
accepted that ("do what makes the most sense long-term and we will reconfigure the games
and sites accordingly"). `archetype_map.json` must carry both axes before anything is
regenerated, and `archetypes_in_scope_v0` needs splitting into per-axis scope lists.

### Family ontology (the bridge that makes cross-sport possible)

```
Volume / Usage          hoops: volume, playmaking   gridiron: usage, opportunity   pitch: attacking
Efficiency              hoops: efficiency, shotmix  gridiron: form, ngs            pitch: attacking, passing_control
Defense / Disruption    hoops: defense, rebounding  gridiron: defense, pfr_adv     pitch: defending_duel
Playmaking / Initiation hoops: playmaking           gridiron: context, opportunity pitch: (see map)
```

---

## 4. Role taxonomy — native, per model

Each sport fits K=8 natively, then maps to the cross layer. The native names are
descriptive of centroid sigmas, not hand-chosen.

**vector-hoops** (`assets/archetypes_time.json`, global K=8, era-z 14-d, seed 7):

| n | native name | → cross |
|---|---|---|
| 0 | Offensive Glass + Rim Protection | A3 |
| 1 | Offensive Glass + Rim Protection + Scoring Efficiency | A11 |
| 2 | Playmaking + Ball Pressure | A2 |
| 3 | Scoring Volume + Free-Throw Shooting | A3 |
| 4 | Perimeter Shooting + Ball Pressure | A1 |
| 5 | Perimeter Shooting + Free-Throw Shooting | A2 |
| 6 | Defensive Glass + Rim Protection | A0 |
| 7 | Perimeter Shooting + Free-Throw Shooting + Ball Pressure | A1 |

⚠ **These names do not match `docs/ARCHETYPE_ERA_RESEARCH.md`**, which lists a different
set ("Three-Point Accuracy (Low Turnovers)" etc.). That doc was committed 2026-07-06 and
the artifact has been regenerated by six commits since. Its §4 career-dynamics table is
also stale — it reports stable 44.4% / migrator 24.1% / reinvention 9.4% where the
artifact says **58.9% / 8.3% / 21.6%**, and an era transition rate of 0.369→0.397 where
the artifact says 0.141→0.162 and PEAKS in the 2010s rather than rising to the 2020s.
**Do not take hoops archetype names or career-class shares from that document.**

**gridiron** native→cross: `0→A1, 1→A5, 2→A2, 3→A1, 4→A2, 5→A0, 6→A11, 7→A11`
**pitch** native→cross: `0→A5, 1→A1, 2→A11, 3→A3, 4→A3, 5→A2, 6→A0, 7→A1`

**vector-equities** is the structural template rather than a role model: `ticker x
fiscal_year` is what `org = team x season` was copied from. It has no athlete layer and
no archetype layer; its contribution to the unified model is the entity SHAPE.

---

## 5. Sector taxonomy — SHIPPED, hand-authored

### Why hand-authored

Measured first, both rejected:

```
Wikidata P279* rollup onto declared anchors   51 / 128 = 39.8%
  and it misses the largest companies: "airline" is not P279* under "transport",
  "software company" is not under "information technology", "cryptocurrency
  exchange" is not under "financial services"
NAICS / ISIC / SIC / GICS                     0 / 128 = 0.0%
```

There is no automatic source. A declared map is the honest remaining option.

### Sectors (28) — SHIPPED as `data/sector_map.json`

Applied: **127 / 128 business-typed companies (99.2%)**, 28 sectors declared and all 28
used. Reported against 128, never against the 98 that carry an industry value.

| sector | companies | athletes |
|---|---|---|
| Financial Services (non-bank) | 22 | 1,898 |
| Automotive | 10 | 1,002 |
| Aviation & Air Transport | 7 | 800 |
| Insurance | 8 | 768 |
| Banking | 9 | 726 |
| Sports & Entertainment Holdings | 9 | 591 |
| Restaurants & Food Service | 3 | 576 |
| Software & IT Services | 6 | 536 |
| Telecommunications | 6 | 516 |
| Mortgage & Lending | 2 | 507 |
| Logistics & Delivery | 2 | 502 |
| Crypto & Digital Assets | 3 | 462 |
| Retail & Consumer Goods | 11 | 431 |
| Personal Care & Cosmetics | 2 | 316 |
| Real Estate & Construction | 2 | 267 |
| Industrial, Chemicals & Manufacturing | 4 | 255 |
| Rail & Ground Transport | 2 | 250 |
| Staffing & HR Services | 4 | 234 |
| Gambling & Betting | 5 | 233 |
| Energy, Petroleum & Utilities | 5 | 149 |
| Investment & Asset Management | 4 | 144 |
| Hospitality & Leisure | 3 | 138 |
| Food & Beverage | 5 | 115 |
| Computer Hardware & Electronics | 4 | 44 |
| Media & Broadcasting | 4 | 40 |
| Video Games & Interactive | 2 | 21 |
| Cybersecurity | 1 | 21 |
| Internet & Satellite Connectivity | 1 | 5 |

**Do not sum that column.** rows 11,586 vs 4,451 distinct athletes = **2.6x inflation**.
Excluding `sports_holdings` (a club's own holding entity is not a sponsor): **4,425
distinct = 71.07% of all athletes**. That second figure is the brand-exposure number.

Fine granularity costs power: six sectors carry one or two companies. Any test split on
sector must report per-cell n rather than assuming 28 categories grants it.

### Original 14-sector proposal (superseded, kept for the reasoning)

Chosen as the axis a sponsorship conversation actually happens along — a rights-holder
sells to categories, and category exclusivity is how deals are priced. Ordered by
measured athlete reach where known.

| sector | covers (observed values) |
|---|---|
| **Financial Services** | financial services, financial sector, economics of banking, financial service activities, credit union, mortgage, investment company |
| **Insurance** | insurance, insurance industry, health insurance |
| **Automotive & Mobility** | automotive industry, motor oil, tyres |
| **Telecom & Connectivity** | telecommunications, mobile phone industry, postal and telecommunications services |
| **Technology & Software** | software company, IT services, consumer electronics industry |
| **Crypto & Digital Assets** | cryptocurrency industry, bitcoin exchange, cryptocurrency exchange |
| **Travel & Transport** | air transport, airline, rail transport, logistics |
| **Food & Beverage** | food industry, gastronomy, brewing, soft drinks |
| **Retail & Consumer Goods** | retail, textile industry, cosmetics industry, multi-level marketing |
| **Energy & Utilities** | energy industry, electricity supply company, oil and gas |
| **Health & Pharma** | pharmaceutical industry, healthcare services |
| **Media & Entertainment** | mass media, broadcasting, gaming |
| **Gambling & Betting** | gambling, sports betting |
| **Real Estate & Construction** | real property, construction, property development |

**Insurance and Crypto are deliberately separate from Financial Services**, against the
usual GICS grouping. Reason: in sponsorship they are different buyers with different
regulatory exposure and different category-exclusivity fights, and both are large enough
here to stand alone — insurance is 4 companies / 584 athletes, crypto is Crypto.com alone
at 419 athletes. Merging them into one finance bucket would hide the single largest
company in the graph inside a category that already dominates.

### What this map does NOT solve

**30 of 128 business-typed companies have no industry value at all** — including Rocket
Companies, United Wholesale Mortgage, American Airlines, Kaseya, Paycor, 777 Partners.
The sector map is a grouping of values that exist; it cannot categorise a company with no
value. Those 30 need either a manual assignment (they are recognisable brands — the list
is short enough to do by hand once) or a second source. **Until they are assigned, any
sector coverage figure must be reported against 128, not against 98.**

**Sector rows will not be disjoint.** AT&T is telecom and media. The existing guard
applies: report the row sum next to the deduplicated union and the inflation factor
(currently 1.58x on the partial rollup, 2.95x on raw industry rows).

---

## 6. What to build next, in order

1. **Assign the 30 uncategorised businesses by hand** into the 14 sectors. Short, and it
   takes coverage from 98/128 to 128/128 — the single highest-value hour here.
2. **Commit the map as `data/sector_map.json`**, mirroring `archetype_map.json`'s shape
   and its "human-authored anchor" note.
3. **Re-run Q8 with sector as the split.** It was UNDERPOWERED because "has a sponsor" is
   96% true in US sport. "Which sector" has real variance and may be testable — that is a
   measurement, not a prediction.
4. **Split trajectory out of the role axis** (A6–A9), so a player can carry both a role
   and a career-stage label instead of the taxonomy forcing a choice.

## 7. Open questions for the operator

1. **Is 14 sectors the right granularity**, or should it collapse to ~8 for a first
   product? More sectors = better category-exclusivity fidelity, worse per-cell power.
2. **Should Insurance and Crypto stay separate from Financial Services?** Argued above,
   but it is a product call, not a data one.
3. **A6–A9 split** — worth doing, or keep the 12-way axis for continuity with the shipped
   game vocabulary?
