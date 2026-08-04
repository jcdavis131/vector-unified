# Tennis + golf as vector-X members — feasibility, measured before building

**Written 2026-08-03.** Operator direction: add tennis and golf MTNNs, on the reasoning
that individual sports bring more locations and sponsors — tournament sponsors, and
*individual athlete* sponsors, which team sports do not have at the athlete level.

The reasoning about the world is right. **The reasoning about the data is not, and the
measurement is cheap enough that it should happen before a line of pipeline code.** What
follows is four queries and five HTTP checks, run today.

---

## 1. The sponsor rationale does not survive contact with Wikidata

The whole point of adding individual sports was athlete-level sponsorship — the thing
7.10 proved the company layer does not have, because team-sport sponsorship attaches to the
club and every teammate inherits an identical vector.

Measured, Wikidata P859 (sponsor) on humans by `P641` sport:

    sport             athletes  with P859     pct
    tennis               15,896          6    0.04%
    golf                  6,157          3    0.05%
    basketball          262,262         12    0.00%
    am_football          49,756          1    0.00%
    assoc_football      395,316         31    0.01%

Tennis and golf are **an order of magnitude better in rate and useless in absolute terms.**
Six athletes. Three athletes. Federer has a Rolex deal and Wikidata does not record it —
P859 is simply not a populated property for athletes in any sport.

Second try, the tournament level, since a title sponsor is a naming fact and those usually
*are* recorded:

    class                    items  sponsor  named_after  location
    tennis_tournament       56,305      164           47     6,786
    golf_tournament          1,576        1            9         7

**0.29% of tennis tournaments and 1 of 1,576 golf tournaments carry a sponsor edge.**

> **CORRECTED 2026-08-03 — this measured the wrong source, and the conclusion drawn from it
> was wrong.** 0.29% is **Wikidata P859 coverage**, and P859 is a sparse editorial property.
> It is not sponsor coverage. The sponsor of a tennis tournament is *its name*, and the name
> was already sitting in the tennis-data.co.uk xlsx this repo had downloaded — no new fetch,
> no key, no API. Re-measured in `pipeline/build_tennis_sponsors.py`:
>
> | signal | count | rate |
> |---|---|---|
> | tournament × location pairs | 307 | — |
> | pairs with a candidate sponsor token | 162 | **52.8%** |
> | locations with a rename across years | 60 of 169 | **35.5%** |
> | CONFIRMED (company-name match **and** rename) | 36 tokens | — |
>
> The renames are dated corporate actions: `BNP Paribas Fortis → European Open` at Antwerp,
> `BB&T → Atlanta Open` (BB&T merged into Truist, 2019), `Sony Ericsson → Miami Open`,
> `Family Circle Cup → Volvo Car Open` at Charleston. Strongest single piece of evidence is
> **AEGON → Viking appearing at Birmingham and Eastbourne in the same year** — one rebrand
> surfacing simultaneously at two unrelated venues, which a tokeniser cannot manufacture.
>
> Same defect class this phase has been chasing throughout: *a real value answering a
> different question than the one it appears to answer.* The 0.29% figure was accurate and
> the inference from it was not. Known false-positive class (toponyms outside the `location`
> column) is documented in the artifact rather than filtered, because a hand-written
> exclusion list would be tuned against the cases it judges.
Locations are better for tennis (12%) and effectively absent for golf.

So both halves of the stated rationale fail on the source the estate actually has. This
does not make the expansion wrong. It means **the sponsor payoff must not be used to
justify it**, because it will not arrive, and 7.9 already set the precedent: a source that
does not exist is recorded as absent rather than approximated.

---

## 2. There is a better rationale, and it comes out of this phase's own failures

Two things individual sports give that the current three cannot, neither of which was in
the original argument:

### 2a. No team confound

The pitch P0/P1 axis (7.32) had to control for team strength because it dominated:

    corr(residual, leave-one-out teammate-mean delivery)   +0.2626
    corr(age, delivery)                                    -0.1671

The confound was **larger than the signal it contaminated.** Every team-sport axis in this
estate carries some version of that problem — a player's per-90 rates, PPR, and per-100
composite all move with the quality of the people around them.

**Tennis and golf have no team.** A stroke average or a service-hold rate is the athlete's
own. That is a materially cleaner delivery measure than anything in hoops, gridiron, or
pitch, and it is worth more to the unified model than a sponsor edge would have been.

### 2b. A ranking is a real market-like prior — the first one outside the drafts

7.9 established that pitch has **no free market prior** and that the age curve is a
*developmental* prior, a different construct, hence `P0/P1` rather than `T0/T1`.

Tennis and golf both publish a **ranking** — ATP/WTA and OWGR. A ranking is not a draft
slot, but it is far closer to one than age is: it is a public, pre-existing, market-facing
valuation of a competitor before the performance being measured. That would give the estate
its **first expectation prior outside the two drafts**, and the first chance to test whether
the T-axis construct generalises to a non-draft valuation.

This is the argument that survives measurement. It should be the stated one.

---

## 3. The data does not exist yet for either, and tennis is much closer

### Tennis — viable, with work

| source | state |
|---|---|
| `JeffSackmann/tennis_atp`, `tennis_wta` — the canonical free datasets | **GONE.** `api.github.com/repos/JeffSackmann/tennis_atp` → 404 on both `main` and `master`; `api.github.com/users/JeffSackmann` reports **`public_repos=1`**. The standard open tennis corpus is no longer published. |
| `tennis-data.co.uk` | **Alive.** Index 200 / 68,710 bytes; per-season archives behind a filename I have not yet pinned (`2024.zip` returned HTTP 300, multiple choices). Carries match results, rankings and closing odds. |
| GitHub search for replacements | Small forks and scrapers, mostly unlicensed. `serve-and-volley/atp-world-tour-tennis-data` (219 stars) is a **scraper**, not a dump, and carries no license. |

**Verdict: buildable, but the foundation has to be re-established first**, and the honest
cost is a polite scraper against `tennis-data.co.uk` plus a licence check — not a `git
clone`. Nobody should plan around Sackmann; it is worth writing down loudly because every
tennis-modelling guide on the internet still points there.

### Golf — no performance source found

OWGR (`owgr.com`, HTTP 200) gives the **ranking** — the prior. Nothing free and open was
found for the **delivery** side: strokes gained, round scoring, or shot-level data. The
GitHub search for a PGA dataset returned nothing usable.

**Verdict: blocked on data, not on modelling.** A golf MTNN needs a per-round performance
matrix and there is no free one identified. Recorded as absent rather than approximated —
same disposition as football transfer values in 7.9.

---

## 4. Recommendation

1. **Do not justify the expansion on sponsors.** Measured above; it will not pay.
2. **Tennis first, golf second**, and the gap between them is data availability, not value.
3. **Before any pipeline code:** pin the `tennis-data.co.uk` archive URLs, check the
   licence, and measure per-season row counts and field coverage — the same
   coverage-before-modelling order that 7.10–7.13 established by getting it wrong first.
4. **Frame the tennis axis as `R0/R1` (ranking), not `T0/T1`,** unless and until the
   construct is shown to behave like a draft slot. Three prefixes for three different priors
   is the whole point of `P0/P1` existing.
5. **The equities/business MTNN is a separate thread** and is not blocked by any of this —
   it already has SEC fundamentals; what it lacks is the analyst-consensus join, which is
   its own expectation-prior problem and closer to solvable than golf's.

---

## 5. MEASURED 2026-08-03 — tennis is the best-covered member of the estate

`acquire_tennis.py` fetched all 28 season files (ATP + WTA, 2013-2026) from
tennis-data.co.uk, respecting a robots.txt that disallows only `/stuff/` and
`/2000/`-`/2005/`, and measured them:

    seasons                                   28 / 28
    matches                                   67,081   (ATP 34,591 / WTA 32,490)
    ranking prior WRank/LRank/WPts/LPts       67,080   100.0%
    closing odds  AvgW/AvgL                   67,012    99.9%
    distinct locations                            169
    surfaces                Hard 39,391 / Clay 19,799 / Grass 7,891
    Location / Tournament / Surface / Round   present in all 28 seasons
    Series                                    present in 14 (ATP only)

Set against the rest of the estate, this is not close:

| member | expectation prior | coverage |
|---|---|---|
| hoops | draft slot | market prior, full |
| gridiron | draft slot | market prior, full |
| pitch | age curve | developmental only; axis scorable on **43.8%** |
| market layer | — | **2.9%** of athletes carry a nonzero value |
| company sponsor reach | — | **14.0%** |
| **tennis** | **ranking AND closing odds** | **100.0% / 99.9%** |

**67,081 matches is over three times the entire current unified corpus** (20,719
player-seasons), and tennis is the only member with *two* market-like priors where no other
has one. `AvgW`/`AvgL` are closing odds — literally what the market priced this specific
match at, which is a closer analogue of a draft slot than a season-level ranking is, and far
closer than pitch's age curve.

The 169 locations also deliver the operator's location angle — from the data file, not from
Wikidata, where the same check found location on 12% of tennis tournaments and 7 golf
tournaments.

**Nothing has been modelled.** This is the coverage number that decides whether a tower is
worth building, obtained before building it — the order 7.10-7.13 established by getting it
wrong first.

---

## 6. The ranking prior works — and the circularity had to be beaten first

`probe_tennis_expectation.py`. A ranking is **computed from results**, so correlating
in-season rank against that season's win rate is close to correlating results with
themselves. A draft slot cannot be contaminated that way. The prior is therefore the
player's **median rank in season t-1** against delivery in season **t**.

    tour        n  lag vs win  lag vs sets  SAME vs win   shrink  verdict
    atp      1253     +0.6088      +0.6277      +0.7527    0.191  USABLE
    wta      1232     +0.4962      +0.5044      +0.6750    0.265  USABLE

**The sanity contrast is the point.** Same-season rank correlates at +0.753 / +0.675;
lagging shrinks it by **19.1% / 26.5%**. The lag removed something real, which is the
evidence that it was needed. Had the two numbers been equal, neither would have been
trustworthy.

Against the matched draft axes — hoops **+0.4934**, gridiron **+0.3950** — the lagged ATP
prior at **+0.6088** is the strongest expectation signal in the estate.

**Closing odds are deliberately not the prior.** They are genuinely pre-match, but a closing
price is a near-optimal forecast of the very outcome being scored, so `corr(odds, result)`
is high by construction and says nothing about over- or under-performance. Odds belong here
as the **baseline a model must beat**, not as the expectation an axis is built against.

**Naming, and the caveat that forces it.** +0.6088 is still partly *persistence* — a good
player last year tends to be good this year, and a ranking encodes exactly that. A draft
slot has no such property: it is a one-time pre-career assessment that never updates. So the
tennis axis measures over/under-performance against **your own recent standing**, not
against what the market paid to acquire you. That is a third construct, and it gets a third
prefix:

    T0 / T1   hoops, gridiron   draft slot        one-time pre-career market valuation
    P0 / P1   pitch             age curve         developmental prior
    R0 / R1   tennis            lagged ranking    self-referential recent standing

**None of the three may be compared against the others.** That is now the estate's standing
rule, and it exists because 7.7b's cross-sport finding reversed twice under exactly this
kind of mismatch.
