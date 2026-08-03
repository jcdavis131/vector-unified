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
