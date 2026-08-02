# Value-signal census — what this estate can and cannot ask

**Measured 2026-08-02.** Every number here was produced by running something, not estimated.
Written so the next attempt does not re-probe four sources that have already been measured
and found insufficient.

## The problem this records

`query_graph.py` returns rigorous answers and has answered three questions "no". The nulls
are real — a positive control (`WIN_PCT` vs `NET_RATING`, n=892) returns r=+0.968 through
the same machinery that returns +0.001 on the questions. The limit is not the method.

The limit is that **business questions need a value signal on the athlete side, and every
free one is superstar-biased and tiny.** Four were measured:

| source | coverage | verdict |
|---|---|---|
| Wikidata `P859` sponsor | 13 / 5,821 athletes (**0.22%**), gridiron literally 0 | unusable |
| Wikipedia bios, brand mentions | 2 of 429 bios, **0** endorsement cues | unusable |
| Wikidata `P166` awards | hoops: 38 winners, 22 matched | MVP-tier only |
| Wikipedia pageviews | 429 players (**7.4%**) | same curated subset as bios |

Against two that work:

| source | coverage | verdict |
|---|---|---|
| `vector-hoops/.../salary_market.json` | **11,408 player-seasons**, `SALARY_TEAM_PCT` as a share | usable (hoops) |
| `acquire_spotrac.py` -> `market_cultural.json` | gridiron salary **93.27%**, 1,509 athletes, 6,171 records | usable (gridiron) |

**Correction made while writing this file.** The first draft said hoops was the only usable
value signal and listed Spotrac as future work. `acquire_spotrac.py` had already been
written AND run: NFL cap hit, 93.27% of gridiron rows. Checking one more thing before
publishing is the only reason this table is right, which is the same failure mode the rest
of this estate keeps catching — a confident claim that nobody re-measured.

The two that work exist because someone did the acquisition. That is the whole difference —
not that basketball and football data are more available in principle. pitch salary is
0.58%, and that is the actual gap.

## What this rules out, concretely

The obvious follow-up to Q3's null was to re-run it in an **uncapped** league. The NBA null
is consistent with the salary cap binding, and pitch has no cap, so pitch should show the
effect if the mechanism is real. That is a clean design: same question, capped vs uncapped,
with the NBA as its own control.

**It cannot be run.** Pitch has 233 orgs with capacity and 1,833 athletes with archetypes,
and no value signal at any usable coverage. The comparison is blocked on acquisition, not
on modelling.

## What the graph CAN still carry

- role (12 cross-sport archetypes, G4 0.978)
- place (163 locations; org country 100%, city 66.3%)
- scale (venue capacity, 86.1% of enriched orgs)
- outcome (hoops only: `NET_RATING`, `WIN_PCT`, 892 team-seasons)
- value (hoops only: salary share, 11,408 player-seasons)

Anything spanning sports is limited to role, place and scale. Anything about value or
outcome is a **basketball** study wearing a cross-sport graph.

## What would unblock it

Acquisition, in rough order of effort:

1. ~~**gridiron**~~ — **already done**: `acquire_spotrac.py`, 93.27% coverage. See above.
2. **pitch** — Transfermarkt market values. Not free-tier friendly and terms-restricted;
   treat licensing as a real question, not an implementation detail.
3. **cross-sport** — Forbes highest-paid athletes is already acquired
   (`forbes_earnings.json`) and IS cross-sport with an explicit salary/endorsement split.
   ~10 athletes/year since 2012, so it is a superstar sample by construction — usable for a
   question ABOUT superstars, not as a general value signal.

## Q3 replicated on gridiron — two capped leagues, two nulls

Q3 (do larger-market clubs concentrate pay in fewer players?) was null on hoops. With
gridiron salary available it is a replication rather than a single result:

| league | cap | n team-seasons | HHI small -> large venue | r | shuffle p95 | verdict |
|---|---|---|---|---|---|---|
| NBA | soft + luxury tax | 744 | 0.1493 -> 0.1509 | +0.001 | 0.082 | no finding |
| NFL | **hard** | 295 | 0.1973 -> 0.2091 | +0.050 | 0.098 | no finding |

Two sports, two independent salary sources, same answer. Market size does not predict how
concentrated a roster's pay is in either league. The cap-binding reading survives a
replication it could have failed, which is worth more than the original null was.

Still association-only, and still bounded by venue capacity being a proxy for market size
rather than a measure of it. The uncapped test remains unrun, because pitch salary is 0.58%.

## The standing rule this estate keeps proving

Measure coverage before building the model. It killed the brand entity at 0.22% before an
encoder existed for it, justified the org entity at 99.98%, and is why the three nulls above
are trustworthy instead of embarrassing. A registry and a coverage number cost an hour; an
encoder trained on a 0.22% edge costs a week and produces a confident wrong answer.
