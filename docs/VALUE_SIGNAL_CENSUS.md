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

## AMENDED 2026-08-03 — the COMPANY edge exists, on the org side

Everything above is about a value signal **on the athlete**, and it stands. But it was
read as "there is no company edge", and that is a claim this file never tested. P859 was
measured on ATHLETES only. Measured on ORGS, by `pipeline/probe_company_edges.py`, over
the same 187 enriched orgs:

| relation | orgs with edge | typed as an ORGANIZATION | distinct targets |
|---|---|---|---|
| `P859` sponsor (on the club) | 27 (14.4%) | 27 (**14.4%**) | 64 |
| `P127` owner | 131 (70.1%) | 75 (**40.1%**) | 157 |
| `P115/P138` venue named-after | 85 (45.5%) | 67 (**35.8%**) | 93 |

Individually moderate; **complementary in practice**, because they fail in different
places — US venues carry naming rights (AT&T, Bank of America, Ford Motor Company) where
European clubs carry shirt sponsors (Adidas AG, Accor, CMA CGM) instead. The union:

    orgs with >= 1 ORGANIZATION-typed company edge : 142 / 187 = 75.9%   (196 companies)
    gridiron 86.7%   hoops 93.5%   pitch 69.0%

Multiplied through `org_entities.json`'s athlete->org edges (99.98% resolved):

    ATHLETES reaching a company in 2 hops : 5,248 / 6,226 = 84.3%
    athlete -> sponsor, direct (above)    :               0.22%
    gridiron 88.0%   hoops 95.7%   pitch 63.4%

**This is the first signal in this estate that does not collapse into basketball.** Every
value and outcome column is hoops-only; the company edge is above 63% in all three sports.

**What it is NOT.** These are INSTITUTIONAL edges — who owns the club, who named the
stadium, who is on the shirt. They are not personal endorsement deals, and 84.3% must
never be described as "84% of athletes have sponsors". The honest reading is *brand
exposure through the employer*, which is a different and weaker claim than the one the
0.22% number was trying to answer.

**Typing is the measurement, not presence.** `owner` is 70.1% populated but only 40.1%
organisations — the rest are individuals (Jeanie Buss, Dan Gilbert). Counting raw
presence would have overstated the company edge by 30 points. `P138` is resolved through
the venue entity rather than by string-matching the label, so "Madison Square Garden"
does not become a company by accident.

**One trap, recorded because it looked exactly like a real negative.** The first
athlete-reach join matched bare team names and returned **0 / 6,226 = 0.0%**. Gridiron
orgs are keyed by CODE (`gridiron::ARI`) while hoops and pitch use full names, so the
join missed every row. A 0.0% here is indistinguishable from "the edge is useless" — the
join key is `sport::team` and the script now asserts how many org keys matched (187/187)
so a silent zero cannot be read as a finding again.

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

## Q5 fired — and the one thing that would make it non-obvious

`query_graph.py --q pay` is the first positive, replicated:

| sport | n | spread | shuffle p95 | ratio |
|---|---|---|---|---|
| hoops | 11,149 | 7.11 pp | 0.65 | **11x** |
| gridiron | 4,965 | 18.10 pp | 1.29 | **14x** |

Face validity it could have failed: gridiron A0 takes **20.41%** of team pay, double the next
role, and A0 was recovered from play data with no position label supplied. A role commanding
a fifth of an NFL payroll is a quarterback.

**It is association-only, and the confound is the whole question.** Role and quality are
entangled: "this role is paid more" and "better players end up in this role" produce the same
table. Without separating them the finding is real but unsurprising — everyone knows QBs are
expensive.

### The exact test that would make it non-obvious, and what it needs

Stratify by a quality/usage proxy and ask whether archetype STILL predicts pay share inside a
narrow band. A yes is a **role premium beyond usage**, which is a genuinely sellable claim; a
no says the archetype effect is just quality wearing a role label.

The proxy is minutes played, and it is nearly free:

- `vector-hoops/pipeline/build_min_gp.py` caches `pergame_<season>.json` with MPG/GP. **It has
  never been run here — 0 such files exist.** Checked, not assumed.
- `pipeline/acquire_hoops_rosters.py` already calls the same endpoint
  (`LeagueDashPlayerStats`) for TEAM_ABBREVIATION. `MIN` and `GP` are columns on the response
  it already receives and discards — the same shape as the bug that capped hoops org coverage
  at 35.53% until 8a4ec34.

So: add two fields to the existing pull, re-run 30 seasons, stratify. Not attempted here
rather than attempted badly — a stratification on a weak proxy (career length, say, which
correlates with role independently) would produce a confounded number that looks like an
answer, which is the failure mode this whole file exists to prevent.

## Quality signals — a separate census, because Q6/Q7 needed one

Q5 (archetype predicts pay) replicated across hoops and gridiron. Q6 and Q7 — which hold
usage and then QUALITY constant, and are the rungs that make the finding non-obvious — are
hoops-only. The blocker is a quality metric, and the two sports are not equally served:

| sport | metric | coverage | verdict |
|---|---|---|---|
| hoops | `PIE` in `vector-hoops/.../dashadvanced_<season>.json` | 14,565 player-seasons, 30 files | **usable** |
| gridiron | fantasy `ppg.ppr` in `vector-gridiron/assets/vectors.json` | 10,700 rows | **not usable — see below** |

### Why fantasy PPR cannot carry Q7, measured rather than assumed

    QB  n=1155  mean ppr 12.98
    RB  n=2916  mean ppr  7.58
    WR  n=4383  mean ppr  7.37
    TE  n=2246  mean ppr  5.15

A 2.5x spread by position, produced by the SCORING SYSTEM — passing yards score at 1/25,
rushing and receiving at 1/10, and quarterbacks accumulate more of the former. A QB is not
2.5x better than a TE. Raw PPR is therefore a position label wearing a quality label, the
same shape as venue capacity being a sport label wearing a market label (Q1).

The obvious repair makes it worse. Ranking PPR WITHIN position would absorb the very signal
Q7 tests, because archetype correlates with position by construction — archetype_map.json's
gridiron_hint for A0 is literally "QB / high-usage RB / WR1". Stratifying on a
position-normalised metric would partial out the archetype effect and return a null that
means nothing.

So the gridiron replication of Q6/Q7 is BLOCKED, not merely unrun, and it is blocked on a
metric rather than on effort. What would unblock it is a play-value measure independent of
fantasy scoring — EPA per play, or PFF-style grades. EPA is derivable from public
play-by-play (nflfastR-class data); PFF is licensed.

Until then the honest scope of the finding is: **the quality-controlled role premium is a
BASKETBALL result.** Q5's cross-sport replication does not extend to Q6/Q7, and saying
otherwise would be the "basketball study wearing a cross-sport graph" failure this file
already names.

## The standing rule this estate keeps proving

Measure coverage before building the model. It killed the brand entity at 0.22% before an
encoder existed for it, justified the org entity at 99.98%, and is why the three nulls above
are trustworthy instead of embarrassing. A registry and a coverage number cost an hour; an
encoder trained on a 0.22% edge costs a week and produces a confident wrong answer.
