# Local artifacts were running years behind the pages that cite them

Four artifacts, found in one afternoon, each holding a much smaller sample than the number
the live page publishes from it. Not one bug — one blind spot, hit four times.

| artifact | stored locally | live page publishes |
|---|---|---|
| `data/gridiron_name_collisions.json` | `names_probed 300`, `colliding_names 15` | "a Wikidata probe of **2,707** corpus names found **117**" |
| `data/hoops_name_collisions.json` | `names_probed 300`, `colliding_names 13` | full corpus, 2,415 |
| `data/pitch_expectation_sources.json` | `distinct_names 374` | "resolved 1,538 of **1,833** distinct names" |
| `data/pitch_age_axis.json` | `pct_scorable 21.2` | "**43.8%** of rows scorable on the age-expectation axis" |

The `300` is the tell. It is a round number in two files at once — a `--limit 300` smoke run
that was never re-run at full scale, and then sat as the local truth for months.

## Every one of them regenerates to agree with the page

That direction matters. These are not cases where the artifact is right and the page drifted;
the page was correct and the LOCAL COPY was stale. Rebuilding moved each one toward what is
already published, which is why adopting the rebuilds was safe:

    gridiron_name_collisions   300/15  ->  2707/117   exactly the published pair
    hoops_name_collisions      300/13  ->  2415/127
    pitch_expectation_sources  374     ->  1833       page says 1,833
    pitch_age_axis             21.2    ->  43.9       page says 43.8

The residual hundredths (43.9 vs 43.8, 1,539 vs published 1,538) are honest input drift
since publication, not error.

### Correction: "regenerates to agree with the page" was measured on ONE field each

Written above after checking the headline number per artifact. `check_prose_values.py`,
built the next hour, checks every cited field and shows the claim does not hold field-wise
for `pitch_age_axis.json`:

    pct_scorable          stored 21.2    regen 43.9    page 43.8    agrees
    report.rows_scorable  stored  516    regen 1066    page 1,040   DISAGREES with both
    corr_age_delivery     stored -0.1593 regen -0.1698 page -0.1671 DISAGREES with both

The page sits BETWEEN the stored and regenerated values, so it was produced from a third
run at a third moment. Adopting the rebuild moved the headline number onto the page's value
and moved these two off it. That is a real trade, not a clean win, and it stays visible in
`prose_values` as `pitch:insights[2]` rather than being quietly absorbed here.

The three-way spread is itself the finding: nothing pins an artifact to the page generated
from it, so page, local artifact and current inputs drift apart independently.

## Why nothing caught it

**`check_cited_fields` compares `value` fields. These numbers live in prose.**

The gridiron claim is a sentence — "a Wikidata probe of 2,707 corpus names found 117" —
inside an insight body. The pitch claim is inside a headline LABEL, "player-seasons embedded
(1,833 distinct names)", where only the `value` beside it (2,430) was ever compared. A number
a reader sees is a published claim regardless of which JSON key it sits in.

So the gate reported `pitch: 11 fields ok, 0 WRONG, 0 values ok` — clean, having compared
nothing at all on that page. And the existing zero-coverage refusal is a GLOBAL count, so
hoops' 24 comparisons kept the gate green while pitch had none. That is now reported per
page (`pitch` 0 of 5, `unified` 0 of 10), which is what made this findable.

**`check_artifact_freshness` compares mtimes, and an mtime cannot see a sample size.** It
said STALE for months without anyone learning that stale meant "nine times smaller". Worse,
`pitch_age_axis.json` sat behind a producer that CRASHED on every run
(`UnicodeEncodeError: 'charmap' codec can't encode character 'ć'`), so it could not have been
refreshed even by someone who noticed — and freshness reports STALE without ever asking
whether a rebuild is possible.

## What is still stale, and why it was not touched

    sponsor_synchrony.json    acquire_venue_sponsors.py fetches live sources. Rerunning is
                              not a no-op test — the input can legitimately have changed —
                              and an unattended network fetch is not a thing to trigger
                              unasked.
    stage2_history.json       written by train_stage2.py, a TRAINER. Overwrites the shipped
                              checkpoint (sport_acc 0.6851, ckpt b055641c03760624).
    assets/unified.json       the SHIPPED asset, 16 MB, cited by every live page.

Each needs an operator decision, not a rebuild.

## The method, if this recurs

Do not trust a rebuild comparison that did not verify the file was actually rewritten. Two
artifacts here reported IDENTICAL because a plain producer run never touched them —
`build_direction_axis.py` and `build_trajectory_axis.py` both default to `--sport hoops`, so
the gridiron variants were never regenerated and trivially matched themselves. Check the
mtime moved before believing the diff.

And strip the artifact's own `built` stamp before comparing. Three artifacts differed ONLY in
that field, which conflates "the output changed" with "the file records when it was written".
