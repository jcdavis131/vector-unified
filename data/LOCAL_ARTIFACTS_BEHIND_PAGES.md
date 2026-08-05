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
