# Published-vs-artifact drift watch

Appended by the drift-watch cron. Question: does every number dumbmodel.com
publishes still match the artifact it cites? The site's fine print says every
number is recomputable from public sources.

A row is evidence the check ran, including when nothing changed. An absent row
is not 'no drift' — it means nobody looked.

Open finding: data/TENNIS_CITATION_GAP.md

| when (UTC) | count | delta | note |
|---|---|---|---|
| 2026-08-05 17:57Z | 6 | unchanged | same 6 tennis values, same both-sides numbers (mtnn_mean 0.1168 vs 0.1157, ridge16_gain 0.0941 vs 0.0846, ridge28_gain 0.1012 vs 0.0896, mtnn_sd 0.0152 vs 0.0132, candidates_add_over_16 0.0071 vs 0.0051). Read-only run: git status unchanged after. |
