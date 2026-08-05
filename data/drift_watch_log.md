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
| 2026-08-05 18:13Z | 6 | unchanged | CONTROLLED TEST of the 9aef6ff restore guard: ran check_gate_inputs_tracked.py (which retrains tennis internally via validate.py), guard reported restoring 2 files, ridge16_all_features_r stayed 0.8427, and cited_fields still reported 6 rather than growing. Closes the 'single observation does not prove it' caveat from the previous row. |
