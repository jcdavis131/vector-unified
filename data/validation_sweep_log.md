# Validation sweep log

Appended by `scripts/validation_sweep.py`. Read-only checks only — see that
file's docstring for what is deliberately NOT run and why.

A row here is evidence a sweep happened, including when it found nothing.
An absent row is not 'clean'; it means nobody looked.

The `interpreter` column was added 2026-08-05. Rows above that date have it EMPTY, and
empty means "not recorded", not "pinned-venv" — the sweep fell back to `sys.executable`
whenever the CUDA venv was missing and no row ever said which one ran. Do not read the
blank as a pass.

| when (UTC) | summary | per-check | semantics counts | interpreter |
|---|---|---|---|---|
| 2026-08-05 12:56Z | 3/5 pass | field_semantics=FAIL cited_fields=FAIL ablation_consistency=PASS merged_careers=PASS superlatives=PASS | semantics {'COUNT': 1, 'files_scanned': 416} |
| 2026-08-05 12:58Z | 4/5 pass | field_semantics=PASS cited_fields=FAIL ablation_consistency=PASS merged_careers=PASS superlatives=PASS | semantics {'files_scanned': 416} |
| 2026-08-05 12:59Z | 4/5 pass | field_semantics=PASS cited_fields=FAIL ablation_consistency=PASS merged_careers=PASS superlatives=PASS | semantics {'files_scanned': 416} |
| 2026-08-05 13:50Z | cron sweep | field_semantics=PASS(0 findings, doc-cov 64.0%) validate=2 FAIL(artifact_freshness, cited_fields) gate_inputs=PASS(0 pass-here-fail-there) | worktree 16P/2F/2S clone 8P/1F/2S/9NA | tennis_forward_report drifted AGAIN (gain_mean_across_cuts 0.0949->0.0863) — reverted, not committed |
| 2026-08-05 13:51Z | 4/5 pass | field_semantics=PASS cited_fields=FAIL ablation_consistency=PASS merged_careers=PASS superlatives=PASS | semantics {'files_scanned': 416} |
| 2026-08-05 17:48Z | cron sweep | sweep 4/5 (cited_fields FAIL) · gate_inputs PASS | worktree 17P/2F/2S · clone 9P/1F/2S/9NA · 0 pass-here-fail-there | semantics 0 findings, 417 files, doc-cov 96.6% | tennis_forward_report drifted AGAIN — reverted. Cause: check_gate_inputs_tracked.py runs validate.py in the WORKING TREE, which retrains tennis. The cron's no-validate.py rule is evaded transitively. |
| 2026-08-05 19:50Z | 4/5 pass | field_semantics=PASS cited_fields=FAIL ablation_consistency=PASS merged_careers=PASS superlatives=PASS | semantics {'files_scanned': 418} |
| 2026-08-05 20:18Z | 4/5 pass | field_semantics=PASS cited_fields=FAIL ablation_consistency=PASS merged_careers=PASS superlatives=PASS | semantics {'files_scanned': 419} | pinned-venv |
| 2026-08-05 20:52Z | 4/5 pass | field_semantics=PASS cited_fields=FAIL ablation_consistency=PASS merged_careers=PASS superlatives=PASS | semantics {'files_scanned': 419} | pinned-venv |
| 2026-08-05 20:57Z | 4/5 pass | field_semantics=PASS cited_fields=FAIL ablation_consistency=PASS merged_careers=PASS superlatives=PASS | semantics {'files_scanned': 420} | pinned-venv |
| 2026-08-05 21:25Z | 4/5 pass | field_semantics=PASS cited_fields=FAIL ablation_consistency=PASS merged_careers=PASS superlatives=PASS | semantics {'files_scanned': 420} | pinned-venv |
| 2026-08-05 21:34Z | 4/5 pass | field_semantics=PASS cited_fields=FAIL ablation_consistency=PASS merged_careers=PASS superlatives=PASS | semantics {'files_scanned': 421} | pinned-venv |
| 2026-08-05 21:43Z | 4/5 pass | field_semantics=PASS cited_fields=FAIL ablation_consistency=PASS merged_careers=PASS superlatives=PASS | semantics {'files_scanned': 421} | pinned-venv |
| 2026-08-05 21:51Z | 4/5 pass | field_semantics=PASS cited_fields=FAIL ablation_consistency=PASS merged_careers=PASS superlatives=PASS | semantics {'files_scanned': 421} | pinned-venv |
| 2026-08-05 22:12Z | 4/5 pass | field_semantics=PASS cited_fields=FAIL ablation_consistency=PASS merged_careers=PASS superlatives=PASS | semantics {'files_scanned': 421} | pinned-venv |
| 2026-08-05 22:52Z | 4/5 pass | field_semantics=PASS cited_fields=FAIL ablation_consistency=PASS merged_careers=PASS superlatives=PASS | semantics {'files_scanned': 422} | pinned-venv |
