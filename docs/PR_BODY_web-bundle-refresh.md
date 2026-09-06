# weekend/web-bundle-refresh

**What and why.** Attempts to refresh `assets/unified_slim.json` + `unified_emb.f32` from
the latest KEPT checkpoint so promotion is possible. Finding: **the currently-committed
export is already byte-identical to a fresh offline re-run** of the export script against
its actual inputs — there is nothing to commit for the bundle files themselves — but those
inputs (`assets/unified.json`) predate every August G2 climb including the latest keep, so
the bundle is real but not current. A true refresh needs a different script that isn't
runnable from a single-repo worktree.

**Measured evidence.**
- Identified the branch's latest KEPT checkpoint: journal `2026-08-15T01:55:35` keep row,
  commits `2281203`/`c2829fd`, on-disk `unified_stage2_best.pt` +
  `stage2_report.json` `best_g2=0.6319980694980695`, both mtime `2026-08-15 09:05:53`.
- Copied `assets/unified.json` + `data/ablation_report.json` **read-only** from the home
  checkout (the only inputs `export_web_slim.py` reads) and re-ran it offline/CPU:
  `--check` reported `CHECK OK`, and a real write produced byte-identical sha256 output —
  confirming the currently-shipped bundle already matches what that script would produce
  from those inputs.
- `pipeline/check_web_slim.py` exits 0 with the pre-existing "g2_status is now 'met'" NOTE.
- `assets/unified.json` (the export's only source, gitignored, not tied to any checkpoint)
  is a stale 2026-07-30 build that predates every August G2 climb, including the
  2026-08-15 keep — so the bundle it produces is real, verified, and internally
  consistent, but **not current** relative to the best available checkpoint.
- A true refresh needs `pipeline/export_unified_stage2.py`, which is **not runnable from a
  worktree**: `load_encoders.py:45` derives sibling-repo paths as `ROOT.parent` (only
  equal to `C:\Users\jcdav` when checked out there — verified it resolves to a nonexistent
  path from this worktree) and needs read access to three sibling repos' home checkouts,
  outside this lane's single-repo scope.
- Checked the page's G2 wording (`assets/unified.js`): it already renders
  `g2_status`/`g2_delta_vs_majority` dynamically with an explicit "quote the margin, not
  the status" caveat — no hardcoded DEFERRED string exists to revise, so no wording edit
  was made.

**Verified, and how.** `check_web_slim.py` exit 0 (verbatim above); bundle sha256 matched
before/after the offline re-run (no diff to commit); checkpoint commit/metric quoted from
the journal (`2281203`/`c2829fd`, `best_g2=0.6319980694980695`, 2026-08-15).

**Explicitly NOT done.** No bundle file was updated — the offline re-run against current
inputs reproduced the existing bytes exactly, so there was nothing to commit. The actual
refresh against the latest checkpoint is blocked on `load_encoders.py`'s
`ROOT.parent`-based sibling-repo path assumption; documented as an operator decision
(fix the path assumption, or run the export from the home checkout directly) rather than
hardcoding a path or junctioning around it.

**Merge target and blocker.** Base: `origin/fix/stage2-best-tracking` (`92e4f9a`, **not**
`master` or `main` — verified: merge-base with `fix/stage2-best-tracking` is `92e4f9a`
itself, 1 commit ahead, 0 behind; merge-base with `master`/`main` is a much older commit,
127 commits ahead / 520+ behind, i.e. this branch is meant to layer on the fix branch, not
on `master` directly). This is a **docs-only branch** (adds
`docs/WEB_BUNDLE_REFRESH_2026-09-06.md`, no code or asset changed) — no merge conflict
risk. The real blocker to a working refresh is the cross-repo path assumption above, not
anything about this branch's mergeability.
