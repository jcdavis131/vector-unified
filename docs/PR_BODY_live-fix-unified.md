# weekend/live-fix-unified

**What and why.** Fixes two live-site honesty defects on vector-unified.vercel.app
confirmed against `origin/main@df126851` (the exact commit L1 blob-matched to the live
bytes, via sha + Vercel's own production-deployment record): a broken cross-repo relative
asset path, and 24 hardcoded roster-tile numbers rendered as if measured. A third defect
(dead SPA rewrites) is a dashboard-only issue with no repo-side fix — documented, not
coded around.

**Measured evidence.**
- **D2** (broken cross-repo path): `public/game/index.html` linked
  `../../vector-hub/packages/vector-tokens/tokens.css`, resolving to a sibling repo not
  part of this deployment (confirmed 404 live via curl). This repo already has the real
  file the link was meant to resolve to: `assets/tokens.css` (repo root, added in
  `6f116d95`, "neobrutalist SSOT for unified") defines exactly the `--void`/`--ink` custom
  properties `/game`'s inline `<style>` depends on; `offline.html`/`sw.js` at the repo root
  already reference it at `/assets/tokens.css`, a path never actually populated under
  `public/`. Fix: copied `assets/tokens.css` byte-identical into `public/assets/tokens.css`
  (sha256 `88fe3fda...` both sides) and repointed the `<link>`. Before: curl of the old path
  404s; after: curl 200, `--void:#080A0F` and `--ink:#111` both present, and the old
  vector-hub path still correctly 404s (proves the link changed, not just that a copy was
  added).
- **D3** (fabricated roster metrics, medium severity): `/`'s inline `#mod-roster` script
  hard-coded 24 floats (a/b per tile) rendered as `<b>{a}</b><i>AR / stretch</i>` styled
  identically to a measured stat. Nothing on this page fetches `unified_slim.json` or
  `unified.json` (only fetch is `news_features.json`) — these numbers could not have come
  from any served or servable file. Fix: removed the two `.metric` divs from the roster
  tile template and the now-unused `a:`/`b:` fields from all 12 `rosterNames` entries.
  Verified via curl of the served bundle: 0 occurrences of `r.a.toFixed`/`r.b.toFixed`/the
  removed div markup remain; the orphaned `.tile-metric` CSS rule was left in place (dead
  styling, not a fabricated-number defect, out of scope for a minimal fix).
- **D1** (dead SPA rewrites, docs-only): `/model`/`/play`/`/model.html` all 404 or
  redirect-loop live even though `origin/main`'s own `vercel.json` already declares the
  correct rewrites block — nothing to edit in the repo. Called `get_project` and
  `get_deployment_build_logs` directly (read-only Vercel API inspection): the project
  response has no `rootDirectory`/`outputDirectory`/`rewrites` field at all, and the build
  log shows a zero-config static build (29ms, no rewrite-manifest step). A later commit on
  this branch re-attributed this evidence to the lane's own re-verification rather than
  reporting L1's tool calls as flat facts.

**Verified, and how.**
- Worktree HEAD pinned to `df1268510f19f4d05bd34f474450e8e345bca06e` (the exact commit L1
  blob-matched to live bytes); served `public/` locally on `127.0.0.1:8934`, curled every
  touched URL before committing.
- Ran the repo's full test suite: 38/38 passed
  (`tests/test_metric_equivalence_gates.py` + `tests/test_vector_core_adoption.py`,
  CPU-only numpy/pytest, no hardcoded home-checkout paths).
- Killed the server and confirmed no listener remained (`Get-NetTCPConnection`).
- Home checkout (`C:\Users\jcdav\vector-unified`) untouched: porcelain empty before and
  after, no fetch/checkout/merge/stash/reset run against it — only `git worktree add` off
  the pinned hash.

**Explicitly NOT done.** D1 (dead rewrites) is out of repo-scope, per above — needs the
Vercel dashboard.

**Merge target and blocker.** Base: **`origin/master`** (`a6c8e4b1`) — verified directly:
`git merge-base weekend/live-fix-unified origin/master` returns `a6c8e4b1`, i.e. `master`'s
current tip exactly, 10 commits ahead. Of those 10, 7 are pre-existing merge/deploy commits
already carrying `origin/main`'s content into `master` (japandi-v4 redesign, news-tower
deploys); the 3 new commits from this lane are on top:
`273beb0` (the D2/D3 code fix), `ba3aaf1` (D1 findings-only), `559af99` (attribution
correction). Clean merge into `master`, no conflicts expected. Live production, per L1, is
actually `origin/main@df126851` — this branch's fix commits target the content that is
live (verified against `df126851`'s bytes directly), and `master` already contains that
content via the pre-existing merges, so merging to `master` does carry the fix forward. No
git-level blocker.
