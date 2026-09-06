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

**Merge target and blocker — correction from an earlier draft of this PR body.** An
earlier version of this section claimed the base was `origin/master` and that `master`
already carried `origin/main`'s content. Re-verified directly and that was **wrong**:
`git merge-base --is-ancestor <main's tip> origin/master` returns false — `master` is
**7 commits behind `main`** (`git rev-list --count origin/master..origin/main` = 7,
reverse = 0) and does not contain the japandi-v4 redesign or the news-tower deploys at
all. The correct base is **`origin/main`** (`df1268510f19f4d05bd34f474450e8e345bca06e`):
`git merge-base weekend/live-fix-unified origin/main` returns `df1268510` exactly (main's
own current tip), **3 commits ahead, 0 behind** — this branch is `main` + this lane's 3
new commits (`273beb0` the D2/D3 code fix, `ba3aaf1` D1 findings-only, `559af99`
attribution correction), matching the L2 commit's own text ("worktree HEAD pinned to
df126851"). Clean merge into `main`, no conflicts expected, and it is the branch that
directly reaches live production (per L1, production is `origin/main@df126851`). Merging
this branch into `master` instead would require first fast-forwarding `master` through
`main`'s 7 commits — a larger, separate reconciliation, not something this branch alone
resolves. No git-level blocker to merging into `main`.
