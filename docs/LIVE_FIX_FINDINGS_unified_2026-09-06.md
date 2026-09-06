# Live-fix findings — vector-unified — 2026-09-06 (lane L2)

Scope: fix lane pipelined from L1's live-site fetch audit
(`C:\Users\jcdav\.claude\jobs\b3fe1852\tmp\lanes\L1-unified\AUDIT_unified.md`).
Live production ref confirmed: `origin/main` @ `df1268510f19f4d05bd34f474450e8e345bca06e`
(blob-hash match + Vercel API `target:"production"` deployment match — see L1 audit §1).

Of L1's three defects, two (D2, D3) were minimal code fixes and are committed on
`weekend/live-fix-unified` (`273beb0`). This file documents the third, **D1**, which
this lane is stopping on rather than attempting a repo-code fix, because the repo's
own configuration is already correct — the break is outside anything a worktree commit
can touch.

## D1 — dead SPA rewrites (not fixed here; needs a Vercel dashboard check)

L1 first surfaced this defect; every claim below was independently re-verified by
this lane (L2) directly — via live curl and via the same read-only Vercel API calls
L1 used — rather than taken on L1's word.

**What's broken (verified live via curl, this lane, 2026-09-06):**

`origin/main`'s own `vercel.json` (confirmed present at the exact commit backing the
live production deployment) declares:

```json
"rewrites": [
  {"source": "/play", "destination": "/index.html"},
  {"source": "/players", "destination": "/index.html"},
  {"source": "/model", "destination": "/index.html"},
  {"source": "/trends", "destination": "/index.html"},
  {"source": "/lab", "destination": "/index.html"},
  {"source": "/dfs", "destination": "/index.html"},
  {"source": "/(.*)", "destination": "/index.html"}
]
```

Live, every one of those six named paths **and** the catch-all 404
(`x-vercel-error: NOT_FOUND`) instead of rewriting to `/index.html`. This is not a
hypothetical unlinked route: `/research` and `/insights` (both real, reachable, 200
live pages) contain their own `href="/play"` links, so a visitor following the site's
own link graph hits a dead end.

**Why this repo's own config is not the bug:** `cleanUrls` — declared in the *same*
`vercel.json` file — **is** honored live (`/model.html` → `308` → `/model`, re-curled
by this lane, confirmed above), which proves the file is read at Vercel's
build/deploy time. Only the `rewrites` block is not being applied. This lane called
`get_project` (`prj_KFZ3qDFdIwmVXZkK4AyzFv0KqQ7w`) directly: the returned object has
no `rootDirectory`, `outputDirectory` override, or rewrites-related field at all
(only `id`, `name`, `framework:null`, `accountId`, `createdAt`, `updatedAt`,
`nodeVersion`, `live:false`, `latestDeployment`, `domains`) — read-only inspection
cannot see whatever dashboard setting is causing the divergence, let alone fix it.
This lane also called `get_deployment_build_logs` on the live production deployment
(`dpl_425fgo3NcpUFwLgW3kromB68eZRB`) directly and confirms the full log: a
zero-config static build, `Build Completed in /vercel/output [29ms]`, with no
rewrite-manifest generation step logged before "Deploying outputs..." — consistent
with (not proof of) a dashboard-level Root Directory or project override causing
`vercel.json`'s `rewrites` specifically to be skipped at deploy time.

**Why no code change is attempted:** there is nothing in the repo to edit — the
`rewrites` block is already exactly what the fix would be. Editing it further (e.g.
duplicating a fixed version elsewhere) would not change what the live deployment
reads, and per this lane's guards, no `vercel` CLI, deploy command, or dashboard
action is permitted from here.

**Reproduction (curl, live site, re-run by this lane 2026-09-06, output quoted
verbatim):**

```
$ curl -s -o /dev/null -w '%{http_code}' https://vector-unified.vercel.app/model
404
$ curl -s -o /dev/null -w '%{http_code}' https://vector-unified.vercel.app/play
404
$ curl -s -D - -o /dev/null https://vector-unified.vercel.app/model.html
HTTP/2 308
location: /model
```

**Recommended operator action:** in the Vercel dashboard for project
`prj_KFZ3qDFdIwmVXZkK4AyzFv0KqQ7w`, check Project Settings → General → Root
Directory / Build & Output Settings for an override that would cause `vercel.json`'s
`rewrites` block specifically (as opposed to `cleanUrls`) to be skipped at deploy
time, and compare against a fresh deploy of the same commit to see whether the
rewrite manifest is generated.

No number, path, or copy was added or removed from the repo for this item — this is
a findings-only entry per the lane's product-decision stop rule.
