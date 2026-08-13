# Unified Alias Fix Runbook — unified.dumbmodel.com 404 107B DEPLOYMENT_NOT_FOUND

**Observed:** 2026-08-11 07:18 CDT `curl unified.dumbmodel.com/` → 404 107B `DEPLOYMENT_NOT_FOUND` x-vercel-id iad1::swxtp-1786450682662-7ffb6c27b4dc. Hoops same Vercel project setup returns 200 MISS 19627B same time → hoops LIVE, unified alias broken.

**Root Cause (code comment decision):** `cleanUrls:true` needs folder `owner/index.html` not just `owner.html`. Root 404 not code crash but DEPLOYMENT_NOT_FOUND alias lost → dashboard Domains→Production re-link. `vector-hoops` canonical remote `master` not `main`. Unified identical symptom: Vercel project has deployment but no Production alias attached → root returns DEPLOYMENT_NOT_FOUND 107B text/plain, sub-routes may still 200 via preview.

**Vercel prod requirements:**
- vercel.json `cleanUrls:true, trailingSlash:false`
- rewrites list must cover `/`, `/owner`, `/player`, `/brand`, `/dfs` → `*/index.html` (folder pattern, not file). Missing rewrites → /owner 404, but root 404 is alias level, not rewrite.
- headers for /assets immutable required for 31536000.
- Folder structure: `owner/index.html` exists (check `vector-unified/owner/index.html` yes). Our vercel.json previously only rewrote `/` → `/index.html`, so /owner relied on cleanUrls auto filesystem mapping but may conflict with `owner.html` vs folder. New vercel.json includes explicit rewrites for all folders.

**Fix Steps (1 click prod):**
1. Open Vercel Dashboard → Project `vector-unified` (or `unified-dumbmodel-com`) → Settings → Domains.
2. Verify `unified.dumbmodel.com` appears. If not: Add Domain → `unified.dumbmodel.com`.
3. Click Domain → Production → Select latest deployment (with our vercel.json 1561B rewrites) → Assign to Production. Redeploy button in Deployments → ... → Redeploy to Production (ensures alias re-link).
4. Wait 30s propagation. `curl -I https://unified.dumbmodel.com/` should return 200 40766B x-vercel-cache MISS/HIT, not 107B.
5. Prune preview duplicates: `vercel --prod` from project root (no push if other agent developing). Helper-only rule: do NOT force push if other agent developing per helper-only rule — only dashboard click + local vercel skip.
6. Verify GOAL Master: hub 20719×64-d dailySeed LCG same-link-same-stars.

**Validation:**
```
curl -s -D - https://unified.dumbmodel.com/ | head -5    # 200 OK expected
curl -s -D - https://unified.dumbmodel.com/owner | head   # 200 via rewrite
curl -s https://unified.dumbmodel.com/ | grep dailySeed   # should contain dailySeed YYYYMMDD
```
Hoops reference 200 sample 2026-08-11 12:18:03 GMT 19627B HIT stable 66th tick since flip 2026-08-09 20:37 CDT. Unified should match 200 pattern post-fix.

**Local check done 07:18 CDT:** hoops 200 ok (v7.4 live, now v8 viral v2 local 40328B unsynced), unified 404 confirmed 107B x-vercel-id swxtp. Local vercel.json updated 1561B parity rewrites added owner/player/brand/dfs/players/model/trends/play/offline/methods headers cache same as hoops. No push forced per helper-only rule; change is candidate locally.

**Next:** Once unified 200, log `vector-unified audit/unified_report_*.json` G2 expected 0.6851→0.64 target.

**Timeline log even no-change:** 7-field mandatory nodeId unified-alias-fix, agentId Scout-swarm-FRONTEND-20260811, attempt 1, latency_ms ~2400, tokens_est 2430, status ok (or diag when blocked), errorClass null or DEPLOYMENT_NOT_FOUND.

**Zero-deps true:** no pip, no torch, inline CSS/JS only for hoops v8, unified static HTML.

**Related:** Hoops FLIP history: 404 79B 2026-08-09 20:19:51 UTC → flip 20:37 CDT 200 36614B → 13:52 200 31600B → 01:37 404 dip → 03:07 200 19627B stable 66th tick. Same infra dip cause suspected for unified 03:07.

