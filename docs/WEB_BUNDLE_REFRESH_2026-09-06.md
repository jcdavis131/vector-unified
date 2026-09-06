# Web bundle refresh attempt — 2026-09-06 (lane L5)

Task: refresh `assets/unified_slim.json` + `assets/unified_emb.f32` on
`fix/stage2-best-tracking` from the branch's latest KEPT checkpoint. Result:
**attempted, verified unchanged, blocked from going further inside a
worktree.** No fabricated number was added; every figure below is quoted from
a file that already existed before this lane ran.

## 1. Latest KEPT checkpoint (identified, not re-derived)

Journal (`C:\Users\jcdav\herdmux\gpu\results.tsv`), the `keep` row for
`vector-unified`:

```
2026-08-15T01:55:35+00:00  vector-unified  22812038  6da99b5ef967  0.6300  0.0019  6  +0.0121  keep  ...
grl-lambda-target 0.5 -> 1.0, host mode. ... GRL is adversarial and --w-sup 0.5 bought G2 out of gridiron
```

Branch history (`git log --oneline --all --grep=KEPT -i`):

```
c2829fd docs(stage2): --grl-lambda-target 1.0 KEPT - fourth climb, and the floors nearly stopped it
2281203 Reapply "exp(stage2): --grl-lambda-target 0.5 -> 1.0"
```

`c2829fd`'s message: baseline n=6 mean 0.6421 ± 0.0062 (host protocol
`6da99b5ef967`) → variant n=6 mean 0.6300 ± 0.0019, delta +0.0121 vs an
acceptance bar of 0.0062 → KEEP. `2281203` is the functional commit (sets
`--grl-lambda-target` default to 1.0); `c2829fd` only rewrites the comment
above it to record the keep. Both are ancestors of this branch's current HEAD
(`92e4f9a`), so the branch's code already reflects the KEPT arm's default.

On disk, `pipeline/data/unified_stage2_best.pt` and `data/stage2_report.json`
(home checkout, both untracked — `data/*` and most of `pipeline/data/` are
gitignored) are mtime **2026-08-15 09:05:53 -0500**, i.e. a canonical
single-seed (seed 7) re-run made *after* the keep landed:

```
best_g2 = 0.6319980694980695   best_epoch = 27
args.grl_lambda_target = 1.0, w_coral = 0.5, w_coral_centroid = 0.5
```

That single-seed 0.6320 matches the keep commit's own per-seed table
(`seed 7 after: 0.6320`).

## 2. What the shipped web bundle is actually built from

`pipeline/export_web_slim.py` reads `assets/unified.json` (its only source)
and writes `unified_slim.json` + `unified_emb.f32`. `assets/unified.json` is
gitignored (`assets/**/*.json`) — it is *not* the KEPT checkpoint's export.
Its own meta fields:

```
built            2026-07-30                                    (a literal string in
                                                                  export_unified_stage2.py,
                                                                  not a real timestamp)
model            UnifiedTrunk Stage 2.1 (unfrozen encoder alignment, best_epoch=58)
g2_sport_acc     0.6850868725868726
g2_target        0.7258
g2_majority_baseline  0.6258
g2_delta_vs_majority  0.0593
g2_status        met
```

This is a **July-30 checkpoint, older than every August G2 climb** (w-coral,
w-sup, grl-lambda-target) including the Aug-14 KEPT arm above. It was never
regenerated from `unified_stage2_best.pt` after that keep. The g2_sport_acc
0.685/+0.0593 on the current bundle is real (a real file produced it) but is
**not** the branch's current best model (0.632/≈+0.006 per the KEPT
checkpoint).

## 3. Re-export result (this lane)

Copied (not junctioned) `assets/unified.json` and `data/ablation_report.json`
read-only from the home checkout into this worktree — the only two inputs
`export_web_slim.py` reads (`SRC`, `ABLATION`). Then, offline, CPU
(`pipeline/.venv` — no torch, no GPU touched):

```
$ python pipeline/export_web_slim.py --check
CHECK OK: both outputs match a fresh export

$ python pipeline/export_web_slim.py
wrote unified_slim.json + unified_emb.f32
```

sha256, before (HEAD) and after (fresh export from the same source):

```
unified_slim.json  5a07e27e24f16aa679d0362693bd2d9ac18fb133b1e31e56439bad431bd16330  (unchanged)
unified_emb.f32    cc7abc4c16869d454748d5dce898235658c9969f25405495bfc7847c13d78e7c  (unchanged)
```

`git status --porcelain` in the worktree: empty, both before and after the
export. **There is nothing to commit for the bundle** — the tracked files
already equal a fresh export of the only source on disk.

```
$ python pipeline/check_web_slim.py
G2   acc 0.6850868725868726 vs majority 0.6258 = +0.0593  [met]
NOTE g2_status is now 'met' — if G2 genuinely passed, the page's DEFERRED
     wording must be revisited deliberately
OK: contract satisfied and no overclaiming text on the page.
(exit 0)
```

## 4. Why this lane did not go further (the real blocker)

Producing a bundle that reflects the Aug-14 KEPT checkpoint requires
`pipeline/export_unified_stage2.py`, which re-runs each sport's live encoder
over the full corpus through `unified_stage2_best.pt`. That script is not
runnable from a worktree without violating this weekend's guards:

`pipeline/load_encoders.py:45`:

```python
ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT.parent  # C:\Users\jcdav
HOOPS = HOME / "vector-hoops"
```

`HOME` is *computed*, not a literal string (the guard-11 grep for
`C:\Users\jcdav\vector-unified` correctly finds nothing), but it only equals
`C:\Users\jcdav` when the repo is checked out directly there. Verified from
this worktree:

```
ROOT        = C:\Users\jcdav\.claude\jobs\b3fe1852\tmp\lanes\L5\wt
ROOT.parent = C:\Users\jcdav\.claude\jobs\b3fe1852\tmp\lanes\L5
HOOPS would resolve to = ...\lanes\L5\vector-hoops   (does not exist)
```

Running it here would need one of: hardcoding `C:\Users\jcdav` into the
script (the exact anti-pattern the guard-11 grep exists to catch), a junction
(forbidden by this lane's brief), or moving the worktree onto the home path
(not attempted). It also needs read access to
`vector-hoops`/`vector-gridiron`/`vector-pitch`'s home checkouts (cached
embeddings + encoder checkpoints) — three repos outside this lane's
single-repo authorization, each with its own queued GPU jobs this weekend.
CUDA is not the blocker (`CUDA_VISIBLE_DEVICES=""` would force CPU cleanly);
the cross-repo path assumption is.

**Operator path to a real refresh:** from the home checkout (`git -C
vector-unified status --porcelain` empty, no unified job running),
`CUDA_VISIBLE_DEVICES= python pipeline/export_unified_stage2.py` to rebuild
`assets/unified.json` from `unified_stage2_best.pt`, then
`python pipeline/export_web_slim.py` to re-slim it.

## 5. Page wording — no change made

`check_web_slim.py`'s NOTE says the page's "DEFERRED wording must be
revisited" if `g2_status` is no longer "defer". Checked what the page
actually renders (`assets/unified.js` lines ~83-91): it does **not** hardcode
a "DEFERRED" string. It tags G2 `WEAK BAR` and renders the live status
dynamically:

```js
note: 'Recorded as "' + (meta.g2_status || '?') + '" against a target of ' +
      fmt(meta.g2_target) + ', but the artifact defines that as within 10
      points of the achievable floor and tells readers to quote the margin,
      not the status. The sport is still recoverable from the geometry.'
```

This already satisfies the repo's own honesty rule (quote
`g2_delta_vs_majority`, not the status; never claim sport-invariance) for
whatever `g2_status` the bundle carries — "met" or "not_met" render through
the same honest sentence. No wording edit was made. Separately: even if a
literal "DEFERRED" string existed, the "met" this bundle carries is from the
stale July-30 checkpoint, not the branch's actual current best (§2) —
changing the page to assert "met, +0.0593" as the branch's live state would
be less honest than what is there now, not more. That is why §4's real
refresh should happen before any wording decision, not instead of it.

## Bottom line

- Bundle: unchanged (sha256 identical, `--check` OK, nothing to commit).
- Gate: `check_web_slim.py` exit 0, with the pre-existing "met" NOTE quoted
  above.
- Wording: unchanged, and should stay unchanged until §4's real refresh runs
  — an operator decision (whether/when to run the full re-export), not a
  swarm one.
