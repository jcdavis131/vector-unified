# Active Tasks - Who's touching what

> One file, one truth. Write your claim BEFORE you edit, clear it when done.
> Format: | Agent | Repo / Area | Since (CT) | What / Why | Branch | Status |

| Agent | Repo / Area | Since | What / Why | Branch | Status |
|-------|-------------|-------|------------|--------|--------|
| Scout | vector-hoops / MTNN v6 fusion | 22:08 CDT | Port transformer fusion + SupCon/VICReg, lift composite 0.7937→0.85 | scout/hoops-v6-fusion | in-progress |
| Scout | vector-gridiron / training pipeline | 22:08 CDT | Bring training in-repo, fix 16-d vs 32-d vs 64-d confusion | scout/gridiron-train-in-repo | in-progress |
| Scout | vector-unified + vector-hub | 22:08 CDT | Push G2 sport-blind 0.685→0.64, verify ablation table | scout/unified-g2-blind | in-progress |
| Scout | dottie / nano 1k + tech debt | 22:08 CDT | First real nano 1k steps, scrub cache, unify checkpoint paths | scout/dottie-nano-1k | in-progress |
| Scout-lane2 | dottie + scout-cli v0.8 polish | 22:43 CDT | Night shift lane 2 verify triple-write + nano smoke deterministic + 1k spec + scaffold | scout/dottie-cli-night2 | done 03:45 CT — triple verified 7-field, 15-dirs scrub 0 left, gitignored pipeline/runs, manifest v0.8.0 fs true net false, 1K spec written |
| Claude-Local | vector-unified / LOCAL-GPU G2 push | 05:0x CDT | MEASURED, NOT PROMOTED. Handoff patch existed nowhere, so I implemented coral_centroid_loss (1st moment on z — coral_loss matched 2nd moments only, leaving sport decodable from the MEAN) + --w-coral/--w-coral-centroid/--grl-lambda-target. PAIRED 3 seeds vs control, same seeds: G2 -0.0458, t=-9.11 df=2 p=0.0118, 95% CI [-0.0674,-0.0241] excludes 0, 3/3, all gates PASS. G3/G1/rank costs are NIL under pairing (my earlier cost claim was a baseline artifact). **N=5 CONFIRMATION RUN (seeds 7/11/13/17/19, all 3 arms): the coral/centroid term I wrote DID NOT CONFIRM — p 0.0298 -> 0.0659, CI [-0.0608,+0.0030] spans zero. Only the lambda schedule holds (-0.0562, p=0.0122, 66%). AND the mean difference is the wrong summary: FULL sd 0.0030 vs CTRL sd 0.0564, a 343x variance ratio (F p=0.00005) — the treatment CLAMPS G2 to ~0.6236 while the bimodal control wanders 0.661-0.778, so the headline (-0.0458 at n=3 -> -0.0851 at n=5) is a fact about which controls were drawn, not an effect size. TWO SELF-CORRECTIONS before that: (1) my floor constant 2.31 was t(0.975,df=8) — the n=9 value — used at n=3, where the correct constant is 4.303; the margin is 2.12x not 3.9x, across 8 stat blocks not just the headline. NO conclusion flips. (2) THIRD ARM RUN: the lambda schedule is 78% of the effect (-0.0356, p=0.0094), the coral/centroid term I wrote is 22% (-0.0102, p=0.0298) and FAILS Bonferroni at 2 tests. Do not credit this result to the centroid loss at n=3.** Shipped model restored byte-for-byte and hash-verified (sport_acc 0.6851, ckpt b055641c03760624) — promotion is yours. Branch local/unified-g2-gpu @252be5d. | local/unified-g2-gpu | done |
| Claude-Local | vector-pitch / verify + push (free lane) | 16:4x CDT | DONE — 13/13 green, json.tool clean, rebased + pushed to pitch master a36b48d. Handoff's 0904a39 / vectors_mtnn.json do not exist here; the seed work shipped instead. | master (ff) | done |

## How to use
1. Add your row before editing
2. Keep main green - work on your own branch, PR or fast-forward only when tests pass
3. New assets = candidate.json -> promote only when eval beats current + gate passes
4. Log even if no change ("checked, no-op") so others know you looked
5. Clear row when done

| Scout-lane1 | vector-* all 4 / honesty pass | 22:43 CDT | equities 4831×500 0.7057 lift6.32 verified fixed d80a716, hoops v6 17 towers d128 4L 4H CLS→64-d leak-free test top1 0.438→0.55 target recall@10 0.977 verified, pitch 588/633 92.9% WC-only 633 2430×11ctx verified, gridiron 32-d native 16-d compat wrapper gate NO promote MAE 8.41 synthetic vs claimed 4.268, branch scout/vector-honesty-night1 4 repos, tests 8p+13p PASS, timeline.jsonl ok | scout/vector-honesty-night1 | done 03:46 CT — pushed 03:49-03:50Z to origin: equities new branch, gridiron main 2bab470..55aacd7, hoops master fcc606e..0c4b039, pitch master a36b48d..cb77f22 (merged Claude-Local a36b48d+bdfa4a0) |

| Scout-push | vector-* honesty branches → origin push | 03:50Z UTC 2026-08-05 | Lane 1 push: 4× scout/vector-honesty-night1 → origin, 3× main/master fast-forward (no force), honest README only, no candidate→vectors promotion per gate, timeline.jsonl private left in ~/workspace/bundles/ultra/runs/night1-honesty, log bundles/research/push-log-night2.md | scout/vector-honesty-night1 | done 03:50Z UTC |
| Scout | dottie / distilled reasoning → nano GRPO | 23:01 CDT | Audit: grpo.py 387L + grpo_torch real + ET-CoT traces + 5 runs exist, no nano→pref collector → scaffolded docs/GRPO_PIPELINE.md + grpo_collect.py numpy-only (groups SHA1, adv (R-mean)/std, chosen/max rej/min margin0.05, thermostat stats) — 29 tests PASS, branch pushed c/dottie-traces-grpo | scout/dottie-traces-grpo | done 23:03 CDT / 04:04Z — spec 8.7k + collector 15k, MANIFEST deterministic, no torch, ready for local GPU grpo 250 steps |
| Scout | vector-hub / chimera daily + provenance depth | 23:01 CDT | Daily LCG rotation polish YYYYMMDD UTC LCG wired hubDailySeed()+hubLcg()+unifiedChimeraDaily() exposed window.UNIFIED_CHIMERA_DAILY, provenance 7-file DM_PROVENANCE [prov] ok/total/bad logger, CSS mode-card--chimera verified, 5th game chimera index live, Vercel 200 six models five daily verified, branch scout/hub-chimera-provenance pushed 23:02 CDT d90788b fast-forward main 3529e7a..d90788b | scout/hub-chimera-provenance | done 23:02 CDT — Vercel propagated 200 sixmodels/fivedaily chimera tile present, hub.js provenance depth 7 files hashes 7/7/10/3/6/14/12 entities ok |
| Scout | vector-unified / G3/G4 + chimera eval | 23:01 CDT | G3 sil 0.683 within>>between, G4 cross-NN 0.9828 lift audit, chimera difficulty band 92.9% verify | scout/unified-g3g4-chimera | done 04:08 CT — G3 0.683 within0.746>>-0.121 sep0.867 PASS, G4 0.9828 vs0.1712 lift0.8116 PASS, pitch 588/633 92.9% median0.4843 +202 PASS, README newly tracked provenance-honest, docs/G3G4_CHIMERA_AUDIT, no torch, pushed branch c2f00b5..6a56132 + master fb4adcb |

## Free lanes right now
- vector-hub / daily 5th puzzle (unified chimera) + provenance checksums
- dottie / distilled reasoning optimizer traces→nano GRPO
- LOCAL GPU heavy trains (OOM in Hatch) — see handoff table above, do NOT pip torch

## 2026-08-04 22:20 CT — HANDOFF to local GPU agent
| LOCAL-GPU | vector-unified / unified G2 0.685->0.64 | 22:20 CT | FULL TRAIN: GRL λ0.3→0.5 + CORAL centroid, missing caches embedding_v3.npz / mtnn_best.pt / pitch_mtnn_embeddings.json, torch OOM workaround → run train_stage2.py --smoke -> train_unified.py 60ep -> eval_unified.py on local GPU | local/unified-g2-gpu | claimed |
| LOCAL-GPU | vector-hoops / v6 transformer 150ep | 22:20 CT | MTNN v6 d_model128 4-head CLS→64-d 17 towers, w-vicreg 0.05, target composite 0.7937→0.85 test top1 0.438→0.55 | local/hoops-v6-gpu | claimed |
| LOCAL-GPU | vector-gridiron / real nflverse | 22:20 CT | nflreadpy 2020-2025 weather+Vegas, 32-d native training, MAE 4.268→3.8 | local/gridiron-real | claimed |
| Orchestrator | scout-cli / harness polish | 23:02 CDT | v0.8 harness manifest fs true net false + cli.sh wrapper perms + shared lib verify, no torch | scout/scout-cli-harness-polish | done 2026-08-04 23:04 CDT / 2026-08-05T04:04Z — manifest v0.8.0 verified fs true net false, wrapper exec 896B rwxrwx---, towers.py 6404B losses.py 2745B honest, doc HARNESS_POLISH_2026-08-05.md committed a4f58f4 pushed origin |
| Orchestrator | vector-pitch / difficulty polish follow-up | 23:05 CDT | WC-only 633 2430×11ctx LoCo 0.797 vs PCA3, difficulty 588/633 92.9% verify, no torch | scout/vector-pitch-polish-followup | done 23:05 CDT 2026-08-04 / 04:05Z 2026-08-05 — vectors.json 633 WC-only 319+314=633 L2 mtnn_v1_24d_l2, vectors_mtnn.json 2430×24d 11ctx WC633+1797other contexts[BL×83 CA24×155 Euro20×256 Euro24×239 LaLiga18/19×77 LaLiga20/21×65 Ligue1×52 PL×418 SA×452] PASS, difficulty 588/633 92.9% median0.4843 es0.6 7easy 38hard slope2.5 warm0.985 salience profile16d PASS, LOCO tm_9ctx 2295rows 9ctx baseline0.7265 vs SupCon0.797 +0.0705 vsPCA3 0.7008 +0.0962 vsPCA16 0.7457 +0.0513 PASS knn5 0.7621→0.7894 nn_role 0.7217→0.7492 recon 0.4773→0.4956, README↔difficulty_calibration.json+eval_scoreboard honest sync no inflated claims, branch scout/vector-pitch-polish-followup pushed cb77f22 no-force — free lane audit |
| Orchestrator | vector-equities / forward IC polish | 23:05 CDT | 4831 FYs 500 tickers 154f 20 towers 64-d transformer purity 0.7057 lift6.32 baseline0.1117 cross0.4013 sil-0.0034 IC 1m0.0051 3m0.0064 6m0.007 gate verify leakage guard — DONE | scout/vector-equities-polish-fwd | done 04:04 CT — FY median-impute per-FY (global fallback only), no ticker future leak, FY-emb 12-d fusion-only, coverage scalar, causal mask, cross-ticker 0.4013 verified, forward IC>0 triple-barrier 0.2189, README sync 0.0051/0.0064/0.007 |
| Orchestrator | vector-pitch / difficulty polish | 2026-08-05T04:33Z | follow-up tournament-z verify median clip, no torch | scout/vector-pitch-polish-cycle2 | in-progress |
| Orchestrator | vector-equities / forward IC polish | 2026-08-05T05:03Z | forward 1m/3m/6m IC gate verify, leakage guard | scout/vector-equities-polish-cycle3 | in-progress |

## Gotchas found the hard way (local GPU box, 2026-08-05)

- **`git checkout` manufactures `artifact_freshness` FAILs.** Switching branches rewrites
  mtimes, and a builder landing 9ms after its own output reads as STALE. Seen as
  `bridge_index.json (0.0h behind)` — every data key byte-identical, only its `built`
  timestamp differed. Do NOT "fix" these by rebuilding: mtime is file-granular, so each
  rebuild cascades and pushes the NEXT artifact to 0.0h. A fresh `artifact_freshness` FAIL
  with `0.0h behind` is checkout noise, not a regression. The real entries carry real
  numbers (`stage2_history.json 114.3h`, `assets/unified.json 24.3h`).
- **Never "refresh" `stage2_history.json` to green the gate.** It can only be regenerated
  by re-running training, which overwrites `unified_stage2_best.pt`. That trades the
  verified shipped model (sport_acc 0.6851, ckpt `b055641c03760624`) for a green line.
- **Restore from an explicit manifest, never by inferring paths from backup filenames.**
  Guessing sent two role-named backups (`before.json`, `unified_report.json.pre_eval`) to
  invented paths, CREATED two junk files, printed "RESTORED" for all six, and left
  `data/unified_report.json` holding a throwaway run's `sport_acc 0.6363`. The one file a
  reader would quote was the one left wrong. Use `pipeline/restore_shipped.py --verify`.
- **`t=2.31` is the n=9 paired constant.** At n=3 (df=2) the two-sided value is 4.303. I
  published "3.9x the floor" off the wrong one across 8 stat blocks. Check the constant
  against `scipy.stats.t.ppf(0.975, df)` — keyed on **df**, not n.

Recorded here rather than in a commit message, because a commit message is not where
anyone looks before running a command.

- **`eval_unified.py --ckpt` takes a NAME, not a path.** It is joined to UCACHE
  (pipeline/data/). LOCAL_GPU_HANDOFF section 1 says
  `--ckpt pipeline/data/unified_stage2_best.pt`, which double-prefixes and dies with
  FileNotFoundError on a doubled pipeline/data segment.
  Correct: `--ckpt unified_stage2_best.pt`
- **`train_stage2.py` writes FOUR things and the fourth is not in that file.**
  data/stage2_baselines.json, data/stage2_history.json,
  pipeline/data/unified_stage2_best.pt, and pipeline/data/gridiron_season_emb.npz —
  the last written two levels down by load_encoders, behind an mtime check. Back up
  all four before any run.
- **data/ is gitignored here.** unified_report.json is overwritten by every eval and
  never shows in `git status`. Hash it if you care; the working tree will not tell you.
- **The shipped sport_acc 0.6851 is ONE draw of a `--grl-lambda 0.05` config.**
  Diffing a 0.3-lambda run against it measures the lambda, not whatever you changed.
  I published a cost claim built on exactly that mistake before catching it — the fix
  was running a real control at the same seeds.
