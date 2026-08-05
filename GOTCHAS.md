# Gotchas — things that cost real time on this box

Every entry below happened. Several happened MORE THAN ONCE, after being written down,
because they were written down in `COORDINATION.md` — which the Hatch mirror overwrites
wholesale. Checked: the gotchas section pushed there earlier today is gone from
`origin/master`, header and all. That is why this file exists at repo root instead.

A lesson recorded in a file that gets overwritten is not recorded.

---

## Shell and tooling

**A pipe eats `$?`.** `cmd | tail -3; echo $?` reports *tail's* exit code. Cost three wrong
"exit=0" reports this session, including one printed directly above a visible traceback.
Never pipe a command whose exit status you intend to quote. Capture, then inspect:

    OUT=$(cmd 2>&1); RC=$?

**`C:\Users` inside a non-raw Python string kills a heredoc.** `SyntaxError: (unicode
error) 'unicodeescape' codec can't decode bytes ... truncated \UXXXXXXXX escape`. Hit six
times. `\U` starts an 8-hex-digit escape. Use the Write/Edit tools for anything containing
Windows paths, or `r"""..."""`. The same applies to `\N`, `\x`, `\u`.

**A `replace()` that does not match still prints your success message.** A patch script
ended with `print("fixed")` outside any condition, so a silently-failed substitution
reported success and the next verification read a stale file. `assert old in s` before
replacing, or use Edit, which fails loudly.

**Never `git checkout` or `git stash` while a background job is running.** Killed three
runs. Worst case: a branch switch deleted files mid-scan, the scan died partway, and the
PREVIOUS output stayed on disk reading as a completed result. A checker that dies mid-sweep
and leaves its old report in place is worse than one that reports partial coverage.

**Commit to `master`, switch back to your branch, and the file is gone.** Bit twice — once
for record files, once for `tools/`. Records belong on master; tooling belongs on both.

---

## Git and artifacts

**`git add -A data/` skips ignored paths SILENTLY.** `.gitignore` has `data/*` with explicit
`!` negations. Three commits reported shipping artifacts that never entered the repo. The
repo's own `.gitignore` already documented this trap and it happened again. Verify with
`git ls-files --error-unmatch <path>`.

**`git checkout` manufactures `artifact_freshness` failures.** Switching branches rewrites
mtimes, and a builder landing 9ms after its own output reads as STALE. A fresh failure at
`0.0h behind` is checkout noise. **Do not "fix" it by rebuilding** — mtime is file-granular,
so each rebuild cascades the next artifact to 0.0h. The real entries carry real numbers
(`stage2_history.json 114.3h`).

**Never refresh `stage2_history.json` to green the gate.** It regenerates only by re-running
training, which overwrites `unified_stage2_best.pt`. That trades a verified shipped model
(`sport_acc 0.6851`, ckpt `b055641c03760624`) for a green line.

**Restore from an explicit manifest, never by inferring paths from backup filenames.**
Guessing sent two role-named backups to invented paths, created two junk files, printed
`RESTORED` for all six, and left `data/unified_report.json` holding a throwaway run's
`sport_acc 0.6363`. The one file a reader would quote was the one left wrong. Use
`pipeline/restore_shipped.py --verify`.

---

## Running things

**`build_*.py`, `probe_*.py` and `acquire_*.py` write with NO FLAG AT ALL.** Screening for
`--write` is not enough. One run of a checker that executed documented commands rewrote ten
artifacts here AND stripped a `CORRECTED` marker from
`vector-hoops/pipeline/seed_floor.json` in a SIBLING repo, taking three green gates red.
Whitelist `check_*.py` and explicit `--check` arms; do not denylist.

**`validate.py` retrains tennis.** It registers `tennis_mtnn` as `train_tennis_mtnn.py
--check`, and that arm retrains. Every full gate run moves `tennis_forward_report.json` off
the value dumbmodel.com cites. Use `scripts/validation_sweep.py`, which excludes it.

**Two servers: the second silently takes the next port.** Start a dashboard while one is
already running and it binds 8001 while 8000 keeps serving old code. Produced a page that
looked current, was not, and cost a wrong verification. Check first:

    curl -s -o NUL -w "%{http_code}" http://localhost:8000/

---

## Numbers and claims

**`t = 2.31` is `t(0.975, df=8)` — the n=9 constant.** It was used at n=3 across 8 stat
blocks, where the correct value is `t(0.975,2) = 4.303`. Every constant should come from
`scipy.stats.t.ppf(0.975, df)`, keyed on **df**, not n. A second off-by-one of the same kind
(looking up by `n` instead of `df`) appeared in the same afternoon.

**A timeout is not a defect.** `probe_company_edges.py` blew a 120s budget while the box was
training, running a 25-agent workflow and building a dashboard; it exits 0 twice over when
idle. A verdict that flips with machine load measured the load, not the command.

**Documentation in a JSON *key* is not prose.** Field names written only as
`stage2_IS_reproducible.numeric_fields_compared` stayed flagged as undocumented, correctly
— checkers scan string VALUES. A name that appears only as a key is a label.

**The seed must be set BEFORE the model is constructed.** `ablation.py` seeds at line 56
after building at line 50, so the seed controls batch sampling and never the weights. That
is the root cause of three artifacts disagreeing about `full@seed7`. Not a CUDA kernel:
under `use_deterministic_algorithms(True, warn_only=False)` torch raises nothing.
`pipeline/check_seed_before_init.py` finds this class — one hit in 329 files.
