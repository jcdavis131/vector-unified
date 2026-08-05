# Scheduling the validation sweep so it survives this session

## The problem with what is currently scheduled

Three cron jobs exist right now — a validation sweep every 4h, a drift watch every 6h, and
a candidate-only dataset expansion daily. **They are session-only.** They live in memory
inside one Claude session, die when it exits, and auto-expire after 7 days regardless.

"Always validating" cannot rest on that. This file is how to make the read-only half
durable. It has NOT been installed — registering a scheduled task modifies the machine, and
that is the operator's call.

## What to schedule

`scripts/validation_sweep.py`. It is a plain script: deterministic, no model in the loop,
no tokens. It runs five read-only checks and appends one row to
`data/validation_sweep_log.md`.

    python scripts/validation_sweep.py --quiet --check

Exit 0 = every check passed. Exit 1 = at least one failed, and the row records which.

**Only the sweep is a script.** The dataset-expansion job genuinely needs judgement — which
free source, what the new fields mean, whether the eval actually won — and stays an agent
prompt. Do not try to cron that as a script.

## Install (Windows Task Scheduler)

Run these yourself; they modify the machine.

    schtasks /Create /TN "vector-validation-sweep" /SC HOURLY /MO 4 /ST 00:13 ^
      /TR "C:\Users\jcdav\vector-hoops\pipeline\.venv\Scripts\python.exe C:\Users\jcdav\vector-unified\scripts\validation_sweep.py --quiet" ^
      /RL LIMITED /F

Check it, run it once by hand, then read the log:

    schtasks /Query /TN "vector-validation-sweep" /V /FO LIST
    schtasks /Run   /TN "vector-validation-sweep"
    type C:\Users\jcdav\vector-unified\data\validation_sweep_log.md

Remove it:

    schtasks /Delete /TN "vector-validation-sweep" /F

`/RL LIMITED` on purpose — this needs no elevation, and a scheduled task that runs as
admin to read JSON files is a larger blast radius than the job deserves.

## What the sweep deliberately does NOT run, and why

Each exclusion is a thing that went wrong once.

| Excluded | Because |
|---|---|
| `build_*.py`, `probe_*.py`, `acquire_*.py` | They write artifacts with **no flag at all**. A checker that executed documented commands rewrote ten artifacts here AND stripped a `CORRECTED` marker from `vector-hoops/pipeline/seed_floor.json`, taking three green gates red. |
| any trainer (`train_*.py`, `ablation.py`) | Overwrites the shipped checkpoint — `sport_acc 0.6851`, ckpt `b055641c03760624`. |
| `validate.py` in full | It runs `train_tennis_mtnn.py --check`, which **retrains tennis** and moves `data/tennis_mtnn_report.json` off the value dumbmodel.com publishes. That is how the current 6-value `cited_fields` disagreement arose. |
| `check_gate_inputs_tracked.py` | Clones the repo and runs `validate.py` twice inside the clone. Minutes per invocation. |

Adding a check to `CHECKS` in the sweep means running it once and diffing `git status`
**across every repo in the estate** first. "It looks read-only" is not the test; that
assumption is exactly what caused the sibling damage.

## Current state, so a first run is not mistaken for a regression

As of 2026-08-05 the sweep reports **3/5 pass**. Both failures are known and neither is
caused by the sweep:

- `field_semantics` — was failing on a real defect it found in `stage2_seed_floor.json`
  (stored `sd 0.0044` beside values whose own sd is `0.0043`). Fixed at source; now clean
  across 416 artifacts estate-wide, with 37 acknowledged blind spots reported every run.
- `cited_fields` — 6 published tennis values disagree with their artifact (page
  `mtnn_mean=0.1168`, artifact `0.1157`). The finding is that the published number does not
  survive a re-run. Not repairable by editing either side.

## The cron that was wrong, and why it is worth writing down

The first scheduled sweep instructed three commands, one of them
`python pipeline/validate.py --offline`. That **contradicted its own hard rule** ("do NOT
run any trainer"): `validate.py` registers `tennis_mtnn` as `train_tennis_mtnn.py --check`,
and that arm retrains. Every sweep therefore moved `data/tennis_forward_report.json` —
`gain_mean_across_cuts` 0.0949 -> 0.0863, observed on two separate runs — and every sweep
had to revert it by hand.

A schedule whose instructions violate its own safety rule is worse than no schedule: it
runs unattended, and the damage is silent because the next run's diff looks like ordinary
churn.

Replaced with `scripts/validation_sweep.py --check` plus
`pipeline/check_gate_inputs_tracked.py`. Verified after the change: a full sweep modifies
`data/validation_sweep_log.md` and nothing else.

**If you add a check to the sweep, run it once and diff `git status` across every repo in
the estate before trusting it.** "It looks read-only" is exactly the assumption that caused
both the sibling-repo damage and this one.

## The operator dashboard

`tools/dashboard/server.py` — one screen, no scrolling, 10s auto-refresh. Launch:

    python tools/dashboard/server.py

Binds `127.0.0.1:8000` and falls back to 8001-8019 if that is taken, printing which port it
got. It coexists with Docker Desktop holding `0.0.0.0:8000`.

**Run exactly one.** A second instance silently takes the next port while the first keeps
serving the old code on 8000 — that happened during development and produced a stale page
that looked current. If the board shows fewer items than you expect, check for a second
server before believing it.

Everything is read at request time from git and the artifacts, cached 8s. A source it cannot
read makes its tile **disappear** rather than showing a last-good value, so a stale
dashboard is not a failure mode. That also means tiles are absent until the artifact behind
them exists: the gate tile needs `data/gate_inputs_tracked_audit.json`, which
`pipeline/check_gate_inputs_tracked.py` writes and which is not carried on master.

It reads only. It runs no checker, no builder and no trainer.

### If an agent is launching it: DETACH, or it dies every few minutes

An agent starting `python server.py` as a background command puts the server inside the
harness's process tree, and the harness reaps that tree. It was killed and hand-restarted
SEVEN times in one session before anyone tried anything different.

`Start-Process` detaches it instead — no scheduled task, no machine change, survives the
harness:

    $pyw = "C:\Users\jcdav\vector-hoops\pipeline\.venv\Scripts\pythonw.exe"
    $srv = "C:\Users\jcdav\vector-unified\tools\dashboard\server.py"
    Start-Process -FilePath $pyw -ArgumentList $srv `
      -WorkingDirectory "C:\Users\jcdav\vector-unified\tools\dashboard" `
      -WindowStyle Hidden -PassThru

`pythonw.exe` so there is no console window. Check before starting, every time — a detached
server does NOT die with the session, so the duplicate-instance trap below is now easier to
hit, not harder:

    curl -s -o NUL -w "%{http_code}" http://localhost:8000/

Stop it by PID: `Stop-Process -Id <pid>`. This is the interim answer; the logon task below
is still the durable one, because a detached process does not survive a reboot.

### Keeping the dashboard up across reboots

The sweep above is a task that runs and exits. The dashboard is a server that must stay
running, so it wants `ONLOGON`, not an interval. Run this yourself; it modifies the machine.

    schtasks /Create /TN "vector-dashboard" /SC ONLOGON ^
      /TR "C:\Users\jcdav\vector-hoops\pipeline\.venv\Scripts\pythonw.exe C:\Users\jcdav\vector-unified\tools\dashboard\server.py" ^
      /RL LIMITED /F

`pythonw.exe`, not `python.exe`, so it runs without a console window.

    schtasks /Run    /TN "vector-dashboard"      # start now, without logging out
    schtasks /Delete /TN "vector-dashboard" /F   # remove

**The duplicate-instance trap applies here more than anywhere.** A logon task plus a
manually started server means two processes: the first holds 8000, the second silently
takes 8001, and the page you are looking at is served by whichever won. That produced a
board during development that looked entirely current and was not. Check before starting
one by hand:

    curl -s -o NUL -w "%{http_code}" http://localhost:8000/

`200` means one is already serving and you should not start another.

**It does not survive a crash.** `ONLOGON` restarts it at the next logon, not when the
process dies. If the board goes blank mid-session,
`schtasks /Run /TN "vector-dashboard"` brings it back.
