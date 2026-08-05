# Agent contacts — who exists, what they can do, and how a message actually reaches them

Every name below was READ OFF `COORDINATION.md` on origin/master, with the line it appears
on. Nothing here is inferred from a name that sounds plausible. If an agent is not in this
file, it is because no evidence of it was found, not because it does not exist.

**Regenerate the evidence:** `git show origin/master:COORDINATION.md` and read the Agent
column. Line numbers below are from the 46-line version current at the time of writing and
will drift — the names and branch conventions are the stable part.

## The channel situation, which is the first thing to know

**`COORDINATION.md` is the only channel, and it is LOSSY IN ONE DIRECTION.**

It is mirrored from Hatch's `bundles/coordination/active-tasks.md`. Every heartbeat commit
checked (`5f10413`, `0001957`, `d1ba9a4`, `32f8313`, `dcebea5`) touches `COORDINATION.md`
and nothing else, and the mirror **overwrites the whole file**. `dcebea5`'s message says
"rebase on 7367a7d" — 7367a7d was a local commit adding a result row, and the row did not
survive.

Consequences, both observed rather than predicted:

- A row written locally can be gone within the hour. It happened twice to the same row.
- While it was gone, three heartbeats (03:43 / 04:13 / 04:43 CDT 2026-08-05) recorded
  "no local GPU measured G2 ... predicted 0.64-0.65" — a measurement existed on a pushed
  branch and the coordination system reported its absence.
- **There is no local -> Hatch path.** An agent on this box can write `COORDINATION.md`,
  but cannot write `active-tasks.md`, so the next mirror wins.

**So: put anything that must survive in a file that is NOT `COORDINATION.md`, and point the
board row at it.** `LOCAL_GPU_G2_RESULT.md` and this file exist for that reason. A board row
is a notification; a repo file is the record.

## Contacts

| Name | Runs on | Owns (evidence: COORDINATION.md line) | Branch convention | Can it train? |
|---|---|---|---|---|
| `Scout` | Hatch VM | dottie GRPO (L21), vector-hub chimera+provenance (L22), vector-unified G3/G4 (L23) | `scout/<area>` | **No** |
| `Scout-lane1` | Hatch VM | vector-* honesty pass, all four repos (L18) | `scout/vector-honesty-night1` | **No** |
| `Scout-lane2` | Hatch VM | dottie + scout-cli v0.8 polish (L8) | `scout/dottie-cli-night2` | **No** |
| `Scout-push` | Hatch VM | pushing honesty branches to origin (L20) | `scout/vector-honesty-night1` | **No** |
| `Orchestrator` | Hatch VM | scout-cli harness, pitch/equities polish cycles (L37-41) | `scout/<repo>-polish-*` | **No** |
| `Heartbeat` | Hatch VM | coordination sweep; clears rows idle >4h; runs the mirror (L25, L43-46) | `heartbeat` | **No** |
| `LOCAL-GPU` | this box | handoff lane: unified G2, hoops v6, gridiron nflverse (L34-36) | `local/<repo>-<task>` | **Yes** |
| `Claude-Local` | this box | whoever is working the local box now; pitch verify (L9), unified G2 (L33) | `local/<...>` or `master (ff)` | **Yes** |


### Host capabilities, stated so a router does not have to discover them

| Host | Has | Cannot reach |
|---|---|---|
| Hatch VM | network, the `bundles/*` tree | **no torch** — the board says "do NOT pip torch"; heavy trains OOM |
| this box | RTX 4080, torch + CUDA, 20+ training runs done this lane | **no `bundles/*`** — a handoff citing a Hatch-only patch is not actionable here, which already cost one lane |

### Git identity does NOT match the board name

Measured with `git log --format="%an <%ae>" -60` in each repo. This matters because it is
the only way to check a "done" row against the tree — and the two do not line up.

| Repo | dominant author | `Scout <jcdavis131@gmail.com>` present? |
|---|---|---|
| vector-unified | `camml210 <camdavis131@gmail.com>` (56/60) | **NO — zero Scout commits** |
| vector-hoops | `camml210 <camdavis131@gmail.com>` (54/60) | yes (3 in window) |
| vector-pitch | `camml210 <camdavis131@gmail.com>` (15) | yes (3) |
| vector-equities | `camml210 <camdavis131@gmail.com>` (25) | yes (6) |

- Work signed **`Claude-Local`** on the board is authored **`camml210 <camdavis131@gmail.com>`**.
  There is no board name in the commit metadata at all.
- **`Scout` has never authored a commit in vector-unified**, although it holds board rows
  claiming vector-unified lanes. Its rows there describe work pushed from elsewhere.
- Two distinct author names share one email: `Scout <jcdavis131@gmail.com>` and
  `Cameron Davis <jcdavis131@gmail.com>`. Email is not a discriminator.

So: **do not verify a row by looking for its agent name in git log.** Match on branch
namespace (`scout/*` vs `local/*`) and the SHA the row cites.

### Addressing, as actually practised

Every item below was found in a file, not inferred:

- **Recipients are named by CAPABILITY, not by name.** `LOCAL_GPU_HANDOFF.md` line 3 reads
  `> **For:** Local non-Hatch agent (Cursor / Claude Code / etc) with GPU`.
- **A reply is a new file.** Claude-Local's answer to that handoff was a separate document,
  `COORDINATION_LOCAL_GPU_BLOCKER.md`, whose "To unblock" section asks Scout to push a
  branch — while naming no recipient and having no delivery guarantee. It sits in a repo
  and waits.
- **Agents refer to each other in the THIRD PERSON, to the operator.** "that is their lane
  and their timeline"; "zero .pt checkpoints — Scout's lane". Nobody writes *to* anyone.
- **The transport is `git push`.** Latency is however long until the other side next pulls.
  There is no notification of any kind.

### The capability split is the load-bearing column

`COORDINATION.md`'s own "Free lanes" section says: *"LOCAL GPU heavy trains (OOM in Hatch)
— see handoff table above, do NOT pip torch."*

That is the whole reason the handoff table exists. **Hatch agents cannot train.** Anything
needing a GPU has to be handed to `LOCAL-GPU` / `Claude-Local` and cannot be done by
whoever noticed it was needed. Conversely, this box has no access to Hatch's bundles, so
work described as living under `bundles/` cannot be read from here — a handoff referencing
a Hatch-only patch is not actionable locally, which has already happened once.

### Before handing work to someone

1. **Check the board first, claim before editing.** `COORDINATION.md`'s own rules: add your
   row before editing, keep main green, `*.candidate.json` before promotion, log even a
   no-op, clear your row when done.
2. **Match the capability.** GPU work only to the two local entries. Do not hand a training
   task to a `Scout` lane; it will bounce on OOM.
3. **Name the artifact, not the outcome.** A row saying "measured G2" is unusable if the
   row is mirrored away. Point at a committed file and a branch SHA.
4. **Assume your row may vanish.** If it matters, it goes in a repo file too.

## What this file does NOT establish

- **No agent-to-agent messaging exists.** There is no inbox, no queue, no mention channel.
  Every "contact" here is reachable only by writing a row that the recipient may or may not
  read before the next mirror. This file documents the addressing scheme; it does not
  create a transport.
- **Identity is a convention, not an authentication.** Anyone can write any name in the
  Agent column. Git authorship on this box is a single local user, so the board name is
  self-asserted.
- **`Scout` vs `Scout-lane1` vs `Scout-lane2` vs `Scout-push` may be one process or four.**
  They hold different branches at overlapping times, which is consistent with either. The
  evidence does not distinguish them and they are listed separately rather than merged on a
  guess.
- **`LOCAL-GPU` vs `Claude-Local`** are both this box. `LOCAL-GPU` is how the handoff table
  addresses the lane; `Claude-Local` is how the agent working it signs. Treat them as one
  recipient with two labels rather than two agents.
