#!/usr/bin/env python3
"""Hill-climb the tennis MTNN's hyperparameters. One knob at a time from the base config.

The base config was a first guess: dim 32, tower_width 16, 400 epochs, lr 3e-3, temp 0.1,
dropout 0.1. It scored 0.0783 (sd 0.0061 over 5 seeds).

MDE for a 5-vs-5 comparison at sd 0.0061 is about 2.78 * 0.0061 * sqrt(2/5) = 0.0107, so
only changes moving the mean by more than ~0.011 are resolvable. Anything smaller gets
reported as "inside the floor" rather than as an improvement.

The trainer writes data/tennis_mtnn_report.json and the embedding on every run; both are
backed up before the sweep and restored after, so the gate artifact is not left holding a
sweep result.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from portable_paths import ESTATE  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[1]

# A FRESH TEMP DIR PER RUN, not a fixed scratch path. This was hardcoded to
# .../Temp/claude/C--Users-jcdav/be69d382-ce38-4d23-b6d1-d92c62546c02/scratchpad — a
# SESSION id, baked into a tracked file. Two ways that goes wrong, and the second is the
# one that matters:
#
#   the directory is gone      shutil.copy2 at the backup step raises, and the sweep dies
#                              before it retrains anything. Loud, and safe.
#   the directory still holds  the restore below copies a *.sweepbak from some earlier
#   stale *.sweepbak files     session back over the gate artifacts, silently reverting
#                              them to whatever that run left. Quiet, and not safe.
#
# These backups only have to survive within one invocation — the restore happens in this
# same process — so a fresh mkdtemp is strictly safer and cannot inherit another run's
# state. It is deliberately NOT cleaned up on exit: if the sweep dies mid-run the backups
# are the only copy of the gate artifacts, and the path is printed below.
SC = Path(tempfile.mkdtemp(prefix="tennis_sweep_"))

# The CUDA venv, resolved from the estate root rather than one laptop. Falls back to the
# interpreter running this file, and SAYS SO, rather than dying.
_VENV = ESTATE / "vector-hoops/pipeline/.venv/Scripts/python.exe"
PY = str(_VENV) if _VENV.exists() else sys.executable
REPORT = REPO / "pipeline" / "data" / "tennis_mtnn_report.json"
REPORT2 = REPO / "data" / "tennis_mtnn_report.json"
EMB = REPO / "pipeline" / "data" / "tennis_mtnn_embedding.npz"

BASE = {"dim": 32, "tower-width": 16, "epochs": 400, "lr": 3e-3,
        "temp": 0.1, "dropout": 0.1}

VARIANTS = [
    ("base", {}),
    ("dim64", {"dim": 64}),
    ("dim16", {"dim": 16}),
    ("tw32", {"tower-width": 32}),
    ("ep800", {"epochs": 800}),
    ("ep200", {"epochs": 200}),
    ("temp05", {"temp": 0.05}),
    ("temp20", {"temp": 0.2}),
    ("drop0", {"dropout": 0.0}),
    ("lr1e-3", {"lr": 1e-3}),
    ("dim64_tw32_ep800", {"dim": 64, "tower-width": 32, "epochs": 800}),
]

for f in (REPORT2, EMB):
    if f.exists():
        shutil.copy2(f, SC / (f.name + ".sweepbak"))

rows = []
for name, override in VARIANTS:
    cfg = {**BASE, **override}
    argv = [PY, "pipeline/train_tennis_mtnn.py"]
    for k, v in cfg.items():
        argv += [f"--{k}", str(v)]
    p = subprocess.run(argv, cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode not in (0, 1):
        print(f"  {name:20} FAILED rc={p.returncode}")
        print((p.stderr or "")[-400:])
        continue
    r = json.loads(REPORT2.read_text(encoding="utf-8"))
    rows.append({"name": name, "cfg": cfg, "mean": r["mtnn_mean"], "sd": r["mtnn_sd"],
                 "per_seed": r["mtnn_per_seed"], "over_bar": r["seeds_over_bar"]})
    print(f"  {name:20} mean {r['mtnn_mean']:.4f}  sd {r['mtnn_sd']:.4f}  "
          f"over_bar {r['seeds_over_bar']}", flush=True)

for f in (REPORT2, EMB):
    b = SC / (f.name + ".sweepbak")
    if b.exists():
        shutil.copy2(b, f)

base = next((r for r in rows if r["name"] == "base"), None)
if base:
    sd = base["sd"]
    mde = 2.78 * sd * (2 / 5) ** 0.5
    print(f"\n  base mean {base['mean']:.4f} sd {sd:.4f} -> 5v5 MDE {mde:.4f}\n")
    print(f"  {'variant':20} {'mean':>8} {'delta':>9}  verdict")
    for r in sorted(rows, key=lambda x: -x["mean"]):
        d = r["mean"] - base["mean"]
        v = "inside floor" if abs(d) <= mde else ("BETTER" if d > 0 else "worse")
        print(f"  {r['name']:20} {r['mean']:>8.4f} {d:>+9.4f}  {v}")

# THE RESULT GOES IN THE REPO, the backups stay in temp. It used to write here into the
# same session scratch dir, which is why check_artifact_freshness.py exempts
# tennis_hparam_sweep.json with "the harness that produced them lives in a scratch
# directory and is not part of the repo". That reason no longer holds for this file.
#
# NOT claiming this is the same artifact as data/tennis_hparam_sweep.json — that one was
# copied in by hand and its schema has not been compared. Different name, its own file.
OUT = REPO / "data" / "tennis_sweep.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
print(f"\nwrote {OUT}  (gate artifact restored from backup)")
print(f"backups kept at {SC}")
