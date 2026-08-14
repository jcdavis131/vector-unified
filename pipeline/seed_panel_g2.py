#!/usr/bin/env python3
"""First multi-seed measurement of the unified G2 gate.

    python pipeline/seed_panel_g2.py --seeds 5 7 13 21 42 99

    # a stage-2 variant, judged against the baseline's own spread
    python pipeline/seed_panel_g2.py \\
        --stage2-extra="--w-coral-centroid 0.5" --compare-to 0.7795 --compare-sd 0.0146

PASSING --stage2-extra: its value starts with a dash, so argparse will not accept
it after a space -- use the `=` form, and quote it, because the value also
contains one. Launching detached from PowerShell needs the whole command as ONE
quoted string; an -ArgumentList array splits on the inner space and argparse then
reports `unrecognized arguments: 0.5`. Both mistakes were made here before this
note existed.

audit_promotion_gates.py records that this model "has never been run at a second
seed" and that its G2 gate "passes by exactly zero: effective_rank 12.0 against
rank_nondeg_floor 12 ... with one run there is no way to tell" whether the floor
was chosen before or after seeing that number.

Both stages now take --seed, so the sweep is possible. This runs the full chain
per seed -- stage 1 then stage 2, because stage 2 consumes stage 1's trunk -- and
reports the spread of every gated quantity. A margin smaller than the spread is
not a margin.

Run it through gpu/train_local.py so each seed is containerised with the sibling
repos mounted read-only and the artifact dirs shadowed; a sweep cannot overwrite
a shipped model no matter how many seeds it tries.
"""
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path

RUNNER = Path(r"C:\Users\jcdav\herdmux\gpu\train_local.py")

PAT = {
    "best_g2": re.compile(r"best_g2=([0-9.]+)"),
    "best_epoch": re.compile(r"best_epoch=(-?\d+)"),
    "final_g2": re.compile(r"G2=([0-9.]+)\s+G3="),
    "final_rank": re.compile(r"rank=([0-9.]+)"),
    "shippable": re.compile(r"SHIPPABLE:\s*(True|False)"),
    "rank_at_best": re.compile(r"rank at best epoch =\s*([0-9.]+)"),
}


def run_seed(seed: int, epochs1: int, epochs2: int, extra: str = "") -> dict:
    cmd = [
        sys.executable, str(RUNNER), "vector-unified",
        "--entry", "pipeline/train_stage2.py",
        "--prepare", f"python -u pipeline/train_unified.py --epochs {epochs1} --seed {seed}",
        "--", "--seed", str(seed), "--epochs", str(epochs2),
    ]
    # Stage-2 variant flags. Kept OUT of the prepare step on purpose: stage 1 is
    # identical across arms, so a difference in G2 is attributable to stage 2
    # alone rather than to a trunk that started somewhere else.
    if extra:
        cmd += extra.split()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        return {"seed": seed, "ok": False, "note": f"exit {p.returncode}"}
    got = {"seed": seed, "ok": True}
    for k, rx in PAT.items():
        hits = rx.findall(out)
        got[k] = hits[-1] if hits else None
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 7, 13, 21, 42, 99])
    ap.add_argument("--epochs1", type=int, default=40)
    ap.add_argument("--epochs2", type=int, default=30)
    ap.add_argument("--stage2-extra", default="",
                    help='variant flags for stage 2, quoted, e.g. "--w-coral-centroid 0.5"')
    ap.add_argument("--compare-to", type=float, default=None,
                    help="baseline mean to compare against; prints the delta against the "
                         "baseline's own seed sd rather than eyeballing two tables")
    ap.add_argument("--compare-sd", type=float, default=None)
    a = ap.parse_args()

    print(f"seeds {a.seeds}   stage1 {a.epochs1}ep   stage2 {a.epochs2}ep", flush=True)
    if a.stage2_extra:
        print(f"stage2 variant: {a.stage2_extra}", flush=True)
    print(f"{'seed':>5} {'best_g2':>9} {'best_ep':>8} {'rank@best':>10} {'shippable':>10}",
          flush=True)

    rows = []
    for s in a.seeds:
        r = run_seed(s, a.epochs1, a.epochs2, a.stage2_extra)
        if not r.get("ok"):
            print(f"{s:>5}  FAILED: {r.get('note')}", flush=True)
            continue
        rows.append(r)
        print(f"{s:>5} {str(r['best_g2']):>9} {str(r['best_epoch']):>8} "
              f"{str(r['rank_at_best']):>10} {str(r['shippable']):>10}", flush=True)

    g2 = [float(r["best_g2"]) for r in rows if r.get("best_g2")]
    rk = [float(r["rank_at_best"]) for r in rows if r.get("rank_at_best")]
    print("\n" + "=" * 62)
    if g2:
        m, sd = statistics.fmean(g2), (statistics.stdev(g2) if len(g2) > 1 else 0.0)
        print(f"best_g2      n={len(g2)}  mean {m:.4f}  sd {sd:.4f}  "
              f"min {min(g2):.4f}  max {max(g2):.4f}")
        # The bar: majority class share + 0.10. Sports are 12,966 / 5,323 / 2,430.
        bar = 12966 / (12966 + 5323 + 2430) + 0.10
        print(f"gate bar     G2 <= {bar:.4f}   -> {sum(1 for v in g2 if v <= bar)}/{len(g2)} seeds pass")
        print(f"margin       mean {bar - m:+.4f} against a seed sd of {sd:.4f}"
              + ("  -- margin is INSIDE the noise" if abs(bar - m) < sd else ""))
    if rk:
        m2, sd2 = statistics.fmean(rk), (statistics.stdev(rk) if len(rk) > 1 else 0.0)
        print(f"rank@best    n={len(rk)}  mean {m2:.2f}  sd {sd2:.2f}  "
              f"floor 12.0 -> {sum(1 for v in rk if v >= 12.0)}/{len(rk)} clear it")

    # Judged against the BASELINE's own spread, not this arm's. A variant that
    # happens to be tightly clustered would otherwise look decisive on its own
    # sd, which says nothing about whether it moved the mean.
    if g2 and a.compare_to is not None:
        m = statistics.fmean(g2)
        delta = m - a.compare_to                      # G2 is lower-is-better
        bar_sd = a.compare_sd if a.compare_sd else 0.0
        print(f"\nvs baseline  {a.compare_to:.4f} -> {m:.4f}   delta {delta:+.4f} "
              f"(negative is better)")
        if bar_sd:
            print(f"             baseline seed sd {bar_sd:.4f}  ->  "
                  + ("BETTER, clears the baseline's own noise" if delta < -bar_sd
                     else "WORSE by more than seed noise" if delta > bar_sd
                     else "INSIDE seed noise; not an improvement"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

