#!/usr/bin/env python3
"""A seed set AFTER the model is built controls nothing. Find that ordering. READ-ONLY.

Solo personal project, no connection to employer, built with public/free-tier only

vector-unified's ablation.py produced three different results for one config at one fixed
seed — 0.6940 / 0.6926 / 0.6827 — and three shipped artifacts disagree about full@seed7
because of it. The cause was not a CUDA kernel. Two hypotheses were tested and falsified
first (nn.Embedding atomicAdd backward; warn_only/CUBLAS_WORKSPACE_CONFIG), and under
`use_deterministic_algorithms(True, warn_only=False)` torch raised no error at all.

It was an ORDERING BUG:

    model = UnifiedTrunk(...)        <- weights drawn from whatever RNG state exists
    opt   = AdamW(...)
    torch.manual_seed(seed)          <- too late; only batch sampling is controlled

vector-realty's train_mtnn.py and vector-unified's train_stage2.py seed FIRST, and both
reproduce bit-identically on the same GPU with no determinism controls at all. That
contrast is what located it.

WHAT THIS CHECKS. Per function: does a seeding call appear AFTER something that consumes
randomness to initialise parameters? Seeding calls are torch.manual_seed,
torch.cuda.manual_seed*, np.random.seed and np.random.default_rng. Initialisers are
recognised by construction of a name ending in a model-ish suffix, a `.to(device)` chain,
or an optimizer constructor.

WHY THIS IS A HEURISTIC AND SAYS SO. It cannot know that `UnifiedTrunk(...)` draws random
weights and some other constructor does not; it matches on shape. A function that seeds
late but builds nothing random is a FALSE POSITIVE, and one that initialises through a
helper this does not recognise is a MISS. Both are stated rather than implied, and the
check is report-only for that reason — a rule that fires on a name it does not understand
is what teaches a reader to skip the report.

    python pipeline/check_seed_before_init.py
    python pipeline/check_seed_before_init.py --estate   # scan sibling repos too
    python pipeline/check_seed_before_init.py --check    # exit 1 on any late seed

Writes: data/seed_order_audit.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ESTATE = ROOT.parent
OUT = ROOT / "data" / "seed_order_audit.json"

SIBLINGS = ["vector-hoops", "vector-gridiron", "vector-pitch", "vector-equities",
            "vector-realty"]

SEED_CALLS = {"manual_seed", "manual_seed_all", "seed", "default_rng"}
OPTIMIZERS = {"AdamW", "Adam", "SGD", "RMSprop", "Adagrad"}
MODELISH = ("MTNN", "Net", "Model", "Trunk", "Encoder", "Tower", "Module")


def _is_seed(node: ast.Call) -> bool:
    f = node.func
    if isinstance(f, ast.Attribute) and f.attr in SEED_CALLS:
        # torch.manual_seed / np.random.seed / np.random.default_rng / rng.manual_seed
        return True
    return False


def _is_init(node: ast.Call) -> bool:
    f = node.func
    if isinstance(f, ast.Name):
        n = f.id
        return n in OPTIMIZERS or any(n.endswith(s) for s in MODELISH)
    if isinstance(f, ast.Attribute):
        if f.attr in OPTIMIZERS:
            return True
        if f.attr == "to":                      # model(...).to(device)
            return True
    return False


def scan(path: Path) -> list[dict]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return []
    out = []
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        seeds, inits = [], []
        for n in ast.walk(fn):
            if isinstance(n, ast.Call):
                if _is_seed(n):
                    seeds.append(n.lineno)
                elif _is_init(n):
                    inits.append(n.lineno)
        if not seeds or not inits:
            continue
        first_init, first_seed = min(inits), min(seeds)
        if first_init < first_seed:
            out.append({"function": fn.name, "first_init_line": first_init,
                        "first_seed_line": first_seed,
                        "gap_lines": first_seed - first_init,
                        "why": "something that initialises parameters runs before the "
                               "first seeding call, so the seed cannot control it"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--estate", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    roots = [ROOT]
    if args.estate:
        roots += [ESTATE / s for s in SIBLINGS if (ESTATE / s).is_dir()]

    findings, n_files = [], 0
    for r in roots:
        for f in sorted((r / "pipeline").glob("*.py")) if (r / "pipeline").is_dir() else []:
            n_files += 1
            for hit in scan(f):
                hit["file"] = str(f.relative_to(ESTATE)).replace("\\", "/")
                findings.append(hit)

    out = {
        "question": "Is any seeding call made AFTER something that initialises parameters?",
        "why": "ablation.py gave 0.6940 / 0.6926 / 0.6827 for one config at one seed. Not a "
               "CUDA kernel — torch raised nothing under strict determinism. The seed was "
               "set after UnifiedTrunk and AdamW were constructed, so it controlled batch "
               "sampling and never the weights.",
        "controls_that_do_it_right": [
            "vector-realty/pipeline/train_mtnn.py — seeds, then builds; reruns bit-identical",
            "vector-unified/pipeline/train_stage2.py — same order; 34 of 37 report fields "
            "bit-identical across reruns",
        ],
        "heuristic_and_its_limits": "Matches on SHAPE, not semantics. It cannot know which "
            "constructors draw random weights, so a function that seeds late but builds "
            "nothing random is a FALSE POSITIVE, and one initialising through an "
            "unrecognised helper is a MISS. Report-only for that reason.",
        "files_scanned": n_files,
        "findings": findings,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"  scanned {n_files} pipeline file(s) across {len(roots)} repo(s)")
    print(f"  late-seed findings: {len(findings)}")
    for x in findings:
        print(f"    {x['file']}::{x['function']}  init@{x['first_init_line']} "
              f"seed@{x['first_seed_line']}  (+{x['gap_lines']} lines)")
    print(f"\nwrote {OUT}")
    if args.check and findings:
        print(f"CHECK FAILED: {len(findings)} function(s) seed after initialising",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
