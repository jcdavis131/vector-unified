#!/usr/bin/env python3
"""The same (config, seed) must not have two different results in two artifacts.

Solo personal project, no connection to employer, built with public/free-tier only

Three ablation artifacts share the config name `full` and overlapping seed sets:

    data/ablation_report.json            seeds [7, 8, 9]
    data/ablation_grl_seeds.json         seeds [7..16]
    data/ablation_coral_vicreg_seeds.json seeds [7..16]

For seeds 8 and 9 the `full` records are BIT-IDENTICAL across all three. For seed 7 they
differ in 6 of 8 fields:

    G2_sport_acc          0.6988   0.7022   0.6754
    G2_delta_vs_majority  0.073    0.0764   0.0496
    G2_rank               13.0     12.9     13.1
    G3_sil                0.6749   0.6736   0.6775
    G1_hoops_z            0.953    0.955    0.953
    G4_hit                0.96     0.96     0.959

RESOLVED BY RE-RUNNING IT. The artifacts could not say which of two explanations held --
(a) training is irreproducible at a fixed seed, or (b) `full` names a different flag set
in each file -- because they record `seeds`, `configs`, `runs` and NOTHING ELSE: no flags,
no timestamp, no code version, no checkpoint hash. So the question was settled on the GPU
instead of argued from the files. `ablation.py --seeds 1 --configs full` run three times
back to back, same code, no change between runs:

    G2_sport_acc   0.6940   0.6926   0.6827      three runs, three DISTINCT values

(a) IS TRUE. `--seed` does not pin the outcome. Root cause is specific: ablation.py sets
torch.manual_seed(seed) and np.random.default_rng(seed) -- the seed IS applied -- but sets
no determinism controls, while train_stage2.py:167 sets cudnn.deterministic = True and
:170 use_deterministic_algorithms(True, warn_only=True). ablation.py has neither.

SEEDS 8 AND 9 WERE THEREFORE NOT RE-RUN. They are bit-identical for `full` across all
three artifacts, and with nondeterminism this large three independent runs agreeing to
four decimals is not plausible. Those entries were carried over. The `full` columns in the
two 10-seed files are NOT independent evidence and must not be pooled.

Full measurement, including a retracted n=2 inference, in data/ablation_determinism.json.

This does NOT invalidate the within-file paired comparisons: those compare arms recorded
in one run set, and `full vs no_grl` at n=10 (+0.0447, p=0.0005) stands. It invalidates
treating `full` as a stable reference ACROSS files, and it means the 3-seed table's
`full` is built on one seed whose value depends on which artifact you read.

    python pipeline/check_ablation_consistency.py
    python pipeline/check_ablation_consistency.py --check   # exit 1 on an UNDECLARED
                                                            # disagreement, or a KNOWN
                                                            # entry that no longer applies

Writes: data/ablation_consistency_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "ablation_consistency_audit.json"

FILES = ["ablation_report.json", "ablation_grl_seeds.json",
         "ablation_coral_vicreg_seeds.json"]

# DISAGREEMENTS THAT ARE EXPLAINED, WITH THE EVIDENCE NAMED.
#
# This gate was registered BLOCKING while the cause of full@seed7 was unknown, and finding
# the cause changed what the gate should do. ablation.py is irreproducible at a fixed seed
# — three runs of one config at one seed gave 0.6940 / 0.6926 / 0.6827 — so these three
# artifacts CAN NEVER AGREE. Left blocking, the gate is permanently red for a condition
# nobody can fix, and a permanently red gate teaches its reader to skip the line. That is
# the same reasoning validate.py already applies to internal_prose.
#
# Deleting the check instead would be worse: it is the only thing that would notice a NEW
# disagreement appearing. So the known one is DECLARED, with its evidence, and everything
# else still fails. Same shape as data/superlative_registry.json, which clears a specific
# claim against named evidence rather than disabling the superlative check.
#
# A declaration is not a fix. It records that this disagreement is understood and that the
# artifacts predate any determinism control. It does not make the numbers agree and does
# not license quoting one file's full@seed7 against another's.
KNOWN: dict[tuple[str, int], str] = {
    ("full", 7): "ablation.py is irreproducible at a fixed seed (measured: 0.6940 / "
                 "0.6926 / 0.6827 on three consecutive runs of the same config, same "
                 "code). These records were produced by a process that does not repeat. "
                 "Evidence: data/ablation_determinism.json. Adding cudnn.deterministic + "
                 "use_deterministic_algorithms did NOT fix it; root cause still unknown.",
}

# Provenance a reader needs to tell (a) from (b) above. Absence is reported, not fixed:
# backfilling a timestamp now would be inventing provenance.
WANTED_PROVENANCE = ["built", "flags", "config_flags", "code_version", "commit",
                     "checkpoint_sha", "generated_by"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any (config, seed) disagrees across artifacts")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    docs = {}
    for f in FILES:
        p = DATA / f
        if not p.exists():
            print(f"FAIL: missing {p}", file=sys.stderr)
            return 2
        docs[f] = json.loads(p.read_text(encoding="utf-8"))

    # (config, seed) -> {file: record}
    seen: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for f, d in docs.items():
        for cfg, runs in d["runs"].items():
            for s, rec in zip(d["seeds"], runs):
                seen[(cfg, int(s))][f] = rec

    agree, disagree, known_hits = [], [], []
    for (cfg, s), byfile in sorted(seen.items()):
        if len(byfile) < 2:
            continue
        fields = sorted(set().union(*(set(r) for r in byfile.values())))
        diff = {}
        for k in fields:
            vals = {f: r.get(k) for f, r in byfile.items()}
            if len({json.dumps(v, sort_keys=True) for v in vals.values()}) > 1:
                diff[k] = vals
        rec = {"config": cfg, "seed": s, "files": sorted(byfile),
               **({"differing_fields": diff} if diff else {})}
        if not diff:
            agree.append(rec)
        elif (cfg, s) in KNOWN:
            rec["declared_reason"] = KNOWN[(cfg, s)]
            known_hits.append(rec)
        else:
            disagree.append(rec)

    missing_prov = {f: [k for k in WANTED_PROVENANCE if k not in d]
                    for f, d in docs.items()}

    out = {
        "question": "Does the same (config, seed) produce the same record in every "
                    "artifact that reports it?",
        "files": FILES,
        "shared_config_seed_pairs": len([k for k, v in seen.items() if len(v) > 1]),
        "agree": len(agree),
        "declared_known": known_hits,
        "undeclared_KNOWN_entries": [f"{c}@seed{sd}" for (c, sd) in KNOWN
                                    if not any(k["config"] == c and k["seed"] == sd
                                               for k in known_hits)],
        "disagree": disagree,
        "provenance_fields_absent": missing_prov,
        "why_this_matters": "With no flags, timestamp, or code version recorded, a "
            "disagreement cannot be attributed. Either training is not reproducible at a "
            "fixed seed (and the agreeing seeds only agree because they were carried "
            "over rather than re-run), or one config name covers several flag sets. Both "
            "change what the ablation table means.",
        "what_this_does_NOT_invalidate": "Within-file paired comparisons. Those compare "
            "arms from one run set: full vs no_grl at n=10 is +0.0447, p=0.0005, and "
            "stands. What is invalid is treating `full` as a stable reference across "
            "files.",
        "no_provenance_was_backfilled": "Absent fields are REPORTED, not filled in. "
            "Writing a timestamp now would be inventing provenance for runs whose actual "
            "conditions are unknown.",
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"  shared (config, seed) pairs: {out['shared_config_seed_pairs']}")
    print(f"  agree: {len(agree)}   declared-known: {len(known_hits)}   UNDECLARED DISAGREE: {len(disagree)}")
    for k in known_hits:
        print(f"    known  {k['config']}@seed{k['seed']} differs in "
              f"{len(k['differing_fields'])} field(s) — declared, see KNOWN")
    for d_ in disagree:
        print(f"    {d_['config']}@seed{d_['seed']} differs in "
              f"{len(d_['differing_fields'])} field(s) across {len(d_['files'])} files:")
        for k, vals in d_["differing_fields"].items():
            print(f"      {k:<22} " + "  ".join(f"{f.split('.')[0][:18]}={v}"
                                                for f, v in vals.items()))
    for f, miss in missing_prov.items():
        if len(miss) == len(WANTED_PROVENANCE):
            print(f"    {f}: NO provenance fields at all "
                  f"(none of {', '.join(WANTED_PROVENANCE)})")
    print(f"\nwrote {OUT}")
    stale = out["undeclared_KNOWN_entries"]
    if stale:
        print(f"  KNOWN declares {stale} which no longer disagrees — remove "
              f"the entry; a declaration outliving its defect hides the next one",
              file=sys.stderr)
    if args.check and (disagree or stale):
        print(f"CHECK FAILED: {len(disagree)} undeclared disagreement(s), "
              f"{len(stale)} stale KNOWN entr(ies)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
