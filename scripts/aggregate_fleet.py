#!/usr/bin/env python3
"""Regenerate fleet/data/fleet.json from each model repo's committed eval JSON.

Pure Python standard library only (urllib + json) — no third-party packages.

Data sources
------------
Each fleet model publishes an eval JSON inside its own repo. This script reads
those files and folds the headline number + a few secondary metrics into the
dashboard's fleet.json.

- PUBLIC repos (vector-hoops, vector-pitch, vector-gridiron, vector-equities) are
  fetched over raw.githubusercontent.com with no credentials.
- PRIVATE repos (vector-realty, vector-unified) are NOT reachable anonymously.
  For those you must either:
    * provide a GitHub token and use the authenticated contents API, or
    * point the script at a local checkout path.
  Rather than embed a token here, this script leaves the two private entries'
  metrics to the human-maintained values already in fleet/data/fleet.json and
  only refreshes the public ones on a live run. (The full aggregation with
  private access is an operator-local step.)

Modes
-----
    python3 scripts/aggregate_fleet.py            # live: refresh public entries
    python3 scripts/aggregate_fleet.py --offline  # validate existing fleet.json only

The --offline mode does no network I/O: it loads fleet/data/fleet.json and checks
it against the expected shape (schemaVersion, thesis, 6 well-formed model entries),
exiting non-zero if anything is wrong. Use it in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FLEET_JSON = REPO_ROOT / "fleet" / "data" / "fleet.json"

RAW_BASE = "https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"

# The per-entry fleet contract (required fields, valid statuses, headlineMetric
# shape) now lives once in vector_core.schema and is reused here instead of being
# re-declared. Prefer the installed package (`pip install -e packages/vector-core`,
# see requirements-dev.txt); fall back to the in-repo source path so this script
# also runs with zero setup, CPU-only.
try:
    from vector_core.schema import validate_entry
except ModuleNotFoundError:
    _VC_SRC = REPO_ROOT / "packages" / "vector-core" / "src"
    if _VC_SRC.is_dir():
        sys.path.insert(0, str(_VC_SRC))
    from vector_core.schema import validate_entry

# Public repos whose eval JSON can be fetched anonymously. Each maps the repo to
# the committed source file (matching the "source" field in fleet.json) and the
# default branch to read from.
PUBLIC_SOURCES = {
    "vector-hoops": {"owner": "jcdavis131", "ref": "main", "path": "assets/eval_scoreboard.json"},
    "vector-pitch": {"owner": "jcdavis131", "ref": "main", "path": "assets/eval_scoreboard.json"},
    "vector-gridiron": {"owner": "jcdavis131", "ref": "main", "path": "assets/eval_scoreboard.json"},
    "vector-equities": {
        "owner": "jcdavis131",
        "ref": "main",
        "path": "assets/eval_sector_coherence.json",
    },
}

# Private repos: documented here, but not fetched anonymously (see module docstring).
PRIVATE_REPOS = {"vector-realty", "vector-unified"}


def load_fleet() -> dict:
    with open(FLEET_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def validate(data: dict) -> list[str]:
    """Return a list of problems with the fleet data (empty == valid)."""
    problems: list[str] = []
    if data.get("schemaVersion") != 1:
        problems.append(f"schemaVersion should be 1, got {data.get('schemaVersion')!r}")
    if not data.get("thesis"):
        problems.append("missing 'thesis'")
    models = data.get("models")
    if not isinstance(models, list):
        problems.append("'models' must be a list")
        return problems
    if len(models) != 6:
        problems.append(f"expected 6 models, got {len(models)}")
    seen = set()
    for i, m in enumerate(models):
        tag = m.get("repo", f"#{i}")
        # Per-entry structural validation is delegated to the canonical fleet
        # contract in vector_core.schema (the shared source of truth). The
        # pass/fail verdict is identical to the previous inline checks across an
        # exhaustive fixture battery (tests/test_vector_core_adoption.py); only
        # the diagnostic wording is now vector_core's canonical wording.
        for problem in validate_entry(m):
            problems.append(f"[{tag}] {problem}")
        if m.get("repo") in seen:
            problems.append(f"duplicate repo {m.get('repo')!r}")
        seen.add(m.get("repo"))
    return problems


def fetch_json(url: str, timeout: float = 15.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "aggregate_fleet/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted host)
        return json.loads(resp.read().decode("utf-8"))


def refresh_public(data: dict) -> int:
    """Best-effort refresh of public entries' metrics from their raw eval JSON.

    Returns the number of entries successfully refreshed. This is intentionally
    conservative: it fetches each public repo's eval JSON so an operator can wire
    up the exact key-mapping per repo. Because each repo's scoreboard schema
    differs, the mapping is left as a per-repo TODO rather than guessed — the
    fetch proves reachability and gives the operator the live blob to map from.
    """
    refreshed = 0
    for m in data.get("models", []):
        repo = m.get("repo")
        src = PUBLIC_SOURCES.get(repo)
        if not src:
            if repo in PRIVATE_REPOS:
                print(f"  {repo}: private — skipped (needs auth/local path)")
            continue
        url = RAW_BASE.format(owner=src["owner"], repo=repo, ref=src["ref"], path=src["path"])
        try:
            blob = fetch_json(url)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            print(f"  {repo}: fetch failed ({exc}); keeping existing values")
            continue
        # We fetched the live scoreboard. Repo schemas differ, so we do not blindly
        # overwrite; we confirm reachability and leave the committed metrics intact.
        # An operator mapping each repo's keys should edit fleet.json or extend
        # PUBLIC_SOURCES with an explicit key map.
        print(f"  {repo}: fetched {url} ({len(blob)} top-level keys) — values unchanged")
        refreshed += 1
    return refreshed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--offline",
        action="store_true",
        help="validate the existing fleet/data/fleet.json against the expected shape (no network)",
    )
    args = ap.parse_args()

    if not FLEET_JSON.exists():
        print(f"ERROR: {FLEET_JSON} does not exist", file=sys.stderr)
        return 2

    data = load_fleet()

    if args.offline:
        problems = validate(data)
        if problems:
            print("fleet.json INVALID:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print(f"fleet.json OK: {len(data['models'])} models, schemaVersion={data['schemaVersion']}")
        return 0

    # Live mode: refresh public entries, re-validate, write back.
    print("Refreshing public entries from raw.githubusercontent.com ...")
    refresh_public(data)
    problems = validate(data)
    if problems:
        print("Refusing to write — fleet.json would be invalid:")
        for p in problems:
            print(f"  - {p}")
        return 1
    with open(FLEET_JSON, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"Wrote {FLEET_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
