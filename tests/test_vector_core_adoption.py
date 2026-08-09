"""Equivalence gate for the first adoption of vector-core in vector-unified.

The swap: ``scripts/aggregate_fleet.py`` no longer re-implements the per-entry
fleet-report contract inline; it delegates to ``vector_core.schema.validate_entry``,
the canonical fleet-wide source of truth.

This file is the gate that made that swap safe. It proves, across an exhaustive
fixture battery, that the *valid/invalid verdict* of the shared validator is
identical to vector-unified's previous inline checks — so the ``--offline`` CI
exit code cannot change — and it locks in the one divergence that had to be fixed
in vector_core first (a present-but-``None`` headlineMetric must be rejected).

All CPU-only, no torch, no sklearn.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import aggregate_fleet  # noqa: E402
from vector_core.schema import validate_entry  # noqa: E402


# --------------------------------------------------------------------------- #
# Frozen copy of aggregate_fleet's ORIGINAL inline per-entry checks (pre-swap).
# Only the verdict-relevant logic is reproduced; message wording is irrelevant
# to the gate, which compares valid vs invalid.
# --------------------------------------------------------------------------- #
_OLD_VALID_STATUS = {"production", "shipped", "wip", "blocked"}
_OLD_REQUIRED = ("repo", "domain", "embeddingDim", "status", "headlineMetric")


def _old_entry_problems(m: dict) -> list[str]:
    problems: list[str] = []
    for f in _OLD_REQUIRED:
        if f not in m:
            problems.append(f"missing {f}")
    if m.get("status") not in _OLD_VALID_STATUS:
        problems.append("bad status")
    if not isinstance(m.get("embeddingDim"), int):
        problems.append("bad dim")
    hm = m.get("headlineMetric")
    if not isinstance(hm, dict) or "name" not in hm or "value" not in hm:
        problems.append("bad headlineMetric")
    return problems


_BASE = {
    "repo": "r",
    "domain": "d",
    "embeddingDim": 64,
    "status": "shipped",
    "headlineMetric": {"name": "x", "value": 0.5},
}


def _battery() -> list[dict]:
    fixtures: list[dict] = [copy.deepcopy(_BASE)]
    # each single required field removed
    for k in list(_BASE):
        f = copy.deepcopy(_BASE)
        del f[k]
        fixtures.append(f)
    # status variants (valid, invalid, empty, None)
    for s in ["production", "shipped", "wip", "blocked", "bogus", "", None]:
        f = copy.deepcopy(_BASE)
        f["status"] = s
        fixtures.append(f)
    # embeddingDim variants (int, bool, float, str, None, list)
    for v in [0, 64, True, False, 3.5, "64", None, [1]]:
        f = copy.deepcopy(_BASE)
        f["embeddingDim"] = v
        fixtures.append(f)
    # headlineMetric variants
    for hm in [
        {"name": "x", "value": 1},
        {"name": "x"},
        {"value": 1},
        {},
        "str",
        None,
        123,
        {"name": "x", "value": 1, "baseline": 2, "source": "s"},
    ]:
        f = copy.deepcopy(_BASE)
        f["headlineMetric"] = hm
        fixtures.append(f)
    # extra unknown field, empty, multi-corrupt
    f = copy.deepcopy(_BASE)
    f["weird"] = 1
    fixtures.append(f)
    fixtures.append({})
    fixtures.append(
        {"repo": "r", "status": "bogus", "embeddingDim": "x", "headlineMetric": "no"}
    )
    return fixtures


def test_vector_core_importable():
    import vector_core

    assert vector_core.__version__
    # numpy-safe API present without importing torch
    assert hasattr(vector_core, "validate_entry")


@pytest.mark.parametrize("entry", _battery())
def test_validate_entry_verdict_matches_repo_inline(entry):
    """The shared validator agrees with the old inline checks on valid vs invalid."""
    old_bad = bool(_old_entry_problems(entry))
    new_bad = bool(validate_entry(entry))
    assert old_bad == new_bad, (
        f"verdict diverged for {entry!r}: "
        f"old={'INVALID' if old_bad else 'valid'} new={'INVALID' if new_bad else 'valid'}"
    )


def test_headline_metric_none_is_rejected():
    """Regression: a present-but-null headlineMetric must be INVALID.

    This is the one divergence the gate found and that was fixed in
    vector_core.schema before the swap: the required-field loop only tested key
    presence, and the old ``elif hm is not None`` let an explicit ``None`` slip
    past the 'must be an object' check. aggregate_fleet always rejected it.
    """
    entry = copy.deepcopy(_BASE)
    entry["headlineMetric"] = None
    assert validate_entry(entry) == ["headlineMetric must be an object"]
    assert bool(_old_entry_problems(entry)) is True


def test_real_fleet_json_still_validates_clean():
    """End-to-end: the committed fleet.json still passes aggregate_fleet.validate()."""
    data = json.loads(aggregate_fleet.FLEET_JSON.read_text(encoding="utf-8"))
    assert aggregate_fleet.validate(data) == []
    # and every entry is clean under the shared contract directly
    for m in data["models"]:
        assert validate_entry(m) == []


def test_aggregate_fleet_flags_bad_entry_with_tag_prefix():
    """The delegated path still tags problems by repo and reports them."""
    data = json.loads(aggregate_fleet.FLEET_JSON.read_text(encoding="utf-8"))
    data["models"][0]["status"] = "bogus"
    problems = aggregate_fleet.validate(data)
    assert any(p.startswith("[vector-hoops]") and "status" in p for p in problems)
