"""Schema for a single fleet-report entry.

Mirrors the shape of each object in the dashboard's ``fleet/data/fleet.json``
``models`` array, so the aggregation script and any consumer can validate /
construct entries with a typed contract. Pure stdlib — no numpy or torch needed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

__all__ = ["HeadlineMetric", "FleetEntry", "FleetEntryDict", "validate_entry"]


class HeadlineMetric(TypedDict, total=False):
    name: str
    value: float
    baseline: float
    source: str


class FleetEntryDict(TypedDict, total=False):
    """TypedDict view of a fleet entry as it appears in fleet.json."""

    repo: str
    domain: str
    visibility: str
    liveUrl: str
    embeddingDim: int
    archTag: str
    status: str
    headlineMetric: HeadlineMetric
    metrics: dict[str, Any]
    strengths: str
    gaps: str


@dataclass
class FleetEntry:
    """Dataclass for a fleet-report entry.

    Fields match ``fleet/data/fleet.json``. ``to_dict`` produces the JSON-ready
    mapping; ``from_dict`` reads one back (ignoring unknown keys defensively).
    """

    repo: str
    domain: str
    embeddingDim: int
    status: str
    headlineMetric: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    strengths: str = ""
    gaps: str = ""
    visibility: str = "public"
    liveUrl: str = ""
    archTag: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FleetEntry:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# Fields that must be present (and non-empty for strings) on every entry.
_REQUIRED = ("repo", "domain", "embeddingDim", "status", "headlineMetric")
_VALID_STATUS = {"production", "shipped", "wip", "blocked"}


def validate_entry(d: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems with ``d`` (empty == valid)."""
    problems: list[str] = []
    for key in _REQUIRED:
        if key not in d:
            problems.append(f"missing required field: {key}")
    if "status" in d and d["status"] not in _VALID_STATUS:
        problems.append(f"invalid status {d['status']!r}; expected one of {sorted(_VALID_STATUS)}")
    if "embeddingDim" in d and not isinstance(d["embeddingDim"], int):
        problems.append("embeddingDim must be an int")
    hm = d.get("headlineMetric")
    if isinstance(hm, dict):
        for key in ("name", "value"):
            if key not in hm:
                problems.append(f"headlineMetric missing {key}")
    elif "headlineMetric" in d:
        # Present but not an object (includes an explicit ``None``). An absent
        # headlineMetric is already reported by the required-field loop above; a
        # present-but-null one must be rejected too — a fleet entry whose headline
        # metric is null is malformed, matching vector-unified's aggregate_fleet
        # validator (the canonical consumer of this contract).
        problems.append("headlineMetric must be an object")
    return problems
