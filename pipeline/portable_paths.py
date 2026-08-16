#!/usr/bin/env python3
"""dumbmodel.com publishes its sources. Those sources must not be paths on my laptop.

Solo personal project, no connection to employer, built with public/free-tier only

WHAT WAS SHIPPING. assets/model.js renders every `source_files` entry under a heading that
makes an explicit promise:

    <h2>Every number above came from these files</h2>
      <li>C:/Users/jcdav/vector-unified/data/g1_position_probe.json</li>
      <li>C:/Users/jcdav/vector-hoops/assets/mtnn_meta.json</li>
      ... 45 of them, across all six pages

Plus 14 more inside insight and headline-stat `source` fields. So a reader was told exactly
where every number came from and handed a path that exists on exactly one computer.

THIS IS THE SAME DEFECT THE REST OF THIS REPO KEEPS FINDING, in a new costume: a real value
answering a different question than the one it appears to answer. The path is real. It is
correct. It resolves — for me. What it appears to be is provenance a reader can follow, and
what it actually is is a note to self. The site's fine print says "Every number is
recomputable from public sources", and a citation nobody but the author can open does not
meet that claim no matter how accurate it is.

It also published the OS username 45 times, which is a smaller and more boring problem, but
not one worth keeping.

THE FIX IS A RELATIVE ROOT, NOT A PRETTIER STRING. Citations are stored repo-relative:

    C:/Users/jcdav/vector-unified/data/hoops_forward_report.json
    ->            vector-unified/data/hoops_forward_report.json

which names the repo AND the path inside it, so it is still a real, checkable citation —
and it resolves for anyone who clones the estate side by side, wherever they put it. ESTATE
is derived from this file's own location rather than hardcoded, so that is true by
construction instead of by my remembering to update a constant.

RESOLUTION FAILS LOUD. resolve() returns None for a prefix that is not a known repo, so a
caller must decide what to do about it and CANNOT accidentally treat "I do not know where
this lives" as "this is fine". check_hub_freshness.py turns that None into a failure. The
alternative — silently skipping unresolvable citations — would make the freshness check
vacuous for exactly the entries most likely to be wrong.
"""

from __future__ import annotations

import re
from pathlib import Path

# This file lives at <estate>/vector-unified/pipeline/portable_paths.py, so the estate root
# is three parents up. Derived, not hardcoded: a constant would be one more thing that is
# true today and silently wrong after a move, and the whole point here is portability.
ROOT = Path(__file__).resolve().parent.parent
ESTATE = ROOT.parent

# The sibling repos whose files the site is allowed to cite. A citation into anything else
# is a mistake worth failing on rather than rewriting into a plausible-looking relative path.
REPOS = (
    "vector-unified",
    "vector-hoops",
    "vector-gridiron",
    "vector-pitch",
    "vector-equities",
    "vector-tennis",
    "vector-hub",
    "vector-golf",
)

# Any drive-letter or UNC path. Deliberately broad: the rule is "no machine-local path in
# published data", not "no path under C:/Users/jcdav".
ABS_RE = re.compile(r"(?:[A-Za-z]:[/\\]|\\\\[A-Za-z0-9_.-]+\\)")


def to_portable(s: str) -> str:
    """Rewrite every absolute estate path inside a string to its repo-relative form.

    Operates on the whole string, not just a bare path, because insight `source` fields
    carry rich citations like `C:/x/y.json -> {field=value}; C:/a/b.json -> ...` and the
    path is only part of the sentence. Non-path text is untouched.
    """
    out = s.replace("\\", "/")
    est = str(ESTATE).replace("\\", "/").rstrip("/")
    for repo in REPOS:
        out = out.replace(f"{est}/{repo}/", f"{repo}/")
    return out


def resolve(cited: str) -> Path | None:
    """Portable citation -> absolute path on THIS machine, or None if the root is unknown.

    None is a real answer meaning "I cannot check this", and callers must not collapse it
    into a pass. An already-absolute path is returned as-is so migration is incremental.
    """
    c = cited.replace("\\", "/")
    if ABS_RE.match(c):
        return Path(c)
    head = c.split("/", 1)[0]
    if head not in REPOS:
        return None
    return ESTATE / c


def find_absolute(obj, path: str = "") -> list[tuple[str, str]]:
    """Every (json_location, string) in a nested structure holding a machine-local path.

    Walks the whole document rather than checking known fields, because the fields that
    carry citations have grown twice already — source_files, then insights[].source, then
    headline_stats[].source — and each time the checker that only knew the old list kept
    reporting clean. A whole-document walk cannot be outgrown that way.
    """
    hits: list[tuple[str, str]] = []
    if isinstance(obj, str):
        if ABS_RE.search(obj):
            hits.append((path or "<root>", obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            hits += find_absolute(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += find_absolute(v, f"{path}[{i}]")
    return hits
