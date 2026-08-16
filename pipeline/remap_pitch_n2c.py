"""Rematch pitch native_to_cross after e_p reclustering (pos_dist L1, 1-1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BAK = ROOT / "pipeline" / "data" / "bak_pre_pitch_con"
KEYS = ("DEF", "MID", "FWD")


def vec(entry: dict) -> np.ndarray:
    return np.array([entry["pos_dist"].get(k, 0.0) for k in KEYS], dtype=float)


def main() -> int:
    old = json.loads((BAK / "native_clusters.json").read_text(encoding="utf-8"))
    new = json.loads((DATA / "native_clusters.json").read_text(encoding="utf-8"))
    amap = json.loads((DATA / "archetype_map.json").read_text(encoding="utf-8"))
    old_map = json.loads((BAK / "archetype_map.json").read_text(encoding="utf-8"))["native_to_cross"]["pitch"]

    O, N = old["pitch"], new["pitch"]
    print("OLD pitch:")
    for e in O:
        print(f"  c{e['cluster']} n={e['n']} {e['pos_dist']}")
    print("NEW pitch:")
    for e in N:
        print(f"  c{e['cluster']} n={e['n']} {e['pos_dist']}")

    cost = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            cost[i, j] = float(np.abs(vec(N[i]) - vec(O[j])).sum())

    used: set[int] = set()
    remap: dict[int, int] = {}
    pairs = sorted((cost[i, j], i, j) for i in range(8) for j in range(8))
    for c, i, j in pairs:
        if i in remap or j in used:
            continue
        remap[i] = j
        used.add(j)
        print(f"new c{i} -> old c{j} cost={c:.3f} " f"n_new={N[i]['n']} n_old={O[j]['n']}")

    new_map = {str(ni): old_map[str(oj)] for ni, oj in sorted(remap.items())}
    print("new pitch native_to_cross", new_map)
    print("gridiron identical?", old["gridiron"] == new["gridiron"])

    amap["native_to_cross"]["pitch"] = new_map
    notes = amap.setdefault("notes", {})
    if not isinstance(notes, dict):
        notes = {"prior": notes}
        amap["notes"] = notes
    notes["pitch_remap_after_supcon"] = (
        "2026-07-11: rematched size-sorted pitch k-means clusters to prior "
        "native_to_cross via pos_dist L1 after e_p SupCon promotion"
    )
    (DATA / "archetype_map.json").write_text(json.dumps(amap, indent=2), encoding="utf-8")
    print("wrote archetype_map.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
