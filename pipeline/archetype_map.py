"""Vector Unified — derive each sport's native role clusters (§4b anchor prep).

The cross-sport archetype taxonomy (data/archetype_map.json) is human-authored and
version-controlled. To author the native→cross mapping we first need each sport's
native clusters described in comparable terms:

  * hoops   — uses the SHIPPED 8 native clusters (embedding_v3.npz `cluster`) +
              the gameArchetypes names from mtnn_arch.json. No re-clustering.
  * pitch   — k-means(8) on e_p (24-d). Pitch has no shipped label in the export.
  * gridiron— k-means(8) on the season-aggregated e_g (32-d). Gridiron ships no
              archetype label (only a 4-way position head).

Each native cluster is characterised by size + position distribution so a human can
map it onto the A0-A11 cross-sport taxonomy. Output: data/native_clusters.json.

This is descriptive, not the mapping itself. The mapping is hand-authored in
data/archetype_map.json::native_to_cross after inspecting this output.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from load_encoders import ROOT, load_all

DATA = ROOT / "data"
SEED = 7
K = 8
CROSS_ARCH_IDS = [
    "A0",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A11",
]  # v0 in-scope order -> idx 0..6
HOOPS_POS = {0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C", -1: "?"}
HOOPS_ARCH = [
    "Offensive Glass + Rim Protection",
    "Offensive Glass (Low Shot Volume)",
    "Three-Point Volume (Low On-Court Impact)",
    "Defensive Glass + Rim Pressure (Fts)",
    "Shot Volume + Three-Point Volume",
    "Three-Point Accuracy + Three-Point Volume",
    "Playmaking + Steals",
    "Scoring Volume + Shot Volume",
]


def _pos_dist(pos_list):
    c = Counter(str(p) for p in pos_list)
    total = sum(c.values()) or 1
    return {p: round(n / total, 3) for p, n in sorted(c.items(), key=lambda kv: -kv[1])}


def _kmeans(E, k=K, seed=SEED):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(E)
    return labels.astype(int)


def native_labels(sport, E, k=K, seed=SEED):
    """Size-sorted k-means labels matching data/native_clusters.json.

    Returns an int array of cluster ids 0..k-1, reassigned so that id 0 is the
    largest cluster, id 1 the next, etc. (stable across runs given seed). Hoops
    returns None because hoops uses its SHIPPED cluster labels (records[
    'native_cluster']) rather than re-clustering.
    """
    if sport == "hoops":
        return None
    labels = _kmeans(E, k=k, seed=seed)
    counts = Counter(labels.tolist())
    order = sorted(range(k), key=lambda c: -counts.get(c, 0))
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[int(c)] for c in labels], dtype=int)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    all_sport = load_all(verbose=False)
    out = {}

    # hoops — shipped labels + names
    recs = all_sport["hoops"]["records"]
    clu = np.array([r["native_cluster"] for r in recs])
    pos = [r["pos"] for r in recs]
    hoops_entries = []
    for c in range(K):
        m = clu == c
        hoops_entries.append(
            {
                "cluster": c,
                "label": HOOPS_ARCH[c] if c < len(HOOPS_ARCH) else f"cluster_{c}",
                "n": int(m.sum()),
                "pos_dist": _pos_dist([HOOPS_POS.get(p, p) for p in np.array(pos)[m]]),
            }
        )
    out["hoops"] = hoops_entries

    # pitch + gridiron — k-means(8), size-sorted via native_labels (matches native_clusters.json)
    for sport in ("pitch", "gridiron"):
        E = all_sport[sport]["E"]
        recs = all_sport[sport]["records"]
        labels = native_labels(sport, E)
        entries = []
        for c in range(K):
            m = labels == c
            entries.append(
                {
                    "cluster": c,
                    "n": int(m.sum()),
                    "pos_dist": _pos_dist([r["pos"] for r in np.array(recs)[m]]),
                }
            )
        out[sport] = entries

    (DATA / "native_clusters.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {DATA / 'native_clusters.json'}")
    for sport, entries in out.items():
        print(f"\n== {sport} ({sum(e['n'] for e in entries)} rows, K={K}) ==")
        for e in entries:
            top = ",".join(f"{p}:{d:.0%}" for p, d in list(e["pos_dist"].items())[:3])
            lbl = f"  {e['label']}" if "label" in e else ""
            print(f"  c{e['cluster']:>2} n={e['n']:>5}  {top}{lbl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
