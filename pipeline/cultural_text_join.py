"""Vector Unified — join cultural text embeddings onto unified corpus order.

Reads:
  assets/unified.json
  data/market_cultural/cultural_text_embeds.npz
  data/market_cultural/wikipedia_bios.json (optional titles)

Writes:
  data/market_cultural/cultural_text_matrix.npz
    T (N, d_text) float32, m_text (N,) float32, extract_chars (N,) int32
  data/market_cultural/cultural_text.json
    lite metadata + coverage (no full vectors — those live in the npz)

Honesty: missing → m_text=0 and T row zeros (head ignores via mask).

Run:  python pipeline/cultural_text_join.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acquire_forbes import norm_name

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "market_cultural"
ASSETS = ROOT / "assets"
OUT_JSON = DATA / "cultural_text.json"
OUT_NPZ = DATA / "cultural_text_matrix.npz"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    U = json.loads((ASSETS / "unified.json").read_text(encoding="utf-8"))
    npz = np.load(DATA / "cultural_text_embeds.npz", allow_pickle=True)
    pks = [str(x) for x in npz["pk"]]
    T_u = npz["t"].astype(np.float32)
    chars_u = {pk: int(c) for pk, c in zip(pks, npz["extract_chars"])}
    emb = {pk: T_u[i] for i, pk in enumerate(pks)}
    d_text = int(T_u.shape[1])

    bios = {}
    bios_path = DATA / "wikipedia_bios.json"
    if bios_path.exists():
        bios = json.loads(bios_path.read_text(encoding="utf-8")).get("players", {})

    n = len(U["players"])
    T = np.zeros((n, d_text), dtype=np.float32)
    m_text = np.zeros(n, dtype=np.float32)
    extract_chars = np.zeros(n, dtype=np.int32)
    meta_rows = []
    n_ok = 0
    for i, p in enumerate(U["players"]):
        pk = f"{p['sport']}::{norm_name(p['name'])}"
        bio = bios.get(pk, {})
        vec = emb.get(pk)
        if vec is not None:
            T[i] = vec
            m_text[i] = 1.0
            extract_chars[i] = chars_u.get(pk, int(bio.get("extract_chars") or 0))
            n_ok += 1
            meta_rows.append({
                "i": i, "sport": p["sport"], "name": p["name"],
                "wiki_title": bio.get("wiki_title"), "m_text": 1,
            })
        else:
            meta_rows.append({
                "i": i, "sport": p["sport"], "name": p["name"],
                "wiki_title": bio.get("wiki_title"), "m_text": 0,
            })

    np.savez_compressed(OUT_NPZ, T=T, m_text=m_text, extract_chars=extract_chars)
    by_sport = {}
    for s in ("hoops", "gridiron", "pitch"):
        idx = [i for i, p in enumerate(U["players"]) if p["sport"] == s]
        by_sport[s] = {
            "n": len(idx),
            "labeled": int(m_text[idx].sum()),
            "rate": round(float(m_text[idx].mean()), 4) if idx else 0.0,
        }
    out = {
        "built": time.strftime("%Y-%m-%d"),
        "n_rows": n,
        "d_text": d_text,
        "model": "all-MiniLM-L6-v2",
        "matrix": OUT_NPZ.name,
        "coverage": {
            "rows_with_text": int(n_ok),
            "row_rate": round(n_ok / n, 4),
            "unique_embeds": len(emb),
            "by_sport": by_sport,
        },
        "honesty": "m_text=0 rows have zero T; head ignores via mask — never imputed",
        "reddit": "deferred — see docs/CULTURAL_TEXT_SCHEMA.md",
        # keep only labeled meta for inspectability (full mask is in npz)
        "labeled_sample": meta_rows[:50],
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT_NPZ} T={T.shape} labeled={n_ok}/{n} "
          f"({100 * n_ok / n:.1f}%) | {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
