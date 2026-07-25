"""Vector Unified — embed Wikipedia leads with frozen MiniLM (transformers).

Avoids sentence_transformers (broken vs local transformers Trainer import).
Uses mean-pooled `sentence-transformers/all-MiniLM-L6-v2` → L2 384-d.

Reads data/market_cultural/wikipedia_bios.json (status=ok extracts).
Writes:
  data/market_cultural/cultural_text_embeds.npz
  data/market_cultural/cultural_text_embeds_meta.json

Run:  python pipeline/embed_cultural_text.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "market_cultural"
BIOS = DATA / "wikipedia_bios.json"
OUT_NPZ = DATA / "cultural_text_embeds.npz"
OUT_META = DATA / "cultural_text_embeds_meta.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not BIOS.exists():
        raise SystemExit(f"missing {BIOS} — run acquire_wikipedia_bios.py first")
    bios = json.loads(BIOS.read_text(encoding="utf-8"))
    rows = [(pk, r) for pk, r in bios["players"].items()
            if r.get("status") == "ok" and r.get("extract")]
    if not rows:
        raise SystemExit("no ok extracts to embed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"embedding {len(rows)} leads with {args.model} on {device} …", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    texts = [(r.get("extract") or "")[:1200] for _, r in rows]
    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch):
            batch = texts[i:i + args.batch]
            enc = tok(batch, padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(device)
            out = model(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            chunks.append(emb.cpu().numpy().astype(np.float32))
            if (i // args.batch) % 10 == 0:
                print(f"  batch {i // args.batch + 1}", flush=True)

    emb = np.concatenate(chunks, axis=0)
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), (float(norms.min()), float(norms.max()))

    pks = np.array([pk for pk, _ in rows], dtype=object)
    chars = np.array([int(r.get("extract_chars") or len(r.get("extract") or ""))
                      for _, r in rows], dtype=np.int32)
    m = np.ones(len(rows), dtype=np.float32)
    np.savez_compressed(OUT_NPZ, pk=pks, t=emb, extract_chars=chars, m=m)
    meta = {
        "built": time.strftime("%Y-%m-%d"),
        "model": args.model,
        "backend": "transformers AutoModel mean-pool",
        "d_text": int(emb.shape[1]),
        "n": int(len(rows)),
        "npz": str(OUT_NPZ.name),
        "sentence_transformers": "skipped (Trainer import broken in this env)",
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {OUT_NPZ} shape={emb.shape} | {OUT_META}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
