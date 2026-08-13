"""Probe cultural-text alignment face-validity on unified_cultural.pt.

Loads cultural_text_matrix.npz + checkpoint, encodes corpus, reports:
  - mean cosine(text_proj(z), t_p) on labeled rows
  - top predicted cosine neighbors for a few named stars
  - G1/G3 quick check via eval_unified if available

Run:  python pipeline/probe_cultural_text.py [--ckpt unified_cultural.pt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from eval_unified import load_model, encode_all
from train_unified import load_matrix as load_train_matrix

DATA = ROOT / "data"
UCACHE = ROOT / "pipeline" / "data"
ASSETS = ROOT / "assets"
OUT = DATA / "market_cultural" / "cultural_text_probe.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="unified_cultural.pt")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # auto: GPU on personal local (CUDA avail), CPU in Hatch VM

    model, ck = load_model(device, args.ckpt)
    if model.text_proj is None:
        raise SystemExit("checkpoint has no text_proj — train with --cultural-text")
    M = load_train_matrix(device, market=False, cultural_text=True)
    z = torch.tensor(encode_all(model, M, device), device=device)
    with torch.no_grad():
        pred = F.normalize(model.text_proj(z), dim=-1)
    T = M["text_t"]
    m = M["text_mask"]
    cos = (pred * T).sum(dim=-1)
    labeled = m > 0.5
    mean_cos = float(cos[labeled].mean()) if labeled.any() else None

    U = json.loads((ASSETS / "unified.json").read_text(encoding="utf-8"))
    # showcase a few labeled stars
    showcase = []
    want = {"LeBron James", "Tom Brady", "Lionel Messi", "Stephen Curry",
            "Aaron Rodgers", "Cristiano Ronaldo"}
    for i, p in enumerate(U["players"]):
        if p["name"] in want and float(m[i]) > 0.5:
            showcase.append({
                "name": p["name"], "sport": p["sport"], "season": p["season"],
                "cos": round(float(cos[i]), 4),
            })

    report = {
        "ckpt": args.ckpt,
        "n_labeled": int(labeled.sum().item()),
        "mean_align_cos": None if mean_cos is None else round(mean_cos, 4),
        "showcase": showcase[:12],
        "note": "Higher mean_align_cos = z carries Wikipedia narrative geometry",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
