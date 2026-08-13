"""Face-validity probe: do unified_market.pt salary/award/reach heads rank known stars?

Loads unified_market.pt + market_cultural.json, encodes full corpus, scores
salary_head, award_head, and reach_head on z. Reports:
  - Spearman corr of pred vs label on labeled rows (per sport + pooled)
  - Top-10 predicted salary / award prestige / reach (should surface Forbes/
    MVP/most-famous names)
  - Cross-sport showcase: top predicted earnings mix of sports?

reach_head predicts log(Wikipedia pageviews) -- a keyless social-reach signal
(acquire_wikipedia_pageviews.py) added 2026-07-30 in place of the originally-
scoped Apify path.

Run: python pipeline/probe_market_heads.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_unified import UnifiedTrunk, load_matrix, SPORTS, UCACHE, DATA, SEED
from eval_unified import load_model, encode_all
from acquire_forbes import norm_name

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # auto: GPU on personal local (CUDA avail), CPU in Hatch VM


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    model, ck = load_model(DEVICE, "unified_market.pt")
    M = load_matrix(DEVICE, market=True)
    z = encode_all(model, M, DEVICE)
    zt = torch.tensor(z, device=DEVICE)
    with torch.no_grad():
        sal_pred = model.salary_head(zt).squeeze(-1).cpu().numpy()
        awd_pred = model.award_head(zt).squeeze(-1).cpu().numpy()
        rch_pred = model.reach_head(zt).squeeze(-1).cpu().numpy()

    mc = json.loads((DATA / "market_cultural" / "market_cultural.json").read_text(encoding="utf-8"))
    rows = mc["rows"]
    sal_m = np.array([r["m_salary"] for r in rows])
    awd_m = np.array([r["m_award"] for r in rows])
    rch_m = np.array([r.get("m_reach", 0) for r in rows])
    sal_z = M["salary_z"].cpu().numpy()
    awd_z = M["award_z"].cpu().numpy()
    rch_z = M["reach_z"].cpu().numpy()
    sid = M["sport_id"].cpu().numpy()

    print("=== Spearman pred vs label (labeled rows) ===")
    for s, name in enumerate(SPORTS):
        ms = (sid == s) & (sal_m == 1)
        ma = (sid == s) & (awd_m == 1)
        if ms.sum() >= 10:
            rho, _ = spearmanr(sal_pred[ms], sal_z[ms])
            print(f"  {name:9s} salary  n={ms.sum():4d}  rho={rho:.3f}")
        else:
            print(f"  {name:9s} salary  n={ms.sum():4d}  (too few)")
        if ma.sum() >= 5:
            rho, _ = spearmanr(awd_pred[ma], awd_z[ma])
            print(f"  {name:9s} award   n={ma.sum():4d}  rho={rho:.3f}")
        mr = (sid == s) & (rch_m == 1)
        if mr.sum() >= 10:
            rho, _ = spearmanr(rch_pred[mr], rch_z[mr])
            print(f"  {name:9s} reach   n={mr.sum():4d}  rho={rho:.3f}")
        else:
            print(f"  {name:9s} reach   n={mr.sum():4d}  (too few)")
    ms = sal_m == 1
    ma = awd_m == 1
    mr = rch_m == 1
    print(f"  pooled    salary  n={ms.sum():4d}  rho={spearmanr(sal_pred[ms], sal_z[ms])[0]:.3f}")
    print(f"  pooled    award   n={ma.sum():4d}  rho={spearmanr(awd_pred[ma], awd_z[ma])[0]:.3f}")
    print(f"  pooled    reach   n={mr.sum():4d}  rho={spearmanr(rch_pred[mr], rch_z[mr])[0]:.3f}")

    # top predicted (dedupe by name+sport, take best season)
    print("\n=== Top-12 predicted salary (unique name+sport) ===")
    order = np.argsort(-sal_pred)
    seen = set()
    for i in order:
        key = (norm_name(rows[i]["name"]), rows[i]["sport"])
        if key in seen:
            continue
        seen.add(key)
        lab = f"sal=${rows[i]['salary_m']:.1f}M" if rows[i]["m_salary"] else "unlabeled"
        print(f"  [{rows[i]['sport'][:2]}] {rows[i]['name']:<22} pred={sal_pred[i]:+.2f}  {lab}")
        if len(seen) >= 12:
            break

    print("\n=== Top-12 predicted award prestige (unique name+sport) ===")
    order = np.argsort(-awd_pred)
    seen = set()
    for i in order:
        key = (norm_name(rows[i]["name"]), rows[i]["sport"])
        if key in seen:
            continue
        seen.add(key)
        lab = f"prestige={rows[i]['award_prestige']}" if rows[i]["m_award"] else "unlabeled"
        print(f"  [{rows[i]['sport'][:2]}] {rows[i]['name']:<22} pred={awd_pred[i]:+.2f}  {lab}")
        if len(seen) >= 12:
            break

    print("\n=== Top-12 predicted reach (unique name+sport) ===")
    order = np.argsort(-rch_pred)
    seen = set()
    for i in order:
        key = (norm_name(rows[i]["name"]), rows[i]["sport"])
        if key in seen:
            continue
        seen.add(key)
        lab = f"views={rows[i]['reach_views']:,}" if rows[i]["m_reach"] else "unlabeled"
        print(f"  [{rows[i]['sport'][:2]}] {rows[i]['name']:<22} pred={rch_pred[i]:+.2f}  {lab}")
        if len(seen) >= 12:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
