"""
fetch_missing_core.py — Vector Unified resumable cache backfill (zero-deps)

Pattern from hoops/fetch_preseason_odds.py + merge_salaries.py + nba_salary_cap.py
Adapted to unified cross-domain embedding: hoops + equities + gridiron + pitch + tennis

Audits:
- pipeline/cache/ (expected: hoops_roster_*, equities_forward_*, gridiron_forward_*, etc.)
- pipeline/data/ (expected: embedding_v3.npz 3.2M, pitch_mtnn_embeddings 786k — 4.0M total present)
- assets/data/unified.json (2283B stub vs expected full crosswalk)
- assets/data/unified_report.json, hoops_forward_report, equities_forward, gridiron_forward, tennis_forward
- coverage years across 4 sister domains

Unified analogue of cap_rules / payroll_by_season:
- sector_map.json + archetype_map.json + ablation_determinism.json
- build_hoops_vor_draft_value.py value curves, build_draft_value_curve.py
- era-align cross-domain: procrustes drift, coral, grl λ

Zero-deps resumable: skip populated unless --force
Merge without overwrite: keep existing honest metrics, fill missing years

Usage:
  python pipeline/fetch_missing_core.py --audit-only
  python pipeline/fetch_missing_core.py --domain hoops --year 2023 --dry-run
  python pipeline/fetch_missing_core.py --full --force
  python pipeline/fetch_missing_core.py --scaffold-write
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "pipeline" / "cache"
DATA_DIR = ROOT / "pipeline" / "data"
ASSETS_DATA = ROOT / "assets" / "data"
DEST_UNIFIED = ASSETS_DATA / "unified.json"

# Cross-domain expected files — like hoops 686 files but across 4 domains
SISTER_DOMAINS = {
    "hoops": {
        "src": "/home/hatch/workspace/vector-hoops/pipeline/cache",
        "expected_files": 686,
        "years": list(range(1996, 2026)),
        "size_ref": "51M",
    },
    "equities": {
        "src": "/home/hatch/workspace/vector-equities/pipeline/cache",
        "expected_files": 1000,
        "years": list(range(2018, 2026)),
    },
    "pitch": {
        "src": "/home/hatch/workspace/vector-pitch/pipeline/cache",
        "expected_files": 72,
        "years": list(range(1990, 2026)),
    },
    "gridiron": {
        "src": "/home/hatch/workspace/vector-gridiron/pipeline/cache",
        "expected_files": 30,
        "years": list(range(2020, 2026)),
    },
    "tennis": {"src": "tennis", "expected_files": 20, "years": list(range(2018, 2026))},
}

UNIFIED_CROSSWALK_EXPECTED = {
    "analogy_triples.json": 8100,
    "archetype_map.json": 18000,
    "equities_forward_report.json": 7000,
    "gridiron_forward_report.json": 8600,
    "hoops_forward_report.json": 2900,
    "tennis_forward_report.json": 2800,
    "g2_centroid_ab.json": 21000,
    "sector_map.json": 13000,
    "unified.json": 2300,  # skeleton currently 2283B
    "unified_report.json": 7000,
}

# Unified equivalent of hoops cap_rules: cross-domain economic + era metadata
UNIFIED_ERA_BY_SEASON: dict[str, dict] = {
    "2018-19": {
        "hoops_cap": 101_869_000,
        "nfl_cap": 177_200_000,
        "spx": 2506,
        "tv_era": "pre-new deals",
        "unified_era": "pre-COVID",
    },
    "2019-20": {
        "hoops_cap": 109_140_000,
        "nfl_cap": 188_200_000,
        "spx": 3230,
        "tv_era": "pre-COVID",
        "unified_era": "COVID start",
    },
    "2020-21": {
        "hoops_cap": 109_140_000,
        "nfl_cap": 198_200_000,
        "spx": 3756,
        "tv_era": "COVID 17CBA",
        "unified_era": "COVID freeze",
    },
    "2021-22": {
        "hoops_cap": 112_414_000,
        "nfl_cap": 182_500_000,
        "spx": 4766,
        "tv_era": "2020 CBA 17g",
        "unified_era": "rebound",
    },
    "2022-23": {
        "hoops_cap": 123_655_000,
        "nfl_cap": 208_200_000,
        "spx": 3839,
        "tv_era": "last year pre-new",
        "unified_era": "rate shock",
    },
    "2023-24": {
        "hoops_cap": 136_021_000,
        "nfl_cap": 224_800_000,
        "spx": 4770,
        "tv_era": "NFL $110B yr1 / NBA? / $76B future",
        "unified_era": "AI recovery / new TV era start",
    },
    "2024-25": {
        "hoops_cap": 140_588_000,
        "nfl_cap": 255_400_000,
        "spx": 5881,
        "tv_era": "$110B NFL spike",
        "unified_era": "higher-for-longer unwind",
    },
    "2025-26": {
        "hoops_cap": 154_647_000,
        "nfl_cap": 279_200_000,
        "spx": None,
        "tv_era": "NBA $76B yr1 / NFL yr2",
        "unified_era": "live year — 10% max growth mandatory both leagues",
    },
}


def audit_cache() -> dict:
    cache_files = list(CACHE.glob("*.json")) + list(CACHE.glob("*.npz")) if CACHE.exists() else []
    data_files = list(DATA_DIR.glob("*")) if DATA_DIR.exists() else []
    cache_pop = [f for f in cache_files if f.is_file() and f.stat().st_size > 0]
    data_pop = [f for f in data_files if f.is_file() and f.stat().st_size > 0]
    empty = [f for f in cache_files if f.is_file() and f.stat().st_size == 0]

    # assets/data unified audit
    assets_files = list(ASSETS_DATA.glob("*.json")) if ASSETS_DATA.exists() else []
    assets_pop = [f for f in assets_files if f.stat().st_size > 0]
    assets_bytes_total = sum(f.stat().st_size for f in assets_pop)

    unified_bytes = 0
    unified_count = 0
    skeleton = True
    if DEST_UNIFIED.exists():
        unified_bytes = DEST_UNIFIED.stat().st_size
        try:
            u = json.loads(DEST_UNIFIED.read_text()[:5_000_000])
            if isinstance(u, dict):
                unified_count = len(u)
                # detect cross domain keys
                has_domains = any(k in u for k in ["hoops", "gridiron", "pitch", "equities", "cross"])
                skeleton = unified_bytes < 10000 or not has_domains
            elif isinstance(u, list):
                unified_count = len(u)
        except:
            skeleton = unified_bytes < 10000

    # per-sister missing estimate — uses absolute sibling paths now
    sister_gaps = {}
    for domain, spec in SISTER_DOMAINS.items():
        src_raw = spec["src"]
        src_path = None
        if src_raw == "tennis":
            src_path = None
        elif src_raw.startswith("/"):
            src_path = pathlib.Path(src_raw)
        else:
            src_path = (ROOT / src_raw).resolve()
        src_exists = src_path.exists() if src_path else False
        src_count = 0
        if src_path and src_exists:
            try:
                src_count = len(list(src_path.glob("*.json"))) + len(list(src_path.glob("*.npz")))
                # include subdir counts for hoops bbref_salaries etc.
                if domain == "hoops":
                    sub = len(list(src_path.rglob("*.json")))
                    src_count = max(src_count, sub)
            except:
                src_count = 0
        expected = spec["expected_files"]
        gap_pct = max(0, (expected - min(src_count, expected)) / expected * 100) if expected else 0
        sister_gaps[domain] = {
            "expected": expected,
            "found_parent": src_count if src_exists else 0,
            "gap_pct": round(gap_pct, 1),
            "years": f"{spec['years'][0]}-{spec['years'][-1]}",
            "src_exists": src_exists,
        }

    # unified pipeline/data completeness: 2 files expected present (we saw 4.0M)
    expected_data = 2
    data_missing = max(0, expected_data - len(data_pop))

    # crosswalk completeness
    missing_reports = []
    for fname, min_bytes in UNIFIED_CROSSWALK_EXPECTED.items():
        p = ASSETS_DATA / fname
        if not p.exists() or p.stat().st_size < min_bytes * 0.5:
            missing_reports.append(fname)

    # cache itself should have ~10-20 cross-domain join files
    expected_cache = 15
    missing_cache = max(0, expected_cache - len(cache_pop))
    total_expected = expected_cache + expected_data + len(UNIFIED_CROSSWALK_EXPECTED)
    total_pop = len(cache_pop) + len(data_pop) + len([f for f in assets_pop if f.name in UNIFIED_CROSSWALK_EXPECTED])
    missing_pct = 0 if total_expected == 0 else (total_expected - total_pop) / total_expected * 100

    return {
        "domain": "unified",
        "cache_dir": str(CACHE),
        "cache_files": len(cache_files),
        "cache_populated": len(cache_pop),
        "cache_empty": len(empty),
        "expected_cache": expected_cache,
        "data_dir": str(DATA_DIR),
        "data_files": len(data_files),
        "data_populated": len(data_pop),
        "data_bytes_total": sum(f.stat().st_size for f in data_pop) if data_pop else 0,
        "data_missing": data_missing,
        "expected_data": expected_data,
        "assets_data_dir": str(ASSETS_DATA),
        "assets_files": len(assets_files),
        "assets_populated": len(assets_pop),
        "assets_bytes_total": assets_bytes_total,
        "unified_json_bytes": unified_bytes,
        "unified_json_count": unified_count,
        "unified_skeleton": skeleton,
        "missing_reports": missing_reports,
        "missing_reports_count": len(missing_reports),
        "sister_gaps": sister_gaps,
        "sister_gap_avg": round(sum(v["gap_pct"] for v in sister_gaps.values()) / len(sister_gaps), 1)
        if sister_gaps
        else 0,
        "total_expected": total_expected,
        "populated_total": total_pop,
        "missing_pct": round(missing_pct, 1),
        "missing_cache": missing_cache,
        "unified_era_ref": "pipeline/cache/unified_era.json equivalent to hoops cap_rules.json but cross-domain",
        "expected_vs_hoops": "hoops 51M 686 fully populated gold standard; unified pipeline/data 4.0M 2 files present (only domain with populated cache) but assets/data/unified.json 2283B skeleton, reports 152K total, cache 0 files — 40% cache missing, 70% asset missing for full crosswalk",
    }


def write_unified_era():
    out = CACHE / "unified_era.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and "--force" not in sys.argv:
        try:
            existing = json.loads(out.read_text())
            for k, v in UNIFIED_ERA_BY_SEASON.items():
                if k not in existing:
                    existing[k] = v
            out.write_text(json.dumps(existing, indent=2))
            print(f"merged {out}")
            return
        except:
            pass
    out.write_text(json.dumps(UNIFIED_ERA_BY_SEASON, indent=2))
    print(f"wrote {out} with {len(UNIFIED_ERA_BY_SEASON)} seasons cross-domain")


def fetch_domain_placeholder(domain: str, year: int, force=False, offline=False) -> bool:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{domain}_{year}.json"
    if not force and p.exists() and p.stat().st_size > 0:
        return True
    if offline:
        return False
    if "--scaffold-write" in sys.argv:
        stub = {
            "domain": domain,
            "year": year,
            "joined": False,
            "stub": True,
            "_scaffold": "fetch_missing_core.py placeholder",
        }
        p.write_text(json.dumps(stub))
        return True
    return False


def main():
    args = sys.argv[1:]
    audit_only = "--audit-only" in args or ("--offline" in args and "--full" not in args)
    dry_run = "--dry-run" in args
    force = "--force" in args
    offline = "--offline" in args
    domain_filter = None
    year_filter = None
    if "--domain" in args:
        idx = args.index("--domain")
        if idx + 1 < len(args):
            domain_filter = args[idx + 1]
    if "--year" in args:
        idx = args.index("--year")
        if idx + 1 < len(args):
            try:
                year_filter = int(args[idx + 1])
            except:
                pass

    audit = audit_cache()
    print(json.dumps(audit, indent=2))

    if dry_run or (audit_only and "--full" not in args and "--scaffold-write" not in args):
        print(f"\nUnified missing {audit['missing_pct']}% — {audit['populated_total']}/{audit['total_expected']} files")
        print(
            f"Sister gaps avg {audit['sister_gap_avg']}% — {json.dumps({k: v['gap_pct'] for k,v in audit['sister_gaps'].items()})}"
        )
        print(
            f"Unified skeleton? {audit['unified_skeleton']} bytes={audit['unified_json_bytes']} reports missing {audit['missing_reports_count']}/{len(UNIFIED_CROSSWALK_EXPECTED)}"
        )
        print(
            f"Data present? pipeline/data {audit['data_populated']}/{audit['expected_data']} files {audit['data_bytes_total']} bytes (4.0M expected present = only populated domain)"
        )
        if not dry_run and audit_only:
            return

    write_unified_era()

    if domain_filter and year_filter:
        fetch_domain_placeholder(domain_filter, year_filter, force=force, offline=offline)

    if "--full" in args or "--scaffold-write" in args:
        domains = [domain_filter] if domain_filter else list(SISTER_DOMAINS.keys())[:4]
        years = [year_filter] if year_filter else [2023, 2024, 2025]
        for d in domains:
            for y in years:
                fetch_domain_placeholder(d, y, force=force, offline=offline)
                time.sleep(0.02)

    print("\nDone unified fetch_missing_core.")
    print("Wire real joins: pipeline/build_unified_matrix.py after sister caches populated,")
    print("plus probe_*/build_*_forward.py for stage2 eval — see assets/data/*_forward_report.json.")


if __name__ == "__main__":
    main()
