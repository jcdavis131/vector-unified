"""Unit tests for pipeline/build_asof_join.py (fixture-only, no estate deps)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import build_asof_join as baj


@pytest.fixture()
def tiny_world(tmp_path: Path):
    venue = {
        "edges": [
            {
                "venue": "Test Field",
                "team_wiki": "Test Team",
                "team_code": "TT",
                "ticker": "AAA",
                "company": "Acme",
                "sponsor_name_from": 2018,
                "naming_year_source": "fixture",
            }
        ],
        "hoops": {"edges": []},
    }
    (tmp_path / "venue_edges.json").write_text(json.dumps(venue), encoding="utf-8")
    mc = {
        "rows": [
            {
                "sport": "gridiron",
                "player_id": "p1",
                "name": "Pat Player",
                "season": "2019",
                "year": 2019,
                "endorse_m": 1.5,
                "m_endorse": 1,
            },
            {
                "sport": "gridiron",
                "player_id": "p2",
                "name": "Early Player",
                "season": "2017",
                "year": 2017,
                "endorse_m": 9.0,
                "m_endorse": 1,
            },
        ]
    }
    mc_path = tmp_path / "market_cultural.json"
    mc_path.write_text(json.dumps(mc), encoding="utf-8")
    emb = tmp_path / "grid.npz"
    np.savez(
        emb,
        E=np.zeros((3, 2), dtype=np.float32),
        gsis=np.array(["p1", "p2", "p3"]),
        name=np.array(["Pat Player", "Early Player", "No Endorse"]),
        season=np.array([2019, 2017, 2020], dtype=np.int32),
        team=np.array(["TT", "TT", "TT"]),
        pos=np.array(["QB", "WR", "RB"]),
    )
    return tmp_path, emb, mc_path


def test_named_window_filters_pre_naming_seasons(tiny_world):
    tmp_path, emb, mc_path = tiny_world
    edges = baj.load_venue_edges(tmp_path / "venue_edges.json")
    endorse = baj.index_endorse(json.loads(mc_path.read_text(encoding="utf-8"))["rows"])
    rows = baj.expand_gridiron(edges["gridiron"], emb, endorse)
    years = sorted(r["season_year"] for r in rows)
    assert years == [2019, 2020]
    assert all(r["ticker"] == "AAA" for r in rows)
    by_id = {r["player_id"]: r for r in rows}
    assert by_id["p1"]["endorse_m"] == 1.5 and by_id["p1"]["m_endorse"] == 1
    assert by_id["p3"]["m_endorse"] == 0 and by_id["p3"]["endorse_m"] is None


def test_cli_writes_jsonl_and_disclaimer(tiny_world, tmp_path: Path):
    world, emb, mc_path = tiny_world
    out_jsonl = tmp_path / "out.jsonl"
    out_meta = tmp_path / "out_meta.json"
    rc = baj.main(
        [
            "--data-root",
            str(world),
            "--venue-edges",
            str(world / "venue_edges.json"),
            "--market-cultural",
            str(mc_path),
            "--grid-emb",
            str(emb),
            "--out-jsonl",
            str(out_jsonl),
            "--out-meta",
            str(out_meta),
            "--skip-hoops",
        ]
    )
    assert rc == 0
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    meta = json.loads(out_meta.read_text(encoding="utf-8"))
    assert meta["join_key"] == "season_year"
    assert "NOT mean an endorsement contract" in meta["disclaimer"]
    assert meta["n_rows"] == 2
