import json, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).parent.parent
RUN_DIR = pathlib.Path.home() / "workspace" / "bundles" / "ultra" / "runs" / "money-equities"
GATE = RUN_DIR / "gate.json"

def test_gate_file_exists_and_valid():
    assert GATE.exists(), "gate.json must exist from pipeline run"
    data=json.loads(GATE.read_text())
    for k in ("ic","sharpe","win","dd","gate_pass","version","n"):
        assert k in data, f"missing {k}"
    assert isinstance(data["gate_pass"], bool)

def test_module_importable_zero_deps():
    import importlib.util
    spec=importlib.util.spec_from_file_location("money_equities_drift", ROOT/"money_equities_drift.py")
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "evaluate")
    assert hasattr(mod, "DriftConfig")
    assert hasattr(mod, "SectorMap")

def test_registry_extensible():
    import importlib.util
    spec=importlib.util.spec_from_file_location("m", ROOT/"money_equities_drift.py")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    @mod.registry.register("test_flag")
    def f(row): return 1.0
    assert "test_flag" in mod.registry._fns

def test_timeline_triple_write():
    tl = RUN_DIR/"timeline.jsonl"
    assert tl.exists()
    lines=tl.read_text().strip().split("\n")
    assert len(lines)>=2
    for line in lines:
        obj=json.loads(line)
        for k in ("ts","nodeId","agentId","attempt","latency_ms","tokens_est","status","errorClass"):
            assert k in obj
