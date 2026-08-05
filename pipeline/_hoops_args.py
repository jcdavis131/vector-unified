import torch, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from portable_paths import ESTATE  # noqa: E402

HOME = ESTATE
ck = torch.load(HOME/"vector-hoops/pipeline/data/mtnn_best.pt", map_location="cpu", weights_only=False)
print("=== hoops checkpoint args (full) ===")
print(json.dumps(ck["args"], indent=2, default=str))
print("\n=== hoops checkpoint weights ===")
print(json.dumps(ck["weights"], indent=2, default=str))
print("\n=== hoops checkpoint top-level keys ===")
print(list(ck.keys()))
print("\n=== hoops feature_manifest ===")
fm = json.loads((HOME/"vector-hoops/pipeline/data/feature_manifest.json").read_text(encoding="utf-8"))
print(json.dumps(fm, indent=2, default=str)[:3000])
