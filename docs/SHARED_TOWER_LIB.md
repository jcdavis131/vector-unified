# Shared Tower Lib — MTNN v6 scaffolding (hoops → unified)

Source: vector-hoops pipeline/train_mtnn.py ResidualTower (canonical)

```python
class ResidualTower(nn.Module):
    def __init__(self, d_in, d_out=32, d_hidden=160):
        self.fc1 = Linear(2*d_in, d_hidden); self.ln1 = LayerNorm(d_hidden)
        self.fc2 = Linear(d_hidden, d_out); self.ln2 = LayerNorm(d_out)
        self.skip = Linear(2*d_in, d_out)
    def forward(self, x, m):
        h = cat([x*m, m], dim=-1)
        return ln2(fc2(gelu(ln1(fc1(h)))) + skip(h))
```

Gridiron same but d_hidden 32 d_out 24 + GatedFusion wrapper.
Pitch same but 3 families (attacking, passing/control, defending/dueling) → 24-d.

UnifiedTrunk reuses adapters + trunk MLP + GRL + SupCon + CORAL centroid+cov pattern from docs/UNIFIED_ARCHITECTURE.md §3.
See chimera_build_spec.json for loss formulas and λ schedule 0.10→0.3→0.5.
