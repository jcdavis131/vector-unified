#!/usr/bin/env bash
# run_local_gpu.sh — Unified chimera — local GPU pickup
set -euo pipefail
EPOCHS="${1:-60}"
cd "$(dirname "$0")/.."
echo "[unified] epochs=$EPOCHS $(date -u)"
DEVICE="cpu"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then DEVICE="cuda"; echo "[unified] CUDA -> $DEVICE"; else echo "[unified] -> cpu"; fi

# unified needs 20,719 chimera from hoops+gridiron+pitch
if [ ! -f data/unified_matrix.npz ]; then
  echo "[unified] Missing data/unified_matrix.npz - build from hoops/pitch/gridiron exports first"
  echo "[unified] Try: python3 pipeline/build_chimera.py (if exists) or wait for hourly cache check"
  exit 0
fi

python3 pipeline/train_unified.py --epochs "$EPOCHS" 2>&1 | tee -a pipeline/cache/train_unified_${EPOCHS}ep.log || \
  python3 pipeline/train_stage2.py --epochs "$EPOCHS" 2>&1 | tee -a pipeline/cache/train_unified_${EPOCHS}ep.log || \
  echo "[unified] graceful exit - see log"

echo "[unified] done"
