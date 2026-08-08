#!/usr/bin/env bash
# R12-H119 read chain on GPU2 (RTX 5000 Ada): the numeric-surface canonicalizer
# applied symmetrically to claim and evidence pre-tokenization, on four frozen
# checkpoints, in both directions. 8 deterministic reads, ~0.25 GPU-h each.
# Idempotent: each stage skips if its result JSON exists.
#
# Launch detached:
#   nohup setsid bash experiments/grounding-semantic/R12-H119_reads.sh \
#       >> logs/R12-H119_reads.log 2>&1 &

set -u
cd "$(dirname "$0")/../.." || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false

E=experiments/grounding-semantic
R=$E/R12-H119_windowed_read.py

stage() {
  local label="$1" outfile="$2"; shift 2
  if [ -s "$E/$outfile" ]; then
    echo "--- $label: already done ($outfile exists), skipping ---"
    return 0
  fi
  echo ""
  echo "--- $label  $(date '+%F %T') ---"
  if ! "$@"; then
    echo "=== FAILED: $label ==="
    exit 1
  fi
}

echo "=== R12-H119 READS $(date '+%F %T') ==="

for dir in strip add; do
  stage "h105d1 $dir" "R12-H119_h105d1_${dir}_windowed_result.json" \
    uv run python "$R" --model models/R9-H105-mmbert-dann-clean --direction "$dir" \
    --out "R12-H119_h105d1_${dir}_windowed_result.json"
  stage "h105d2 $dir" "R12-H119_h105d2_${dir}_windowed_result.json" \
    uv run python "$R" --model models/R9-H105-draw2 --direction "$dir" \
    --out "R12-H119_h105d2_${dir}_windowed_result.json"
  stage "h108d1 $dir" "R12-H119_h108d1_${dir}_windowed_result.json" \
    uv run python "$R" --model models/R10-H108-lane-draw1 --direction "$dir" \
    --out "R12-H119_h108d1_${dir}_windowed_result.json"
  stage "h108d2 $dir" "R12-H119_h108d2_${dir}_windowed_result.json" \
    uv run python "$R" --model models/R10-H108-lane-draw2 --direction "$dir" \
    --out "R12-H119_h108d2_${dir}_windowed_result.json"
done

echo ""
echo "=== R12-H119 READS COMPLETE $(date '+%F %T') ==="
